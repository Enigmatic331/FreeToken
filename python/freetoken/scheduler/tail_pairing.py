"""Default-off layer-major tail pairing for DeepSeek-V4.1 prefill.

With ``FREETOKEN_DSV41_TAIL_PAIRING=1`` the final short chunk of a prompt (the *tail*) runs inside
the forward of the preceding full scheduler chunk (the *head*), layer by layer: the head's layer L,
then the tail's layer L, then the head's layer L+1. Both logical chunks then use each layer's
streamed expert bank once instead of streaming the whole bank twice. Chunk boundaries, every
per-chunk shape, state publication and output stay exactly those of the unpaired schedule.

This module holds the scheduler's only pairing decision (``plan_tail_pair``) and the cross-rank
agreement on it. Every EP rank runs the same scheduler, so each derives the same plan from its own
replicated state; ``agree_tail_pair`` fails closed, on every rank at once, if they ever differ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Callable

import torch

from freetoken.utils import div_ceil

TAIL_PAIRING_ENV = "FREETOKEN_DSV41_TAIL_PAIRING"
# The head chunk (--max-extend-tokens) the capacity bound and the GPU gate are qualified for.
QUALIFIED_HEAD_TOKENS = 8192
# Tails at most one EP route tile: the byte-bound regime the pairing targets. A longer tail costs
# nearly as much compute as the stream it saves while holding more cross-layer state.
MAX_TAIL_TOKENS = 4096
# Deep prompts never pair: the tail's extra cross-layer state then stays below the deepest
# (520K-class) unpaired peak that bounds authority VRAM, where the tail gains least anyway.
MAX_PROMPT_END = 131_072

_WIRE_MAGIC = 0x7A11_0001
_WIRE_VERSION = 1


def tail_pairing_enabled() -> bool:
    """Strict parse: unset or ``0`` is off, ``1`` is on, anything else is a startup error."""
    value = os.environ.get(TAIL_PAIRING_ENV)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{TAIL_PAIRING_ENV} must be '0' or '1', got {value!r}")


class TailPairMismatch(RuntimeError):
    """EP ranks disagree on, or received malformed, paired-prefill geometry."""


def _require_int(name: str, value) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


@dataclass(frozen=True)
class TailPairLimits:
    head_tokens: int  # the full scheduler chunk (max_extend_tokens)
    sparse_max_tokens: int  # tails this short take the sparse slot path and never stream
    page_size: int
    max_tail_tokens: int = MAX_TAIL_TOKENS
    max_prompt_end: int = MAX_PROMPT_END

    def __post_init__(self) -> None:
        for item in fields(self):
            _require_int(item.name, getattr(self, item.name))
        if self.page_size <= 0 or self.head_tokens <= 0 or self.head_tokens % self.page_size:
            raise ValueError("head_tokens must be a positive multiple of page_size")
        if not 0 <= self.sparse_max_tokens < self.max_tail_tokens <= self.head_tokens:
            raise ValueError("need 0 <= sparse_max_tokens < max_tail_tokens <= head_tokens")
        if self.max_prompt_end < self.head_tokens + self.max_tail_tokens:
            raise ValueError("max_prompt_end cannot hold one head and one tail")


def tail_pair_limits(config) -> TailPairLimits:
    """The pairing limits of ``config``; raises ValueError worded as the unmet requirement.

    Only the qualified head chunk pairs, and the sparse threshold must leave tails that stream
    the bank. The engine resolves this before the weight load and the scheduler again at init.
    """
    head_tokens = getattr(config, "max_extend_tokens", None)
    if head_tokens != QUALIFIED_HEAD_TOKENS:
        raise ValueError(
            f"--max-extend-tokens {QUALIFIED_HEAD_TOKENS}, the qualified head chunk "
            f"(got {head_tokens})"
        )
    sparse_max_tokens = int(getattr(config, "moe_sparse_prefill_max_tokens", 0) or 0)
    if not 0 <= sparse_max_tokens < MAX_TAIL_TOKENS:
        raise ValueError(
            f"--moe-sparse-prefill-max-tokens below {MAX_TAIL_TOKENS}, so that tails stream "
            f"the bank (got {sparse_max_tokens})"
        )
    return TailPairLimits(
        head_tokens=QUALIFIED_HEAD_TOKENS,
        sparse_max_tokens=sparse_max_tokens,
        page_size=int(config.page_size),
    )


@dataclass(frozen=True)
class TailPairPlan:
    """Agreed geometry of one paired forward: head [head_start, tail_start), tail
    [tail_start, prompt_end), both of request ``uid`` in page-table row ``table_idx``."""

    uid: int
    table_idx: int
    head_start: int
    head_len: int
    tail_len: int

    def __post_init__(self) -> None:
        for item in fields(self):
            _require_int(item.name, getattr(self, item.name))
        if self.uid < 0 or self.table_idx < 0 or self.head_start < 0:
            raise ValueError(f"negative paired-prefill field in {self}")
        if self.head_len <= 0 or self.tail_len <= 0:
            raise ValueError(f"empty paired-prefill sub-chunk in {self}")

    @property
    def tail_start(self) -> int:
        return self.head_start + self.head_len

    @property
    def prompt_end(self) -> int:
        return self.tail_start + self.tail_len

    def encode(self) -> torch.Tensor:
        return torch.tensor(
            [_WIRE_MAGIC, _WIRE_VERSION, 1, self.uid, self.table_idx,
             self.head_start, self.head_len, self.tail_len],
            dtype=torch.int64,
        )

    @staticmethod
    def encode_none() -> torch.Tensor:
        return torch.tensor([_WIRE_MAGIC, _WIRE_VERSION, 0, 0, 0, 0, 0, 0], dtype=torch.int64)

    @classmethod
    def decode(cls, wire: torch.Tensor) -> "TailPairPlan | None":
        if (not isinstance(wire, torch.Tensor) or wire.dtype != torch.int64
                or tuple(wire.shape) != (8,) or wire.device.type != "cpu"):
            raise TailPairMismatch(f"malformed paired-prefill plan wire: {wire!r}")
        magic, version, present, *values = (int(v) for v in wire.tolist())
        if magic != _WIRE_MAGIC or version != _WIRE_VERSION or present not in (0, 1):
            raise TailPairMismatch(f"unknown paired-prefill plan wire header: {wire.tolist()}")
        if not present:
            if any(values):
                raise TailPairMismatch(f"absent paired-prefill plan carries data: {wire.tolist()}")
            return None
        try:
            return cls(*values)
        except (TypeError, ValueError) as exc:
            raise TailPairMismatch(f"invalid paired-prefill plan {wire.tolist()}: {exc}") from exc


def plan_tail_pair(
    batch,
    *,
    limits: TailPairLimits,
    prompt_len: int,
    prefill_budget: int,
    decode_running: bool,
    pending_matches: bool,
    swa_free_tokens: int | None,
) -> TailPairPlan | None:
    """The pairing decision for a just-prepared prefill ``batch``; ``None`` keeps it unpaired.

    ``prompt_len`` is the complete prompt of the pending request whose continuation the batch's
    request is (``pending_matches``). ``swa_free_tokens`` is the window pool's free list alone,
    excluding the evictable tree, so a pair never forces an eviction the unpaired schedule would
    not; ``None`` means the model has no paged window pool.
    """
    from .prefill import ChunkedReq

    if not batch.is_prefill or getattr(batch, "speculative", False) or len(batch.reqs) != 1:
        return None
    if decode_running or not pending_matches:
        return None
    head = batch.reqs[0]
    if not isinstance(head, ChunkedReq):  # a final or single chunk has no tail to pair
        return None
    if head.extend_len != limits.head_tokens or prefill_budget != limits.head_tokens:
        return None
    tail_len = prompt_len - head.device_len
    if not limits.sparse_max_tokens < tail_len <= limits.max_tail_tokens:
        return None
    if prompt_len > limits.max_prompt_end:
        return None
    page = limits.page_size
    if head.device_len % page:
        return None
    if swa_free_tokens is not None:
        # The tail's pages plus one page for the first decode step, all before the head's
        # out-of-window pages return at the drain.
        needed = (div_ceil(prompt_len, page) - head.device_len // page + 1) * page
        if swa_free_tokens < needed:
            return None
    if head.uid < 0 or head.table_idx < 0:
        return None  # not a served request row; decline rather than fail the scheduler
    return TailPairPlan(
        uid=int(head.uid),
        table_idx=int(head.table_idx),
        head_start=int(head.cached_len),
        head_len=int(head.extend_len),
        tail_len=int(tail_len),
    )


def validate_paired_batches(head_batch, tail_batch) -> TailPairPlan:
    """Check that a paired forward's two batches are exactly the agreed plan's chunks.

    Runs on every rank before the forward's first collective. Every rank holds the same
    replicated batches, so a malformed pair raises ``TailPairMismatch`` on all of them alike.
    """
    plan = getattr(tail_batch, "tail_pair_plan", None)
    if (
        not isinstance(plan, TailPairPlan)
        or getattr(head_batch, "tail_pair_plan", None) is not plan
        or getattr(head_batch, "tail_pair_role", None) != "head"
        or getattr(tail_batch, "tail_pair_role", None) != "tail"
    ):
        raise TailPairMismatch("paired forward without one agreed plan on both chunks")
    for role, batch in (("head", head_batch), ("tail", tail_batch)):
        if not batch.is_moe_prefill or len(batch.reqs) != 1:
            raise TailPairMismatch(f"paired {role} must be exactly one bulk-prefill request")
    head_req, tail_req = head_batch.reqs[0], tail_batch.reqs[0]
    if head_req.can_decode or not tail_req.can_decode:
        raise TailPairMismatch(
            "paired head must be a non-final chunk and the tail the final, decoding chunk "
            f"(head decodes {head_req.can_decode}, tail decodes {tail_req.can_decode})"
        )
    actual = tuple(
        (req.uid, req.table_idx, req.cached_len, req.device_len)
        for req in (head_req, tail_req)
    )
    expected = (
        (plan.uid, plan.table_idx, plan.head_start, plan.tail_start),
        (plan.uid, plan.table_idx, plan.tail_start, plan.prompt_end),
    )
    if actual != expected:
        raise TailPairMismatch(f"paired chunks {actual} do not match {plan}")
    sizes = (head_batch.input_ids.numel(), tail_batch.input_ids.numel())
    if sizes != (plan.head_len, plan.tail_len):
        raise TailPairMismatch(f"paired token counts {sizes} do not match {plan}")
    return plan


def agree_tail_pair(
    plan: TailPairPlan | None,
    *,
    rank: int,
    world_size: int,
    broadcast: Callable[[torch.Tensor], torch.Tensor],
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor],
    unready: str | None = None,
) -> TailPairPlan | None:
    """Return rank 0's plan once every rank has proven it computed the same one.

    Rank 0 broadcasts its encoded plan (or "no pair"); each rank compares it with its own and the
    mismatch count is summed, so every rank reaches the same verdict and either all proceed or all
    raise ``TailPairMismatch`` before any GPU collective of the forward is issued. ``unready``
    is why this rank could not execute its own plan (rank-local engine state); it counts as a
    mismatch, so that too fails closed on every rank instead of hanging the others mid-forward.
    """
    local = plan.encode() if plan is not None else TailPairPlan.encode_none()
    if world_size <= 1:
        if unready is not None:
            raise TailPairMismatch(f"rank {rank} cannot execute paired prefill: {unready}")
        return plan
    reference = broadcast(local.clone() if rank == 0 else torch.zeros_like(local))
    disagrees = not torch.equal(reference, local) or unready is not None
    mismatch = torch.tensor([int(disagrees)], dtype=torch.int64)
    mismatch = all_reduce_sum(mismatch)
    if int(mismatch.item()) != 0:
        raise TailPairMismatch(
            f"rank {rank}: {int(mismatch.item())} rank(s) disagree with or cannot execute rank "
            f"0's paired-prefill plan (rank 0 {reference.tolist()}, this rank {local.tolist()}"
            f"{'' if unready is None else ', unready: ' + unready})"
        )
    return TailPairPlan.decode(reference)


__all__ = [
    "MAX_PROMPT_END",
    "MAX_TAIL_TOKENS",
    "QUALIFIED_HEAD_TOKENS",
    "TAIL_PAIRING_ENV",
    "TailPairLimits",
    "TailPairMismatch",
    "TailPairPlan",
    "agree_tail_pair",
    "plan_tail_pair",
    "tail_pair_limits",
    "tail_pairing_enabled",
    "validate_paired_batches",
]
