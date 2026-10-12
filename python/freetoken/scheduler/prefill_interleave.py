"""Default-off resident-decode interleaving between prefill chunks under DeepSeek-V4.1 gated C2.

C2 forwards one prefill request at a time and the scheduler is prefill-first, so a resident
decode row otherwise waits for the other request's whole prompt. With
``FREETOKEN_DSV41_C2_PREFILL_INTERLEAVE=N`` (1..8), each *intermediate* prefill chunk grants up to
N ordinary decode steps for the resident rows before the prompt's next chunk. Nothing runs beside
a prefill forward, and a prompt's first chunk, final chunk and paired head+tail forward are
scheduled exactly as before; only the order of whole forwards changes.

The decision uses replicated scheduler state only (a step count, never a clock), and
``agree_interleave`` proves every EP rank made it before anything is scheduled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Callable

import torch

PREFILL_INTERLEAVE_ENV = "FREETOKEN_DSV41_C2_PREFILL_INTERLEAVE"
MAX_INTERLEAVE_STEPS = 8

_WIRE_MAGIC = 0x1D7E_0001
_WIRE_VERSION = 1
_WIRE_LEN = 7


def prefill_interleave_steps() -> int:
    """Strict parse: unset or ``0`` is off, ``1``..``8`` decode steps per chunk boundary."""
    value = os.environ.get(PREFILL_INTERLEAVE_ENV)
    if value is None or value == "0":
        return 0
    try:
        steps = int(value)
    except ValueError:
        steps = -1
    if str(steps) != value or not 1 <= steps <= MAX_INTERLEAVE_STEPS:
        raise ValueError(
            f"{PREFILL_INTERLEAVE_ENV} must be an integer from 0 to {MAX_INTERLEAVE_STEPS}, "
            f"got {value!r}"
        )
    return steps


class PrefillInterleaveMismatch(RuntimeError):
    """EP ranks disagree on, or received malformed, an interleave decision."""


@dataclass(frozen=True)
class InterleaveDecision:
    """One scheduling decision at a chunk boundary, identical on every EP rank."""

    decode: bool  # True: run the resident decode rows now; False: resume the prompt
    uid: int  # the prefilling prompt whose continuation heads the queue
    resume_at: int  # that continuation's next chunk start (the chunk Req's cached_len)
    budget: int  # decode steps left at this boundary before the decision
    decode_rows: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not (bool if item.name == "decode" else int):
                raise TypeError(f"{item.name} must be a {item.type}, got {value!r}")
        if self.uid < 0 or self.resume_at < 0 or self.decode_rows < 1:
            raise ValueError(f"invalid interleave decision {self}")
        if not 0 <= self.budget <= MAX_INTERLEAVE_STEPS or (self.decode and self.budget < 1):
            raise ValueError(f"invalid interleave budget in {self}")

    def encode(self) -> torch.Tensor:
        return torch.tensor(
            [_WIRE_MAGIC, _WIRE_VERSION, int(self.decode), self.uid, self.resume_at,
             self.budget, self.decode_rows],
            dtype=torch.int64,
        )

    @classmethod
    def decode_wire(cls, wire: torch.Tensor) -> "InterleaveDecision":
        if (not isinstance(wire, torch.Tensor) or wire.dtype != torch.int64
                or tuple(wire.shape) != (_WIRE_LEN,) or wire.device.type != "cpu"):
            raise PrefillInterleaveMismatch(f"malformed interleave wire: {wire!r}")
        magic, version, decode, uid, resume_at, budget, rows = (int(v) for v in wire.tolist())
        if magic != _WIRE_MAGIC or version != _WIRE_VERSION or decode not in (0, 1):
            raise PrefillInterleaveMismatch(f"unknown interleave wire header: {wire.tolist()}")
        try:
            return cls(bool(decode), uid, resume_at, budget, rows)
        except (TypeError, ValueError) as exc:
            raise PrefillInterleaveMismatch(f"invalid interleave wire {wire.tolist()}") from exc


def plan_interleave(
    *, budget: int, pending_head, decode_rows: int
) -> InterleaveDecision | None:
    """The decision at a scheduling point, or None when no interleave question arises.

    The question arises only when the queue's head is a prompt's continuation (it has run an
    intermediate chunk) and other requests are resident in decode. Under C2 that is one row.
    """
    chunk = getattr(pending_head, "chunked_req", None) if pending_head is not None else None
    if chunk is None or decode_rows < 1:
        return None
    return InterleaveDecision(
        decode=budget > 0,
        uid=int(chunk.uid),
        resume_at=int(chunk.cached_len),
        budget=int(budget),
        decode_rows=int(decode_rows),
    )


def agree_interleave(
    decision: InterleaveDecision,
    *,
    rank: int,
    world_size: int,
    broadcast: Callable[[torch.Tensor], torch.Tensor],
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor],
) -> InterleaveDecision:
    """Return rank 0's decision once every rank has proven it computed the same one; otherwise
    every rank raises ``PrefillInterleaveMismatch`` before scheduling anything."""
    if world_size <= 1:
        return decision
    local = decision.encode()
    reference = broadcast(local.clone() if rank == 0 else torch.zeros_like(local))
    mismatch = torch.tensor([int(not torch.equal(reference, local))], dtype=torch.int64)
    mismatch = all_reduce_sum(mismatch)
    if int(mismatch.item()) != 0:
        raise PrefillInterleaveMismatch(
            f"rank {rank}: {int(mismatch.item())} rank(s) disagree with rank 0's prefill "
            f"interleave decision (rank 0 {reference.tolist()}, this rank {local.tolist()})"
        )
    return InterleaveDecision.decode_wire(reference)


def prefill_interleave_problems(config, *, requested_max_running_req: int) -> list[str]:
    """Why this execution cannot interleave, as unmet requirements; [] when it can.

    The engine checks this before the weight load. Interleaving only exists where a decode row
    can be resident beside another request's chunked prefill: DeepSeek-V4.1's gated C2.
    """
    from freetoken.engine.engine import (
        _dsv41_feature_concurrency,
        _feature_concurrency_problems,
    )

    args = getattr(config.model_config, "dsv41_args", None)
    if args is None:
        return ["DeepSeek-V4.1"]
    problems = []
    if _dsv41_feature_concurrency(config) != 2:
        problems.append(
            "DeepSeek-V4.1 gated C2 (FREETOKEN_DSV41_CONCURRENCY=1, --max-running-requests 2)"
        )
    else:
        problems.extend(_feature_concurrency_problems(config, requested_max_running_req, 2))
    if getattr(config, "speculative_dspark", False) or getattr(args, "dspark_enabled", False):
        problems.append("MTP/DSpark off")
    return problems


__all__ = [
    "MAX_INTERLEAVE_STEPS",
    "PREFILL_INTERLEAVE_ENV",
    "InterleaveDecision",
    "PrefillInterleaveMismatch",
    "agree_interleave",
    "plan_interleave",
    "prefill_interleave_problems",
    "prefill_interleave_steps",
]
