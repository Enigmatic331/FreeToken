"""Correctness-first hierarchical index selection for V4.1 CSA2."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


INDEXER_PREFILL_MAX_LOGITS_MB_ENV = "FREETOKEN_DSV41_INDEXER_MAX_LOGITS_MB"
DEFAULT_INDEXER_PREFILL_MAX_LOGITS_MB = 512


def indexer_prefill_max_logits_bytes() -> int:
    """Configured cap for one fused prefill-logits allocation.

    The score matrix is fp32.  Keeping the limit in bytes makes the row planner
    explicit and mirrors the limit used by the other V4 indexer runtimes.
    """

    raw = os.getenv(
        INDEXER_PREFILL_MAX_LOGITS_MB_ENV,
        str(DEFAULT_INDEXER_PREFILL_MAX_LOGITS_MB),
    )
    try:
        mib = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{INDEXER_PREFILL_MAX_LOGITS_MB_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if mib <= 0:
        raise ValueError(
            f"{INDEXER_PREFILL_MAX_LOGITS_MB_ENV} must be positive, got {mib}"
        )
    return mib * 1024 * 1024


def indexer_prefill_chunk_rows(
    query_rows: int, key_rows: int, max_logits_bytes: int
) -> int:
    """Largest query-row chunk whose fp32 ``[M, N]`` logits fit the cap.

    A single query row is the indivisible fallback when its key width alone is
    larger than the cap.  Zero-width keys need no score allocation.
    """

    if query_rows < 0 or key_rows < 0:
        raise ValueError("indexer query/key row counts cannot be negative")
    if max_logits_bytes <= 0:
        raise ValueError("indexer logits byte cap must be positive")
    if query_rows == 0:
        return 0
    if key_rows == 0:
        return query_rows
    return min(query_rows, max(1, max_logits_bytes // (key_rows * 4)))


def visible_compressed_lengths(positions: torch.Tensor, ratio: int) -> torch.Tensor:
    """Number of completed compressed groups visible to each query position."""

    if ratio not in (1, 2):
        raise ValueError(f"V4.1 indexer requires ratio 1 or 2, got {ratio}")
    return torch.div(positions + 1, ratio, rounding_mode="floor")


def select_candidate_blocks(
    logits: torch.Tensor,
    compressed_lengths: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Publish DeepSeek's level-one mask, pinning each query's newest block."""

    if block_size <= 0 or topk_blocks <= 0:
        raise ValueError("candidate block size and top-k must be positive")
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1)
    num_blocks = scores.shape[-1]
    if not torch.is_tensor(compressed_lengths):
        compressed_lengths = torch.as_tensor(
            compressed_lengths, device=logits.device
        )
    newest = torch.div(
        compressed_lengths - 1, block_size, rounding_mode="floor"
    )
    block_ids = torch.arange(num_blocks, device=logits.device)
    scores = scores.masked_fill(block_ids == newest, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_index_topk(
    logits: torch.Tensor,
    compressed_lengths: torch.Tensor | int,
    topk: int,
    *,
    offset: int = 0,
    candidate_mask: torch.Tensor | None = None,
    inplace: bool = False,
) -> torch.Tensor:
    """Mask, select, position-sort, and ``-1``-pad CSA2 compressed rows.

    ``inplace`` lets the long-prefill path reuse its disposable fp32 logits
    buffer instead of allocating another full score matrix for each mask.  The
    default remains non-mutating for decode/verification and general callers.
    """

    width = logits.shape[-1]
    if width == 0 or topk <= 0:
        return torch.empty(*logits.shape[:-1], 0, dtype=torch.int32, device=logits.device)
    if not torch.is_tensor(compressed_lengths):
        compressed_lengths = torch.as_tensor(
            compressed_lengths, device=logits.device
        )
    live = compressed_lengths
    while live.ndim < logits.ndim:
        live = live.unsqueeze(-1)
    columns = torch.arange(width, device=logits.device)
    scores = logits
    if inplace:
        scores.masked_fill_(columns >= live, -torch.inf)
    else:
        scores = scores.masked_fill(columns >= live, -torch.inf)
    if candidate_mask is not None:
        if candidate_mask.shape != logits.shape:
            raise ValueError(
                f"candidate mask {candidate_mask.shape} != logits {logits.shape}"
            )
        if inplace:
            scores.masked_fill_(~candidate_mask, -torch.inf)
        else:
            scores = scores.masked_fill(~candidate_mask, -torch.inf)
    k = min(int(topk), width)
    picked = scores.topk(k, dim=-1, sorted=False).indices
    selected_scores = scores.gather(-1, picked)
    picked = picked.masked_fill(selected_scores == -torch.inf, width)
    picked = picked.sort(dim=-1).values
    return torch.where(picked < live, picked + offset, -1).to(torch.int32)


class CandidateRuntime:
    """Per-forward candidate publication shared by layer 20 and its consumers."""

    def __init__(self) -> None:
        self.mask: torch.Tensor | None = None

    def publish(
        self,
        logits: torch.Tensor,
        compressed_lengths: torch.Tensor | int,
        topk_blocks: int,
        block_size: int,
    ) -> torch.Tensor:
        self.mask = select_candidate_blocks(
            logits, compressed_lengths, topk_blocks, block_size
        )
        return self.mask

    def consume(self, logits: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            raise RuntimeError("candidate consumer ran before the source layer")
        if self.mask.shape != logits.shape:
            raise ValueError(
                f"published candidate mask {self.mask.shape} != logits {logits.shape}"
            )
        return logits.masked_fill(~self.mask, -torch.inf)


__all__ = [
    "CandidateRuntime",
    "DEFAULT_INDEXER_PREFILL_MAX_LOGITS_MB",
    "INDEXER_PREFILL_MAX_LOGITS_MB_ENV",
    "indexer_prefill_chunk_rows",
    "indexer_prefill_max_logits_bytes",
    "select_candidate_blocks",
    "select_index_topk",
    "visible_compressed_lengths",
]
