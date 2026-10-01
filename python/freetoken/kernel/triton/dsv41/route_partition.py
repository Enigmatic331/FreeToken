"""Fused EP route localization for one-token DeepSeek-V4.1 decode.

The ordinary tensor implementation is intentionally retained for prefill.  Decode has
only ``top_k`` route entries per row, however, and expressing partition localization
plus cache-safe inactive-id replacement as PyTorch tensor operations creates a long
chain of tiny CUDA graph nodes.  This kernel performs the same transform in one CTA per
row without changing route order, weights, or cache admission semantics.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "global_offset", "local_count"])
def _localize_cache_safe_routes_kernel(
    weights_ptr,
    ids_ptr,
    out_weights_ptr,
    out_ids_ptr,
    rows,
    global_offset,
    local_count,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_K)
    mask = (row < rows) & (lane < TOP_K)
    offset = row * TOP_K + lane

    weight = tl.load(weights_ptr + offset, mask=mask, other=0.0)
    global_id = tl.load(ids_ptr + offset, mask=mask, other=0).to(tl.int32)
    local_id = global_id - global_offset
    owned = mask & (local_id >= 0) & (local_id < local_count)
    active = owned & (weight != 0.0)

    # cache_safe_route_ids uses the first non-zero route in each row as the harmless
    # replacement for every inactive/sentinel position, or local row zero when this
    # rank owns no route.  The replacement is mathematically inert because the
    # corresponding route weight remains zero.
    first = tl.min(tl.where(active, lane, BLOCK_K), axis=0)
    fallback = tl.sum(
        tl.where(active & (lane == first), local_id, 0), axis=0
    ).to(tl.int32)

    tl.store(out_weights_ptr + offset, tl.where(owned, weight, 0.0), mask=mask)
    tl.store(out_ids_ptr + offset, tl.where(active, local_id, fallback), mask=mask)


def fused_localize_cache_safe_routes(
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    global_offset: int,
    local_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rank-local weights and cache-safe int32 ids in one CUDA launch.

    This is the fused equivalent of ``localize_expert_routes`` followed by
    ``cache_safe_route_ids``.  It is decode-only: callers retain the ordinary path for
    prefill, whose inactive sentinel is useful to its compact route schedule.
    """
    if not weights.is_cuda or not ids.is_cuda:
        raise ValueError("fused route localization requires CUDA tensors")
    if weights.ndim != 2 or ids.shape != weights.shape:
        raise ValueError(
            "route weights and ids must be matching 2D tensors, got "
            f"{tuple(weights.shape)} and {tuple(ids.shape)}"
        )
    if weights.dtype != torch.float32:
        raise ValueError(f"route weights must be float32, got {weights.dtype}")
    if ids.dtype != torch.int32:
        raise ValueError(f"route ids must be int32, got {ids.dtype}")
    if not weights.is_contiguous() or not ids.is_contiguous():
        raise ValueError("route weights and ids must be contiguous")
    if local_count < 0:
        raise ValueError(f"local_count must be non-negative, got {local_count}")

    rows, top_k = weights.shape
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    out_weights = torch.empty_like(weights)
    out_ids = torch.empty_like(ids)
    _localize_cache_safe_routes_kernel[(rows,)](
        weights,
        ids,
        out_weights,
        out_ids,
        rows,
        int(global_offset),
        int(local_count),
        TOP_K=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        num_warps=1,
    )
    return out_weights, out_ids


__all__ = ["fused_localize_cache_safe_routes"]
