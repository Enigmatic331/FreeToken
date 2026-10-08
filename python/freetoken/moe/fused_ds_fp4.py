"""Routed FP4 expert compute on the offload-cache banks (DeepSeek-V4).

Composes the DeepSeek-FP4 fused-MoE kernels (``dsv4_fused_moe``) into the full
routed-expert path: ``gate_up -> swiglu(limit) -> down``, reading expert weights
directly from the resident banks (no bf16 materialization). ``slots`` maps each
(token, route) to a bank row: a cache slot for decode or selective short prefill,
or a streamed full-layer position (== expert id) for bulk prefill.
"""

from __future__ import annotations

from functools import lru_cache
import os

import torch
import triton

from freetoken.kernel.triton.dsv4.fp8_linear import (
    act_quant_fp8_inplace,
    act_quant_fp8_roundtrip,
)
from freetoken.kernel.triton.dsv4.fused_moe import (
    _decode_dsfp4_moe_kernel,
    _e2m1_lut,
    _prefill_dsfp4_moe_kernel,
    fused_swiglu,
)
from freetoken.moe.fused import moe_align_block_size

_TL_DTYPE = None
_SKIP_ZERO_ROUTES = os.environ.get("FREETOKEN_DSV41_SKIP_ZERO_ROUTES", "0") == "1"
_SM89_WIDE_GRID = os.environ.get("FREETOKEN_DSV41_SM89_WIDE_GRID", "0") == "1"
_PREFILL_SKIP_INACTIVE = os.environ.get("FREETOKEN_DSV41_PREFILL_SKIP_INACTIVE", "0") == "1"


@lru_cache(maxsize=None)
def _zero_route_device_supported(device: torch.device) -> bool:
    # Query only the input's device, never another rank's CUDA device.
    return torch.cuda.get_device_capability(device) in ((12, 0), (8, 9))


def _use_zero_route_skip(tokens: int, activation_block_size: int, device: torch.device) -> bool:
    return (
        _SKIP_ZERO_ROUTES and tokens == 1 and activation_block_size == 32
        and device.type == "cuda" and _zero_route_device_supported(device)
    )


@lru_cache(maxsize=None)
def _decode_launch_geometry(device: torch.device) -> tuple[int, int, int]:
    """Return ``(N tile, packed-K-byte tile, warps)`` for decode FP4 GEMVs.

    The opt-in wider grid is deliberately limited to high-SM-count Ada.  A
    real-checkpoint screen found N=8 faster on the 128-SM RTX 4090 but slower
    on the 80-SM RTX 4080 SUPER; SM120 and smaller Ada therefore retain the
    qualified production geometry.
    """
    if (
        _SM89_WIDE_GRID
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) == (8, 9)
        and torch.cuda.get_device_properties(device).multi_processor_count >= 120
    ):
        return 8, 128, 1
    return 16, 128, 1


def _compute_type(dtype: torch.dtype):
    import triton.language as tl

    return {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[dtype]


def _grouped_decode(
    a: torch.Tensor,            # [A_rows, K] compute dtype
    packed_cache: torch.Tensor,  # [S, N, K//2] uint8
    scale_cache: torch.Tensor,   # [S, N, K//32] e8m0
    slots: torch.Tensor,         # [T, top_k] int32 -> cache slot
    topk_weights: torch.Tensor | None,
    *,
    a_row_is_route: bool,
    mul_routed_weight: bool,
    skip_zero_weight: bool = False,
) -> torch.Tensor:
    """Grouped per-route GEMM ``c[t,r,:] = a[row] @ dequant(W[slot])^T``.

    Returns ``[T, top_k, N]``. Used for both gate_up (a_row=token) and down
    (a_row=route). FP4 weights are dequantized inline from the slot cache.
    """
    T, top_k = slots.shape
    N = packed_cache.shape[1]
    K = packed_cache.shape[2] * 2
    total_routes = T * top_k
    dtype = a.dtype
    out = torch.empty((T, top_k, N), dtype=dtype, device=a.device)
    if topk_weights is None:
        assert not skip_zero_weight, "Zero-route skipping needs real route weights"
        topk_weights = out.new_empty((1, 1), dtype=torch.float32)
    scale_u8 = scale_cache.view(torch.uint8)

    # Decode is a per-route GEMV, bound by inline FP4 dequant.  The qualified
    # default is BN=16/BKB=128/1 warp.  An opt-in 128-SM Ada path uses BN=8 to
    # expose twice as many CTAs; the device discriminator is cached outside the
    # kernel and leaves the 4080 SUPER and both 5090s unchanged.
    BLOCK_SIZE_N, BLOCK_SIZE_KB, _NW = _decode_launch_geometry(a.device)
    assert (K // 2) % BLOCK_SIZE_KB == 0, (K, BLOCK_SIZE_KB)
    grid = (total_routes, triton.cdiv(N, BLOCK_SIZE_N))
    _decode_dsfp4_moe_kernel[grid](
        a, packed_cache, scale_u8, out, topk_weights, slots,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_cache.stride(0), packed_cache.stride(1), packed_cache.stride(2),
        scale_u8.stride(0), scale_u8.stride(1), scale_u8.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        topk_weights.stride(0) if topk_weights.ndim == 2 else 0,
        topk_weights.stride(1) if topk_weights.ndim == 2 else 0,
        slots.stride(0), slots.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_KB=BLOCK_SIZE_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_compute_type(dtype),
        SKIP_ZERO_WEIGHT=skip_zero_weight,
        num_warps=_NW,
    )
    return out


def routed_experts_fp4(
    x: torch.Tensor,             # [T, H] compute dtype
    slots: torch.Tensor,         # [T, top_k] int32 -> cache slot
    topk_weights: torch.Tensor,  # [T, top_k] fp32 (incl. route_scale + renorm)
    gate_up_packed: torch.Tensor,  # [S, 2I, H//2] uint8
    gate_up_scale: torch.Tensor,   # [S, 2I, H//32] e8m0
    down_packed: torch.Tensor,     # [S, H, I//2] uint8
    down_scale: torch.Tensor,      # [S, H, I//32] e8m0
    swiglu_limit: float,
    return_route_outputs: bool = False,
    activation_block_size: int = 128,
    *, skip_zero_weight: bool | None = None,
) -> torch.Tensor:
    """Full routed-expert output (summed over the top-k routes), excludes shared expert.

    Precision matches the reference ``fp4_gemm(act_quant(x, 128), W_fp4)``: the gate_up
    and down activations are FP8-round-tripped (block 128, ue8m0) before each GEMM. Since
    an fp8 value x pow2 scale is exact in bf16, the round-tripped activation entering the
    bf16 decode kernel is bit-identical to the reference's dequantized FP8 activation
    (validated max diff = 0 vs the tilelang ``fp4_gemm`` reference)."""
    T, top_k = slots.shape
    H = x.shape[1]
    two_I = gate_up_packed.shape[1]
    I = two_I // 2
    skip_zero = (
        _use_zero_route_skip(T, activation_block_size, x.device)
        if skip_zero_weight is None else skip_zero_weight
    )

    x = act_quant_fp8_roundtrip(
        x, activation_block_size
    )  # gate_up activation -> FP8 round-trip (no clone)
    gate_up = _grouped_decode(
        x, gate_up_packed, gate_up_scale, slots, topk_weights if skip_zero else None,
        a_row_is_route=False, mul_routed_weight=False,
        skip_zero_weight=skip_zero,
    )  # [T, top_k, 2I]
    act = fused_swiglu(gate_up, swiglu_limit)  # [T, top_k, I]
    # SwiGLU writes independent storage.  End the larger gate/up tensor's
    # lifetime before allocating the down projection so the caching allocator
    # can reuse this block instead of overlapping both route-output buffers.
    del gate_up

    act = act.reshape(T * top_k, I)
    act_quant_fp8_inplace(
        act, activation_block_size
    )  # down activation -> FP8 round-trip
    down = _grouped_decode(
        act, down_packed, down_scale, slots, topk_weights,
        a_row_is_route=True, mul_routed_weight=True,
        skip_zero_weight=skip_zero,
    )  # [T, top_k, H]
    return down if return_route_outputs else down.sum(dim=1)


# Above this the grouped GEMM beats the per-route GEMV despite its padding;
# below, the GEMV's exact routes*N work wins (short streaming chunks). The
# grouped kernel sits on its dequant floor (~6.4-6.9ms/GEMM-pair at DSV4
# geometry) for any chunk size, so the crossover is where the GEMV's
# routes-proportional cost reaches that floor (H100 sweep).
_GROUPED_MIN_ROUTES = 768


def _align_prefill_routes(slots, block_size: int, num_rows: int, skip_inactive: bool):
    # The optional native aligner does not preserve the extra sentinel bucket
    # uniformly: small expert counts can emit expert_id=-1, while larger
    # counts can omit those routes. Our zero-writing branch requires every
    # route exactly once and the explicit positive sentinel == num_rows.
    # Keep the accepted native OFF path; use the explicit contract only for
    # inactive skipping or the already-qualified expanded cache geometries.
    if skip_inactive or num_rows >= 1024:
        from freetoken.kernel.triton.moe_align import moe_align_block_size as align

        return align(slots, block_size, num_rows, include_sentinel=True)
    return moe_align_block_size(slots, block_size, num_rows)


def _grouped_prefill(
    a: torch.Tensor,             # [A_rows, K] compute dtype (FP8 round-tripped)
    packed_cache: torch.Tensor,  # [S, N, K//2] uint8
    scale_cache: torch.Tensor,   # [S, N, K//32] e8m0
    c: torch.Tensor,             # [T, top_k, N] output, flat-indexed over routes
    tw_flat: torch.Tensor,       # [T*top_k] fp32
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: dict,
    inactive_expert: int = -1,
) -> None:
    N = packed_cache.shape[1]
    K = packed_cache.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale_u8 = scale_cache.view(torch.uint8)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_dsfp4_moe_kernel[grid](
        a, packed_cache, scale_u8, c, tw_flat,
        sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid,
        a.stride(0), a.stride(1),
        packed_cache.stride(0), packed_cache.stride(1), packed_cache.stride(2),
        scale_u8.stride(0), scale_u8.stride(1), scale_u8.stride(2),
        c.stride(-2), c.stride(-1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_compute_type(a.dtype),
        INACTIVE_EXPERT=inactive_expert,
        num_warps=cfg.get("num_warps", 4),
        num_stages=cfg.get("num_stages", 3),
    )


def routed_experts_fp4_prefill(
    x: torch.Tensor,             # [T, H] compute dtype
    slots: torch.Tensor,         # [T, top_k] int32 -> bank row in [0, num_rows)
    topk_weights: torch.Tensor,  # [T, top_k] fp32 (incl. route_scale + renorm)
    gate_up_packed: torch.Tensor,  # [S, 2I, H//2] uint8
    gate_up_scale: torch.Tensor,   # [S, 2I, H//32] e8m0
    down_packed: torch.Tensor,     # [S, H, I//2] uint8
    down_scale: torch.Tensor,      # [S, H, I//32] e8m0
    swiglu_limit: float,
    num_rows: int,
    return_route_outputs: bool = False,
    activation_block_size: int = 128,
) -> torch.Tensor:
    """Grouped-GEMM counterpart of :func:`routed_experts_fp4` for dense prefill
    chunks: one moe_align sort shared by both GEMMs, each expert's weights
    dequantized once per N-tile instead of once per route. Same FP8
    round-tripped activations; differs from the GEMV only in fp32 accumulation
    order (tl.dot tree vs sequential K-walk)."""
    T, top_k = slots.shape
    if T * top_k < _GROUPED_MIN_ROUTES:
        return routed_experts_fp4(
            x, slots, topk_weights,
            gate_up_packed, gate_up_scale, down_packed, down_scale, swiglu_limit,
            return_route_outputs=return_route_outputs,
            activation_block_size=activation_block_size,
        )
    H = x.shape[1]
    two_I = gate_up_packed.shape[1]
    I = two_I // 2
    routes = T * top_k
    # One static config for every density (no autotune): the kernel is
    # dequant-floor-bound, so per-expert padding at BLOCK_M=64 costs the same
    # as tighter tiles while keeping the wgmma-wide M tile on sm_90.
    cfg = dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
        num_warps=8, num_stages=1,
    )
    skip_inactive = _PREFILL_SKIP_INACTIVE and activation_block_size == 32
    if skip_inactive:
        # The caller can retain cache-safe IDs; zero-weight routes are sorted
        # into a separate bucket rather than mixed with useful expert rows.
        # Keep the aligner's sentinel bin and every route output initialized.
        slots = torch.where(topk_weights == 0, num_rows, slots).contiguous()
    sorted_ids, expert_ids, ntpp = _align_prefill_routes(
        slots, cfg["BLOCK_SIZE_M"], num_rows, skip_inactive
    )
    inactive_expert = num_rows if skip_inactive else -1
    tw = topk_weights.reshape(-1).contiguous()

    x = act_quant_fp8_roundtrip(
        x, activation_block_size
    )  # gate_up activation -> FP8 round-trip (no clone)
    gate_up = torch.empty((T, top_k, two_I), dtype=x.dtype, device=x.device)
    _grouped_prefill(
        x, gate_up_packed, gate_up_scale, gate_up, tw,
        sorted_ids, expert_ids, ntpp, routes, top_k, False, cfg,
        inactive_expert=inactive_expert,
    )
    act = fused_swiglu(gate_up, swiglu_limit)  # [T, top_k, I]

    act = act.reshape(routes, I)
    act_quant_fp8_inplace(
        act, activation_block_size
    )  # down activation -> FP8 round-trip
    down_numel = T * top_k * H
    if gate_up.numel() >= down_numel:
        # The gate/up result is dead after SwiGLU and, at DSV4 geometry, is
        # wider than the down result.  Reinterpret its storage instead of
        # asking the fragmented long-prefill allocator for another ~480 MiB
        # block.  Both kernels use the same CUDA stream, so the down write is
        # ordered after SwiGLU's final read from this storage.
        down = gate_up.reshape(-1)[:down_numel].view(T, top_k, H)
        del gate_up
    else:
        del gate_up
        down = torch.empty((T, top_k, H), dtype=x.dtype, device=x.device)
    _grouped_prefill(
        act, down_packed, down_scale, down, tw,
        sorted_ids, expert_ids, ntpp, routes, 1, True, cfg,
        inactive_expert=inactive_expert,
    )
    return down if return_route_outputs else down.sum(dim=1)


__all__ = ["routed_experts_fp4", "routed_experts_fp4_prefill", "_grouped_decode"]
