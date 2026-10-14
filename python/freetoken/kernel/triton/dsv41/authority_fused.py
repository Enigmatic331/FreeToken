"""Exact decode-only primitives for the DSV4.1 authority-fusion path.

The wrappers in this module deliberately preserve the production tensor shapes
and reduction order.  Package-A is opt-in; these kernels are never reached by
the default path.  Their numerical/code-generation contract is qualified by
the G2 suite before any live use.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_TL = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


@triton.jit
def _hc_prepare_mix_kernel(x, flat, square, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    value = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(flat + offs, value, mask=mask)
    tl.store(square + offs, value * value, mask=mask)


@triton.jit
def _hc_post_mixprep_kernel(
    a_ptr,
    routes_ptr,
    shared_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    output_ptr,
    flat_ptr,
    square_ptr,
    M,
    D,
    stride_am,
    stride_rtm,
    stride_rtk,
    stride_sm,
    stride_rm,
    stride_rh,
    stride_pm,
    stride_cm,
    stride_ym,
    stride_yh,
    stride_fm,
    HC: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    OUT: tl.constexpr,
    HAS_ROUTE_SUM: tl.constexpr,
    EMIT_PREP: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    if HAS_ROUTE_SUM:
        routed = tl.zeros((BLOCK,), dtype=tl.float32)
        for index in tl.static_range(TOPK):
            routed += tl.load(
                routes_ptr + m * stride_rtm + index * stride_rtk + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        shared = tl.load(
            shared_ptr + m * stride_sm + offs, mask=mask, other=0.0
        ).to(tl.float32)
        # Match the old route reduction store followed by the BF16 add store.
        activation = (shared + routed.to(OUT).to(tl.float32)).to(OUT).to(tl.float32)
    else:
        activation = tl.load(
            a_ptr + m * stride_am + offs, mask=mask, other=0.0
        ).to(tl.float32)

    for q in tl.static_range(HC):
        acc = tl.load(post_ptr + m * stride_pm + q).to(tl.float32) * activation
        for p in tl.static_range(HC):
            coefficient = tl.load(
                comb_ptr + m * stride_cm + p * HC + q
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + m * stride_rm + p * stride_rh + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += coefficient * residual
        value = acc.to(OUT)
        tl.store(
            output_ptr + m * stride_ym + q * stride_yh + offs,
            value,
            mask=mask,
        )
        if EMIT_PREP:
            widened = value.to(tl.float32)
            prep = flat_ptr + m * stride_fm + q * D + offs
            tl.store(prep, widened, mask=mask)
            tl.store(
                square_ptr + m * stride_fm + q * D + offs,
                widened * widened,
                mask=mask,
            )


@triton.jit
def _hc_sinkhorn_rmsprep_kernel(
    mixes_ptr,
    square_mean_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    n,
    stride_m,
    stride_mean,
    stride_pre,
    stride_post,
    stride_comb,
    NORM_EPS,
    HC: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n:
        return
    h = tl.arange(0, HC)
    mix = mixes_ptr + row * stride_m
    rms = tl.rsqrt(tl.load(square_mean_ptr + row * stride_mean) + NORM_EPS)
    scale_pre = tl.load(scale_ptr)
    scale_post = tl.load(scale_ptr + 1)
    scale_comb = tl.load(scale_ptr + 2)

    pre_mix = tl.load(mix + h).to(tl.float32) * rms
    pre = tl.sigmoid(pre_mix * scale_pre + tl.load(base_ptr + h)) + EPS
    tl.store(pre_ptr + row * stride_pre + h, pre)

    post_mix = tl.load(mix + HC + h).to(tl.float32) * rms
    post = 2.0 * tl.sigmoid(
        post_mix * scale_post + tl.load(base_ptr + HC + h)
    )
    tl.store(post_ptr + row * stride_post + h, post)

    index = h[:, None] * HC + h[None, :]
    comb_mix = tl.load(mix + 2 * HC + index).to(tl.float32) * rms
    value = comb_mix * scale_comb + tl.load(base_ptr + 2 * HC + index)
    value = value - tl.max(value, axis=1)[:, None]
    value = tl.exp(value)
    value = value / tl.sum(value, axis=1)[:, None]
    value += EPS
    value = value / (tl.sum(value, axis=0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        value = value / (tl.sum(value, axis=1)[:, None] + EPS)
        value = value / (tl.sum(value, axis=0)[None, :] + EPS)
    tl.store(comb_ptr + row * stride_comb + index, value)


@triton.jit
def _router_scores_exact_kernel(
    logits_ptr,
    bias_ptr,
    scores_ptr,
    biased_ptr,
    M,
    N: tl.constexpr,
    stride_lm,
    stride_sm,
    stride_bm,
    INV_TEMPERATURE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    value = tl.load(
        logits_ptr + row * stride_lm + offs, mask=mask, other=0.0
    ).to(tl.float32)
    value *= INV_TEMPERATURE
    softplus = tl.where(
        value > 20.0,
        value,
        libdevice.log1p(libdevice.exp(value)),
    )
    score = tl.sqrt_rn(softplus)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(scores_ptr + row * stride_sm + offs, score, mask=mask)
    tl.store(biased_ptr + row * stride_bm + offs, score + bias, mask=mask)


@triton.jit
def _router_wire_tail_kernel(
    gathered_ptr,
    denominator_ptr,
    indices_ptr,
    weights_ptr,
    ids_ptr,
    M,
    K: tl.constexpr,
    stride_gm,
    stride_dm,
    stride_im,
    stride_wm,
    stride_idm,
    EPS,
    ROUTE_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    denominator = tl.load(denominator_ptr + row * stride_dm) + EPS
    gathered = tl.load(
        gathered_ptr + row * stride_gm + offs, mask=mask, other=0.0
    ).to(tl.float32)
    weights = tl.div_rn(gathered, denominator) * ROUTE_SCALE
    indices = tl.load(
        indices_ptr + row * stride_im + offs, mask=mask, other=0
    ).to(tl.int32)
    tl.store(weights_ptr + row * stride_wm + offs, weights, mask=mask)
    tl.store(ids_ptr + row * stride_idm + offs, indices, mask=mask)


def hc_prepare_mix(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Widen ``x`` and square it into distinct contiguous FP32 buffers."""
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("hc_prepare_mix requires a CUDA BF16/FP16 tensor")
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"HC tensor must be [M,1,HC,D], got {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("hc_prepare_mix requires contiguous input")
    leading = x.shape[:-2]
    width = x.shape[-2] * x.shape[-1]
    source = x.contiguous()
    flat = torch.empty((*leading, width), dtype=torch.float32, device=x.device)
    square = torch.empty_like(flat)
    n = source.numel()
    block = 256
    _hc_prepare_mix_kernel[(triton.cdiv(n, block),)](
        source, flat, square, n, BLOCK=block, num_warps=4
    )
    return flat, square


def hc_post_mixprep(
    a_or_shared: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    routes: torch.Tensor | None = None,
    prepare: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Exact HC post mix with optional ordered route sum and prep epilogue."""
    if not residual.is_cuda:
        raise ValueError("hc_post_mixprep requires CUDA tensors")
    if residual.ndim != 4 or residual.shape[1] != 1:
        raise ValueError(
            f"HC residual must be [M,1,HC,D], got {tuple(residual.shape)}"
        )
    if residual.dtype not in _TL or residual.dtype == torch.float32:
        raise ValueError("HC residual must be BF16 or FP16")
    if not residual.is_contiguous():
        raise ValueError("HC residual must be contiguous")
    leading = residual.shape[:-2]
    hc, dim = residual.shape[-2:]
    residual_3d = residual.reshape(-1, hc, dim).contiguous()
    rows = residual_3d.shape[0]
    if a_or_shared.numel() != rows * dim or a_or_shared.dtype != residual.dtype:
        raise ValueError("HC activation/shared tensor geometry or dtype mismatch")
    if post.numel() != rows * hc or post.dtype != torch.float32:
        raise ValueError("HC post tensor geometry or dtype mismatch")
    if comb.numel() != rows * hc * hc or comb.dtype != torch.float32:
        raise ValueError("HC comb tensor geometry or dtype mismatch")
    activation = a_or_shared.reshape(rows, dim).contiguous()
    post_2d = post.reshape(rows, hc).contiguous()
    comb_3d = comb.reshape(rows, hc, hc).contiguous()
    output = torch.empty_like(residual_3d)
    flat = (
        torch.empty((rows, hc * dim), dtype=torch.float32, device=residual.device)
        if prepare
        else None
    )
    square = torch.empty_like(flat) if flat is not None else None
    if routes is None:
        route_view = activation
        shared = activation
        topk = 1
    else:
        if (
            routes.ndim != 3
            or routes.shape[0] != rows
            or routes.shape[-1] != dim
            or routes.dtype != residual.dtype
        ):
            raise ValueError("HC route tensor geometry or dtype mismatch")
        route_view = routes.reshape(rows, routes.shape[-2], dim).contiguous()
        shared = activation
        topk = route_view.shape[1]
    block = 1024
    _hc_post_mixprep_kernel[(rows, triton.cdiv(dim, block))](
        activation,
        route_view,
        shared,
        residual_3d,
        post_2d,
        comb_3d,
        output,
        flat if flat is not None else output,
        square if square is not None else output,
        rows,
        dim,
        activation.stride(0),
        route_view.stride(0),
        route_view.stride(1) if routes is not None else 0,
        shared.stride(0),
        residual_3d.stride(0),
        residual_3d.stride(1),
        post_2d.stride(0),
        comb_3d.stride(0),
        output.stride(0),
        output.stride(1),
        flat.stride(0) if flat is not None else 0,
        HC=hc,
        TOPK=topk,
        BLOCK=block,
        OUT=_TL[residual.dtype],
        HAS_ROUTE_SUM=routes is not None,
        EMIT_PREP=prepare,
        num_warps=4,
    )
    return (
        output.view_as(residual),
        flat.view(*leading, hc * dim) if flat is not None else None,
        square.view(*leading, hc * dim) if square is not None else None,
    )


def hc_sinkhorn_rmsprep(
    mixes: torch.Tensor,
    square_mean: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the old RMS tail and sinkhorn body in one row-local launch."""
    if not mixes.is_cuda or mixes.dtype != torch.float32:
        raise ValueError("HC mixes must be CUDA FP32")
    if hc_mult <= 0 or hc_mult & (hc_mult - 1):
        raise ValueError("hc_mult must be a positive power of two")
    if mixes.shape[-1] != (2 + hc_mult) * hc_mult:
        raise ValueError("HC mix width does not match hc_mult")
    if sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be at least one")
    if scale.numel() != 3 or base.numel() != mixes.shape[-1]:
        raise ValueError("HC scale/base geometry mismatch")
    leading = mixes.shape[:-1]
    rows = mixes.reshape(-1, mixes.shape[-1]).contiguous().float()
    means = square_mean.reshape(-1, 1).contiguous().float()
    if rows.shape[0] != means.shape[0]:
        raise ValueError("mix and square-mean row counts differ")
    count = rows.shape[0]
    pre = torch.empty((count, hc_mult), dtype=torch.float32, device=mixes.device)
    post = torch.empty_like(pre)
    comb = torch.empty((count, hc_mult, hc_mult), dtype=torch.float32, device=mixes.device)
    _hc_sinkhorn_rmsprep_kernel[(count,)](
        rows,
        means,
        scale.float().contiguous(),
        base.float().contiguous(),
        pre,
        post,
        comb,
        count,
        rows.stride(0),
        means.stride(0),
        pre.stride(0),
        post.stride(0),
        comb.stride(0),
        float(norm_eps),
        HC=hc_mult,
        ITERS=sinkhorn_iters,
        EPS=float(eps),
        num_warps=1,
    )
    return (
        pre.view(*leading, hc_mult),
        post.view(*leading, hc_mult),
        comb.view(*leading, hc_mult, hc_mult),
    )


def router_scores_exact(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    inv_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unbiased and bias-ranked exact sqrt-softplus scores."""
    if not logits.is_cuda or logits.dtype != torch.float32:
        raise ValueError("router logits must be CUDA FP32")
    if logits.ndim != 2 or bias.shape != (logits.shape[1],):
        raise ValueError("router logits/bias geometry mismatch")
    if bias.dtype != torch.float32 or not logits.is_contiguous() or not bias.is_contiguous():
        raise ValueError("router logits/bias must be contiguous FP32")
    if inv_temperature <= 0:
        raise ValueError("inverse temperature must be positive")
    rows, experts = logits.shape
    scores = torch.empty_like(logits, dtype=torch.float32)
    biased = torch.empty_like(scores)
    block = triton.next_power_of_2(experts)
    _router_scores_exact_kernel[(rows,)](
        logits,
        bias,
        scores,
        biased,
        rows,
        N=experts,
        stride_lm=logits.stride(0),
        stride_sm=scores.stride(0),
        stride_bm=biased.stride(0),
        INV_TEMPERATURE=float(inv_temperature),
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return scores, biased


def router_wire_tail(
    gathered: torch.Tensor,
    denominator: torch.Tensor,
    indices: torch.Tensor,
    wire_weights: torch.Tensor,
    wire_ids: torch.Tensor,
    *,
    eps: float,
    route_scale: float,
) -> None:
    """Normalize route weights and write weights/ids directly to the EP wire."""
    if not gathered.is_cuda or gathered.dtype != torch.float32:
        raise ValueError("router tail requires CUDA FP32 weights")
    if gathered.shape != indices.shape or gathered.shape != wire_weights.shape:
        raise ValueError("router tail tensors must have identical [M,K] shapes")
    if wire_ids.shape != indices.shape or denominator.shape != (gathered.shape[0], 1):
        raise ValueError("router denominator or id wire geometry mismatch")
    if (
        denominator.dtype != torch.float32
        or wire_weights.dtype != torch.float32
        or wire_ids.dtype != torch.int32
        or indices.dtype != torch.int64
        or not all(
            tensor.is_contiguous()
            for tensor in (gathered, denominator, indices, wire_weights, wire_ids)
        )
    ):
        raise ValueError("router tail dtype/contiguity mismatch")
    rows, topk = gathered.shape
    block = triton.next_power_of_2(topk)
    _router_wire_tail_kernel[(rows,)](
        gathered,
        denominator,
        indices,
        wire_weights,
        wire_ids,
        rows,
        K=topk,
        stride_gm=gathered.stride(0),
        stride_dm=denominator.stride(0),
        stride_im=indices.stride(0),
        stride_wm=wire_weights.stride(0),
        stride_idm=wire_ids.stride(0),
        EPS=float(eps),
        ROUTE_SCALE=float(route_scale),
        BLOCK=block,
        num_warps=1,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )


__all__ = [
    "hc_post_mixprep",
    "hc_prepare_mix",
    "hc_sinkhorn_rmsprep",
    "router_scores_exact",
    "router_wire_tail",
]
