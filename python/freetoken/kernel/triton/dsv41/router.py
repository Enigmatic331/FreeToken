"""Fused DeepSeek-V4.1 sqrt-softplus top-k router.

The gate GEMV remains a separate launch; this kernel collapses the temperature
scale, softplus, sqrt, bias-only ranking, top-k gather, normalization, and route
scale that otherwise become a chain of small PyTorch kernels on every layer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sqrtsoftplus_topk_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    indices_ptr,
    M,
    stride_lm,
    stride_ln,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    TEMPERATURE: tl.constexpr,
    ROUTE_SCALE: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    logits = tl.load(
        logits_ptr + row * stride_lm + offs_n * stride_ln,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    logits = logits / TEMPERATURE
    # Match torch.nn.functional.softplus(beta=1, threshold=20).
    softplus = tl.where(
        logits > 20.0,
        logits,
        tl.log(1.0 + tl.exp(logits)),
    )
    scores = tl.sqrt(softplus)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    ranked = tl.where(mask_n, scores + bias, -float("inf"))
    ranked = tl.where(ranked == ranked, ranked, -1.0e30)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    selected_scores = tl.zeros((BLOCK_K,), dtype=tl.float32)
    selected_indices = tl.zeros((BLOCK_K,), dtype=tl.int32)
    current = ranked
    for k in tl.static_range(K):
        maximum = tl.max(current, axis=0)
        lanes = tl.where(current == maximum, offs_n, N + 1)
        winner = tl.min(lanes, axis=0).to(tl.int32)
        unbiased = tl.sum(tl.where(offs_n == winner, scores, 0.0), axis=0)
        slot = offs_k == k
        selected_scores = tl.where(slot, unbiased, selected_scores)
        selected_indices = tl.where(slot, winner, selected_indices)
        current = tl.where(offs_n == winner, -float("inf"), current)

    if RENORMALIZE:
        denom = tl.sum(tl.where(mask_k, selected_scores, 0.0), axis=0) + 1.0e-20
        selected_scores = selected_scores / denom
    selected_scores *= ROUTE_SCALE

    out_mask = mask_k & (row < M)
    tl.store(
        weights_ptr + row * stride_wm + offs_k * stride_wk,
        selected_scores,
        mask=out_mask,
    )
    tl.store(
        indices_ptr + row * stride_im + offs_k * stride_ik,
        selected_indices,
        mask=out_mask,
    )


def fused_sqrtsoftplus_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    renormalize: bool,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select by ``sqrt(softplus(logits / temperature)) + bias``.

    Returned weights use the unbiased sqrt-softplus scores, matching the model
    definition.  Indices are int32, the native EP wire and offload-cache type.
    """
    if logits.ndim != 2:
        raise ValueError(f"router logits must be 2D, got {tuple(logits.shape)}")
    if bias.ndim != 1 or bias.shape[0] != logits.shape[1]:
        raise ValueError(
            f"router bias must be [{logits.shape[1]}], got {tuple(bias.shape)}"
        )
    if not 0 < topk <= logits.shape[1]:
        raise ValueError(f"topk must be in [1,{logits.shape[1]}], got {topk}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    M, N = logits.shape
    weights = torch.empty((M, topk), dtype=torch.float32, device=logits.device)
    indices = torch.empty((M, topk), dtype=torch.int32, device=logits.device)
    block_n = triton.next_power_of_2(N)
    block_k = triton.next_power_of_2(topk)
    _sqrtsoftplus_topk_kernel[(M,)](
        logits,
        bias,
        weights,
        indices,
        M,
        logits.stride(0),
        logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        TEMPERATURE=float(temperature),
        ROUTE_SCALE=float(route_scale),
        N=N,
        K=topk,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        RENORMALIZE=renormalize and topk > 1,
        num_warps=4,
    )
    return weights, indices


__all__ = ["fused_sqrtsoftplus_topk"]
