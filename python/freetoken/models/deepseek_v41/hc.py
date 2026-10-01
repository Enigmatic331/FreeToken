"""Exact V4.1 predecessor-carried manifold Hyper-Connection transitions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    result = torch.zeros(*x.shape[:-2], hc_mult, dtype=torch.float32, device=x.device)
    result[..., 0] = 1.0
    return result


def hc_mixes(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive the pre/post/combination coefficients produced by this sublayer."""

    leading = x.shape[:-2]
    flat = x.flatten(-2).float()
    mixes = F.linear(flat, weight.float())
    # The FP32 projection input is dead after ``linear``.  Reuse it for the
    # RMS reduction rather than materializing another full-size ``square``
    # tensor (640 MiB for an 8K DSV4.1 prefill chunk).  CUDA stream ordering
    # keeps the in-place write behind the linear kernel's final read.
    flat.square_()
    mixes *= torch.rsqrt(flat.mean(-1, keepdim=True) + norm_eps)
    rows = mixes.reshape(-1, mixes.shape[-1])
    if rows.is_cuda:
        from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn

        pre, post, comb = hc_split_sinkhorn(
            rows, scale, base, hc_mult, sinkhorn_iters, eps
        )
    else:
        pre = torch.sigmoid(rows[:, :hc_mult] * scale[0] + base[:hc_mult]) + eps
        post = 2 * torch.sigmoid(
            rows[:, hc_mult : 2 * hc_mult] * scale[1]
            + base[hc_mult : 2 * hc_mult]
        )
        comb = rows[:, 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]
        comb = comb.view(-1, hc_mult, hc_mult).softmax(-1) + eps
        comb /= comb.sum(-2, keepdim=True) + eps
        for _ in range(sinkhorn_iters - 1):
            comb /= comb.sum(-1, keepdim=True) + eps
            comb /= comb.sum(-2, keepdim=True) + eps
    return (
        pre.view(*leading, hc_mult),
        post.view(*leading, hc_mult),
        comb.view(*leading, hc_mult, hc_mult),
    )


def hc_pre(x: torch.Tensor, incoming_pre: torch.Tensor) -> torch.Tensor:
    """Collapse streams with the mix emitted by the preceding sublayer."""

    leading, hc_mult, dim = x.shape[:-2], x.shape[-2], x.shape[-1]
    if x.is_cuda:
        from freetoken.kernel.triton.dsv4.hc import hc_pre_combine

        result = hc_pre_combine(
            x.reshape(-1, hc_mult, dim).float(), incoming_pre.reshape(-1, hc_mult), x.dtype
        )
        return result.view(*leading, dim)
    return (incoming_pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)


def hc_post(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Inject this sublayer output and mix its predecessor residual streams."""

    leading, hc_mult, dim = residual.shape[:-2], residual.shape[-2], residual.shape[-1]
    if output.is_cuda:
        from freetoken.kernel.triton.dsv4.hc import hc_post_combine

        result = hc_post_combine(
            output.reshape(-1, dim),
            residual.reshape(-1, hc_mult, dim),
            post.reshape(-1, hc_mult),
            comb.reshape(-1, hc_mult, hc_mult),
        )
        return result.view(*leading, hc_mult, dim)
    result = post.unsqueeze(-1) * output.unsqueeze(-2)
    result += (comb.unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(-3)
    return result.to(output.dtype)


__all__ = ["hc_mixes", "hc_post", "hc_pre", "identity_pre_mix"]
