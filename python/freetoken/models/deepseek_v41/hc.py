"""Exact V4.1 predecessor-carried manifold Hyper-Connection transitions."""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from freetoken.utils import init_logger

logger = init_logger(__name__)

_DIRECT_PRE_ENV = "FREETOKEN_DSV41_HC_DIRECT_PRE"


def _direct_pre_mode() -> str:
    """Strict parse of the opt-in direct BF16/FP16 HC-pre input.

    Unset or ``0`` is ``off``. ``prefill`` feeds the original-dtype streams to the unchanged
    combine kernel only from the prefill call sites, selected by the explicit execution phase
    (the ``prefill=True`` keyword those call sites pass), never by row count: a C2 decode batch
    may have two rows. ``1`` is the earlier all-phase variant, kept for reproducibility.
    """
    value = os.environ.get(_DIRECT_PRE_ENV)
    if value is None or value == "0":
        return "off"
    if value == "1":
        return "all"
    if value == "prefill":
        return "prefill"
    raise ValueError(f"{_DIRECT_PRE_ENV} must be '0', '1' or 'prefill', got {value!r}")


_DIRECT_PRE_MODE = _direct_pre_mode()
# All phases (legacy ``1``); tests toggle this name directly.
_DIRECT_PRE = _DIRECT_PRE_MODE == "all"
# Prefill call sites only (``prefill``).
_DIRECT_PRE_PREFILL = _DIRECT_PRE_MODE == "prefill"


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


def _check_prefill_phase() -> None:
    """Defence in depth: a prefill-only path must never run inside a decode forward."""
    from freetoken import core

    batch = getattr(getattr(core, "_GLOBAL_CTX", None), "_batch", None)
    if batch is not None and not batch.is_prefill:
        raise RuntimeError("HC prefill direct-pre requested from a non-prefill forward")


def hc_pre(x: torch.Tensor, incoming_pre: torch.Tensor, *, prefill: bool = False) -> torch.Tensor:
    """Collapse streams with the mix emitted by the preceding sublayer.

    ``prefill`` is the caller's explicit execution phase (``True`` only from the prefill
    block/model paths); it selects the opt-in prefill-only direct input and nothing else.
    """

    leading, hc_mult, dim = x.shape[:-2], x.shape[-2], x.shape[-1]
    if x.is_cuda:
        from freetoken.kernel.triton.dsv4.hc import hc_pre_combine

        streams = x.reshape(-1, hc_mult, dim)
        direct = x.dtype in (torch.bfloat16, torch.float16) and (
            _DIRECT_PRE or (prefill and _DIRECT_PRE_PREFILL)
        )
        if direct and prefill:
            _check_prefill_phase()
        if not direct:
            streams = streams.float()
        # The combine kernel promotes each stream to FP32 before the same
        # multiply/add sequence. Avoid materializing an entire FP32 copy.
        result = hc_pre_combine(streams, incoming_pre.reshape(-1, hc_mult), x.dtype)
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


def warmup_direct_pre(device: torch.device, dim: int, hc_mult: int) -> None:
    """Load authority-local BF16-input modules before any unmatched receive."""
    if not (_DIRECT_PRE or _DIRECT_PRE_PREFILL) or device.type != "cuda":
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("HC direct-pre warmup must precede graph capture")
    # Cover Triton's scalar-one, divisible-by-16, and generic row variants.
    for rows in (1, 2, 16):
        x = torch.zeros((rows, hc_mult, dim), device=device, dtype=torch.bfloat16)
        pre = torch.zeros((rows, hc_mult), device=device, dtype=torch.float32)
        # Startup has no active forward; the prefill keyword selects the direct variant.
        hc_pre(x, pre, prefill=True)
    torch.cuda.synchronize(device)
    logger.info("DSV4.1 HC direct-pre mode=%s: BF16 combine warmed for rows 1/2/16",
                _DIRECT_PRE_MODE)


__all__ = ["hc_mixes", "hc_post", "hc_pre", "identity_pre_mix", "warmup_direct_pre"]
