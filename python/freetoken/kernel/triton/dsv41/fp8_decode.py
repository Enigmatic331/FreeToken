"""Opt-in, exact-order native FP8 decode tuning for V4.1.

The native FP8 bytes, E8M0 scales, activation quantizer and split-K boundaries
are unchanged. Only launch geometry and loop unrolling differ. Unsupported
devices/shapes and batches larger than two rows retain the established
implementation; the two-row path evaluates each row independently.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dsv4.fp8_linear import (
    _splitk_reduce_kernel,
    _TL_DTYPE,
    act_quant_fp8,
)
from freetoken.kernel.triton.e4m3_compat import FORCE_EMU


# Read once before startup warmup/capture; never switch an already loaded graph.
ENABLED = os.getenv("FREETOKEN_DSV41_FP8_DECODE_TUNING", "0") == "1"
SM89_ENABLED = os.getenv("FREETOKEN_DSV41_SM89_FP8_DECODE_TUNING", "0") == "1"

# (N, K): (rows per CTA, unchanged split-K, warps, loop-unroll factor).
# Qualified in isolation on all 290 native checkpoint projections. Whole-model
# qualification is additionally required before enabling a deployment.
SM120_CONFIGS = {
    (512, 5120): (16, 32, 1, 1),
    (1280, 5120): (32, 16, 8, 1),
    (2304, 5120): (16, 8, 4, 2),
    (4096, 1280): (32, 4, 8, 2),
    (5120, 2304): (64, 4, 8, 2),
    (5120, 8192): (64, 4, 8, 2),
    (25600, 6144): (64, 1, 4, 2),
    (32768, 1280): (32, 1, 2, 2),
}

# Qualified in isolation on four official-checkpoint weights per shape and
# 1,006 byte-identical inputs per weight.  Full 40+40 checkpoint qualification
# and whole-model serving qualification are required before deployment.
SM89_CONFIGS = {
    (5120, 8192): (16, 4, 2, 2),
    (32768, 1280): (32, 1, 2, 2),
}


def _tuned_config(x: torch.Tensor, weight: torch.Tensor):
    """Return this tensor-local device's opt-in config, or ``None``."""
    if (not ENABLED and not SM89_ENABLED) or FORCE_EMU:
        return None
    if not x.is_cuda or weight.ndim != 2 or x.ndim < 1:
        return None
    if not x.shape[-1]:
        return None
    rows = x.numel() // x.shape[-1]
    if rows not in (1, 2):
        return None
    shape = tuple(weight.shape)
    if shape not in SM120_CONFIGS and shape not in SM89_CONFIGS:
        return None
    capability = torch.cuda.get_device_capability(x.device)
    if capability == (12, 0) and ENABLED:
        return SM120_CONFIGS.get(shape)
    if capability == (8, 9) and SM89_ENABLED:
        return SM89_CONFIGS.get(shape)
    return None


def use_tuned_decode(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Host-only routing; inspect the tensor's device, never a foreign rank."""
    return _tuned_config(x, weight) is not None


@triton.jit
def _gemv_unroll(
    a_ptr, sa_ptr, w_ptr, sb_ptr, part_ptr,
    N, K, stride_ak, stride_wn, stride_wk, stride_sbn, stride_sbk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr, SCALE_BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    sn = offs_n // SCALE_BLOCK
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in tl.range(0, k_per, SCALE_BLOCK, loop_unroll_factor=UNROLL):
        offs_k = k_start + k0 + tl.arange(0, SCALE_BLOCK)
        a = tl.load(a_ptr + offs_k * stride_ak).to(tl.float32)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        kb = (k_start + k0) // SCALE_BLOCK
        sb_code = tl.load(sb_ptr + sn * stride_sbn + kb * stride_sbk, mask=n_mask, other=0)
        scb = tl.exp2(sb_code.to(tl.float32) - 127.0)
        sa_code = tl.load(sa_ptr + kb)
        sca = tl.exp2(sa_code.to(tl.float32) - 127.0)
        acc += tl.sum(w * a[None, :], axis=1) * scb * sca
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


def tuned_fp8_linear_32(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """One/two-row tuned path; caller must check ``use_tuned_decode`` first.

    Two-row decode deliberately reuses the qualified one-row kernel independently
    for each row.  Besides avoiding the tiny generic M=2 GEMM, this preserves the
    exact C1 split-K boundaries and reduction order for both concurrent requests.
    """
    assert weight.dtype == torch.float8_e4m3fn
    *lead, k = x.shape
    n, weight_k = weight.shape
    rows = x.numel() // k
    assert weight_k == k and rows in (1, 2)
    assert scale.shape == (n // 32, k // 32)
    if rows == 2:
        flat = x.reshape(2, k)
        out = torch.cat(
            [
                tuned_fp8_linear_32(flat[i : i + 1], weight, scale, bias)
                for i in range(2)
            ],
            dim=0,
        )
        return out.reshape(*lead, n)
    config = _tuned_config(x, weight)
    assert config is not None
    bn, sk, nw, unroll = config
    compute_dtype = x.dtype if x.dtype in _TL_DTYPE else torch.bfloat16
    sb = scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale
    sb = sb.contiguous()
    aq, sa = act_quant_fp8(x, 32)
    a, s = aq[0], sa[0]
    part = torch.empty((sk, n), device=x.device, dtype=torch.float32)
    _gemv_unroll[(triton.cdiv(n, bn), sk)](
        a, s, weight, sb, part, n, k, a.stride(0), weight.stride(0), weight.stride(1),
        sb.stride(0), sb.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=bn, SPLIT_K=sk, SCALE_BLOCK=32, UNROLL=unroll, num_warps=nw,
    )
    out = torch.empty(n, device=x.device, dtype=compute_dtype)
    _splitk_reduce_kernel[(triton.cdiv(n, 256),)](
        part, out, n, sk, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[compute_dtype], num_warps=2,
    )
    out = out.reshape(*lead, n)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


__all__ = ["use_tuned_decode", "tuned_fp8_linear_32"]
