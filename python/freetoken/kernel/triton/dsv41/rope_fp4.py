"""Fused RoPE-tail plus V4.1 FP4 fake-quant round trip.

Adapted from SGLang's Apache-2.0 DeepSeek-V4.1 kernel.  Index queries/keys use
32-wide E8M0 scales; compressed KV uses the checkpoint-faithful 16-wide E4M3
scale.  The output remains BF16 because FreeToken stores dequantized KV rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, round_e4m3


@triton.jit
def _rope_fp4_kernel(
    x_ptr,
    freq_ptr,
    out_ptr,
    x_stride_row,
    out_stride_row,
    freq_stride_token,
    rows_per_token,
    D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK: tl.constexpr,
    INVERSE: tl.constexpr,
    COMPRESSED_KV: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // rows_per_token
    offsets = tl.arange(0, D)
    value = tl.load(x_ptr + row * x_stride_row + offsets).to(tl.float32)

    head = D - ROPE_D
    in_tail = offsets >= head
    pair = (offsets - head) // 2
    imaginary = ((offsets - head) % 2) == 1
    real = tl.load(
        x_ptr + row * x_stride_row + head + 2 * pair,
        mask=in_tail,
        other=0.0,
    ).to(tl.float32)
    imag = tl.load(
        x_ptr + row * x_stride_row + head + 2 * pair + 1,
        mask=in_tail,
        other=0.0,
    ).to(tl.float32)
    fr = tl.load(
        freq_ptr + token * freq_stride_token + 2 * pair,
        mask=in_tail,
        other=1.0,
    )
    fi = tl.load(
        freq_ptr + token * freq_stride_token + 2 * pair + 1,
        mask=in_tail,
        other=0.0,
    )
    if INVERSE:
        fi = -fi
    rotated = tl.where(imaginary, real * fi + imag * fr, real * fr - imag * fi)
    # Reference rope_tail returns in x.dtype before fake quantization.
    rotated = rotated.to(tl.bfloat16).to(tl.float32)
    value = tl.where(in_tail, rotated, value)

    blocks = tl.reshape(value, (D // BLOCK, BLOCK))
    amax = tl.max(tl.abs(blocks), axis=1)
    if COMPRESSED_KV:
        scale = tl.minimum(tl.maximum(amax * (1.0 / 6.0), 2.0**-9), 448.0)
        if e4m3_native_cx():
            scale = scale.to(tl.float8e4nv).to(tl.float32)
        else:
            scale = round_e4m3(scale)
        scaled = tl.div_rn(blocks, scale[:, None])
    else:
        amax = tl.maximum(amax, 6.0 * (2.0**-126)) * (1.0 / 6.0)
        bits = amax.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 0xFF) - 127
        exponent += ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
        scaled = blocks / scale[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -6.0), 6.0)
    magnitude = tl.abs(scaled)
    step = tl.where(magnitude < 2.0, 0.5, tl.where(magnitude < 4.0, 1.0, 2.0))
    sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
    quantized = libdevice.rint(magnitude / step) * step * sign
    out = tl.reshape(quantized * scale[:, None], (D,))
    tl.store(
        out_ptr + row * out_stride_row + offsets,
        out.to(out_ptr.dtype.element_ty),
    )


def rope_fp4_roundtrip(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool = False,
    compressed_kv: bool = False,
) -> torch.Tensor:
    """Return RoPE-rotated and FP4-rounded values in ``x.dtype``.

    ``x`` is ``[tokens, ..., dim]`` and ``freqs`` is complex
    ``[tokens, rope_dim // 2]``.  Compressed KV selects E4M3-per-16 scaling;
    otherwise the indexer selects E8M0-per-32 scaling.
    """

    if not x.is_cuda or not freqs.is_cuda:
        raise ValueError("V4.1 fused RoPE/FP4 requires CUDA tensors")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"V4.1 fused RoPE/FP4 requires BF16 input, got {x.dtype}")
    block = 16 if compressed_kv else 32
    dim = x.shape[-1]
    if dim % block or rope_dim % 2 or rope_dim > dim:
        raise ValueError(
            f"invalid FP4/RoPE geometry: dim={dim}, block={block}, rope_dim={rope_dim}"
        )
    x = x.contiguous()
    freq_real = torch.view_as_real(freqs.contiguous()).contiguous()
    out = torch.empty_like(x)
    rows = x.numel() // dim
    if rows == 0:
        return out
    rows_per_token = rows // x.shape[0]
    _rope_fp4_kernel[(rows,)](
        x.reshape(-1, dim),
        freq_real,
        out.reshape(-1, dim),
        dim,
        dim,
        freq_real.stride(0),
        rows_per_token,
        D=dim,
        ROPE_D=rope_dim,
        BLOCK=block,
        INVERSE=inverse,
        COMPRESSED_KV=compressed_kv,
        num_warps=4,
    )
    return out


__all__ = ["rope_fp4_roundtrip"]
