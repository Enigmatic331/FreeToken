"""Fused RoPE-tail plus V4.1 FP4 fake-quant round trip.

Adapted from SGLang's Apache-2.0 DeepSeek-V4.1 kernel.  Index queries/keys use
32-wide E8M0 scales; compressed KV uses the checkpoint-faithful 16-wide E4M3
scale.  ``rope_fp4_roundtrip`` returns the rounded values as BF16;
``rope_fp4_pack`` emits the same compressed-KV values packed (nibbles plus raw E4M3
scale bytes, ``kvcache/dsv41_packed_kv.py``), which unpack to identical BF16 bits.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from freetoken.kernel.triton.e4m3_compat import e4m3_native, e4m3_native_cx, round_e4m3


@triton.jit
def _rope_fp4_kernel(
    x_ptr,
    freq_ptr,
    out_ptr,
    q_ptr,
    s_ptr,
    dest_rows_ptr,
    x_stride_row,
    out_stride_row,
    freq_stride_token,
    rows_per_token,
    D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK: tl.constexpr,
    INVERSE: tl.constexpr,
    COMPRESSED_KV: tl.constexpr,
    PACK: tl.constexpr,
    INDEXED_PACK: tl.constexpr,
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
    grid_value = libdevice.rint(magnitude / step) * step
    if PACK:
        # E2M1 code of the grid value, sign from the pre-rounding value so a rounded-away
        # negative keeps the -0.0 the BF16 path stores. 16-bit words (kvcache/dsv41_packed_kv.py):
        # nibble word w holds channels w, w + D/4, w + D/2, w + 3D/4; scale word g holds the
        # E4M3 bytes of groups 2g and 2g + 1.
        code = tl.where(
            grid_value < 2.0,
            grid_value * 2.0,
            tl.where(grid_value < 4.0, grid_value + 2.0, grid_value * 0.5 + 4.0),
        ).to(tl.int32)
        nibble = tl.reshape(code | ((scaled < 0).to(tl.int32) << 3), (4, D // 4))
        words = tl.sum(nibble << (tl.arange(0, 4) * 4)[:, None], axis=0)  # disjoint bit fields
        dest_row = row
        if INDEXED_PACK:
            dest_row = tl.load(dest_rows_ptr + row)
        tl.store(q_ptr + dest_row * (D // 4) + tl.arange(0, D // 4), words.to(tl.int16))
        scale_bytes = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True).to(tl.int32)
        low, high = tl.split(tl.reshape(scale_bytes, (D // (2 * BLOCK), 2)))
        tl.store(
            s_ptr + dest_row * (D // (2 * BLOCK)) + tl.arange(0, D // (2 * BLOCK)),
            (low | (high << 8)).to(tl.int16),
        )
    else:
        quantized = grid_value * sign
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
        out,
        out,
        out,
        dim,
        dim,
        freq_real.stride(0),
        rows_per_token,
        D=dim,
        ROPE_D=rope_dim,
        BLOCK=block,
        INVERSE=inverse,
        COMPRESSED_KV=compressed_kv,
        PACK=False,
        INDEXED_PACK=False,
        num_warps=4,
    )
    return out


def rope_fp4_pack(x: torch.Tensor, freqs: torch.Tensor, rope_dim: int):
    """``rope_fp4_roundtrip(x, freqs, rope_dim, compressed_kv=True)`` for a packed pool:
    the same rounding, emitted as ``PackedRows`` (``x`` is ``[rows, dim]``)."""
    from freetoken.kvcache.dsv41_packed_kv import GROUP, PackedRows

    if not x.is_cuda or not freqs.is_cuda:
        raise ValueError("V4.1 fused RoPE/FP4 requires CUDA tensors")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"V4.1 fused RoPE/FP4 requires BF16 input, got {x.dtype}")
    if not e4m3_native(x.device):
        raise ValueError("packed V4.1 KV needs native E4M3 conversion (sm_89 or newer)")
    if x.dim() != 2:
        raise ValueError(f"rope_fp4_pack takes [rows, dim], got {tuple(x.shape)}")
    rows, dim = x.shape
    if dim % (8 * GROUP) or rope_dim % 2 or rope_dim > dim:
        raise ValueError(f"invalid FP4/RoPE geometry: dim={dim}, rope_dim={rope_dim}")
    out = PackedRows.empty(rows, dim, x.device)
    if rows == 0:
        return out
    x = x.contiguous()
    freq_real = torch.view_as_real(freqs.contiguous()).contiguous()
    _rope_fp4_kernel[(rows,)](
        x,
        freq_real,
        x,
        out.q,
        out.s,
        out.q,
        dim,
        dim,
        freq_real.stride(0),
        1,
        D=dim,
        ROPE_D=rope_dim,
        BLOCK=GROUP,
        INVERSE=False,
        COMPRESSED_KV=True,
        PACK=True,
        INDEXED_PACK=False,
        num_warps=4,
    )
    return out


def rope_fp4_pack_into(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    out,
    rows: torch.Tensor,
):
    """Pack directly into ``out[rows]`` without temporary rows or indexed-copy kernels.

    This is the graph-captured serving writer.  ``rows`` may contain duplicate scratch
    destinations: as with the former pair of ``index_copy_`` calls, their final contents are
    unspecified and harmless because scratch rows are never addressable compressed history.
    """
    from freetoken.kvcache.dsv41_packed_kv import GROUP, PackedRows

    if not isinstance(out, PackedRows):
        raise TypeError("rope_fp4_pack_into requires a PackedRows destination")
    if not x.is_cuda or not freqs.is_cuda or not rows.is_cuda:
        raise ValueError("V4.1 fused RoPE/FP4 requires CUDA tensors")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"V4.1 fused RoPE/FP4 requires BF16 input, got {x.dtype}")
    if not e4m3_native(x.device):
        raise ValueError("packed V4.1 KV needs native E4M3 conversion (sm_89 or newer)")
    if x.dim() != 2:
        raise ValueError(f"rope_fp4_pack_into takes [rows, dim], got {tuple(x.shape)}")
    n_rows, dim = x.shape
    if rows.dim() != 1 or rows.numel() != n_rows:
        raise ValueError(
            f"destination rows must be [{n_rows}], got {tuple(rows.shape)}"
        )
    if rows.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"destination rows must be int32 or int64, got {rows.dtype}")
    if x.device != freqs.device or x.device != rows.device or x.device != out.device:
        raise ValueError("input, frequencies, rows and packed destination must share a device")
    if dim != out.head_dim:
        raise ValueError(f"packed destination has dim={out.head_dim}, input has dim={dim}")
    if dim % (8 * GROUP) or rope_dim % 2 or rope_dim > dim:
        raise ValueError(f"invalid FP4/RoPE geometry: dim={dim}, rope_dim={rope_dim}")
    if not rows.is_contiguous():
        raise ValueError("destination rows must be contiguous")
    if n_rows == 0:
        return out
    x = x.contiguous()
    freq_real = torch.view_as_real(freqs.contiguous()).contiguous()
    _rope_fp4_kernel[(n_rows,)](
        x,
        freq_real,
        x,
        out.q,
        out.s,
        rows,
        dim,
        dim,
        freq_real.stride(0),
        1,
        D=dim,
        ROPE_D=rope_dim,
        BLOCK=GROUP,
        INVERSE=False,
        COMPRESSED_KV=True,
        PACK=True,
        INDEXED_PACK=True,
        num_warps=4,
    )
    return out


__all__ = ["rope_fp4_pack", "rope_fp4_pack_into", "rope_fp4_roundtrip"]
