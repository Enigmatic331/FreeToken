# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (sglang/kernels/ops/embeddings/engram_gather.py).
"""UVA gather for DeepSeek-V4.1's FP8/E8M0 Engram tables.

The raw pointers may address CUDA memory or a mapped, pinned host allocation.  A
rank owns the half-open global row range ``[row_lo, row_hi)``.  Non-owned ids
produce zero rows so the small BF16 result can be summed across the EP/TP pair.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32

_E8M0_ZERO = 2.0**-127


@triton.jit
def _engram_gather_kernel(
    weight_ptr,
    scale_ptr,
    ids_ptr,
    out_ptr,
    row_lo,
    row_hi,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    global_id = tl.load(ids_ptr + row).to(tl.int64)
    owned = (global_id >= row_lo) & (global_id < row_hi)
    local_id = tl.where(owned, global_id - row_lo, 0)
    offsets = tl.arange(0, DIM)

    if e4m3_native_cx():
        weight = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
        values = tl.load(
            weight + local_id * DIM + offsets,
            mask=owned,
            other=0.0,
        ).to(tl.float32)
    else:
        weight = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        values = e4m3_u8_to_f32(
            tl.load(weight + local_id * DIM + offsets, mask=owned, other=0)
        )

    scales = scale_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    exponents = tl.load(
        scales + local_id * (DIM // BLOCK) + offsets // BLOCK,
        mask=owned,
        other=0,
    ).to(tl.int32)
    # Construct 2**(exponent - 127) exactly from IEEE-754 exponent bits.  E8M0
    # code zero is the exceptional 2**-127 value rather than FP32 zero.
    factor = (exponents << 23).to(tl.float32, bitcast=True)
    factor = tl.where(exponents == 0, E8M0_ZERO, factor)
    values = tl.where(owned, values * factor, 0.0)
    tl.store(out_ptr + row * DIM + offsets, values.to(tl.bfloat16))


def engram_gather_rows(
    weight_ptr: int,
    scale_ptr: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
    *,
    dim: int,
    block_size: int = 32,
    row_lo: int = 0,
    row_hi: int = 2**62,
) -> torch.Tensor:
    """Gather flat device ``row_ids`` into contiguous BF16 ``out [N, dim]``."""

    if dim <= 0 or dim & (dim - 1) or dim % block_size:
        raise ValueError(f"Engram dim must be power-of-two and block aligned: {dim=}, {block_size=}")
    if row_ids.device.type != "cuda":
        raise ValueError("Engram UVA gather requires device row ids")
    if out.shape != (row_ids.numel(), dim) or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(
            f"Engram output must be contiguous bf16 [{row_ids.numel()}, {dim}], "
            f"got {tuple(out.shape)} {out.dtype} contiguous={out.is_contiguous()}"
        )
    if out.device != row_ids.device:
        raise ValueError("Engram ids and output must use the same CUDA device")
    if row_ids.numel():
        _engram_gather_kernel[(row_ids.numel(),)](
            weight_ptr,
            scale_ptr,
            row_ids,
            out,
            row_lo,
            row_hi,
            DIM=dim,
            BLOCK=block_size,
            E8M0_ZERO=_E8M0_ZERO,
            num_warps=1,
        )
    return out


__all__ = ["engram_gather_rows"]
