"""Graph-safe transient BF16 staging for packed V4.1 decode attention.

The exact packed attention kernel preserves the BF16 accumulation order by issuing four
quarter-width QK and PV dot chains.  At decode batch sizes that arithmetic costs much more than
the bytes it saves.  This path gathers only the selected packed rows into a small contiguous
BF16 slab, remaps their columns, then reuses the original full-width BF16 attention kernel.
The packed representation is lossless, so the staged rows and attention output remain bit exact.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_bf16(nibble, scale_byte):
    """Packed E2M1 times E4M3 -> its exact BF16 bits, including signed zero."""
    code = nibble & 7
    fraction = tl.where(code == 1, 0, code & 1)
    exponent = (code >> 1) - 1
    scale_exponent = (scale_byte >> 3) & 0xF
    scale_mantissa = scale_byte & 7
    significand = tl.where(
        scale_exponent > 0,
        (2 + fraction) * (8 + scale_mantissa),
        (2 + fraction) * scale_mantissa,
    )
    power = tl.where(
        scale_exponent > 0,
        exponent + scale_exponent - 11,
        exponent - 10,
    )
    msb = tl.where(
        significand >= 32,
        5,
        tl.where(
            significand >= 16,
            4,
            tl.where(
                significand >= 8,
                3,
                tl.where(significand >= 4, 2, tl.where(significand >= 2, 1, 0)),
            ),
        ),
    )
    bits = ((msb + power + 127) << 7) | ((significand << (7 - msb)) & 0x7F)
    bits = tl.where((code == 0) | (significand == 0), 0, bits)
    bits = bits | ((nibble >> 3) << 15)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _stage_packed_decode_kernel(
    q_ptr,
    s_ptr,
    idx_ptr,
    count_ptr,
    staged_ptr,
    remapped_ptr,
    stride_idx_query,
    N_WINDOW: tl.constexpr,
    N_COMPRESSED: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
):
    row_block = tl.program_id(0)
    query = tl.program_id(1)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    dims = tl.arange(0, D)
    live = N_COMPRESSED
    if HAS_COUNTS:
        live = tl.load(count_ptr + query)
    within = rows < N_COMPRESSED
    source = tl.load(
        idx_ptr + query * stride_idx_query + N_WINDOW + rows,
        mask=within,
        other=-1,
    )
    valid = within & (rows < live) & (source >= 0)
    words = tl.load(
        q_ptr + source[:, None] * (D // 4) + (dims % (D // 4))[None, :],
        mask=valid[:, None],
        other=0,
    ).to(tl.int32) & 0xFFFF
    nibble = (words >> (4 * (dims // (D // 4)))[None, :]) & 0xF
    group = dims // 16
    scale_words = tl.load(
        s_ptr + source[:, None] * (D // 32) + (group // 2)[None, :],
        mask=valid[:, None],
        other=0,
    ).to(tl.int32) & 0xFFFF
    scale_byte = (scale_words >> ((group % 2) * 8)[None, :]) & 0xFF
    value = _fp4_bf16(nibble, scale_byte)
    destination = query * N_COMPRESSED + rows
    tl.store(
        staged_ptr + destination[:, None] * D + dims[None, :],
        value,
        mask=within[:, None],
    )
    tl.store(
        remapped_ptr + query * stride_idx_query + N_WINDOW + rows,
        tl.where(valid, destination, -1),
        mask=within,
    )

    # One row block per query also carries over its small window-index prefix.
    window_columns = tl.arange(0, N_WINDOW)
    first_block = row_block == 0
    tl.store(
        remapped_ptr + query * stride_idx_query + window_columns,
        tl.load(
            idx_ptr + query * stride_idx_query + window_columns,
            mask=first_block,
            other=-1,
        ),
        mask=first_block,
    )


def stage_packed_decode(
    packed,
    idx: torch.Tensor,
    n_window: int,
    cmp_counts: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a transient BF16 compressed pool and identically ordered remapped indices.

    The serving use is ``m == 1`` with 512 compressed columns.  The implementation also handles
    C2 batch two; each query owns a disjoint slice of the transient pool.
    """
    from freetoken.kvcache.dsv41_packed_kv import PackedRows

    if not isinstance(packed, PackedRows):
        raise TypeError("stage_packed_decode requires PackedRows")
    if idx.dtype != torch.int32 or not idx.is_cuda or not idx.is_contiguous():
        raise ValueError("stage_packed_decode requires contiguous CUDA int32 indices")
    if idx.dim() != 3 or idx.shape[1] != 1:
        raise ValueError(f"packed decode staging requires [batch, 1, topk], got {tuple(idx.shape)}")
    if packed.head_dim != 512 or n_window != 128:
        raise ValueError(
            f"packed decode staging is qualified for D=512, window=128; got {packed.head_dim}, {n_window}"
        )
    n_compressed = idx.shape[-1] - n_window
    if n_compressed <= 0:
        raise ValueError("packed decode staging requires compressed columns")
    batch = idx.shape[0]
    staged = torch.empty(
        batch * n_compressed,
        packed.head_dim,
        dtype=torch.bfloat16,
        device=idx.device,
    )
    remapped = torch.empty_like(idx)
    has_counts = cmp_counts is not None
    if has_counts:
        counts = cmp_counts.contiguous().to(torch.int32).view(-1)
        if counts.numel() != batch:
            raise ValueError(f"compressed counts must have {batch} elements, got {counts.numel()}")
    else:
        counts = idx
    block_rows = 2
    _stage_packed_decode_kernel[(triton.cdiv(n_compressed, block_rows), batch)](
        packed.q,
        packed.s,
        idx,
        counts,
        staged,
        remapped,
        idx.stride(1),
        N_WINDOW=n_window,
        N_COMPRESSED=n_compressed,
        D=packed.head_dim,
        BLOCK_R=block_rows,
        HAS_COUNTS=has_counts,
        num_warps=8,
        num_stages=1,
    )
    return staged, remapped


__all__ = ["stage_packed_decode"]
