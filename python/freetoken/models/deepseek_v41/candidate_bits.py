"""Lossless eager-prefill candidate bits; imported only by the opt-in path."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def packed_width(key_width: int, block_size: int) -> int:
    if key_width < 0 or block_size <= 0:
        raise ValueError("nonnegative key width and positive block size required")
    blocks = (key_width + block_size - 1) // block_size
    return (blocks + 7) // 8


@triton.jit(do_not_specialize=["WIDTH", "BYTES", "IN_STRIDE", "OUT_STRIDE"],
            do_not_specialize_on_alignment=["mask", "packed"])
def _pack(mask, packed, WIDTH, BYTES, IN_STRIDE, OUT_STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    byte = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.zeros((BLOCK,), dtype=tl.int32)
    for bit in tl.static_range(8):
        column = byte * 8 + bit
        flag = tl.load(mask + row * IN_STRIDE + column,
                       mask=column < WIDTH, other=0).to(tl.int32)
        value = value | (flag << bit)
    tl.store(packed + row * OUT_STRIDE + byte, value.to(tl.uint8), mask=byte < BYTES)


@triton.jit(do_not_specialize=["WIDTH", "IN_STRIDE", "OUT_STRIDE"],
            do_not_specialize_on_alignment=["packed", "output"])
def _expand(packed, output, WIDTH, BLOCK_SIZE: tl.constexpr,
            IN_STRIDE, OUT_STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    bit_index = column // BLOCK_SIZE
    byte = tl.load(packed + row * IN_STRIDE + bit_index // 8,
                   mask=column < WIDTH, other=0).to(tl.int32)
    flag = ((byte >> (bit_index % 8)) & 1) != 0
    tl.store(output + row * OUT_STRIDE + column, flag, mask=column < WIDTH)


def pack_into(mask: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    if (mask.ndim != 2 or mask.dtype != torch.bool or not mask.is_cuda
            or mask.stride(1) != 1):
        raise ValueError("expected CUDA row-major bool block mask")
    expected = (mask.shape[0], (mask.shape[1] + 7) // 8)
    if (output.shape != expected or output.dtype != torch.uint8
            or output.device != mask.device or output.stride(1) != 1):
        raise ValueError("expected matching CUDA byte publication")
    if mask.numel():
        _pack[(mask.shape[0], triton.cdiv(expected[1], 256))](
            mask, output, WIDTH=mask.shape[1], BYTES=expected[1],
            IN_STRIDE=mask.stride(0), OUT_STRIDE=output.stride(0), BLOCK=256,
            num_warps=4)
    return output


def expand(packed: torch.Tensor, key_width: int, block_size: int) -> torch.Tensor:
    expected = packed_width(key_width, block_size)
    if (packed.ndim != 2 or packed.dtype != torch.uint8 or not packed.is_cuda
            or packed.shape[1] != expected or packed.stride(1) != 1):
        raise ValueError("expected CUDA row-major packed byte publication")
    output = torch.empty((packed.shape[0], key_width), dtype=torch.bool, device=packed.device)
    if output.numel():
        _expand[(output.shape[0], triton.cdiv(key_width, 1024))](
            packed, output, WIDTH=key_width, BLOCK_SIZE=block_size,
            IN_STRIDE=packed.stride(0), OUT_STRIDE=output.stride(0), BLOCK=1024,
            num_warps=4)
    return output


def warmup(device: torch.device, *, block_sizes: tuple[int, ...] = (8,)) -> None:
    """Load shape-independent eager kernels before model collectives begin.

    Dimensions, strides and pointer alignment cannot create later variants.
    Each model-configured block size is warmed; OFF never imports this module.
    Repeat on runtime rebind so every authority CUDA context is covered.
    """
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("candidate-bit warmup must run outside CUDA graph capture")
        for block_size in block_sizes:
            if block_size <= 0:
                raise ValueError("candidate block size must be positive")
            mask = torch.ones((2, 65), dtype=torch.bool, device=device)
            packed = torch.empty((2, 9), dtype=torch.uint8, device=device)
            pack_into(mask, packed)
            expanded = expand(packed, 65 * block_size, block_size)
        torch.cuda.current_stream(device).synchronize()
