"""DeepSeek-V4.1 32x32 block-scaled FP8 linear wrapper."""

from __future__ import annotations

import torch

from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear
from freetoken.kernel.triton.dsv41.fp8_decode import use_tuned_decode, tuned_fp8_linear_32


def block_fp8_linear_32(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply V4.1's FP8-e4m3/E8M0 linear with 32x32 scale blocks."""

    if use_tuned_decode(x, weight):
        return tuned_fp8_linear_32(x, weight, scale, bias)
    return block_fp8_linear(x, weight, scale, bias, block_size=32)


__all__ = ["block_fp8_linear_32"]
