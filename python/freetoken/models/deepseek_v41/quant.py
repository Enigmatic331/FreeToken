"""Torch precision oracles for V4.1's two FP4 activation formats."""

from __future__ import annotations

import torch


def _ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    bits = x.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    exponent += ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((exponent + 127) << 23).view(torch.float32)


def round_fp4(x: torch.Tensor) -> torch.Tensor:
    magnitude = x.abs()
    step = torch.where(
        magnitude < 2.0,
        0.5,
        torch.where(magnitude < 4.0, 1.0, 2.0),
    )
    return torch.round(magnitude / step) * step * torch.sign(x)


def fake_quant_fp4(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(-1, keepdim=True)
    scale = _ceil_pow2(amax.clamp_min(6.0 * 2.0**-126) * (1.0 / 6.0))
    return (
        round_fp4((blocks / scale).clamp(-6.0, 6.0)) * scale
    ).flatten(-2).to(x.dtype)


def fake_quant_compressed_kv(x: torch.Tensor) -> torch.Tensor:
    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(-1, keepdim=True)
    scale = (amax * (1.0 / 6.0)).clamp(2**-9, 448.0)
    scale = scale.to(torch.float8_e4m3fn).float()
    return (
        round_fp4((blocks / scale).clamp(-6.0, 6.0)) * scale
    ).flatten(-2).to(x.dtype)


__all__ = ["fake_quant_compressed_kv", "fake_quant_fp4", "round_fp4"]
