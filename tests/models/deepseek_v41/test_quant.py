from __future__ import annotations

import torch

from freetoken.models.deepseek_v41.quant import (
    fake_quant_compressed_kv,
    fake_quant_fp4,
)


def test_indexer_fp4_uses_independent_power_of_two_scales_per_32():
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    x[0, :32] = 3.0
    x[0, 32:] = 12.0
    got = fake_quant_fp4(x)
    assert got.dtype == x.dtype and got.shape == x.shape
    assert torch.equal(got[0, :32], torch.full((32,), 3.0, dtype=x.dtype))
    assert torch.equal(got[0, 32:], torch.full((32,), 12.0, dtype=x.dtype))
    assert not got[1].any()


def test_compressed_kv_uses_e4m3_scales_per_16_not_indexer_scaling():
    torch.manual_seed(41)
    x = (torch.randn(3, 32) * 0.37).to(torch.bfloat16)
    compressed = fake_quant_compressed_kv(x)
    indexer = fake_quant_fp4(x)
    assert compressed.dtype == x.dtype
    assert not torch.equal(compressed, indexer)


def test_compressed_kv_zero_group_remains_zero_with_nonzero_scale_floor():
    x = torch.zeros(2, 32, dtype=torch.bfloat16)
    assert torch.equal(fake_quant_compressed_kv(x), x)
