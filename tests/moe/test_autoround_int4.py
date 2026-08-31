import pytest
import torch
import torch.nn.functional as F

from freetoken.moe.fused_autoround import (
    fused_experts_autoround,
    fused_experts_decode_autoround,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _pack(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.int32).unflatten(-1, (-1, 8))
    shifts = torch.arange(8, device=codes.device, dtype=torch.int32) * 4
    return torch.sum(codes << shifts, dim=-1).to(torch.int32)


def _dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    group = torch.arange(codes.shape[-1], device=codes.device) // 128
    return (codes.float() - 8.0) * scales[..., group].float()


def test_autoround_decode_matches_torch():
    torch.manual_seed(7)
    device = torch.device("cuda")
    experts, tokens, top_k, hidden, inter = 4, 2, 2, 256, 128
    dtype = torch.bfloat16

    gu_codes = torch.randint(0, 16, (experts, 2 * inter, hidden), device=device)
    dn_codes = torch.randint(0, 16, (experts, hidden, inter), device=device)
    gu_scale = torch.rand(experts, 2 * inter, hidden // 128, device=device,
                          dtype=torch.float16) * 0.03 + 0.005
    dn_scale = torch.rand(experts, hidden, inter // 128, device=device,
                          dtype=torch.float16) * 0.03 + 0.005
    gu_packed, dn_packed = _pack(gu_codes), _pack(dn_codes)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    ids = torch.tensor([[0, 2], [1, 3]], device=device, dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], device=device)

    actual = fused_experts_decode_autoround(
        x, gu_packed, gu_scale, dn_packed, dn_scale, weights, ids,
    )
    gu = _dequant(gu_codes, gu_scale)
    dn = _dequant(dn_codes, dn_scale)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for token in range(tokens):
        for route in range(top_k):
            expert = int(ids[token, route])
            proj = x[token].float() @ gu[expert].T
            activated = F.silu(proj[:inter]) * proj[inter:]
            expected[token] += weights[token, route] * (activated @ dn[expert].T)

    torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.08)


def test_autoround_prefill_matches_torch():
    torch.manual_seed(11)
    device = torch.device("cuda")
    experts, tokens, top_k, hidden, inter = 4, 17, 2, 256, 128
    dtype = torch.bfloat16
    gu_codes = torch.randint(0, 16, (experts, 2 * inter, hidden), device=device)
    dn_codes = torch.randint(0, 16, (experts, hidden, inter), device=device)
    gu_scale = torch.rand(experts, 2 * inter, hidden // 128, device=device,
                          dtype=torch.float16) * 0.02 + 0.005
    dn_scale = torch.rand(experts, hidden, inter // 128, device=device,
                          dtype=torch.float16) * 0.02 + 0.005
    gu_packed, dn_packed = _pack(gu_codes), _pack(dn_codes)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    ids = torch.randint(0, experts, (tokens, top_k), device=device, dtype=torch.int32)
    weights = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1)

    actual = fused_experts_autoround(
        x, gu_packed, gu_scale, dn_packed, dn_scale, weights, ids, experts,
    )
    gu = _dequant(gu_codes, gu_scale)
    dn = _dequant(dn_codes, dn_scale)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for token in range(tokens):
        for route in range(top_k):
            expert = int(ids[token, route])
            proj = x[token].float() @ gu[expert].T
            activated = F.silu(proj[:inter]) * proj[inter:]
            expected[token] += weights[token, route] * (activated @ dn[expert].T)

    torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.08)
