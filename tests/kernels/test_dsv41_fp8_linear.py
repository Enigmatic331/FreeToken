from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _dequant_weight(weight: torch.Tensor, scale: torch.Tensor, block: int) -> torch.Tensor:
    factors = scale.view(torch.uint8).float().sub(127).exp2()
    factors = factors.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return weight.float() * factors


@pytest.mark.parametrize(
    ("tokens", "n"),
    [(1, 256), (7, 256), (3, 320)],
    ids=["decode", "prefill", "tail-output-tile"],
)
def test_v41_block32_linear_matches_dequantized_torch_reference(tokens, n):
    from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8
    from freetoken.kernel.triton.dsv41 import block_fp8_linear_32

    torch.manual_seed(41 + tokens)
    m, k = tokens, 256
    x = (torch.randn(m, k, device="cuda") * 1.5).to(torch.bfloat16)
    weight = (torch.randn(n, k, device="cuda") * 1.25).to(torch.float8_e4m3fn)
    scale_codes = torch.randint(124, 130, (n // 32, k // 32), device="cuda", dtype=torch.uint8)
    scale = scale_codes.view(torch.float8_e8m0fnu)

    got = block_fp8_linear_32(x, weight, scale)
    aq, ascales = act_quant_fp8(x, 32)
    activation = aq.float() * ascales.float().sub(127).exp2().repeat_interleave(32, 1)
    want = (activation @ _dequant_weight(weight, scale, 32).T).to(torch.bfloat16)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=0.5)


def test_existing_dsv4_block128_default_is_unchanged():
    from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8, block_fp8_linear

    torch.manual_seed(4)
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, 256, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.full((1, 2), 127, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    got = block_fp8_linear(x, weight, scale)
    aq, ascales = act_quant_fp8(x, 128)
    activation = aq.float() * ascales.float().sub(127).exp2().repeat_interleave(128, 1)
    want = activation @ weight.float().T
    torch.testing.assert_close(got, want.to(torch.bfloat16), rtol=2e-2, atol=0.5)
