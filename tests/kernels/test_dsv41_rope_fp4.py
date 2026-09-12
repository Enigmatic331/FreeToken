from __future__ import annotations

import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _round_fp4(x):
    magnitude = x.abs()
    step = torch.where(
        magnitude < 2.0,
        torch.tensor(0.5, device=x.device),
        torch.where(
            magnitude < 4.0,
            torch.tensor(1.0, device=x.device),
            torch.tensor(2.0, device=x.device),
        ),
    )
    return torch.round(magnitude / step) * step * torch.sign(x)


def _reference(x, freqs, rope_dim, compressed):
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    complex_tail = torch.view_as_complex(
        tail.float().unflatten(-1, (-1, 2)).contiguous()
    )
    f = freqs.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
    rotated = torch.view_as_real(complex_tail * f).flatten(-2).to(x.dtype)
    value = torch.cat([head, rotated], dim=-1)
    block = 16 if compressed else 32
    blocks = value.float().unflatten(-1, (-1, block))
    amax = blocks.abs().amax(-1, keepdim=True)
    if compressed:
        scale = (amax * (1.0 / 6.0)).clamp(2**-9, 448.0)
        scale = scale.to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6.0 * 2.0**-126) * (1.0 / 6.0)
        bits = amax.contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 0xFF) - 127
        exponent += ((bits & 0x7FFFFF) != 0).to(torch.int32)
        scale = ((exponent + 127) << 23).view(torch.float32)
    return (_round_fp4((blocks / scale).clamp(-6, 6)) * scale).flatten(-2).to(x.dtype)


@pytest.mark.parametrize("compressed,dim", [(False, 128), (True, 512)])
def test_fused_rope_fp4_matches_torch_reference(compressed, dim):
    from freetoken.kernel.triton.dsv41 import rope_fp4_roundtrip

    torch.manual_seed(410)
    tokens, heads, rope_dim = 5, 3, 64
    x = (torch.randn(tokens, heads, dim, device="cuda") * 2).to(torch.bfloat16)
    angle = torch.randn(tokens, rope_dim // 2, device="cuda")
    freqs = torch.polar(torch.ones_like(angle), angle)
    got = rope_fp4_roundtrip(x, freqs, rope_dim, compressed_kv=compressed)
    want = _reference(x, freqs, rope_dim, compressed)
    assert torch.equal(got, want)


def test_fused_rope_fp4_rejects_wrong_dtype():
    from freetoken.kernel.triton.dsv41 import rope_fp4_roundtrip

    x = torch.zeros(1, 128, device="cuda", dtype=torch.float32)
    freqs = torch.ones(1, 32, device="cuda", dtype=torch.complex64)
    with pytest.raises(ValueError, match="BF16"):
        rope_fp4_roundtrip(x, freqs, 64)
