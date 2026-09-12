from __future__ import annotations

import torch

from freetoken.models.deepseek_v41.hc import (
    hc_mixes,
    hc_post,
    hc_pre,
    identity_pre_mix,
)


def test_hc_cpu_matches_reference_and_carries_predecessor_mix():
    torch.manual_seed(41)
    batch, seq, hc, dim = 2, 3, 4, 8
    x = torch.randn(batch, seq, hc, dim, dtype=torch.bfloat16)
    weight = torch.randn((2 + hc) * hc, hc * dim)
    scale = torch.randn(3)
    base = torch.randn((2 + hc) * hc)
    incoming = identity_pre_mix(x, hc)

    pre, post, comb = hc_mixes(
        x,
        weight,
        scale,
        base,
        hc_mult=hc,
        sinkhorn_iters=5,
        eps=1e-6,
        norm_eps=1e-20,
    )
    collapsed = hc_pre(x, incoming)
    torch.testing.assert_close(collapsed, x[..., 0, :])

    block_output = torch.randn(batch, seq, dim, dtype=torch.bfloat16)
    got = hc_post(block_output, x, post, comb)
    want = post.unsqueeze(-1) * block_output.float().unsqueeze(-2)
    want += (comb.unsqueeze(-1) * x.float().unsqueeze(-2)).sum(-3)
    torch.testing.assert_close(got, want.bfloat16())
    assert pre.shape == incoming.shape
    assert not torch.equal(pre, incoming)
