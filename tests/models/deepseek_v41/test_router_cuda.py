from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv41.router import fused_sqrtsoftplus_topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,experts", [(1, 385), (17, 385), (64, 384)])
def test_fused_sqrtsoftplus_router_matches_torch(rows: int, experts: int):
    generator = torch.Generator().manual_seed(41)
    logits = torch.randn(rows, experts, generator=generator, dtype=torch.float32)
    bias = torch.randn(experts, generator=generator, dtype=torch.float32) * 0.1
    scores = F.softplus(logits / 0.7).sqrt()
    expected_ids = (scores + bias).topk(8, dim=-1).indices
    expected_weights = scores.gather(-1, expected_ids)
    expected_weights /= expected_weights.sum(-1, keepdim=True) + 1e-20
    expected_weights *= 1.5

    weights, ids = fused_sqrtsoftplus_topk(
        logits.cuda(),
        bias.cuda(),
        topk=8,
        temperature=0.7,
        renormalize=True,
        route_scale=1.5,
    )

    torch.testing.assert_close(ids.cpu().long(), expected_ids)
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=2e-5, atol=2e-6)
