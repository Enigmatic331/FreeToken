"""Fused Lightning-Indexer prefill scoring against a float32 reference."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.dsv4.indexer import indexer_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TOL = dict(atol=5e-2, rtol=5e-2)


def _inputs(*, batch=2, query_rows=129, key_rows=257, heads=32, dim=128):
    generator = torch.Generator(device="cuda").manual_seed(4101)
    q = torch.randn(
        batch,
        query_rows,
        heads,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    keys = torch.randn(
        batch,
        key_rows,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weights = torch.randn(
        batch,
        query_rows,
        heads,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return q, keys, weights


def _reference(q, keys, weights):
    scores = torch.einsum("bshd,btd->bsht", q.float(), keys.float()).relu_()
    return (scores * weights.float().unsqueeze(-1)).sum(2)


def test_prefill_logits_match_head_reduced_reference():
    q, keys, weights = _inputs()
    got = indexer_logits(q, keys, weights)
    assert got.shape == (2, 129, 257)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, _reference(q, keys, weights), **TOL)


def test_prefill_logits_support_unaligned_tiles_and_reusable_output():
    q, keys, weights = _inputs(batch=1, query_rows=3, key_rows=131)
    out = torch.full((1, 3, 131), torch.nan, device="cuda", dtype=torch.float32)
    got = indexer_logits(q, keys, weights, out=out)
    assert got.data_ptr() == out.data_ptr()
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, _reference(q, keys, weights), **TOL)
