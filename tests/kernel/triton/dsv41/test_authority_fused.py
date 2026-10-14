"""G2 exactness/codegen qualification for Package-A primitives (SM120 only)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv41.authority_fused import (
    hc_post_mixprep,
    hc_prepare_mix,
    hc_sinkhorn_rmsprep,
    router_scores_exact,
    router_wire_tail,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="G2 needs CUDA")


@pytest.mark.parametrize("rows", [1, 2])
def test_prepare_mix_is_row_local_and_exact(rows):
    x = torch.randn(rows, 1, 4, 5120, dtype=torch.bfloat16, device="cuda")
    flat, square = hc_prepare_mix(x)
    expected = x.flatten(-2).float()
    assert flat.data_ptr() != square.data_ptr()
    assert torch.equal(flat.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(square.view(torch.int32), expected.square().view(torch.int32))


def test_prepare_mix_exhausts_all_bf16_patterns():
    bits = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
    x = bits.view(torch.bfloat16).view(1, 1, 4, 16384)
    flat, square = hc_prepare_mix(x)
    expected = x.flatten(-2).float()
    assert torch.equal(flat.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(square.view(torch.int32), expected.square().view(torch.int32))


def test_prepare_mix_partner_payload_cannot_change_row_a():
    x = torch.randn(2, 1, 4, 5120, dtype=torch.bfloat16, device="cuda")
    first = hc_prepare_mix(x)
    saved = tuple(value[0].clone() for value in first)
    x[1].normal_().mul_(32)
    second = hc_prepare_mix(x)
    assert all(torch.equal(old, new[0]) for old, new in zip(saved, second))


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("routes,prepare", [(False, True), (True, True), (True, False)])
def test_post_mixprep_matches_ordered_reference(rows, routes, prepare):
    torch.manual_seed(6106 + rows)
    residual = torch.randn(rows, 1, 4, 5120, dtype=torch.bfloat16, device="cuda")
    post = torch.randn(rows, 1, 4, dtype=torch.float32, device="cuda")
    comb = torch.randn(rows, 1, 4, 4, dtype=torch.float32, device="cuda")
    shared = torch.randn(rows, 5120, dtype=torch.bfloat16, device="cuda")
    route_values = (
        torch.randn(rows, 6, 5120, dtype=torch.bfloat16, device="cuda")
        if routes else None
    )
    activation = shared
    if route_values is not None:
        reduced = torch.zeros(rows, 5120, dtype=torch.float32, device="cuda")
        for slot in range(6):
            reduced += route_values[:, slot].float()
        activation = (shared.float() + reduced.to(torch.bfloat16).float()).to(torch.bfloat16)
    from freetoken.models.deepseek_v41.hc import hc_post

    expected = hc_post(activation.view(rows, 1, 5120), residual, post, comb)
    hidden, flat, square = hc_post_mixprep(
        shared,
        residual,
        post,
        comb,
        routes=route_values,
        prepare=prepare,
    )
    assert torch.equal(hidden.view(torch.int16), expected.view(torch.int16))
    if prepare:
        widened = hidden.flatten(-2).float()
        assert torch.equal(flat.view(torch.int32), widened.view(torch.int32))
        assert torch.equal(square.view(torch.int32), widened.square().view(torch.int32))
    else:
        assert flat is square is None


@pytest.mark.parametrize("rows", [1, 2])
def test_sinkhorn_rmsprep_matches_composed_reference(rows):
    from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn

    torch.manual_seed(7000 + rows)
    mixes = torch.randn(rows, 1, 24, device="cuda", dtype=torch.float32)
    mean = torch.rand(rows, 1, 1, device="cuda", dtype=torch.float32)
    scale = torch.randn(3, device="cuda", dtype=torch.float32)
    base = torch.randn(24, device="cuda", dtype=torch.float32)
    actual = hc_sinkhorn_rmsprep(
        mixes, mean, scale, base,
        hc_mult=4, sinkhorn_iters=20, eps=1e-6, norm_eps=1e-20,
    )
    value = mixes * torch.rsqrt(mean + 1e-20)
    reference = hc_split_sinkhorn(
        value.reshape(-1, 24), scale, base, 4, 20, 1e-6
    )
    reference = (
        reference[0].view(rows, 1, 4),
        reference[1].view(rows, 1, 4),
        reference[2].view(rows, 1, 4, 4),
    )
    assert len(actual) == len(reference) == 3
    assert all(a.shape == b.shape for a, b in zip(actual, reference))
    assert all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(actual, reference))


@pytest.mark.parametrize("rows", [1, 2])
def test_router_score_and_wire_bytes_exact(rows):
    torch.manual_seed(8000 + rows)
    logits = torch.randn(rows, 384, dtype=torch.float32, device="cuda") * 20
    bias = torch.randn(384, dtype=torch.float32, device="cuda")
    scores, ranked = router_scores_exact(logits, bias, inv_temperature=1.0)
    expected = F.softplus(logits).sqrt()
    assert torch.equal(scores.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(ranked.view(torch.int32), (expected + bias).view(torch.int32))
    ids = ranked.topk(6, -1).indices
    gathered = scores.gather(-1, ids)
    denominator = gathered.sum(-1, keepdim=True)
    weights = torch.empty(rows, 6, dtype=torch.float32, device="cuda")
    wire_ids = torch.empty(rows, 6, dtype=torch.int32, device="cuda")
    router_wire_tail(
        gathered, denominator, ids, weights, wire_ids, eps=1e-20, route_scale=1.5
    )
    expected_weights = gathered / (denominator + 1e-20) * 1.5
    assert torch.equal(weights.view(torch.int32), expected_weights.view(torch.int32))
    assert torch.equal(wire_ids, ids.to(torch.int32))
    # Planted defect: permuting gathered values must be visible in the wire.
    assert not torch.equal(weights, expected_weights.flip(-1))


def test_primitives_capture_and_replay_m2():
    x = torch.randn(2, 1, 4, 5120, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        hc_prepare_mix(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flat, square = hc_prepare_mix(x)
    for seed in range(8):
        x.normal_().mul_(2.0 ** (seed - 4))
        expected = x.flatten(-2).float()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(flat.view(torch.int32), expected.view(torch.int32))
        assert torch.equal(square.view(torch.int32), expected.square().view(torch.int32))
