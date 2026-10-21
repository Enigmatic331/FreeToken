"""Explicit-window GPU qualification for the isolated packed prefill candidate.

Collection skips these tests unless FREETOKEN_TEST_GPU_WINDOW=1 is explicitly
set after serving admission is closed.  Ordinary CPU test runs never launch or
even query CUDA through this module.  Every case compares output bits with the
production BF16-pool kernel on the same inputs.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("FREETOKEN_TEST_GPU_WINDOW") != "1",
    reason="experimental GPU kernel: requires an approved admission-closed GPU window",
)


@pytest.fixture(autouse=True)
def _tests_this_checkout():
    """An editable install elsewhere would otherwise be what runs: set PYTHONPATH=<repo>/python."""
    import freetoken

    repo = Path(__file__).resolve().parents[2]
    assert Path(freetoken.__file__).resolve().is_relative_to(repo / "python"), freetoken.__file__


def _case(seed, m, batch, **kwargs):
    # tests/kvcache is a package: a bare ``import test_dsv41_packed_kv`` does not resolve.
    from .test_dsv41_packed_kv import _attention_case

    return _attention_case(seed, m, batch, **kwargs)


def _launch(q, window, packed, sink, idx, n_window, scale, counts=None, buffers=2):
    from freetoken.kernel.triton.dsv41.sparse_attn_gluon import sparse_attn_paged_packed_gluon

    assert torch.cuda.get_device_capability(q.device) == (12, 0)
    return sparse_attn_paged_packed_gluon(
        q, window, packed, sink, idx, n_window, scale, counts, buffers=buffers
    )


def _oracle(q, window, bf16, sink, idx, n_window, scale, counts=None):
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

    return sparse_attn_paged(q, window, bf16, sink, idx, n_window, scale, cmp_counts=counts)


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("buffers", [1, 2])
@pytest.mark.parametrize("m,batch,live", [
    (2, 1, None), (6, 1, 0), (6, 1, 1), (6, 1, 31),
    (6, 1, 32), (6, 1, 33), (64, 2, 300), (256, 1, 512),
    (2048, 1, None),
])
def test_prefill_is_bitwise_identical(m, batch, live, buffers):
    window, bf16, packed, q, idx, sink = _case(29, m, batch)
    counts = None if live is None else torch.full(
        (batch, m), live, dtype=torch.int32, device=q.device
    )
    expected = _oracle(q, window, bf16, sink, idx, 128, 0.04, counts)
    actual = _launch(q, window, packed, sink, idx, 128, 0.04, counts, buffers)
    torch.cuda.synchronize()
    assert _bits_equal(actual, expected)


@pytest.mark.parametrize("pattern", ["all_window", "all_packed", "empty_tile", "all_masked"])
def test_prefill_invalid_tile_patterns(pattern):
    window, bf16, packed, q, idx, sink = _case(31, 3, 1)
    if pattern == "all_window":
        idx[..., 128:] = -1
    elif pattern == "all_packed":
        idx[..., :128] = -1
    elif pattern == "empty_tile":
        idx[..., 160:192] = -1
    else:
        idx.fill_(-1)
    expected = _oracle(q, window, bf16, sink, idx, 128, 0.04)
    actual = _launch(q, window, packed, sink, idx, 128, 0.04)
    torch.cuda.synchronize()
    assert _bits_equal(actual, expected)


@pytest.mark.parametrize("buffers", [1, 2])
def test_prefill_long_candidate_list_wraps_the_ring(buffers):
    """132 tiles: every ready/consumed barrier completes dozens of phases."""
    window, bf16, packed, q, idx, sink = _case(41, 64, 1, topk_cmp=4096)
    expected = _oracle(q, window, bf16, sink, idx, 128, 0.04)
    actual = _launch(q, window, packed, sink, idx, 128, 0.04, buffers=buffers)
    torch.cuda.synchronize()
    assert _bits_equal(actual, expected)


def test_prefill_partial_head_block():
    """40 heads: the last head block masks 8 rows of queries and outputs."""
    window, bf16, packed, q, idx, sink = _case(43, 6, 1, h=40)
    expected = _oracle(q, window, bf16, sink, idx, 128, 0.04)
    actual = _launch(q, window, packed, sink, idx, 128, 0.04)
    torch.cuda.synchronize()
    assert _bits_equal(actual, expected)


def test_prefill_repeated_launches_stay_identical():
    """A producer/consumer race would show as intermittent bit flips across launches."""
    window, bf16, packed, q, idx, sink = _case(47, 2048, 1)
    expected = _oracle(q, window, bf16, sink, idx, 128, 0.04)
    for launch in range(20):
        actual = _launch(q, window, packed, sink, idx, 128, 0.04)
        torch.cuda.synchronize()
        assert _bits_equal(actual, expected), launch


def test_prefill_graph_replays_live_counts():
    window, bf16, packed, q, idx, sink = _case(37, 6, 2)
    counts = torch.full((2, 6), 300, dtype=torch.int32, device=q.device)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch(q, window, packed, sink, idx, 128, 0.04, counts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = _launch(q, window, packed, sink, idx, 128, 0.04, counts)
    for live in (0, 1, 17, 32, 33, 300, 512, 1):
        counts.fill_(live)
        expected = _oracle(q, window, bf16, sink, idx, 128, 0.04, counts)
        graph.replay()
        torch.cuda.synchronize()
        assert _bits_equal(actual, expected), live


def test_dispatch_flag_routes_packed_prefill_to_gluon(monkeypatch):
    """The serving entry point with FREETOKEN_DSV41_PACKED_PREFILL_GLUON=1: prefill goes
    through the Gluon launcher, decode stays on Triton, and both keep the BF16 bits."""
    from freetoken.kernel.triton.dsv41 import sparse_attn_gluon
    from freetoken.kvcache import dsv41_packed_kv

    calls = []
    launcher = sparse_attn_gluon.sparse_attn_paged_packed_gluon
    monkeypatch.setattr(
        sparse_attn_gluon, "sparse_attn_paged_packed_gluon",
        lambda *a, **k: calls.append(a[0].shape) or launcher(*a, **k),
    )
    monkeypatch.setenv("FREETOKEN_DSV41_PACKED_PREFILL_GLUON", "1")
    dsv41_packed_kv.packed_prefill_gluon_enabled.cache_clear()
    try:
        for m in (64, 1):
            window, bf16, packed, q, idx, sink = _case(53, m, 1)
            expected = _oracle(q, window, bf16, sink, idx, 128, 0.04)
            from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

            actual = sparse_attn_paged(q, window, packed, sink, idx, 128, 0.04)
            torch.cuda.synchronize()
            assert _bits_equal(actual, expected), m
    finally:
        dsv41_packed_kv.packed_prefill_gluon_enabled.cache_clear()
    assert [shape[1] for shape in calls] == [64]
