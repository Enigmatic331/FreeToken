"""CPU-only policy checks for finalizing the synthetic prefill warmup."""

from __future__ import annotations

from unittest.mock import Mock
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import (
    _finalize_moe_prefill_warmup,
    _needs_prefill_warmup,
    _prefill_warmup_slot_start,
)


def test_sparse_prefill_requests_warmup_with_non_triton_attention():
    cache = Mock(sparse_prefill_max_tokens=256)

    assert _needs_prefill_warmup("qsa_sparse", cache)


def test_triton_attention_preserves_existing_warmup_behavior():
    cache = Mock(sparse_prefill_max_tokens=0)

    assert _needs_prefill_warmup("triton,flashinfer", cache)


def test_non_triton_dense_path_does_not_add_startup_work():
    cache = Mock(sparse_prefill_max_tokens=0)

    assert not _needs_prefill_warmup("qsa_sparse", cache)


def test_sparse_prefill_keeps_warmed_experts_and_clears_stats():
    cache = Mock(sparse_prefill_max_tokens=256)

    _finalize_moe_prefill_warmup(cache)

    cache.reset_stats.assert_called_once_with()
    cache.reset.assert_not_called()


def test_dense_prefill_preserves_cold_reset_behavior():
    cache = Mock(sparse_prefill_max_tokens=0)

    _finalize_moe_prefill_warmup(cache)

    cache.reset.assert_called_once_with()
    cache.reset_stats.assert_not_called()


def test_prefill_warmup_without_moe_cache_is_a_noop():
    _finalize_moe_prefill_warmup(None)


@pytest.mark.parametrize("length", [80, 128])
def test_dsv41_prefill_warmup_uses_bound_reserved_page(length):
    from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache

    pool = object.__new__(DSV41PagedKVCache)
    pool.P = 128
    pool._device = torch.device("cpu")
    pool.full_to_window = torch.full((1025,), -1, dtype=torch.int64)
    pool.bind_window_pages(896, 256)
    start = _prefill_warmup_slot_start(pool, 896, length)
    slots = pool.translate_full_to_window(torch.arange(start, start + length))
    assert torch.equal(slots, torch.arange(256, 256 + length))
    assert torch.all(pool.full_to_window[:896] == -1)  # no real request mapping touched
    with pytest.raises(ValueError, match="dummy page"):
        _prefill_warmup_slot_start(pool, 896, 129)


def test_generic_prefill_warmup_preserves_existing_slot_range():
    assert _prefill_warmup_slot_start(object(), 896, 128) == 0


@pytest.mark.parametrize("limit", [1, 16, 256])
def test_dsfp4_sparse_modules_warm_before_collectives_and_reset_cache(monkeypatch, limit):
    from freetoken.engine.engine import Engine
    import freetoken.kernel as kernels

    events = []
    cache = SimpleNamespace(quant_format="ds_fp4", reset=lambda: events.append("reset"))
    rows_seen = []

    def prefill(hidden, weights, ids):
        rows_seen.append(hidden.shape[0])
        assert weights.shape == ids.shape == (hidden.shape[0], 6)
        assert torch.all(weights[:, 0] == 1)
        assert torch.all(weights[:, 1:] == 0)
        assert torch.all(ids == 0)
        events.append("local_compute")
        return hidden

    layer = SimpleNamespace(top_k=6, hidden_size=4, _prefill_routed=prefill)
    engine = SimpleNamespace(moe_offload_cache=cache,
        model=SimpleNamespace(_iter_offload_moe_layers=lambda: iter([layer])),
        dtype=torch.bfloat16, device=torch.device("cpu"),
        _execution_plan=SimpleNamespace(enabled=True), tp_cpu_group=object())
    monkeypatch.setattr(kernels, "moe_sum_reduce_triton", lambda *_: events.append("reduce"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: events.append("sync"))
    monkeypatch.setattr(torch.distributed, "barrier", lambda **_: events.append("barrier"))

    Engine._warmup_sparse_prefill_moe_kernels(
        engine, SimpleNamespace(moe_sparse_prefill_max_tokens=limit))
    assert rows_seen == sorted(set(rows_seen))
    assert rows_seen[0] == 1 and rows_seen[-1] == limit
    if limit == 256:
        assert {127, 128, 129, 170, 171}.issubset(rows_seen)
    assert events[-3:] == ["reset", "sync", "barrier"]
    assert events.count("barrier") == 1
    assert events.count("reset") == len(rows_seen) + 1


@pytest.mark.parametrize("hybrid_limit", [None, 1, 3, 5, 6, 0])
def test_dspark_warms_local_hybrid_before_collective_barrier(monkeypatch, hybrid_limit):
    from freetoken.engine.engine import Engine
    import freetoken.kernel.triton.dsv41 as kernels

    events, gpu_rows, hybrid_rows = [], [], []
    cache = SimpleNamespace(
        num_experts=8, is_cpu_layer=lambda _: False,
        uses_hybrid=lambda rows: hybrid_limit is not None and (
            hybrid_limit == 0 or rows <= hybrid_limit),
        reset=lambda: events.append("reset"),
        ensure_experts=lambda *_: events.append("ensure"),
        copy_missing=lambda: events.append("copy"),
        bank_views=lambda: (), alphas_for_slots=lambda _: None,
    )

    def gemm(_cache, hidden, *_args, **_kwargs):
        gpu_rows.append(hidden.shape[0])
        events.append("gpu")
        return hidden

    def cpu_overlap(_cache, hidden, weights, ids):
        rows = hidden.shape[0]
        hybrid_rows.append(rows)
        assert hidden.shape == (rows, 4)
        assert weights.shape == ids.shape == (rows, 6)
        assert torch.allclose(weights, torch.full((rows, 6), 1 / 6))
        assert torch.equal(ids, torch.arange(6, dtype=torch.int32).repeat(rows, 1))
        events.append("hybrid")
        return hidden

    layer = SimpleNamespace(layer_id=0, top_k=6, _expert_gemm=gemm, _decode_hybrid=cpu_overlap)
    engine = SimpleNamespace(
        moe_offload_cache=cache,
        model=SimpleNamespace(_iter_offload_moe_layers=lambda: iter([layer])),
        dtype=torch.bfloat16, device=torch.device("cpu"), stream=None,
        _execution_plan=SimpleNamespace(enabled=True), tp_cpu_group=object(),
    )
    monkeypatch.setattr(kernels, "fused_localize_cache_safe_routes", lambda w, ids, **_: (w, ids))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "Event", lambda **_: SimpleNamespace(
        record=lambda *_: events.append("event"), elapsed_time=lambda _: 0.0))
    monkeypatch.setattr(torch.distributed, "barrier", lambda **_: events.append("barrier"))
    Engine._warmup_dspark_moe_kernels(engine, SimpleNamespace(
        model_config=SimpleNamespace(dsv41_args=SimpleNamespace(dspark_block_size=5, dim=4)),
        speculative_dspark=True))
    assert gpu_rows == list(range(1, 7))
    assert events[-1] == "barrier"
    expected = [] if hybrid_limit is None else list(range(
        1, (hybrid_limit or 6) + 1))
    assert hybrid_rows == expected
    assert events.count("hybrid") == len(expected)
    if expected:
        # Synthetic CPU work is drained and the real cache starts cold.
        last_hybrid = max(i for i, event in enumerate(events) if event == "hybrid")
        assert events[last_hybrid + 1:][:3] == ["sync", "reset", "sync"]
