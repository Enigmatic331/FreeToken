from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache


def test_uniform_glm_iq_stage_still_aligns_grouped_mmq_cache_rows():
    """A PP stage can contain one IQ recipe even when the full model is mixed."""
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=2,
        cache_size=2,
        device=torch.device("cpu"),
        quant_format="gguf_glm_iq",
    )
    cache.set_bank_sources(
        {
            "gate_up": [torch.empty((2, 3), dtype=torch.uint8) for _ in range(2)],
            "down": [torch.empty((2, 5), dtype=torch.uint8) for _ in range(2)],
        }
    )

    assert cache.bank_caches["gate_up"].shape == (2, 400)
    assert cache.bank_caches["down"].shape == (2, 13328)
    assert cache._variable_stride_banks == {"gate_up", "down"}

    cache.rebuild(3)
    assert cache.bank_caches["gate_up"].shape == (3, 400)
    assert cache.bank_caches["down"].shape == (3, 13328)


def _stats_only_cache(*, target: str, rows: list[list[int]]) -> OffloadMoeCache:
    cache = object.__new__(OffloadMoeCache)
    cache.decode_target = target
    cache.lru_stats = torch.tensor(rows, dtype=torch.int64)
    cache.stat_active = torch.tensor(0, dtype=torch.int64)
    cache.stat_missing = torch.tensor(0, dtype=torch.int64)
    cache.stat_calls = torch.tensor(0, dtype=torch.int64)
    cache.stat_fetched = torch.tensor(0, dtype=torch.int64)
    cache.stat_steps_layer = torch.zeros(len(rows), dtype=torch.int64)
    cache.stat_active_layer = torch.zeros(len(rows), dtype=torch.int64)
    cache.stat_missing_layer = torch.zeros(len(rows), dtype=torch.int64)
    cache.stat_fetched_layer = torch.zeros(len(rows), dtype=torch.int64)
    cache.prefill_hit_rows = 0
    cache.prefill_total_rows = 0
    cache.num_layers = len(rows)
    return cache


def test_plain_gpu_cache_reports_every_miss_as_a_fetch():
    # flashlib Stat columns are ACTIVE, MISS, CALLS.
    cache = _stats_only_cache(target="gpu", rows=[[12, 3, 2], [8, 1, 2]])

    total = cache.decode_miss_stats()
    per_layer = cache.decode_miss_stats_per_layer()["per_layer"]

    assert total["missing_per_layer"] == 1.0
    assert total["fetched_per_layer"] == 1.0
    assert total["fetch_rate"] == 1.0
    assert [layer["fetched_per_step"] for layer in per_layer] == [1.5, 0.5]


def test_routing_histogram_excludes_inactive_ep_sentinels():
    cache = _stats_only_cache(target="gpu", rows=[[0, 0, 0]])
    cache.num_experts = 4
    cache.cache_size = 2
    cache.decode_freq = torch.zeros((1, 4), dtype=torch.int64)
    ids = torch.tensor([[0, 4, 2, 4]], dtype=torch.int32)
    active = torch.tensor([[True, False, True, False]])

    cache.record_decode_routes(0, ids, active)

    assert cache.decode_freq.tolist() == [[1, 0, 1, 0]]


def test_routing_stats_report_global_static_coverage():
    cache = _stats_only_cache(target="gpu", rows=[[0, 0, 0], [0, 0, 0]])
    cache.num_experts = 4
    cache.cache_size = 2
    cache.decode_freq = torch.tensor([[8, 2, 0, 0], [5, 3, 1, 1]], dtype=torch.int64)

    stats = cache.decode_routing_stats()

    assert stats["working_set_mean"] == 3.0
    assert stats["static_coverage"]["2"] == pytest.approx(13 / 20)
    assert stats["expert_pairs_for_90pct"] == 4
