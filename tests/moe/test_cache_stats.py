from __future__ import annotations

import torch

from freetoken.moe.offload_cache import OffloadMoeCache


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
