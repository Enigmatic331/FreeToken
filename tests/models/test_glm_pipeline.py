from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _glm_indexers() -> tuple[str, ...]:
    return tuple(
        "full" if i < 3 or (i - 2) % 4 == 0 else "shared" for i in range(78)
    )


def test_pp2_boundary_starts_on_indexshare_leader(monkeypatch):
    from freetoken.models.glm_moe_dsa import execution

    monkeypatch.setattr(execution, "_ENABLED", True)
    monkeypatch.setattr(
        execution, "try_get_tp_info", lambda: SimpleNamespace(rank=1, size=2)
    )

    plan = execution.glm_pipeline_plan(78, _glm_indexers())

    assert (plan.start_layer, plan.stop_layer) == (38, 78)
    assert plan.layer_ids[0] == 38
    assert _glm_indexers()[plan.start_layer] == "full"
    assert plan.moe_index(38, first_moe_layer=3) == 0
    assert plan.moe_index(77, first_moe_layer=3) == 39


def test_pp3_boundaries_are_safe_and_cover_every_layer(monkeypatch):
    from freetoken.models.glm_moe_dsa import execution

    monkeypatch.setattr(execution, "_ENABLED", True)
    plans = []
    for rank in range(3):
        monkeypatch.setattr(
            execution,
            "try_get_tp_info",
            lambda rank=rank: SimpleNamespace(rank=rank, size=3),
        )
        plans.append(execution.glm_pipeline_plan(78, _glm_indexers()))

    assert [p.start_layer for p in plans] == [0, 26, 50]
    assert [p.stop_layer for p in plans] == [26, 50, 78]
    assert tuple(i for p in plans for i in p.layer_ids) == tuple(range(78))
    assert all(p.start_layer == 0 or _glm_indexers()[p.start_layer] == "full" for p in plans)


def test_pipeline_plan_is_disabled_before_worker_rank_exists(monkeypatch):
    from freetoken.models.glm_moe_dsa import execution

    monkeypatch.setattr(execution, "_ENABLED", True)
    monkeypatch.setattr(execution, "try_get_tp_info", lambda: None)

    plan = execution.glm_pipeline_plan(78, _glm_indexers())

    assert not plan.enabled
    assert plan.layer_ids == tuple(range(78))


def test_stage_local_mla_pool_uses_global_layer_ids():
    from freetoken.kvcache.dsa_pool import MLAKVCache

    pool = MLAKVCache(
        latent_dim=4,
        num_layers=2,
        num_pages=3,
        page_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_ids=(38, 39),
    )
    pool.k_cache(38).fill_(1)
    pool.k_cache(39).fill_(2)

    assert pool.num_layers == 2
    assert pool.k_cache(38)[0, 0].tolist() == [1, 1, 1, 1]
    assert pool.k_cache(39)[0, 0].tolist() == [2, 2, 2, 2]
    with pytest.raises(KeyError):
        pool.k_cache(0)


def test_op_list_preserves_global_checkpoint_indices():
    from freetoken.layers import BaseOP, OPList

    class Weighted(BaseOP):
        def __init__(self, value: float):
            self.weight = torch.tensor(value)

    ops = OPList([Weighted(1), Weighted(2)], start_index=38)

    assert set(ops.state_dict(prefix="model.layers")) == {
        "model.layers.38.weight",
        "model.layers.39.weight",
    }


def test_stage_local_expert_bank_indices():
    from freetoken.models.glm_moe_dsa.weight import _layer_to_bank

    config = SimpleNamespace(
        first_k_dense_replace=3,
        num_layers=78,
        local_layer_ids=tuple(range(38, 78)),
    )

    assert _layer_to_bank(37, config) is None
    assert _layer_to_bank(38, config) == 0
    assert _layer_to_bank(77, config) == 39
