from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.execution import DeepseekV41ExecutionPlan
from freetoken.models.deepseek_v41.moe import Gate, _prepare_partitioned_routes
from freetoken.moe.partition import ExpertPartition, ExpertStorageRange


def test_execution_plan_balances_global_experts_and_keeps_workers_in_engram():
    left = DeepseekV41ExecutionPlan(0, 2, backbone_rank=0).partition(385)
    right_plan = DeepseekV41ExecutionPlan(1, 2, backbone_rank=0)
    right = right_plan.partition(385)
    assert (left.global_offset, left.local_count) == (0, 193)
    assert (right.global_offset, right.local_count) == (193, 192)
    assert right_plan.is_expert_worker
    assert right_plan.participates_in_engram


def test_execution_plan_can_keep_ep3_engram_on_two_ranks():
    plans = [
        DeepseekV41ExecutionPlan(
            rank, 3, backbone_rank=0, engram_ranks=(0, 1)
        )
        for rank in range(3)
    ]
    assert [plan.participates_in_engram for plan in plans] == [True, True, False]
    assert plans[0].engram_rank == 0
    assert plans[1].engram_rank == 1
    assert plans[2].engram_world_size == 2
    with pytest.raises(RuntimeError, match="does not participate"):
        _ = plans[2].engram_rank


def test_asymmetric_ep_uses_decode_ownership_for_prefill_without_phase_split():
    plan = DeepseekV41ExecutionPlan(
        rank=2,
        world_size=3,
        backbone_rank=0,
        expert_shards=(3, 3, 2),
    )
    decode = plan.partition(8)
    prefill = plan.partition(8, prefill=True)
    assert (prefill.global_offset, prefill.local_count) == (
        decode.global_offset,
        decode.local_count,
    )


def test_execution_plan_rejects_invalid_engram_subsets():
    for ranks, message in (
        ((), "must not be empty"),
        ((1, 2), "include the backbone"),
        ((0, 1, 1), "duplicates"),
        ((0, 3), "outside the EP world"),
    ):
        with pytest.raises(ValueError, match=message):
            DeepseekV41ExecutionPlan(
                0, 3, backbone_rank=0, engram_ranks=ranks
            )


def test_execution_plan_phase_splits_prefill_from_decode_storage():
    plans = [
        DeepseekV41ExecutionPlan(
            rank,
            3,
            backbone_rank=0,
            expert_shards=(3, 3, 2),
            prefill_expert_shards=(4, 4, 0),
            expert_storage_ranges=((0, 4), (3, 5), (6, 2)),
            engram_ranks=(0, 1),
        )
        for rank in range(3)
    ]
    assert plans[0].prefill_active_ranks == (0, 1)
    assert all(plan.supports_packed_prefill for plan in plans)
    assert [plan.participates_in_prefill for plan in plans] == [True, True, False]
    assert [
        (plan.storage_partition(8).global_offset, plan.storage_partition(8).local_count)
        for plan in plans
    ] == [(0, 4), (3, 5), (6, 2)]
    assert [plan.partition(8, prefill=True).local_count for plan in plans] == [4, 4, 0]
    assert [plan.partition(8).local_count for plan in plans] == [3, 3, 2]


def test_execution_plan_rejects_storage_that_misses_phase_ownership():
    plan = DeepseekV41ExecutionPlan(
        1,
        3,
        backbone_rank=0,
        expert_shards=(3, 3, 2),
        prefill_expert_shards=(4, 4, 0),
        expert_storage_ranges=((0, 4), (4, 4), (6, 2)),
    )
    with pytest.raises(ValueError, match="decode ownership"):
        plan.storage_partition(8)


def test_fused_cache_safe_fallback_indexes_from_overlapping_storage():
    ownership = ExpertPartition(8, world_size=2, rank=1, shard_counts=(4, 4))
    storage = ExpertStorageRange(8, global_offset=3, local_count=5)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    indices = torch.tensor([[3, 4, 7, 1]])
    local_weights, safe_ids = _prepare_partitioned_routes(
        weights,
        indices,
        ownership,
        fused_cache_safe=True,
        storage=storage,
    )
    assert torch.equal(local_weights, torch.tensor([[0.0, 0.2, 0.3, 0.0]]))
    assert torch.equal(safe_ids, torch.tensor([[1, 1, 4, 1]]))


def test_attention_tp2_plan_keeps_both_backbones_but_root_owns_shared_path():
    root = DeepseekV41ExecutionPlan(
        0, 2, backbone_rank=0, attention_tp2_ep2=True
    )
    peer = DeepseekV41ExecutionPlan(
        1, 2, backbone_rank=0, attention_tp2_ep2=True
    )
    assert root.is_backbone and peer.is_backbone
    assert not root.is_expert_worker and not peer.is_expert_worker
    assert root.attention_parallel and peer.attention_parallel
    assert not root.shared_expert_parallel and not peer.shared_expert_parallel
    assert not root.uses_authority_transport and not peer.uses_authority_transport


def test_dense_parallel_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        DeepseekV41ExecutionPlan(
            0,
            2,
            backbone_rank=0,
            tp2_ep2=True,
            attention_tp2_ep2=True,
        )


def test_router_bias_selects_but_unbiased_score_scales_routes():
    args = DeepseekV41Args(
        dim=4,
        n_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        compress_ratios=(0,),
        engram_layer_ids=(),
        engram_num_embeddings=(),
        route_scale=1.5,
    )
    gate = Gate(args)
    gate.weight.data.copy_(torch.eye(4, dtype=torch.bfloat16))
    gate.bias.data.copy_(torch.tensor([0.0, 100.0, 0.0, 50.0]))
    hidden = torch.tensor([[4.0, -3.0, 2.0, 1.0]], dtype=torch.bfloat16)
    got_weights, got_ids = gate(hidden)

    scores = F.softplus(hidden.float()).sqrt()
    want_ids = (scores + gate.bias).topk(2, -1).indices
    want_weights = scores.gather(-1, want_ids)
    want_weights = want_weights / (want_weights.sum(-1, keepdim=True) + 1e-20) * 1.5
    torch.testing.assert_close(got_ids, want_ids)
    torch.testing.assert_close(got_weights, want_weights)
    assert got_ids.tolist() == [[1, 3]]


def test_authority_decode_refill_is_forked_around_shared_expert(monkeypatch):
    import freetoken.models.deepseek_v41.model as model_module
    from freetoken.models.deepseek_v41.model import MoEState

    calls = []

    class FakeGate(nn.Module):
        def forward(self, hidden):
            calls.append("router")
            rows = hidden.shape[0]
            return (
                torch.ones((rows, 2), dtype=torch.float32),
                torch.tensor([[0, 1]], dtype=torch.int32).expand(rows, -1).clone(),
            )

    class FakeShared(nn.Module):
        def forward(self, hidden):
            calls.append("shared")
            return hidden + 1

    class FakeExperts(nn.Module):
        top_k = 2

        def prepare_packed_prefill_receive(self, weights, route_dtype):
            calls.append("prepare")

        def begin_routed_decode(self, weights, ids):
            calls.append("refill-fork")
            return object()

        def finish_routed_decode(self, hidden, weights, plan):
            calls.append("refill-join+routed")
            return torch.zeros_like(hidden)

        def routed_forward(self, hidden, weights, ids):
            raise AssertionError("split decode must not run the serial route path")

    state = MoEState.__new__(MoEState)
    nn.Module.__init__(state)
    state.dim = 4
    state.gate = FakeGate()
    state.shared_experts = FakeShared()
    state.experts = FakeExperts()
    state.execution = type(
        "Plan",
        (),
        {
            "uses_authority_transport": True,
            "backbone_rank": 0,
            "rank": 0,
            "enabled": True,
            "tp2_ep2": False,
            "attention_tp2_ep2": False,
        },
    )()
    state.partition = object()
    state._comm = type("Comm", (), {"broadcast": staticmethod(lambda value, root: value)})()
    state.decode_refill_overlap = True

    monkeypatch.setattr(
        "freetoken.moe.partition.localize_expert_routes",
        lambda weights, ids, partition: (weights, ids),
    )
    monkeypatch.setattr(
        model_module,
        "get_global_ctx",
        lambda: type("Ctx", (), {"batch": type("Batch", (), {"is_prefill": False})()})(),
    )

    hidden = torch.zeros((1, 1, 4), dtype=torch.bfloat16)
    output = state(hidden)

    assert calls == [
        "router",
        "prepare",
        "refill-fork",
        "shared",
        "refill-join+routed",
    ]
    torch.testing.assert_close(output, hidden + 1)
