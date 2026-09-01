from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.moe import ExpertParallelOffloadMoELayer, OffloadMoELayer
from freetoken.models.qwen3_5_moe.weight import _expert_partition
from freetoken.models.qwen4_exp import execution
from freetoken.models.qwen4_exp.execution import Qwen4ExpExecutionPlan
from freetoken.moe.partition import (
    ExpertPartition,
    cache_safe_route_ids,
    localize_expert_routes,
)


def test_ep_offload_preserves_route_slots_through_collective():
    assert ExpertParallelOffloadMoELayer.return_route_outputs is True


def test_balanced_partition_and_local_routes():
    partition = ExpertPartition(512, world_size=2, rank=1)
    assert partition.local_count == 256
    assert partition.global_offset == 256
    assert partition.global_stop == 512

    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    indices = torch.tensor([[10, 256, 511, 100]])
    local_weights, local_ids = localize_expert_routes(weights, indices, partition)
    assert torch.equal(local_weights, torch.tensor([[0.0, 0.2, 0.3, 0.0]]))
    assert torch.equal(local_ids, torch.tensor([[256, 0, 255, 256]]))
    assert torch.equal(
        cache_safe_route_ids(local_weights, local_ids),
        torch.tensor([[0, 0, 255, 0]]),
    )


def test_heterogeneous_partition_validation():
    partition = ExpertPartition(512, world_size=2, rank=0, shard_counts=(96, 416))
    assert (partition.global_offset, partition.global_stop) == (0, 96)
    with pytest.raises(ValueError, match="must sum"):
        ExpertPartition(512, world_size=2, rank=0, shard_counts=(100, 400))


def test_qwen_fp8_loader_uses_rank_local_partition(monkeypatch):
    monkeypatch.setattr(
        execution,
        "_PLAN",
        Qwen4ExpExecutionPlan(1, 2, backbone_rank=0, expert_shards=(96, 416)),
    )
    config = SimpleNamespace(
        num_experts=416,
        qwen4_args=SimpleNamespace(num_experts=512),
    )
    partition = _expert_partition(config)
    assert partition.local_count == 416
    assert partition.global_offset == 96


def test_qwen_fp8_loader_rejects_config_partition_disagreement(monkeypatch):
    monkeypatch.setattr(
        execution,
        "_PLAN",
        Qwen4ExpExecutionPlan(0, 2, backbone_rank=0, expert_shards=(96, 416)),
    )
    config = SimpleNamespace(
        num_experts=256,
        qwen4_args=SimpleNamespace(num_experts=512),
    )
    with pytest.raises(ValueError, match="disagree"):
        _expert_partition(config)


@pytest.mark.parametrize("path", ["_prefill_routed", "_decode_routed"])
def test_ep_offload_sanitizes_inactive_route_ids_before_kernels(monkeypatch, path):
    captured = {}

    def capture(_self, hidden_states, topk_weights, topk_ids):
        captured["weights"] = topk_weights.clone()
        captured["ids"] = topk_ids.clone()
        return hidden_states

    monkeypatch.setattr(OffloadMoELayer, path, capture)
    layer = object.__new__(ExpertParallelOffloadMoELayer)
    hidden = torch.zeros((2, 4), dtype=torch.bfloat16)
    weights = torch.tensor([[0.2, 0.0, 0.3], [0.0, 0.0, 0.0]])
    # 256 is the one-past-the-end sentinel for a 256-expert local bank.
    ids = torch.tensor([[7, 256, 9], [256, 256, 256]], dtype=torch.int32)

    output = getattr(layer, path)(hidden, weights, ids)

    assert output is hidden
    assert torch.equal(captured["weights"], weights)
    assert torch.equal(
        captured["ids"],
        torch.tensor([[7, 7, 9], [0, 0, 0]], dtype=torch.int32),
    )
