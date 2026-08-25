from types import SimpleNamespace

import pytest
import torch

from freetoken.models.deepseek_v4.execution import DSV4ExecutionPlan
from freetoken.moe.partition import ExpertPartition


def test_execution_plan_preserves_legacy_replicated_backbone_by_default():
    plan = DSV4ExecutionPlan(rank=1, world_size=3)

    assert not plan.enabled
    assert plan.is_backbone
    assert not plan.is_expert_worker


def test_execution_plan_assigns_one_backbone_and_remaining_workers():
    plans = [DSV4ExecutionPlan(rank=r, world_size=3, backbone_rank=0) for r in range(3)]

    assert [p.is_backbone for p in plans] == [True, False, False]
    assert [p.is_expert_worker for p in plans] == [False, True, True]


def test_dummy_weights_materialize_e8m0_as_unit_scale():
    from freetoken.engine.engine import _make_dummy_weight_state_dict

    state = _make_dummy_weight_state_dict(
        {"scale": torch.empty(2, 3, dtype=torch.float8_e8m0fnu)},
        device=torch.device("cpu"),
    )

    assert state["scale"].dtype == torch.float8_e8m0fnu
    assert state["scale"].view(torch.uint8).tolist() == [[127] * 3] * 2


@pytest.mark.parametrize("backbone_rank", [-1, 3])
def test_execution_plan_rejects_invalid_backbone_rank(backbone_rank):
    with pytest.raises(ValueError, match="backbone_rank"):
        DSV4ExecutionPlan(rank=0, world_size=3, backbone_rank=backbone_rank)


def test_expert_worker_model_owns_only_routed_expert_shell(monkeypatch):
    from freetoken.models.deepseek_v4 import config as dsv4_config
    from freetoken.models.deepseek_v4 import model as dsv4_model
    from freetoken.models.deepseek_v4 import moe as dsv4_moe
    from freetoken.models.deepseek_v4.args import DeepseekV4Args
    import freetoken.layers.moe as moe_layers

    plan = DSV4ExecutionPlan(rank=1, world_size=3, backbone_rank=0)
    monkeypatch.setattr(dsv4_model, "get_dsv4_execution_plan", lambda: plan)
    monkeypatch.setattr(dsv4_moe, "get_dsv4_execution_plan", lambda: plan)
    monkeypatch.setattr(
        moe_layers, "get_tp_info", lambda: SimpleNamespace(rank=1, size=3)
    )
    monkeypatch.setattr(
        dsv4_config,
        "ep_partition",
        lambda total: ExpertPartition(total, world_size=3, rank=1),
    )

    # Keep every quantized dimension block-aligned while making construction cheap.
    args = DeepseekV4Args(
        vocab_size=256,
        dim=128,
        moe_inter_dim=128,
        n_layers=2,
        n_hash_layers=1,
        n_heads=1,
        n_routed_experts=8,
        n_activated_experts=2,
        head_dim=128,
        rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        window_size=128,
        compress_ratios=(0, 0),
    )
    config = SimpleNamespace(dsv4_args=args)

    with torch.device("meta"):
        model = dsv4_model.DeepseekV4ForCausalLM(config)

    keys = set(model.state_dict())
    assert keys == set()
    assert all(block.ffn.gate is None for block in model._transformer.layers)
    assert all(block.ffn.shared_experts is None for block in model._transformer.layers)
    assert not hasattr(model._transformer, "embed")
