from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.execution import DeepseekV41ExecutionPlan
from freetoken.models.deepseek_v41.moe import Gate


def test_execution_plan_balances_global_experts_and_keeps_workers_in_engram():
    left = DeepseekV41ExecutionPlan(0, 2, backbone_rank=0).partition(385)
    right_plan = DeepseekV41ExecutionPlan(1, 2, backbone_rank=0)
    right = right_plan.partition(385)
    assert (left.global_offset, left.local_count) == (0, 193)
    assert (right.global_offset, right.local_count) == (193, 192)
    assert right_plan.is_expert_worker
    assert right_plan.participates_in_engram


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
