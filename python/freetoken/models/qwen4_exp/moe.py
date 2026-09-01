from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from freetoken.distributed import DistributedCommunicator
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers import ExpertParallelOffloadMoELayer, LinearReplicated
from freetoken.layers.moe import make_moe_layer
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE, _SharedExpert
from freetoken.moe.fused import fused_topk
from freetoken.moe.partition import localize_expert_routes

from .execution import get_qwen4_exp_execution_plan

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None) -> None:
        self.execution = get_qwen4_exp_execution_plan()
        self.partition = self.execution.partition(config.qwen4_args.num_experts)
        self.renormalize = config.norm_topk_prob
        self._comm = DistributedCommunicator()
        if self.execution.enabled:
            # The worker owns only its routed-expert shard. The authority owns the
            # full global router and BF16 shared expert, exactly as in TP1.
            self.gate = (
                None
                if self.execution.is_expert_worker
                else LinearReplicated(
                    config.hidden_size, config.qwen4_args.num_experts, has_bias=False
                )
            )
            shared_config = (
                replace(config, expert_quant="none")
                if getattr(config, "expert_quant", "none") == "fp8_block"
                else config
            )
            self.shared_expert = (
                None
                if self.execution.is_expert_worker
                else _SharedExpert(
                    shared_config,
                    config.hidden_size,
                    config.shared_expert_intermediate_size,
                )
            )
            self.shared_expert_gate = (
                None
                if self.execution.is_expert_worker
                else LinearReplicated(config.hidden_size, 1, has_bias=False)
            )
            self.experts = make_moe_layer(
                config,
                layer_id=layer_id,
                num_experts=self.partition.local_count,
                renormalize=config.norm_topk_prob,
                weight_format=(
                    "fp8_block"
                    if getattr(config, "expert_quant", "none") == "fp8_block"
                    else "bf16"
                ),
                offload_cls=ExpertParallelOffloadMoELayer,
            )
            self.experts.packed_prefill_root = self.execution.backbone_rank
            return
        if getattr(config, "expert_quant", "none") != "fp8_block":
            super().__init__(config, layer_id=layer_id)
            return
        # Qwen3.8's block-fp8 checkpoint quantizes only the routed experts; the shared
        # expert stays bf16, so hide expert_quant from _SharedExpert's fp8 branch and
        # rebuild the routed experts with the fp8_block bank layout.
        super().__init__(replace(config, expert_quant="none"), layer_id=layer_id)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format="fp8_block",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.execution.enabled:
            return self._forward_tp1(hidden_states)
        if self.execution.is_expert_worker:
            raise RuntimeError("Qwen expert-worker ranks must call worker_forward")

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        hidden_states = self._comm.broadcast(
            hidden_states.contiguous(), self.execution.backbone_rank
        )
        assert self.gate is not None
        router_logits = self.gate.forward(hidden_states)
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.experts.top_k,
            renormalize=self.renormalize,
        )
        topk_weights = self._comm.broadcast(
            topk_weights.float().contiguous(), self.execution.backbone_rank
        )
        topk_ids = self._comm.broadcast(
            topk_ids.to(torch.int32).contiguous(), self.execution.backbone_rank
        )
        topk_weights, topk_ids = localize_expert_routes(
            topk_weights, topk_ids, self.partition
        )
        self.experts.prepare_packed_prefill_receive(topk_weights, hidden_states.dtype)
        assert self.shared_expert is not None and self.shared_expert_gate is not None
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.routed_forward(
            hidden_states,
            topk_weights.float().contiguous(),
            topk_ids.to(torch.int32).contiguous(),
        )
        routed = routed.to(hidden_states.dtype)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)

    def _forward_tp1(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)

    def worker_forward(self, hidden_shape: tuple[int, ...], device: torch.device) -> None:
        if not self.execution.is_expert_worker:
            raise RuntimeError("worker_forward is valid only on Qwen expert-worker ranks")
        hidden_states = self._comm.broadcast(
            torch.empty(hidden_shape, dtype=torch.bfloat16, device=device),
            self.execution.backbone_rank,
        ).view(-1, hidden_shape[-1])
        route_shape = (hidden_states.shape[0], self.experts.top_k)
        topk_weights = self._comm.broadcast(
            torch.empty(route_shape, dtype=torch.float32, device=device),
            self.execution.backbone_rank,
        )
        topk_ids = self._comm.broadcast(
            torch.empty(route_shape, dtype=torch.int32, device=device),
            self.execution.backbone_rank,
        )
        topk_weights, topk_ids = localize_expert_routes(
            topk_weights, topk_ids, self.partition
        )
        self.experts.routed_forward(
            hidden_states,
            topk_weights.float().contiguous(),
            topk_ids.to(torch.int32).contiguous(),
        )


__all__ = ["Qwen4ExpMoE"]
