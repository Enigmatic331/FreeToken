"""V4.1 sqrt-softplus router, shared expert, and Qwen-style EP transport."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator
from freetoken.layers import ExpertParallelOffloadMoELayer

from .args import DeepseekV41Args
from .execution import get_execution_plan
from .layers import Linear
from .profile import profile_range


def _fused_route_prep_enabled() -> bool:
    return os.getenv("FREETOKEN_DSV41_FUSED_ROUTE_PREP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prepare_partitioned_routes(
    weights: torch.Tensor,
    ids: torch.Tensor,
    ownership,
    *,
    fused_cache_safe: bool,
    storage=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fused_cache_safe:
        if storage is None or storage.global_offset == ownership.global_offset:
            from freetoken.kernel.triton.dsv41 import fused_localize_cache_safe_routes

            return fused_localize_cache_safe_routes(
                weights,
                ids,
                global_offset=ownership.global_offset,
                local_count=ownership.local_count,
            )
        # The fused kernel indexes from the ownership offset. If storage starts
        # earlier, preserve the cache-safe contract through the exact composed
        # path instead of returning ids relative to the wrong origin.
        from freetoken.moe.partition import (
            cache_safe_route_ids,
            localize_expert_routes_to_storage,
        )

        weights, ids = localize_expert_routes_to_storage(
            weights, ids, ownership, storage
        )
        return weights, cache_safe_route_ids(weights, ids)
    if storage is not None:
        from freetoken.moe.partition import localize_expert_routes_to_storage

        return localize_expert_routes_to_storage(weights, ids, ownership, storage)
    from freetoken.moe.partition import localize_expert_routes

    return localize_expert_routes(weights, ids, ownership)


class Gate(nn.Module):
    """Bias-corrected selection with unbiased, normalized route weights."""

    def __init__(self, args: DeepseekV41Args) -> None:
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(
            torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(args.n_routed_experts, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.is_cuda:
            from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32

            scores = bf16_linear_fp32(x, self.weight)
            fused = os.getenv(
                "FREETOKEN_DSV41_FUSED_ROUTER", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if fused:
                from freetoken.kernel.triton.dsv41.router import (
                    fused_sqrtsoftplus_topk,
                )

                if self.score_func != "sqrtsoftplus":
                    raise ValueError(
                        f"unsupported V4.1 route score {self.score_func}"
                    )
                return fused_sqrtsoftplus_topk(
                    scores,
                    self.bias,
                    topk=self.topk,
                    temperature=self.gate_temp,
                    renormalize=self.norm_topk_prob,
                    route_scale=self.route_scale,
                )
            scores = scores / self.gate_temp
        else:
            scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        else:
            raise ValueError(f"unsupported V4.1 route score {self.score_func}")
        indices = (scores + self.bias).topk(self.topk, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= weights.sum(-1, keepdim=True) + 1e-20
        return weights * self.route_scale, indices


class SharedExpert(nn.Module):
    def __init__(self, args: DeepseekV41Args) -> None:
        super().__init__()
        execution = get_execution_plan()
        parallel = execution.shared_expert_parallel
        self.w1 = Linear(
            args.dim,
            args.moe_inter_dim,
            parallel="column" if parallel else None,
        )
        self.w2 = Linear(
            args.moe_inter_dim,
            args.dim,
            parallel="row" if parallel else None,
        )
        self.w3 = Linear(
            args.dim,
            args.moe_inter_dim,
            parallel="column" if parallel else None,
        )
        self.limit = args.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu

        hidden = fused_swiglu(self.w1(x), self.w3(x), self.limit, x.dtype)
        return self.w2(hidden)


class RoutedExperts(ExpertParallelOffloadMoELayer):
    def __init__(self, layer_id: int, args: DeepseekV41Args, local_experts: int) -> None:
        super().__init__(
            layer_id=layer_id,
            num_experts=local_experts,
            top_k=args.n_activated_experts,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=True,
            activation="silu",
        )
        self.swiglu_limit = args.swiglu_limit


class MoE(nn.Module):
    """Authority computes router/shared expert; all ranks compute owned routes."""

    def __init__(self, layer_id: int, args: DeepseekV41Args) -> None:
        super().__init__()
        self.dim = args.dim
        self.execution = get_execution_plan()
        self.decode_partition = self.execution.partition(args.n_routed_experts)
        self.prefill_partition = self.execution.partition(
            args.n_routed_experts, prefill=True
        )
        self.partition = self.decode_partition
        self.storage = self.execution.storage_partition(args.n_routed_experts)
        self._comm = DistributedCommunicator()
        self.gate = None if self.execution.is_expert_worker else Gate(args)
        self.shared_experts = (
            None if self.execution.is_expert_worker else SharedExpert(args)
        )
        self.experts = RoutedExperts(layer_id, args, self.storage.local_count)
        self.experts.packed_prefill_root = (
            self.execution.backbone_rank
            if self.execution.supports_packed_prefill
            else None
        )
        if self.execution.supports_packed_prefill:
            peers = tuple(
                rank
                for rank in self.execution.prefill_active_ranks
                if rank != self.execution.backbone_rank
            )
            if len(peers) != 1:
                raise RuntimeError("packed prefill requires exactly one active peer")
            self.experts.packed_prefill_peer_rank = peers[0]
        self.fused_route_prep = _fused_route_prep_enabled()

    def _phase_broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        root = self.execution.backbone_rank
        assert root is not None
        batch = get_global_ctx().batch
        if not batch.is_prefill or not self.execution.phase_aware:
            return self._comm.broadcast(tensor, root)
        if self.execution.rank == root:
            for peer in self.execution.prefill_active_ranks:
                if peer != root:
                    self._comm.send(tensor, peer)
            return tensor
        if self.execution.participates_in_prefill:
            return self._comm.recv(tensor, root)
        raise RuntimeError("inactive prefill rank attempted a phase broadcast")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.execution.is_expert_worker:
            raise RuntimeError("V4.1 expert workers must call worker_forward")
        shape = x.shape
        hidden = x.view(-1, self.dim)
        if self.execution.uses_authority_transport:
            hidden = self._phase_broadcast(hidden.contiguous())
        assert self.gate is not None and self.shared_experts is not None
        weights, ids = self.gate(hidden)
        if self.execution.uses_authority_transport:
            weights = self._phase_broadcast(weights.float().contiguous())
            ids = self._phase_broadcast(ids.to(torch.int32).contiguous())
        fused_cache_safe = (
            self.fused_route_prep
            and weights.is_cuda
            and not get_global_ctx().batch.is_prefill
        )
        ownership = (
            self.prefill_partition
            if get_global_ctx().batch.is_prefill
            else self.decode_partition
        )
        weights, ids = _prepare_partitioned_routes(
            weights,
            ids,
            ownership,
            fused_cache_safe=fused_cache_safe,
            storage=self.storage,
        )
        if not self.execution.tp2_ep2:
            self.experts.prepare_packed_prefill_receive(weights, hidden.dtype)
        shared = self.shared_experts(hidden)
        if self.execution.tp2_ep2:
            self.experts.prepare_packed_prefill_receive(weights, hidden.dtype)
        routed_weights = weights.float().contiguous()
        routed_ids = ids.to(torch.int32).contiguous()
        if fused_cache_safe:
            routed = self.experts.routed_decode_cache_safe(
                hidden, routed_weights, routed_ids
            )
        else:
            routed = self.experts.routed_forward(hidden, routed_weights, routed_ids)
        if self.execution.tp2_ep2 and get_global_ctx().batch.is_prefill:
            if self.execution.rank != self.execution.backbone_rank:
                routed = torch.empty_like(hidden)
            routed = self._comm.broadcast(
                routed.contiguous(), self.execution.backbone_rank
            )
        return (shared + routed.to(shared.dtype)).view(shape)

    def worker_forward(self, hidden_shape: tuple[int, ...], device: torch.device) -> None:
        if not self.execution.is_expert_worker:
            raise RuntimeError("worker_forward is valid only on an expert worker")
        with profile_range("DSV41/EP/WorkerReceive"):
            hidden = self._phase_broadcast(
                torch.empty(hidden_shape, dtype=torch.bfloat16, device=device)
            ).view(-1, self.dim)
            route_shape = (hidden.shape[0], self.experts.top_k)
            weights = self._phase_broadcast(
                torch.empty(route_shape, dtype=torch.float32, device=device)
            )
            ids = self._phase_broadcast(
                torch.empty(route_shape, dtype=torch.int32, device=device)
            )
            fused_cache_safe = (
                self.fused_route_prep
                and weights.is_cuda
                and not get_global_ctx().batch.is_prefill
            )
            ownership = (
                self.prefill_partition
                if get_global_ctx().batch.is_prefill
                else self.decode_partition
            )
            weights, ids = _prepare_partitioned_routes(
                weights,
                ids,
                ownership,
                fused_cache_safe=fused_cache_safe,
                storage=self.storage,
            )
        with profile_range("DSV41/MoE/WorkerRoutedExpert"):
            if fused_cache_safe:
                self.experts.routed_decode_cache_safe(
                    hidden, weights.contiguous(), ids.contiguous()
                )
            else:
                self.experts.routed_forward(
                    hidden, weights.contiguous(), ids.contiguous()
                )


__all__ = ["Gate", "MoE", "RoutedExperts", "SharedExpert"]
