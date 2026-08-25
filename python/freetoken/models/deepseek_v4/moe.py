"""DSV4 MoE: sqrtsoftplus/hash router, shared SwiGLU expert, offloaded FP4 routed
experts (GPU slot-cache / cpu / hybrid decode paths)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.distributed import DistributedCommunicator
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import OffloadMoELayer
from freetoken.moe.partition import ExpertPartition

from .args import DeepseekV4Args
from .execution import get_dsv4_execution_plan
from .layers import Linear


def localize_expert_routes(
    weights: torch.Tensor,
    indices: torch.Tensor,
    partition: ExpertPartition,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep this rank's routes and map their global ids to its local bank.

    Foreign routes use the one-past-the-end expert id reserved by the grouped
    MoE alignment kernels. Cache decode replaces the sentinel with an already-
    requested local id immediately before admission, while retaining the zero
    route weight.
    """
    local = indices - partition.global_offset
    owned = (local >= 0) & (local < partition.local_count)
    local = torch.where(owned, local, local.new_full((), partition.local_count))
    weights = torch.where(owned, weights, weights.new_zeros(()))
    return weights, local


class Gate(nn.Module):
    """MoE router: sqrtsoftplus scoring + hash routing (first ``n_hash_layers``)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16), requires_grad=False)
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int64), requires_grad=False
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        scores = bf16_linear_fp32(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Dense SwiGLU expert (the shared expert; routed experts are offloaded FP4)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, kind="fp8")
        self.w2 = Linear(inter_dim, dim, kind="fp8")
        self.w3 = Linear(dim, inter_dim, kind="fp8")
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = fused_swiglu(self.w1(x), self.w3(x), self.swiglu_limit, x.dtype)
        return self.w2(h)


class DSV4OffloadMoELayer(OffloadMoELayer):
    """Routed FP4 experts on the shared offload cache: the base whole-layer
    streaming prefill (grouped inline-dequant GEMM for dense chunks, GEMV
    below the route crossover) and slot-cache / cpu / hybrid decode paths
    (per-route dequant GEMV)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        from .config import ep_partition

        partition = ep_partition(args.n_routed_experts)

        super().__init__(
            layer_id=layer_id,
            # EP under TP>1: this layer computes only this rank's expert shard
            # (banks, cache and streaming are all local); partial outputs are
            # summed across ranks by routed_forward's _maybe_all_reduce.
            num_experts=partition.local_count,
            top_k=args.n_activated_experts,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=True,
            activation="silu",
        )
        self.swiglu_limit = args.swiglu_limit

    @staticmethod
    def _cache_safe_route_ids(
        topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        """Replace skipped sentinel routes before calling the slot-cache LRU.

        Each skipped position duplicates the row's first live local expert, so
        it creates no additional cache admission.  The all-foreign edge case
        falls back to expert zero; its route weights remain zero and the DSV4
        kernel masks its weight reads.
        """
        active = topk_weights != 0
        first_pos = active.to(torch.int64).argmax(dim=-1, keepdim=True)
        fallback = topk_ids.gather(-1, first_pos)
        fallback = torch.where(
            active.any(dim=-1, keepdim=True), fallback, fallback.new_zeros(())
        )
        return torch.where(active, topk_ids, fallback)

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        cache = self.offload_cache
        assert cache is not None
        if cache.is_cpu_layer(self.layer_id):
            # The CPU executor natively skips negative expert ids; translate
            # the grouped-kernel sentinel only at this boundary.
            cpu_ids = torch.where(
                topk_weights != 0, topk_ids, topk_ids.new_full((), -1)
            )
            return super()._decode_routed(hidden_states, topk_weights, cpu_ids)
        return super()._decode_routed(
            hidden_states,
            topk_weights,
            self._cache_safe_route_ids(topk_weights, topk_ids),
        )

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Whole-layer streaming moves all num_experts rows per layer; a small
        # chunk touches at most T*top_k of them, so below that crossover the
        # decode-style on-demand slot path strictly moves fewer bytes (and
        # keeps short-prompt slot residency -- hence hybrid decode's GPU/CPU
        # route split -- unchanged). Mixing modes across chunks is safe: the
        # streaming buffers disown their borrowed slots on invalidation.
        if hidden_states.shape[0] * self.top_k >= self.num_experts:
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        cache = self.offload_cache
        assert cache is not None
        topk_ids = self._cache_safe_route_ids(topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)  # in-place expert-id -> slot
        cache.copy_missing()
        if cache.collect_stats:
            cache.record_decode_stats(self.layer_id)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=True,
        )


class MoE(nn.Module):
    """Sparse MoE: hash/score router -> offloaded FP4 routed experts + shared expert."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        from .config import ep_partition

        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.execution = get_dsv4_execution_plan()
        self._comm = DistributedCommunicator()
        self.gate = None if self.execution.is_expert_worker else Gate(layer_id, args)
        self.shared_experts = (
            None
            if self.execution.is_expert_worker
            else Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
        )
        self.experts = DSV4OffloadMoELayer(layer_id, args)
        self.partition = ep_partition(args.n_routed_experts)
        self.ep_active = self.partition.world_size > 1

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self.execution.is_expert_worker:
            raise RuntimeError("expert-worker ranks must call worker_forward")
        shape = x.size()
        if self.execution.enabled:
            # Workers are already waiting in the matching collective with an empty
            # tensor. The authority is the sole source, so broadcast is exact even
            # when PyNCCL implements it as source-only SUM.
            x = self._comm.broadcast(x.contiguous(), self.execution.backbone_rank)
        x = x.view(-1, self.dim)
        assert self.gate is not None
        weights, indices = self.gate(x, input_ids.flatten())
        if self.execution.enabled:
            # Route once on the authority. Sending int32 global ids avoids both
            # duplicated router weights and mixed-SKU route disagreement.
            weights = self._comm.broadcast(
                weights.contiguous(), self.execution.backbone_rank
            )
            indices = self._comm.broadcast(
                indices.to(torch.int32).contiguous(), self.execution.backbone_rank
            )
        # Shared expert enqueued before routed_forward: hybrid decode blocks on the
        # CPU pool inside routed_forward, so this GEMM must already be on the stream
        # to overlap the CPU overflow compute.
        assert self.shared_experts is not None
        shared = self.shared_experts(x)
        if self.ep_active:
            # EP: keep only this rank's routes — remap to local bank ids and mark
            # foreign routes with the alignment sentinel. Routes stay normalized
            # globally, so the cross-rank all-reduce reconstructs the full sum.
            # (The shared expert is replicated and added after the reduce.)
            weights, indices = localize_expert_routes(weights, indices, self.partition)
        # routed_forward may mutate the ids in place (offload decode slot remap);
        # indices.to(int32) always copies (int64 source), so no clone needed here.
        routed = self.experts.routed_forward(
            x, weights.float().contiguous(), indices.to(torch.int32).contiguous()
        )
        return (routed + shared).view(shape)

    def worker_forward(
        self, input_ids: torch.Tensor, hidden_shape: tuple[int, ...]
    ) -> None:
        """Receive one layer's normalized MoE input and contribute local experts.

        The routed-expert all-reduce inside ``routed_forward`` is deliberately
        retained: the authority consumes the reconstructed routed output while
        workers discard it and wait for the next layer's broadcast.
        """
        if not self.execution.is_expert_worker:
            raise RuntimeError("worker_forward is valid only on expert-worker ranks")
        x = torch.empty(
            hidden_shape,
            dtype=torch.bfloat16,
            device=input_ids.device,
        )
        x = self._comm.broadcast(x, self.execution.backbone_rank).view(-1, self.dim)
        route_shape = (x.shape[0], self.topk)
        weights = self._comm.broadcast(
            torch.empty(route_shape, dtype=torch.float32, device=x.device),
            self.execution.backbone_rank,
        )
        indices = self._comm.broadcast(
            torch.empty(route_shape, dtype=torch.int32, device=x.device),
            self.execution.backbone_rank,
        )
        if self.ep_active:
            weights, indices = localize_expert_routes(weights, indices, self.partition)
        self.experts.routed_forward(
            x, weights.float().contiguous(), indices.to(torch.int32).contiguous()
        )


__all__ = ["DSV4OffloadMoELayer", "Gate", "MoE", "localize_expert_routes"]
