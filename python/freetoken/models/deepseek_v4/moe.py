"""DSV4 MoE: sqrtsoftplus/hash router, shared SwiGLU expert, offloaded FP4 routed
experts (GPU slot-cache / cpu / hybrid decode paths)."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.distributed import DistributedCommunicator
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import ExpertParallelOffloadMoELayer
from freetoken.moe.partition import ExpertPartition, localize_expert_routes

from .args import DeepseekV4Args
from .execution import get_dsv4_execution_plan
from .layers import Linear


_EP_PACKED_PREFILL_OVERLAP = os.getenv("FREETOKEN_EP_PACKED_OVERLAP", "1") != "0"
_EP_PACKED_RECV_STREAM: torch.cuda.Stream | None = None


def _ep_packed_recv_stream(device: torch.device) -> torch.cuda.Stream:
    global _EP_PACKED_RECV_STREAM
    if _EP_PACKED_RECV_STREAM is None:
        _EP_PACKED_RECV_STREAM = torch.cuda.Stream(device=device)
    return _EP_PACKED_RECV_STREAM


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


class DSV4OffloadMoELayer(ExpertParallelOffloadMoELayer):
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
        # EP must preserve canonical route slots through communication. Every
        # route has exactly one owner, so summing the zero-filled route tensor
        # copies each BF16 route contribution without first changing its
        # association. The reducer below then reproduces TP1's top-k order.
        self.return_route_outputs = partition.world_size > 1
        self.packed_prefill_root: int | None = None
        self._packed_prefill_pending = None

    def prepare_packed_prefill_receive(
        self,
        local_topk_weights: torch.Tensor,
        route_dtype: torch.dtype,
    ) -> None:
        """Post rank 0's packed receive before local expert work starts.

        The worker still sends the same owned BF16 route rows and rank 0 still
        inserts them into the same canonical route slots. Only the receive's
        scheduling changes, so the strict FP32 top-k reduction is untouched.
        """
        from freetoken.core import get_global_ctx
        from freetoken.distributed import get_tp_info

        root = self.packed_prefill_root
        info = get_tp_info()
        if (
            root is None
            or info.rank != root
            or not get_global_ctx().batch.is_prefill
            or not _EP_PACKED_PREFILL_OVERLAP
        ):
            return
        if info.size != 2:
            raise RuntimeError("packed DSV4 prefill routes require exactly two EP ranks")
        if self._packed_prefill_pending is not None:
            raise RuntimeError("packed DSV4 prefill receive already pending")

        owned = local_topk_weights.ne(0).reshape(-1)
        remote_indices = torch.nonzero(~owned, as_tuple=False).flatten()
        if remote_indices.numel() == 0:
            self._packed_prefill_pending = (remote_indices, None, None)
            return

        packed = torch.empty(
            (remote_indices.numel(), self.hidden_size),
            dtype=route_dtype,
            device=local_topk_weights.device,
        )
        current = torch.cuda.current_stream(local_topk_weights.device)
        recv_stream = _ep_packed_recv_stream(local_topk_weights.device)
        recv_stream.wait_stream(current)
        with torch.cuda.stream(recv_stream):
            self._comm.recv(packed, 1 - root)
        self._packed_prefill_pending = (remote_indices, packed, recv_stream)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        if get_global_ctx().batch.is_prefill:
            routes = self._prefill_routed(hidden_states, topk_weights, topk_ids)
            return self._maybe_combine_packed_prefill_routes(routes, topk_weights)
        routes = self._decode_routed(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(routes)

    def _maybe_combine_packed_prefill_routes(
        self,
        routes: torch.Tensor,
        local_topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        root = self.packed_prefill_root
        if root is None:
            return self._maybe_all_reduce(routes)

        from freetoken.distributed import get_tp_info

        info = get_tp_info()
        if info.size != 2:
            raise RuntimeError("packed DSV4 prefill routes require exactly two EP ranks")
        if routes.ndim != 3:
            raise RuntimeError(f"expected per-route outputs, got {tuple(routes.shape)}")
        if routes.shape[:2] != local_topk_weights.shape:
            raise RuntimeError(
                "route output and weight shapes disagree: "
                f"{tuple(routes.shape[:2])} != {tuple(local_topk_weights.shape)}"
            )

        flat_routes = routes.view(-1, routes.shape[-1])
        owned = local_topk_weights.ne(0).reshape(-1)
        if info.rank != root:
            owned_indices = torch.nonzero(owned, as_tuple=False).flatten()
            if owned_indices.numel() != 0:
                packed = flat_routes.index_select(0, owned_indices).contiguous()
                self._comm.send(packed, root)
            return routes

        pending = getattr(self, "_packed_prefill_pending", None)
        self._packed_prefill_pending = None
        if pending is not None:
            remote_indices, packed, recv_stream = pending
            if packed is not None:
                torch.cuda.current_stream(routes.device).wait_stream(recv_stream)
                flat_routes.index_copy_(0, remote_indices, packed)
        else:
            remote_indices = torch.nonzero(~owned, as_tuple=False).flatten()
            if remote_indices.numel() != 0:
                packed = routes.new_empty((remote_indices.numel(), routes.shape[-1]))
                self._comm.recv(packed, 1 - root)
                flat_routes.index_copy_(0, remote_indices, packed)

        from freetoken.kernel import moe_sum_reduce_triton

        output = routes.new_empty((routes.shape[0], routes.shape[-1]))
        moe_sum_reduce_triton(routes, output)
        return output

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            return super()._maybe_all_reduce(hidden_states)
        if self.tp_size > 1:
            hidden_states = self._comm.all_reduce(hidden_states)

        from freetoken.kernel import moe_sum_reduce_triton

        num_tokens, _, hidden_dim = hidden_states.shape
        output = torch.empty(
            (num_tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        moe_sum_reduce_triton(hidden_states, output)
        return output

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
        if self.execution.enabled:
            self.experts.packed_prefill_root = self.execution.backbone_rank
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
        if self.ep_active:
            # EP: keep only this rank's routes — remap to local bank ids and mark
            # foreign routes with the alignment sentinel. Routes stay normalized
            # globally, so the cross-rank all-reduce reconstructs the full sum.
            # (The shared expert is replicated and added after the reduce.)
            weights, indices = localize_expert_routes(weights, indices, self.partition)
            self.experts.prepare_packed_prefill_receive(weights, x.dtype)
        # Shared expert enqueued before routed_forward: hybrid decode blocks on the
        # CPU pool inside routed_forward, so this GEMM must already be on the stream
        # to overlap the CPU overflow compute. During prefill, rank 0 has already
        # posted the packed worker-route receive on its side stream.
        assert self.shared_experts is not None
        shared = self.shared_experts(x)
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

        Decode retains the fixed-shape exact all-reduce for CUDA-graph safety.
        Prefill instead sends only this worker's owned route rows to the authority;
        the worker discards its local return value in either case.
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
