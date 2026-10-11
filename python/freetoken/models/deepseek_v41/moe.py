"""V4.1 sqrt-softplus router, shared expert, and Qwen-style EP transport."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator
from freetoken.layers import ExpertParallelOffloadMoELayer
from freetoken.moe import prefill_trace

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


def _fused_decode_dispatch_enabled() -> bool:
    return os.getenv(
        "FREETOKEN_DSV41_FUSED_DECODE_DISPATCH", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _background_admission_enabled(execution) -> bool:
    # Authority/backbone, other model families, prefill and multi-row paths keep
    # their qualified implementations. This experimental policy is off by default.
    return execution.is_expert_worker and os.getenv(
        "FREETOKEN_DSV41_BACKGROUND_ADMISSION", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _background_admission_max_misses() -> int:
    value = int(os.getenv("FREETOKEN_DSV41_BACKGROUND_MAX_MISSES", "1"))
    if not 1 <= value <= 6:
        raise ValueError("V4.1 background CPU miss limit must be between 1 and 6")
    return value


_PREFILL_EARLY_REFILL_ENV = "FREETOKEN_DSV41_PREFILL_EARLY_REFILL"
# freetoken.scheduler.tail_pairing.TAIL_PAIRING_ENV; strictly parsed and validated at startup by
# the scheduler and engine. Frozen here so the default-off path never inspects the batch.
_TAIL_PAIRING_ENABLED = os.getenv("FREETOKEN_DSV41_TAIL_PAIRING", "0") == "1"


def prefill_early_refill_enabled() -> bool:
    """Strict parse at model creation: unset/``0`` is off and ``1`` is on."""
    value = os.environ.get(_PREFILL_EARLY_REFILL_ENV)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(
        f"{_PREFILL_EARLY_REFILL_ENV} must be '0' or '1', got {value!r}"
    )


def _broadcast_decode_dispatch(
    communicator,
    execution,
    hidden: torch.Tensor,
    top_k: int,
    weights: torch.Tensor | None = None,
    ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast BF16 hidden state and FP32/int32 routes as one exact payload."""

    root = execution.backbone_rank
    if root is None:
        raise RuntimeError("fused decode dispatch requires an authority root")
    if hidden.dtype != torch.bfloat16 or hidden.numel() % 2:
        raise RuntimeError(
            "fused decode dispatch requires an even BF16 hidden-state size"
        )
    route_shape = (hidden.shape[0], top_k)
    route_elements = hidden.shape[0] * top_k
    hidden_words = hidden.numel() // 2
    wire = torch.empty(
        hidden_words + 2 * route_elements,
        dtype=torch.int32,
        device=hidden.device,
    )
    wire_hidden = wire[:hidden_words].view(torch.bfloat16).view_as(hidden)
    wire_weights = wire[
        hidden_words : hidden_words + route_elements
    ].view(torch.float32).view(route_shape)
    wire_ids = wire[hidden_words + route_elements :].view(route_shape)
    if execution.rank == root:
        if weights is None or ids is None:
            raise RuntimeError("authority must provide decode routing tensors")
        if weights.shape != route_shape or ids.shape != route_shape:
            raise RuntimeError(
                "decode route shape does not match hidden state: "
                f"weights={tuple(weights.shape)} ids={tuple(ids.shape)} "
                f"expected={route_shape}"
            )
        wire_hidden.copy_(hidden)
        wire_weights.copy_(weights)
        wire_ids.copy_(ids)
    wire = communicator.broadcast(wire, root)
    return wire_hidden, wire_weights, wire_ids


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

    def __init__(self, args: DeepseekV41Args, *, enable_vision: bool = False) -> None:
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
        self.bias_vl = (
            nn.Parameter(torch.empty_like(self.bias), requires_grad=False)
            if enable_vision
            else None
        )

    def forward(
        self, x: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias = self.bias
        if image_mask is not None:
            if self.bias_vl is None:
                raise RuntimeError("image routing requested without vision router bias")
            bias = torch.where(image_mask.reshape(-1, 1), self.bias_vl, bias)
        if x.is_cuda:
            from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32

            scores = bf16_linear_fp32(x, self.weight)
            fused = os.getenv(
                "FREETOKEN_DSV41_FUSED_ROUTER", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            # The fused kernel consumes one shared bias vector. A multimodal
            # prefill selects bias_vl per row, so keep that correctness path in
            # PyTorch until the router kernel grows a row-strided bias input.
            if fused and image_mask is None:
                from freetoken.kernel.triton.dsv41.router import (
                    fused_sqrtsoftplus_topk,
                )

                if self.score_func != "sqrtsoftplus":
                    raise ValueError(
                        f"unsupported V4.1 route score {self.score_func}"
                    )
                return fused_sqrtsoftplus_topk(
                    scores,
                    bias,
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
        indices = (scores + bias).topk(self.topk, dim=-1).indices
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
        # V4.1 uses one activation scale per 32 values.  Keep this model-local;
        # the shared DS-FP4 kernels retain V4's 128-value default.
        self.dsfp4_activation_block_size = int(args.weight_block_size[1])
        self.skip_empty_cache = os.getenv("FREETOKEN_DSV41_SKIP_EMPTY_CACHE", "0") == "1"
        self.background_admission_enabled = _background_admission_enabled(get_execution_plan())
        self.background_admission_max_misses = (
            _background_admission_max_misses() if self.background_admission_enabled else 1
        )

    def _decode_hybrid(self, cache, hidden_states, topk_weights, topk_ids):
        if self.background_admission_enabled and hidden_states.shape[0] == 1:
            from freetoken.moe.background_admission import BackgroundAdmission

            state = getattr(cache, "_dsv41_background_admission", None)
            if state is None:
                state = BackgroundAdmission(cache, max_cpu_misses=self.background_admission_max_misses)
                cache._dsv41_background_admission = state
            return state.decode(self, hidden_states, topk_weights, topk_ids)
        return super()._decode_hybrid(cache, hidden_states, topk_weights, topk_ids)

    def _maybe_all_reduce(self, routes: torch.Tensor) -> torch.Tensor:
        with profile_range("DSV41/EP/RouteAllReduce"):
            return super()._maybe_all_reduce(routes)

    # Bound by PrefillEarlyRefill when FREETOKEN_DSV41_PREFILL_EARLY_REFILL=1.
    prefill_early_refill = None

    def uses_prefill_double_buffer(self, num_tokens: int) -> bool:
        """Whether this row count takes the real overlap double-buffer path."""
        cache = self.offload_cache
        if cache is None or not cache.prefill_overlap or not cache.prefill_bank_buffers:
            return False
        if self._prefill_route_tile_tokens(num_tokens):
            return True
        return not (
            cache.sparse_prefill_max_tokens > 0
            and num_tokens <= cache.sparse_prefill_max_tokens
            and cache.quant_format in ("fp8_block", "ds_fp4")
            and not cache.is_unpinned_layer(self.layer_id)
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        routed = super().routed_forward(hidden_states, topk_weights, topk_ids)
        refill = self.prefill_early_refill
        if refill is not None:
            refill.layer_released(self, hidden_states.shape[0])
        return routed

    # -- opt-in prefill trace (FREETOKEN_DSV41_PREFILL_TRACE); each override
    # defers to the base class unchanged unless a traced prefill chunk is active.

    def _wait_prefill_overlap(self, cache):
        if (
            _TAIL_PAIRING_ENABLED
            and getattr(get_global_ctx().batch, "tail_pair_role", None) == "tail"
        ):
            # Paired tail chunk: the head chunk streamed this layer moments ago in the same
            # forward and the slot stays resident until the head's next layer issues L+2.
            return cache.reuse_prefill_layer(self.layer_id)
        # Keep the default-OFF path identical: in particular, do not even require an active
        # global batch when the feature is disabled (standalone kernel tools call this method).
        cone = None
        if os.getenv("FREETOKEN_DSV41_DEPENDENCY_CONE", "0") == "1":
            cone = getattr(get_global_ctx().batch, "dsv41_cone", None)
        if cone is not None:
            # Encoder-only cone chunks stop before the source layer.  The generic overlap
            # choreography always queues L+1, which would stream the unused source-layer expert
            # bank after layer source-1.  Bound both traced and ordinary prefetch to the layers
            # this rank will actually execute; decoder chunks retain the normal full-depth path.
            limit = int(cone.executed_layers)
            if not 0 <= self.layer_id < limit:
                raise RuntimeError(
                    f"dependency-cone prefill reached MoE layer {self.layer_id} "
                    f"outside its {limit}-layer segment"
                )
            if self.layer_id == 0:
                cache.begin_prefill()
            if prefill_trace.active():
                self._traced_prefetch(cache, self.layer_id)
                if self.layer_id + 1 < limit:
                    self._traced_prefetch(cache, self.layer_id + 1)
            else:
                cache.prefetch_prefill_layer(self.layer_id)
                if self.layer_id + 1 < limit:
                    cache.prefetch_prefill_layer(self.layer_id + 1)
            return cache.wait_prefill_layer(self.layer_id)
        refill = self.prefill_early_refill
        began_early = (
            self.layer_id == 0 and refill is not None and refill.began(cache)
        )
        if not prefill_trace.active():
            if began_early:
                # Layer 0 was issued before embedding/attention. Queue layer 1
                # only now, after attention 0, so a host-blocking invalidation
                # cannot serialize that attention behind copy 0.
                cache.prefetch_prefill_layer(0)
                cache.prefetch_prefill_layer(1)
                return cache.wait_prefill_layer(0)
            return super()._wait_prefill_overlap(cache)
        # Mirrors OffloadMoELayer._wait_prefill_overlap, split by trace marks.
        prefill_trace.mark("moe.prefetch.begin")
        if self.layer_id == 0 and not began_early:
            cache.begin_prefill()
        if self.layer_id == 0:
            prefill_trace.note("bank_feat_bytes", [
                int(buffer[0, 0].numel() * buffer.element_size())
                for buffer in cache.prefill_bank_buffers
            ])
            prefill_trace.note("prefill_expert_count", int(cache.prefill_expert_count))
        self._traced_prefetch(cache, self.layer_id)
        self._traced_prefetch(cache, self.layer_id + 1)
        prefill_trace.mark("moe.prefetch.end")
        views = cache.wait_prefill_layer(self.layer_id)
        prefill_trace.mark("moe.expert_wait.end")
        return views

    @staticmethod
    def _traced_prefetch(cache, layer: int) -> None:
        stream = cache.prefill_copy_stream
        buffer_id = layer % 2
        if (
            stream is None
            or layer >= cache.num_layers
            or cache._prefill_buffer_layer[buffer_id] == layer
        ):
            cache.prefetch_prefill_layer(layer)  # already issued, or nothing to copy
            return
        if cache._prefill_buffer_has_release_event[buffer_id]:
            # The prefetch begins with this same wait; repeating it first lets
            # the begin mark observe when the copy can actually start.
            stream.wait_event(cache.prefill_release_events[buffer_id])
        prefill_trace.mark("h2d.begin", layer=layer, stream=stream, tag="copy")
        hits = cache.prefill_hit_rows
        cache.prefetch_prefill_layer(layer)
        prefill_trace.mark("h2d.end", layer=layer, stream=stream, tag="copy")
        prefill_trace.note("h2d.hit_rows", cache.prefill_hit_rows - hits, layer=layer)

    def _expert_gemm(self, cache, hidden_states, *args, is_prefill, **kwargs):
        if not (is_prefill and prefill_trace.active()):
            return super()._expert_gemm(
                cache, hidden_states, *args, is_prefill=is_prefill, **kwargs
            )
        prefill_trace.mark("moe.gemm.begin")
        out = super()._expert_gemm(
            cache, hidden_states, *args, is_prefill=is_prefill, **kwargs
        )
        prefill_trace.mark("moe.gemm.end")
        return out

    def prepare_packed_prefill_receive(self, *args, **kwargs) -> None:
        if not prefill_trace.active():
            return super().prepare_packed_prefill_receive(*args, **kwargs)
        prefill_trace.mark("moe.prep.begin")
        super().prepare_packed_prefill_receive(*args, **kwargs)
        pending = getattr(self, "_packed_prefill_pending", None)
        if pending is not None:
            # Completes when every peer's packed routes for this tile arrived.
            prefill_trace.mark("moe.recv.end", stream=pending[1], tag="recv")
        prefill_trace.mark("moe.prep.end")

    def _maybe_combine_packed_prefill_routes(self, routes, local_topk_weights):
        if not prefill_trace.active():
            return super()._maybe_combine_packed_prefill_routes(
                routes, local_topk_weights
            )
        out = super()._maybe_combine_packed_prefill_routes(routes, local_topk_weights)
        # Root: after peer wait, index_copy and FP32 reduction; worker: after send.
        prefill_trace.mark("moe.combine.end")
        return out


class PrefillEarlyRefill:
    """Issue expert banks as early as double-buffer ownership safely permits."""

    def __init__(self) -> None:
        self._layers: tuple[RoutedExperts, ...] = ()
        self._cache = None
        self._tokens = 0

    def bind(self, layers) -> None:
        self._layers = tuple(layers)
        for layer in self._layers:
            layer.prefill_early_refill = self

    def began(self, cache) -> bool:
        return cache is not None and cache is self._cache

    def _eligible_cache(self, num_tokens: int):
        if not self._layers or num_tokens <= 0:
            return None
        if not get_global_ctx().batch.is_moe_prefill:
            return None
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return None
        cache = self._layers[0].offload_cache
        if (
            cache is None
            or cache.num_layers != len(self._layers)
            or any(
                layer.offload_cache is not cache or layer.layer_id != index
                for index, layer in enumerate(self._layers)
            )
        ):
            return None
        if not all(
            layer.uses_prefill_double_buffer(num_tokens) for layer in self._layers
        ):
            return None
        return cache

    def begin(self, num_tokens: int) -> bool:
        self._cache = None
        cache = self._eligible_cache(num_tokens)
        if cache is None:
            return False
        # Publish before the first copy so an abort inside begin is fenced.
        self._cache, self._tokens = cache, int(num_tokens)
        with profile_range("DSV41/MoE/EarlyRefillBegin"):
            cache.begin_prefill()
            if prefill_trace.active():
                self._layers[0]._traced_prefetch(cache, 0)
            else:
                cache.prefetch_prefill_layer(0)
        return True

    def layer_released(self, layer: RoutedExperts, num_tokens: int) -> None:
        cache = self._cache
        if (
            cache is None
            or layer.offload_cache is not cache
            or num_tokens != self._tokens
        ):
            return
        # routed_forward released buffer L % 2 after its last GEMM; the next
        # owner of that buffer is L + 2, so this is its earliest safe issue.
        next_layer = layer.layer_id + 2
        if prefill_trace.active():
            layer._traced_prefetch(cache, next_layer)
        else:
            cache.prefetch_prefill_layer(next_layer)

    def end(self, *, aborted: bool) -> None:
        cache, self._cache = self._cache, None
        if aborted and cache is not None and cache.prefill_copy_stream is not None:
            try:
                # Copies left by an aborted chunk must precede later slot reuse.
                torch.cuda.current_stream(cache.device).wait_stream(
                    cache.prefill_copy_stream
                )
            except Exception:
                # Preserve the original chunk error if CUDA is already sticky.
                pass

    @contextmanager
    def chunk(self, num_tokens: int):
        try:
            yield self.begin(num_tokens)
        except BaseException:
            self.end(aborted=True)
            raise
        self.end(aborted=False)


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
        self._prefill_comm = None
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
            self.experts.packed_prefill_peer_ranks = peers
            self.experts.packed_prefill_peer_ranges = tuple(
                (
                    peer,
                    partition.global_offset,
                    partition.global_stop,
                )
                for peer in peers
                for partition in (
                    self.execution.partition_for_rank(
                        args.n_routed_experts, peer, prefill=True
                    ),
                )
            )
            if len(peers) == 1:
                self.experts.packed_prefill_peer_rank = peers[0]
        self.fused_route_prep = _fused_route_prep_enabled()
        self.fused_decode_dispatch = _fused_decode_dispatch_enabled()

    def attach_prefill_communicator(self, communicator) -> None:
        self._prefill_comm = communicator
        self.experts.prefill_communicator = communicator

    def _phase_broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        root = self.execution.backbone_rank
        assert root is not None
        batch = get_global_ctx().batch
        if not batch.is_moe_prefill or not self.execution.uses_prefill_subgroup:
            return self._comm.broadcast(tensor, root)
        if self._prefill_comm is None and self.execution.participates_in_prefill:
            raise RuntimeError("V4.1 active prefill rank has no subgroup communicator")
        communicator = self._prefill_comm
        if self.execution.rank == root:
            for peer in self.execution.prefill_active_ranks:
                if peer != root:
                    communicator.send(tensor, peer)
            return tensor
        if self.execution.participates_in_prefill:
            return communicator.recv(tensor, root)
        raise RuntimeError("inactive prefill rank attempted a phase broadcast")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.execution.is_expert_worker:
            raise RuntimeError("V4.1 expert workers must call worker_forward")
        shape = x.shape
        hidden = x.view(-1, self.dim)
        fused_dispatch = (
            self.fused_decode_dispatch
            and self.execution.uses_authority_transport
            and not get_global_ctx().batch.is_moe_prefill
        )
        if self.execution.uses_authority_transport and not fused_dispatch:
            hidden = self._phase_broadcast(hidden.contiguous())
        assert self.gate is not None and self.shared_experts is not None
        weights, ids = self.gate(hidden)
        if fused_dispatch:
            with profile_range("DSV41/EP/FusedDecodeDispatch"):
                hidden, weights, ids = _broadcast_decode_dispatch(
                    self._comm,
                    self.execution,
                    hidden,
                    self.experts.top_k,
                    weights,
                    ids,
                )
        elif self.execution.uses_authority_transport:
            weights = self._phase_broadcast(weights.float().contiguous())
            ids = self._phase_broadcast(ids.to(torch.int32).contiguous())
        fused_cache_safe = (
            self.fused_route_prep
            and weights.is_cuda
            and not get_global_ctx().batch.is_moe_prefill
        )
        ownership = (
            self.prefill_partition
            if get_global_ctx().batch.is_moe_prefill
            else self.decode_partition
        )
        global_ids = ids
        weights, ids = _prepare_partitioned_routes(
            weights,
            ids,
            ownership,
            fused_cache_safe=fused_cache_safe,
            storage=self.storage,
        )
        packed_peers = getattr(self.experts, "packed_prefill_peer_ranks", ())
        if not self.execution.tp2_ep2:
            if len(packed_peers) > 1:
                self.experts.prepare_packed_prefill_receive(
                    weights, hidden.dtype, global_topk_ids=global_ids
                )
            else:
                self.experts.prepare_packed_prefill_receive(weights, hidden.dtype)
        shared = self.shared_experts(hidden)
        if self.execution.tp2_ep2:
            if len(packed_peers) > 1:
                self.experts.prepare_packed_prefill_receive(
                    weights, hidden.dtype, global_topk_ids=global_ids
                )
            else:
                self.experts.prepare_packed_prefill_receive(weights, hidden.dtype)
        routed_weights = weights.float().contiguous()
        routed_ids = ids.to(torch.int32).contiguous()
        if fused_cache_safe:
            routed = self.experts.routed_decode_cache_safe(
                hidden, routed_weights, routed_ids
            )
        else:
            routed = self.experts.routed_forward(hidden, routed_weights, routed_ids)
        if self.execution.tp2_ep2 and get_global_ctx().batch.is_moe_prefill:
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
            hidden = torch.empty(hidden_shape, dtype=torch.bfloat16, device=device)
            if self.fused_decode_dispatch and not get_global_ctx().batch.is_moe_prefill:
                hidden, weights, ids = _broadcast_decode_dispatch(
                    self._comm,
                    self.execution,
                    hidden,
                    self.experts.top_k,
                )
            else:
                hidden = self._phase_broadcast(hidden).view(-1, self.dim)
                route_shape = (hidden.shape[0], self.experts.top_k)
                weights = self._phase_broadcast(
                    torch.empty(route_shape, dtype=torch.float32, device=device)
                )
                ids = self._phase_broadcast(
                    torch.empty(route_shape, dtype=torch.int32, device=device)
                )
            prefill_trace.mark("moe.input_recv.end")
            fused_cache_safe = (
                self.fused_route_prep
                and weights.is_cuda
                and not get_global_ctx().batch.is_moe_prefill
            )
            ownership = (
                self.prefill_partition
                if get_global_ctx().batch.is_moe_prefill
                else self.decode_partition
            )
            weights, ids = _prepare_partitioned_routes(
                weights,
                ids,
                ownership,
                fused_cache_safe=fused_cache_safe,
                storage=self.storage,
            )
            prefill_trace.mark("moe.localize.end")
        with profile_range("DSV41/MoE/WorkerRoutedExpert"):
            if fused_cache_safe:
                self.experts.routed_decode_cache_safe(
                    hidden, weights.contiguous(), ids.contiguous()
                )
            else:
                self.experts.routed_forward(
                    hidden, weights.contiguous(), ids.contiguous()
                )


__all__ = ["Gate", "MoE", "PrefillEarlyRefill", "RoutedExperts", "SharedExpert"]
