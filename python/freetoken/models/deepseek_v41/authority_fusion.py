"""Opt-in exact authority-side decode fusion for DeepSeek-V4.1.

The production selector lives outside the existing decode implementation, so
flag-off and ineligible batches call the qualified path literally.  This module
contains the complete Package-A path and owns all forward-scoped state.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


AUTHORITY_FUSION_ENV = "FREETOKEN_DSV41_AUTHORITY_FUSION"
PACKAGE_A_LEVERS = frozenset(
    {"R1", "R2", "R3", "R4", "R8", "R10", "R11", "R12-prime", "R13", "P1"}
)
FALLBACK_LAYERS = frozenset({0, 1, 14})
SUPPRESSED_PRODUCERS = frozenset({0, 13, 39})
PUBLICATION_LAYERS = frozenset({2, 8, 14, 20, 24, 28, 32, 36})


def _parse_flag() -> bool:
    value = os.environ.get(AUTHORITY_FUSION_ENV)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{AUTHORITY_FUSION_ENV} must be '0' or '1', got {value!r}")


_ENABLED = _parse_flag()


@dataclass(frozen=True, slots=True)
class AuthorityFusionPolicy:
    """Test-only lever selection; production always uses the complete package."""

    levers: frozenset[str] = PACKAGE_A_LEVERS

    def __post_init__(self) -> None:
        unknown = self.levers - PACKAGE_A_LEVERS
        if unknown:
            raise ValueError(f"unknown Package-A levers: {sorted(unknown)}")

    def has(self, lever: str) -> bool:
        return lever in self.levers


FULL_POLICY = AuthorityFusionPolicy()


def policy_without(*levers: str) -> AuthorityFusionPolicy:
    """Return a G3/G4 leave-one-out policy without adding an environment knob."""

    unknown = set(levers) - PACKAGE_A_LEVERS
    if unknown:
        raise ValueError(f"unknown Package-A levers: {sorted(unknown)}")
    return AuthorityFusionPolicy(PACKAGE_A_LEVERS - set(levers))


def authority_fusion_enabled() -> bool:
    return _ENABLED


def authority_fusion_eligible(batch, execution) -> bool:
    """Frozen runtime selector for an authority M=1/M=2 ordinary decode."""

    return bool(
        _ENABLED
        and execution.uses_authority_transport
        and execution.rank == execution.backbone_rank
        and batch.is_decode
        and not batch.speculative
        and int(batch.padded_size) in (1, 2)
    )


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(getattr(tensor, "_version", 0))
    except RuntimeError as exc:
        # CUDA-graph warmup runs under inference_mode(), whose tensors
        # deliberately have no version counter.  Identity plus the forward
        # epoch still scopes the memo; use a stable sentinel for that mode.
        if "Inference tensors do not track version counter" not in str(exc):
            raise
        return -1


@dataclass(slots=True)
class PreparedMix:
    flat: torch.Tensor
    square: torch.Tensor
    source: torch.Tensor
    source_version: int
    epoch: int

    @classmethod
    def capture(
        cls,
        flat: torch.Tensor,
        square: torch.Tensor,
        source: torch.Tensor,
        epoch: int,
    ) -> "PreparedMix":
        return cls(flat, square, source, _tensor_version(source), epoch)

    def require(self, source: torch.Tensor, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.epoch != epoch or self.source is not source:
            raise RuntimeError("stale or foreign authority HC preparation")
        if self.source_version != _tensor_version(source):
            raise RuntimeError("authority HC preparation source was modified in place")
        if self.flat is self.square or self.flat.data_ptr() == self.square.data_ptr():
            raise RuntimeError("authority HC preparation buffers alias")
        return self.flat, self.square


@dataclass(slots=True)
class AuthorityMoEOutput:
    shared: torch.Tensor
    routes: torch.Tensor


@dataclass(slots=True)
class ForwardMemo:
    """Strong-reference, epoch-scoped R4/R8 memo."""

    epoch: int = 0
    active: bool = False
    _freq_table: torch.Tensor | None = None
    _freq_positions: torch.Tensor | None = None
    _freq_value: torch.Tensor | None = None
    _window_rows: torch.Tensor | None = None
    _topk_rows: torch.Tensor | None = None
    _window_version: int | None = None
    _topk_version: int | None = None
    _combined_rows: torch.Tensor | None = None
    _cmp_counts: torch.Tensor | None = None

    def begin(self) -> int:
        if self.active:
            raise RuntimeError("authority forward memo re-entered")
        self.clear()
        self.epoch += 1
        self.active = True
        return self.epoch

    def clear(self) -> None:
        self.active = False
        self._freq_table = None
        self._freq_positions = None
        self._freq_value = None
        self._window_rows = None
        self._topk_rows = None
        self._window_version = None
        self._topk_version = None
        self._combined_rows = None
        self._cmp_counts = None

    def frequencies(
        self, table: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if not self.active:
            raise RuntimeError("authority forward memo is inactive")
        if self._freq_table is table and self._freq_positions is positions:
            assert self._freq_value is not None
            return self._freq_value
        value = table.index_select(0, positions)
        self._freq_table = table
        self._freq_positions = positions
        self._freq_value = value
        return value

    def attention_rows(
        self, window_rows: torch.Tensor, topk_rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.active:
            raise RuntimeError("authority forward memo is inactive")
        if (
            self._window_rows is window_rows
            and self._topk_rows is topk_rows
            and self._window_version == _tensor_version(window_rows)
            and self._topk_version == _tensor_version(topk_rows)
        ):
            assert self._combined_rows is not None and self._cmp_counts is not None
            return self._combined_rows, self._cmp_counts
        combined = torch.cat([window_rows, topk_rows], -1).int()
        counts = (topk_rows >= 0).sum(-1).to(torch.int32)
        self._window_rows = window_rows
        self._topk_rows = topk_rows
        self._window_version = _tensor_version(window_rows)
        self._topk_version = _tensor_version(topk_rows)
        self._combined_rows = combined
        self._cmp_counts = counts
        return combined, counts


_FORWARD_MEMO = ForwardMemo()
_KV_STREAM: torch.cuda.Stream | None = None
_WARMED = False


def _derived_layer_sets(args) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    fallback = frozenset({0, *map(int, args.engram_layer_ids)})
    suppressed = frozenset(
        layer
        for layer in range(int(args.n_layers))
        if layer == int(args.n_layers) - 1 or layer + 1 in args.engram_layer_ids
    )
    publications = frozenset(map(int, args.index_source_layers))
    return fallback, suppressed, publications


def validate_authority_geometry(transformer) -> None:
    """Fail closed unless the loaded model is the exact frozen Package-A geometry."""

    args = transformer.args
    expected = {
        "n_layers": 40,
        "dim": 5120,
        "hc_mult": 4,
        "n_routed_experts": 384,
        "n_activated_experts": 6,
        "score_func": "sqrtsoftplus",
        "gate_temp": 1.0,
        "norm_topk_prob": True,
        "route_scale": 1.5,
        "norm_eps": 1e-20,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "engram_layer_ids": (1, 14),
        "kv_source_layers": (2, 8, 14, 20),
        "dspark_enabled": False,
    }
    mismatches = [
        f"{name}={getattr(args, name)!r} expected {value!r}"
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    fallback, suppressed, publications = _derived_layer_sets(args)
    if fallback != FALLBACK_LAYERS:
        mismatches.append(f"fallback layers {sorted(fallback)}")
    if suppressed != SUPPRESSED_PRODUCERS:
        mismatches.append(f"suppressed producers {sorted(suppressed)}")
    if publications != PUBLICATION_LAYERS:
        mismatches.append(f"publication layers {sorted(publications)}")
    from .execution import get_execution_plan

    execution = get_execution_plan()
    if not execution.uses_authority_transport or execution.dense_parallel:
        mismatches.append("execution is not authority EP")
    if execution.rank != execution.backbone_rank:
        mismatches.append("enabled warmup is not on the authority rank")
    if any(not getattr(block.ffn, "fused_decode_dispatch", False) for block in transformer.layers):
        mismatches.append("fused decode dispatch is not enabled on every layer")
    for block in transformer.layers:
        experts = getattr(block.ffn, "experts", None)
        cache = getattr(experts, "offload_cache", None)
        if cache is not None and cache.is_cpu_layer(block.layer_id):
            mismatches.append(f"layer {block.layer_id} is a CPU-only expert layer")
    if os.getenv("FREETOKEN_DSV41_FUSED_ROUTER", "0") not in ("", "0"):
        mismatches.append("legacy fused router must be off")
    if os.getenv("FREETOKEN_DSV41_HC_DIRECT_PRE", "0") == "1":
        mismatches.append("legacy all-phase HC direct-pre must be off")
    if "TRITON_DEFAULT_FP_FUSION" in os.environ:
        mismatches.append("TRITON_DEFAULT_FP_FUSION must be absent")
    if mismatches:
        raise RuntimeError("Package-A authority geometry refusal: " + "; ".join(mismatches))


def _legacy_prepare(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = hidden.flatten(-2).float()
    square = flat.square()
    return flat, square


def _prepare(hidden: torch.Tensor, *, fused: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if fused:
        from freetoken.kernel.triton.dsv41 import hc_prepare_mix

        return hc_prepare_mix(hidden)
    return _legacy_prepare(hidden)


def _mixes(block, hidden: torch.Tensor, prepared: PreparedMix, sublayer: str, epoch: int, *, fused_rms: bool):
    from .hc import hc_mixes_from_prepared

    flat, square = prepared.require(hidden, epoch)
    return hc_mixes_from_prepared(
        flat,
        square,
        getattr(block, f"hc_{sublayer}_fn"),
        getattr(block, f"hc_{sublayer}_scale"),
        getattr(block, f"hc_{sublayer}_base"),
        hc_mult=block.hc_mult,
        sinkhorn_iters=block.hc_sinkhorn_iters,
        eps=block.hc_eps,
        norm_eps=block.norm_eps,
        fused_rms=fused_rms,
    )


@contextmanager
def _authority_route_passthrough(experts):
    previous = getattr(experts, "_authority_route_passthrough", False)
    if previous:
        raise RuntimeError("authority route passthrough re-entered")
    experts._authority_route_passthrough = True
    try:
        yield
    finally:
        experts._authority_route_passthrough = previous


def _dispatch_wire(hidden: torch.Tensor, topk: int):
    if hidden.dtype != torch.bfloat16 or hidden.numel() % 2:
        raise RuntimeError("Package-A dispatch requires an even BF16 hidden tensor")
    rows = hidden.shape[0]
    hidden_words = hidden.numel() // 2
    route_elements = rows * topk
    wire = torch.empty(
        hidden_words + 2 * route_elements,
        dtype=torch.int32,
        device=hidden.device,
    )
    wire_hidden = wire[:hidden_words].view(torch.bfloat16).view_as(hidden)
    wire_weights = wire[
        hidden_words : hidden_words + route_elements
    ].view(torch.float32).view(rows, topk)
    wire_ids = wire[hidden_words + route_elements :].view(rows, topk)
    if wire_ids.data_ptr() % (16 if rows == 2 else 8):
        raise RuntimeError("Package-A route-id wire alignment drift")
    return wire, wire_hidden, wire_weights, wire_ids


def _authority_gate(gate, hidden, wire_weights, wire_ids, policy):
    from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32

    logits = bf16_linear_fp32(hidden, gate.weight)
    if policy.has("R12-prime"):
        from freetoken.kernel.triton.dsv41 import router_scores_exact

        scores, biased = router_scores_exact(
            logits, gate.bias, inv_temperature=1.0 / float(gate.gate_temp)
        )
    else:
        scores = logits / gate.gate_temp
        scores = F.softplus(scores).sqrt()
        biased = scores + gate.bias
    indices = biased.topk(gate.topk, dim=-1).indices
    gathered = scores.gather(-1, indices)
    if gate.norm_topk_prob and gate.topk > 1:
        denominator = gathered.sum(-1, keepdim=True)
    else:
        denominator = torch.ones(
            (gathered.shape[0], 1), dtype=torch.float32, device=gathered.device
        )
    if policy.has("R13"):
        from freetoken.kernel.triton.dsv41 import router_wire_tail

        router_wire_tail(
            gathered,
            denominator,
            indices,
            wire_weights,
            wire_ids,
            eps=1e-20 if gate.norm_topk_prob and gate.topk > 1 else 0.0,
            route_scale=float(gate.route_scale),
        )
    else:
        weights = gathered
        if gate.norm_topk_prob and gate.topk > 1:
            weights = weights / (denominator + 1e-20)
        wire_weights.copy_(weights * gate.route_scale)
        wire_ids.copy_(indices)


def authority_moe_forward(state, x: torch.Tensor, policy: AuthorityFusionPolicy) -> AuthorityMoEOutput:
    """Mirror ``MoEState.forward`` while retaining all-reduced route slots."""

    from freetoken.moe import prefill_trace

    from .moe import (
        _broadcast_prefilled_decode_dispatch,
        _prepare_partitioned_routes,
    )
    from .profile import profile_range

    if state.experts is None or state.execution is None:
        raise RuntimeError("V4.1 routed experts have not been attached")
    execution = state.execution
    if not execution.uses_authority_transport or execution.rank != execution.backbone_rank:
        raise RuntimeError("Package-A MoE path is authority-only")
    hidden = x.view(-1, state.dim)
    wire, wire_hidden, wire_weights, wire_ids = _dispatch_wire(
        hidden, state.experts.top_k
    )
    wire_hidden.copy_(hidden)
    prefill_trace.mark("moe.hidden_bcast.end")
    assert state.gate is not None
    with profile_range("DSV41/MoE/Router"):
        _authority_gate(state.gate, hidden, wire_weights, wire_ids, policy)
    prefill_trace.mark("moe.router.end")
    with profile_range("DSV41/EP/FusedDecodeDispatch"):
        hidden, weights, ids = _broadcast_prefilled_decode_dispatch(
            state._comm,
            execution,
            wire,
            tuple(hidden.shape),
            state.experts.top_k,
        )
    prefill_trace.mark("moe.route_bcast.end")
    global_ids = ids
    ownership = getattr(state, "decode_partition", state.partition)
    fused_cache_safe = bool(getattr(state, "fused_route_prep", False))
    weights, ids = _prepare_partitioned_routes(
        weights,
        ids,
        ownership,
        fused_cache_safe=fused_cache_safe,
        storage=getattr(state, "storage", None),
    )
    prefill_trace.mark("moe.localize.end")
    packed_peers = getattr(state.experts, "packed_prefill_peer_ranks", ())
    with profile_range("DSV41/MoE/PrepareRouted"):
        if len(packed_peers) > 1:
            state.experts.prepare_packed_prefill_receive(
                weights, hidden.dtype, global_topk_ids=global_ids
            )
        else:
            state.experts.prepare_packed_prefill_receive(weights, hidden.dtype)
    prefill_trace.mark("moe.prep_recv.end")
    routed_weights = weights.float().contiguous()
    routed_ids = ids.to(torch.int32).contiguous()
    refill_plan = None
    if getattr(state, "decode_refill_overlap", False):
        with profile_range("DSV41/MoE/DecodeRefillFork"):
            if fused_cache_safe:
                refill_plan = state.experts.begin_routed_decode_cache_safe(
                    routed_weights, routed_ids
                )
            else:
                refill_plan = state.experts.begin_routed_decode(
                    routed_weights, routed_ids
                )
    with profile_range("DSV41/MoE/SharedExpert"):
        assert state.shared_experts is not None
        shared = state.shared_experts(hidden)
    prefill_trace.mark("moe.shared.end")
    with _authority_route_passthrough(state.experts), profile_range(
        "DSV41/MoE/RoutedExpert"
    ):
        if refill_plan is None:
            if fused_cache_safe:
                routes = state.experts.routed_decode_cache_safe(
                    hidden, routed_weights, routed_ids
                )
            else:
                routes = state.experts.routed_forward(
                    hidden, routed_weights, routed_ids
                )
        else:
            routes = state.experts.finish_routed_decode(
                hidden, routed_weights, refill_plan
            )
    if routes.ndim != 3 or routes.shape[0] != hidden.shape[0]:
        raise RuntimeError(f"authority route passthrough returned {tuple(routes.shape)}")
    return AuthorityMoEOutput(shared.view(-1, state.dim), routes)


def authority_attention_decode(
    attention,
    x: torch.Tensor,
    positions: torch.Tensor,
    rows: torch.Tensor,
    cmp_stage_cap: int,
    window_ctx,
    memo: ForwardMemo,
    policy: AuthorityFusionPolicy,
) -> torch.Tensor:
    """Mirror ``Attention.decode_step`` with R4/R8 memo and P1 side stream."""

    from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
    from freetoken.models.deepseek_v4.ops import apply_rotary_emb_decode

    assert attention.freqs_cis is not None
    window_slots, _previous, window_rows = window_ctx
    freqs = (
        memo.frequencies(attention.freqs_cis, positions)
        if policy.has("R4")
        else attention.freqs_cis.index_select(0, positions)
    )
    launched = False
    kv_keepalive = None
    current = torch.cuda.current_stream(x.device)
    try:
        if policy.has("P1"):
            if _KV_STREAM is None:
                raise RuntimeError("Package-A KV stream was not warmed")
            _KV_STREAM.wait_stream(current)
            launched = True
            with torch.cuda.stream(_KV_STREAM):
                kv_keepalive = attention.kv_norm(attention.wkv(x))
                apply_rotary_emb_decode(
                    kv_keepalive[..., -attention.rope_head_dim :], freqs
                )
                act_quant_fp8_inplace(kv_keepalive, attention.fp8_block_size)
                attention.attn.store_window(
                    kv_keepalive.view(x.shape[0], attention.head_dim),
                    attention.layer_id,
                    window_slots,
                )
        q_lora = attention.q_norm(attention.wq_a(x))
        q = attention.wq_b(q_lora).unflatten(
            -1, (attention.n_heads, attention.head_dim)
        )
        apply_rotary_emb_decode(q[..., -attention.rope_head_dim :], freqs)
        if not policy.has("P1"):
            kv_keepalive = attention.kv_norm(attention.wkv(x))
            apply_rotary_emb_decode(
                kv_keepalive[..., -attention.rope_head_dim :], freqs
            )
            act_quant_fp8_inplace(kv_keepalive, attention.fp8_block_size)
            attention.attn.store_window(
                kv_keepalive.view(x.shape[0], attention.head_dim),
                attention.layer_id,
                window_slots,
            )

        ratio = attention.plan.compress_ratio
        cmp_counts = None
        if attention.plan.owns_kv:
            attention._publish_source_decode(x, positions, window_slots, rows, freqs)
        if ratio:
            n_stage = (cmp_stage_cap + 1) // ratio
            compressed_rows = attention._index_decode(
                x, q_lora, positions, rows, n_stage, freqs
            )
            if policy.has("R8"):
                topk, cmp_counts = memo.attention_rows(window_rows, compressed_rows)
            else:
                topk = torch.cat([window_rows, compressed_rows], -1).int()
                cmp_counts = (compressed_rows >= 0).sum(-1).to(torch.int32)
        else:
            topk = window_rows.int()
    finally:
        if launched:
            current.wait_stream(_KV_STREAM)
    output = attention.attn.attend(
        q,
        attention.layer_id,
        topk,
        attention.window_size,
        attention.attn_sink,
        attention.softmax_scale,
        cmp_counts=cmp_counts,
        has_compression=bool(ratio),
    )
    apply_rotary_emb_decode(
        output[..., -attention.rope_head_dim :], freqs, inverse=True
    )
    return attention._project_output(output)


def _post_and_prepare(
    activation,
    residual,
    post,
    comb,
    *,
    prepare,
    fused,
):
    if fused:
        from freetoken.kernel.triton.dsv41 import hc_post_mixprep

        return hc_post_mixprep(
            activation, residual, post, comb, prepare=prepare
        )
    from .hc import hc_post

    hidden = hc_post(activation, residual, post, comb)
    if prepare:
        flat, square = _legacy_prepare(hidden)
    else:
        flat = square = None
    return hidden, flat, square


def _ffn_post_and_prepare(
    moe: AuthorityMoEOutput,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    prepare: bool,
    policy: AuthorityFusionPolicy,
):
    """Compose independently selectable R1 and R2 without changing arithmetic."""

    from .hc import hc_post

    if policy.has("R1"):
        from freetoken.kernel.triton.dsv41 import hc_post_mixprep

        hidden, flat, square = hc_post_mixprep(
            moe.shared,
            residual,
            post,
            comb,
            routes=moe.routes,
            prepare=prepare and policy.has("R2"),
        )
        if prepare and not policy.has("R2"):
            flat, square = _legacy_prepare(hidden)
        return hidden, flat, square

    from freetoken.kernel import moe_sum_reduce_triton

    rows, _, dim = moe.routes.shape
    reduced = torch.empty(
        (rows, dim), dtype=moe.routes.dtype, device=moe.routes.device
    )
    moe_sum_reduce_triton(moe.routes, reduced)
    activation = moe.shared + reduced.to(moe.shared.dtype)
    if policy.has("R2"):
        from freetoken.kernel.triton.dsv41 import hc_post_mixprep

        return hc_post_mixprep(
            activation, residual, post, comb, prepare=prepare
        )
    hidden = hc_post(activation, residual, post, comb)
    if prepare:
        flat, square = _legacy_prepare(hidden)
    else:
        flat = square = None
    return hidden, flat, square


def authority_block_decode(
    block,
    hidden: torch.Tensor,
    incoming_pre: torch.Tensor,
    prepared: PreparedMix | None,
    *,
    epoch: int,
    positions: torch.Tensor,
    rows: torch.Tensor,
    cmp_stage_cap: int,
    window_ctx,
    memo: ForwardMemo,
    policy: AuthorityFusionPolicy,
    engram_rows: torch.Tensor | None,
    token_mask: torch.Tensor | None,
):
    from .hc import hc_pre, hc_pre_decode_direct

    layer = int(block.layer_id)
    hidden = block._engram(hidden, engram_rows, token_mask)
    _require_preparation_schedule(layer, prepared)
    if prepared is None:
        flat, square = _prepare(hidden, fused=policy.has("R2"))
        prepared = PreparedMix.capture(flat, square, hidden, epoch)
    attn_pre, attn_post, attn_comb = _mixes(
        block, hidden, prepared, "attn", epoch, fused_rms=policy.has("R3")
    )
    residual = hidden
    combined = (
        hc_pre_decode_direct(hidden, incoming_pre)
        if policy.has("R2")
        else hc_pre(hidden, incoming_pre)
    )
    block_input = block.attn_norm(combined)
    block_output = authority_attention_decode(
        block.attn,
        block_input,
        positions,
        rows,
        cmp_stage_cap,
        window_ctx,
        memo,
        policy,
    )
    hidden, flat, square = _post_and_prepare(
        block_output,
        residual,
        attn_post,
        attn_comb,
        prepare=True,
        fused=policy.has("R10"),
    )
    assert flat is not None and square is not None
    ffn_prepared = PreparedMix.capture(flat, square, hidden, epoch)
    ffn_pre, ffn_post, ffn_comb = _mixes(
        block, hidden, ffn_prepared, "ffn", epoch, fused_rms=policy.has("R11")
    )
    residual = hidden
    combined = (
        hc_pre_decode_direct(hidden, attn_pre)
        if policy.has("R10")
        else hc_pre(hidden, attn_pre)
    )
    block_input = block.ffn_norm(combined)
    moe = authority_moe_forward(block.ffn, block_input, policy)
    emit = layer not in SUPPRESSED_PRODUCERS
    hidden, flat, square = _ffn_post_and_prepare(
        moe,
        residual,
        ffn_post,
        ffn_comb,
        prepare=emit,
        policy=policy,
    )
    next_prepared = (
        PreparedMix.capture(flat, square, hidden, epoch)
        if emit and flat is not None and square is not None
        else None
    )
    return hidden, ffn_pre, next_prepared


def _require_preparation_schedule(layer: int, prepared: PreparedMix | None) -> None:
    expects_fallback = int(layer) in FALLBACK_LAYERS
    if (prepared is None) != expects_fallback:
        raise RuntimeError(
            f"Package-A preparation schedule drift at layer {layer}: "
            f"prepared={prepared is not None}"
        )


def decode_authority_fused(
    transformer,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    cmp_stage_cap: int,
    *,
    engram_rows: dict[int, torch.Tensor] | None = None,
    token_mask: torch.Tensor | None = None,
    capture_dspark: bool = False,
    policy: AuthorityFusionPolicy = FULL_POLICY,
) -> torch.Tensor:
    """Full Package-A authority decode; the caller has already checked eligibility."""

    from . import model as model_module
    from .hc import hc_pre, identity_pre_mix
    from .head import project_logits

    if capture_dspark:
        raise RuntimeError("Package-A refuses DSpark/MTP")
    if not _WARMED or _KV_STREAM is None:
        raise RuntimeError("Package-A decode entered before enabled warmup")
    memo = _FORWARD_MEMO
    epoch = memo.begin()
    try:
        model_module.shared_attention.reset()
        batch_size = input_ids.shape[0]
        if batch_size not in (1, 2):
            raise RuntimeError(f"Package-A requires M=1/2, got {batch_size}")
        rows = torch.arange(batch_size, device=input_ids.device)
        from freetoken.core import get_global_ctx

        window_ctx = get_global_ctx().batch.attn_metadata.window_ctx(
            positions, rows
        )
        hidden = transformer.embed(input_ids).unsqueeze(2).repeat(
            1, 1, transformer.args.hc_mult, 1
        )
        incoming = identity_pre_mix(hidden, transformer.args.hc_mult)
        prepared = None
        for layer, block in enumerate(transformer.layers):
            hidden, incoming, prepared = authority_block_decode(
                block,
                hidden,
                incoming,
                prepared,
                epoch=epoch,
                positions=positions,
                rows=rows,
                cmp_stage_cap=cmp_stage_cap,
                window_ctx=window_ctx,
                memo=memo,
                policy=policy,
                engram_rows=(engram_rows or {}).get(layer),
                token_mask=token_mask,
            )
        # The final HC pre-combine is outside frozen R2/R10 Package-A scope.
        hidden = transformer.norm(hc_pre(hidden, incoming))
        transformer._dspark_hidden = None
        return project_logits(hidden[:, -1], transformer.head)
    finally:
        memo.clear()


def warmup_authority_fusion(transformer, device: torch.device) -> None:
    """Create the single side stream and precompile all Package-A variants."""

    global _KV_STREAM, _WARMED
    if not _ENABLED:
        return
    if _WARMED:
        return
    if device.type != "cuda":
        raise RuntimeError("Package-A enabled warmup requires CUDA")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Package-A warmup must precede CUDA graph capture")
    validate_authority_geometry(transformer)
    _KV_STREAM = torch.cuda.Stream(device=device)
    from freetoken.kernel.triton.dsv41 import (
        hc_post_mixprep,
        hc_prepare_mix,
        hc_sinkhorn_rmsprep,
        router_scores_exact,
        router_wire_tail,
    )
    from .hc import hc_pre_decode_direct
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
    from freetoken.kernel.triton.dsv4.hc import hc_post_combine
    from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn

    block = transformer.layers[0]
    for rows in (1, 2, 16):
        hidden = torch.zeros(
            (rows, 1, transformer.args.hc_mult, transformer.args.dim),
            dtype=torch.bfloat16,
            device=device,
        )
        flat, square = hc_prepare_mix(hidden)
        incoming = torch.zeros(
            (rows, 1, transformer.args.hc_mult),
            dtype=torch.float32,
            device=device,
        )
        hc_pre_decode_direct(hidden, incoming)
        mixes = F.linear(flat, block.hc_attn_fn.float())
        means = square.mean(-1, keepdim=True)
        hc_sinkhorn_rmsprep(
            mixes,
            means,
            block.hc_attn_scale,
            block.hc_attn_base,
            hc_mult=transformer.args.hc_mult,
            sinkhorn_iters=transformer.args.hc_sinkhorn_iters,
            eps=transformer.args.hc_eps,
            norm_eps=transformer.args.norm_eps,
        )
        normalized = mixes * torch.rsqrt(means + transformer.args.norm_eps)
        hc_split_sinkhorn(
            normalized.reshape(-1, normalized.shape[-1]),
            block.hc_attn_scale,
            block.hc_attn_base,
            transformer.args.hc_mult,
            transformer.args.hc_sinkhorn_iters,
            transformer.args.hc_eps,
        )
        residual = hidden
        activation = torch.zeros(
            (rows, transformer.args.dim), dtype=torch.bfloat16, device=device
        )
        post = torch.zeros(
            (rows, transformer.args.hc_mult), dtype=torch.float32, device=device
        )
        comb = torch.zeros(
            (rows, transformer.args.hc_mult, transformer.args.hc_mult),
            dtype=torch.float32,
            device=device,
        )
        routes = torch.zeros(
            (rows, transformer.args.n_activated_experts, transformer.args.dim),
            dtype=torch.bfloat16,
            device=device,
        )
        hc_post_mixprep(activation, residual, post, comb, prepare=True)
        hc_post_mixprep(activation, residual, post, comb, prepare=False)
        hc_post_mixprep(
            activation, residual, post, comb, routes=routes, prepare=True
        )
        hc_post_mixprep(
            activation, residual, post, comb, routes=routes, prepare=False
        )
        reduced = torch.empty_like(activation)
        moe_sum_reduce_triton(routes, reduced)
        hc_post_combine(
            activation,
            residual.reshape(rows, transformer.args.hc_mult, transformer.args.dim),
            post.reshape(rows, transformer.args.hc_mult),
            comb.reshape(rows, transformer.args.hc_mult, transformer.args.hc_mult),
        )
        logits = torch.zeros(
            (rows, transformer.args.n_routed_experts),
            dtype=torch.float32,
            device=device,
        )
        scores, _ = router_scores_exact(
            logits, block.ffn.gate.bias, inv_temperature=1.0
        )
        bf16_linear_fp32(activation, block.ffn.gate.weight)
        indices = torch.arange(
            transformer.args.n_activated_experts, device=device
        ).repeat(rows, 1)
        gathered = scores.gather(-1, indices)
        denominator = gathered.sum(-1, keepdim=True)
        _, _, wire_weights, wire_ids = _dispatch_wire(
            activation, transformer.args.n_activated_experts
        )
        router_wire_tail(
            gathered,
            denominator,
            indices,
            wire_weights,
            wire_ids,
            eps=1e-20,
            route_scale=transformer.args.route_scale,
        )
    torch.cuda.synchronize(device)
    _WARMED = True


__all__ = [
    "AUTHORITY_FUSION_ENV",
    "AuthorityFusionPolicy",
    "AuthorityMoEOutput",
    "FALLBACK_LAYERS",
    "ForwardMemo",
    "FULL_POLICY",
    "PACKAGE_A_LEVERS",
    "PUBLICATION_LAYERS",
    "PreparedMix",
    "SUPPRESSED_PRODUCERS",
    "authority_fusion_eligible",
    "authority_fusion_enabled",
    "decode_authority_fused",
    "policy_without",
    "validate_authority_geometry",
    "warmup_authority_fusion",
]
