"""Checkpoint-native DSpark/MTP drafter for DeepSeek-V4.1-Flash.

The V4.1 drafter is deliberately resident on an auxiliary GPU.  Its three
128-expert DS-FP4 blocks occupy about 7.4 GiB and are small enough for the
CMP-170HX, while placing them in the target EP cache would evict hot 384-expert
target rows on the already memory-bound 5090/4080 ranks.

The proposal and verification rules follow FreeToken upstream PRs #69/#71.
V4.1-specific model geometry and weight names follow the checkpoint's official
``inference/model.py`` reference.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import replace
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.models.loader import ShardReader, drop_page_cache
from freetoken.utils import init_logger

from .args import DeepseekV41Args
from .hc import hc_mixes, hc_post, hc_pre, identity_pre_mix
from .layers import Linear, RMSNorm
from .moe import Gate, SharedExpert


logger = init_logger(__name__)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def sampling_probs(
    logits: torch.Tensor, temperature: float, top_p: float, top_k: int = -1
) -> torch.Tensor:
    """The exact post-temperature/top-k/top-p distribution used for p/q tests."""

    if temperature <= 0:
        out = torch.zeros_like(logits, dtype=torch.float32)
        out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return out
    scaled = logits.float() if temperature == 1.0 else logits.float() / temperature
    if top_k and top_k > 0:
        kth = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    if 0 < top_p < 1:
        ordered, indices = probs.sort(dim=-1, descending=True)
        cumulative = ordered.cumsum(dim=-1)
        ordered = ordered.masked_fill(cumulative - ordered > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, indices, ordered)
        probs /= probs.sum(dim=-1, keepdim=True)
    return probs


def accepted_prefix(
    proposed: torch.Tensor, target_argmax: torch.Tensor
) -> tuple[int, int]:
    """Return the greedy accepted prefix length and target bonus token."""

    n = int(proposed.numel())
    for i in range(n):
        if int(proposed[i]) != int(target_argmax[i]):
            return i, int(target_argmax[i])
    return n, int(target_argmax[n]) if target_argmax.numel() > n else -1


def rejection_accept_device(
    proposed: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Distribution-exact speculative rejection sampling on the logits device."""

    n = int(proposed.numel())
    if q.shape[0] != n or p.shape[0] < n + 1:
        raise ValueError(
            f"rejection sampler shape mismatch: proposed={n}, q={q.shape}, p={p.shape}"
        )
    if n:
        rows = torch.arange(n, device=proposed.device)
        token_ids = proposed.long()
        p_token = p[:n][rows, token_ids]
        q_token = q[:n][rows, token_ids]
        uniforms = torch.rand(
            n, device=p.device, dtype=torch.float32, generator=generator
        )
        positions = torch.arange(n, device=p.device, dtype=torch.int64)
        n_acc_device = torch.where(
            p_token <= uniforms * q_token,
            positions,
            torch.full_like(positions, n),
        ).amin()
        rejected_row = n_acc_device.clamp_max(n - 1)
        residual = torch.clamp(p[rejected_row] - q[rejected_row], min=0.0)
        total = residual.sum()
        residual_dist = torch.where(
            total > 0,
            residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
            p[rejected_row],
        )
        dist = torch.where(n_acc_device < n, residual_dist, p[n])
    else:
        n_acc_device = torch.zeros((), dtype=torch.int64, device=p.device)
        dist = p[0]
    bonus = torch.multinomial(dist, 1, generator=generator).squeeze(0)
    result = torch.stack((n_acc_device, bonus.to(torch.int64))).cpu()
    return int(result[0]), int(result[1])


class DSparkAcceptanceFallback:
    """Request-local acceptance-rate circuit breaker from upstream PR #71."""

    def __init__(
        self,
        threshold: float,
        min_drafted: int,
        cooldown_steps: int,
        *,
        cumulative: bool = False,
    ) -> None:
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("DSpark fallback threshold must be in (0, 1]")
        if min_drafted < 1 or cooldown_steps < 1:
            raise ValueError("DSpark fallback counters must be positive")
        self.threshold = float(threshold)
        self.min_drafted = int(min_drafted)
        self.cooldown_steps = int(cooldown_steps)
        self.cumulative = bool(cumulative)
        self.reset()

    def reset(self) -> None:
        self.accepted = 0
        self.drafted = 0
        self.cooldown_left = 0
        self.disabled = False

    def should_speculate(self) -> bool:
        if self.cumulative:
            return not self.disabled
        if self.cooldown_left:
            self.cooldown_left -= 1
            return False
        return True

    def record(self, accepted: int, drafted: int) -> float | None:
        if self.cumulative and self.disabled:
            return None
        self.accepted += int(accepted)
        self.drafted += int(drafted)
        if self.drafted < self.min_drafted:
            return None
        rate = self.accepted / self.drafted
        if self.cumulative:
            if rate >= self.threshold:
                return None
            self.disabled = True
            return rate
        self.accepted = self.drafted = 0
        if rate >= self.threshold:
            return None
        self.cooldown_left = self.cooldown_steps
        return rate


class DSparkAdaptiveVerification:
    """Choose the next single-request verification prefix from DSpark confidence.

    DSpark's confidence head emits one conditional acceptance logit per draft
    position.  A prefix of length ``k`` therefore has expected useful output
    ``1 + sum(cumprod(sigmoid(logits))[:k])``: one target bonus plus the
    expected accepted draft prefix.  Dividing that value by a measured total
    step cost gives the same throughput objective used by vLLM's adaptive
    verifier, specialized to FreeToken's one-request V4.1 path.

    The caller applies the result to the *next* decode step.  This avoids a
    device-to-host synchronization in front of the current target pass.  A
    zero-length choice is intentionally excluded: FreeToken decides whether to
    launch the auxiliary drafter before confidence exists, while its separate
    acceptance fallback already provides a target-only circuit breaker.
    """

    def __init__(self, step_costs_ms: Sequence[float]) -> None:
        if not step_costs_ms:
            raise ValueError("DSpark adaptive verification needs step costs")
        costs = []
        previous = 0.0
        for value in step_costs_ms:
            cost = float(value)
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("DSpark adaptive step costs must be positive")
            # Profiling noise must not make a longer target pass appear cheaper.
            previous = max(previous, cost)
            costs.append(previous)
        self.step_costs_ms = tuple(costs)
        self._cost_tensors: dict[torch.device, torch.Tensor] = {}

    @property
    def max_length(self) -> int:
        return len(self.step_costs_ms)

    def choose_tensor_from_probabilities(
        self, confidence_probs: torch.Tensor
    ) -> torch.Tensor:
        probs = confidence_probs.detach().float().view(-1)[: self.max_length]
        if probs.numel() != self.max_length:
            raise ValueError(
                f"DSpark confidence has {probs.numel()} positions, expected "
                f"{self.max_length}"
            )
        probs = probs.clamp_(0.0, 1.0)
        survival = probs.cumprod(dim=0)
        expected_output = 1.0 + survival.cumsum(dim=0)
        costs = self._cost_tensors.get(expected_output.device)
        if costs is None:
            costs = torch.tensor(
                self.step_costs_ms,
                dtype=expected_output.dtype,
                device=expected_output.device,
            )
            self._cost_tensors[expected_output.device] = costs
        return (expected_output / costs).argmax().to(torch.int32) + 1

    def choose_from_probabilities(self, confidence_probs: torch.Tensor) -> int:
        return int(self.choose_tensor_from_probabilities(confidence_probs).item())

    def choose_tensor(self, confidence_logits: torch.Tensor) -> torch.Tensor:
        return self.choose_tensor_from_probabilities(confidence_logits.sigmoid())

    def choose(self, confidence_logits: torch.Tensor) -> int:
        return int(self.choose_tensor(confidence_logits).item())

    def initial_length(self) -> int:
        return self.choose_from_probabilities(torch.ones(self.max_length))


class ResidentDSFP4MoE(nn.Module):
    """One complete V4.1 draft MoE with resident packed expert banks."""

    def __init__(self, args: DeepseekV41Args) -> None:
        super().__init__()
        draft_args = replace(
            args,
            n_routed_experts=args.dspark_n_routed_experts,
            n_activated_experts=args.dspark_n_activated_experts,
        )
        self.gate = Gate(draft_args)
        self.shared_experts = SharedExpert(draft_args)
        e, h, inter = (
            args.dspark_n_routed_experts,
            args.dim,
            args.moe_inter_dim,
        )
        e8m0 = torch.float8_e8m0fnu
        self.register_buffer(
            "gate_up_packed",
            torch.empty(e, 2 * inter, h // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "gate_up_scale",
            torch.empty(e, 2 * inter, h // 32, dtype=e8m0),
        )
        self.register_buffer(
            "down_packed",
            torch.empty(e, h, inter // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "down_scale",
            torch.empty(e, h, inter // 32, dtype=e8m0),
        )
        self.n_experts = e
        self.swiglu_limit = args.swiglu_limit
        self.activation_block_size = int(args.weight_block_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

        shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        weights, ids = self.gate(flat)
        shared = self.shared_experts(flat)
        routed = routed_experts_fp4(
            flat,
            ids.to(torch.int32).contiguous(),
            weights.float().contiguous(),
            self.gate_up_packed,
            self.gate_up_scale,
            self.down_packed,
            self.down_scale,
            self.swiglu_limit,
            activation_block_size=self.activation_block_size,
        )
        return (shared + routed.to(shared.dtype)).view(shape)


class DSparkAttention(nn.Module):
    """Non-causal block attention over a resident 128-token context ring."""

    def __init__(self, args: DeepseekV41Args) -> None:
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
        self.fp8_block_size = int(args.weight_block_size[1])
        self.softmax_scale = args.head_dim**-0.5
        self.attn_sink = nn.Parameter(
            torch.empty(args.n_heads, dtype=torch.float32), requires_grad=False
        )
        self.wq_a = Linear(args.dim, args.q_lora_rank)
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = Linear(args.q_lora_rank, args.n_heads * args.head_dim)
        self.wkv = Linear(args.dim, args.head_dim)
        self.kv_norm = RMSNorm(args.head_dim, args.norm_eps)
        self.wo_a = nn.Parameter(
            torch.empty(
                args.o_groups * args.o_lora_rank,
                args.n_heads * args.head_dim // args.o_groups,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.wo_b = Linear(args.o_groups * args.o_lora_rank, args.dim)
        self.register_buffer(
            "window_kv_cache",
            torch.zeros(args.window_size, args.head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "_window_offsets",
            torch.arange(-args.window_size, 0, dtype=torch.long),
            persistent=False,
        )
        self.freqs_cis: torch.Tensor | None = None

    def bind(self, device: torch.device) -> None:
        from freetoken.models.deepseek_v4.ops import get_freqs_cis

        self.freqs_cis = get_freqs_cis(
            self.args.rope_head_dim,
            self.args.max_seq_len,
            0,
            self.args.rope_theta,
            self.args.rope_factor,
            self.args.beta_fast,
            self.args.beta_slow,
            device,
        )

    def _kv(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
        from freetoken.models.deepseek_v4.ops import apply_rotary_emb_decode

        assert self.freqs_cis is not None
        kv = self.kv_norm(self.wkv(x))
        freqs = self.freqs_cis.index_select(0, positions.long())
        apply_rotary_emb_decode(
            kv[..., -self.rope_head_dim :].reshape(-1, 1, self.rope_head_dim),
            freqs,
        )
        act_quant_fp8_inplace(kv, self.fp8_block_size)
        return kv

    def catch_up(self, main_x: torch.Tensor, positions: torch.Tensor) -> None:
        if positions.numel() == 0:
            return
        kv = self._kv(main_x, positions)
        # Only the last window can survive.  Trimming first makes the modulo rows
        # unique, avoiding duplicate-index index_copy semantics on long prefills.
        if positions.numel() > self.window_size:
            positions = positions[-self.window_size :]
            kv = kv[-self.window_size :]
        self.window_kv_cache.index_copy_(
            0, torch.remainder(positions.long(), self.window_size), kv
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        fixed_window: bool = False,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
        from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged
        from freetoken.models.deepseek_v4.ops import apply_rotary_emb_decode

        assert self.freqs_cis is not None
        if x.shape[0] != 1:
            raise ValueError("V4.1 auxiliary DSpark currently supports one stream")
        block = x.shape[1]
        flat_pos = positions.long().view(-1)
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        freqs = self.freqs_cis.index_select(0, flat_pos)
        apply_rotary_emb_decode(
            q[..., -self.rope_head_dim :].reshape(-1, self.n_heads, self.rope_head_dim),
            freqs,
        )
        block_kv = self._kv(x[0], flat_pos)

        if fixed_window:
            # CUDA graph replay must derive the live ring rows from device data;
            # baking ``positions[0].item()`` into capture would replay stale KV.
            # The graph is used only once a complete 128-token window exists.
            context_pos = flat_pos[:1] + self._window_offsets
        else:
            start = int(flat_pos[0].item())
            lo = max(0, start - self.window_size)
            context_pos = torch.arange(lo, start, device=x.device, dtype=torch.long)
        context = self.window_kv_cache.index_select(
            0, torch.remainder(context_pos, self.window_size)
        )
        gathered = torch.cat([context, block_kv], dim=0).contiguous()
        width = gathered.shape[0]
        topk = torch.arange(width, device=x.device, dtype=torch.int32).view(1, 1, -1)
        topk = topk.expand(1, block, -1).contiguous()
        # The paged kernel needs two equal-stride pools. Every column is marked as
        # window, so the one-row second pool is never read.
        dummy = torch.empty(1, self.head_dim, dtype=gathered.dtype, device=x.device)
        output = sparse_attn_paged(
            q,
            gathered,
            dummy,
            self.attn_sink,
            topk,
            width,
            self.softmax_scale,
        )
        apply_rotary_emb_decode(
            output[..., -self.rope_head_dim :].reshape(
                -1, self.n_heads, self.rope_head_dim
            ),
            freqs,
            inverse=True,
        )
        output = output.reshape(
            1,
            block,
            self.n_groups,
            self.n_heads * self.head_dim // self.n_groups,
        )
        weight = self.wo_a.view(self.n_groups, self.o_lora_rank, -1)
        output = torch.einsum("bsgd,grd->bsgr", output, weight).flatten(-2)
        return self.wo_b(output)


class DSparkBlock(nn.Module):
    def __init__(self, args: DeepseekV41Args) -> None:
        super().__init__()
        self.attn = DSparkAttention(args)
        self.ffn = ResidentDSFP4MoE(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.norm_eps
        mix = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.dim
        for sublayer in ("attn", "ffn"):
            setattr(
                self,
                f"hc_{sublayer}_fn",
                nn.Parameter(torch.empty(mix, hc_dim, dtype=torch.float32), requires_grad=False),
            )
            setattr(
                self,
                f"hc_{sublayer}_base",
                nn.Parameter(torch.empty(mix, dtype=torch.float32), requires_grad=False),
            )
            setattr(
                self,
                f"hc_{sublayer}_scale",
                nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False),
            )

    def _mixes(self, x: torch.Tensor, sublayer: str):
        return hc_mixes(
            x,
            getattr(self, f"hc_{sublayer}_fn"),
            getattr(self, f"hc_{sublayer}_scale"),
            getattr(self, f"hc_{sublayer}_base"),
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
            norm_eps=self.norm_eps,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        incoming: torch.Tensor,
        positions: torch.Tensor,
        *,
        fixed_window: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden
        attn_pre, attn_post, attn_comb = self._mixes(hidden, "attn")
        output = self.attn(
            self.attn_norm(hc_pre(hidden, incoming)),
            positions,
            fixed_window=fixed_window,
        )
        hidden = hc_post(output, residual, attn_post, attn_comb)

        residual = hidden
        ffn_pre, ffn_post, ffn_comb = self._mixes(hidden, "ffn")
        output = self.ffn(self.ffn_norm(hc_pre(hidden, attn_pre)))
        return hc_post(output, residual, ffn_post, ffn_comb), ffn_pre


class DSparkDrafter(nn.Module):
    """Three-stage V4.1 block drafter resident on one auxiliary CUDA device."""

    def __init__(self, args: DeepseekV41Args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.empty(0).device
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.target_layer_ids = tuple(args.dspark_target_layer_ids)
        self.main_proj = Linear(args.dim * len(self.target_layer_ids), args.dim)
        self.main_norm = RMSNorm(args.dim, args.norm_eps)
        self.stages = nn.ModuleList([DSparkBlock(args) for _ in range(args.n_mtp_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.markov_embed = nn.Parameter(
            torch.empty(args.vocab_size, args.dspark_markov_rank, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.markov_head = nn.Parameter(
            torch.empty(args.vocab_size, args.dspark_markov_rank, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.confidence_proj = nn.Parameter(
            torch.empty(1, args.dim + args.dspark_markov_rank, dtype=torch.float32),
            requires_grad=False,
        )
        self._target_embed = None
        self._target_logits = None
        self._draft_graph_requested = _env_flag(
            "FREETOKEN_DSV41_DSPARK_DRAFT_CUDA_GRAPH"
        )
        self.register_buffer(
            "_local_target_embed_weight", None, persistent=False
        )
        self.register_buffer(
            "_local_target_head_weight", None, persistent=False
        )
        self._draft_graph: torch.cuda.CUDAGraph | None = None
        self._draft_graph_input_ids: torch.Tensor | None = None
        self._draft_graph_positions: torch.Tensor | None = None
        self._draft_graph_hidden: torch.Tensor | None = None
        self._draft_graph_logits: torch.Tensor | None = None
        self._greedy_graphs: dict[
            int,
            tuple[
                torch.cuda.CUDAGraph,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ],
        ] = {}
        self._catch_up_graphs: dict[
            int,
            tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor],
        ] = {}
        self._needs_confidence = True

    def bind(
        self,
        target_embed,
        target_logits,
        *,
        target_embed_weight: torch.Tensor | None = None,
        target_head_weight: torch.Tensor | None = None,
        graph_lengths: Sequence[int] | None = None,
        needs_confidence: bool = True,
    ) -> None:
        self.device = self.main_proj.weight.device
        self._target_embed = target_embed
        self._target_logits = target_logits
        self._needs_confidence = bool(needs_confidence)
        for stage in self.stages:
            stage.attn.bind(self.device)
        if self._draft_graph_requested:
            if target_embed_weight is None or target_head_weight is None:
                raise RuntimeError(
                    "DSpark draft CUDA graph requires target embedding and head weights"
                )
            with torch.cuda.device(self.device):
                self._local_target_embed_weight = (
                    target_embed_weight.detach().to(self.device)
                )
                self._local_target_head_weight = (
                    target_head_weight.detach().to(self.device)
                )
                lengths = tuple(
                    sorted(set(graph_lengths or (self.block_size,)))
                )
                if not lengths or any(
                    length < 1 or length > self.block_size for length in lengths
                ):
                    raise ValueError(
                        f"DSpark draft graph lengths must be in [1, {self.block_size}]"
                    )
                # Adaptive verification always needs all confidence positions,
                # regardless of the target prefix selected for the current step.
                if self._needs_confidence:
                    lengths = (self.block_size,)
                self._capture_draft_graphs(lengths)
                self._capture_catch_up_graphs()

    def _forward_local_backbone(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        fixed_window: bool,
        logits_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the three-stage draft stack entirely on the auxiliary GPU."""

        if (
            self._local_target_embed_weight is None
            or self._local_target_head_weight is None
        ):
            raise RuntimeError("DSpark local target weights are not loaded")
        embeds = F.embedding(input_ids.long(), self._local_target_embed_weight)
        hidden = embeds.unsqueeze(2).repeat(1, 1, self.args.hc_mult, 1)
        incoming = identity_pre_mix(hidden, self.args.hc_mult)
        for stage in self.stages:
            hidden, incoming = stage(
                hidden,
                incoming,
                positions,
                fixed_window=fixed_window,
            )
        head_hidden = hc_pre(hidden, incoming)
        if logits_length is not None:
            head_hidden = head_hidden[:, :logits_length]
        normalized = self.norm(head_hidden)
        base_logits = F.linear(normalized.float(), self._local_target_head_weight)
        return head_hidden, base_logits

    def _capture_draft_graphs(self, greedy_lengths: Sequence[int]) -> None:
        """Capture the backbone and common full greedy draft shapes.

        The backbone-only graph remains the exact fallback for arbitrary sampled
        requests.  Greedy requests use one end-to-end graph per configured K,
        including the sequential Markov dependency and argmax chain, matching
        the scope of vLLM's DSpark graph rather than graphing verification alone.
        """

        ids = torch.full(
            (1, self.block_size),
            self.noise_token_id,
            dtype=torch.long,
            device=self.device,
        )
        ids[0, 0] = 0
        positions = torch.arange(
            self.args.window_size,
            self.args.window_size + self.block_size,
            dtype=torch.long,
            device=self.device,
        )
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(capture_stream):
            # Warm every lazy module and allocator path before capture.
            for _ in range(2):
                self._forward_local_backbone(
                    ids, positions, fixed_window=True
                )
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            hidden, logits = self._forward_local_backbone(
                ids, positions, fixed_window=True
            )
        capture_stream.synchronize()
        self._draft_graph = graph
        self._draft_graph_input_ids = ids
        self._draft_graph_positions = positions
        self._draft_graph_hidden = hidden
        self._draft_graph_logits = logits

        for length in greedy_lengths:
            graph_ids = ids.clone()
            graph_positions = positions.clone()
            proposed = torch.empty(
                length, dtype=torch.long, device=self.device
            )
            confidence = (
                torch.empty(length, dtype=torch.float32, device=self.device)
                if self._needs_confidence
                else None
            )
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    head_hidden, base_logits = self._forward_local_backbone(
                        graph_ids,
                        graph_positions,
                        fixed_window=True,
                        logits_length=length,
                    )
                    self._sample_greedy(
                        graph_ids[0, :1],
                        head_hidden,
                        base_logits,
                        proposed,
                        confidence,
                    )
            capture_stream.synchronize()

            greedy_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(greedy_graph, stream=capture_stream):
                head_hidden, base_logits = self._forward_local_backbone(
                    graph_ids,
                    graph_positions,
                    fixed_window=True,
                    logits_length=length,
                )
                self._sample_greedy(
                    graph_ids[0, :1],
                    head_hidden,
                    base_logits,
                    proposed,
                    confidence,
                )
            capture_stream.synchronize()
            self._greedy_graphs[length] = (
                greedy_graph,
                graph_ids,
                graph_positions,
                proposed,
                confidence,
            )
        resident = (
            self._local_target_embed_weight.numel()
            * self._local_target_embed_weight.element_size()
            + self._local_target_head_weight.numel()
            * self._local_target_head_weight.element_size()
        )
        logger.info_rank0(
            "Captured DeepSeek-V4.1 DSpark draft backbone and greedy K=%s on %s; "
            "local target embed/head %.2f GiB",
            tuple(greedy_lengths),
            self.device,
            resident / (1 << 30),
        )

    def _catch_up_local(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> None:
        main_x = self.combine_target_hidden(hidden.view(-1, hidden.shape[-1]))
        for stage in self.stages:
            stage.attn.catch_up(main_x, positions)

    def _capture_catch_up_graphs(self) -> None:
        """Capture the small steady-state target-context updates on the drafter."""

        width = self.args.dim * len(self.target_layer_ids)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        for count in range(1, self.block_size + 2):
            hidden = torch.zeros(
                count, width, dtype=torch.bfloat16, device=self.device
            )
            positions = torch.arange(
                count, dtype=torch.long, device=self.device
            )
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self._catch_up_local(hidden, positions)
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                self._catch_up_local(hidden, positions)
            capture_stream.synchronize()
            self._catch_up_graphs[count] = (graph, hidden, positions)
        for stage in self.stages:
            stage.attn.window_kv_cache.zero_()
        logger.info_rank0(
            "Captured DeepSeek-V4.1 DSpark context catch-up graphs for rows 1..%d",
            self.block_size + 1,
        )

    def _can_replay_draft_graph(self, positions: torch.Tensor) -> bool:
        if self._draft_graph is None or positions.numel() < self.block_size:
            return False
        # Short prefixes have a variable-width context and retain the eager path.
        return int(positions[0].item()) >= int(self.args.window_size)

    def _replay_greedy_graph(
        self,
        anchor: torch.Tensor,
        positions: torch.Tensor,
        proposal_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        entry = self._greedy_graphs.get(proposal_length)
        if entry is None or not self._can_replay_draft_graph(positions):
            return None
        graph, ids, graph_positions, proposed, confidence = entry
        ids.fill_(self.noise_token_id)
        ids[0, 0].copy_(anchor.to(self.device).long().reshape(()))
        graph_positions.copy_(positions[: self.block_size].to(self.device))
        graph.replay()
        return proposed, confidence

    def _draft_backbone(
        self,
        anchor: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._can_replay_draft_graph(positions):
            assert self._draft_graph_input_ids is not None
            assert self._draft_graph_positions is not None
            assert self._draft_graph_hidden is not None
            assert self._draft_graph_logits is not None
            self._draft_graph_input_ids.fill_(self.noise_token_id)
            self._draft_graph_input_ids[0, 0].copy_(
                anchor.to(self.device).long().reshape(())
            )
            self._draft_graph_positions.copy_(
                positions[: self.block_size].to(self.device)
            )
            self._draft_graph.replay()
            return self._draft_graph_hidden, self._draft_graph_logits

        ids = torch.full(
            (1, self.block_size),
            self.noise_token_id,
            dtype=torch.long,
            device=self.device,
        )
        ids[0, 0].copy_(anchor.to(self.device).long().reshape(()))
        local_positions = positions[: self.block_size].to(self.device).view(-1)
        if self._local_target_embed_weight is not None:
            return self._forward_local_backbone(
                ids, local_positions, fixed_window=False
            )

        embeds = self._target_embed(ids.to(anchor.device)).to(self.device)
        hidden = embeds.unsqueeze(2).repeat(1, 1, self.args.hc_mult, 1)
        incoming = identity_pre_mix(hidden, self.args.hc_mult)
        for stage in self.stages:
            hidden, incoming = stage(hidden, incoming, local_positions)
        head_hidden = hc_pre(hidden, incoming)
        normalized = self.norm(head_hidden)
        base_logits = self._target_logits(normalized.to(anchor.device)).to(self.device)
        return head_hidden, base_logits

    def combine_target_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        expect = self.args.dim * len(self.target_layer_ids)
        if hidden.shape[-1] != expect:
            raise ValueError(f"DSpark target hidden width {hidden.shape[-1]} != {expect}")
        return self.main_norm(self.main_proj(hidden.to(self.device)))

    @torch.inference_mode()
    def catch_up_context(self, hidden: torch.Tensor, positions: torch.Tensor) -> None:
        if hidden is None or positions.numel() == 0:
            return
        # FreeToken's engine stream belongs to the text authority.  Triton launches
        # on torch's current CUDA device rather than inferring it from every pointer,
        # so explicitly select the auxiliary GPU around all resident-drafter work.
        with torch.cuda.device(self.device):
            count = int(positions.numel())
            entry = self._catch_up_graphs.get(count)
            if entry is not None:
                graph, graph_hidden, graph_positions = entry
                graph_hidden.copy_(hidden.view_as(graph_hidden).to(self.device))
                graph_positions.copy_(positions.view_as(graph_positions).to(self.device))
                graph.replay()
                return
            self._catch_up_local(
                hidden.view(-1, hidden.shape[-1]).to(self.device),
                positions.to(self.device),
            )

    def _sample_greedy(
        self,
        anchor: torch.Tensor,
        head_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        proposed: torch.Tensor,
        confidence: torch.Tensor | None,
    ) -> None:
        previous = anchor.long().reshape(1)
        for k in range(proposed.numel()):
            markov = F.embedding(previous, self.markov_embed)
            logits = (
                base_logits[0, k].float()
                + F.linear(markov, self.markov_head).squeeze(0).float()
            )
            token = logits.argmax().view(1)
            proposed[k].copy_(token[0])
            if confidence is not None:
                confidence[k].copy_(
                    F.linear(
                        torch.cat(
                            [head_hidden[0, k].float(), markov[0].float()]
                        ),
                        self.confidence_proj,
                    ).squeeze()
                )
            previous = token

    @torch.inference_mode()
    def propose(
        self,
        anchor: torch.Tensor,
        positions: torch.Tensor,
        sampling_params: Sequence,
        *,
        proposal_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        with torch.cuda.device(self.device):
            return self._propose(
                anchor,
                positions,
                sampling_params,
                proposal_length=proposal_length,
            )

    def _propose(
        self,
        anchor: torch.Tensor,
        positions: torch.Tensor,
        sampling_params: Sequence,
        *,
        proposal_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if len(sampling_params) != 1 or anchor.numel() != 1:
            raise ValueError("V4.1 auxiliary DSpark currently supports one request")
        if self._target_embed is None or self._target_logits is None:
            raise RuntimeError("DSpark target embedding/head callbacks are not bound")
        params = sampling_params[0]
        requested = self.block_size if proposal_length is None else int(proposal_length)
        if requested < 1 or requested > self.block_size:
            raise ValueError(
                f"DSpark proposal length {requested} is outside [1, {self.block_size}]"
            )
        length = self.block_size if self._needs_confidence else requested
        if params.is_greedy:
            replayed = self._replay_greedy_graph(anchor, positions, length)
            if replayed is not None:
                proposed, confidence = replayed
                target_device = anchor.device
                return (
                    proposed.to(target_device),
                    None,
                    confidence.to(target_device)
                    if confidence is not None
                    else None,
                )
        head_hidden, base_logits = self._draft_backbone(anchor, positions)
        head_hidden = head_hidden[:, :length]
        base_logits = base_logits[:, :length]
        proposed = torch.empty(length, dtype=torch.long, device=self.device)
        q = (
            None
            if params.is_greedy
            else torch.empty(
                length,
                self.args.vocab_size,
                dtype=torch.float32,
                device=self.device,
            )
        )
        confidence = (
            torch.empty(length, dtype=torch.float32, device=self.device)
            if self._needs_confidence
            else None
        )
        previous = anchor.to(self.device).long().reshape(1)
        for k in range(length):
            markov = F.embedding(previous, self.markov_embed)
            logits = base_logits[0, k].float() + F.linear(markov, self.markov_head).squeeze(0).float()
            if params.is_greedy:
                token = logits.argmax().view(1)
            else:
                assert q is not None
                q_k = sampling_probs(
                    logits.unsqueeze(0),
                    params.temperature,
                    params.top_p,
                    params.top_k,
                )[0]
                q[k].copy_(q_k)
                token = torch.multinomial(q_k, 1)
            proposed[k] = token[0]
            if confidence is not None:
                confidence[k] = F.linear(
                    torch.cat([head_hidden[0, k].float(), markov[0].float()]),
                    self.confidence_proj,
                ).squeeze()
            previous = token
        target_device = anchor.device
        return (
            proposed.to(target_device),
            q.to(target_device) if q is not None else None,
            confidence.to(target_device) if confidence is not None else None,
        )


_MTP_EXPERT_RE = re.compile(
    r"^mtp\.(?P<stage>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)


def _raw_parameter_map(drafter: DSparkDrafter) -> dict[str, nn.Parameter]:
    result: dict[str, nn.Parameter] = {
        "mtp.0.main_proj.weight": drafter.main_proj.weight,
        "mtp.0.main_proj.scale": drafter.main_proj.scale,
        "mtp.0.main_norm.weight": drafter.main_norm.weight,
        f"mtp.{len(drafter.stages) - 1}.norm.weight": drafter.norm.weight,
        f"mtp.{len(drafter.stages) - 1}.markov_head.embed.weight": drafter.markov_embed,
        f"mtp.{len(drafter.stages) - 1}.markov_head.head.weight": drafter.markov_head,
        f"mtp.{len(drafter.stages) - 1}.confidence_head.proj.weight": drafter.confidence_proj,
    }
    for stage_id, stage in enumerate(drafter.stages):
        for name, parameter in stage.named_parameters():
            if name == "attn.wo_a":
                result[f"mtp.{stage_id}.attn.wo_a.weight"] = parameter
            else:
                result[f"mtp.{stage_id}.{name}"] = parameter
    return result


def _copy_expert_tensor(
    drafter: DSparkDrafter, raw_name: str, value: torch.Tensor
) -> None:
    match = _MTP_EXPERT_RE.match(raw_name)
    if match is None:
        raise ValueError(f"not an MTP expert tensor: {raw_name}")
    stage = drafter.stages[int(match.group("stage"))].ffn
    expert = int(match.group("expert"))
    projection, kind = match.group("proj"), match.group("kind")
    inter = drafter.args.moe_inter_dim
    if kind == "weight":
        value = value.view(torch.uint8)
        if projection == "w1":
            stage.gate_up_packed[expert, :inter].copy_(value)
        elif projection == "w3":
            stage.gate_up_packed[expert, inter:].copy_(value)
        else:
            stage.down_packed[expert].copy_(value)
    elif projection == "w1":
        stage.gate_up_scale[expert, :inter].copy_(value)
    elif projection == "w3":
        stage.gate_up_scale[expert, inter:].copy_(value)
    else:
        stage.down_scale[expert].copy_(value)


@torch.inference_mode()
def load_dspark_checkpoint(
    model_path: str, drafter: DSparkDrafter, device: torch.device
) -> int:
    """Load and validate the complete ``mtp.*`` payload directly on ``device``."""

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    raw_names = sorted(name for name in weight_map if name.startswith("mtp."))
    parameters = _raw_parameter_map(drafter)
    expected_experts = (
        len(drafter.stages) * drafter.args.dspark_n_routed_experts * 6
    )
    expert_names = [name for name in raw_names if _MTP_EXPERT_RE.match(name)]
    if len(expert_names) != expected_experts:
        raise ValueError(
            f"MTP expert contract has {len(expert_names)} tensors, expected {expected_experts}"
        )
    optional = {
        f"mtp.{stage}.ffn.gate.bias_vl" for stage in range(len(drafter.stages))
    }
    consumed_scales = {
        f"mtp.{stage}.attn.wo_a.scale" for stage in range(len(drafter.stages))
    }
    expected = set(parameters) | set(expert_names) | optional | consumed_scales
    missing = sorted(set(parameters) - set(raw_names))
    unexpected = sorted(set(raw_names) - expected)
    if missing or unexpected:
        raise ValueError(
            f"MTP checkpoint contract differs: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )

    reader = ShardReader(model_path, device)
    try:
        for raw_name, parameter in parameters.items():
            value = reader.get_tensor(raw_name)
            if raw_name.endswith(".attn.wo_a.weight"):
                from .weight import _dequant_fp8_block32

                scale = reader.get_tensor(raw_name.removesuffix(".weight") + ".scale")
                value = _dequant_fp8_block32(value, scale)
            parameter.copy_(value.to(dtype=parameter.dtype))
        for raw_name in expert_names:
            _copy_expert_tensor(drafter, raw_name, reader.get_tensor(raw_name))
    finally:
        files = reader.files()
        reader.close()
        for path in files:
            drop_page_cache(path)

    total = 0
    for parameter in drafter.parameters():
        total += parameter.numel() * parameter.element_size()
    for _name, buffer in drafter.named_buffers():
        total += buffer.numel() * buffer.element_size()
    logger.info_rank0(
        "Loaded DeepSeek-V4.1 DSpark on %s: %.2f GiB resident",
        device,
        total / (1 << 30),
    )
    return int(total)


__all__ = [
    "DSparkAcceptanceFallback",
    "DSparkAdaptiveVerification",
    "DSparkDrafter",
    "accepted_prefix",
    "load_dspark_checkpoint",
    "rejection_accept_device",
    "sampling_probs",
]
