"""DeepSeek-V4.1 ordinary-generation text runtime.

MTP/DSpark and vision are intentionally outside this first serving path.  The
registered adapter below runs one dense TP1 authority plus rank-local routed
experts and row-sharded, resident Engram tables on every EP rank.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.models.blocks import BaseLLMModel

from .args import DeepseekV41Args
from .attention_layout import AttentionLayout
from .compress import Compressor
from .engram_layer import Engram
from .indexer import (
    CandidateRuntime,
    indexer_prefill_chunk_rows,
    indexer_prefill_max_logits_bytes,
    select_candidate_blocks,
    select_index_topk,
)
from .layers import Linear, RMSNorm
from .moe import Gate, MoE, SharedExpert
from .profile import profile, profile_range


class SharedAttentionRuntime:
    """Per-forward publications consumed by later CSA2 layers."""

    def __init__(self) -> None:
        self.topk_rows: torch.Tensor | None = None
        self.candidates = CandidateRuntime()

    def reset(self) -> None:
        self.topk_rows = None
        self.candidates.mask = None


shared_attention = SharedAttentionRuntime()


class Indexer(nn.Module):
    def __init__(self, args: DeepseekV41Args, layer_id: int) -> None:
        super().__init__()
        self.compress_ratio = int(args.compress_ratios[layer_id])
        self.owns_k = layer_id in args.kv_source_layers
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.n_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.prefill_max_logits_bytes = indexer_prefill_max_logits_bytes()
        self.head_weight_scale = args.index_head_dim**-0.5 * args.index_n_heads**-0.5
        self.wq_b = Linear(args.q_lora_rank, args.index_n_heads * args.index_head_dim)
        self.weights_proj = Linear(args.dim, args.index_n_heads, kind="bf16")
        if self.owns_k:
            self.wk = Linear(args.head_dim, args.index_head_dim, kind="bf16")
            self.k_norm = RMSNorm(args.index_head_dim, args.norm_eps)

    @staticmethod
    def _rope_fp4(x: torch.Tensor, freqs: torch.Tensor, rope_dim: int) -> torch.Tensor:
        if x.is_cuda:
            from freetoken.kernel.triton.dsv41 import rope_fp4_roundtrip

            return rope_fp4_roundtrip(x, freqs, rope_dim)
        head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
        value = torch.view_as_complex(
            tail.float().unflatten(-1, (-1, 2)).contiguous()
        )
        f = freqs.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
        tail = torch.view_as_real(value * f).flatten(-2).to(x.dtype)
        value = torch.cat([head, tail], dim=-1)
        # CPU tooling uses the exact fake-quant oracle; serving takes the fused path.
        from .quant import fake_quant_fp4

        return fake_quant_fp4(value, block_size=32)

    def index_keys(self, latent: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        if not self.owns_k:
            raise RuntimeError("only a KV source layer can publish V4.1 index keys")
        key = self.k_norm(self.wk(latent))
        return self._rope_fp4(key, freqs, self.rope_head_dim)

    def queries(self, q_lora: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        query = self.wq_b(q_lora).view(
            q_lora.shape[0], self.n_heads, self.index_head_dim
        )
        return self._rope_fp4(query, freqs, self.rope_head_dim)

    def head_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights_proj(x) * self.head_weight_scale

    @staticmethod
    def scores(
        query: torch.Tensor, keys: torch.Tensor, head_weights: torch.Tensor
    ) -> torch.Tensor:
        score = torch.einsum("bhd,nd->bhn", query, keys)
        return (score.relu() * head_weights.unsqueeze(-1)).sum(1).float()

    def select(
        self,
        scores: torch.Tensor,
        compressed_lengths: torch.Tensor | int,
        *,
        offset: int = 0,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return select_index_topk(
            scores,
            compressed_lengths,
            self.index_topk,
            offset=offset,
            candidate_mask=candidate_mask,
        )

    def select_candidate_mask(
        self, scores: torch.Tensor, compressed_lengths: torch.Tensor | int
    ) -> torch.Tensor:
        return select_candidate_blocks(
            scores,
            compressed_lengths,
            self.candidate_topk_blocks,
            self.candidate_block_size,
        )


class Attention(nn.Module):
    def __init__(
        self,
        args: DeepseekV41Args,
        layer_id: int,
        layout: AttentionLayout,
    ) -> None:
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.plan = layout[layer_id]
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
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
        self.compressor = Compressor(args, layer_id) if self.plan.owns_kv else None
        self.indexer = Indexer(args, layer_id) if self.plan.owns_index else None
        if self.plan.compress_ratio:
            original_len, theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_len, theta = 0, args.rope_theta
        self._freqs_params = (
            args.rope_head_dim,
            args.max_seq_len,
            original_len,
            theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.freqs_cis: torch.Tensor | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, device: torch.device) -> None:
        from freetoken.models.deepseek_v4.ops import get_freqs_cis

        self.freqs_cis = get_freqs_cis(*self._freqs_params, device)

    def _project_q(self, x: torch.Tensor, freqs: torch.Tensor):
        from freetoken.models.deepseek_v4.ops import apply_rotary_emb

        q_lora = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_lora).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -self.rope_head_dim :], freqs)
        return q_lora, q

    def _project_window_kv(self, x: torch.Tensor, freqs: torch.Tensor):
        from freetoken.models.deepseek_v4.ops import apply_rotary_emb

        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs)
        act_quant_fp8_inplace(kv, 64)
        return kv

    def _project_output(self, output: torch.Tensor) -> torch.Tensor:
        output = output.reshape(
            *output.shape[:-2], self.n_groups, self.n_heads * self.head_dim // self.n_groups
        )
        weight = self.wo_a.view(self.n_groups, self.o_lora_rank, -1)
        output = torch.einsum("...gd,grd->...gr", output, weight).flatten(-2)
        return self.wo_b(output)

    def _publish_source_prefill(
        self,
        x: torch.Tensor,
        start_pos: int,
        table_idx: int,
        window_slots: torch.Tensor,
    ) -> None:
        assert self.compressor is not None and self.indexer is not None
        latent, rows = self.compressor.prefill_paged(
            x,
            start_pos,
            window_slots,
            layer_id=self.layer_id,
            table_idx=table_idx,
            backend=self.attn,
        )
        if latent.shape[1] == 0:
            return
        ratio = self.plan.compress_ratio
        positions = start_pos + torch.arange(
            0, latent.shape[1] * ratio, ratio, device=x.device
        )
        freqs = self.freqs_cis.index_select(0, positions)
        index_k = self.indexer.index_keys(latent[0], freqs)
        self.attn.compress_pool(self.layer_id, "idx").index_copy_(0, rows, index_k)
        from freetoken.kernel.triton.dsv41 import rope_fp4_roundtrip

        compressed = rope_fp4_roundtrip(
            latent[0], freqs, self.rope_head_dim, compressed_kv=True
        )
        self.attn.compress_pool(self.layer_id, "attn").index_copy_(
            0, rows, compressed
        )

    def _index_prefill(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        start_pos: int,
        table_idx: int,
    ) -> torch.Tensor:
        if not self.plan.owns_index:
            if shared_attention.topk_rows is None:
                raise RuntimeError(
                    f"V4.1 layer {self.layer_id} consumed top-k before publication"
                )
            return shared_attention.topk_rows
        assert self.indexer is not None
        ratio = self.plan.compress_ratio
        end = start_pos + x.shape[1]
        n_blocks = end // ratio
        if n_blocks == 0:
            selected = torch.empty(
                x.shape[1], 0, dtype=torch.int32, device=x.device
            )
        else:
            keys = self.attn.indexer_keys(
                table_idx, n_blocks, ratio, self.layer_id, 1
            )[0]
            positions = start_pos + torch.arange(x.shape[1], device=x.device)
            freqs = self.freqs_cis.index_select(0, positions)
            query = self.indexer.queries(q_lora[0], freqs)
            weights = self.indexer.head_weights(x[0])
            live = torch.div(positions + 1, ratio, rounding_mode="floor")
            rows_per_chunk = indexer_prefill_chunk_rows(
                query.shape[0], n_blocks, self.indexer.prefill_max_logits_bytes
            )
            selected = torch.empty(
                query.shape[0],
                min(self.indexer.index_topk, n_blocks),
                dtype=torch.int32,
                device=x.device,
            )
            published_candidates = (
                torch.empty(
                    query.shape[0], n_blocks, dtype=torch.bool, device=x.device
                )
                if self.plan.is_candidate_source
                else None
            )
            consumed_candidates = (
                shared_attention.candidates.mask if self.plan.uses_candidates else None
            )
            if self.plan.uses_candidates:
                if consumed_candidates is None:
                    raise RuntimeError("candidate consumer ran before the source layer")
                expected = (query.shape[0], n_blocks)
                if consumed_candidates.shape != expected:
                    raise ValueError(
                        f"published candidate mask {consumed_candidates.shape} "
                        f"!= logits {expected}"
                    )

            for row_start in range(0, query.shape[0], rows_per_chunk):
                row_end = min(row_start + rows_per_chunk, query.shape[0])
                rows = slice(row_start, row_end)
                scores = self.attn.indexer_prefill_logits(
                    query[rows].unsqueeze(0),
                    keys.unsqueeze(0),
                    weights[rows].unsqueeze(0),
                )[0]
                if published_candidates is not None:
                    mask = self.indexer.select_candidate_mask(
                        scores, live[rows].unsqueeze(-1)
                    )
                    published_candidates[rows].copy_(mask)
                elif consumed_candidates is not None:
                    mask = consumed_candidates[rows]
                else:
                    mask = None
                selected[rows].copy_(
                    self.indexer.select(scores, live[rows], candidate_mask=mask)
                )
            if published_candidates is not None:
                shared_attention.candidates.mask = published_candidates
        rows = self.attn.blocks_to_global(selected, ratio, ti=table_idx).unsqueeze(0)
        shared_attention.topk_rows = rows
        return rows

    def prefill_single(
        self, x: torch.Tensor, start_pos: int, table_idx: int
    ) -> torch.Tensor:
        """Correctness-first single-request paged prefill/extend."""

        from freetoken.models.deepseek_v4.ops import apply_rotary_emb

        assert self.freqs_cis is not None
        n = x.shape[1]
        end = start_pos + n
        freqs = self.freqs_cis[start_pos:end]
        q_lora, q = self._project_q(x, freqs)
        kv = self._project_window_kv(x, freqs)
        slots = self.attn.window_slots_of(table_idx, start_pos, end)
        self.attn.store_window(kv[0], self.layer_id, slots)

        if start_pos == 0:
            width = min(n, self.window_size)
            query_pos = torch.arange(n, device=x.device).unsqueeze(1)
            candidates = (query_pos - self.window_size + 1).clamp(0) + torch.arange(
                width, device=x.device
            )
            local = torch.where(candidates <= query_pos, candidates, -1).unsqueeze(0)
            window_rows = self.attn.win_cols_to_global(local, slots)
        else:
            lo = max(0, start_pos - self.window_size + 1)
            pool_slots = self.attn.window_slots_of(table_idx, lo, end)
            query_pos = start_pos + torch.arange(n, device=x.device).unsqueeze(1)
            candidates = (
                (query_pos - self.window_size + 1).clamp(min=lo)
                + torch.arange(self.window_size, device=x.device)
            )
            local = torch.where(candidates <= query_pos, candidates - lo, -1).unsqueeze(0)
            window_rows = self.attn.win_cols_to_global(local, pool_slots)

        ratio = self.plan.compress_ratio
        if self.plan.owns_kv:
            self._publish_source_prefill(x, start_pos, table_idx, slots)
        if ratio:
            compressed_rows = self._index_prefill(x, q_lora, start_pos, table_idx)
            topk = torch.cat([window_rows, compressed_rows], -1).int()
        else:
            topk = window_rows.int()
        output = self.attn.attend(
            q,
            self.layer_id,
            topk,
            window_rows.shape[-1],
            self.attn_sink,
            self.softmax_scale,
            has_compression=bool(ratio),
        )
        apply_rotary_emb(output[..., -self.rope_head_dim :], freqs, inverse=True)
        return self._project_output(output)

    def _publish_source_decode(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        window_slots: torch.Tensor,
        rows: torch.Tensor,
        freqs: torch.Tensor,
    ) -> None:
        assert self.compressor is not None and self.indexer is not None
        ratio = self.plan.compress_ratio
        latent, cmp_dest, completed = self.compressor.decode_paged(
            x,
            positions,
            window_slots,
            rows,
            layer_id=self.layer_id,
            backend=self.attn,
        )
        group_pos = (positions + 1 - ratio).clamp_min(0)
        group_freqs = self.freqs_cis.index_select(0, group_pos)
        index_k = self.indexer.index_keys(latent[:, 0], group_freqs)
        idx_dest = self.attn.decode_compress_rows(
            rows,
            positions,
            ratio,
            self.layer_id,
            "idx",
            completed,
        )
        self.attn.scatter_compressed(self.layer_id, "idx", idx_dest, index_k)
        from freetoken.kernel.triton.dsv41 import rope_fp4_roundtrip

        compressed = rope_fp4_roundtrip(
            latent[:, 0],
            group_freqs,
            self.rope_head_dim,
            compressed_kv=True,
        )
        self.attn.scatter_compressed(
            self.layer_id, "attn", cmp_dest, compressed
        )

    def _index_decode(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        rows: torch.Tensor,
        n_stage: int,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        if not self.plan.owns_index:
            if shared_attention.topk_rows is None:
                raise RuntimeError(
                    f"V4.1 layer {self.layer_id} consumed top-k before publication"
                )
            return shared_attention.topk_rows
        ratio = self.plan.compress_ratio
        if n_stage == 0:
            global_rows = torch.empty(
                x.shape[0], 1, 0, dtype=torch.int64, device=x.device
            )
        else:
            assert self.indexer is not None
            query = self.indexer.queries(q_lora[:, 0], freqs)
            weights = self.indexer.head_weights(x[:, 0])
            valid = torch.div(positions + 1, ratio, rounding_mode="floor")
            scores = self.attn.indexer_decode_scores(
                query, weights, valid, n_stage, ratio, self.layer_id
            )
            if self.plan.is_candidate_source:
                shared_attention.candidates.publish(
                    scores,
                    valid.unsqueeze(-1),
                    self.indexer.candidate_topk_blocks,
                    self.indexer.candidate_block_size,
                )
            mask = (
                shared_attention.candidates.mask
                if self.plan.uses_candidates
                else None
            )
            blocks = self.indexer.select(
                scores, valid, candidate_mask=mask
            ).unsqueeze(1)
            global_rows = self.attn.blocks_to_global(
                blocks, ratio, rows=rows
            )
        shared_attention.topk_rows = global_rows
        return global_rows

    def decode_step(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rows: torch.Tensor,
        cmp_stage_cap: int,
        window_ctx=None,
    ) -> torch.Tensor:
        """Eager/captured one-token CSA2 decode over source-owned paged pools."""

        from freetoken.models.deepseek_v4.ops import apply_rotary_emb_decode

        assert self.freqs_cis is not None
        if window_ctx is None:
            window_ctx = get_global_ctx().batch.attn_metadata.window_ctx(
                positions, rows
            )
        window_slots, _previous, window_rows = window_ctx
        freqs = self.freqs_cis.index_select(0, positions)
        q_lora = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_lora).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb_decode(q[..., -self.rope_head_dim :], freqs)
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb_decode(kv[..., -self.rope_head_dim :], freqs)
        act_quant_fp8_inplace(kv, 64)
        self.attn.store_window(
            kv.view(x.shape[0], self.head_dim), self.layer_id, window_slots
        )

        ratio = self.plan.compress_ratio
        cmp_counts = None
        if self.plan.owns_kv:
            self._publish_source_decode(
                x, positions, window_slots, rows, freqs
            )
        if ratio:
            n_stage = (cmp_stage_cap + 1) // ratio
            compressed_rows = self._index_decode(
                x, q_lora, positions, rows, n_stage, freqs
            )
            topk = torch.cat([window_rows, compressed_rows], -1).int()
            cmp_counts = (compressed_rows >= 0).sum(-1).to(torch.int32)
        else:
            topk = window_rows.int()
        output = self.attn.attend(
            q,
            self.layer_id,
            topk,
            self.window_size,
            self.attn_sink,
            self.softmax_scale,
            cmp_counts=cmp_counts,
            has_compression=bool(ratio),
        )
        apply_rotary_emb_decode(
            output[..., -self.rope_head_dim :], freqs, inverse=True
        )
        return self._project_output(output)


class MoEState(nn.Module):
    """Resident router/shared-expert state; routed experts attach at runtime."""

    def __init__(self, args: DeepseekV41Args) -> None:
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(args)
        self.shared_experts = SharedExpert(args)
        self.experts = None
        self.execution = None
        self.partition = None
        self._comm = None

    def attach_routed_experts(self, layer_id: int, args: DeepseekV41Args) -> None:
        from freetoken.distributed import DistributedCommunicator

        from .execution import get_execution_plan
        from .moe import RoutedExperts

        self.execution = get_execution_plan()
        self.partition = self.execution.partition(args.n_routed_experts)
        self._comm = DistributedCommunicator()
        with self.execution.model_tp_context():
            self.experts = RoutedExperts(layer_id, args, self.partition.local_count)
        self.experts.packed_prefill_root = self.execution.backbone_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.experts is None or self.execution is None:
            raise RuntimeError("V4.1 routed experts have not been attached")
        shape = x.shape
        hidden = x.view(-1, self.dim)
        if self.execution.enabled:
            with profile_range("DSV41/EP/HiddenBroadcast"):
                hidden = self._comm.broadcast(
                    hidden.contiguous(), self.execution.backbone_rank
                )
        with profile_range("DSV41/MoE/Router"):
            weights, ids = self.gate(hidden)
        if self.execution.enabled:
            with profile_range("DSV41/EP/RouteBroadcast"):
                weights = self._comm.broadcast(
                    weights.float().contiguous(), self.execution.backbone_rank
                )
                ids = self._comm.broadcast(
                    ids.to(torch.int32).contiguous(), self.execution.backbone_rank
                )
                from freetoken.moe.partition import localize_expert_routes

                weights, ids = localize_expert_routes(weights, ids, self.partition)
        with profile_range("DSV41/MoE/PrepareRouted"):
            self.experts.prepare_packed_prefill_receive(weights, hidden.dtype)
        with profile_range("DSV41/MoE/SharedExpert"):
            shared = self.shared_experts(hidden)
        with profile_range("DSV41/MoE/RoutedExpert"):
            routed = self.experts.routed_forward(
                hidden,
                weights.float().contiguous(),
                ids.to(torch.int32).contiguous(),
            )
        return (shared + routed.to(shared.dtype)).view(shape)


class Block(nn.Module):
    def __init__(
        self,
        args: DeepseekV41Args,
        layer_id: int,
        layout: AttentionLayout,
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.attn = Attention(args, layer_id, layout)
        self.engram = Engram(args, layer_id) if layer_id in args.engram_layer_ids else None
        self.ffn = MoEState(args)
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
        from .hc import hc_mixes

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

    def _engram(self, hidden, engram_rows, token_mask):
        if self.engram is None:
            return hidden
        if engram_rows is None:
            raise RuntimeError(f"Engram rows missing at layer {self.attn.layer_id}")
        with profile_range("DSV41/Engram/Inject"):
            return self.engram(hidden, engram_rows, token_mask)

    @profile("DSV41/Layer_{}/Prefill", layer_id_field="layer_id")
    def prefill_single(
        self,
        hidden: torch.Tensor,
        incoming_pre: torch.Tensor,
        *,
        start_pos: int,
        table_idx: int,
        engram_rows: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .hc import hc_post, hc_pre

        hidden = self._engram(hidden, engram_rows, token_mask)
        residual = hidden
        attn_pre, attn_post, attn_comb = self._mixes(hidden, "attn")
        block_input = self.attn_norm(hc_pre(hidden, incoming_pre))
        with profile_range("DSV41/Attention/Prefill"):
            block_output = self.attn.prefill_single(
                block_input, start_pos, table_idx
            )
        hidden = hc_post(block_output, residual, attn_post, attn_comb)

        residual = hidden
        ffn_pre, ffn_post, ffn_comb = self._mixes(hidden, "ffn")
        block_input = self.ffn_norm(hc_pre(hidden, attn_pre))
        with profile_range("DSV41/MoE/Authority"):
            block_output = self.ffn(block_input)
        return hc_post(block_output, residual, ffn_post, ffn_comb), ffn_pre

    @profile("DSV41/Layer_{}/Decode", layer_id_field="layer_id")
    def decode_step(
        self,
        hidden: torch.Tensor,
        incoming_pre: torch.Tensor,
        *,
        positions: torch.Tensor,
        rows: torch.Tensor,
        cmp_stage_cap: int,
        window_ctx,
        engram_rows: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .hc import hc_post, hc_pre

        hidden = self._engram(hidden, engram_rows, token_mask)
        residual = hidden
        attn_pre, attn_post, attn_comb = self._mixes(hidden, "attn")
        block_input = self.attn_norm(hc_pre(hidden, incoming_pre))
        with profile_range("DSV41/Attention/Decode"):
            block_output = self.attn.decode_step(
                block_input, positions, rows, cmp_stage_cap, window_ctx
            )
        hidden = hc_post(block_output, residual, attn_post, attn_comb)

        residual = hidden
        ffn_pre, ffn_post, ffn_comb = self._mixes(hidden, "ffn")
        block_input = self.ffn_norm(hc_pre(hidden, attn_pre))
        with profile_range("DSV41/MoE/Authority"):
            block_output = self.ffn(block_input)
        return hc_post(block_output, residual, ffn_post, ffn_comb), ffn_pre


class Transformer(nn.Module):
    """Ordinary-generation V4.1 backbone; MTP and vision are intentionally absent."""

    def __init__(self, args: DeepseekV41Args) -> None:
        super().__init__()
        self.args = args
        self.attention_layout = AttentionLayout(args)
        self.embed = nn.Embedding(args.vocab_size, args.dim, dtype=torch.bfloat16)
        self.embed.weight.requires_grad_(False)
        self.layers = nn.ModuleList(
            [Block(args, layer, self.attention_layout) for layer in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Parameter(
            torch.empty(args.vocab_size, args.dim, dtype=torch.float32), requires_grad=False
        )
        # RoutedExperts owns no resident model tensors. Construct the shells with
        # the engine model so the generic offload-cache walker can attach before
        # runtime preparation. Standalone meta tooling has no TP context and only
        # needs the resident parameter contract, so it defers this attachment.
        from freetoken.distributed import get_tp_info

        try:
            get_tp_info()
        except RuntimeError:
            pass
        else:
            self._attach_routed_experts()

    def _attach_routed_experts(self) -> None:
        for layer_id, block in enumerate(self.layers):
            if block.ffn.experts is None:
                block.ffn.attach_routed_experts(layer_id, self.args)

    def attach_runtime(self, device: torch.device) -> None:
        self._attach_routed_experts()
        for block in self.layers:
            block.attn.bind(device)

    def attach_engram_table(self, layer_id: int, table) -> None:
        block = self.layers[layer_id]
        if block.engram is None:
            raise ValueError(f"layer {layer_id} has no Engram module")
        block.engram.attach_table(table)

    def prefill_single(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int,
        table_idx: int,
        engram_rows: dict[int, torch.Tensor] | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from .hc import hc_pre, identity_pre_mix

        shared_attention.reset()
        hidden = self.embed(input_ids).unsqueeze(2).repeat(
            1, 1, self.args.hc_mult, 1
        )
        incoming = identity_pre_mix(hidden, self.args.hc_mult)
        for layer_id, block in enumerate(self.layers):
            hidden, incoming = block.prefill_single(
                hidden,
                incoming,
                start_pos=start_pos,
                table_idx=table_idx,
                engram_rows=(engram_rows or {}).get(layer_id),
                token_mask=token_mask,
            )
        hidden = self.norm(hc_pre(hidden, incoming))
        return F.linear(hidden[:, -1].float(), self.head)

    def decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cmp_stage_cap: int,
        *,
        engram_rows: dict[int, torch.Tensor] | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from .hc import hc_pre, identity_pre_mix

        shared_attention.reset()
        batch = input_ids.shape[0]
        rows = torch.arange(batch, device=input_ids.device)
        window_ctx = get_global_ctx().batch.attn_metadata.window_ctx(
            positions, rows
        )
        hidden = self.embed(input_ids).unsqueeze(2).repeat(
            1, 1, self.args.hc_mult, 1
        )
        incoming = identity_pre_mix(hidden, self.args.hc_mult)
        for layer_id, block in enumerate(self.layers):
            hidden, incoming = block.decode_step(
                hidden,
                incoming,
                positions=positions,
                rows=rows,
                cmp_stage_cap=cmp_stage_cap,
                window_ctx=window_ctx,
                engram_rows=(engram_rows or {}).get(layer_id),
                token_mask=token_mask,
            )
        hidden = self.norm(hc_pre(hidden, incoming))
        return F.linear(hidden[:, -1].float(), self.head)


class _CoordinatedEngramTable:
    """Engram module facade whose lookup enters the EP collective."""

    def __init__(self, coordinator, layer_id: int) -> None:
        self.coordinator = coordinator
        self.layer_id = int(layer_id)

    def prefetch(self, row_ids: torch.Tensor) -> None:
        # The worker must enter the row-id broadcast at this exact layer, so an
        # authority-only early prefetch would reorder collectives.
        return None

    def lookup(self, row_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.coordinator.authority_lookup(self.layer_id, row_ids)


class _ZeroEngramTable:
    """Collective-correct table used by the engine's dummy-weight path."""

    nbytes = 0

    def __init__(self, dim: int, world_size: int) -> None:
        self.dim = int(dim)
        self.world_size = int(world_size)

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None

    def lookup(self, row_ids, *, out=None, reduce=True, communicator=None):
        rows = torch.zeros(
            (*row_ids.shape, self.dim), dtype=torch.bfloat16, device=row_ids.device
        )
        if reduce and self.world_size > 1:
            from freetoken.distributed import DistributedCommunicator

            rows = (communicator or DistributedCommunicator()).all_reduce(rows)
        if out is None:
            return rows
        out.copy_(rows)
        return out


class DeepseekV41ExpertWorkerModel:
    """Only the rank-local routed experts and Engram collective calls."""

    def __init__(self, args: DeepseekV41Args) -> None:
        self.args = args
        self.layers = [MoE(layer_id, args) for layer_id in range(args.n_layers)]
        self.engram_coordinator = None

    def attach_engram_coordinator(self, coordinator) -> None:
        self.engram_coordinator = coordinator

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.engram_coordinator is None:
            raise RuntimeError("V4.1 worker has no Engram tables")
        num_tokens = input_ids.numel()
        device = input_ids.device
        hidden_shape = (num_tokens, self.args.dim)
        for layer_id, layer in enumerate(self.layers):
            with profile_range(f"DSV41/Layer_{layer_id}/ExpertWorker"):
                if layer_id in self.args.engram_layer_ids:
                    with profile_range("DSV41/Engram/Worker"):
                        self.engram_coordinator.worker_lookup(
                            layer_id,
                            num_tokens=num_tokens,
                            hashes_per_token=self.args.engram_hashes_per_token,
                            device=device,
                        )
                with profile_range("DSV41/MoE/Worker"):
                    layer.worker_forward(hidden_shape, device)
        return torch.zeros((1, 1), dtype=torch.float32, device=device)


class DeepseekV41ForCausalLM(BaseLLMModel):
    """Engine adapter for text-only, non-speculative DeepSeek-V4.1 serving."""

    def __init__(self, config) -> None:
        from .execution import get_execution_plan

        self._config = config
        self._args: DeepseekV41Args = config.dsv41_args
        self._execution = get_execution_plan()
        self._model = (
            DeepseekV41ExpertWorkerModel(self._args)
            if self._execution.is_expert_worker
            else Transformer(self._args)
        )
        self._bound = False
        self._engram_hasher = None
        self._engram_tables = None
        self._engram_coordinator = None

    def state_dict(self, *, prefix: str = "", result=None):
        result = {} if result is None else result
        if self._execution.is_expert_worker:
            return result
        for name, parameter in self._model.named_parameters():
            result[f"{prefix}.{name}" if prefix else name] = parameter
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        if self._execution.is_expert_worker:
            if state_dict and not _internal:
                raise RuntimeError(f"Unexpected V4.1 worker weights: {list(state_dict)}")
            return
        casted = {}
        for name, parameter in self._model.named_parameters():
            key = f"{prefix}.{name}" if prefix else name
            if key not in state_dict:
                raise RuntimeError(f"Missing weight for DeepSeek-V4.1 parameter: {key}")
            casted[name] = state_dict.pop(key).to(parameter.dtype)
        if state_dict and not _internal:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict)}")
        self._model.load_state_dict(casted, assign=True, strict=False)

    def _iter_offload_moe_layers(self):
        if self._execution.is_expert_worker:
            for layer in self._model.layers:
                yield layer.experts
            return
        for block in self._model.layers:
            yield block.ffn.experts

    def prepare_for_runtime(self) -> None:
        if not self._execution.is_expert_worker and not self._bound:
            device = self._model.embed.weight.device
            self._model.attach_runtime(device)
            self._bound = True

    def mark_for_rebind(self) -> None:
        self._bound = False

    def load_host_tables(self, engine_config) -> int:
        """Load this rank's two resident Engram row intervals and attach them."""

        if not self._execution.enabled:
            raise RuntimeError(
                "DeepSeek-V4.1 Engram requires heterogeneous EP; pass "
                "--tensor-parallel-size 2 --dsv41-backbone-rank 0"
            )
        from freetoken.checkpoint.ftw import is_ftw_checkpoint

        if is_ftw_checkpoint(engine_config.model_path):
            raise RuntimeError(
                "DeepSeek-V4.1 FTW checkpoints are not supported yet because "
                "Engram tables need rank-sharded raw-checkpoint reads; serve the "
                "original safetensors checkpoint directly"
            )
        from .engram import EngramHasher, EngramLayout, load_engram_host_table
        from .engram_runtime import EngramCoordinator

        rank, world = self._execution.rank, self._execution.world_size
        args = self._args
        if getattr(engine_config, "use_dummy_weight", False):
            tables = {
                layer_id: _ZeroEngramTable(args.engram_head_dim, world)
                for layer_id in args.engram_layer_ids
            }
        else:
            tables = {}
            # One NVMe reader at a time prevents two 94-GiB scans from competing.
            load_order = tuple(r for r in range(world) if r != self._execution.backbone_rank)
            load_order += (self._execution.backbone_rank,)
            for load_rank in load_order:
                if rank == load_rank:
                    for layer_id, rows in zip(
                        args.engram_layer_ids, args.engram_num_embeddings
                    ):
                        tables[layer_id] = load_engram_host_table(
                            engine_config.model_path,
                            layer_id=layer_id,
                            num_embeddings=rows,
                            dim=args.engram_head_dim,
                            rank=rank,
                            world_size=world,
                        )
                torch.distributed.barrier()

        coordinator = EngramCoordinator(tables, execution=self._execution)
        self._engram_tables = tables
        self._engram_coordinator = coordinator
        if self._execution.is_expert_worker:
            self._model.attach_engram_coordinator(coordinator)
        else:
            layout = EngramLayout.build(
                layer_ids=args.engram_layer_ids,
                num_embeddings=args.engram_num_embeddings,
                max_ngram_size=args.engram_max_ngram_size,
                n_heads=args.engram_n_heads,
                head_dim=args.engram_head_dim,
                vocab_size=args.engram_vocab_size,
            )
            from freetoken.utils import load_tokenizer

            self._engram_hasher = EngramHasher(
                layout,
                load_tokenizer(engine_config.model_path),
                pad_token_id=args.engram_pad_id,
                compressed_vocab_size=args.engram_compressed_vocab_size,
            ).to(self._model.embed.weight.device)
            for layer_id in args.engram_layer_ids:
                self._model.attach_engram_table(
                    layer_id, _CoordinatedEngramTable(coordinator, layer_id)
                )
        return sum(table.nbytes for table in tables.values())

    @profile("DSV41/Engram/Hash")
    def _engram_rows(self, batch, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        if self._engram_hasher is None:
            raise RuntimeError("V4.1 Engram hasher was not initialized")
        reqs = batch.padded_reqs
        if len(reqs) != 1:
            raise RuntimeError("V4.1 ordinary-generation baseline supports one stream")
        ctx = self._args.engram_max_ngram_size - 1
        pin = {"pin_memory": torch.cuda.is_available()}
        history = torch.full(
            (1, ctx), self._args.engram_pad_id, dtype=torch.int64, **pin
        )
        req = reqs[0]
        prefix = req.input_ids[max(0, req.cached_len - ctx) : req.cached_len].long()
        if prefix.numel():
            history[0, -prefix.numel() :] = prefix
        history = history.to(input_ids.device, non_blocking=True)
        cu = torch.tensor(
            [0, input_ids.numel()], dtype=torch.int32, **pin
        ).to(input_ids.device, non_blocking=True)
        all_rows = self._engram_hasher.row_ids(
            input_ids.view(-1), batch.positions.view(-1), cu, history
        )
        return {
            layer_id: all_rows[:, index]
            for index, layer_id in enumerate(self._args.engram_layer_ids)
        }

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        if self._execution.is_expert_worker:
            with profile_range("DSV41/Batch/ExpertWorker"):
                return self._model.forward(batch.input_ids)
        self.prepare_for_runtime()
        input_ids = batch.input_ids.long()
        engram_rows = self._engram_rows(batch, input_ids)
        if batch.is_prefill:
            req = batch.reqs[0]
            with profile_range("DSV41/Batch/Prefill"):
                return self._model.prefill_single(
                    input_ids.view(1, -1),
                    start_pos=req.cached_len,
                    table_idx=req.table_idx,
                    engram_rows=engram_rows,
                )
        positions = batch.positions.long().view(-1)[: batch.padded_size]
        with profile_range("DSV41/Batch/Decode"):
            return self._model.decode(
                input_ids.view(batch.padded_size, 1),
                positions,
                int(positions.max().item()),
                engram_rows=engram_rows,
            )


__all__ = [
    "Attention",
    "Block",
    "DeepseekV41ExpertWorkerModel",
    "DeepseekV41ForCausalLM",
    "Indexer",
    "Transformer",
]
