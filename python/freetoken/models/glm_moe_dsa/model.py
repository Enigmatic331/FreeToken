from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import GlmMoeDsaAttention
from .mlp import GlmDsaGatedMLP
from .moe import GlmMoeDsaSparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class GlmFp8LMHead(ParallelLMHead):
    """W8A16 lm_head (fp8-e4m3 weight + per-row scale, quantized at load).

    The full-vocab logits GEMV reads the whole ~1.9 GiB bf16 head every decode step;
    fp8 halves that. Selected by ``ModelConfig.lm_head_quant == "fp8_pertensor"``
    (weight.py quantizes at load off the same field). GLM-5.2 does not tie embeddings,
    so the fp8 weight is head-only.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim, tie_word_embeddings=False)
        self.weight = torch.empty(num_embeddings, embedding_dim, dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty(num_embeddings, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fp8_pertensor_linear(x, self.weight, self.weight_scale)


class GlmFullEmbedding(BaseOP):
    """Unsharded embedding for pipeline rank 0 (no vocab collective)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import indexing

        return indexing(weights=self.weight, indices=x, vocab_range=None)


class GlmFullLMHead(BaseOP):
    """Unsharded bf16 fallback head for the final pipeline rank."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return F.linear(x, self.weight)


class GlmMoeDsaDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = GlmMoeDsaAttention(config, layer_id)
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = GlmMoeDsaSparseBlock(config, layer_id)
        else:
            self.mlp = GlmDsaGatedMLP(
                config.hidden_size, config.intermediate_size, quant=config.dense_quant
            )
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class GlmMoeDsaModel(BaseOP):
    def __init__(self, config: ModelConfig):
        from .execution import glm_pipeline_plan

        self._plan = glm_pipeline_plan(config.num_layers, config.glm_dsa_args.indexer_types)
        self._hidden_size = config.hidden_size
        self._comm = DistributedCommunicator()
        if self._plan.is_first:
            self.embed_tokens: BaseOP | None = (
                GlmFullEmbedding(config.vocab_size, config.hidden_size)
                if self._plan.enabled
                else VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            )
        else:
            self.embed_tokens = None
        layer_ids = self._plan.layer_ids
        self.layers = OPList(
            [GlmMoeDsaDecoderLayer(config, layer_id) for layer_id in layer_ids],
            start_index=self._plan.start_layer,
        )
        self.norm = (
            RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
            if self._plan.is_last
            else None
        )

    def _run_local(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = get_global_ctx()
        cache = ctx.moe_offload_cache
        phase = "prefill" if ctx.batch.is_prefill else "decode"
        if cache is not None:
            cache.profile_begin(f"{phase}_stage")
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        if cache is not None:
            cache.profile_end(f"{phase}_stage")
        assert residual is not None
        return x, residual

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self._plan.enabled:
            assert self.embed_tokens is not None and self.norm is not None
            x = self.embed_tokens.forward(input_ids)
            x, residual = self._run_local(x, None)
            return self.norm.forward(x, residual)[0]

        rows = input_ids.numel()
        if self._plan.is_first:
            assert self.embed_tokens is not None
            x, residual = self._run_local(self.embed_tokens.forward(input_ids), None)
        else:
            dtype = self.layers.op_list[0].input_layernorm.weight.dtype
            x = torch.empty(rows, self._hidden_size, dtype=dtype, device=input_ids.device)
            residual = torch.empty_like(x)

        # Every rank participates in every boundary collective. The next rank runs
        # its local block after receiving both halves of the fused-residual stream.
        for src in range(self._plan.world_size - 1):
            cache = get_global_ctx().moe_offload_cache
            phase = "prefill" if get_global_ctx().batch.is_prefill else "decode"
            if cache is not None:
                cache.profile_begin(f"{phase}_boundary", src)
            x = self._comm.broadcast(x, src)
            residual = self._comm.broadcast(residual, src)
            if cache is not None:
                cache.profile_end(f"{phase}_boundary", src)
            if self._plan.rank == src + 1:
                x, residual = self._run_local(x, residual)

        if self._plan.is_last:
            assert self.norm is not None
            return self.norm.forward(x, residual)[0]
        return x


class GlmMoeDsaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        from .execution import glm_pipeline_plan

        self._plan = glm_pipeline_plan(config.num_layers, config.glm_dsa_args.indexer_types)
        self.model = GlmMoeDsaModel(config)
        if self._plan.enabled and config.tie_word_embeddings:
            raise ValueError("GLM pipeline currently requires untied embeddings")
        if not self._plan.is_last:
            self.lm_head: BaseOP | None = None
        elif config.lm_head_quant == "fp8_pertensor" and not config.tie_word_embeddings:
            self.lm_head: BaseOP = GlmFp8LMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
        elif self._plan.enabled:
            self.lm_head = GlmFullLMHead(config.vocab_size, config.hidden_size)
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing hook (engine calls it before the pool family's solve_num_pages):
        materialize every layer's bmm-ready kv_b split and free the checkpoint-layout
        originals, so the ~2.2 GiB repack is measured by the sizing pass instead of
        overcommitting the KV budget on the first forward (gpt_oss precedent)."""
        import torch

        for layer in self.model.layers.op_list:
            layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        if self.lm_head is None:
            return output.new_empty((get_global_ctx().batch.size, 1))
        return self.lm_head.forward(output)


__all__ = ["GlmMoeDsaForCausalLM"]
