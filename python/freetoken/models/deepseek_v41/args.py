"""DeepSeek-V4.1-Flash hyperparameters from the official inference config."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Literal


@dataclass
class DeepseekV41Args:
    # Runtime overlays.
    max_batch_size: int = 1
    max_seq_len: int = 4096
    temperature: float = 1.0
    dtype: Literal["bf16", "fp8"] = "fp8"
    expert_dtype: Literal[None, "fp4"] = "fp4"
    weight_block_size: tuple[int, int] = (32, 32)
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"

    # Text model.
    vocab_size: int = 129_280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_mtp_layers: int = 3
    n_heads: int = 64

    # Routed/shared experts.
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["sqrtsoftplus"] = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    # MLA + CSA2.
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple[int, ...] = ()
    kv_source_layers: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    compress_rope_theta: float = 160_000.0

    # RoPE/YaRN.
    original_seq_len: int = 65_536
    rope_theta: float = 10_000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1

    # Hierarchical lightning indexer.
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    # Manifold-constrained Hyper-Connections.
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # Engram.
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_vocab_size: int = 16_000_000
    engram_num_embeddings: tuple[int, ...] = (384_006_168, 384_016_682)
    engram_max_ngram_size: int = 4
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99_092
    engram_n_heads: int = 8
    engram_head_dim: int = 256

    # DSpark/MTP (kept disabled until ordinary generation passes).
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128_799
    dspark_target_layer_ids: tuple[int, ...] = (37, 38, 39)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_n_activated_experts: int = 3

    # Vision geometry is parsed but the first serving gate is text-only.
    vision_n_layers: int = 32
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 295_936
    vision_max_wh_ratio: float | None = None
    vision_rope_theta: float = 10_000.0
    image_token_id: int = 129_264

    def __post_init__(self) -> None:
        for name in (
            "weight_block_size",
            "compress_ratios",
            "kv_source_layers",
            "index_source_layers",
            "engram_layer_ids",
            "engram_num_embeddings",
            "dspark_target_layer_ids",
        ):
            value = getattr(self, name)
            if isinstance(value, list):
                setattr(self, name, tuple(value))
        if self.weight_block_size != (32, 32):
            raise ValueError(f"DeepSeek-V4.1 requires 32x32 FP8 blocks, got {self.weight_block_size}")
        if len(self.compress_ratios) < self.n_layers:
            raise ValueError("compress_ratios does not cover every main-model layer")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("Engram layer/table counts differ")

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def engram_hashes_per_token(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_heads

    def engram_table_bytes(self, rank: int = 0, world_size: int = 1) -> int:
        """Exact FP8 payload + per-32 E8M0 bytes owned by one rank."""

        row_bytes = self.engram_head_dim + self.engram_head_dim // self.weight_block_size[1]
        return sum(
            (rows * (rank + 1) // world_size - rows * rank // world_size) * row_bytes
            for rows in self.engram_num_embeddings
        )

    def expert_bank_bytes(self, rank: int = 0, world_size: int = 1) -> int:
        """Exact main-model FP4 expert-bank bytes owned by one contiguous EP shard."""

        lo = self.n_routed_experts * rank // world_size
        hi = self.n_routed_experts * (rank + 1) // world_size
        experts = hi - lo
        h, inter = self.dim, self.moe_inter_dim
        per_expert_layer = (
            2 * inter * (h // 2)
            + 2 * inter * (h // 32)
            + h * (inter // 2)
            + h * (inter // 32)
        )
        return self.n_layers * experts * per_expert_layer


def _config_path(model_path: str) -> str:
    candidates = (
        os.path.join(model_path, "inference", "config.json"),
        os.path.join(model_path, "config.json"),
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No DeepSeek-V4.1 config found under {model_path}")


def load_args(model_path: str, **overrides) -> DeepseekV41Args:
    """Read the official inference dialect and overlay top-level quant metadata."""

    path = _config_path(model_path)
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if "text_config" in raw:
        top_level = raw
        text = dict(raw["text_config"])
        quant = top_level.get("quantization_config") or {}
        aliases = {
            "hidden_size": "dim",
            "moe_intermediate_size": "moe_inter_dim",
            "num_hidden_layers": "n_layers",
            "num_attention_heads": "n_heads",
            "num_experts_per_tok": "n_activated_experts",
            "scoring_func": "score_func",
            "routed_scaling_factor": "route_scale",
            "qk_rope_head_dim": "rope_head_dim",
            "rms_norm_eps": "norm_eps",
            "sliding_window": "window_size",
            "kv_source_layer_ids": "kv_source_layers",
            "index_source_layer_ids": "index_source_layers",
            "candidate_source_layer_id": "candidate_source_layer",
            "engram_pad_token_id": "engram_pad_id",
            "num_nextn_predict_layers": "n_mtp_layers",
            "dspark_num_experts_per_tok": "dspark_n_activated_experts",
        }
        raw = {aliases.get(key, key): value for key, value in text.items()}
        raw.update(
            dtype=str(top_level.get("dtype", "fp8")).replace("bfloat16", "bf16"),
            expert_dtype=quant.get("expert_dtype", "fp4"),
            weight_block_size=quant.get("weight_block_size", (32, 32)),
            scale_fmt=quant.get("scale_fmt", "ue8m0"),
        )
    valid = {field.name for field in fields(DeepseekV41Args)}
    kwargs = {key: value for key, value in raw.items() if key in valid}
    kwargs.update(overrides)
    return DeepseekV41Args(**kwargs)


__all__ = ["DeepseekV41Args", "load_args"]
