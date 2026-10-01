"""Engine-facing configuration for DeepSeek-V4.1-Flash."""

from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    DSV4AttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    vision_load_enabled,
)

from .args import load_args


def parse_config(hf_config: Any) -> ModelConfig:
    model_path = getattr(hf_config, "_name_or_path", None) or getattr(
        hf_config, "name_or_path", None
    )
    if not model_path:
        raise ValueError("DeepSeek-V4.1 config needs the local checkpoint path")
    args = load_args(model_path, max_batch_size=1)
    text = getattr(hf_config, "text_config", hf_config)
    max_position = int(
        getattr(text, "max_position_embeddings", 0)
        or args.original_seq_len * args.rope_factor
    )
    rope_scaling = {
        "rope_type": "yarn",
        "factor": args.rope_factor,
        "beta_fast": args.beta_fast,
        "beta_slow": args.beta_slow,
        "original_max_position_embeddings": args.original_seq_len,
    }
    vision_config = getattr(hf_config, "vision_config", None)
    if not vision_load_enabled():
        vision_config = None
    return ModelConfig(
        num_layers=args.n_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,
        head_dim=args.head_dim,
        hidden_size=args.dim,
        vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim,
        rms_norm_eps=args.norm_eps,
        rotary_config=RotaryConfig(
            head_dim=args.head_dim,
            rotary_dim=args.rope_head_dim,
            max_position=max_position,
            base=args.rope_theta,
            scaling=rope_scaling,
        ),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=args.norm_topk_prob,
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        moe_enabled=True,
        expert_quant="ds_fp4",
        attn_sm_scale=args.head_dim**-0.5,
        swiglu_limit=args.swiglu_limit,
        vision_config=vision_config,
        # Every EP rank needs the placeholder id for deterministic radix keys and
        # Engram masking, even though only the authority builds the vision tower.
        # ``vision_config`` remains the sole switch for allocating/loading vision.
        image_token_id=args.image_token_id,
        dsv4_args=args,
        dsv41_args=args,
        single_stream_only=True,
        attention_groups=(
            DSV4AttentionGroupConfig(
                name="dsv41",
                layer_ids=tuple(range(args.n_layers)),
                num_kv_heads=1,
                head_dim=args.head_dim,
                sliding_window=args.window_size,
            ),
        ),
    )


__all__ = ["parse_config"]
