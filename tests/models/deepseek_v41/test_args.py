from __future__ import annotations

import json
import os

import pytest

from freetoken.models.deepseek_v41.args import load_args


MODEL_PATH = os.environ.get("FREETOKEN_DSV41_MODEL_PATH", "")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="official checkpoint absent")
def test_official_checkpoint_args_and_memory_arithmetic():
    args = load_args(MODEL_PATH)
    assert (args.dim, args.n_layers, args.n_routed_experts) == (5120, 40, 384)
    assert args.weight_block_size == (32, 32)
    assert args.compress_ratios[:3] == (0, 0, 2)
    assert args.compress_ratios[20:22] == (1, 1)
    assert args.kv_source_layers == (2, 8, 14, 20)
    assert args.index_source_layers == (2, 8, 14, 20, 24, 28, 32, 36)
    assert args.engram_hashes_per_token == 24
    assert args.engram_table_bytes() == 202_758_032_400
    assert args.engram_table_bytes(rank=0, world_size=2) == 101_379_016_200
    assert args.engram_table_bytes(rank=1, world_size=2) == 101_379_016_200
    assert args.expert_bank_bytes(rank=0, world_size=2) == 144_388_915_200


def test_top_level_hf_dialect_aliases(tmp_path):
    config = {
        "dtype": "bfloat16",
        "quantization_config": {
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "vocab_size": 100,
            "hidden_size": 64,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "qk_rope_head_dim": 16,
            "rms_norm_eps": 1e-6,
            "sliding_window": 128,
            "compress_ratios": [0, 0],
            "engram_layer_ids": [],
            "engram_num_embeddings": [],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = load_args(str(tmp_path), max_seq_len=8192)
    assert args.dtype == "bf16"
    assert (args.dim, args.moe_inter_dim, args.n_layers, args.n_heads) == (64, 32, 2, 4)
    assert args.n_activated_experts == 2
    assert args.rope_head_dim == 16
    assert args.max_seq_len == 8192
