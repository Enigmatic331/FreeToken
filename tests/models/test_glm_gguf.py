from types import SimpleNamespace


def _shim():
    p = "glm-dsa."
    metadata = {
        p + "block_count": 79,
        p + "nextn_predict_layers": 1,
        p + "leading_dense_block_count": 3,
        p + "attention.key_length_mla": 256,
        p + "rope.dimension_count": 64,
        p + "embedding_length": 6144,
        p + "vocab_size": 154880,
        p + "feed_forward_length": 12288,
        p + "expert_feed_forward_length": 2048,
        p + "attention.layer_norm_rms_epsilon": 1e-5,
        p + "attention.head_count": 64,
        p + "expert_count": 256,
        p + "expert_used_count": 8,
        p + "expert_shared_count": 1,
        p + "expert_weights_norm": True,
        p + "expert_weights_scale": 2.5,
        p + "expert_group_count": 1,
        p + "expert_group_used_count": 1,
        p + "attention.q_lora_rank": 2048,
        p + "attention.kv_lora_rank": 512,
        p + "attention.value_length_mla": 256,
        p + "context_length": 1048576,
        p + "rope.freq_base": 8000000.0,
        p + "attention.indexer.head_count": 32,
        p + "attention.indexer.key_length": 128,
        p + "attention.indexer.top_k": 2048,
    }
    return SimpleNamespace(
        metadata=metadata,
        architectures=["GlmMoeDsaGGUFForCausalLM"],
        vocab_size=154880,
        tie_word_embeddings=False,
    )


def test_glm_gguf_config_excludes_mtp_and_expands_indexshare():
    from freetoken.models.glm_moe_dsa.config import parse_gguf_config

    config = parse_gguf_config(_shim())
    assert config.num_layers == 78
    assert config.num_moe_layers == 75
    assert config.expert_quant == "gguf_q2_k_xl"
    assert config.glm_dsa_args.qk_nope_head_dim == 192
    full = [
        i for i, kind in enumerate(config.glm_dsa_args.indexer_types) if kind == "full"
    ]
    assert full[:7] == [0, 1, 2, 6, 10, 14, 18]
    assert full[-1] == 74


def test_q2_k_xl_exception_profile_matches_release():
    from freetoken.models.gguf.dequant import GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ4_XS
    from freetoken.models.glm_moe_dsa.weight import _gguf_q2_k_xl_types

    assert _gguf_q2_k_xl_types(7) == (GGML_IQ2_XS, GGML_IQ3_XXS)
    assert _gguf_q2_k_xl_types(8) == (GGML_IQ3_XXS, GGML_IQ4_XS)
    assert _gguf_q2_k_xl_types(75) == (GGML_IQ2_XS, GGML_IQ4_XS)
    assert _gguf_q2_k_xl_types(77) == (GGML_IQ2_XS, GGML_IQ4_XS)


def test_glm_gguf_uses_embedded_gpt2_bpe_converter():
    from freetoken.models.gguf.tokenizer import _TOKENIZER_ARCH

    assert _TOKENIZER_ARCH["glm-dsa"] == "gpt2"
