"""Native mixed-Q2 GLM expert kernels over synthetic GGUF block rows."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _valid_iq_rows(shape: tuple[int, ...], block_bytes: int, scale: float) -> torch.Tensor:
    raw = torch.randint(0, 256, shape, dtype=torch.uint8)
    blocks = raw.view(*shape[:-1], -1, block_bytes)
    d = torch.tensor([scale], dtype=torch.float16).view(torch.uint8)
    blocks[..., :2] = d
    return raw.cuda()


def _valid_iq_rows_pinned(
    shape: tuple[int, ...], block_bytes: int, scale: float
) -> torch.Tensor:
    raw = torch.randint(0, 256, shape, dtype=torch.uint8).pin_memory()
    blocks = raw.view(*shape[:-1], -1, block_bytes)
    d = torch.tensor([scale], dtype=torch.float16).view(torch.uint8)
    blocks[..., :2] = d
    return raw


def test_glm_q2_k_xl_moe_matches_dense_dequant_reference():
    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.models.gguf.dequant import (
        GGML_IQ2_XS,
        GGML_IQ3_XXS,
        row_bytes,
    )
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_q2_k_xl

    torch.manual_seed(52)
    experts, hidden, intermediate = 4, 256, 256
    tokens, top_k = 2, 2
    gate_up = _valid_iq_rows(
        (experts, 2 * intermediate, row_bytes(hidden, GGML_IQ2_XS)), 74, 0.02
    )
    down = _valid_iq_rows(
        (experts, hidden, row_bytes(intermediate, GGML_IQ3_XXS)), 98, 0.02
    )
    x = (torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32, device="cuda")
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32, device="cuda")

    actual = fused_experts_gguf_q2_k_xl(x, gate_up, down, weights, ids, "silu")

    gate_dense = ggml_dequantize(
        gate_up.view(experts * 2 * intermediate, -1),
        GGML_IQ2_XS,
        experts * 2 * intermediate,
        hidden,
        torch.bfloat16,
    ).view(experts, 2 * intermediate, hidden)
    down_dense = ggml_dequantize(
        down.view(experts * hidden, -1),
        GGML_IQ3_XXS,
        experts * hidden,
        intermediate,
        torch.bfloat16,
    ).view(experts, hidden, intermediate)
    expected = torch.zeros_like(actual)
    for token in range(tokens):
        for route in range(top_k):
            expert = int(ids[token, route])
            gu = F.linear(x[token], gate_dense[expert])
            inter = F.silu(gu[:intermediate]) * gu[intermediate:]
            expected[token] += F.linear(inter, down_dense[expert]) * weights[token, route]

    rel = (actual - expected).abs().max() / (expected.abs().max() + 1e-6)
    assert rel < 0.12, f"mixed-Q2 MMVQ relative error {rel.item()}"


def test_glm_q2_k_xl_format_geometry():
    from freetoken.models.gguf.dequant import (
        GGML_IQ2_XS,
        GGML_IQ3_XXS,
        row_bytes,
    )
    from freetoken.moe.offload_cache import _BANK_SCHEMAS

    assert row_bytes(6144, GGML_IQ2_XS) == 1776
    assert row_bytes(2048, GGML_IQ3_XXS) == 784
    assert _BANK_SCHEMAS["gguf_q2_k_xl"] == ("gate_up", "down")
    assert _BANK_SCHEMAS["gguf_glm_iq"] == ("gate_up", "down")


def test_iq1_layers_share_one_max_stride_offload_cache():
    from freetoken.models.gguf.dequant import (
        GGML_IQ1_S,
        GGML_IQ2_XXS,
        GGML_IQ3_XXS,
        row_bytes,
    )
    from freetoken.moe.fused_q4_0 import fused_experts_gguf
    from freetoken.moe.offload_cache import OffloadMoeCache

    experts, hidden, intermediate = 2, 256, 256
    gu_iq1 = _valid_iq_rows_pinned(
        (experts, 2 * intermediate, row_bytes(hidden, GGML_IQ1_S)), 50, 0.02
    )
    gu_iq2 = _valid_iq_rows_pinned(
        (experts, 2 * intermediate, row_bytes(hidden, GGML_IQ2_XXS)), 66, 0.02
    )
    down = _valid_iq_rows_pinned(
        (experts, hidden, row_bytes(intermediate, GGML_IQ3_XXS)), 98, 0.02
    )
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=experts,
        cache_size=experts,
        device=torch.device("cuda"),
        quant_format="gguf_glm_iq",
    )
    cache.set_bank_sources(
        {"gate_up": [gu_iq1, gu_iq2], "down": [down, down.clone()]}
    )
    cache.materialize_layer(0)
    cache.copy_missing()
    torch.cuda.synchronize()
    gu_cache, down_cache = cache.bank_views(experts)

    x = torch.randn(1, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    ids = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    weights = torch.ones((1, 1), dtype=torch.float32, device="cuda")
    expected = fused_experts_gguf(
        x,
        gu_iq1.cuda(),
        down.cuda(),
        weights,
        ids,
        "silu",
        gate_up_type=GGML_IQ1_S,
        down_type=GGML_IQ3_XXS,
    )
    actual = fused_experts_gguf(
        x,
        gu_cache,
        down_cache,
        weights,
        ids,
        "silu",
        gate_up_type=GGML_IQ1_S,
        down_type=GGML_IQ3_XXS,
        intermediate_size=intermediate,
    )
    torch.testing.assert_close(actual, expected)


def test_mixed_q2_layers_share_one_max_stride_offload_cache():
    from freetoken.models.gguf.dequant import (
        GGML_IQ2_XS,
        GGML_IQ3_XXS,
        GGML_IQ4_XS,
        row_bytes,
    )
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_q2_k_xl
    from freetoken.moe.offload_cache import OffloadMoeCache

    experts, hidden, intermediate = 2, 256, 256
    gu_common = _valid_iq_rows_pinned(
        (experts, 2 * intermediate, row_bytes(hidden, GGML_IQ2_XS)), 74, 0.02
    )
    gu_special = _valid_iq_rows_pinned(
        (experts, 2 * intermediate, row_bytes(hidden, GGML_IQ3_XXS)), 98, 0.02
    )
    down_common = _valid_iq_rows_pinned(
        (experts, hidden, row_bytes(intermediate, GGML_IQ3_XXS)), 98, 0.02
    )
    down_special = _valid_iq_rows_pinned(
        (experts, hidden, row_bytes(intermediate, GGML_IQ4_XS)), 136, 0.02
    )
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=experts,
        cache_size=experts,
        device=torch.device("cuda"),
        quant_format="gguf_q2_k_xl",
    )
    cache.set_bank_sources(
        {
            "gate_up": [gu_common, gu_special],
            "down": [down_common, down_special],
        }
    )
    assert cache.bank_caches["gate_up"].shape == (
        experts,
        (gu_special[0].numel() + 399) // 400 * 400,
    )
    assert cache.bank_caches["down"].shape == (
        experts,
        (down_special[0].numel() + 783) // 784 * 784,
    )
    assert cache.bank_caches["gate_up"].stride(0) % 50 == 0
    assert cache.bank_caches["down"].stride(0) % 98 == 0
    assert cache.bank_caches["gate_up"].stride(0) % 16 == 0
    assert cache.bank_caches["down"].stride(0) % 16 == 0

    cache.materialize_layer(1)
    cache.copy_missing()
    torch.cuda.synchronize()
    gu_cache, down_cache = cache.bank_views(experts)
    assert torch.equal(
        gu_cache[:, : gu_special[0].numel()].cpu(), gu_special.reshape(experts, -1)
    )
    assert torch.equal(
        down_cache[:, : down_special[0].numel()].cpu(),
        down_special.reshape(experts, -1),
    )

    x = torch.randn(1, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    ids = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    weights = torch.ones((1, 1), dtype=torch.float32, device="cuda")
    expected = fused_experts_gguf_q2_k_xl(
        x,
        gu_special.cuda(),
        down_special.cuda(),
        weights,
        ids,
        "silu",
        gate_up_type=GGML_IQ3_XXS,
        down_type=GGML_IQ4_XS,
    )
    actual = fused_experts_gguf_q2_k_xl(
        x,
        gu_cache,
        down_cache,
        weights,
        ids,
        "silu",
        gate_up_type=GGML_IQ3_XXS,
        down_type=GGML_IQ4_XS,
        intermediate_size=intermediate,
    )
    torch.testing.assert_close(actual, expected)

    from freetoken.layers.moe import OffloadMoELayer

    layer = object.__new__(OffloadMoELayer)
    layer.layer_id = 1
    layer.hidden_size = hidden
    layer.intermediate_size = intermediate
    layer.activation = "silu"
    dispatched = layer._expert_gemm(
        cache,
        x,
        weights,
        ids,
        views=(gu_cache, down_cache),
        n=experts,
        alphas=None,
        is_prefill=True,
    )
    torch.testing.assert_close(dispatched, expected)


def test_glm_q2_loader_materializes_only_stage_layers(monkeypatch):
    from types import SimpleNamespace

    from freetoken.models.gguf.dequant import (
        GGML_IQ2_XS,
        GGML_IQ3_XXS,
        row_bytes,
    )
    from freetoken.models.gguf import reader
    from freetoken.models.glm_moe_dsa.weight import load_gguf_q2_k_xl_expert_sources

    experts = 2
    hidden = intermediate = 256
    h_bytes = row_bytes(hidden, GGML_IQ2_XS)
    i_bytes = row_bytes(intermediate, GGML_IQ3_XXS)

    class Tensor:
        def __init__(self, name: str, ggml_type: int, packed: torch.Tensor):
            self.name = name
            self.ggml_type = ggml_type
            self._packed = packed
            self.cache_drops = 0

        def packed(self) -> torch.Tensor:
            return self._packed

        def drop_cache(self) -> None:
            self.cache_drops += 1

    tensors = []
    for layer in (3, 4, 5):
        tensors.extend(
            [
                Tensor(
                    f"blk.{layer}.ffn_gate_exps.weight",
                    GGML_IQ2_XS,
                    torch.full((experts * intermediate, h_bytes), 10 + layer, dtype=torch.uint8),
                ),
                Tensor(
                    f"blk.{layer}.ffn_up_exps.weight",
                    GGML_IQ2_XS,
                    torch.full((experts * intermediate, h_bytes), 20 + layer, dtype=torch.uint8),
                ),
                Tensor(
                    f"blk.{layer}.ffn_down_exps.weight",
                    GGML_IQ3_XXS,
                    torch.full((experts * hidden, i_bytes), 30 + layer, dtype=torch.uint8),
                ),
            ]
        )
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda path: iter(tensors))
    config = SimpleNamespace(
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        num_layers=6,
        first_k_dense_replace=3,
        local_layer_ids=(4, 5),
    )
    completed = []

    banks = load_gguf_q2_k_xl_expert_sources(
        "unused.gguf", config, layer_sink=lambda layer, layer_banks: completed.append(layer)
    )

    assert completed == [0, 1]
    assert len(banks["gate_up"]) == len(banks["down"]) == 2
    assert banks["gate_up"][0][0, 0, 0].item() == 14
    assert banks["gate_up"][0][0, intermediate, 0].item() == 24
    assert banks["down"][1][0, 0, 0].item() == 35
    assert sum(t.cache_drops for t in tensors) == 6


def test_split_gguf_paths_require_complete_set(tmp_path):
    from freetoken.models.gguf.reader import gguf_shard_paths

    paths = [tmp_path / f"model-{part:05d}-of-00003.gguf" for part in range(1, 4)]
    for path in paths:
        path.touch()

    assert gguf_shard_paths(str(paths[1])) == tuple(str(path) for path in paths)
    paths[-1].unlink()
    with pytest.raises(FileNotFoundError, match="missing 1/3 shards"):
        gguf_shard_paths(str(paths[0]))
