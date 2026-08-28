from __future__ import annotations

import pytest
import torch


def test_llama_iq_mmq_is_opt_in(monkeypatch):
    from freetoken.kernel.llama_iq_mmq import supported

    monkeypatch.delenv("FREETOKEN_LLAMA_CPP_DIR", raising=False)
    assert not supported(19, 4096)
    monkeypatch.setenv("FREETOKEN_LLAMA_CPP_DIR", "/tmp/llama.cpp")
    assert supported(19, 4096)
    assert supported(18, 7)
    assert not supported(19, 6)
    assert not supported(17, 4096)


def test_llama_iq_mmq_checkout_validation(monkeypatch, tmp_path):
    from freetoken.kernel import llama_iq_mmq

    monkeypatch.setenv("FREETOKEN_LLAMA_CPP_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="built llama.cpp checkout"):
        llama_iq_mmq._checkout()


def test_llama_iq_mmq_chunks_route_sort_without_concatenating(monkeypatch):
    from freetoken.kernel import llama_iq_mmq

    calls = []

    class FakeModule:
        def grouped_iq_mmq_out(self, weight, x, ids, quant_type, rows, out):
            calls.append((x.shape[0], ids.shape, out.shape))
            out.zero_()
            return out

    monkeypatch.setattr(llama_iq_mmq, "_module", lambda: FakeModule())
    x = torch.zeros(8200, 16, dtype=torch.bfloat16)
    ids = torch.zeros(8200, 1, dtype=torch.int32)
    weight = torch.zeros(4, 32, dtype=torch.uint8)
    out = llama_iq_mmq.grouped_iq_mmq(weight, x, ids, 19, 8)
    assert out.shape == (8200, 8)
    assert out.dtype == torch.bfloat16
    assert [call[0] for call in calls] == [4096, 4096, 8]


def test_single_token_decode_keeps_both_projections_on_mmvq(monkeypatch):
    from freetoken.moe import fused_q4_0

    calls = []

    def fake_mmvq(x, weight, ids, top_k, quant_type, rows, tokens):
        calls.append((top_k, quant_type, rows, tokens))
        return torch.zeros(tokens * top_k, rows, dtype=x.dtype)

    monkeypatch.setattr("freetoken.kernel.gguf.ggml_moe_a8_vec", fake_mmvq)
    monkeypatch.setitem(fused_q4_0._ACT, "silu", lambda gate_up: gate_up[:, :8])
    monkeypatch.setenv("FREETOKEN_LLAMA_CPP_DIR", "/tmp/llama.cpp")
    x = torch.zeros(1, 16, dtype=torch.bfloat16)
    gate_up = torch.zeros(4, 32, dtype=torch.uint8)
    down = torch.zeros(4, 32, dtype=torch.uint8)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.5]])
    fused_q4_0.fused_experts_gguf(
        x, gate_up, down, weights, ids, "silu",
        gate_up_type=19, down_type=18, intermediate_size=8,
    )
    assert calls == [(2, 19, 16, 1), (1, 18, 16, 2)]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not __import__("os").environ.get("FREETOKEN_LLAMA_CPP_DIR"),
    reason="needs CUDA and a built FREETOKEN_LLAMA_CPP_DIR",
)
def test_llama_iq_mmq_matches_existing_unique_route_path(monkeypatch):
    from freetoken.models.gguf.dequant import GGML_IQ1_S, GGML_IQ3_XXS, row_bytes
    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    def valid_rows(shape, block_bytes):
        raw = torch.randint(0, 256, shape, dtype=torch.uint8)
        blocks = raw.view(*shape[:-1], -1, block_bytes)
        blocks[..., :2] = torch.tensor([0.02], dtype=torch.float16).view(torch.uint8)
        return raw.cuda()

    torch.manual_seed(304)
    experts, hidden, intermediate, tokens, top_k = 4, 256, 256, 8, 2
    gate_up = valid_rows(
        (experts, 2 * intermediate, row_bytes(hidden, GGML_IQ1_S)), 50
    )
    down = valid_rows(
        (experts, hidden, row_bytes(intermediate, GGML_IQ3_XXS)), 98
    )
    x = (torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    ids = torch.stack(
        [torch.randperm(experts, device="cuda", dtype=torch.int32)[:top_k] for _ in range(tokens)]
    )
    weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=-1)
    llama_dir = __import__("os").environ["FREETOKEN_LLAMA_CPP_DIR"]

    monkeypatch.delenv("FREETOKEN_LLAMA_CPP_DIR")
    expected = fused_experts_gguf(
        x, gate_up, down, weights, ids, "silu",
        gate_up_type=GGML_IQ1_S, down_type=GGML_IQ3_XXS,
    )
    monkeypatch.setenv("FREETOKEN_LLAMA_CPP_DIR", llama_dir)
    gate_flat = gate_up.flatten(1)
    down_flat = down.flatten(1)
    gate_cache = torch.zeros(
        experts, gate_flat.shape[1] + 400, dtype=torch.uint8, device="cuda"
    )
    down_cache = torch.zeros(
        experts, down_flat.shape[1] + 784, dtype=torch.uint8, device="cuda"
    )
    gate_cache[:, : gate_flat.shape[1]] = gate_flat
    down_cache[:, : down_flat.shape[1]] = down_flat
    actual = fused_experts_gguf(
        x, gate_cache, down_cache, weights, ids, "silu",
        gate_up_type=GGML_IQ1_S, down_type=GGML_IQ3_XXS,
        intermediate_size=intermediate,
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
