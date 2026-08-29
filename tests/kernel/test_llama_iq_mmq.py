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
    assert supported(17, 4096)
    assert supported(23, 4096)
    assert not supported(19, 6)
    assert not supported(16, 4096)


def test_llama_iq_mmq_checkout_validation(monkeypatch, tmp_path):
    from freetoken.kernel import llama_iq_mmq

    monkeypatch.setenv("FREETOKEN_LLAMA_CPP_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="built llama.cpp checkout"):
        llama_iq_mmq._checkout()


def test_llama_iq_mmq_rejects_unpinned_revision(monkeypatch, tmp_path):
    from freetoken.kernel import llama_iq_mmq

    required = (
        tmp_path / "ggml" / "include" / "ggml.h",
        tmp_path / "ggml" / "src" / "ggml-cuda" / "mmq.cuh",
        tmp_path / "build" / "bin" / "libggml-cuda.so",
        tmp_path / "build" / "bin" / "libggml.so",
        tmp_path / "build" / "bin" / "libggml-base.so",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    class Result:
        stdout = "deadbeef\n"

    monkeypatch.setattr(llama_iq_mmq.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setenv("FREETOKEN_LLAMA_CPP_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="unsupported llama.cpp revision deadbeef"):
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


@pytest.mark.parametrize("gate_type,down_type", [(19, 18), (17, 23)])
def test_single_token_decode_keeps_both_projections_on_mmvq(
    monkeypatch, gate_type, down_type
):
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
        gate_up_type=gate_type, down_type=down_type, intermediate_size=8,
    )
    assert calls == [(2, gate_type, 16, 1), (1, down_type, 16, 2)]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not __import__("os").environ.get("FREETOKEN_LLAMA_CPP_DIR"),
    reason="needs CUDA and a built FREETOKEN_LLAMA_CPP_DIR",
)
@pytest.mark.parametrize(
    "gate_type,down_type,gate_block_bytes,down_block_bytes,gate_alignment,down_alignment",
    [
        (19, 18, 50, 98, 400, 784),
        (17, 23, 74, 136, 29008, 13328),
    ],
)
def test_llama_iq_mmq_matches_existing_unique_route_path(
    monkeypatch,
    gate_type,
    down_type,
    gate_block_bytes,
    down_block_bytes,
    gate_alignment,
    down_alignment,
):
    from freetoken.models.gguf.dequant import row_bytes
    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    def valid_rows(shape, block_bytes):
        raw = torch.randint(0, 256, shape, dtype=torch.uint8)
        blocks = raw.view(*shape[:-1], -1, block_bytes)
        blocks[..., :2] = torch.tensor([0.02], dtype=torch.float16).view(torch.uint8)
        return raw.cuda()

    torch.manual_seed(304)
    experts, hidden, intermediate, tokens, top_k = 4, 256, 256, 8, 2
    gate_up = valid_rows(
        (experts, 2 * intermediate, row_bytes(hidden, gate_type)), gate_block_bytes
    )
    down = valid_rows(
        (experts, hidden, row_bytes(intermediate, down_type)), down_block_bytes
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
        gate_up_type=gate_type, down_type=down_type,
    )
    monkeypatch.setenv("FREETOKEN_LLAMA_CPP_DIR", llama_dir)
    gate_flat = gate_up.flatten(1)
    down_flat = down.flatten(1)
    gate_stride = (gate_flat.shape[1] + gate_alignment - 1) // gate_alignment * gate_alignment
    down_stride = (down_flat.shape[1] + down_alignment - 1) // down_alignment * down_alignment
    gate_cache = torch.zeros(experts, gate_stride, dtype=torch.uint8, device="cuda")
    down_cache = torch.zeros(experts, down_stride, dtype=torch.uint8, device="cuda")
    gate_cache[:, : gate_flat.shape[1]] = gate_flat
    down_cache[:, : down_flat.shape[1]] = down_flat
    actual = fused_experts_gguf(
        x, gate_cache, down_cache, weights, ids, "silu",
        gate_up_type=gate_type, down_type=down_type,
        intermediate_size=intermediate,
    )
    # MMQ and MMVQ quantize activations with different tilings, so two low-bit
    # projections plus SiLU can amplify harmless BF16 elementwise differences.
    # Qualify the whole result by scale and direction rather than penalizing
    # near-zero reference elements with an uninformative relative error.
    actual_f = actual.float()
    expected_f = expected.float()
    assert torch.isfinite(actual_f).all()
    max_error = (actual_f - expected_f).abs().max()
    reference_peak = expected_f.abs().max()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    assert max_error / reference_peak < 0.01
    assert cosine > 0.999
