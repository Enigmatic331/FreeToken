from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from freetoken import core
from freetoken.core import Batch, Context
from freetoken.models.deepseek_v41.compress import Compressor
from freetoken.models.deepseek_v41.dspark import (
    DSparkAcceptanceFallback,
    DSparkAdaptiveVerification,
    _MTP_EXPERT_RE,
    accepted_prefix,
    sampling_probs,
)
from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM
from freetoken.engine.graph import GraphRunner, _dspark_graph_lengths
import freetoken.models.deepseek_v41.model as dsv41_model


def _t(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def test_greedy_acceptance_keeps_only_the_matching_prefix_and_bonus():
    assert accepted_prefix(_t(5, 6, 7), _t(5, 99, 7, 8)) == (1, 99)


def test_greedy_acceptance_uses_free_bonus_after_full_match():
    assert accepted_prefix(_t(5, 6, 7), _t(5, 6, 7, 8)) == (3, 8)


def test_sampling_probs_applies_temperature_top_k_and_top_p():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    probs = sampling_probs(logits, temperature=1.0, top_p=0.8, top_k=3)
    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(-1), torch.ones(1))
    assert probs[0, 2] == 0 and probs[0, 3] == 0


def test_acceptance_fallback_enters_and_leaves_cooldown():
    fallback = DSparkAcceptanceFallback(0.5, min_drafted=4, cooldown_steps=3)
    assert fallback.record(1, 4) == pytest.approx(0.25)
    assert [fallback.should_speculate() for _ in range(4)] == [False, False, False, True]


def test_adaptive_verification_maximizes_expected_tokens_per_step_cost():
    adaptive = DSparkAdaptiveVerification([5, 6, 7, 8, 9])

    assert adaptive.initial_length() == 5
    assert adaptive.choose_from_probabilities(
        torch.tensor([0.8, 0.1, 0.1, 0.1, 0.1])
    ) == 1
    assert adaptive.choose(torch.full((5,), 10.0)) == 5


def test_adaptive_verification_makes_noisy_cost_curve_monotonic():
    adaptive = DSparkAdaptiveVerification([5, 4, 7, 6, 9])

    assert adaptive.step_costs_ms == (5.0, 5.0, 7.0, 7.0, 9.0)


def test_dspark_draft_positions_start_at_the_live_anchor(monkeypatch):
    class RecordingDrafter:
        def __init__(self):
            self.anchor = None
            self.positions = None

        def propose(self, anchor, positions, sampling_params):
            self.anchor = anchor.clone()
            self.positions = positions.clone()
            return "proposal"

    drafter = RecordingDrafter()
    batch = SimpleNamespace(
        spec_block=5,
        input_ids=torch.tensor([71, 0, 0, 0, 0, 0]),
        positions=torch.arange(100, 106),
    )
    monkeypatch.setattr(
        dsv41_model,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=batch),
    )
    model = SimpleNamespace(
        _drafter=drafter,
        _args=SimpleNamespace(dspark_block_size=5),
    )

    assert DeepseekV41ForCausalLM.draft(model, [object()]) == "proposal"
    assert drafter.anchor.tolist() == [71]
    assert drafter.positions.tolist() == [100, 101, 102, 103, 104]


def test_dspark_draft_keeps_full_confidence_span_when_verify_is_trimmed(monkeypatch):
    class RecordingDrafter:
        def propose(self, anchor, positions, sampling_params):
            del anchor, sampling_params
            return positions.clone()

    batch = SimpleNamespace(
        spec_block=2,
        input_ids=torch.tensor([71, 0, 0]),
        positions=torch.arange(100, 103),
    )
    monkeypatch.setattr(
        dsv41_model,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=batch),
    )
    model = SimpleNamespace(
        _drafter=RecordingDrafter(),
        _args=SimpleNamespace(dspark_block_size=5),
    )

    positions = DeepseekV41ForCausalLM.draft(model, [object()])
    assert positions.tolist() == [100, 101, 102, 103, 104]


def test_dsfp4_experts_honor_v41_activation_block_size(monkeypatch):
    import freetoken.moe.fused_ds_fp4 as fused

    seen = []

    def roundtrip(value, block):
        seen.append(("gate_up", block))
        return value

    def inplace(value, block):
        seen.append(("down", block))
        return value

    def grouped(value, packed, scales, slots, weights, **kwargs):
        del scales, weights, kwargs
        return value.new_zeros(slots.shape[0], slots.shape[1], packed.shape[1])

    monkeypatch.setattr(fused, "act_quant_fp8_roundtrip", roundtrip)
    monkeypatch.setattr(fused, "act_quant_fp8_inplace", inplace)
    monkeypatch.setattr(fused, "_grouped_decode", grouped)
    monkeypatch.setattr(
        fused,
        "fused_swiglu",
        lambda value, limit: value[..., : value.shape[-1] // 2],
    )

    fused.routed_experts_fp4(
        torch.zeros(1, 4, dtype=torch.bfloat16),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.ones(1, 1),
        torch.zeros(1, 4, 2, dtype=torch.uint8),
        torch.zeros(1, 4, 1, dtype=torch.uint8),
        torch.zeros(1, 4, 1, dtype=torch.uint8),
        torch.zeros(1, 4, 1, dtype=torch.uint8),
        10.0,
        activation_block_size=32,
    )

    assert seen == [("gate_up", 32), ("down", 32)]


class _TakeTwo(nn.Module):
    def forward(self, value):
        return value[..., :2]


class _ZeroTwo(nn.Module):
    def forward(self, value):
        return torch.zeros_like(value[..., :2], dtype=torch.float32)


class _CarryBackend:
    window_size = 128

    def __init__(self):
        self.blocks = {}
        self.starts = None

    def read_carry(self, layer_id, tier, window_slot, ring_size):
        key = (layer_id, tier, window_slot // self.window_size)
        return self.blocks.get(key, torch.zeros(ring_size, 4)).clone()

    def write_carry(self, layer_id, tier, window_slot, ring_size, block):
        key = (layer_id, tier, window_slot // self.window_size)
        self.blocks[key] = block.clone()

    def compress_rows_of(self, table_idx, starts, ratio):
        self.starts = starts.clone()
        return torch.div(starts, ratio, rounding_mode="floor")

    def read_carry_blocks(self, layer_id, tier, window_slots, ring_size):
        return torch.stack(
            [self.read_carry(layer_id, tier, int(slot), ring_size) for slot in window_slots]
        )

    def write_carry_blocks(self, layer_id, tier, window_slots, ring_size, blocks):
        for slot, block in zip(window_slots, blocks, strict=True):
            self.write_carry(layer_id, tier, int(slot), ring_size, block)

    def verify_compress_rows(self, positions, ratio, layer_id, tier, completed):
        del layer_id, tier
        rows = torch.div(positions, ratio, rounding_mode="floor")
        return torch.where(completed, rows, torch.full_like(rows, 99))


def test_speculative_compressor_resumes_odd_position_and_journals_each_row(
    monkeypatch,
):
    compressor = Compressor.__new__(Compressor)
    nn.Module.__init__(compressor)
    compressor.compress_ratio = 2
    compressor.head_dim = 2
    compressor.norm = nn.Identity()
    compressor.wkv = _TakeTwo()
    compressor.wgate = _ZeroTwo()

    batch = Batch(reqs=[], phase="prefill")
    batch.speculative = True
    batch.spec_carry_states = {}
    ctx = Context(page_size=128)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    backend = _CarryBackend()
    x = torch.arange(18, dtype=torch.float32).view(1, 6, 3)
    window_slots = torch.arange(1, 7, dtype=torch.int64)
    with ctx.forward_batch(batch):
        latent, rows = compressor.prefill_paged(
            x,
            start_pos=1,
            window_slots=window_slots,
            layer_id=2,
            table_idx=0,
            backend=backend,
        )

    assert latent.shape == (1, 3, 2)
    assert rows.tolist() == [0, 1, 2]
    assert backend.starts.tolist() == [0, 2, 4]
    journal = batch.spec_carry_states[(2, "attn", 2)]
    assert len(journal) == 6
    assert all(piece.shape == (1, 2, 4) for piece in journal)


def test_graph_compressor_journals_each_fixed_row_and_restores_selected_state():
    compressor = Compressor.__new__(Compressor)
    nn.Module.__init__(compressor)
    compressor.compress_ratio = 2
    compressor.head_dim = 2
    compressor.norm = nn.Identity()
    compressor.wkv = _TakeTwo()
    compressor.wgate = _ZeroTwo()
    compressor.register_buffer(
        "spec_graph_journal", torch.empty(6, 2, 4), persistent=False
    )
    backend = _CarryBackend()
    x = torch.arange(15, dtype=torch.float32).view(1, 5, 3)
    positions = torch.arange(1, 6)
    slots = torch.arange(1, 6)

    latent, rows, completed = compressor.verify_paged(
        x, positions, slots, layer_id=2, backend=backend
    )

    assert latent.shape == (5, 2)
    assert completed.tolist() == [True, False, True, False, True]
    assert rows.tolist() == [0, 99, 1, 99, 2]
    expected = compressor.spec_graph_journal[2].clone()
    compressor.restore_graph_carry(2, 3, layer_id=2, backend=backend)
    assert torch.equal(backend.read_carry(2, "attn", 3, 2), expected)


def test_dspark_graph_lengths_are_explicit_bounded_and_deduplicated():
    assert _dspark_graph_lengths("4, 2,4", 5) == (2, 4)
    assert _dspark_graph_lengths("", 5) == ()
    with pytest.raises(ValueError, match="outside"):
        _dspark_graph_lengths("6", 5)


def test_dspark_only_graph_does_not_require_decode_padding_sizes():
    runner = GraphRunner.__new__(GraphRunner)
    runner.graph_bs_list = []
    runner.dummy_req = object()
    runner.can_use_cuda_graph = lambda batch: True
    req = object()
    batch = SimpleNamespace(
        speculative=True, size=1, reqs=[req], padded_reqs=None
    )

    runner.pad_batch(batch)

    assert batch.padded_reqs == [req]


def test_real_checkpoint_mtp_namespace_matches_loader_contract():
    configured_path = os.environ.get("FREETOKEN_DSV41_TEST_MODEL_PATH")
    if not configured_path:
        pytest.skip("set FREETOKEN_DSV41_TEST_MODEL_PATH for the checkpoint gate")
    model_path = Path(configured_path)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip("local DeepSeek-V4.1 checkpoint is not installed")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    names = {name for name in weight_map if name.startswith("mtp.")}
    expert_names = {name for name in names if _MTP_EXPERT_RE.match(name)}
    assert len(names) == 2401
    assert len(expert_names) == 3 * 128 * 6
    assert "mtp.0.main_proj.weight" in names
    assert "mtp.2.markov_head.head.weight" in names
    assert "mtp.2.confidence_head.proj.weight" in names
