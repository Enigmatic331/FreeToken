"""CPU-only policy, memo, schedule, and default-off gates for Package A."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.deepseek_v41 import authority_fusion as fusion


def _execution(**overrides):
    values = dict(
        uses_authority_transport=True,
        rank=0,
        backbone_rank=0,
        dense_parallel=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch(**overrides):
    values = dict(is_decode=True, speculative=False, padded_size=1)
    values.update(overrides)
    return SimpleNamespace(**values)


def _args(**overrides):
    values = dict(
        n_layers=40,
        dim=5120,
        hc_mult=4,
        n_routed_experts=384,
        n_activated_experts=6,
        score_func="sqrtsoftplus",
        gate_temp=1.0,
        norm_topk_prob=True,
        route_scale=1.5,
        norm_eps=1e-20,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
        engram_layer_ids=(1, 14),
        kv_source_layers=(2, 8, 14, 20),
        index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
        dspark_enabled=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_strict_flag_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(fusion.AUTHORITY_FUSION_ENV, raising=False)
    else:
        monkeypatch.setenv(fusion.AUTHORITY_FUSION_ENV, value)
    assert fusion._parse_flag() is expected


@pytest.mark.parametrize("value", ["", "true", "yes", "2", " 1"])
def test_invalid_flag_refuses(monkeypatch, value):
    monkeypatch.setenv(fusion.AUTHORITY_FUSION_ENV, value)
    with pytest.raises(ValueError, match="must be '0' or '1'"):
        fusion._parse_flag()


def test_default_off_warmup_touches_nothing(monkeypatch):
    monkeypatch.setattr(fusion, "_ENABLED", False)

    class Bomb:
        def __getattribute__(self, name):
            raise AssertionError(f"default-off warmup touched transformer.{name}")

    fusion.warmup_authority_fusion(Bomb(), torch.device("cuda:0"))


def test_g1_process_does_not_inherit_enabled_flag():
    assert fusion.AUTHORITY_FUSION_ENV not in os.environ
    assert fusion._ENABLED is False


@pytest.mark.parametrize("rows", [1, 2])
def test_flag_off_rejects_otherwise_eligible_batch(monkeypatch, rows):
    monkeypatch.setattr(fusion, "_ENABLED", False)
    assert not fusion.authority_fusion_eligible(
        _batch(padded_size=rows), _execution()
    )


def test_invalid_flag_fails_real_module_import():
    env = dict(os.environ)
    env[fusion.AUTHORITY_FUSION_ENV] = "2"
    result = subprocess.run(
        [sys.executable, "-c", "import freetoken.models.deepseek_v41.authority_fusion"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must be '0' or '1'" in result.stderr


@pytest.mark.parametrize("rows", [1, 2])
def test_eligibility_accepts_only_frozen_rows(monkeypatch, rows):
    monkeypatch.setattr(fusion, "_ENABLED", True)
    assert fusion.authority_fusion_eligible(_batch(padded_size=rows), _execution())


@pytest.mark.parametrize(
    "batch, execution",
    [
        (_batch(is_decode=False), _execution()),
        (_batch(speculative=True), _execution()),
        (_batch(padded_size=3), _execution()),
        (_batch(), _execution(uses_authority_transport=False)),
        (_batch(), _execution(rank=1)),
    ],
)
def test_eligibility_rejects_ineligible_routes(monkeypatch, batch, execution):
    monkeypatch.setattr(fusion, "_ENABLED", True)
    assert not fusion.authority_fusion_eligible(batch, execution)


def test_layer_sets_are_derived_and_frozen():
    assert fusion._derived_layer_sets(_args()) == (
        fusion.FALLBACK_LAYERS,
        fusion.SUPPRESSED_PRODUCERS,
        fusion.PUBLICATION_LAYERS,
    )


def test_geometry_accepts_exact_configuration(monkeypatch):
    from freetoken.models.deepseek_v41 import execution

    monkeypatch.setattr(execution, "get_execution_plan", lambda: _execution())
    monkeypatch.delenv("TRITON_DEFAULT_FP_FUSION", raising=False)
    monkeypatch.setenv("FREETOKEN_DSV41_FUSED_ROUTER", "0")
    monkeypatch.setenv("FREETOKEN_DSV41_HC_DIRECT_PRE", "0")
    transformer = SimpleNamespace(
        args=_args(),
        layers=[SimpleNamespace(ffn=SimpleNamespace(fused_decode_dispatch=True)) for _ in range(40)],
    )
    fusion.validate_authority_geometry(transformer)


@pytest.mark.parametrize(
    "change",
    [
        {"n_layers": 39},
        {"dim": 4096},
        {"hc_mult": 2},
        {"n_routed_experts": 256},
        {"n_activated_experts": 4},
        {"score_func": "softmax"},
        {"gate_temp": 0.9},
        {"norm_topk_prob": False},
        {"route_scale": 1.0},
        {"norm_eps": 1e-6},
        {"hc_eps": 1e-5},
        {"hc_sinkhorn_iters": 10},
        {"engram_layer_ids": (1, 13)},
        {"kv_source_layers": (2, 8)},
        {"index_source_layers": (2, 8)},
        {"dspark_enabled": True},
    ],
)
def test_geometry_refuses_drift(monkeypatch, change):
    from freetoken.models.deepseek_v41 import execution

    monkeypatch.setattr(execution, "get_execution_plan", lambda: _execution())
    transformer = SimpleNamespace(
        args=_args(**change),
        layers=[SimpleNamespace(ffn=SimpleNamespace(fused_decode_dispatch=True)) for _ in range(40)],
    )
    with pytest.raises(RuntimeError, match="geometry refusal"):
        fusion.validate_authority_geometry(transformer)


@pytest.mark.parametrize(
    "execution_plan",
    [_execution(dense_parallel=True), _execution(rank=1)],
)
def test_geometry_refuses_execution_drift(monkeypatch, execution_plan):
    from freetoken.models.deepseek_v41 import execution

    monkeypatch.setattr(execution, "get_execution_plan", lambda: execution_plan)
    transformer = SimpleNamespace(
        args=_args(),
        layers=[SimpleNamespace(ffn=SimpleNamespace(fused_decode_dispatch=True)) for _ in range(40)],
    )
    with pytest.raises(RuntimeError, match="geometry refusal"):
        fusion.validate_authority_geometry(transformer)


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("FREETOKEN_DSV41_FUSED_ROUTER", "1", "fused router"),
        ("FREETOKEN_DSV41_HC_DIRECT_PRE", "1", "direct-pre"),
        ("TRITON_DEFAULT_FP_FUSION", "0", "TRITON_DEFAULT_FP_FUSION"),
    ],
)
def test_geometry_refuses_conflicting_environment(monkeypatch, name, value, match):
    from freetoken.models.deepseek_v41 import execution

    monkeypatch.setattr(execution, "get_execution_plan", lambda: _execution())
    monkeypatch.setenv(name, value)
    transformer = SimpleNamespace(
        args=_args(),
        layers=[SimpleNamespace(ffn=SimpleNamespace(fused_decode_dispatch=True)) for _ in range(40)],
    )
    with pytest.raises(RuntimeError, match=match):
        fusion.validate_authority_geometry(transformer)


def test_prepared_mix_requires_identity_epoch_and_version():
    source = torch.zeros(1, 1, 4, 8, dtype=torch.bfloat16)
    flat = source.flatten(-2).float()
    square = flat.square()
    prepared = fusion.PreparedMix.capture(flat, square, source, epoch=7)
    assert prepared.require(source, 7) == (flat, square)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        prepared.require(source.clone(), 7)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        prepared.require(source, 8)
    source.add_(1)
    with pytest.raises(RuntimeError, match="modified in place"):
        prepared.require(source, 7)


def test_prepared_mix_accepts_inference_tensor_without_version_counter():
    with torch.inference_mode():
        source = torch.zeros(1, 1, 4, 8, dtype=torch.bfloat16)
        flat = source.flatten(-2).float()
        square = flat.square()
        prepared = fusion.PreparedMix.capture(flat, square, source, epoch=7)
        assert prepared.source_version == -1
        assert prepared.require(source, 7) == (flat, square)


def test_prepared_mix_rejects_aliasing():
    source = torch.zeros(1, 1, 4, 8, dtype=torch.bfloat16)
    flat = source.flatten(-2).float()
    prepared = fusion.PreparedMix.capture(flat, flat, source, epoch=1)
    with pytest.raises(RuntimeError, match="alias"):
        prepared.require(source, 1)


def test_forward_memo_uses_strong_identity_keys_and_resets():
    memo = fusion.ForwardMemo()
    epoch = memo.begin()
    table = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    positions = torch.tensor([1, 3])
    first = memo.frequencies(table, positions)
    assert memo.frequencies(table, positions) is first
    equal_table = table.clone()
    table_ref = weakref.ref(equal_table)
    second = memo.frequencies(equal_table, positions)
    assert second is not first and torch.equal(second, first)
    del table, equal_table
    gc.collect()
    assert table_ref() is not None
    memo.clear()
    gc.collect()
    assert not memo.active and memo._freq_table is None
    assert memo.begin() == epoch + 1
    memo.clear()


def test_forward_memo_attention_rows_identity_not_value():
    memo = fusion.ForwardMemo()
    memo.begin()
    window = torch.tensor([[[0, 1]]], dtype=torch.int64)
    compressed = torch.tensor([[[2, -1]]], dtype=torch.int64)
    first = memo.attention_rows(window, compressed)
    assert memo.attention_rows(window, compressed)[0] is first[0]
    compressed.add_(1)
    mutated = memo.attention_rows(window, compressed)
    assert mutated[0] is not first[0]
    second = memo.attention_rows(window.clone(), compressed.clone())
    assert second[0] is not mutated[0] and torch.equal(second[0], mutated[0])
    memo.clear()
    with pytest.raises(RuntimeError, match="inactive"):
        memo.attention_rows(window, compressed)


def test_test_policy_has_no_environment_surface():
    reduced = fusion.policy_without("P1", "R8")
    assert not reduced.has("P1") and not reduced.has("R8")
    assert reduced.has("R1")
    assert list(name for name in os.environ if "LEVER" in name) == []
    with pytest.raises(ValueError, match="unknown"):
        fusion.policy_without("R13h")


def test_full_forty_layer_preparation_schedule_and_planted_defects():
    marker = object()
    for layer in range(40):
        expected = None if layer in fusion.FALLBACK_LAYERS else marker
        fusion._require_preparation_schedule(layer, expected)
        wrong = marker if expected is None else None
        with pytest.raises(RuntimeError, match=f"layer {layer}"):
            fusion._require_preparation_schedule(layer, wrong)
