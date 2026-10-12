"""Engine side of DeepSeek-V4.1 C2 with the C1 adaptive prefill optimisations (CPU).

Under the gated C2 (``FREETOKEN_DSV41_CONCURRENCY=1`` and two running requests) tail pairing and
the dependency cone keep every C1 startup check, at concurrency 2 and with one request per
prefill forward. The one-request calls, and every refusal outside the gated C2, are unchanged.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

import freetoken.engine.engine as engine_mod
import freetoken.models.deepseek_v41.moe as dsv41_moe
from freetoken.engine.engine import (
    Engine,
    _dsv41_feature_concurrency,
    _feature_concurrency_problems,
    _validate_tail_pairing,
)
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.moe import prefill_trace
from freetoken.scheduler import dependency_cone as dc
from freetoken.scheduler.tail_pairing import TAIL_PAIRING_ENV

from .test_dsv41_tail_pairing_engine import (  # noqa: F401 -- startup is a fixture
    _PRODUCTION_CONCURRENCY,
    _PastStartupChecks,
    _config,
    _hf_config,
    _plan,
    _refusal,
    _serve_config,
    startup,
)

CONCURRENCY_ENV = "FREETOKEN_DSV41_CONCURRENCY"
REAL_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)  # the checkpoint's layer plan
C2_ARGS = ("--max-running-requests", "2")
C2_ONE_REQUEST_PREFILL = "one request per prefill forward under C2 (prefill_batch_size_limit 1)"
_G0 = f"{TAIL_PAIRING_ENV}=1 requires: "


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", True)
    monkeypatch.setattr(prefill_trace, "enabled", lambda: False)
    for name in (CONCURRENCY_ENV, TAIL_PAIRING_ENV, dc.DEPENDENCY_CONE_ENV,
                 dc.DEPENDENCY_CONE_MIN_TOKENS_ENV, "FREETOKEN_DSV41_PREFILL_EARLY_REFILL"):
        monkeypatch.delenv(name, raising=False)


def _c2_config(limit=1, **overrides):
    model_config = SimpleNamespace(dsv41_args=object(), prefill_batch_size_limit=limit)
    return _config(**{"max_running_req": 2, "model_config": model_config, **overrides})


# ----------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("gate", "max_running", "dsv41", "expected"),
    [
        ("1", 2, True, 2),
        ("1", 1, True, 1),
        (None, 2, True, 1),
        ("0", 2, True, 1),
        ("1", 2, False, 1),  # only DeepSeek-V4.1 has the gated C2
        ("1", 4, True, 1),
    ],
)
def test_feature_concurrency_is_two_only_for_the_gated_dsv41_c2(
    monkeypatch, gate, max_running, dsv41, expected
):
    if gate is not None:
        monkeypatch.setenv(CONCURRENCY_ENV, gate)
    config = SimpleNamespace(
        model_config=SimpleNamespace(dsv41_args=object() if dsv41 else None),
        max_running_req=max_running,
    )
    assert _dsv41_feature_concurrency(config) == expected
    assert _dsv41_feature_concurrency(SimpleNamespace(max_running_req=2)) == 1


def test_concurrency_problems_keep_the_one_request_wording():
    # The exact strings the C1 validators produced, and the live gate judged.
    assert _feature_concurrency_problems(_config(max_running_req=1), 2, 1) == [
        "--max-running-requests 1 (got 2)"]
    assert _feature_concurrency_problems(_config(max_running_req=2), 1, 1) == [
        "--max-running-requests 1 (effective 2)"]
    assert _feature_concurrency_problems(_config(max_running_req=1), 1, 1) == []
    # C2: exactly two, and one request per prefill forward.
    assert _feature_concurrency_problems(_c2_config(), 2, 2) == []
    assert _feature_concurrency_problems(_c2_config(limit=None), 2, 2) == [C2_ONE_REQUEST_PREFILL]
    assert _feature_concurrency_problems(_c2_config(limit=2), 2, 2) == [C2_ONE_REQUEST_PREFILL]
    assert _feature_concurrency_problems(_c2_config(), 1, 2) == [
        "--max-running-requests 2 (got 1)"]
    assert _feature_concurrency_problems(_c2_config(max_running_req=1), 2, 2) == [
        "--max-running-requests 2 (effective 1)"]


# ----------------------------------------------------------------------------- validators


def test_tail_pairing_validation_accepts_serialized_c2():
    _validate_tail_pairing(_c2_config(), _plan(), requested_max_running_req=2, concurrency=2)


@pytest.mark.parametrize(
    ("config", "plan", "message"),
    [
        (_c2_config(limit=None), _plan(), C2_ONE_REQUEST_PREFILL),
        # Every C1 requirement still applies at concurrency 2.
        (_c2_config(), _plan(uses_authority_transport=False), "authority-EP"),
        (_c2_config(speculative_dspark=True), _plan(), "MTP/DSpark off"),
        (_c2_config(max_extend_tokens=16_384), _plan(), "--max-extend-tokens 8192"),
        (_c2_config(moe_sparse_prefill_max_tokens=4096), _plan(), "below 4096"),
        (_c2_config(moe_prefill_overlap=False), _plan(), "prefill overlap"),
    ],
)
def test_tail_pairing_validation_refuses_unsafe_c2(config, plan, message):
    with pytest.raises(ValueError, match=re.escape(message)) as excinfo:
        _validate_tail_pairing(config, plan, requested_max_running_req=2, concurrency=2)
    assert str(excinfo.value).startswith(_G0)


def test_tail_pairing_validation_without_c2_keeps_the_one_request_refusal():
    with pytest.raises(ValueError) as excinfo:
        _validate_tail_pairing(_c2_config(), _plan(), requested_max_running_req=2)
    assert str(excinfo.value) == f"{_G0}--max-running-requests 1 (got 2)"


def _cone_config(**overrides):
    model_config = SimpleNamespace(dsv41_args=DeepseekV41Args(compress_ratios=REAL_RATIOS),
                                   prefill_batch_size_limit=overrides.pop("limit", 1))
    values = dict(model_config=model_config, page_size=128, max_running_req=2,
                  speculative_dspark=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cone_problems_accept_serialized_c2_and_refuse_the_rest(monkeypatch):
    plan = SimpleNamespace(uses_authority_transport=True)
    kwargs = dict(requested_max_running_req=2, concurrency=2)
    assert dc.dependency_cone_problems(_cone_config(), plan, **kwargs) == []
    assert dc.dependency_cone_problems(_cone_config(limit=None), plan, **kwargs) == [
        C2_ONE_REQUEST_PREFILL]
    assert dc.dependency_cone_problems(_cone_config(speculative_dspark=True), plan, **kwargs) == [
        "MTP/DSpark off"]
    # Without the C2 concurrency the old refusal stands, word for word.
    assert dc.dependency_cone_problems(_cone_config(), plan, requested_max_running_req=2) == [
        "--max-running-requests 1 (got 2)"]
    monkeypatch.setenv("FREETOKEN_DSV41_PREFILL_EARLY_REFILL", "1")
    assert dc.dependency_cone_problems(_cone_config(), plan, **kwargs) == [
        "FREETOKEN_DSV41_PREFILL_EARLY_REFILL off"]


# --------------------------------------------- startup through the real Engine.__init__


def _real_layer_plan(monkeypatch):
    """The checkpoint's 40-layer encoder-decoder plan, which the cone validates at startup."""
    from freetoken.models.deepseek_v41 import config as dsv41_config

    monkeypatch.setattr(dsv41_config, "load_args",
                        lambda path, **_: DeepseekV41Args(compress_ratios=REAL_RATIOS))


def _record(monkeypatch, module, name):
    calls = []
    real = getattr(module, name)

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, recording)
    return calls


def test_gated_c2_runs_the_tail_pairing_checks_at_concurrency_two(monkeypatch, startup):
    start, validations = startup
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    config = _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS)
    with pytest.raises(_PastStartupChecks):
        start(config)
    assert validations == [{"requested_max_running_req": 2, "concurrency": 2}]
    assert config.max_running_req == 2


def test_gated_c2_runs_the_cone_checks_at_concurrency_two(monkeypatch, startup):
    start, _ = startup
    _real_layer_plan(monkeypatch)
    calls = _record(monkeypatch, dc, "dependency_cone_problems")
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    monkeypatch.setenv(dc.DEPENDENCY_CONE_ENV, "1")
    monkeypatch.setenv(dc.DEPENDENCY_CONE_MIN_TOKENS_ENV, "32768")
    config = _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS)
    with pytest.raises(_PastStartupChecks):
        start(config)
    assert calls == [{"requested_max_running_req": 2, "concurrency": 2}]


@pytest.mark.parametrize(
    ("env", "trace", "extra", "requirement"),
    [
        ({"FREETOKEN_DSV41_PREFILL_EARLY_REFILL": "1"}, False, (),
         "FREETOKEN_DSV41_PREFILL_EARLY_REFILL off"),
        ({}, True, (), "FREETOKEN_DSV41_PREFILL_TRACE off"),
        ({}, False, ("--max-prefill-length", "16384"),
         "--max-extend-tokens 8192, the qualified head chunk (got 16384)"),
        ({}, False, ("--moe-sparse-prefill-max-tokens", "4096"),
         "--moe-sparse-prefill-max-tokens below 4096, so that tails stream the bank (got 4096)"),
    ],
    ids=["early_refill", "prefill_trace", "head_16384", "sparse_4096"],
)
def test_gated_c2_refuses_unqualified_geometry_before_the_weight_load(
    monkeypatch, startup, env, trace, extra, requirement
):
    start, _ = startup
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(prefill_trace, "enabled", lambda: trace)
    message = _refusal(start, _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS, *extra))
    assert message.startswith(_G0) and requirement in message.split(_G0, 1)[1]


@pytest.mark.parametrize("flag", [TAIL_PAIRING_ENV, dc.DEPENDENCY_CONE_ENV])
def test_gated_c2_without_one_request_prefill_is_refused_before_the_weight_load(
    monkeypatch, startup, flag
):
    start, _ = startup
    _real_layer_plan(monkeypatch)
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    monkeypatch.setenv(flag, "1")
    config = _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS)
    config.__dict__["hf_config"] = _hf_config()
    # A future ragged prefill that batches prompts must not inherit these optimisations.
    config.__dict__["model_config"] = replace(config.model_config, prefill_batch_size_limit=None)
    message = _refusal(start, config)
    assert message == f"{flag}=1 requires: {C2_ONE_REQUEST_PREFILL}"


def test_gated_c2_keeps_mtp_refused_before_the_weight_load(monkeypatch, startup):
    start, validations = startup
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    config = _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS, "--speculative-dspark",
                           "--dspark-device", "3")
    with pytest.raises(ValueError, match="DSpark"):
        start(config)
    assert not validations  # _adjust_config refused first: DSpark needs one running request


def test_one_request_calls_are_unchanged_under_the_gate(monkeypatch, startup):
    start, validations = startup
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    config = _serve_config(*_PRODUCTION_CONCURRENCY)
    with pytest.raises(_PastStartupChecks):
        start(config)
    assert validations == [{"requested_max_running_req": 1}]  # no concurrency keyword


def test_an_ungated_two_request_launch_keeps_the_one_request_refusal(startup, monkeypatch):
    start, _ = startup
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    assert _refusal(start, _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS)) == (
        f"{_G0}--max-running-requests 1 (got 2)")


def test_phase_cache_priority_follows_the_same_concurrency_gate():
    source = inspect.getsource(Engine.__init__)
    guard = source.index("if phase_priority_requested and (")
    create = source.index("PhaseCachePriority.from_engine(self)")
    assert guard < create
    assert "config.max_running_req == _dsv41_feature_concurrency(config)" in source[guard:create]
    assert engine_mod._dsv41_feature_concurrency is _dsv41_feature_concurrency
