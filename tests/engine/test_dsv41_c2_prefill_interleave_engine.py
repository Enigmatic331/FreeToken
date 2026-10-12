"""Engine startup gate of the DeepSeek-V4.1 C2 prefill interleave (CPU, real Engine.__init__).

``FREETOKEN_DSV41_C2_PREFILL_INTERLEAVE`` is accepted only under the gated C2 and is refused before
the weight load everywhere else; unset or ``0`` adds no check and changes nothing.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.moe import prefill_trace
from freetoken.scheduler import prefill_interleave as pi
from freetoken.scheduler.tail_pairing import TAIL_PAIRING_ENV

from .test_dsv41_tail_pairing_engine import (  # noqa: F401 -- startup is a fixture
    _PRODUCTION_CONCURRENCY,
    _PastStartupChecks,
    _hf_config,
    _refusal,
    _serve_config,
    startup,
)

CONCURRENCY_ENV = "FREETOKEN_DSV41_CONCURRENCY"
C2_ARGS = ("--max-running-requests", "2")
_G = f"{pi.PREFILL_INTERLEAVE_ENV} requires: "
_C2 = "DeepSeek-V4.1 gated C2 (FREETOKEN_DSV41_CONCURRENCY=1, --max-running-requests 2)"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(prefill_trace, "enabled", lambda: False)
    for name in (CONCURRENCY_ENV, TAIL_PAIRING_ENV, pi.PREFILL_INTERLEAVE_ENV):
        monkeypatch.delenv(name, raising=False)


def _problems_spy(monkeypatch):
    calls = []
    real = pi.prefill_interleave_problems

    def spy(config, **kwargs):
        calls.append(kwargs)
        return real(config, **kwargs)

    monkeypatch.setattr(pi, "prefill_interleave_problems", spy)
    return calls


def test_gated_c2_passes_the_interleave_checks(monkeypatch, startup):
    start, _ = startup
    calls = _problems_spy(monkeypatch)
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    monkeypatch.setenv(pi.PREFILL_INTERLEAVE_ENV, "1")
    config = _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS)
    with pytest.raises(_PastStartupChecks):
        start(config)
    assert calls == [{"requested_max_running_req": 2}]


@pytest.mark.parametrize("value", ["0", None])
def test_gate_off_adds_no_check(monkeypatch, startup, value):
    start, _ = startup
    calls = _problems_spy(monkeypatch)
    if value is not None:
        monkeypatch.setenv(pi.PREFILL_INTERLEAVE_ENV, value)
    with pytest.raises(_PastStartupChecks):
        start(_serve_config(*_PRODUCTION_CONCURRENCY))
    assert calls == []


@pytest.mark.parametrize("c2_gate", [None, "1"])  # the C2 gate alone does not make C1 into C2
def test_c1_with_the_gate_is_refused_before_the_weight_load(monkeypatch, startup, c2_gate):
    start, _ = startup
    monkeypatch.setenv(pi.PREFILL_INTERLEAVE_ENV, "1")
    if c2_gate is not None:
        monkeypatch.setenv(CONCURRENCY_ENV, c2_gate)
    assert _refusal(start, _serve_config(*_PRODUCTION_CONCURRENCY)) == _G + _C2


def test_an_ungated_two_request_launch_keeps_its_existing_refusal(monkeypatch, startup):
    start, _ = startup
    monkeypatch.setenv(pi.PREFILL_INTERLEAVE_ENV, "1")
    message = _refusal(start, _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS))
    assert "requires the experimental" in message and pi.PREFILL_INTERLEAVE_ENV not in message


@pytest.mark.parametrize("value", ["9", "x", "-1", "01", ""])
def test_malformed_gate_is_refused_before_the_weight_load(monkeypatch, startup, value):
    start, _ = startup
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    monkeypatch.setenv(pi.PREFILL_INTERLEAVE_ENV, value)
    message = _refusal(start, _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS))
    assert pi.PREFILL_INTERLEAVE_ENV in message


def test_gated_c2_without_one_request_prefill_is_refused(monkeypatch, startup):
    start, _ = startup
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    monkeypatch.setenv(pi.PREFILL_INTERLEAVE_ENV, "2")
    config = _serve_config(*_PRODUCTION_CONCURRENCY, *C2_ARGS)
    config.__dict__["hf_config"] = _hf_config()
    config.__dict__["model_config"] = replace(config.model_config, prefill_batch_size_limit=None)
    assert _refusal(start, config) == (
        _G + "one request per prefill forward under C2 (prefill_batch_size_limit 1)")


def test_problems_refuse_non_dsv41_and_dspark():
    from types import SimpleNamespace

    assert pi.prefill_interleave_problems(
        SimpleNamespace(model_config=SimpleNamespace(dsv41_args=None), max_running_req=2),
        requested_max_running_req=2) == ["DeepSeek-V4.1"]
    args = DeepseekV41Args(compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0))
    config = SimpleNamespace(model_config=SimpleNamespace(dsv41_args=args, prefill_batch_size_limit=1),
                             max_running_req=2, speculative_dspark=True)
    problems = pi.prefill_interleave_problems(config, requested_max_running_req=2)
    assert "MTP/DSpark off" in problems
