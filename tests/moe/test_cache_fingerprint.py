"""Opt-in MoE admission-state fingerprint (measurement only, default off)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.moe import cache_fingerprint as fp
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.scheduler.scheduler import Scheduler


def _cache() -> OffloadMoeCache:
    cache = OffloadMoeCache(2, 8, 10, torch.device("cpu"), decode_target="hybrid")
    cache.slot_for_id[0, 3] = 4
    cache.id_of_slot[4] = 3
    cache.usage[4] = 7
    cache.step.fill_(7)
    cache.expert_recency[0, 3] = 7
    return cache


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True)])
def test_strict_parse(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(fp.FINGERPRINT_ENV, raising=False)
    else:
        monkeypatch.setenv(fp.FINGERPRINT_ENV, value)
    assert fp.fingerprint_enabled() is expected


@pytest.mark.parametrize("value", ["", "true", "on", "2", " 1"])
def test_invalid_values_are_refused(monkeypatch, value):
    monkeypatch.setenv(fp.FINGERPRINT_ENV, value)
    with pytest.raises(ValueError):
        fp.fingerprint_enabled()


def test_equal_state_equal_digest_and_counters_do_not_matter():
    a, b = _cache(), _cache()
    b.lru_stats.fill_(5)
    b.stat_missing_layer.fill_(3)
    b.num_indices.fill_(2)  # per-call scratch, not persistent admission state
    fa, fb = fp.cache_fingerprint(a), fp.cache_fingerprint(b)
    assert fa == fb
    assert fa["step"] == 7 and fa["resident"] == 1 and len(fa["sha256"]) == 64


@pytest.mark.parametrize("field,mutate", [
    ("slot_for_id", lambda c: c.slot_for_id.__setitem__((1, 2), 9)),
    ("id_of_slot", lambda c: c.id_of_slot.__setitem__(9, 10)),
    ("usage", lambda c: c.usage.__setitem__(4, 6)),
    ("step", lambda c: c.step.fill_(8)),
    ("expert_recency", lambda c: c.expert_recency.__setitem__((1, 1), 2)),
])
def test_every_persistent_field_changes_the_digest(field, mutate):
    a, b = _cache(), _cache()
    mutate(b)
    assert fp.cache_fingerprint(a)["sha256"] != fp.cache_fingerprint(b)["sha256"], field


def test_format_is_stable():
    line = fp.format_fingerprint(2, dict(step=9, resident=3, sha256="ab" * 32))
    assert line == f"MoE cache fingerprint rank=2 step=9 resident=3 sha256={'ab' * 32}"


def _scheduler(enabled: bool, logs: list[str]):
    sched = Scheduler.__new__(Scheduler)
    sched._idle_checks_done = False
    sched._moe_cache_fingerprint = enabled
    sched.engine = SimpleNamespace(moe_offload_cache=_cache())
    sched.config = SimpleNamespace(tp_info=SimpleNamespace(rank=1), moe_collect_stats=False)
    sched.cache_manager = SimpleNamespace(check_integrity=lambda: None)
    return sched


@pytest.mark.parametrize("enabled", [False, True])
def test_idle_epoch_logs_once_only_when_enabled(monkeypatch, enabled):
    from freetoken.scheduler import scheduler as module

    logs: list[str] = []
    monkeypatch.setattr(module.logger, "info", lambda msg, *a: logs.append(msg % a if a else msg))
    monkeypatch.setattr(module.logger, "info_rank0", lambda *a, **k: None)
    sched = _scheduler(enabled, logs)
    Scheduler.run_when_idle(sched)
    Scheduler.run_when_idle(sched)  # same idle epoch: no second digest
    lines = [line for line in logs if line.startswith("MoE cache fingerprint")]
    assert len(lines) == (1 if enabled else 0)
    if enabled:
        assert lines[0].startswith("MoE cache fingerprint rank=1 step=7 resident=1 sha256=")


def test_default_constructed_scheduler_has_no_fingerprint_attribute_dependency(monkeypatch):
    from freetoken.scheduler import scheduler as module

    monkeypatch.setattr(module.logger, "info_rank0", lambda *a, **k: None)
    sched = _scheduler(False, [])
    del sched._moe_cache_fingerprint  # e.g. older harness objects built with __new__
    Scheduler.run_when_idle(sched)
    assert sched._idle_checks_done
