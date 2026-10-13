"""Opt-in bounded admission trace: default off, exact CPU-reference records, content-free files."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe import admission_trace as at
from freetoken.moe.offload_cache import OffloadMoeCache

LAYERS, EXPERTS, SLOTS, TOPK, HIDDEN, INTER = 3, 40, 48, 4, 8, 4


def _cache():
    cache = OffloadMoeCache(LAYERS, EXPERTS, SLOTS, torch.device("cpu"), quant_format="bf16",
                            decode_target="hybrid", hybrid_max_fetch=1)
    gen = torch.Generator().manual_seed(3)
    cache.set_bank_sources({
        "gate_up": [torch.randn(EXPERTS, 2 * INTER, HIDDEN, generator=gen).bfloat16() for _ in range(LAYERS)],
        "down": [torch.randn(EXPERTS, HIDDEN, INTER, generator=gen).bfloat16() for _ in range(LAYERS)],
    })

    def copy_missing(*, num_indices=None):
        layer = cache._pending_src_layer
        for i in range(int((cache.num_indices if num_indices is None else num_indices).reshape(-1)[0])):
            for per_layer, bank in cache.banks:
                bank[int(cache.evict_slots[i])].copy_(per_layer[layer][int(cache.src_indices[i])])

    cache.copy_missing = copy_missing
    executor = SimpleNamespace(return_route_outputs=False,
                               decode_submit=lambda layer, hidden, w, ids: hidden,
                               decode_sync=lambda pending: torch.zeros_like(pending))
    cache.set_cpu_executor(executor)
    return cache


def _layer(layer_id, skip_empty):
    gemm = lambda cache, hidden, weights, slots, **_: hidden  # noqa: E731
    return SimpleNamespace(layer_id=layer_id, skip_empty_cache=skip_empty, _expert_gemm=gemm)


def _trace(cache, tmp_path, *, capacity=4096, max_files=8):
    trace = at.AdmissionTrace(cache, directory=tmp_path, rank=1, top_k=TOPK, max_rows=2,
                              capacity=capacity, max_files=max_files)
    cache._admission_trace = trace
    return trace


def _batch(rows, prefill=False):
    return SimpleNamespace(is_prefill=prefill, speculative=False, reqs=[SimpleNamespace(uid=10 + r) for r in range(rows)],
                           log_new_tokens=0)


# ------------------------------------------------------------------ settings

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(at.TRACE_DIR_ENV, raising=False)
    assert at.trace_settings() is None
    assert OffloadMoeCache(1, 4, 4, torch.device("cpu"))._admission_trace is None
    assert at.AdmissionTrace.from_env(_cache(), rank=0, top_k=TOPK, max_rows=2) is None


@pytest.mark.parametrize("value", ["relative/dir", "/nonexistent/four-lever"])
def test_bad_directories_are_refused(monkeypatch, value):
    monkeypatch.setenv(at.TRACE_DIR_ENV, value)
    with pytest.raises(ValueError):
        at.trace_settings()


def test_bounds_are_enforced(monkeypatch, tmp_path):
    monkeypatch.setenv(at.TRACE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(at.TRACE_RECORDS_ENV, "16")
    with pytest.raises(ValueError):
        at.trace_settings()
    monkeypatch.setenv(at.TRACE_RECORDS_ENV, "2048")
    monkeypatch.setenv(at.TRACE_FILES_ENV, "3")
    assert at.trace_settings() == dict(directory=tmp_path, capacity=2048, max_files=3)


def test_from_env_warms_before_returning_and_warm_is_inert_on_cpu(monkeypatch, tmp_path):
    monkeypatch.setenv(at.TRACE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(at.TRACE_RECORDS_ENV, "1024")
    calls = []
    real = at.AdmissionTrace.warm
    monkeypatch.setattr(at.AdmissionTrace, "warm", lambda self: calls.append(self) or real(self))
    trace = at.AdmissionTrace.from_env(_cache(), rank=1, top_k=TOPK, max_rows=2)
    assert calls == [trace] and not trace.armed
    assert int(trace.cursor.item()) == 0 and bool((trace.ring == -1).all())
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ exact records

STATE = ("slot_for_id", "id_of_slot", "usage", "step", "expert_recency")


def _drive(cache, trace, steps=40, seed=7):
    """Real CPU hybrid admission; returns the expected record of every call."""
    rng = random.Random(seed)
    expected = []
    for step in range(steps):
        rows = 1 if step % 3 else 2
        skip = rows == 1 and step % 2 == 0
        if trace is not None:
            trace.begin_forward(_batch(rows))
        for layer in range(LAYERS):
            ids = torch.tensor([[rng.randrange(EXPERTS) for _ in range(TOPK)] for _ in range(rows)], dtype=torch.int32)
            weights = torch.tensor([[rng.choice((0.0, 0.5, 1.0)) for _ in range(TOPK)] for _ in range(rows)])
            if step % 7 == 3:
                weights.zero_()
            raw = ids.clone()
            OffloadMoELayer._decode_hybrid(_layer(layer, skip), cache, torch.ones(rows, HIDDEN), weights, ids)
            n_fetch = int(cache.num_indices[0])
            expected.append(dict(layer=layer, ids=raw.reshape(-1).tolist(), guarded=skip,
                                 nonzero=int((weights != 0).sum()) if skip else -1,
                                 missing=int(cache.num_missing_full[0]), fetch=n_fetch,
                                 plan_slots=cache.evict_slots[:n_fetch].tolist(),
                                 plan_experts=cache.src_indices[:n_fetch].tolist()))
    return expected


def test_tracing_does_not_change_admission(tmp_path):
    plain, traced = _cache(), _cache()
    trace = _trace(traced, tmp_path)
    trace.arm()
    _drive(plain, None)
    _drive(traced, trace)
    for name in STATE:
        assert torch.equal(getattr(plain, name), getattr(traced, name)), name


def test_records_match_every_admission_and_flush_is_content_free(tmp_path):
    cache = _cache()
    trace = _trace(cache, tmp_path)
    _drive(cache, trace, steps=3)          # before arming: nothing is recorded
    start = {name: getattr(cache, name).clone() for name in STATE}
    trace.arm()
    expected = _drive(cache, trace)
    trace.note_invalidate(0)
    path = trace.flush("test")
    data = np.load(path.with_suffix(".npz"))
    meta = json.loads(path.with_suffix(".json").read_text())
    records = data["records"]
    assert meta["valid"] and meta["records"] == len(expected) == len(records)
    rk = meta["rk"]
    assert rk == max(2 * TOPK, (EXPERTS + 31) // 32) and records.shape[1] == 8 + 3 * rk
    for rec, exp in zip(records, expected):
        seq, kind, layer, n_ids, flags, nonzero, missing, fetch = rec[:8].tolist()
        assert kind == at.KIND_HYBRID and layer == exp["layer"] and n_ids == len(exp["ids"])
        assert rec[8:8 + n_ids].tolist() == exp["ids"]
        assert (flags & 1) == int(exp["guarded"]) and nonzero == exp["nonzero"]
        assert (missing, fetch) == (exp["missing"], exp["fetch"])
        assert rec[8 + rk:8 + rk + fetch].tolist() == exp["plan_slots"]
        assert rec[8 + 2 * rk:8 + 2 * rk + fetch].tolist() == exp["plan_experts"]
    seqs = sorted({int(r[0]) for r in records})
    assert seqs == [f["seq"] for f in meta["forwards"]]
    assert meta["forwards"][-1]["invalidations"] == [0]
    for name in STATE:
        assert np.array_equal(data[f"start_{name}"], start[name].numpy()), name
        assert np.array_equal(data[f"end_{name}"], getattr(cache, name).numpy()), name
    allowed = {"rank", "flush", "reason", "wall_time", "records", "overflow", "valid", "width", "rk",
               "top_k", "max_rows", "layers", "experts", "slots", "decode_target", "hybrid_max_fetch",
               "hybrid_fetch_fraction", "hybrid_fetch_by_recency", "prefill_overlap",
               "prefill_expert_start", "prefill_expert_count", "sparse_prefill_max_tokens", "forwards"}
    assert set(meta) == allowed
    assert all(np.issubdtype(data[k].dtype, np.integer) for k in data.files)


def test_sparse_prefill_bitmask_and_post_counts(tmp_path):
    cache = _cache()
    trace = _trace(cache, tmp_path)
    trace.arm()
    trace.begin_forward(_batch(1, prefill=True))
    routes = torch.tensor([[0, 31, 32, 39], [31, 5, 0, 7]], dtype=torch.int32)
    trace.record_sparse_pre(2, routes)
    cache.num_indices.fill_(6)
    trace.record_post(at.KIND_SPARSE)
    rec = trace.ring[0].tolist()
    assert rec[1:4] == [at.KIND_SPARSE, 2, EXPERTS] and rec[6:8] == [6, 6]
    word0 = rec[8] & 0xFFFFFFFF
    assert word0 == (1 << 0) | (1 << 5) | (1 << 7) | (1 << 31)
    assert rec[9] == (1 << 0) | (1 << 7)   # experts 32 and 39
    assert all(v == -1 for v in rec[8 + trace.rk:])


def test_overflow_is_bounded_and_marked_invalid(tmp_path):
    cache = _cache()
    trace = _trace(cache, tmp_path, capacity=1024)
    trace.arm()
    trace.capacity = 5  # shrink the CPU reference ring to force an overflow
    trace.begin_forward(_batch(1))
    for layer in range(8):
        trace.record_pre(at.KIND_LRU, layer % LAYERS, torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))
        trace.record_post(at.KIND_LRU)
    meta = json.loads(trace.flush("test").with_suffix(".json").read_text())
    assert meta["overflow"] == 3 and not meta["valid"]


def test_max_files_disarms(tmp_path):
    cache = _cache()
    trace = _trace(cache, tmp_path, max_files=1)
    trace.arm()
    trace.begin_forward(_batch(1))
    assert trace.flush("test") is not None and not trace.armed
    trace.arm()
    assert not trace.armed and int(trace.enabled.item()) == 0


def test_empty_flush_writes_nothing(tmp_path):
    cache = _cache()
    trace = _trace(cache, tmp_path)
    trace.arm()
    assert trace.flush("idle") is None and list(tmp_path.iterdir()) == []


def test_scheduler_idle_arms_then_flushes(tmp_path, monkeypatch):
    from freetoken.scheduler import scheduler as module
    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setattr(module.logger, "info", lambda *a, **k: None)
    cache = _cache()
    trace = _trace(cache, tmp_path)
    sched = Scheduler.__new__(Scheduler)
    sched.engine = SimpleNamespace(moe_offload_cache=cache)
    sched._admission_trace_idle()
    assert trace.armed and list(tmp_path.iterdir()) == []
    _drive(cache, trace, steps=2)
    sched._admission_trace_idle()
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".json", ".npz"]
    bare = Scheduler.__new__(Scheduler)
    bare._admission_trace_idle()  # no engine/cache/trace: nothing happens
