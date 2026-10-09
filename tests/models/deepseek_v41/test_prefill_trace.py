"""CPU-only tests for the opt-in DSV4.1 prefill critical-path trace.

CUDA events are replaced by a deterministic fake clock, so no device is touched.
"""

from __future__ import annotations

import json
from contextlib import contextmanager as contextmanager_factory
from types import SimpleNamespace

import pytest
import torch

import freetoken.models.deepseek_v41.moe as dsv41_moe
import freetoken.moe.prefill_trace as trace
from freetoken.layers.moe import ExpertParallelOffloadMoELayer


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.created = 0
        self.synchronized = 0
        self.stream_waits = []


class FakeComputeStream:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock

    def wait_event(self, event) -> None:
        self.clock.stream_waits.append(event)


class FakeEvent:
    clock: FakeClock

    def __init__(self, enable_timing: bool = False) -> None:
        assert enable_timing
        type(self).clock.created += 1
        self.time = None
        self.stream = None

    def record(self, stream) -> None:
        clock = type(self).clock
        clock.now += 1.5
        self.time = clock.now
        self.stream = stream

    def query(self) -> bool:
        return False

    def synchronize(self) -> None:
        type(self).clock.synchronized += 1

    def elapsed_time(self, other) -> float:
        if self.time is None or other.time is None:
            raise RuntimeError("event not recorded")
        return other.time - self.time


@pytest.fixture
def fake_cuda(monkeypatch, tmp_path):
    clock = FakeClock()
    event_type = type("Event", (FakeEvent,), {"clock": clock})
    compute = FakeComputeStream(clock)
    monkeypatch.setattr(trace.torch.cuda, "Event", event_type)
    monkeypatch.setattr(trace.torch.cuda, "current_stream", lambda device=None: compute)
    monkeypatch.setattr(trace.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(trace.torch.cuda, "get_device_name", lambda device=None: "fake")

    def device_sync(device=None):
        raise AssertionError("the trace must never synchronize the whole device")

    monkeypatch.setattr(trace.torch.cuda, "synchronize", device_sync)
    monkeypatch.setattr(trace, "_ENABLED", True)
    monkeypatch.setattr(trace, "_DIR", str(tmp_path))
    trace._reset_for_tests()
    yield clock
    trace._reset_for_tests()


def records(tmp_path):
    rows = []
    for path in sorted(tmp_path.glob("prefill-trace-rank*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def test_flag_off_creates_no_events_files_or_active_trace(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("tracing must stay inert when disabled")

    monkeypatch.setattr(trace, "_ENABLED", False)
    monkeypatch.setattr(trace, "_DIR", str(tmp_path))
    monkeypatch.setattr(trace.torch.cuda, "Event", forbidden)
    monkeypatch.setattr(trace.torch.cuda, "synchronize", forbidden)
    trace._reset_for_tests()
    with trace.chunk(rank=0, tokens=8192, device="cuda:0"):
        assert not trace.active()
        trace.set_layer(3)
        trace.mark("layer.begin")
        trace.mark("h2d.begin", layer=4, stream="copy", tag="copy")
        trace.note("h2d.hit_rows", 7)
    assert list(tmp_path.iterdir()) == []


def test_default_environment_is_off():
    assert trace.os.getenv("FREETOKEN_DSV41_PREFILL_TRACE") is None
    assert trace.enabled() is False


def test_event_accounting_and_schema(fake_cuda, tmp_path):
    with trace.chunk(rank=2, tokens=8192, device="cuda:0", start_pos=16384, prompt_len=520000):
        assert trace.active()
        trace.set_layer(0)
        trace.mark("layer.begin")
        trace.mark("h2d.begin", layer=1, stream="copy", tag="copy")
        trace.note("h2d.hit_rows", 5, layer=1)
        trace.mark("layer.end")
    assert not trace.active()
    chunk, write = records(tmp_path)
    assert chunk["kind"] == "chunk" and write["kind"] == "write"
    assert write["seq"] == chunk["seq"] == 0 and write["write_ns"] >= 0
    for key in (
        "schema", "rank", "pid", "seq", "tokens", "start_pos", "prompt_len", "device",
        "side_streams_joined",
        "begin_host_ns", "end_enqueue_host_ns", "anchor_host_ns",
        "anchor_complete_before_sync", "sync_wait_ns", "gpu_total_ms", "marks",
        "notes", "mark_count", "mark_host_ns",
    ):
        assert key in chunk, key
    assert chunk["schema"] == trace.SCHEMA
    assert (chunk["rank"], chunk["tokens"], chunk["start_pos"], chunk["prompt_len"]) == (
        2, 8192, 16384, 520000,
    )
    # begin=1.5, marks at 3.0/4.5/6.0, end=7.5: elapsed is measured from begin.
    assert [m[:4] for m in chunk["marks"]] == [
        [0, "layer.begin", "compute", 1.5],
        [1, "h2d.begin", "copy", 3.0],
        [0, "layer.end", "compute", 4.5],
    ]
    assert chunk["gpu_total_ms"] == 6.0
    assert chunk["mark_count"] == 3
    assert chunk["notes"] == [[1, "h2d.hit_rows", 5]]
    assert chunk["anchor_host_ns"] >= chunk["end_enqueue_host_ns"] >= chunk["begin_host_ns"]
    assert all(m[4] >= chunk["begin_host_ns"] for m in chunk["marks"])
    # 1 begin + 3 marks + 1 end; exactly one bounded event sync per chunk, after
    # the compute stream joined the copy stream's last mark on the device.
    assert fake_cuda.created == 5
    assert fake_cuda.synchronized == 1
    assert [e.stream for e in fake_cuda.stream_waits] == ["copy"]
    assert chunk["side_streams_joined"] == ["copy"]


def test_failed_forward_is_dropped_without_sync_and_next_chunk_is_clean(fake_cuda, tmp_path):
    with pytest.raises(ValueError, match="boom"):
        with trace.chunk(rank=0, tokens=8192, device="cuda:0"):
            trace.mark("layer.begin")
            raise ValueError("boom")
    assert not trace.active()
    assert fake_cuda.synchronized == 0 and fake_cuda.stream_waits == []
    with trace.chunk(rank=0, tokens=4096, device="cuda:0"):
        trace.mark("layer.begin")
    aborted, chunk, _write = records(tmp_path)
    assert aborted == {
        "schema": trace.SCHEMA, "kind": "aborted", "rank": 0, "pid": aborted["pid"],
        "seq": 0, "tokens": 8192, "start_pos": -1, "error": "ValueError",
    }
    assert chunk["seq"] == 1 and chunk["tokens"] == 4096 and chunk["mark_count"] == 1


def test_finalize_fault_never_fails_serving(fake_cuda, tmp_path, monkeypatch):
    def broken(_trace):
        raise RuntimeError("elapsed unavailable")

    monkeypatch.setattr(trace, "_finish", broken)
    with trace.chunk(rank=0, tokens=8192, device="cuda:0"):
        trace.mark("layer.begin")
    assert not trace.active()
    assert records(tmp_path) == []


def test_stream_capture_is_never_traced(fake_cuda, tmp_path, monkeypatch):
    monkeypatch.setattr(trace.torch.cuda, "is_current_stream_capturing", lambda: True)
    with trace.chunk(rank=0, tokens=1, device="cuda:0"):
        assert not trace.active()
        trace.mark("layer.begin")
    assert fake_cuda.created == 0
    assert records(tmp_path) == []


def test_missing_directory_is_reported_without_raising(fake_cuda, tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "_DIR", str(tmp_path / "missing"))
    with trace.chunk(rank=0, tokens=8192, device="cuda:0"):
        trace.mark("layer.begin")
    assert not trace.active()
    assert not (tmp_path / "missing").exists()


def test_start_filter_traces_only_registered_chunks(fake_cuda, tmp_path):
    trace._reset_for_tests("0,253952,507904")
    for start in (0, 8192, 245760, 253952, 262144, 507904, 516096, -1):
        with trace.chunk(rank=1, tokens=8192, device="cuda:1", start_pos=start,
                         prompt_len=520000):
            trace.set_layer(0)
            trace.mark("layer.begin")
            trace.note("h2d.hit_rows", 1)
    chunks = [r for r in records(tmp_path) if r["kind"] == "chunk"]
    # Sequence numbers count every prefill forward so ranks still align.
    assert [(r["seq"], r["start_pos"]) for r in chunks] == [(0, 0), (3, 253952), (5, 507904)]
    # Unregistered chunks allocate nothing: 3 traced x (begin + mark + end).
    assert fake_cuda.created == 9
    assert fake_cuda.synchronized == 3


def test_start_filter_parse_and_default():
    assert trace._parse_starts("") is None
    assert trace._parse_starts(" 0,253952 ,507904") == frozenset({0, 253952, 507904})
    with pytest.raises(ValueError):
        trace._parse_starts("0,-8192")
    assert trace._STARTS is None  # module default: flag off, no filter parsed


def test_worker_forward_passes_request_offsets(monkeypatch):
    import freetoken.models.deepseek_v41.model as model

    seen = {}

    @contextmanager_factory
    def fake_chunk(**kwargs):
        seen.update(kwargs)
        yield

    batch = SimpleNamespace(
        is_moe_prefill=True,
        reqs=[SimpleNamespace(cached_len=253952, input_ids=torch.zeros(520000))],
    )
    monkeypatch.setattr(model, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    monkeypatch.setattr(trace, "enabled", lambda: True)
    monkeypatch.setattr(trace, "chunk", fake_chunk)
    worker = model.DeepseekV41ExpertWorkerModel.__new__(model.DeepseekV41ExpertWorkerModel)
    worker.args = SimpleNamespace(dim=4)
    worker.engram_coordinator = SimpleNamespace(execution=SimpleNamespace(
        phase_aware=False, participates_in_prefill=True, rank=3))
    worker._forward_layers = lambda *a: seen.setdefault("layers_called", True)
    worker.forward(torch.zeros(8192, dtype=torch.long))
    assert seen["rank"] == 3 and seen["tokens"] == 8192
    assert seen["start_pos"] == 253952 and seen["prompt_len"] == 520000
    assert seen["layers_called"] is True


def test_worker_forward_untouched_when_disabled(monkeypatch):
    import freetoken.models.deepseek_v41.model as model

    monkeypatch.setattr(trace, "enabled", lambda: False)
    monkeypatch.setattr(model, "get_global_ctx",
                        lambda: pytest.fail("flag-off worker must not read the batch for tracing"))
    monkeypatch.setattr(trace, "chunk", lambda **k: pytest.fail("no trace when disabled"))
    worker = model.DeepseekV41ExpertWorkerModel.__new__(model.DeepseekV41ExpertWorkerModel)
    worker.args = SimpleNamespace(dim=4)
    worker.engram_coordinator = SimpleNamespace(execution=SimpleNamespace(phase_aware=False))
    called = []
    worker._forward_layers = lambda *a: called.append(a)
    worker.forward(torch.zeros(8, dtype=torch.long))
    assert len(called) == 1


# -- RoutedExperts overrides -------------------------------------------------


def routed(layer_id: int = 0):
    layer = dsv41_moe.RoutedExperts.__new__(dsv41_moe.RoutedExperts)
    layer.layer_id = layer_id
    return layer


def test_overrides_defer_to_base_when_inactive(monkeypatch):
    calls = []
    base = ExpertParallelOffloadMoELayer
    monkeypatch.setattr(base, "_wait_prefill_overlap",
                        lambda self, cache: calls.append("wait") or "views")
    monkeypatch.setattr(base, "_expert_gemm",
                        lambda self, cache, hidden, *a, is_prefill, **k: calls.append("gemm") or "out")
    monkeypatch.setattr(base, "prepare_packed_prefill_receive",
                        lambda self, *a, **k: calls.append("prep"))
    monkeypatch.setattr(base, "_maybe_combine_packed_prefill_routes",
                        lambda self, routes, weights: calls.append("combine") or "combined")
    monkeypatch.setattr(trace, "_active", None)
    monkeypatch.setattr(trace, "mark", lambda *a, **k: pytest.fail("no marks when inactive"))
    layer = routed()
    assert layer._wait_prefill_overlap(object()) == "views"
    assert layer._expert_gemm(object(), "h", "w", "i", views=(), n=1, alphas=None,
                              is_prefill=True) == "out"
    assert layer.prepare_packed_prefill_receive("w", torch.bfloat16) is None
    assert layer._maybe_combine_packed_prefill_routes("r", "w") == "combined"
    assert calls == ["wait", "gemm", "prep", "combine"]


class FakeCache:
    def __init__(self, *, stream=None, released=(False, False)) -> None:
        self.calls = []
        self.prefill_copy_stream = stream
        self.num_layers = 2
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_has_release_event = list(released)
        self.prefill_release_events = ["release0", "release1"]
        self.prefill_bank_buffers = [torch.zeros(2, 3, 4, dtype=torch.uint8)]
        self.prefill_expert_count = 3
        self.prefill_hit_rows = 0

    def begin_prefill(self):
        self.calls.append(("begin",))

    def prefetch_prefill_layer(self, layer):
        self.calls.append(("prefetch", layer))
        if layer < self.num_layers:
            self._prefill_buffer_layer[layer % 2] = layer
            self.prefill_hit_rows += 2

    def wait_prefill_layer(self, layer):
        self.calls.append(("wait", layer))
        return ("views", layer)


class FakeStream:
    def __init__(self, log) -> None:
        self.log = log

    def wait_event(self, event):
        self.log.append(("stream_wait", event))


def test_traced_overlap_mirrors_base_order_and_marks_copies(monkeypatch):
    log = []
    monkeypatch.setattr(trace, "_active", object())
    monkeypatch.setattr(trace, "mark", lambda name, **k: log.append(("mark", name, k.get("layer"))))
    monkeypatch.setattr(trace, "note", lambda key, value, **k: log.append(("note", key, value)))
    cache = FakeCache(stream=FakeStream(log), released=(True, True))
    cache.calls = log
    layer = routed(0)
    assert layer._wait_prefill_overlap(cache) == ("views", 0)
    assert log == [
        ("mark", "moe.prefetch.begin", None),
        ("begin",),
        ("note", "bank_feat_bytes", [4]),
        ("note", "prefill_expert_count", 3),
        ("stream_wait", "release0"),
        ("mark", "h2d.begin", 0),
        ("prefetch", 0),
        ("mark", "h2d.end", 0),
        ("note", "h2d.hit_rows", 2),
        ("stream_wait", "release1"),
        ("mark", "h2d.begin", 1),
        ("prefetch", 1),
        ("mark", "h2d.end", 1),
        ("note", "h2d.hit_rows", 2),
        ("mark", "moe.prefetch.end", None),
        ("wait", 0),
        ("mark", "moe.expert_wait.end", None),
    ]
    # Layer 1: its own buffer is already issued and layer 2 does not exist.
    log.clear()
    layer = routed(1)
    assert layer._wait_prefill_overlap(cache) == ("views", 1)
    assert log == [
        ("mark", "moe.prefetch.begin", None),
        ("prefetch", 1),
        ("prefetch", 2),
        ("mark", "moe.prefetch.end", None),
        ("wait", 1),
        ("mark", "moe.expert_wait.end", None),
    ]


def test_traced_gemm_and_combine_marks_bracket_base(monkeypatch):
    log = []
    base = ExpertParallelOffloadMoELayer
    monkeypatch.setattr(base, "_expert_gemm",
                        lambda self, cache, hidden, *a, is_prefill, **k: log.append("gemm") or "out")
    monkeypatch.setattr(base, "_maybe_combine_packed_prefill_routes",
                        lambda self, routes, weights: log.append("combine") or "combined")

    def prep(self, *a, **k):
        log.append("prep")
        self._packed_prefill_pending = ((), SimpleNamespace(name="recv"))

    monkeypatch.setattr(base, "prepare_packed_prefill_receive", prep)
    monkeypatch.setattr(trace, "_active", object())
    monkeypatch.setattr(trace, "mark", lambda name, **k: log.append((name, k.get("tag", "compute"))))
    layer = routed()
    layer._expert_gemm(object(), "h", "w", "i", views=(), n=1, alphas=None, is_prefill=True)
    layer._expert_gemm(object(), "h", "w", "i", views=(), n=None, alphas=None, is_prefill=False)
    layer.prepare_packed_prefill_receive("w", torch.bfloat16)
    layer._maybe_combine_packed_prefill_routes("r", "w")
    assert log == [
        ("moe.gemm.begin", "compute"), "gemm", ("moe.gemm.end", "compute"),
        "gemm",  # decode/GEMV entry stays unmarked
        ("moe.prep.begin", "compute"), "prep", ("moe.recv.end", "recv"),
        ("moe.prep.end", "compute"),
        "combine", ("moe.combine.end", "compute"),
    ]
