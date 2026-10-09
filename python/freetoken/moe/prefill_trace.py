"""Opt-in CUDA-event critical-path trace for DeepSeek-V4.1 bulk prefill chunks.

Disabled unless ``FREETOKEN_DSV41_PREFILL_TRACE=1``; then
``FREETOKEN_DSV41_PREFILL_TRACE_DIR`` names an existing directory that receives one
JSONL file per rank process.  Every ``mark`` records a timing event on a named
stream plus the host monotonic time at which it was enqueued.  Nothing synchronizes
until the end of a traced chunk.  There the compute stream first waits (on the
device, no host sync) for the last mark of each side stream, then one
``Event.synchronize`` on the chunk's final compute-stream event anchors that
device's event clock to the shared host ``CLOCK_MONOTONIC`` (all rank processes
run on one host).  That is the only synchronization per traced chunk.

``FREETOKEN_DSV41_PREFILL_TRACE_STARTS`` optionally lists the chunk start
positions to trace (for example ``0,253952,507904``).  Other chunks only advance
the per-rank chunk sequence: they allocate no events, never synchronize and
write nothing.  A chunk with an unknown start (-1) is not traced under a filter.

With the flag off, ``chunk`` yields immediately and ``mark``/``note`` return on a
single ``None`` check: no events, no synchronization, no files.  A failed forward
discards its pending events without synchronizing and leaves no active trace.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

SCHEMA = "dsv41-prefill-trace/1"
_ENABLED = os.getenv("FREETOKEN_DSV41_PREFILL_TRACE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DIR = os.getenv("FREETOKEN_DSV41_PREFILL_TRACE_DIR", "")


def _parse_starts(text: str) -> frozenset[int] | None:
    text = text.strip()
    if not text:
        return None
    starts = frozenset(int(item) for item in text.split(","))
    if any(start < 0 for start in starts):
        raise ValueError("FREETOKEN_DSV41_PREFILL_TRACE_STARTS must be non-negative")
    return starts


_STARTS = (
    _parse_starts(os.getenv("FREETOKEN_DSV41_PREFILL_TRACE_STARTS", ""))
    if _ENABLED
    else None
)


class _ChunkTrace:
    __slots__ = (
        "rank", "seq", "tokens", "start_pos", "prompt_len", "device", "begin",
        "begin_host_ns", "layer", "marks", "notes", "mark_host_ns",
    )

    def __init__(self, rank, seq, tokens, start_pos, prompt_len, device) -> None:
        self.rank = rank
        self.seq = seq
        self.tokens = tokens
        self.start_pos = start_pos
        self.prompt_len = prompt_len
        self.device = device
        self.layer = -1
        # (layer, name, stream_tag, event, host_ns); repeated names within one
        # layer (per route tile) are distinguished by their occurrence order.
        self.marks: list[tuple] = []
        self.notes: list[tuple] = []
        self.mark_host_ns = 0
        self.begin = torch.cuda.Event(enable_timing=True)
        self.begin_host_ns = time.monotonic_ns()
        self.begin.record(torch.cuda.current_stream(device))


_active: _ChunkTrace | None = None
_seq = 0
_file = None


def enabled() -> bool:
    return _ENABLED


def active() -> bool:
    return _active is not None


def _output(rank: int):
    global _file
    if _file is None:
        if not _DIR or not Path(_DIR).is_dir():
            raise RuntimeError(
                "FREETOKEN_DSV41_PREFILL_TRACE_DIR must name an existing directory"
            )
        path = Path(_DIR) / f"prefill-trace-rank{rank}-pid{os.getpid()}.jsonl"
        _file = path.open("a", buffering=1 << 20)
    return _file


def set_layer(layer: int) -> None:
    trace = _active
    if trace is None:
        return
    trace.layer = int(layer)


def mark(name: str, *, layer: int | None = None, stream=None, tag: str = "compute") -> None:
    """Record ``name`` for the current (or explicit) layer on ``stream``."""

    trace = _active
    if trace is None:
        return
    started = time.monotonic_ns()
    event = torch.cuda.Event(enable_timing=True)
    event.record(stream if stream is not None else torch.cuda.current_stream(trace.device))
    trace.marks.append(
        (
            trace.layer if layer is None else int(layer),
            name,
            tag,
            event,
            started,
        )
    )
    trace.mark_host_ns += time.monotonic_ns() - started


def note(key: str, value, *, layer: int | None = None) -> None:
    trace = _active
    if trace is None:
        return
    trace.notes.append((trace.layer if layer is None else int(layer), key, value))


def _finish(trace: _ChunkTrace) -> dict:
    stream = torch.cuda.current_stream(trace.device)
    # Join each side stream's last mark into the compute stream so the single
    # end-event wait below covers every event read by elapsed_time.
    last_side = {}
    for _layer, _name, tag, event, _host_ns in trace.marks:
        if tag != "compute":
            last_side[tag] = event
    for event in last_side.values():
        stream.wait_event(event)
    end = torch.cuda.Event(enable_timing=True)
    end_enqueue_ns = time.monotonic_ns()
    end.record(stream)
    complete_before_sync = bool(end.query())
    sync_begin_ns = time.monotonic_ns()
    end.synchronize()
    anchor_ns = time.monotonic_ns()
    total_ms = trace.begin.elapsed_time(end)
    marks = [
        [layer, name, tag, round(trace.begin.elapsed_time(event), 4), host_ns]
        for layer, name, tag, event, host_ns in trace.marks
    ]
    return dict(
        schema=SCHEMA,
        kind="chunk",
        rank=trace.rank,
        pid=os.getpid(),
        seq=trace.seq,
        tokens=trace.tokens,
        start_pos=trace.start_pos,
        prompt_len=trace.prompt_len,
        device=torch.cuda.get_device_name(trace.device),
        begin_host_ns=trace.begin_host_ns,
        end_enqueue_host_ns=end_enqueue_ns,
        anchor_host_ns=anchor_ns,
        anchor_complete_before_sync=complete_before_sync,
        sync_wait_ns=anchor_ns - sync_begin_ns,
        side_streams_joined=sorted(last_side),
        gpu_total_ms=round(total_ms, 4),
        marks=marks,
        notes=[list(item) for item in trace.notes],
        mark_count=len(marks),
        mark_host_ns=trace.mark_host_ns,
    )


@contextmanager
def chunk(*, rank: int, tokens: int, device, start_pos: int = -1, prompt_len: int = -1):
    """Trace one prefill forward on this rank; a no-op unless enabled."""

    global _active, _seq
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        yield
        return
    if _active is not None:
        logger.warning("DSV41 prefill trace: discarding an unfinished chunk")
        _active = None
    seq = _seq
    _seq += 1  # every rank counts every prefill forward, traced or not
    if _STARTS is not None and int(start_pos) not in _STARTS:
        yield
        return
    _active = _ChunkTrace(rank, seq, int(tokens), int(start_pos), int(prompt_len), device)
    try:
        yield
    except BaseException as exc:
        trace, _active = _active, None
        # Never synchronize a failed forward; its events are simply dropped.
        try:
            _output(trace.rank).write(json.dumps(dict(
                schema=SCHEMA, kind="aborted", rank=trace.rank, pid=os.getpid(),
                seq=trace.seq, tokens=trace.tokens, start_pos=trace.start_pos,
                error=type(exc).__name__,
            )) + "\n")
        except Exception:  # noqa: BLE001 -- the original failure must propagate
            pass
        raise
    trace, _active = _active, None
    try:
        record = _finish(trace)
        write_begin = time.monotonic_ns()
        handle = _output(trace.rank)
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        record_write_ns = time.monotonic_ns() - write_begin
        handle.write(json.dumps(dict(
            schema=SCHEMA, kind="write", rank=trace.rank, seq=trace.seq,
            write_ns=record_write_ns,
        )) + "\n")
        handle.flush()
    except Exception as exc:  # noqa: BLE001 -- a trace fault must not fail serving
        logger.warning(f"DSV41 prefill trace failed for chunk {trace.seq}: {exc!r}")


def _reset_for_tests(starts: str = "") -> None:
    global _active, _seq, _file, _STARTS
    if _file is not None:
        _file.close()
    _active, _seq, _file = None, 0, None
    _STARTS = _parse_starts(starts)


__all__ = ["active", "chunk", "enabled", "mark", "note", "set_layer"]
