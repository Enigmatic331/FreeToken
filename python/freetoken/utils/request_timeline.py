"""Opt-in, content-free request boundary timestamps (measurement only, default off).

``FREETOKEN_REQUEST_TIMELINE=1`` makes each serving process keep stamps for request boundaries
and log ONE line per request per process when its part of the request ends::

    REQUEST_TIMELINE {"proc":"frontend","uid":12,"t":{"uid_assigned":...},"n":{...}}

Clock domain: every stamp is ``time.monotonic_ns()`` (Linux ``CLOCK_MONOTONIC``), which is
shared by all processes on the host -- the API frontend, the tokenizer/detokenizer worker, the
scheduler ranks -- and by a benchmark client running on the same host, so stamps taken in
different processes are directly subtractable. GPU work is only bracketed by host stamps; the
scheduler may add CUDA-event *durations* (device-side elapsed time), never device timestamps.

Content: only the integer request uid, fixed stage names, monotonic stamps and integer counts
(token counts, chunk sizes) are recorded. No text, token ids, headers or client identities.
Bounded: at most ``max_open`` requests per process are tracked; the oldest is dropped and
counted if one never finishes. Unset or ``0`` disables everything; other values are refused.
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from typing import Callable

TIMELINE_ENV = "FREETOKEN_REQUEST_TIMELINE"
TIMELINE_PREFIX = "REQUEST_TIMELINE "
MAX_OPEN = 512


def timeline_enabled() -> bool:
    """Strict parse: unset/``0`` is off and ``1`` is on."""
    value = os.environ.get(TIMELINE_ENV)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{TIMELINE_ENV} must be '0' or '1', got {value!r}")


def timeline_now(owner) -> int:
    """CLOCK_MONOTONIC ns if ``owner._request_timeline`` is enabled, else 0 (no further work).

    A plain function rather than a method so that any object -- including the light stubs
    tests pass to unbound scheduler methods -- simply reads as "disabled".
    """
    timeline = getattr(owner, "_request_timeline", None)
    return time.monotonic_ns() if timeline is not None and timeline.enabled else 0


class RequestTimeline:
    """Per-process stamps keyed by request uid. Every method is a no-op when disabled.

    ``begin`` opens a request; stamps for unknown uids are ignored, so a request that already
    emitted its line (or was never opened in this process) cannot be recreated by late stamps.
    """

    def __init__(self, proc: str, log: Callable[[str], None], *, enabled: bool | None = None,
                 clock: Callable[[], int] = time.monotonic_ns, max_open: int = MAX_OPEN) -> None:
        self.proc = proc
        self.enabled = timeline_enabled() if enabled is None else bool(enabled)
        self._log = log
        self._clock = clock
        self._max_open = max_open
        self._open: OrderedDict[int, dict] = OrderedDict()
        self.dropped = 0

    def now(self) -> int:
        return self._clock()

    def begin(self, uid: int, stage: str | None = None, *, t: int | None = None) -> None:
        if not self.enabled or uid in self._open:
            return
        if len(self._open) >= self._max_open:
            self._open.popitem(last=False)
            self.dropped += 1
        self._open[uid] = dict(t={}, n={})
        if stage is not None:
            self.stamp(uid, stage, t=t)

    def active(self, uid: int) -> bool:
        return self.enabled and uid in self._open

    def stamp(self, uid: int, stage: str, *, t: int | None = None, once: bool = True) -> None:
        if not self.enabled:
            return
        entry = self._open.get(uid)
        if entry is None or (once and stage in entry["t"]):
            return
        entry["t"][stage] = self._clock() if t is None else int(t)

    def count(self, uid: int, name: str, value) -> None:
        if not self.enabled:
            return
        entry = self._open.get(uid)
        if entry is not None:
            entry["n"][name] = value

    def add(self, uid: int, name: str, value=1) -> None:
        if not self.enabled:
            return
        entry = self._open.get(uid)
        if entry is not None:
            entry["n"][name] = entry["n"].get(name, 0) + value

    def append(self, uid: int, name: str, value) -> None:
        if not self.enabled:
            return
        entry = self._open.get(uid)
        if entry is not None:
            entry["n"].setdefault(name, []).append(value)

    def finish(self, uid: int, stage: str | None = None) -> dict | None:
        """Emit the uid's line (once) and forget it; returns the emitted record."""
        if not self.enabled:
            return None
        if stage is not None:
            self.stamp(uid, stage)
        entry = self._open.pop(uid, None)
        if entry is None:
            return None
        record = dict(proc=self.proc, uid=uid, t=entry["t"], n=entry["n"])
        if self.dropped:
            record["dropped"] = self.dropped
        self._log(TIMELINE_PREFIX + json.dumps(record, separators=(",", ":"), sort_keys=True))
        return record

    def discard(self, uid: int) -> None:
        if self.enabled:
            self._open.pop(uid, None)


def parse_timeline_line(message: str) -> dict | None:
    """Inverse of ``finish``'s log line; tolerates logger prefixes before the marker."""
    index = message.find(TIMELINE_PREFIX)
    if index < 0:
        return None
    return json.loads(message[index + len(TIMELINE_PREFIX):])


__all__ = [
    "MAX_OPEN",
    "RequestTimeline",
    "TIMELINE_ENV",
    "TIMELINE_PREFIX",
    "parse_timeline_line",
    "timeline_enabled",
    "timeline_now",
]
