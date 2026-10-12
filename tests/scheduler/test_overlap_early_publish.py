"""Overlap scheduling must not hold a completed reply behind another request's prefill.

``overlap_loop`` launches batch N+1 before draining batch N. A prefill forward blocks the host
for the whole prefill (DSV4.1 sparse-prefill expert copies: ~1.7 s for a 128-token tail at 4K),
so when N+1 is request B's prefill, request A's already-computed token from batch N used to be
published only after B's prefill finished: A's TTFT under concurrency became P_A + P_B
(analysis-preeos-20261012). The fix drains N before scheduling whenever a prefill is pending
and N has a reply to publish -- normal_loop's order, already used for speculative steps.

These tests drive the real ``overlap_loop`` / ``_process_last_data`` / abort handler against
CPU-built radix managers. Only scheduling-side preparation and the forward are scripted: the
fake forward advances each request like the engine (``complete_one`` + ``filter_reqs``) and
records, at launch time, which earlier publishable batches are still undrained.
"""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import AbortBackendMsg, DetokenizeMsg, ErrorReplyMsg
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

WIDTH = 96
MAX_RUNNING = 4
A, B, C = 1, 2, 3


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _token(uid: int, step: int) -> int:
    return 1000 * uid + step


class _Harness:
    def __init__(self, *, prefill_budget: int = 64, eos=()):
        _setup_context()
        pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32)
        cm = CacheManager(num_pages=256, page_size=1, page_table=pt, type="radix")
        tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
        dm = DecodeManager(page_size=1)
        pm = PrefillManager(cm, tm, dm)
        self.cm, self.tm, self.dm, self.pm = cm, tm, dm, pm
        self.log: list[tuple] = []
        self.reqs: dict[int, object] = {}
        self.steps: dict[int, int] = {}
        self.admit_hits: dict[int, int] = {}
        self.launched: list = []
        self.drained: list[int] = []  # id() of drained batches
        self.inbox: list = []
        self.wait_calls_per_iter: list[int] = []

        s = Scheduler.__new__(Scheduler)
        s._idle_checks_done = False
        s._last_data = None
        s._pending_rebuild = None
        s._speculative = None
        s._pending_abort_acks = set()
        s.prefill_manager, s.decode_manager = pm, dm
        s.cache_manager, s.table_manager = cm, tm
        s.prefill_budget = prefill_budget
        s.finished_reqs = set()
        s.eos_token_ids = set(eos)
        s.toolcall_anchor_id = None
        s.config = SimpleNamespace(page_size=1)
        s.status_reporter = SimpleNamespace(report_batch=lambda *_, **__: None)
        s.stream = Mock()
        s.engine = SimpleNamespace(stream=Mock())
        s.engine_stream_ctx = nullcontext()
        s._kv_usage_pages = cm.page_usage
        s._mamba_slot_usage = lambda: None
        s._swa_token_usage = lambda: None
        s._gpu_mem_bytes = lambda: 0
        s._restore_linear_states = Mock()
        s.receive_msg = self._receive
        s.send_result = self._send
        s._process_one_msg = self._process_one_msg
        s._schedule_next_batch = self._schedule
        s._forward = self._forward
        s._process_last_data = self._drain
        self.s = s

    # -- scripted pieces -------------------------------------------------------------------
    def _receive(self, *, blocking):
        msgs, self.inbox = self.inbox, []
        return msgs

    def _process_one_msg(self, msg):
        if isinstance(msg, tuple):  # ("user", uid, prompt_len, max_tokens[, prompt_key])
            _, uid, prompt_len, max_tokens, *key = msg
            base = (key[0] if key else uid) * 100
            prompt = torch.arange(base, base + prompt_len, dtype=torch.int32)
            self.pm.pending_list.append(
                PendingReq(uid, prompt, SamplingParams(max_tokens=max_tokens))
            )
        else:
            Scheduler._process_one_msg(self.s, msg)

    def _schedule(self):
        batch = self.pm.schedule_next_batch(self.s.prefill_budget) or self.dm.schedule_next_batch()
        if batch is None:
            return None
        self.cm.allocate_paged(batch.reqs)  # _prepare_batch's allocation
        for req in batch.reqs:
            self.reqs[req.uid] = req  # a final chunk replaces its ChunkedReq predecessors
            if batch.is_prefill:
                self.admit_hits.setdefault(req.uid, req.cached_len)  # first chunk's prefix hit
        return SimpleNamespace(batch=batch)

    def _forward(self, forward_input):
        batch = forward_input.batch
        undrained = [
            b for b in self.launched  # Batch is a value-eq dataclass: compare identities
            if id(b) not in self.drained and any(not isinstance(r, ChunkedReq) for r in b.reqs)
        ]
        uids = tuple(r.uid for r in batch.reqs)
        self.log.append(("launch", batch.phase, uids, len(undrained)))
        self.launched.append(batch)
        tokens = []
        for req in batch.reqs:  # engine.forward_batch advances the shared Req
            req.complete_one()
            if isinstance(req, ChunkedReq):  # an intermediate chunk's sample is discarded
                tokens.append(0)
                continue
            self.steps[req.uid] = self.steps.get(req.uid, 0) + 1
            tokens.append(_token(req.uid, self.steps[req.uid]))
        self.dm.filter_reqs(batch.reqs)
        return (None, torch.tensor(tokens, dtype=torch.int32),
                SimpleNamespace(synchronize=lambda: None))

    def _drain(self, last_data):
        if last_data is not None:
            batch = last_data[0].batch
            self.log.append(("drain", batch.phase, tuple(r.uid for r in batch.reqs)))
            assert id(batch) not in self.drained, "batch drained twice"
            self.drained.append(id(batch))
        Scheduler._process_last_data(self.s, last_data)

    def _send(self, msgs):
        for m in msgs:
            if isinstance(m, DetokenizeMsg):
                self.log.append(("publish", m.uid, m.next_token, m.finished, m.finish_reason))
            elif isinstance(m, ErrorReplyMsg):
                self.log.append(("ack", m.uid))
            else:
                raise AssertionError(f"unexpected reply {m!r}")

    # -- driver ----------------------------------------------------------------------------
    def step(self, data, *msgs):
        self.inbox = list(msgs)
        before = self.s.stream.wait_stream.call_count
        data = self.s.overlap_loop(data)
        self.wait_calls_per_iter.append(self.s.stream.wait_stream.call_count - before)
        return data

    def run_to_idle(self, data, limit=200):
        for _ in range(limit):
            if data is None and not (self.pm.runnable or self.dm.runnable):
                return
            data = self.step(data)
        raise AssertionError("scheduler did not go idle")

    # -- views -----------------------------------------------------------------------------
    def index(self, pred) -> int:
        return next(i for i, e in enumerate(self.log) if pred(e))

    def publishes(self, uid):
        return [e for e in self.log if e[0] == "publish" and e[1] == uid]

    def assert_clean_finish(self, expected):
        for uid, (n_tokens, reason) in expected.items():
            pubs = self.publishes(uid)
            assert [e[2] for e in pubs] == [_token(uid, k) for k in range(1, n_tokens + 1)]
            assert [e[3] for e in pubs] == [False] * (n_tokens - 1) + [True]
            assert pubs[-1][4] == reason
            assert self.reqs[uid].table_idx == -1
        assert not self.dm.runnable and not self.pm.runnable
        assert len(self.drained) == len(self.launched)
        self.cm.check_integrity()


def _is_launch(phase, uid):
    return lambda e: e[0] == "launch" and e[1] == phase and uid in e[2]


def _is_publish(uid, step):
    return lambda e: e[0] == "publish" and e[1] == uid and e[2] == _token(uid, step)


def test_prior_prefill_reply_is_published_before_next_prefill_launches():
    """The measured B2 case: B arrives while A's prefill runs. A's first token must be
    published before B's (host-blocking) prefill is launched, not after it returns."""
    h = _Harness()
    data = h.step(None, ("user", A, 16, 4))
    data = h.step(data, ("user", B, 16, 4))

    assert h.index(_is_publish(A, 1)) < h.index(_is_launch("prefill", B))
    b_launch = h.log[h.index(_is_launch("prefill", B))]
    assert b_launch[3] == 0  # nothing publishable was left undrained behind B's prefill

    h.run_to_idle(data)
    h.assert_clean_finish({A: (4, "length"), B: (4, "length")})
    assert h.index(_is_publish(A, 1)) < h.index(_is_publish(B, 1))


def test_running_decode_token_is_published_before_new_prefill_launches():
    """A is decoding when B arrives: A's in-flight decode token is published first."""
    h = _Harness()
    data = h.step(None, ("user", A, 16, 8))
    for _ in range(3):
        data = h.step(data)
    assert h.steps[A] == 4  # prefill + 3 decode forwards launched; the last is in flight
    data = h.step(data, ("user", B, 16, 4))

    assert h.index(_is_publish(A, 4)) < h.index(_is_launch("prefill", B))
    assert h.log[h.index(_is_launch("prefill", B))][3] == 0

    h.run_to_idle(data)
    h.assert_clean_finish({A: (8, "length"), B: (4, "length")})


def test_decode_only_overlap_ordering_is_unchanged():
    """Control: with no prefill pending, decode N+1 still launches before decode N drains."""
    h = _Harness()
    data = h.step(None, ("user", A, 16, 12), ("user", B, 16, 12))  # one batched prefill
    data = h.step(data)                                             # first decode
    start = len(h.log)
    start_iter = len(h.wait_calls_per_iter)
    for _ in range(6):
        data = h.step(data)

    events = [e for e in h.log[start:] if e[0] in ("launch", "drain")]
    assert [e[0] for e in events] == ["launch", "drain"] * 6
    assert all(e[1] == "decode" and e[2] == (A, B) for e in events)
    assert all(e[3] == 1 for e in events if e[0] == "launch")  # exactly one batch overlapped
    assert h.wait_calls_per_iter[start_iter:] == [2] * 6         # no extra drain-side wait

    h.run_to_idle(data)
    h.assert_clean_finish({A: (12, "length"), B: (12, "length")})


def test_intermediate_chunks_keep_overlap_but_final_chunk_publishes_first():
    """An intermediate-chunk-only batch has nothing to publish, so chunk k+1 still launches
    before chunk k drains. A final chunk is a reply: it drains before the next prefill."""
    h = _Harness(prefill_budget=8)
    data = h.step(None, ("user", A, 8, 12))
    data = h.step(data)  # A decoding
    data = h.step(data, ("user", B, 24, 4), ("user", C, 8, 4))

    assert h.index(_is_publish(A, 2)) < h.index(_is_launch("prefill", B))
    for _ in range(3):
        data = h.step(data)
    prefills = [e for e in h.log if e[0] in ("launch", "drain") and e[1] == "prefill"
                and B in e[2]]
    # chunk 1 launch, chunk 2 launch, chunk 1 drain, chunk 3 (final) launch, chunk 2 drain
    assert [e[0] for e in prefills[:5]] == ["launch", "launch", "drain", "launch", "drain"]
    # B's final chunk carries B's first token: it drains before C's prefill launches.
    assert h.index(_is_publish(B, 1)) < h.index(_is_launch("prefill", C))
    assert h.log[h.index(_is_launch("prefill", C))][3] == 0

    h.run_to_idle(data)
    h.assert_clean_finish({A: (12, "length"), B: (4, "length"), C: (4, "length")})


@pytest.mark.parametrize("decode_steps", [0, 2], ids=["abort-inflight-prefill",
                                                      "abort-inflight-decode"])
def test_abort_of_inflight_batch_acks_before_next_prefill(decode_steps):
    """Abort arriving with B: the early drain frees A once, sends no DetokenizeMsg for the
    aborted forward, and the ack (A's terminal reply) precedes B's prefill launch."""
    h = _Harness()
    data = h.step(None, ("user", A, 16, 8))
    for _ in range(decode_steps):
        data = h.step(data)
    a_pubs_before = len(h.publishes(A))
    data = h.step(data, AbortBackendMsg(uid=A), ("user", B, 16, 4))

    assert h.index(lambda e: e == ("ack", A)) < h.index(_is_launch("prefill", B))
    assert h.reqs[A].table_idx == -1
    assert all(A not in e[2] for e in h.log if e[0] == "launch" and e[1] == "prefill"
               and B in e[2])

    h.run_to_idle(data)
    assert len(h.publishes(A)) == a_pubs_before          # nothing after the abort
    assert [e for e in h.log if e[0] == "ack"] == [("ack", A)]
    assert all(A not in e[2] for e in h.log[h.index(lambda e: e == ("ack", A)):]
               if e[0] == "launch")
    h.assert_clean_finish({B: (4, "length")})


@pytest.mark.parametrize("reason", ["length", "stop"])
def test_terminal_reply_is_published_once_before_next_prefill(reason):
    """A's in-flight forward is its last (max_tokens or EOS) when B arrives: A finishes
    exactly once, before B's prefill, and is never scheduled again."""
    eos = (_token(A, 3),) if reason == "stop" else ()
    h = _Harness(eos=eos)
    data = h.step(None, ("user", A, 16, 3 if reason == "length" else 16))
    data = h.step(data)
    data = h.step(data)
    assert h.steps[A] == 3  # A's 3rd token is in flight
    data = h.step(data, ("user", B, 16, 4))

    final = h.index(_is_publish(A, 3))
    assert h.log[final][3:] == (True, reason)
    assert final < h.index(_is_launch("prefill", B))
    assert all(A not in e[2] for e in h.log[final:] if e[0] == "launch")

    h.run_to_idle(data)
    h.assert_clean_finish({A: (3, reason), B: (4, "length")})


def test_duplicate_prompt_admitted_after_early_drain_reuses_committed_prefix():
    """Behaviour change pinned on purpose: draining A's prefill now commits A's prompt to the
    radix cache (cache_req(finished=False)) before B is admitted -- normal_loop's order -- so
    an identical prompt B gets a prefix hit it used to miss. Locks and frees stay balanced."""
    h = _Harness()
    data = h.step(None, ("user", A, 16, 4, 7))
    data = h.step(data, ("user", B, 16, 4, 7))

    assert h.admit_hits[A] == 0
    assert h.admit_hits[B] == 15  # match_req keeps the prompt's last token to compute
    h.run_to_idle(data)
    h.assert_clean_finish({A: (4, "length"), B: (4, "length")})
    assert h.cm.prefix_cache.size_info.protected_size == 0


def test_speculative_drain_still_runs_once_with_prefill_pending():
    """Speculative steps already drained before scheduling; a pending prefill must not add
    a second drain of the same batch."""
    s = Scheduler.__new__(Scheduler)
    s._idle_checks_done = False
    s._last_data = None
    s._pending_rebuild = None
    s._speculative = object()
    s.prefill_manager = SimpleNamespace(runnable=True)
    s.decode_manager = SimpleNamespace(runnable=True)
    s.stream = Mock()
    s.engine = SimpleNamespace(stream=Mock())
    s.engine_stream_ctx = nullcontext()
    s.receive_msg = Mock(return_value=[])
    s._process_one_msg = Mock()
    s._schedule_next_batch = Mock(return_value=SimpleNamespace(batch=object()))
    s._restore_linear_states = Mock()
    s._forward = Mock(return_value=object())
    s._process_last_data = Mock()
    s._flush_abort_acks = Mock()
    last_data = object()

    s.overlap_loop(last_data)

    s._process_last_data.assert_called_once_with(last_data)
    s._flush_abort_acks.assert_called_once()
