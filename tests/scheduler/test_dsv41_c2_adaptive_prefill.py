"""DeepSeek-V4.1 gated C2 with the C1 adaptive prefill optimisations (CPU, no model, no GPU).

C2 keeps up to two requests resident but forwards one prefill request at a time, and a pending
prefill pauses decode. Another request may therefore be resident -- holding its row, pages and
window -- without being forwarded while a prompt pairs its tail or runs its dependency cone.

These tests drive the real ``overlap_loop`` (early publish included), batch preparation, tail-pair
and cone planning, forward bookkeeping, abort handling and drains over a real DSV4.1 page pool and
SWA radix tree with two table rows. Each C2 run with an optimisation is compared with the same C2
run without it: the same logical chunks, decode batches, replies and final cache.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import freetoken.scheduler.scheduler as scheduler_mod
from freetoken.core import Batch, Req, SamplingParams
from freetoken.message import AbortBackendMsg, DetokenizeMsg, PromptAdmittedMsg
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.scheduler import dependency_cone as dc
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.tail_pairing import (
    MAX_TAIL_TOKENS,
    TAIL_PAIRING_ENV,
    plan_tail_pair,
)
from freetoken.scheduler.utils import PendingReq

from .test_dsv41_dependency_cone_lifecycle import REAL_RATIOS, _geometry
from .test_dsv41_tail_pairing import (  # noqa: F401 -- cpu_scheduler_helpers is a fixture
    CHUNK,
    P,
    SPARSE,
    _limits,
    _req,
    _scheduled_chunk,
    _scheduler,
    _StubEngine,
    _tree_state,
    cpu_scheduler_helpers,
)

pytestmark = pytest.mark.usefixtures("cpu_scheduler_helpers")

A, B, C = 1, 2, 3
CONCURRENCY_ENV = "FREETOKEN_DSV41_CONCURRENCY"


class _C2Engine(_StubEngine):
    """Every row of every launched forward; a C2 decode forward may hold two rows."""

    def __init__(self, page_table, events):
        super().__init__(page_table)
        self.stream = Mock()
        self.launches = []
        self.events = events

    def forward_batch(self, batch, args):
        paired = batch.paired_head
        launch = SimpleNamespace(
            phase=batch.phase,
            rows=_rows(batch),
            head=None if paired is None else _rows(paired.batch),
            cone=batch.dsv41_cone,
            role=batch.tail_pair_role,
        )
        self.launches.append(launch)
        self.events.append(("launch", launch))
        return super().forward_batch(batch, args)


def _rows(batch):
    return tuple((r.uid, r.cached_len, r.device_len) for r in batch.reqs)


def _c2(*, pairing=False, cone=None, num_pages=512, win_pages=320):
    """A C2 scheduler: two table rows, one request per prefill forward (DeepSeek-V4.1's
    prefill_batch_size_limit), the C2 tail-pair limits, and overlap_loop's stream surface."""
    sched, pool = _scheduler(pairing=pairing, num_pages=num_pages, win_pages=win_pages,
                             max_running=2)
    cm, tm, dm = sched.cache_manager, sched.table_manager, sched.decode_manager
    sched.prefill_manager = PrefillManager(cm, tm, dm, max_batch_size=1)
    sched._tail_pairing = replace(_limits(), resident_decode_rows=1) if pairing else None
    sched._dependency_cone = cone
    sched.events = []
    sched.engine = _C2Engine(sched.engine.page_table, sched.events)
    sched.stream = Mock()
    sched.engine_stream_ctx = nullcontext()
    sched._idle_checks_done = False
    sched._pending_rebuild = None
    sched.prompt_lens = {}
    sched.inbox = []
    send = sched.send_result

    def record(msgs):
        sched.events.extend(("sent", m) for m in msgs)
        send(msgs)

    def receive(*, blocking):
        msgs, sched.inbox = sched.inbox, []
        return msgs

    sched.send_result = record
    sched.receive_msg = receive
    return sched, pool


def _submit(sched, uid, prompt_len, *, max_tokens):
    """``uid``'s prompt arrives; different uids share no prefix."""
    ids = (torch.arange(prompt_len, dtype=torch.int32) + 13 * uid) % 90 + 1
    sched.prompt_lens[uid] = prompt_len
    sched.prefill_manager.pending_list.append(PendingReq(
        uid=uid, input_ids=ids, sampling_params=SamplingParams(max_tokens=max_tokens)))
    return ids


def _idle(sched, data):
    return data is None and not (sched.prefill_manager.runnable or sched.decode_manager.runnable)


def _run(sched, data=None, *, until=None, limit=800):
    """Real ``overlap_loop`` iterations until ``until(sched)`` holds, or until idle."""
    for _ in range(limit):
        if until(sched) if until is not None else _idle(sched, data):
            return data
        data = sched.overlap_loop(data)
    raise AssertionError("the C2 scheduler did not reach the condition")


def _abort(sched, data, uid):
    """One overlap iteration that receives ``uid``'s abort."""
    sched.inbox = [AbortBackendMsg(uid=uid)]
    return sched.overlap_loop(data)


def _published(uid):
    return lambda s: sum(isinstance(m, DetokenizeMsg) and m.uid == uid for m in s.sent)


def _launched(pred):
    return lambda s: any(pred(launch) for launch in s.engine.launches)


def _replies(sched, uid=None):
    return [(m.uid, m.next_token, m.finished, m.finish_reason) for m in sched.sent
            if isinstance(m, DetokenizeMsg) and uid in (None, m.uid)]


def _acks(sched):
    return [m.uid for m in sched.sent if getattr(m, "error", None) == "request aborted"]


def _prefills(sched):
    return [launch for launch in sched.engine.launches if launch.phase == "prefill"]


def _decodes(sched):
    return [tuple(row[0] for row in launch.rows)
            for launch in sched.engine.launches if launch.phase == "decode"]


def _spans(sched, uid):
    """``uid``'s logical prefill chunks, whatever forward grouping ran them."""
    return [row[1:] for launch in _prefills(sched)
            for row in (launch.head or ()) + launch.rows if row[0] == uid]


def _evictions(sched):
    calls = []
    pc = sched.cache_manager.prefix_cache
    real = pc.evict_swa
    pc.evict_swa = lambda n: calls.append(n) or real(n)
    return calls


def _assert_c2_invariants(sched, aborted=()):
    """On every launched forward: a prefill forward is one request (a paired tail and its head
    are one request's chunks) with no decode row; nothing decodes while a prompt is between its
    first and final chunk (unless that prompt was aborted); a decode forward holds one or two
    rows and no tail/cone role."""
    mid_prefill = None
    for launch in sched.engine.launches:
        if launch.phase == "decode":
            assert mid_prefill in (None, *aborted), "a decode ran during a serialized prefill"
            mid_prefill = None
            assert 1 <= len(launch.rows) <= 2
            assert launch.head is None and launch.cone is None and launch.role is None
            continue
        (uid, _start, end), = launch.rows
        assert launch.head is None or {row[0] for row in launch.head} == {uid}
        assert mid_prefill in (None, uid), "two prompts interleaved their prefill chunks"
        mid_prefill = None if end == sched.prompt_lens[uid] else uid


def _finish(sched, *expected, aborted=()):
    """Every listed request ran to its length limit; rows, pages and window are all returned."""
    for uid in expected:
        replies = _replies(sched, uid)
        assert [finished for *_, finished, _reason in replies] == [False] * (len(replies) - 1) + [
            True]
        assert replies[-1][3] == "length"
    _assert_c2_invariants(sched, aborted)
    assert not sched.decode_manager.runnable and not sched.prefill_manager.runnable
    assert sched.table_manager.available_size == 2
    sched.cache_manager.check_integrity()  # free + tree == capacity: nothing leaked or doubled


def _two_lane_run(*, pairing=False, cone=None, prompt_len, a_len=300, a_tokens=24, b_tokens=6,
                  **pools):
    """A decodes; B arrives after A's third token and prefills beside the paused A."""
    sched, pool = _c2(pairing=pairing, cone=cone, **pools)
    a = _submit(sched, A, a_len, max_tokens=a_tokens)
    data = _run(sched, until=lambda s: _published(A)(s) >= 3)
    evictions = _evictions(sched)
    b = _submit(sched, B, prompt_len, max_tokens=b_tokens)
    _run(sched, data)
    _finish(sched, A, B)
    return sched, dict(
        spans=(_spans(sched, A), _spans(sched, B)),
        decodes=_decodes(sched),
        replies=_replies(sched),
        admissions=[(m.uid, m.prompt_tokens, m.cached_tokens)
                    for m in sched.sent if isinstance(m, PromptAdmittedMsg)],
        tree=(_tree_state(sched.cache_manager, pool, a),
              _tree_state(sched.cache_manager, pool, b)),
        evictions=evictions,
    )


# ----------------------------------------------------------------------------- planner


def test_planner_reserves_a_window_page_per_resident_decode_row():
    head = _scheduled_chunk(17_490)  # tail [16384, 17490): 9 pages, + this request's first step
    c1, c2 = _limits(), replace(_limits(), resident_decode_rows=1)

    def plan(limits, rows, pages):
        return plan_tail_pair(head, limits=limits, prompt_len=17_490, prefill_budget=CHUNK,
                              decode_running=rows, pending_matches=True,
                              swa_free_tokens=pages * P)

    # One running request: any other decode row declines, exactly as before.
    assert plan(c1, True, 10**6) is None and plan(c1, 1, 10**6) is None
    assert plan(c1, False, 10) == plan(c1, 0, 10) is not None and plan(c1, 0, 9) is None
    # C2: the paused row may stay resident, with one more page for its first decode step,
    # which can be scheduled before the pair's drain returns the head's window.
    assert plan(c2, 0, 10) == plan(c2, 1, 11) == plan(c1, 0, 10)
    assert plan(c2, 1, 10) is None
    assert plan(c2, 2, 10**6) is None  # never more resident rows than C2 holds


def test_c2_limits_and_planner_stay_closed_outside_c2():
    assert replace(_limits(), resident_decode_rows=1) != _limits()
    assert "resident_decode_rows" not in repr(replace(_limits(), resident_decode_rows=1))
    for bad in (2, -1):
        with pytest.raises(ValueError, match="resident_decode_rows"):
            replace(_limits(), resident_decode_rows=bad)
    with pytest.raises(TypeError):
        replace(_limits(), resident_decode_rows=True)
    c2 = replace(_limits(), resident_decode_rows=1)
    head = _scheduled_chunk(17_490)
    kwargs = dict(limits=c2, prompt_len=17_490, prefill_budget=CHUNK, decode_running=0,
                  pending_matches=True, swa_free_tokens=None)
    assert plan_tail_pair(head, **kwargs) is not None
    other = _req(ChunkedReq, 300, 0, 256, uid=A, table_idx=1)
    for batch in (Batch(reqs=[head.reqs[0], other], phase="prefill"),  # never two prompts
                  Batch(reqs=[head.reqs[0], other], phase="decode")):  # never a decode batch
        assert plan_tail_pair(batch, **kwargs) is None


def test_a_two_row_decode_batch_never_reaches_a_cone_segment():
    """A C2 decode batch may carry a cone prompt's request (its state rides the Req into
    decode); the planner must decline it rather than raise its shared-prefill guard."""
    g = _geometry()
    cone_req = _req(Req, 26_000, 25_999, 26_000, uid=B)
    cone_req.dsv41_cone = dc.ConeRequestState(g, B, 26_000, 0, g.latest_decoder_start(26_000))
    other = _req(Req, 300, 299, 300, uid=A, table_idx=1)
    assert dc.plan_prefill_cone(Batch(reqs=[other, cone_req], phase="decode"), g) is None
    assert dc.plan_prefill_cone(Batch(reqs=[cone_req], phase="decode"), g) is None


# ----------------------------------------------------------------------------- flag selection


def _config(max_running_req):
    model_config = SimpleNamespace(dsv41_args=DeepseekV41Args(compress_ratios=REAL_RATIOS),
                                   prefill_batch_size_limit=1)
    return SimpleNamespace(model_config=model_config, max_running_req=max_running_req,
                           max_extend_tokens=CHUNK, moe_sparse_prefill_max_tokens=SPARSE,
                           page_size=P)


@pytest.mark.parametrize(
    ("gate", "max_running", "selected"),
    [(None, 1, "c1"), ("1", 1, "c1"), ("1", 2, "c2"), (None, 2, None), ("0", 2, None)],
    ids=["c1", "c1-gate", "c2", "two-no-gate", "two-gate-0"],
)
def test_flag_selection(monkeypatch, gate, max_running, selected):
    for env in (TAIL_PAIRING_ENV, dc.DEPENDENCY_CONE_ENV):
        monkeypatch.setenv(env, "1")
    monkeypatch.delenv(dc.DEPENDENCY_CONE_MIN_TOKENS_ENV, raising=False)
    if gate is None:
        monkeypatch.delenv(CONCURRENCY_ENV, raising=False)
    else:
        monkeypatch.setenv(CONCURRENCY_ENV, gate)
    warnings = []
    monkeypatch.setattr(scheduler_mod.logger, "warning_rank0", warnings.append)
    config = _config(max_running)
    limits = scheduler_mod._tail_pairing_limits(config)
    geometry = scheduler_mod._dependency_cone_geometry(config)
    if selected is None:  # an ungated two-request config is never served with either
        assert limits is None and geometry is None
        assert len(warnings) == 2 and all("bypassed" in w for w in warnings)
        return
    assert not warnings
    assert geometry == _geometry()  # the exact C1 cone geometry, also under C2
    assert limits == replace(_limits(), resident_decode_rows=max_running - 1)
    assert repr(limits) == repr(_limits())  # the qualified startup log line is unchanged
    if selected == "c1":
        assert limits == _limits()


def test_flags_off_select_nothing_under_c2(monkeypatch):
    for env in (TAIL_PAIRING_ENV, dc.DEPENDENCY_CONE_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(CONCURRENCY_ENV, "1")
    warnings = []
    monkeypatch.setattr(scheduler_mod.logger, "warning_rank0", warnings.append)
    assert scheduler_mod._tail_pairing_limits(_config(2)) is None
    assert scheduler_mod._dependency_cone_geometry(_config(2)) is None
    assert warnings == []


# ----------------------------------------------------------------------------- tail pairing


@pytest.mark.parametrize("prompt_len", [17_490, 9_298, 12_288, 126_976, 16_384, 8_448])
def test_pairing_beside_a_resident_decode_row_matches_unpaired_c2(prompt_len):
    off_sched, off = _two_lane_run(prompt_len=prompt_len,
                                   num_pages=1088 if prompt_len > 20_000 else 512)
    on_sched, on = _two_lane_run(pairing=True, prompt_len=prompt_len,
                                 num_pages=1088 if prompt_len > 20_000 else 512)
    for key in ("spans", "decodes", "replies", "admissions", "tree"):
        assert on[key] == off[key], key
    assert on["evictions"] == []  # the pair forced no window eviction
    tail = prompt_len % CHUNK
    pairs = [launch for launch in _prefills(on_sched) if launch.head is not None]
    if prompt_len > CHUNK and SPARSE < tail <= MAX_TAIL_TOKENS:
        # Paired while A stayed resident: A's last token before B's prompt came in a lone
        # decode batch, and A resumed beside B only after the paired forward.
        (pair,) = pairs
        assert pair.rows == ((B, prompt_len - tail, prompt_len),)
        assert on["decodes"][:3] == [(A,)] * 3 and on["decodes"][3] == (A, B)
        assert len(_prefills(on_sched)) == len(_prefills(off_sched)) - 1
    else:
        assert pairs == []
        assert _prefills(on_sched) == _prefills(off_sched)


def test_pairing_beside_a_resident_row_needs_one_more_free_window_page(monkeypatch):
    """At the C1 threshold (tail + one page) a pair beside a resident decode row declines: that
    row may take a page before the pair's drain. One more free page pairs; neither evicts."""
    seen = []
    real = scheduler_mod.plan_tail_pair

    def spy(batch, **kwargs):
        plan = real(batch, **kwargs)
        if kwargs["prompt_len"] == 17_490 and batch.reqs[0].cached_len == CHUNK:  # the head
            seen.append((kwargs["decode_running"], kwargs["swa_free_tokens"], plan is not None))
        return plan

    monkeypatch.setattr(scheduler_mod, "plan_tail_pair", spy)
    _two_lane_run(pairing=True, prompt_len=17_490)
    ((rows, free, paired),) = seen
    assert rows == 1 and paired and free % P == 0
    for free_pages, expect_pair in ((9 + 1 + 1, True), (9 + 1, False)):
        seen.clear()
        sched, pool = _c2(pairing=True)
        held = pool._win_alloc.alloc(free // P - free_pages)
        _submit(sched, A, 300, max_tokens=24)
        data = _run(sched, until=lambda s: _published(A)(s) >= 3)
        evictions = _evictions(sched)
        _submit(sched, B, 17_490, max_tokens=6)
        _run(sched, data)
        assert seen == [(1, free_pages * P, expect_pair)]
        assert evictions == []
        assert [launch.head is not None for launch in _prefills(sched)] == (
            [False, False, True] if expect_pair else [False, False, False, False])
        pool._win_alloc.free(held)
        _finish(sched, A, B)


def test_a_prompt_queued_behind_a_paired_prefill_is_admitted_after_it_and_pairs_beside_it():
    sched, _pool = _c2(pairing=True)
    _submit(sched, A, 17_490, max_tokens=12)
    data = _run(sched, until=_launched(lambda launch: launch.rows[0][0] == A))  # A's chunk 0
    _submit(sched, B, 12_000, max_tokens=6)  # queued: A's continuation stays first
    _run(sched, data)
    _finish(sched, A, B)
    order = [(e[1].phase, e[1].rows[0][0]) if e[0] == "launch"
             else ("sent", e[1].uid, type(e[1]).__name__) for e in sched.events]
    a_pair = order.index(("prefill", A), order.index(("prefill", A)) + 1)
    b_first = order.index(("prefill", B))
    assert a_pair < b_first  # B was admitted only after A's prompt finished prefilling
    # Early publish: A's first token left before B's first chunk launched.
    assert order.index(("sent", A, "DetokenizeMsg")) < b_first
    # A became resident and paused; B still paired beside it: head [0, 8192), tail to 12000.
    pairs = [launch for launch in _prefills(sched) if launch.head is not None]
    assert [(p.head, p.rows) for p in pairs] == [
        (((A, 8192, 16_384),), ((A, 16_384, 17_490),)),
        (((B, 0, 8192),), ((B, 8192, 12_000),)),
    ]
    assert [m.uid for m in sched.sent if isinstance(m, PromptAdmittedMsg)] == [A, B]


def test_a_third_prompt_waits_for_a_free_row_and_then_pairs_beside_the_resident_one():
    sched, _pool = _c2(pairing=True, num_pages=768)
    _submit(sched, A, 300, max_tokens=10)
    data = _run(sched, until=lambda s: _published(A)(s) >= 2)
    _submit(sched, B, 17_490, max_tokens=30)
    data = _run(sched, data, until=lambda s: _published(B)(s) >= 2)
    _submit(sched, C, 9_298, max_tokens=4)  # both rows are taken
    _run(sched, data)
    _finish(sched, A, B, C)
    order = [(e[1].phase, e[1].rows[0][0]) if e[0] == "launch"
             else ("finished", e[1].uid) if getattr(e[1], "finished", False) else None
             for e in sched.events]
    assert order.index(("finished", A)) < order.index(("prefill", C))  # a row had to free first
    pairs = [launch.rows[0][0] for launch in _prefills(sched) if launch.head is not None]
    assert pairs == [B, C]  # each beside the other's paused decode row


# ----------------------------------------------------------------------------- dependency cone


@pytest.mark.parametrize("prompt_len", [26_000, 17_490, 9_000])
def test_cone_beside_a_resident_decode_row_matches_plain_c2(prompt_len):
    g = _geometry()
    _plain_sched, plain = _two_lane_run(prompt_len=prompt_len)
    cone_sched, cone = _two_lane_run(cone=g, prompt_len=prompt_len)
    for key in ("spans", "decodes", "replies", "admissions", "tree"):
        assert cone[key] == plain[key], key
    latest = g.latest_decoder_start(prompt_len)
    segments = [launch.cone for launch in _prefills(cone_sched) if launch.rows[0][0] == B]
    spans = _spans(cone_sched, B)
    if spans[0][1] > latest:  # the first chunk crosses R*: the plain prefill
        assert segments == [None] * len(spans)
    else:
        r0 = next(s.start for s in segments if not s.is_encoder)
        for (start, end), seg in zip(spans, segments):
            assert (seg.uid, seg.start, seg.end) == (B, start, end)
            assert seg.is_encoder == (end <= latest)
            assert seg.is_encoder or seg.decoder_start == r0
    assert all(launch.cone is None for launch in _prefills(cone_sched) if launch.rows[0][0] == A)


def test_adaptive_policy_under_c2_runs_one_optimisation_per_prompt():
    """FREETOKEN_DSV41_DEPENDENCY_CONE_MIN_TOKENS=32768 with tail pairing: the short prompt pairs
    beside the resident request and the long one runs its cone beside the other resident one."""
    sched, _pool = _c2(pairing=True, cone=replace(_geometry(), min_uncached_tokens=32_768),
                       num_pages=1024)
    _submit(sched, A, 300, max_tokens=10)
    data = _run(sched, until=lambda s: _published(A)(s) >= 2)
    _submit(sched, B, 17_490, max_tokens=40)
    data = _run(sched, data, until=lambda s: _published(B)(s) >= 2)
    _submit(sched, C, 40_000, max_tokens=4)  # waits for A's row
    _run(sched, data)
    _finish(sched, A, B, C)
    by_uid = {uid: [launch for launch in _prefills(sched) if launch.rows[0][0] == uid]
              for uid in (A, B, C)}
    assert [launch.head is not None for launch in by_uid[B]] == [False, True]
    assert all(launch.cone is None for launch in by_uid[B])
    assert all(launch.head is None and launch.cone is not None for launch in by_uid[C])
    assert [launch.cone.is_encoder for launch in by_uid[C]] == [True] * 4 + [False]
    assert all(launch.cone is None and launch.head is None for launch in by_uid[A])


# ----------------------------------------------------------------------------- cancellation


def _abort_point(feature, point):
    """The launch after which the abort arrives: B's first chunk, its optimised forward (the
    pair, or the first decoder chunk), or B's first decode step."""
    if point == "first_chunk":
        return lambda launch: launch.phase == "prefill" and launch.rows[0][:2] == (B, 0)
    if point == "optimised":
        if feature == "pair":
            return lambda launch: launch.head is not None
        return lambda launch: launch.cone is not None and not launch.cone.is_encoder
    return lambda launch: launch.phase == "decode" and B in (row[0] for row in launch.rows)


@pytest.mark.parametrize("point", ["first_chunk", "optimised", "decode"])
@pytest.mark.parametrize("victim", [A, B], ids=["resident", "prefilling"])
@pytest.mark.parametrize("feature", ["pair", "cone"])
def test_abort_beside_a_resident_request_frees_once(feature, victim, point):
    g = _geometry()
    sched, pool = _c2(pairing=feature == "pair", cone=g if feature == "cone" else None)
    _submit(sched, A, 300, max_tokens=24)
    data = _run(sched, until=lambda s: _published(A)(s) >= 3)
    a_req = next(iter(sched.decode_manager.running_reqs))
    b_prompt = _submit(sched, B, 17_490 if feature == "pair" else 26_000, max_tokens=6)
    data = _run(sched, data, until=_launched(_abort_point(feature, point)))
    in_flight = data[0].batch.reqs
    replies_before = len(_replies(sched, victim))
    data = _abort(sched, data, victim)
    if victim == A and point != "decode":
        # Resident, not forwarded: freed at once, under nobody's in-flight forward.
        assert a_req not in in_flight and a_req.table_idx == -1
    _run(sched, data)
    sched._flush_abort_acks()
    survivor = B if victim == A else A
    assert _acks(sched) == [victim]  # exactly one terminal acknowledgement
    victim_replies = _replies(sched, victim)
    assert len(victim_replies) <= replies_before + 1  # at most the step already in flight
    assert not any(finished for *_, finished, _ in victim_replies)
    _finish(sched, survivor, aborted=(victim,))
    if feature == "cone" and victim == B:
        # Nothing inexact reached the tree: any reusable prefix lies a live window past the
        # frontier of the decoder start that actually ran.
        segments = [launch.cone for launch in _prefills(sched) if launch.rows[0][0] == B]
        r0 = next((s.start for s in segments if s is not None and not s.is_encoder), None)
        pc = sched.cache_manager.prefix_cache
        for m in range(P, len(b_prompt) + 1, P):
            got = pc.match_prefix(b_prompt[:m]).cached_len
            if got:
                assert r0 is not None and got - P >= g.tombstone_frontier(r0), (m, got)


# ----------------------------------------------------------------------------- decode isolation


def test_decode_batches_never_reach_tail_or_cone_planning(monkeypatch):
    planned = []
    for name in ("plan_tail_pair", "plan_prefill_cone"):
        real = getattr(scheduler_mod, name)

        def spy(batch, *args, real=real, name=name, **kwargs):
            planned.append((name, batch.phase, len(batch.reqs)))
            return real(batch, *args, **kwargs)

        monkeypatch.setattr(scheduler_mod, name, spy)
    sched, _pool = _c2(pairing=True, cone=_geometry())
    _submit(sched, A, 300, max_tokens=8)
    data = _run(sched, until=lambda s: _published(A)(s) >= 2)
    _submit(sched, B, 17_490, max_tokens=8)  # cone (min 0): pairing is not planned for it
    data = _run(sched, data, until=lambda s: _published(B)(s) >= 2)
    _run(sched, data)
    _submit(sched, C, 9_298, max_tokens=4)  # the cone declines; the tail pairs
    _run(sched)
    _finish(sched, A, B, C)
    assert planned and all(phase == "prefill" and n == 1 for _name, phase, n in planned)
    assert {name for name, *_ in planned} == {"plan_tail_pair", "plan_prefill_cone"}
    assert any(len(launch.rows) == 2 for launch in sched.engine.launches
               if launch.phase == "decode")
    assert [launch.rows[0][0] for launch in _prefills(sched) if launch.head] == [C]
    assert {launch.rows[0][0] for launch in _prefills(sched) if launch.cone} == {B}
