"""DeepSeek-V4.1 gated C2 resident-decode interleaving between prefill chunks (CPU, no GPU).

``FREETOKEN_DSV41_C2_PREFILL_INTERLEAVE=N`` lets a resident decode row take up to N ordinary decode
steps after each intermediate prefill chunk of the other request, instead of waiting for its whole
prompt. These tests drive the real ``overlap_loop`` (early publish included), batch preparation,
tail-pair / cone planning, abort handling and drains over a real DSV4.1 page pool and SWA radix
tree with two table rows. Each interleaved run is compared with the same run without the gate:
the same logical chunks and chunk inputs, the same replies per request, the same final cache.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import freetoken.scheduler.scheduler as scheduler_mod
from freetoken.message import DetokenizeMsg, PromptAdmittedMsg
from freetoken.scheduler import prefill_interleave as pi
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.utils import PendingReq
from freetoken.core import SamplingParams

from .test_dsv41_c2_adaptive_prefill import (
    A,
    B,
    C,
    CONCURRENCY_ENV,
    _abort,
    _acks,
    _c2,
    _config,
    _decodes,
    _evictions,
    _launched,
    _prefills,
    _published,
    _replies,
    _run,
    _spans,
    _submit,
)
from .test_dsv41_dependency_cone_lifecycle import _geometry
from .test_dsv41_tail_pairing import (  # noqa: F401 -- cpu_scheduler_helpers is a fixture
    CHUNK,
    IMAGE,
    P,
    _tree_state,
    cpu_scheduler_helpers,
)

pytestmark = pytest.mark.usefixtures("cpu_scheduler_helpers")


def _sched(steps, **kw):
    sched, pool = _c2(**kw)
    sched._prefill_interleave = steps
    sched._interleave_budget = 0
    return sched, pool


def _interleave_invariants(sched, steps, *, aborted=()):
    """Every forward: a prefill is one request; a decode never holds a prefilling request; at most
    ``steps`` decode forwards separate two chunks of one prompt; nothing runs between a pair's head
    and tail (they are one forward); a prompt's first chunk follows its admission directly."""
    mid, since_chunk = None, 0
    for launch in sched.engine.launches:
        if launch.phase == "decode":
            assert 1 <= len(launch.rows) <= 2
            assert launch.head is None and launch.cone is None and launch.role is None
            if mid is not None:
                assert mid not in {row[0] for row in launch.rows}
                since_chunk += 1
                assert mid in aborted or since_chunk <= steps, "a chunk boundary exceeded its budget"
            continue
        (uid, _start, end), = launch.rows
        assert launch.head is None or {row[0] for row in launch.head} == {uid}
        assert mid in (None, uid, *aborted), "two prompts interleaved their prefill chunks"
        mid = None if end == sched.prompt_lens[uid] else uid
        since_chunk = 0


def _finish(sched, steps, *expected, aborted=()):
    for uid in expected:
        replies = _replies(sched, uid)
        assert [f for *_, f, _r in replies] == [False] * (len(replies) - 1) + [True]
        assert replies[-1][3] == "length"
    _interleave_invariants(sched, steps, aborted=aborted)
    assert not sched.decode_manager.runnable and not sched.prefill_manager.runnable
    assert sched.table_manager.available_size == 2
    sched.cache_manager.check_integrity()


def _chunk_inputs(sched, uid):
    """The token ids every logical chunk of ``uid`` forwarded."""
    out = []
    for record in sched.engine.forwards:
        if record["phase"] != "prefill":
            continue
        for span, tokens in zip(record["spans"], record["tokens"]):
            out.append((span, tuple(tokens)))
    return [item for item in out if item in {(s, t) for s, t in out}]


def _two_lane(steps, *, pairing=False, cone=None, prompt_len, a_tokens=40, b_tokens=6, **pools):
    """A decodes; B arrives after A's third token and prefills beside the resident A."""
    sched, pool = _sched(steps, pairing=pairing, cone=cone, **pools)
    a = _submit(sched, A, 300, max_tokens=a_tokens)
    data = _run(sched, until=lambda s: _published(A)(s) >= 3)
    evictions = _evictions(sched)
    b = _submit(sched, B, prompt_len, max_tokens=b_tokens)
    _run(sched, data)
    _finish(sched, steps, A, B)
    return sched, dict(
        spans=(_spans(sched, A), _spans(sched, B)),
        b_inputs=[(span, tokens) for record in sched.engine.forwards if record["phase"] == "prefill"
                  for span, tokens in zip(record["spans"], record["tokens"])],
        replies={uid: _replies(sched, uid) for uid in (A, B)},
        admissions=[(m.uid, m.prompt_tokens, m.cached_tokens)
                    for m in sched.sent if isinstance(m, PromptAdmittedMsg)],
        tree=(_tree_state(sched.cache_manager, pool, a), _tree_state(sched.cache_manager, pool, b)),
        evictions=evictions,
        decodes=_decodes(sched),
    )


def _order(sched):
    """('P', uid, start, end) for prefill forwards (a pair is its head then tail), ('D', uids)."""
    out = []
    for launch in sched.engine.launches:
        if launch.phase == "decode":
            out.append(("D", tuple(row[0] for row in launch.rows)))
        else:
            for row in (launch.head or ()) + launch.rows:
                out.append(("P", *row))
    return out


CASES = [
    dict(prompt_len=17_490),  # three plain chunks
    dict(prompt_len=17_490, pairing=True),  # chunk 0, then the paired head+tail forward
    dict(prompt_len=26_000, cone="cone"),  # encoder chunks, then decoder chunks
    dict(prompt_len=40_000, cone="adaptive", pairing=True, num_pages=1024),  # production policy
    dict(prompt_len=9_000),  # one boundary
    dict(prompt_len=8_192),  # a single chunk: no boundary at all
]


def _case(case):
    case = dict(case)
    cone = case.pop("cone", None)
    if cone == "cone":
        case["cone"] = _geometry()
    elif cone == "adaptive":
        case["cone"] = replace(_geometry(), min_uncached_tokens=32_768)
    if case["prompt_len"] > 20_000:
        case.setdefault("num_pages", 1088)
    return case


# ----------------------------------------------------------------------------- scheduling order


@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()))
def test_interleaving_preserves_every_request_and_the_cache(case, steps):
    _off_sched, off = _two_lane(0, **_case(case))
    on_sched, on = _two_lane(steps, **_case(case))
    # The prompt's chunks, their inputs, every reply, admission accounting and the final cache
    # are exactly the uninterleaved C2 run's; only the order of whole forwards moved.
    for key in ("spans", "b_inputs", "replies", "admissions", "tree"):
        assert on[key] == off[key], key
    assert on["evictions"] == off["evictions"] == []
    order = _order(on_sched)
    b_chunks = [i for i, item in enumerate(order) if item[0] == "P" and item[1] == B]
    # The newcomer's first chunk is not delayed: it is the first forward scheduled after B
    # arrived that is not A's in-flight decode step.
    first = b_chunks[0]
    assert all(item == ("D", (A,)) for item in order[first - 1:first])
    # Between consecutive chunks of B's prompt: exactly min(steps, A's remaining) lone-A decodes,
    # except inside a paired forward (head and tail share one forward).
    paired = {(launch.head[0][1], launch.rows[0][1]) for launch in _prefills(on_sched) if launch.head}
    for i, j in zip(b_chunks, b_chunks[1:]):
        between = order[i + 1:j]
        if (order[i][2], order[j][2]) in paired:
            assert between == []
        else:
            assert between == [("D", (A,))] * len(between) and 1 <= len(between) <= steps
    # Nothing waits after the final chunk: B joins A's decode at once.
    assert order[b_chunks[-1] + 1] == ("D", (A, B))
    # A's extra lone steps were taken from its later shared steps, not added.
    assert len(on["decodes"]) >= len(off["decodes"])
    assert sum(len(d) for d in on["decodes"]) == sum(len(d) for d in off["decodes"])
    if len(b_chunks) == 1:
        assert on["decodes"] == off["decodes"]


def test_the_resident_waits_at_most_one_prefill_forward():
    sched, _ = _two_lane(1, prompt_len=40_000, cone=_geometry(), num_pages=1088, a_tokens=60)
    since_a = 0
    worst = 0
    for launch in sched.engine.launches:
        if launch.phase == "decode" and A in {row[0] for row in launch.rows}:
            worst, since_a = max(worst, since_a), 0
        elif launch.phase == "prefill" and launch.rows[0][0] == B:
            since_a += 1
    assert worst == 1  # versus the whole 40,000-token prompt (5 forwards) without the gate
    off, _ = _two_lane(0, prompt_len=40_000, cone=_geometry(), num_pages=1088, a_tokens=60)
    longest = 0
    since_a = 0
    for launch in off.engine.launches:
        if launch.phase == "decode" and A in {row[0] for row in launch.rows}:
            longest, since_a = max(longest, since_a), 0
        elif launch.phase == "prefill" and launch.rows[0][0] == B:
            since_a += 1
    assert longest == 5


def test_a_lone_prompt_never_interleaves_and_the_gate_off_is_unchanged():
    runs = {}
    for steps in (None, 0, 2):
        sched, _ = _c2()
        if steps is not None:
            sched._prefill_interleave, sched._interleave_budget = steps, 0
        _submit(sched, B, 26_000, max_tokens=4)
        _run(sched)
        runs[steps] = (_order(sched), _replies(sched))
    assert runs[None] == runs[0] == runs[2]  # nothing resident: identical forwards


def test_the_resident_finishing_mid_prompt_ends_the_interleave():
    sched, _ = _sched(2)
    _submit(sched, A, 300, max_tokens=6)
    data = _run(sched, until=lambda s: _published(A)(s) >= 3)
    _submit(sched, B, 40_000, max_tokens=4)
    _run(sched, data)
    _finish(sched, 2, A, B)
    order = _order(sched)
    last_a = max(i for i, item in enumerate(order) if item[0] == "D" and A in item[1])
    assert all(item[0] == "P" for item in order[last_a + 1:-3])  # then B's chunks back to back


# ----------------------------------------------------------------------------- queued third client


def test_two_active_plus_a_queued_third_client():
    sched, _ = _sched(1, pairing=True, num_pages=1024)
    _submit(sched, A, 300, max_tokens=30)
    data = _run(sched, until=lambda s: _published(A)(s) >= 2)
    _submit(sched, B, 26_000, max_tokens=8)
    data = _run(sched, data, until=_launched(lambda l: l.phase == "prefill" and l.rows[0][0] == B))
    _submit(sched, C, 9_298, max_tokens=4)  # both rows taken: queued behind B's continuation
    _run(sched, data)
    _finish(sched, 1, A, B, C)
    order = _order(sched)
    b_last = max(i for i, item in enumerate(order) if item[0] == "P" and item[1] == B)
    c_first = min(i for i, item in enumerate(order) if item[0] == "P" and item[1] == C)
    done = [max(i for i, item in enumerate(order) if item[0] == "D" and uid in item[1])
            for uid in (A, B)]
    assert b_last < c_first and min(done) < c_first  # admission order; C needed a freed row
    # A was interleaved while C waited in the queue.
    assert any(item == ("D", (A,)) for item in order[:b_last])
    assert [m.uid for m in sched.sent if isinstance(m, PromptAdmittedMsg)] == [A, B, C]


# ----------------------------------------------------------------------------- tail pairing / cone


def test_pairing_beside_an_interleaved_resident_keeps_its_window_reservation(monkeypatch):
    seen = []
    real = scheduler_mod.plan_tail_pair

    def spy(batch, **kwargs):
        plan = real(batch, **kwargs)
        seen.append((kwargs["decode_running"], plan is not None))
        return plan

    monkeypatch.setattr(scheduler_mod, "plan_tail_pair", spy)
    sched, on = _two_lane(1, pairing=True, prompt_len=17_490)
    assert (1, True) in seen and on["evictions"] == []
    pairs = [l for l in _prefills(sched) if l.head is not None]
    assert [(p.head, p.rows) for p in pairs] == [(((B, 8192, 16_384),), ((B, 16_384, 17_490),))]


def test_cone_segments_are_unchanged_by_interleaving():
    g = _geometry()
    off, _ = _two_lane(0, cone=g, prompt_len=40_000, num_pages=1088)
    on, _ = _two_lane(2, cone=g, prompt_len=40_000, num_pages=1088)
    seg = lambda s: [l.cone for l in _prefills(s) if l.rows[0][0] == B]
    assert seg(on) == seg(off) and any(s.is_encoder for s in seg(on))
    assert all(l.cone is None for l in _prefills(on) if l.rows[0][0] == A)


# ----------------------------------------------------------------------------- multimodal


def test_multimodal_chunks_gather_the_same_vision_rows(monkeypatch):
    def run(steps):
        gathered = []
        sched, _ = _sched(steps)
        real = sched._gather_multimodal

        def spy(batch):
            real(batch)
            mm = getattr(batch, "mm_embeds", None)
            gathered.append((batch.reqs[0].uid, batch.reqs[0].cached_len,
                             None if mm is None else mm.clone()))

        sched._gather_multimodal = spy
        _submit(sched, A, 300, max_tokens=30)
        data = _run(sched, until=lambda s: _published(A)(s) >= 3)
        ids = (torch.arange(17_490, dtype=torch.int32) % 90 + 1)
        image_at = [8_100 + i for i in range(200)]  # an image straddling the first chunk boundary
        ids[torch.tensor(image_at)] = IMAGE
        embeds = torch.arange(200 * 4, dtype=torch.float32).view(200, 4)
        sched.prompt_lens[B] = 17_490
        sched.prefill_manager.pending_list.append(PendingReq(
            uid=B, input_ids=ids, sampling_params=SamplingParams(max_tokens=4),
            mm_embeds=embeds, is_multimodal=True))
        _run(sched, data)
        _finish(sched, steps, A, B)
        return sched, [item for item in gathered if item[0] == B]

    off_sched, off = run(0)
    on_sched, on = run(1)
    assert len(on) == len(off) == 3
    for (u1, c1, m1), (u2, c2, m2) in zip(on, off):
        assert (u1, c1) == (u2, c2) and (m1 is None) == (m2 is None)
        assert m1 is None or torch.equal(m1, m2)
    assert any(item == ("D", (A,)) for item in _order(on_sched)[:-4])


# ----------------------------------------------------------------------------- cancellation


def _point(name, b_len):
    """The launch (after B's first chunk) at which the abort arrives."""
    if name == "interleaved_decode":  # A's lone step in flight; B's continuation between chunks
        return lambda l: l.phase == "decode" and [row[0] for row in l.rows] == [A]
    if name == "later_chunk":  # a B chunk that followed an interleaved step
        return lambda l: l.phase == "prefill" and l.rows[0][0] == B and l.rows[0][1] > 0
    if name == "final_forward":  # B's last chunk, or its paired head+tail forward
        return lambda l: l.phase == "prefill" and l.rows[0][0] == B and l.rows[0][2] == b_len
    return lambda l: l.phase == "decode" and len(l.rows) == 2  # the first joint decode


POINTS = ["first_chunk", "interleaved_decode", "later_chunk", "final_forward", "joint_decode"]


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("victim", [A, B], ids=["resident", "prefilling"])
@pytest.mark.parametrize("feature", ["plain", "pair", "cone"])
def test_abort_at_every_boundary_frees_once(feature, victim, point):
    g = _geometry()
    b_len = {"plain": 17_490, "pair": 17_490, "cone": 26_000}[feature]
    sched, _pool = _sched(1, pairing=feature == "pair", cone=g if feature == "cone" else None,
                          num_pages=768)
    launches = sched.engine.launches
    _submit(sched, A, 300, max_tokens=40)
    data = _run(sched, until=lambda s: _published(A)(s) >= 3)
    _submit(sched, B, b_len, max_tokens=6)
    data = _run(sched, data, until=_launched(lambda l: l.phase == "prefill" and l.rows[0][0] == B))
    if point != "first_chunk":
        since, match = len(launches), _point(point, b_len)
        data = _run(sched, data, until=lambda s: any(match(l) for l in launches[since:]))
    batch = data[0].batch
    pending = sched.prefill_manager.pending_list
    b_open = pending[0].chunked_req if pending and pending[0].uid == B else None
    if point == "interleaved_decode":  # the state the gate creates: B open between its chunks
        assert batch.is_decode and [r.uid for r in batch.reqs] == [A] and b_open is not None
    if point == "later_chunk":
        assert ("D", (A,)) in _order(sched)[_order(sched).index(("P", B, 0, 8192)):]
    data = _abort(sched, data, victim)
    if victim == B and b_open is not None and b_open not in batch.reqs:
        assert b_open.table_idx == -1  # a continuation between its chunks is freed at once
    _run(sched, data)
    sched._flush_abort_acks()
    survivor = B if victim == A else A
    assert _acks(sched) == [victim]
    assert not any(f for *_, f, _ in _replies(sched, victim))
    _finish(sched, 1, survivor, aborted=(victim,))


# ----------------------------------------------------------------------------- agreement


class _Ranks:
    """Two replicated schedulers' collectives on one host: rank 0's broadcast wins, the mismatch
    flags sum, exactly as the Gloo CPU group would combine them."""

    def __init__(self):
        self.calls = []

    def run(self, decisions):
        wires = [d.encode() for d in decisions]
        flags = [int(not torch.equal(wires[0], w)) for w in wires]
        results = []
        for rank, decision in enumerate(decisions):
            try:
                results.append(pi.agree_interleave(
                    decision, rank=rank, world_size=len(decisions),
                    broadcast=lambda t: wires[0].clone(),
                    all_reduce_sum=lambda t: torch.tensor([sum(flags)], dtype=torch.int64)))
            except pi.PrefillInterleaveMismatch as exc:
                results.append(exc)
        return results


def _decision(**kw):
    values = dict(decode=True, uid=B, resume_at=8192, budget=1, decode_rows=1)
    values.update(kw)
    return pi.InterleaveDecision(**values)


def test_agreement_returns_rank_zeros_decision_or_fails_closed_everywhere():
    ranks = _Ranks()
    assert ranks.run([_decision()] * 4) == [_decision()] * 4
    results = ranks.run([_decision(), _decision(), _decision(decode=False, budget=0), _decision()])
    assert all(isinstance(r, pi.PrefillInterleaveMismatch) for r in results)
    for bad in (torch.zeros(7, dtype=torch.int64), torch.zeros(6, dtype=torch.int64),
                torch.zeros(7, dtype=torch.int32)):
        with pytest.raises(pi.PrefillInterleaveMismatch):
            pi.InterleaveDecision.decode_wire(bad)
    assert pi.InterleaveDecision.decode_wire(_decision().encode()) == _decision()
    for bad in (dict(decode=True, budget=0), dict(budget=9), dict(decode_rows=0), dict(uid=-1),
                dict(decode=1)):
        with pytest.raises((TypeError, ValueError)):
            _decision(**bad)


def test_every_boundary_decision_is_agreed_across_ranks(monkeypatch):
    sched, _ = _sched(2)
    sched.config.tp_info = type(sched.config.tp_info)(rank=0, size=4, is_primary=lambda: True)
    labels = []
    sched._cpu_broadcast = lambda t, label: labels.append(label) or t
    sched._cpu_all_reduce_sum = lambda t, label: labels.append(label) or t
    _submit(sched, A, 300, max_tokens=40)
    data = _run(sched, until=lambda s: _published(A)(s) >= 3)
    _submit(sched, B, 26_000, max_tokens=4)  # 4 chunks: 3 boundaries x (2 decodes + 1 resume)
    _run(sched, data)
    _finish(sched, 2, A, B)
    assert labels == ["prefill_interleave", "prefill_interleave_mismatch"] * 9


def test_plan_asks_only_at_a_continuation_with_a_resident_row():
    head = PendingReq(uid=B, input_ids=torch.ones(10, dtype=torch.int32),
                      sampling_params=SamplingParams(max_tokens=1))
    assert pi.plan_interleave(budget=1, pending_head=None, decode_rows=1) is None
    assert pi.plan_interleave(budget=1, pending_head=head, decode_rows=1) is None  # first chunk
    head.chunked_req = ChunkedReq.__new__(ChunkedReq)
    head.chunked_req.uid, head.chunked_req.cached_len = B, 8192
    assert pi.plan_interleave(budget=1, pending_head=head, decode_rows=0) is None
    assert pi.plan_interleave(budget=1, pending_head=head, decode_rows=1) == _decision()
    assert pi.plan_interleave(budget=0, pending_head=head, decode_rows=1) == _decision(
        decode=False, budget=0)


# ----------------------------------------------------------------------------- gate


@pytest.mark.parametrize(("value", "steps"), [(None, 0), ("0", 0), ("1", 1), ("8", 8)])
def test_gate_parses_strictly(monkeypatch, value, steps):
    if value is None:
        monkeypatch.delenv(pi.PREFILL_INTERLEAVE_ENV, raising=False)
    else:
        monkeypatch.setenv(pi.PREFILL_INTERLEAVE_ENV, value)
    assert pi.prefill_interleave_steps() == steps


@pytest.mark.parametrize("value", ["", "9", "-1", "01", " 1", "1.0", "true", "on"])
def test_gate_refuses_malformed_values(monkeypatch, value):
    monkeypatch.setenv(pi.PREFILL_INTERLEAVE_ENV, value)
    with pytest.raises(ValueError, match=pi.PREFILL_INTERLEAVE_ENV):
        pi.prefill_interleave_steps()


@pytest.mark.parametrize(("gate", "max_running", "ok"),
                         [("1", 2, True), ("1", 1, False), (None, 2, False), ("0", 2, False)])
def test_scheduler_fails_closed_outside_gated_c2(monkeypatch, gate, max_running, ok):
    monkeypatch.setenv(pi.PREFILL_INTERLEAVE_ENV, "1")
    if gate is None:
        monkeypatch.delenv(CONCURRENCY_ENV, raising=False)
    else:
        monkeypatch.setenv(CONCURRENCY_ENV, gate)
    if ok:
        assert scheduler_mod._prefill_interleave_steps(_config(max_running)) == 1
    else:
        with pytest.raises(ValueError, match="gated C2"):
            scheduler_mod._prefill_interleave_steps(_config(max_running))
    monkeypatch.delenv(pi.PREFILL_INTERLEAVE_ENV)
    assert scheduler_mod._prefill_interleave_steps(_config(max_running)) == 0
