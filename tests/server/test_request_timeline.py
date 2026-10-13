"""Opt-in request boundary timeline: measurement only, content-free, default off, C2-safe."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import BatchBackendMsg, TokenizeMsg, UserMsg, UserReply
from freetoken.utils import request_timeline as rt
from freetoken.utils.request_timeline import RequestTimeline, parse_timeline_line


class Clock:
    def __init__(self):
        self.t = 1_000

    def __call__(self):
        self.t += 10
        return self.t


def _timeline(proc="unit", enabled=True, max_open=rt.MAX_OPEN):
    lines: list[str] = []
    return RequestTimeline(proc, lines.append, enabled=enabled, clock=Clock(), max_open=max_open), lines


# ------------------------------------------------------------------ core

@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True)])
def test_strict_parse(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(rt.TIMELINE_ENV, raising=False)
    else:
        monkeypatch.setenv(rt.TIMELINE_ENV, value)
    assert rt.timeline_enabled() is expected
    assert RequestTimeline("p", print).enabled is expected


@pytest.mark.parametrize("value", ["", "true", "on", "2", "1 "])
def test_invalid_values_are_refused(monkeypatch, value):
    monkeypatch.setenv(rt.TIMELINE_ENV, value)
    with pytest.raises(ValueError):
        rt.timeline_enabled()
    with pytest.raises(ValueError):
        RequestTimeline("p", print)


def test_disabled_does_nothing():
    tl, lines = _timeline(enabled=False)
    tl.begin(1, "a")
    tl.stamp(1, "b")
    tl.count(1, "n", 3)
    tl.add(1, "k")
    tl.append(1, "l", [1])
    assert tl.finish(1) is None and lines == [] and not tl.active(1)


def test_one_line_per_request_with_only_stages_counts_and_stamps():
    tl, lines = _timeline("frontend")
    tl.begin(7, "uid_assigned")
    tl.stamp(7, "x")
    tl.stamp(7, "x")  # once: the first stamp wins
    tl.stamp(7, "y", t=5)
    tl.count(7, "prompt_tokens", 17490)
    tl.add(7, "completion_tokens", 2)
    tl.add(7, "completion_tokens", 1)
    tl.append(7, "chunks", [1, 2, 8192])
    record = tl.finish(7)
    assert len(lines) == 1 and lines[0].startswith(rt.TIMELINE_PREFIX)
    parsed = parse_timeline_line("prefix junk " + lines[0])
    assert parsed == record
    assert parsed["proc"] == "frontend" and parsed["uid"] == 7
    assert set(parsed) == {"proc", "uid", "t", "n"}
    assert parsed["t"]["y"] == 5 and parsed["t"]["x"] < parsed["t"]["y"] + 10_000
    assert parsed["n"] == {"prompt_tokens": 17490, "completion_tokens": 3, "chunks": [[1, 2, 8192]]}
    assert tl.finish(7) is None and len(lines) == 1  # emitted once


def test_unknown_or_finished_uids_cannot_be_recreated_by_late_stamps():
    tl, lines = _timeline()
    tl.stamp(3, "late")
    tl.count(3, "n", 1)
    assert tl.finish(3) is None
    tl.begin(4)
    tl.finish(4)
    tl.stamp(4, "after_finish")
    assert tl.finish(4) is None and len(lines) == 1


def test_bounded_and_counts_drops():
    tl, lines = _timeline(max_open=2)
    for uid in range(3):
        tl.begin(uid)
    assert not tl.active(0) and tl.active(1) and tl.active(2)
    record = tl.finish(2)
    assert record["dropped"] == 1


def test_c2_interleaved_requests_stay_separate():
    tl, lines = _timeline()
    tl.begin(1, "rx")
    tl.begin(2, "rx")
    tl.append(1, "chunks", [0])
    tl.add(2, "interleaved_decodes")
    tl.append(1, "chunks", [1])
    tl.finish(2)
    tl.finish(1)
    a, b = (parse_timeline_line(line) for line in lines)
    assert a["uid"] == 2 and a["n"] == {"interleaved_decodes": 1}
    assert b["uid"] == 1 and b["n"] == {"chunks": [[0], [1]]}


# ------------------------------------------------------------------ frontend

def _manager(enabled=True):
    from freetoken.server.api_server import FrontendManager

    lines: list[str] = []
    timeline = RequestTimeline("frontend", lines.append, enabled=enabled)
    manager = FrontendManager(config=SimpleNamespace(served_model_name="m"), send_tokenizer=None,
                              recv_tokenizer=None, maintenance_state="serving", timeline=timeline)
    return manager, lines


def test_frontend_default_constructs_a_disabled_timeline(monkeypatch):
    from freetoken.server.api_server import FrontendManager

    monkeypatch.delenv(rt.TIMELINE_ENV, raising=False)
    manager = FrontendManager(config=SimpleNamespace(served_model_name="m"), send_tokenizer=None,
                              recv_tokenizer=None)
    assert isinstance(manager.timeline, RequestTimeline) and not manager.timeline.enabled


def test_frontend_replies_and_ack_cleanup_emit_one_content_free_line():
    manager, lines = _manager()

    async def flow():
        uid = manager.new_user()
        replies = [
            UserReply(uid=uid, incremental_output="", finished=False, prompt_tokens_delta=17490,
                      cached_tokens=0),
            UserReply(uid=uid, incremental_output="SECRET TEXT", finished=False,
                      completion_tokens_delta=1),
            UserReply(uid=uid, incremental_output="", finished=True, completion_tokens_delta=2),
        ]
        received = []
        gen = manager.wait_for_ack(uid)
        for reply in replies:
            manager._timeline_reply(reply)
            manager.ack_map[uid].append(reply)
            manager.event_map[uid].set()
        async for ack in gen:
            received.append(ack)
            if ack.finished:
                break
        await gen.aclose()
        return uid, received

    uid, received = asyncio.run(flow())
    assert len(received) == 3 and len(lines) == 1
    assert "SECRET" not in lines[0]
    record = parse_timeline_line(lines[0])
    assert record["uid"] == uid
    assert {"uid_assigned", "first_reply_rx", "admitted_rx", "first_text_rx", "finished_rx"} <= set(record["t"])
    assert record["n"] == {"prompt_tokens": 17490, "cached_tokens": 0, "completion_tokens": 3}


class _CompletionState:
    def __init__(self, timeline):
        self.timeline = timeline
        self.config = SimpleNamespace(model_path="/models/m", served_model_name="m",
                                      max_output_tokens=None, max_seq_len=1 << 20,
                                      enable_cache_report=False)
        self.sent = None

    def new_user(self):
        self.timeline.begin(5, "uid_assigned")
        return 5

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid):
        try:
            yield UserReply(uid=uid, incremental_output="", finished=False, prompt_tokens_delta=3)
            yield UserReply(uid=uid, incremental_output=" ORBIT", finished=False, completion_tokens_delta=1)
            yield UserReply(uid=uid, incremental_output="", finished=True, finish_reason="stop")
        finally:
            self.timeline.finish(uid)


def test_streaming_completion_stamps_the_path_to_the_first_text_chunk():
    from freetoken.server.openai_api import CompletionRequest, handle_completion

    lines: list[str] = []
    state = _CompletionState(RequestTimeline("frontend", lines.append, enabled=True))

    async def flow():
        req = CompletionRequest(model="m", prompt="hello", max_tokens=4, stream=True)
        response = await handle_completion(req, None, state, {})
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(flow())
    assert any(b"ORBIT" in c for c in chunks)
    record = parse_timeline_line(lines[0])
    assert {"uid_assigned", "handler_entry", "tokenize_sent", "sse_first_text_yield"} <= set(record["t"])
    assert record["t"]["handler_entry"] <= record["t"]["tokenize_sent"] <= record["t"]["sse_first_text_yield"]
    assert "http_arrival" not in record["t"]  # no request object was supplied


def test_timeline_entry_uses_the_middleware_arrival():
    from freetoken.server.openai_api import _timeline_entry

    lines: list[str] = []
    tl = RequestTimeline("frontend", lines.append, enabled=True)
    tl.begin(9)
    state = SimpleNamespace(timeline=tl)
    request = SimpleNamespace(state=SimpleNamespace(ft_arrival_ns=111))
    _timeline_entry(state, 9, request, 222, prerender_start=333, prerender_end=0)
    record = tl.finish(9)
    assert record["t"] == {"http_arrival": 111, "handler_entry": 222, "prerender_start": 333}


def test_fake_states_without_a_timeline_take_the_default_path():
    from freetoken.server.openai_api import _timeline, _timeline_entry

    state = SimpleNamespace()
    assert _timeline(state) is None
    _timeline_entry(state, 1, None, 5)  # no attribute: nothing happens


# ------------------------------------------------------------------ tokenizer worker

def test_tokenize_requests_stamps_render_and_encode():
    from freetoken.tokenizer.server import _tokenize_requests

    tl, lines = _timeline("tokenizer_0")
    msgs = [TokenizeMsg(uid=u, text="x", sampling_params=SamplingParams()) for u in (1, 2)]
    for m in msgs:
        tl.begin(m.uid, "tok_rx")

    class Manager:
        def tokenize(self, batch):
            if batch[0].uid == 2:
                raise ValueError("bad template")
            return [torch.arange(11, dtype=torch.int32)]

    ok, inputs, errors = _tokenize_requests(Manager(), msgs, SimpleNamespace(warning=lambda *_: None), tl)
    assert [m.uid for m in ok] == [1] and [e.uid for e in errors] == [2]
    one, two = tl.finish(1), tl.finish(2)
    assert {"tok_rx", "tok_start", "tok_done"} <= set(one["t"]) and one["n"] == {"prompt_tokens": 11}
    assert "tok_done" not in two["t"]


def test_tokenize_requests_without_a_timeline_is_unchanged():
    from freetoken.tokenizer.server import _tokenize_requests

    class Manager:
        def tokenize(self, batch):
            return [torch.arange(3, dtype=torch.int32)]

    msgs = [TokenizeMsg(uid=1, text="x", sampling_params=SamplingParams())]
    ok, inputs, errors = _tokenize_requests(Manager(), msgs, SimpleNamespace(warning=print))
    assert len(ok) == 1 and not errors


# ------------------------------------------------------------------ scheduler (rank 0)

def _scheduler(enabled=True):
    from freetoken.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    lines: list[str] = []
    sched._request_timeline = RequestTimeline("scheduler", lines.append, enabled=enabled)
    return sched, lines


def test_timeline_now_is_zero_without_or_with_a_disabled_timeline():
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.utils.request_timeline import timeline_now

    bare = Scheduler.__new__(Scheduler)
    assert timeline_now(bare) == 0
    assert timeline_now(SimpleNamespace()) == 0  # unbound-method stubs read as disabled
    sched, _ = _scheduler(enabled=False)
    assert timeline_now(sched) == 0
    sched, _ = _scheduler(enabled=True)
    assert timeline_now(sched) > 0


def _user(uid, n):
    return UserMsg(uid=uid, input_ids=torch.arange(n, dtype=torch.int32), sampling_params=SamplingParams())


def test_receive_path_begins_requests_with_idle_work_context():
    sched, lines = _scheduler()
    sched._timeline_idle_work = (10, 20)
    queue = [_user(1, 17490), BatchBackendMsg(data=[_user(2, 31)])]

    class Queue:
        def empty(self):
            return not queue

        def get(self):
            return queue.pop(0)

    sched._recv_from_tokenizer = Queue()
    sched.run_when_idle = lambda: None
    msgs = sched._recv_msg_single_rank(blocking=False)
    assert len(msgs) == 2
    one = sched._request_timeline.finish(1)
    two = sched._request_timeline.finish(2)
    assert one["n"] == {"prompt_tokens": 17490} and two["n"] == {"prompt_tokens": 31}
    assert one["t"]["idle_work_start"] == 10 and one["t"]["idle_work_end"] == 20
    assert "sched_rx" in one["t"] and "sched_rx" in two["t"]


def test_receive_path_is_inert_when_disabled():
    sched, lines = _scheduler(enabled=False)
    queue = [_user(1, 5)]
    sched._recv_from_tokenizer = SimpleNamespace(empty=lambda: not queue, get=lambda: queue.pop(0))
    sched.run_when_idle = lambda: None
    assert len(sched._recv_msg_single_rank(blocking=False)) == 1
    assert not sched._request_timeline.active(1) and lines == []


class _Event:
    def __init__(self):
        self.synced = False

    def synchronize(self):
        self.synced = True

    def elapsed_time(self, end):
        return 1234.5678


def test_prefill_chunks_and_final_drain_publish_one_scheduler_line(monkeypatch):
    from freetoken.scheduler import scheduler as module

    class Chunk:  # stands in for ChunkedReq
        def __init__(self, uid):
            self.uid = uid

    class Final:
        def __init__(self, uid):
            self.uid = uid

    monkeypatch.setattr(module, "ChunkedReq", Chunk)
    sched, lines = _scheduler()
    sched.prefill_manager = SimpleNamespace(pending_list=[SimpleNamespace(uid=8)])
    tl = sched._request_timeline
    tl.begin(8, "sched_rx")
    # head chunk scheduled, forwarded and drained (no reply)
    head = SimpleNamespace(is_prefill=True, reqs=[Chunk(8)], log_new_tokens=8192, dsv41_cone=None)
    sched._timeline_scheduled(head, 100)
    head._timeline_forward = (200, 300, _Event(), _Event())
    sched._timeline_drained(head, 310, 320)
    assert lines == []
    # a resident decode step interleaved before the paired tail
    decode = SimpleNamespace(is_prefill=False, reqs=[Final(3), Final(4)])
    sched._timeline_scheduled(decode, 400)
    tail = SimpleNamespace(is_prefill=True, reqs=[Final(8)], log_new_tokens=9298,
                           paired_head=object(), dsv41_cone=None)
    sched._timeline_scheduled(tail, 500)
    end = _Event()
    tail._timeline_forward = (600, 700, _Event(), end)
    sched._timeline_drained(tail, 710, 720)
    assert end.synced and len(lines) == 1
    record = parse_timeline_line(lines[0])
    assert record["uid"] == 8 and "first_token_tx" in record["t"]
    assert [c[2:] for c in record["n"]["chunks"]] == [[8192, 0, 0], [9298, 1, 0]]
    assert record["n"]["interleaved_decodes"] == 1
    assert [f[:4] for f in record["n"]["forwards"]] == [[200, 300, 310, 320], [600, 700, 710, 720]]
    assert record["n"]["forwards"][0][5] == 1234.568
