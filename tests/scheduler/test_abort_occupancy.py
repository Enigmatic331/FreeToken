"""Terminal abort occupancy survives scheduler -> tokenizer -> frontend.

CPU-only tests use the real hybrid resource-drain path where possible. Empty
heartbeats and early rejections must not clear another request's live usage.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from freetoken.core import Batch
from freetoken.message import AbortBackendMsg, ErrorReplyMsg, UserReply
from freetoken.scheduler.scheduler import Scheduler
from freetoken.server.stats import StatsTracker
from freetoken.tokenizer.server import _error_reply
from .test_abort_inflight_prefill import UID, _as_last_data, _launch_req, _setup


def test_aborted_inflight_request_reports_released_real_pool():
    pool, cm, tm, dm, _, sent, scheduler = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32))
    dm.filter_reqs([req])
    scheduler._last_data = _as_last_data(Batch(reqs=[req], phase="prefill"))
    tracker = StatsTracker()
    tracker.on_new_user(UID)
    before_used, before_total = cm.page_usage()
    assert before_used > 0
    tracker.observe(UserReply(uid=UID, incremental_output="x", finished=False,
                              kv_used_pages=before_used, kv_total_pages=before_total))
    tracker.on_abort(UID)
    Scheduler._process_one_msg(scheduler, AbortBackendMsg(uid=UID))
    assert req.table_idx != -1  # no release under a live forward
    Scheduler._process_last_data(scheduler, scheduler._last_data)
    assert req.table_idx == -1 and sent == []
    Scheduler._flush_abort_acks(scheduler)
    assert len(sent) == 1
    snapshot = cm.page_usage()
    assert snapshot[0] == 0
    assert (sent[0].kv_used_pages, sent[0].kv_total_pages) == snapshot
    wire = ErrorReplyMsg.decoder(ErrorReplyMsg.encoder(sent[0]))
    tracker.observe(_error_reply(wire))
    assert tracker.active == 0 and tracker.completed == 0
    assert (tracker.kv_used_pages, tracker.kv_total_pages) == snapshot
    cm.check_integrity()


@pytest.mark.parametrize("used", [0, 3])
def test_abort_snapshot_preserves_all_pools_and_other_live_requests(used):
    sent = []
    scheduler = SimpleNamespace(
        _pending_abort_acks={5, 7}, send_result=sent.extend,
        _kv_usage_pages=Mock(return_value=(used, 64)),
        _mamba_slot_usage=Mock(return_value=(used, 16)),
        _swa_token_usage=Mock(return_value=(used * 128, 1024)),
        _gpu_mem_bytes=Mock(return_value=4096),
    )
    tracker = StatsTracker()
    for uid in (5, 7, 9):
        tracker.on_new_user(uid)
    tracker.on_abort(5)
    tracker.on_abort(7)
    Scheduler._flush_abort_acks(scheduler)
    assert [msg.uid for msg in sent] == [5, 7]
    assert not scheduler._pending_abort_acks
    for msg in sent:
        tracker.observe(_error_reply(ErrorReplyMsg.decoder(ErrorReplyMsg.encoder(msg))))
    assert tracker.inflight_uids == (9,) and tracker.completed == 0
    assert (tracker.kv_used_pages, tracker.kv_total_pages) == (used, 64)
    assert (tracker.mamba_used_slots, tracker.mamba_total_slots) == (used, 16)
    assert (tracker.swa_used_tokens, tracker.swa_total_tokens) == (used * 128, 1024)
    assert tracker.vram_bytes == 4096
    for name in ("_kv_usage_pages", "_mamba_slot_usage", "_swa_token_usage", "_gpu_mem_bytes"):
        getattr(scheduler, name).assert_called_once()


def test_early_error_does_not_invent_a_zero_occupancy_snapshot():
    tracker = StatsTracker()
    tracker.observe(UserReply(uid=9, incremental_output="x", finished=False,
                              kv_used_pages=3, kv_total_pages=64,
                              swa_used_tokens=384, swa_total_tokens=1024))
    tracker.observe(_error_reply(ErrorReplyMsg(uid=5, error="prompt too long",
                                              code="context_length_exceeded")))
    assert (tracker.kv_used_pages, tracker.kv_total_pages) == (3, 64)
    assert (tracker.swa_used_tokens, tracker.swa_total_tokens) == (384, 1024)


def test_empty_idle_abort_flush_does_not_query_any_pool_or_gpu():
    scheduler = SimpleNamespace(_pending_abort_acks=set(), send_result=Mock())
    for name in ("_kv_usage_pages", "_mamba_slot_usage", "_swa_token_usage", "_gpu_mem_bytes"):
        setattr(scheduler, name, Mock(side_effect=AssertionError("idle query")))
    Scheduler._flush_abort_acks(scheduler)
    scheduler.send_result.assert_not_called()
    for name in ("_kv_usage_pages", "_mamba_slot_usage", "_swa_token_usage", "_gpu_mem_bytes"):
        getattr(scheduler, name).assert_not_called()
