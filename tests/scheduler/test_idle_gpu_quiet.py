from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from freetoken.scheduler.scheduler import Scheduler


def _scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._idle_checks_done = False
    scheduler._last_data = None
    scheduler._pending_rebuild = None
    scheduler._speculative = None
    scheduler.prefill_manager = SimpleNamespace(runnable=False)
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.cache_manager = SimpleNamespace(check_integrity=Mock())
    scheduler._log_moe_cache_stats = Mock()
    scheduler.stream = Mock()
    scheduler.engine = SimpleNamespace(stream=Mock())
    scheduler.engine_stream_ctx = nullcontext()
    scheduler._process_one_msg = Mock()
    scheduler._schedule_next_batch = Mock(return_value=None)
    scheduler._restore_linear_states = Mock()
    scheduler._forward = Mock(return_value=object())
    scheduler._process_last_data = Mock()
    scheduler._flush_abort_acks = Mock()
    return scheduler


def _loop(scheduler: Scheduler, overlap: bool, last_data=None):
    if overlap:
        return scheduler.overlap_loop(last_data)
    return scheduler.normal_loop()


@pytest.mark.parametrize("overlap", [False, True])
def test_repeated_idle_heartbeats_do_not_submit_gpu_work(overlap):
    scheduler = _scheduler()

    def heartbeat(*, blocking):
        assert blocking
        scheduler.run_when_idle()
        return []

    scheduler.receive_msg = Mock(side_effect=heartbeat)
    for _ in range(20):
        assert _loop(scheduler, overlap) is None

    assert scheduler.receive_msg.call_count == 20
    scheduler.cache_manager.check_integrity.assert_called_once()
    scheduler._log_moe_cache_stats.assert_called_once()
    scheduler.stream.wait_stream.assert_not_called()
    scheduler.engine.stream.wait_stream.assert_not_called()
    scheduler._schedule_next_batch.assert_not_called()


@pytest.mark.parametrize("overlap", [False, True])
def test_message_resume_preserves_stream_ordering_and_rechecks_idle(overlap):
    scheduler = _scheduler()
    scheduler.run_when_idle()
    message = object()
    forward_input = SimpleNamespace(batch=object())
    scheduler.receive_msg = Mock(return_value=[message])
    scheduler._schedule_next_batch.return_value = forward_input

    _loop(scheduler, overlap)

    scheduler._process_one_msg.assert_called_once_with(message)
    scheduler._restore_linear_states.assert_called_once_with(forward_input.batch)
    scheduler._forward.assert_called_once_with(forward_input)
    if overlap:
        assert scheduler.stream.wait_stream.call_count == 2
        scheduler.engine.stream.wait_stream.assert_called_once_with(scheduler.stream)
    else:
        scheduler._process_last_data.assert_called_once_with(
            (forward_input, scheduler._forward.return_value)
        )
    assert not scheduler._idle_checks_done
    scheduler.run_when_idle()
    assert scheduler.cache_manager.check_integrity.call_count == 2


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("manager", ["prefill_manager", "decode_manager"])
def test_already_queued_work_resumes_without_a_new_message(overlap, manager):
    scheduler = _scheduler()
    scheduler.run_when_idle()
    getattr(scheduler, manager).runnable = True
    scheduler.receive_msg = Mock(return_value=[])
    scheduler._schedule_next_batch.return_value = SimpleNamespace(batch=object())

    _loop(scheduler, overlap)

    scheduler.receive_msg.assert_called_once_with(blocking=False)
    scheduler._forward.assert_called_once()
    assert not scheduler._idle_checks_done


def test_normal_idle_still_flushes_cpu_abort_acknowledgements():
    scheduler = _scheduler()
    scheduler.receive_msg = Mock(return_value=[])

    scheduler.normal_loop()

    scheduler._flush_abort_acks.assert_called_once()
    scheduler._process_last_data.assert_not_called()
    scheduler._schedule_next_batch.assert_not_called()


def test_inflight_speculative_batch_is_drained_before_idle():
    scheduler = _scheduler()
    scheduler._idle_checks_done = True
    scheduler._speculative = object()
    scheduler.receive_msg = Mock(return_value=[])
    last_data = object()

    assert scheduler.overlap_loop(last_data) is None

    scheduler.receive_msg.assert_called_once_with(blocking=False)
    scheduler._process_last_data.assert_called_once_with(last_data)
    scheduler._flush_abort_acks.assert_called_once()
    assert scheduler.stream.wait_stream.call_count == 3
    assert not scheduler._idle_checks_done


@pytest.mark.parametrize("overlap", [False, True])
def test_pending_rebuild_is_not_skipped_and_invalidates_idle_check(overlap):
    scheduler = _scheduler()
    scheduler.run_when_idle()
    scheduler._pending_rebuild = object()
    scheduler.receive_msg = Mock(return_value=[])

    def rebuild():
        scheduler._pending_rebuild = None

    scheduler._execute_pending_rebuild = Mock(side_effect=rebuild)
    _loop(scheduler, overlap)

    scheduler.receive_msg.assert_called_once_with(blocking=False)
    scheduler._execute_pending_rebuild.assert_called_once()
    assert not scheduler._idle_checks_done
    scheduler.run_when_idle()
    assert scheduler.cache_manager.check_integrity.call_count == 2


def test_failed_integrity_check_is_not_marked_complete():
    scheduler = _scheduler()
    scheduler.cache_manager.check_integrity.side_effect = RuntimeError("bad cache")

    with pytest.raises(RuntimeError, match="bad cache"):
        scheduler.run_when_idle()

    assert not scheduler._idle_checks_done
