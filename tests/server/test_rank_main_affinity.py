from types import SimpleNamespace
import threading

import pytest

from freetoken.server import rank_affinity as affinity


def config():
    return SimpleNamespace(moe_backend="hybrid", tp_info=SimpleNamespace(size=4),
                           moe_hybrid_ranks=(1, 2), moe_cpu_threads=24)


@pytest.fixture
def fake_host(monkeypatch):
    current = [set(range(112))]
    calls = []
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda pid: current[0].copy())
    def set_affinity(pid, cpus):
        assert pid == 0
        calls.append((pid, set(cpus)))
        current[0] = set(cpus)
    monkeypatch.setattr(affinity.os, "sched_setaffinity", set_affinity)
    return current, calls


def test_default_is_noop_without_config_or_topology(monkeypatch):
    monkeypatch.delenv(affinity.ENV_NAME, raising=False)
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _: pytest.fail("default touched affinity"))
    assert affinity.prepare_rank_main_affinity(None) is None


@pytest.mark.parametrize("raw", ["", "1,", "-1", "2-1", "1-2-3", "abc", "1,1", "1-3,3", "0-1000000"])
def test_bad_lists_rejected(raw):
    with pytest.raises(ValueError):
        affinity.parse_cpu_list(raw)


def test_smt_list_and_whitespace():
    assert affinity.parse_cpu_list(" 56-82, 84-110 ") == frozenset(range(56, 83)) | frozenset(range(84, 111))


def test_prepare_does_not_pin_and_apply_changes_only_caller(monkeypatch, fake_host):
    current, calls = fake_host
    monkeypatch.setenv(affinity.ENV_NAME, "56-82,84-110")
    import freetoken.moe.cpu_executor as cpu
    seen = []
    def partition(rank, ranks, threads):
        assert current[0] == set(range(112))
        seen.append(rank)
        return (list(range(24)), 27) if rank == 1 else (list(range(28, 52)), 55)
    monkeypatch.setattr(cpu, "partition_rank_affinity", partition)
    prepared = affinity.prepare_rank_main_affinity(config())
    assert seen == [1, 2] and calls == []
    prepared.apply()
    assert calls == [(0, set(prepared.cpus))]


def test_exact_worker_overlap_rejected_but_smt_allowed(monkeypatch, fake_host):
    monkeypatch.setattr(affinity, "_reserved_worker_cpus", lambda _: {0, 27, 55})
    for raw in ("0", "27", "55"):
        monkeypatch.setenv(affinity.ENV_NAME, raw)
        with pytest.raises(ValueError, match="overlap"):
            affinity.prepare_rank_main_affinity(config())
    monkeypatch.setenv(affinity.ENV_NAME, "56")
    assert affinity.prepare_rank_main_affinity(config()).cpus == {56}


def test_outside_cpuset_rejected(monkeypatch, fake_host):
    monkeypatch.setenv(affinity.ENV_NAME, "112")
    with pytest.raises(ValueError, match="outside"):
        affinity.prepare_rank_main_affinity(config())


def test_changed_mask_during_init_rejected(monkeypatch, fake_host):
    current, calls = fake_host
    monkeypatch.setenv(affinity.ENV_NAME, "56")
    monkeypatch.setattr(affinity, "_reserved_worker_cpus", lambda _: set())
    prepared = affinity.prepare_rank_main_affinity(config())
    current[0] = {56, 57}
    with pytest.raises(RuntimeError, match="during initialization"):
        prepared.apply()
    assert not calls


def test_mismatched_tid_rejected(monkeypatch, fake_host):
    prepared = affinity.RankMainAffinity(frozenset({56}), frozenset(range(112)),
                                         affinity.os.getpid(), threading.get_native_id() + 1)
    with pytest.raises(RuntimeError, match="preparing thread"):
        prepared.apply()


def test_failed_application_restores_mask(monkeypatch, fake_host):
    current, calls = fake_host
    prepared = affinity.RankMainAffinity(frozenset({56, 57}), frozenset(range(112)),
                                         affinity.os.getpid(), threading.get_native_id())
    def partial(pid, cpus):
        calls.append((pid, set(cpus)))
        current[0] = {56} if len(cpus) == 2 else set(cpus)
    monkeypatch.setattr(affinity.os, "sched_setaffinity", partial)
    with pytest.raises(RuntimeError, match="exactly"):
        prepared.apply()
    assert current[0] == set(range(112)) and len(calls) == 2


def test_launcher_applies_after_initialization_before_readiness():
    import inspect
    from freetoken.server.launch import _run_scheduler
    source = inspect.getsource(_run_scheduler)
    assert source.index("prepare_rank_main_affinity(args)") < source.index("scheduler = Scheduler(args)")
    assert source.index("scheduler.sync_all_ranks()") < source.index("main_affinity.apply()")
    assert source.index("main_affinity.apply()") < source.index('ack_queue.put("Scheduler is ready")')


def test_coordinator_only_exclusion_preserves_workers_and_smt(monkeypatch, fake_host):
    current, calls = fake_host
    monkeypatch.delenv(affinity.ENV_NAME, raising=False)
    monkeypatch.setenv(affinity.COORDINATOR_ENV_NAME, "1")
    import freetoken.moe.cpu_executor as cpu
    seen = []
    def partition(rank, ranks, threads):
        assert current[0] == set(range(112))
        seen.append((rank, ranks, threads))
        return (list(range(24)), 27) if rank == 1 else (list(range(28, 52)), 55)
    monkeypatch.setattr(cpu, "partition_rank_affinity", partition)
    prepared = affinity.prepare_rank_main_affinity(config())
    assert prepared.cpus == frozenset(range(112)) - {27, 55}
    assert {0, 23, 28, 51, 83, 111} <= prepared.cpus
    assert seen == [(1, (1, 2), 24), (2, (1, 2), 24)] and not calls
    prepared.apply()
    assert calls == [(0, set(prepared.cpus))]


def test_explicit_disabled_coordinator_switch_is_noop(monkeypatch):
    monkeypatch.delenv(affinity.ENV_NAME, raising=False)
    monkeypatch.setenv(affinity.COORDINATOR_ENV_NAME, "0")
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _: pytest.fail("disabled touched affinity"))
    assert affinity.prepare_rank_main_affinity(None) is None


@pytest.mark.parametrize("raw", ["", "true", "2", "-1"])
def test_invalid_coordinator_switch_rejected(monkeypatch, raw):
    monkeypatch.setenv(affinity.COORDINATOR_ENV_NAME, raw)
    with pytest.raises(ValueError, match="expected 0 or 1"):
        affinity.prepare_rank_main_affinity(None)


def test_two_affinity_policies_rejected(monkeypatch):
    monkeypatch.setenv(affinity.ENV_NAME, "56-111")
    monkeypatch.setenv(affinity.COORDINATOR_ENV_NAME, "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        affinity.prepare_rank_main_affinity(None)


@pytest.mark.parametrize("backend,size,ranks", [("offload", 4, (1, 2)), ("hybrid", 1, (0,)), ("hybrid", 4, ())])
def test_coordinator_switch_requires_distributed_cpu_pools(monkeypatch, fake_host, backend, size, ranks):
    monkeypatch.delenv(affinity.ENV_NAME, raising=False)
    monkeypatch.setenv(affinity.COORDINATOR_ENV_NAME, "1")
    c = config()
    c.moe_backend, c.tp_info.size, c.moe_hybrid_ranks = backend, size, ranks
    with pytest.raises(ValueError, match="distributed"):
        affinity.prepare_rank_main_affinity(c)


def test_coordinator_exclusion_uses_restricted_inherited_topology(monkeypatch, fake_host):
    current, _ = fake_host
    current[0] = {4, 6, 8, 12}
    monkeypatch.delenv(affinity.ENV_NAME, raising=False)
    monkeypatch.setenv(affinity.COORDINATOR_ENV_NAME, "1")
    import freetoken.moe.cpu_executor as cpu
    monkeypatch.setattr(cpu, "partition_rank_affinity", lambda r, ranks, threads: ([4], 6) if r == 1 else ([8], 12))
    assert affinity.prepare_rank_main_affinity(config()).cpus == {4, 8}


@pytest.mark.parametrize("reserved", [{112}, set(range(112))])
def test_invalid_or_empty_remaining_mask_rejected(monkeypatch, fake_host, reserved):
    monkeypatch.delenv(affinity.ENV_NAME, raising=False)
    monkeypatch.setenv(affinity.COORDINATOR_ENV_NAME, "1")
    monkeypatch.setattr(affinity, "_reserved_worker_cpus", lambda c, **kw: reserved)
    with pytest.raises(ValueError):
        affinity.prepare_rank_main_affinity(config())
