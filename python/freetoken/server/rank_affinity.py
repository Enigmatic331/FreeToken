"""Opt-in Linux affinity for rank-main threads, after worker initialization.

This is logical-CPU separation, not physical-core isolation: SMT siblings may
be selected explicitly. Existing helper threads are never repinned; helpers
created later may inherit the main thread's mask. Keep the switch unset for
the original scheduler behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
import threading

ENV_NAME = "FREETOKEN_RANK_MAIN_CPUS"
COORDINATOR_ENV_NAME = "FREETOKEN_RANK_MAIN_EXCLUDE_COORDINATORS"


def parse_cpu_list(value: str) -> frozenset[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", part):
            raise ValueError(f"{ENV_NAME}: expected a nonempty CPU list, e.g. 56-82,84-110")
        bounds = part.split("-")
        first, last = int(bounds[0]), int(bounds[-1])
        if last < first or last - first > 65535:
            raise ValueError(f"{ENV_NAME}: invalid CPU range {part!r}")
        values = set(range(first, last + 1))
        if cpus & values:
            raise ValueError(f"{ENV_NAME}: overlapping CPU entries")
        cpus.update(values)
    return frozenset(cpus)


def _reserved_worker_cpus(config, *, coordinators_only: bool = False) -> set[int]:
    if config.moe_backend not in ("cpu", "hybrid"):
        return set()
    if config.tp_info.size <= 1:
        raise ValueError(f"{ENV_NAME}: CPU/hybrid affinity currently requires distributed ranks")
    from freetoken.moe.cpu_executor import partition_rank_affinity

    ranks = (config.moe_hybrid_ranks
             if config.moe_backend == "hybrid" and config.moe_hybrid_ranks is not None
             else tuple(range(config.tp_info.size)))
    reserved: set[int] = set()
    for rank in ranks:
        workers, coordinator = partition_rank_affinity(rank, ranks, config.moe_cpu_threads)
        if not coordinators_only:
            reserved.update(workers)
        reserved.add(coordinator)
    return reserved


@dataclass(frozen=True)
class RankMainAffinity:
    cpus: frozenset[int]
    original: frozenset[int]
    pid: int
    tid: int

    def apply(self) -> None:
        if os.getpid() != self.pid or threading.get_native_id() != self.tid:
            raise RuntimeError("Rank-main affinity must be applied by the preparing thread")
        if frozenset(os.sched_getaffinity(0)) != self.original:
            raise RuntimeError("Rank-main affinity changed during initialization")
        try:
            # Linux pid=0 means this calling thread, NOT all process threads.
            os.sched_setaffinity(0, self.cpus)
            if frozenset(os.sched_getaffinity(0)) != self.cpus:
                raise RuntimeError("Rank-main CPU mask was not applied exactly")
        except BaseException:
            os.sched_setaffinity(0, self.original)
            raise


def prepare_rank_main_affinity(config) -> RankMainAffinity | None:
    raw = os.environ.get(ENV_NAME, "").strip()
    exclude_coordinators = os.environ.get(COORDINATOR_ENV_NAME, "0").strip()
    if exclude_coordinators not in ("0", "1"):
        raise ValueError(f"{COORDINATOR_ENV_NAME}: expected 0 or 1")
    if raw and exclude_coordinators == "1":
        raise ValueError(f"{ENV_NAME} and {COORDINATOR_ENV_NAME} are mutually exclusive")
    if not raw and exclude_coordinators == "0":
        return None  # No affinity/topology work on the default path.
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError(f"{ENV_NAME} requires Linux thread affinity")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Rank-main affinity must be prepared on the main thread")
    original = frozenset(os.sched_getaffinity(0))
    if exclude_coordinators == "1":
        # Derive from the SAME unrestricted topology used by the CPU pools.
        # Only protect their coordinator logical CPUs; workers and SMT siblings
        # remain available to main threads. Existing helpers are never repinned.
        reserved = _reserved_worker_cpus(config, coordinators_only=True)
        if not reserved or not reserved <= original:
            raise ValueError(f"{COORDINATOR_ENV_NAME}: requires distributed CPU pools in the inherited mask")
        cpus = original - reserved
        if not cpus:
            raise ValueError(f"{COORDINATOR_ENV_NAME}: no main-thread CPUs remain")
        return RankMainAffinity(cpus, original, os.getpid(), threading.get_native_id())
    cpus = parse_cpu_list(raw)
    if not cpus <= original:
        raise ValueError(f"{ENV_NAME}: requested CPUs are outside the inherited allowed mask")
    # Compute the existing pool allocation BEFORE narrowing the main mask;
    # physical_core_cpus() intentionally consults the calling thread's mask.
    if cpus & _reserved_worker_cpus(config):
        raise ValueError(f"{ENV_NAME}: main CPUs overlap worker/coordinator logical CPUs")
    return RankMainAffinity(cpus, original, os.getpid(), threading.get_native_id())
