from __future__ import annotations

import os
import time
from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from freetoken.scheduler.io import SchedulerIOMixin


class _IdlePullQueue:
    def wait(self, timeout_ms: int) -> bool:
        # Stay below the deliberately short process-group timeout while still
        # making the worker spend real time waiting in Gloo.
        time.sleep(0.6)
        return False

    def empty(self) -> bool:
        return True


def _idle_rank(rank: int, init_path: str) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=2),
    )
    try:
        io = SchedulerIOMixin.__new__(SchedulerIOMixin)
        io.tp_cpu_group = dist.distributed_c10d._get_default_group()
        io.run_when_idle = lambda: None
        if rank == 0:
            io._recv_from_tokenizer = _IdlePullQueue()
        for _ in range(5):
            received = (
                io._recv_msg_multi_rank0(blocking=True)
                if rank == 0
                else io._recv_msg_multi_rank1(blocking=True)
            )
            assert received == []
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_idle_heartbeat_survives_longer_than_group_timeout(tmp_path):
    # Five 0.6-second idle intervals exceed the two-second Gloo timeout in
    # aggregate. Each heartbeat is matched, so no individual collective times
    # out even though the scheduler remains idle for longer than that timeout.
    mp.spawn(_idle_rank, args=(str(tmp_path / "gloo-init"),), nprocs=2, join=True)
