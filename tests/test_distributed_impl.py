from __future__ import annotations

import torch
import torch.distributed as dist

from freetoken.distributed.impl import TorchDistributedImpl


class _Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def test_torch_p2p_uses_batched_nccl_group_boundaries(monkeypatch):
    calls = []
    works = []
    isend = object()
    irecv = object()

    monkeypatch.setattr(dist, "isend", isend)
    monkeypatch.setattr(dist, "irecv", irecv)
    monkeypatch.setattr(
        dist,
        "P2POp",
        lambda op, tensor, peer: (op, tensor, peer),
    )

    def batch(ops):
        calls.append(ops)
        work = _Work()
        works.append(work)
        return [work]

    monkeypatch.setattr(dist, "batch_isend_irecv", batch)

    communicator = TorchDistributedImpl()
    tensor = torch.arange(4)
    assert communicator.send(tensor, 1) is tensor
    assert communicator.recv(tensor, 0) is tensor

    assert calls == [[(isend, tensor, 1)], [(irecv, tensor, 0)]]
    assert all(work.waited for work in works)
