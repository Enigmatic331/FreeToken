#!/usr/bin/env python3
"""Stress the V4.1 heterogeneous-EP CUDA-graph collective sequence.

The ordinary PyNCCL graph smoke test covers three adjacent all-reduces.  V4.1
instead captures 40 asymmetric authority/worker layers and then immediately
issues an eager next-token broadcast on the same communicator.  This probe
reproduces that ordering without loading the checkpoint or expert banks.
"""

from __future__ import annotations

import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _broadcast(comm, tensor: torch.Tensor, rank: int) -> None:
    if rank != 0:
        tensor.zero_()
    comm.all_reduce(tensor, "sum")


def _worker(
    rank: int,
    world_size: int,
    port: int,
    layers: int,
    replays: int,
    sync_before_eager: bool,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )

    from freetoken.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=rank,
        tp_size=world_size,
        tp_cpu_group=dist.group.WORLD,
        max_size_bytes=1 << 20,
    )
    device = torch.device("cuda", rank)
    hidden = torch.full((5120,), rank + 1.0, dtype=torch.bfloat16, device=device)
    weights = torch.full((6,), rank + 1.0, dtype=torch.float32, device=device)
    ids = torch.full((6,), rank + 1, dtype=torch.int32, device=device)
    routes = torch.full((6, 5120), rank + 1.0, dtype=torch.bfloat16, device=device)
    engram_ids = torch.full((48,), rank + 1, dtype=torch.int32, device=device)
    engram_rows = torch.full((6144,), rank + 1.0, dtype=torch.bfloat16, device=device)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    dist.barrier()
    with torch.cuda.stream(capture_stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for layer in range(layers):
                if layer in (1, 14):
                    _broadcast(comm, engram_ids, rank)
                    comm.all_reduce(engram_rows, "sum")
                _broadcast(comm, hidden, rank)
                _broadcast(comm, weights, rank)
                _broadcast(comm, ids, rank)
                # Preserve V4.1's asymmetric interval between collectives: the
                # authority runs router/shared+dense work, while the worker runs
                # only its local routed experts.
                torch.cuda._sleep(120_000 if rank == 0 else 80_000)
                comm.all_reduce(routes, "sum")
            if rank == 0:
                torch.cuda._sleep(300_000)  # final norm/head before sampling

    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)
    dist.barrier()
    print(f"rank={rank} capture=complete", flush=True)

    for replay in range(replays):
        graph.replay()
        print(f"rank={rank} replay={replay} submitted", flush=True)
        if sync_before_eager:
            torch.cuda.synchronize(device)
            print(f"rank={rank} replay={replay} synchronized", flush=True)

        # Engine.forward_batch performs this eager source-only all-reduce after
        # GraphRunner.replay (and authority sampling) to distribute next_tokens.
        next_token = torch.tensor(
            [17 if rank == 0 else 0], dtype=torch.int32, device=device
        )
        _broadcast(comm, next_token, rank)
        print(f"rank={rank} replay={replay} eager-broadcast-submitted", flush=True)
        torch.cuda.synchronize(device)
        if int(next_token.item()) != 17:
            raise RuntimeError(
                f"rank {rank}: replay {replay} received {int(next_token.item())}"
            )
        dist.barrier()
        print(f"rank={rank} replay={replay} complete", flush=True)

    dist.destroy_process_group()
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--sync-before-eager", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    mp.spawn(
        _worker,
        args=(2, _free_port(), args.layers, args.replays, args.sync_before_eager),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    main()
