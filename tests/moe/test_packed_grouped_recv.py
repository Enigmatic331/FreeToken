from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import freetoken.kernel as kernel_module
import freetoken.layers.moe as moe_module
from freetoken.distributed.impl import DistributedCommunicator, TorchDistributedImpl
from freetoken.layers.moe import ExpertParallelOffloadMoELayer

HIDDEN = 4
PEER_RANGES = ((1, 4, 8), (2, 8, 12), (3, 12, 16))


class _Stream:
    def __init__(self, name):
        self.name = name
        self.waited_on = []

    def wait_stream(self, other):
        self.waited_on.append(other)


class _Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


@pytest.fixture
def ep4_root(monkeypatch):
    """EP4 authority with the overlapped receive path on fake CPU streams."""
    compute, recv = _Stream("compute"), _Stream("recv")
    monkeypatch.setattr(moe_module, "_EP_PACKED_PREFILL_OVERLAP", True)
    monkeypatch.setattr(moe_module, "_EP_PREFILL_ROUTE_TILE_TOKENS", 0)
    monkeypatch.setattr(moe_module, "_EP_PACKED_WIRE_DTYPE", "bf16")
    monkeypatch.setattr(moe_module, "_ep_packed_recv_stream", lambda _device: recv)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: compute)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        moe_module, "get_tp_info", lambda: SimpleNamespace(rank=0, size=4)
    )
    monkeypatch.setattr(
        moe_module,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_moe_prefill=True)),
    )
    monkeypatch.setattr(
        kernel_module,
        "moe_sum_reduce_triton",
        lambda routes, output: output.copy_(routes.float().sum(dim=1)),
    )
    return SimpleNamespace(compute=compute, recv=recv)


def _layer(peers=(1, 2, 3), communicator=None):
    layer = ExpertParallelOffloadMoELayer(
        layer_id=0, num_experts=4, top_k=4, hidden_size=HIDDEN, intermediate_size=8
    )
    layer.packed_prefill_root = 0
    layer.packed_prefill_peer_ranks = peers
    layer.packed_prefill_peer_ranges = tuple(
        entry for entry in PEER_RANGES if entry[0] in peers
    )
    if communicator is not None:
        layer.prefill_communicator = communicator
    return layer


def _routes(ids, local_weights):
    """Per-route outputs on the root plus each peer's owned rows in send order."""
    full = (
        torch.arange(ids.numel() * HIDDEN, dtype=torch.float32)
        .view(*ids.shape, HIDDEN)
        .mul_(0.37)
        .sin_()
        .to(torch.bfloat16)
    )
    flat_ids = ids.reshape(-1)
    owned = local_weights.ne(0).reshape(-1)
    root = full.clone()
    root.view(-1, HIDDEN)[~owned] = 0
    sends = {}
    for peer, start, stop in PEER_RANGES:
        rows = torch.nonzero(
            (flat_ids >= start) & (flat_ids < stop), as_tuple=False
        ).flatten()
        sends[peer] = full.view(-1, HIDDEN).index_select(0, rows).contiguous()
    return full, root, sends


def _wire(sends, wire_dtype):
    """Each peer's ordered send queue as produced by the worker."""
    if wire_dtype == "bf16":
        return {peer: [packed] for peer, packed in sends.items() if packed.numel()}
    queues = {}
    for peer, packed in sends.items():
        if packed.numel():
            queues[peer] = list(moe_module._quantize_ep_routes(packed))
    return queues


class _Recorder:
    """Shared fake NCCL: every receive pops the next tensor sent by that peer."""

    def __init__(self, queues):
        self.queues = {peer: list(items) for peer, items in queues.items()}
        self.events = []

    def deliver(self, tensor, peer):
        payload = self.queues[peer].pop(0)
        # NCCL moves contiguous elements; FP8 workers send [rows, hidden/4]
        # int32 words while the root receives the same words flattened.
        assert payload.numel() == tensor.numel() and payload.dtype == tensor.dtype
        tensor.copy_(payload.reshape(tensor.shape))


class _SequentialCommunicator:
    def __init__(self, recorder):
        self.recorder = recorder

    def recv(self, tensor, peer):
        self.recorder.events.append(("recv", peer, tensor.dtype))
        self.recorder.deliver(tensor, peer)
        return tensor


class _GroupedCommunicator(_SequentialCommunicator):
    def __init__(self, recorder):
        super().__init__(recorder)
        self.buffers = []

    def recv_many(self, ops):
        ops = list(ops)
        self.buffers.append([tensor for tensor, _ in ops])
        self.recorder.events.append(
            ("recv_many", tuple((peer, tensor.dtype) for tensor, peer in ops))
        )
        for tensor, peer in ops:
            self.recorder.deliver(tensor, peer)


def _ep4_case():
    ids = torch.tensor(
        [[0, 4, 8, 12], [3, 7, 11, 15], [1, 9, 13, 5], [2, 6, 14, 10]],
        dtype=torch.int32,
    )
    local_weights = ids.lt(4).float()
    return ids, local_weights


def _run(layer, ids, local_weights, root_routes):
    layer.prepare_packed_prefill_receive(
        local_weights, torch.bfloat16, global_topk_ids=ids
    )
    entries, _stream = layer._packed_prefill_pending
    output = layer._maybe_combine_packed_prefill_routes(
        root_routes.clone(), local_weights
    )
    return entries, output


@pytest.mark.parametrize("wire_dtype", ["bf16", "fp8"])
def test_grouped_receive_matches_sequential_bitwise(monkeypatch, ep4_root, wire_dtype):
    monkeypatch.setattr(moe_module, "_EP_PACKED_WIRE_DTYPE", wire_dtype)
    ids, local_weights = _ep4_case()
    _full, root_routes, sends = _routes(ids, local_weights)

    monkeypatch.setattr(moe_module, "_EP_PACKED_GROUPED_P2P", False)
    sequential = _Recorder(_wire(sends, wire_dtype))
    _entries, expected = _run(
        _layer(communicator=_SequentialCommunicator(sequential)),
        ids, local_weights, root_routes,
    )

    monkeypatch.setattr(moe_module, "_EP_PACKED_GROUPED_P2P", True)
    grouped = _Recorder(_wire(sends, wire_dtype))
    communicator = _GroupedCommunicator(grouped)
    entries, output = _run(
        _layer(communicator=communicator), ids, local_weights, root_routes
    )

    assert torch.equal(output, expected)
    assert all(not queue for queue in grouped.queues.values())
    # One grouped launch, peers ascending, and per-peer scale before payload.
    per_peer = (
        ((torch.bfloat16, torch.int32) if wire_dtype == "fp8" else (torch.bfloat16,))
    )
    assert grouped.events == [
        (
            "recv_many",
            tuple((peer, dtype) for peer in (1, 2, 3) for dtype in per_peer),
        )
    ]
    assert [peer for peer, *_ in entries] == [1, 2, 3]
    # The pending entries own exactly the buffers handed to the launch.
    launched = communicator.buffers[0]
    expected_buffers = []
    for _peer, _indices, packed, scale in entries:
        if scale is not None:
            expected_buffers.append(scale)
        expected_buffers.append(packed)
    assert len(launched) == len(expected_buffers)
    assert all(a is b for a, b in zip(launched, expected_buffers))
    assert len({tensor.data_ptr() for tensor in launched}) == len(launched)
    assert sequential.events == [
        ("recv", peer, dtype) for peer in (1, 2, 3) for dtype in per_peer
    ]


def test_grouped_receive_uses_one_torch_batch_and_waits_every_work(
    monkeypatch, ep4_root
):
    ids, local_weights = _ep4_case()
    _full, root_routes, sends = _routes(ids, local_weights)

    monkeypatch.setattr(moe_module, "_EP_PACKED_GROUPED_P2P", False)
    _entries, expected = _run(
        _layer(communicator=_SequentialCommunicator(_Recorder(_wire(sends, "bf16")))),
        ids, local_weights, root_routes,
    )

    recorder = _Recorder(_wire(sends, "bf16"))
    calls = []
    works = []
    irecv = object()
    monkeypatch.setattr(dist, "irecv", irecv)
    monkeypatch.setattr(dist, "P2POp", lambda op, tensor, peer: (op, tensor, peer))

    def batch(ops):
        calls.append([(op, peer) for op, _tensor, peer in ops])
        for _op, tensor, peer in ops:
            recorder.deliver(tensor, peer)
        # NCCL coalescing may return one or several handles; wait them all.
        launched = [_Work(), _Work()]
        works.extend(launched)
        return launched

    monkeypatch.setattr(dist, "batch_isend_irecv", batch)
    monkeypatch.setattr(DistributedCommunicator, "plugins", [TorchDistributedImpl()])
    monkeypatch.setattr(moe_module, "_EP_PACKED_GROUPED_P2P", True)
    ep4_root.recv.waited_on.clear()
    ep4_root.compute.waited_on.clear()
    entries, output = _run(_layer(), ids, local_weights, root_routes)

    assert calls == [[(irecv, 1), (irecv, 2), (irecv, 3)]]
    assert works and all(work.waited for work in works)
    assert torch.equal(output, expected)
    assert ep4_root.recv.waited_on == [ep4_root.compute]
    assert ep4_root.compute.waited_on == [ep4_root.recv]
    assert [peer for peer, *_ in entries] == [1, 2, 3]


def test_flag_off_keeps_original_sequential_path(monkeypatch, ep4_root):
    ids, local_weights = _ep4_case()
    _full, root_routes, sends = _routes(ids, local_weights)
    monkeypatch.setattr(moe_module, "_EP_PACKED_GROUPED_P2P", False)
    recorder = _Recorder(_wire(sends, "bf16"))

    class NoGroupedCalls(_GroupedCommunicator):
        def recv_many(self, ops):
            raise AssertionError("flag-off path must not use grouped receives")

    _run(_layer(communicator=NoGroupedCalls(recorder)), ids, local_weights, root_routes)
    assert recorder.events == [("recv", peer, torch.bfloat16) for peer in (1, 2, 3)]


def test_grouped_flag_falls_back_without_recv_many(monkeypatch, ep4_root):
    # e.g. the two-rank prefill subgroup communicator, which has no grouped API.
    ids, local_weights = _ep4_case()
    _full, root_routes, sends = _routes(ids, local_weights)
    monkeypatch.setattr(moe_module, "_EP_PACKED_GROUPED_P2P", False)
    _entries, expected = _run(
        _layer(communicator=_SequentialCommunicator(_Recorder(_wire(sends, "bf16")))),
        ids, local_weights, root_routes,
    )
    monkeypatch.setattr(moe_module, "_EP_PACKED_GROUPED_P2P", True)
    recorder = _Recorder(_wire(sends, "bf16"))
    _entries, output = _run(
        _layer(communicator=_SequentialCommunicator(recorder)),
        ids, local_weights, root_routes,
    )
    assert torch.equal(output, expected)
    assert recorder.events == [("recv", peer, torch.bfloat16) for peer in (1, 2, 3)]


def test_grouped_receive_skips_empty_peers_and_zero_peer_tiles(monkeypatch, ep4_root):
    monkeypatch.setattr(moe_module, "_EP_PACKED_GROUPED_P2P", True)
    calls = []
    monkeypatch.setattr(dist, "batch_isend_irecv", lambda ops: calls.append(ops))
    monkeypatch.setattr(DistributedCommunicator, "plugins", [TorchDistributedImpl()])

    # Every route is local: all peers are present but have nothing to send.
    ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    local_weights = torch.ones(1, 4)
    _full, root_routes, _sends = _routes(ids, local_weights)
    entries, output = _run(_layer(), ids, local_weights, root_routes)
    assert calls == []
    assert [(peer, packed) for peer, _i, packed, _s in entries] == [
        (1, None), (2, None), (3, None)
    ]
    assert ep4_root.compute.waited_on == []
    torch.testing.assert_close(output, root_routes.float().sum(dim=1).to(output.dtype))

    # Only peer 2 owns a route: the launch carries exactly that one receive.
    recorder = _Recorder({})
    communicator = _GroupedCommunicator(recorder)
    ids = torch.tensor([[0, 1, 9, 3]], dtype=torch.int32)
    local_weights = ids.lt(4).float()
    _full, root_routes, sends = _routes(ids, local_weights)
    recorder.queues = _wire(sends, "bf16")
    entries, output = _run(
        _layer(communicator=communicator), ids, local_weights, root_routes
    )
    assert recorder.events == [("recv_many", ((2, torch.bfloat16),))]
    expected = root_routes.clone()
    expected[0, 2] = sends[2][0]
    torch.testing.assert_close(output, expected.float().sum(dim=1).to(output.dtype))


def test_grouped_receive_single_peer(monkeypatch, ep4_root):
    monkeypatch.setattr(moe_module, "_EP_PACKED_GROUPED_P2P", True)
    monkeypatch.setattr(
        moe_module, "get_tp_info", lambda: SimpleNamespace(rank=0, size=2)
    )
    payload = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).bfloat16()
    recorder = _Recorder({1: [payload]})
    communicator = _GroupedCommunicator(recorder)
    layer = _layer(peers=(1,), communicator=communicator)
    local_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    layer.prepare_packed_prefill_receive(local_weights, torch.bfloat16)
    routes = torch.zeros(2, 2, HIDDEN, dtype=torch.bfloat16)
    routes[0, 0] = 10
    routes[1, 1] = 20
    output = layer._maybe_combine_packed_prefill_routes(routes, local_weights)
    assert recorder.events == [("recv_many", ((1, torch.bfloat16),))]
    torch.testing.assert_close(
        output,
        torch.stack([payload[0] + 10, payload[1] + 20]).to(output.dtype),
    )


def test_torch_recv_many_single_batch_and_empty_noop(monkeypatch):
    calls = []
    works = []
    irecv = object()
    monkeypatch.setattr(dist, "irecv", irecv)
    monkeypatch.setattr(dist, "P2POp", lambda op, tensor, peer: (op, tensor, peer))

    def batch(ops):
        calls.append(ops)
        launched = [_Work() for _ in ops]
        works.extend(launched)
        return launched

    monkeypatch.setattr(dist, "batch_isend_irecv", batch)
    impl = TorchDistributedImpl()
    a, b, c = torch.zeros(1), torch.zeros(2), torch.zeros(3)
    impl.recv_many([(a, 1), (b, 1), (c, 3)])
    impl.recv_many([])
    assert calls == [[(irecv, a, 1), (irecv, b, 1), (irecv, c, 3)]]
    assert len(works) == 3 and all(work.waited for work in works)


def test_pynccl_recv_many_keeps_ordered_sequential_fallback():
    from freetoken.distributed.impl import PyNCCLDistributedImpl

    received = []
    comm = SimpleNamespace(recv=lambda tensor, peer: received.append((tensor, peer)))
    impl = PyNCCLDistributedImpl(comm)
    a, b, c = torch.zeros(1), torch.zeros(2), torch.zeros(3)
    impl.recv_many([(a, 1), (b, 2), (c, 2)])
    assert [(id(tensor), peer) for tensor, peer in received] == [
        (id(a), 1), (id(b), 2), (id(c), 2)
    ]


def test_distributed_communicator_delegates_recv_many(monkeypatch):
    seen = []
    monkeypatch.setattr(
        DistributedCommunicator,
        "plugins",
        [SimpleNamespace(recv_many=lambda ops: seen.append(list(ops)))],
    )
    tensor = torch.zeros(1)
    DistributedCommunicator().recv_many([(tensor, 2)])
    assert seen == [[(tensor, 2)]]


def _import_moe_with(monkeypatch, value):
    monkeypatch.setenv("FREETOKEN_EP_PACKED_GROUPED_P2P", value)
    spec = importlib.util.spec_from_file_location(
        "freetoken.layers._grouped_p2p_flag_probe", moe_module.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value, enabled", [("0", False), ("1", True)])
def test_grouped_flag_values(monkeypatch, value, enabled):
    assert _import_moe_with(monkeypatch, value)._EP_PACKED_GROUPED_P2P is enabled


def test_grouped_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("FREETOKEN_EP_PACKED_GROUPED_P2P", raising=False)
    spec = importlib.util.spec_from_file_location(
        "freetoken.layers._grouped_p2p_flag_probe", moe_module.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._EP_PACKED_GROUPED_P2P is False


@pytest.mark.parametrize("value", ["", "true", "on", "2", " 1", "yes"])
def test_grouped_flag_rejects_invalid_values(monkeypatch, value):
    with pytest.raises(ValueError, match="FREETOKEN_EP_PACKED_GROUPED_P2P"):
        _import_moe_with(monkeypatch, value)
