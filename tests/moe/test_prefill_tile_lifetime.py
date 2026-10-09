"""EP prefill must not retain a completed tile into the next GEMM allocation."""

from contextlib import contextmanager
import importlib.util
from types import SimpleNamespace
import weakref

import pytest
import torch

import freetoken.layers.moe as moe_module
from freetoken.layers.moe import ExpertParallelOffloadMoELayer


def _worker_layer(communicator=None, gemm=None):
    """EP worker rank 1 of root 0 over one fake, already materialized layer."""
    layer = object.__new__(ExpertParallelOffloadMoELayer)
    layer.packed_prefill_root = 0
    if communicator is not None:
        layer.prefill_communicator = communicator
    layer.layer_id, layer.num_experts = 7, 8
    layer.skip_inactive_prefill_routes = False
    layer._packed_prefill_peers = lambda: (1,)
    layer.offload_cache = SimpleNamespace(prefill_overlap=False,
        materialize_layer=lambda _: None, copy_missing=lambda: None,
        bank_views=lambda _: None, alphas_for_layer=lambda _: None)
    layer._expert_gemm = gemm or (
        lambda cache, x, *args, **kwargs: torch.stack((x, x * 2, x * 3), dim=1)
    )
    return layer


class _Event:
    def __init__(self, log, number):
        self.log, self.number = log, number

    def synchronize(self):
        self.log.append(("synchronize", self.number))


class _Stream:
    def __init__(self, log, name):
        self.log, self.name, self.events = log, name, 0

    def wait_stream(self, other):
        self.log.append(("wait_stream", self.name, other.name))

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event.number))

    def record_event(self):
        self.events += 1
        self.log.append(("record_event", self.name, self.events))
        return _Event(self.log, self.events)


@pytest.fixture
def fake_streams(monkeypatch):
    """CPU stand-ins that log where gathers, quantization and sends run."""
    log, recorded = [], []
    compute, send = _Stream(log, "compute"), _Stream(log, "send")
    current = [compute]

    @contextmanager
    def use(stream):
        log.append(("enter", stream.name))
        current.append(stream)
        try:
            yield
        finally:
            current.pop()
            log.append(("exit", stream.name))

    index_select = torch.Tensor.index_select
    quantize = moe_module._quantize_ep_routes

    def logged_index_select(tensor, dim, index):
        log.append(("gather", current[-1].name))
        return index_select(tensor, dim, index)

    def logged_quantize(packed):
        log.append(("quantize", current[-1].name))
        return quantize(packed)

    def record_stream(tensor, stream):
        recorded.append(tensor)
        log.append(("record_stream", stream.name))

    monkeypatch.setattr(moe_module, "get_tp_info", lambda: SimpleNamespace(rank=1))
    monkeypatch.setattr(moe_module, "_ep_packed_send_stream", lambda _device: send)
    monkeypatch.setattr(moe_module, "_quantize_ep_routes", logged_quantize)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: current[-1])
    monkeypatch.setattr(torch.cuda, "stream", use)
    monkeypatch.setattr(torch.Tensor, "index_select", logged_index_select)
    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    return SimpleNamespace(log=log, recorded=recorded, current=current)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("tokens", [8, 9])
def test_completed_prefill_tiles_are_released_before_next_gemm(monkeypatch, rank, overlap, tokens):
    """Keep the root's exact full output and worker's final tile, not earlier tiles."""
    monkeypatch.setattr(moe_module, "get_tp_info", lambda: SimpleNamespace(rank=rank))
    monkeypatch.setattr(moe_module, "_EP_PACKED_ASYNC_SEND", False)
    calls = []
    route_refs, combined_refs = [], []
    hidden = torch.arange(tokens * 4, dtype=torch.float32).reshape(tokens, 4)
    weights = torch.ones(tokens, 2)
    ids = torch.zeros(tokens, 2, dtype=torch.int32)
    global_ids = torch.arange(tokens * 2, dtype=torch.int32).reshape(tokens, 2)
    views = object()
    cache = SimpleNamespace(
        prefill_overlap=overlap,
        materialize_layer=lambda layer: calls.append(("materialize", layer)),
        copy_missing=lambda: calls.append(("copy",)),
        bank_views=lambda rows: views,
        alphas_for_layer=lambda layer: None,
        release_prefill_layer=lambda layer: calls.append(("release", layer)),
    )
    layer = object.__new__(ExpertParallelOffloadMoELayer)
    layer.offload_cache = cache
    layer.layer_id, layer.num_experts = 7, 8
    layer.packed_prefill_root = 0
    layer.skip_inactive_prefill_routes = False
    layer._packed_prefill_tiled_global_ids = global_ids
    layer._packed_prefill_peers = lambda: (1, 2)
    layer._wait_prefill_overlap = lambda supplied: views

    def prepare(tile_weights, dtype, *, global_topk_ids):
        start = len(route_refs) * 4
        assert torch.equal(global_topk_ids, global_ids[start:start + len(tile_weights)])
        calls.append(("prepare", len(tile_weights)))

    def gemm(supplied, tile_hidden, tile_weights, tile_ids, **kwargs):
        # A weakref is sufficient to catch the Python lifetime that made a new
        # 240-MiB projection overlap the previous tile on the authority.
        assert all(ref() is None for ref in route_refs), "previous route tile is still live"
        assert all(ref() is None for ref in combined_refs), "previous combined tile is still live"
        assert supplied is cache and kwargs["views"] is views and kwargs["is_prefill"]
        result = torch.stack((tile_hidden, tile_hidden * 2), dim=1)
        route_refs.append(weakref.ref(result))
        return result

    def combine(routes, tile_weights):
        if rank:
            return routes
        result = routes.sum(dim=1)
        combined_refs.append(weakref.ref(result))
        return result

    layer.prepare_packed_prefill_receive = prepare
    layer._expert_gemm = gemm
    layer._maybe_combine_packed_prefill_routes = combine
    result = layer._prefill_routed_tiled(hidden, weights, ids, 4)
    assert len(route_refs) == (tokens + 3) // 4
    assert layer._packed_prefill_tiled_global_ids is None
    if rank == 0:
        assert torch.equal(result, hidden * 3)
        assert all(ref() is None for ref in route_refs + combined_refs)
    else:
        final_start = ((tokens - 1) // 4) * 4
        assert torch.equal(result, torch.stack((hidden[final_start:], hidden[final_start:] * 2), dim=1))
        assert all(ref() is None for ref in route_refs[:-1])
        assert route_refs[-1]() is result
    if overlap:
        assert ("release", 7) in calls and ("materialize", 7) not in calls
    else:
        assert calls.count(("materialize", 7)) == calls.count(("copy",)) == 1


@pytest.mark.parametrize("wire", ["bf16", "fp8"])
def test_async_worker_packs_on_compute_and_keeps_one_send_in_flight(
    monkeypatch, fake_streams, wire
):
    """Only sends move to the side stream; payloads match the sync path bitwise."""
    monkeypatch.setattr(moe_module, "_EP_PACKED_WIRE_DTYPE", wire)
    hidden = torch.arange(12 * 8, dtype=torch.float32).reshape(12, 8).mul(0.37).sin().bfloat16()
    # Tile 1 owns no route on this worker: it must neither wait nor send.
    weights = torch.tensor([[1.0, 0.0, 1.0]] * 4 + [[0.0] * 3] * 4 + [[0.0, 1.0, 1.0]] * 4)
    ids = torch.zeros(12, 3, dtype=torch.int32)

    def run(async_send):
        monkeypatch.setattr(moe_module, "_EP_PACKED_ASYNC_SEND", async_send)
        fake_streams.log.clear()
        fake_streams.recorded.clear()
        sent, route_refs = [], []

        class Communicator:
            def send(self, tensor, root):
                assert root == 0
                fake_streams.log.append(("send", fake_streams.current[-1].name))
                sent.append(tensor)

        def gemm(cache, x, *args, **kwargs):
            assert all(ref() is None for ref in route_refs), "previous route tile is still live"
            routes = torch.stack((x, x * 2, x * 3), dim=1)
            route_refs.append(weakref.ref(routes))
            return routes

        result = _worker_layer(Communicator(), gemm)._prefill_routed_tiled(hidden, weights, ids, 4)
        assert torch.equal(result, torch.stack((hidden[8:], hidden[8:] * 2, hidden[8:] * 3), dim=1))
        return sent, list(fake_streams.log)

    expected_sent, sync_log = run(False)
    sent, log = run(True)

    payloads = 2 if wire == "fp8" else 1
    quantize = [("quantize", "compute")] if wire == "fp8" else []
    assert len(sent) == len(expected_sent) == 2 * payloads
    assert all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(sent, expected_sent))
    # Flag off: the unchanged synchronous path never touches a side stream.
    assert sync_log == ([("gather", "compute")] + quantize + [("send", "compute")] * payloads) * 2

    def tile(number):
        return (
            [("gather", "compute")] + quantize + [("wait_stream", "send", "compute")]
            + [("record_stream", "send")] * payloads + [("enter", "send")]
            + [("send", "send")] * payloads + [("exit", "send"), ("record_event", "send", number)]
        )

    # Tile 2 allocates its payload only after the host saw tile 0's send finish.
    assert log == tile(1) + [("synchronize", 1)] + tile(2) + [("wait_event", "compute", 2)]
    # Only sent payloads are protected for the side stream -- never routes.
    assert [id(tensor) for tensor in fake_streams.recorded] == [id(tensor) for tensor in sent]


def test_async_send_failure_leaves_payload_protected(monkeypatch, fake_streams):
    """A send that raises mid-tile cannot let compute recycle a queued buffer."""
    monkeypatch.setattr(moe_module, "_EP_PACKED_WIRE_DTYPE", "fp8")

    class Communicator:
        def send(self, tensor, root):
            fake_streams.log.append(("send", fake_streams.current[-1].name))
            if tensor.dtype == torch.int32:
                raise RuntimeError("injected send failure")

    routes = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8).bfloat16()
    weights = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    layer = _worker_layer(Communicator())
    with pytest.raises(RuntimeError, match="injected send failure"):
        layer._send_packed_prefill_routes_async(routes, weights, None)
    assert fake_streams.log == [
        ("gather", "compute"), ("quantize", "compute"), ("wait_stream", "send", "compute"),
        ("record_stream", "send"), ("record_stream", "send"), ("enter", "send"),
        ("send", "send"), ("send", "send"), ("exit", "send"),
    ]
    assert [stream.name for stream in fake_streams.current] == ["compute"]


def test_async_send_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("FREETOKEN_EP_PACKED_ASYNC_SEND", raising=False)
    spec = importlib.util.spec_from_file_location(
        "freetoken.layers._async_send_flag_probe", moe_module.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._EP_PACKED_ASYNC_SEND is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("wire", ["bf16", "fp8"])
def test_async_worker_tiles_remain_valid_after_python_release(monkeypatch, wire):
    """Exercise real record_stream lifetimes with delayed side-stream sends."""
    from freetoken.distributed import DistributedCommunicator

    monkeypatch.setattr(moe_module, "get_tp_info", lambda: SimpleNamespace(rank=1))
    monkeypatch.setattr(moe_module, "_EP_PACKED_ASYNC_SEND", True)
    monkeypatch.setattr(moe_module, "_EP_PACKED_WIRE_DTYPE", wire)
    monkeypatch.setattr(moe_module, "_EP_PACKED_SEND_STREAM", None)
    done = []

    class Communicator:
        def __init__(self):
            self.sent = []

        def send(self, tensor, root):
            assert root == 0
            # Every earlier tile's send finished before this tile was packed.
            assert all(event.query() for event in done)
            self.sent.append(tensor.clone())

    communicator = Communicator()
    monkeypatch.setattr(DistributedCommunicator, "plugins", [communicator])
    layer = _worker_layer()
    hidden = torch.arange(65 * 128, dtype=torch.float32, device="cuda").reshape(65, 128)
    weights = torch.tensor([1., 0., 1.], device="cuda").expand(65, 3).contiguous()
    ids = torch.zeros(65, 3, dtype=torch.int32, device="cuda")
    send = layer._send_packed_prefill_routes_async

    def delayed_send(routes, tile_weights, in_flight):
        # Force sends to remain pending while the compute stream advances.
        with torch.cuda.stream(moe_module._ep_packed_send_stream(routes.device)):
            torch.cuda._sleep(500_000)
        event = send(routes, tile_weights, in_flight)
        done.append(event)
        return event

    layer._send_packed_prefill_routes_async = delayed_send
    result = layer._prefill_routed_tiled(hidden, weights, ids, 16)
    torch.cuda.synchronize()
    expected = []
    for index in range(5):
        x = hidden[index * 16:(index + 1) * 16]
        packed = torch.stack((x, x * 3), dim=1).reshape(-1, 128)
        expected.extend(moe_module._quantize_ep_routes(packed) if wire == "fp8" else (packed,))
    assert len(communicator.sent) == len(expected) == 5 * (2 if wire == "fp8" else 1)
    for actual, want in zip(communicator.sent, expected):
        assert actual.dtype == want.dtype and torch.equal(actual, want)
    assert torch.equal(result, torch.stack((hidden[-1:], hidden[-1:] * 2, hidden[-1:] * 3), dim=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_worker_memory_does_not_grow_with_lagging_sends(monkeypatch):
    """A lagging side stream must not retain a route tile or payload per tile."""
    from freetoken.distributed import DistributedCommunicator

    monkeypatch.setattr(moe_module, "get_tp_info", lambda: SimpleNamespace(rank=1))
    monkeypatch.setattr(moe_module, "_EP_PACKED_ASYNC_SEND", True)
    monkeypatch.setattr(moe_module, "_EP_PACKED_WIRE_DTYPE", "bf16")
    monkeypatch.setattr(moe_module, "_EP_PACKED_SEND_STREAM", None)

    class Communicator:
        def send(self, tensor, root):
            torch.cuda._sleep(5_000_000)  # each send outlasts the next tile's GEMM

    monkeypatch.setattr(DistributedCommunicator, "plugins", [Communicator()])
    layer = _worker_layer()
    tile, width = 1024, 4096

    def peak_growth(tiles):
        hidden = torch.ones(tiles * tile, width, dtype=torch.bfloat16, device="cuda")
        weights = torch.tensor([1., 0., 1.], device="cuda").expand(tiles * tile, 3).contiguous()
        ids = torch.zeros(tiles * tile, 3, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_reserved()
        layer._prefill_routed_tiled(hidden, weights, ids, tile)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_reserved() - base

    # Retaining each lagging route tile (the first async candidate) would add
    # four 24-MiB route tiles here; the bound keeps the steady state flat.
    route_bytes = tile * 3 * width * 2
    peak_growth(2)  # create the side stream and kernels outside the comparison
    assert peak_growth(6) - peak_growth(2) <= route_bytes
