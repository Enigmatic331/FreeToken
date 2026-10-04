"""EP prefill must not retain a completed tile into the next GEMM allocation."""

from types import SimpleNamespace
import weakref

import pytest
import torch

import freetoken.layers.moe as moe_module
from freetoken.layers.moe import ExpertParallelOffloadMoELayer


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_worker_tiles_remain_valid_after_python_release(monkeypatch):
    """Exercise real record_stream lifetimes with delayed side-stream gathers."""
    from freetoken.distributed import DistributedCommunicator

    monkeypatch.setattr(moe_module, "get_tp_info", lambda: SimpleNamespace(rank=1))
    monkeypatch.setattr(moe_module, "_EP_PACKED_ASYNC_SEND", True)
    monkeypatch.setattr(moe_module, "_EP_PACKED_WIRE_DTYPE", "bf16")
    monkeypatch.setattr(moe_module, "_EP_PACKED_SEND_STREAM", None)

    class Communicator:
        def __init__(self):
            self.sent = []

        def send(self, tensor, root):
            assert root == 0
            self.sent.append(tensor.clone())

    communicator = Communicator()
    monkeypatch.setattr(DistributedCommunicator, "plugins", [communicator])
    layer = object.__new__(ExpertParallelOffloadMoELayer)
    layer.packed_prefill_root = 0
    layer.layer_id, layer.num_experts = 7, 8
    layer.skip_inactive_prefill_routes = False
    layer._packed_prefill_peers = lambda: (1,)
    layer.offload_cache = SimpleNamespace(prefill_overlap=False,
        materialize_layer=lambda _: None, copy_missing=lambda: None,
        bank_views=lambda _: None, alphas_for_layer=lambda _: None)
    hidden = torch.arange(65 * 128, dtype=torch.float32, device="cuda").reshape(65, 128)
    weights = torch.tensor([1., 0., 1.], device="cuda").expand(65, 3).contiguous()
    ids = torch.zeros(65, 3, dtype=torch.int32, device="cuda")
    layer._expert_gemm = lambda cache, x, *args, **kwargs: torch.stack((x, x * 2, x * 3), dim=1)
    send = layer._send_packed_prefill_routes_async

    def delayed_send(routes, tile_weights):
        # Force gathers to remain pending while the compute stream advances.
        with torch.cuda.stream(moe_module._ep_packed_send_stream(routes.device)):
            torch.cuda._sleep(500_000)
        return send(routes, tile_weights)

    layer._send_packed_prefill_routes_async = delayed_send
    result = layer._prefill_routed_tiled(hidden, weights, ids, 16)
    torch.cuda.synchronize()
    assert len(communicator.sent) == 5
    for index, actual in enumerate(communicator.sent):
        x = hidden[index * 16:(index + 1) * 16]
        expected = torch.stack((x, x * 3), dim=1).reshape(-1, 128)
        assert torch.equal(actual, expected)
    assert torch.equal(result, torch.stack((hidden[-1:], hidden[-1:] * 2, hidden[-1:] * 3), dim=1))
