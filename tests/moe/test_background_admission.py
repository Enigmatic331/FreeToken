"""Admission must retain normal LRU decisions while computing misses on CPU."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fetch", [0, 1, 3])
@pytest.mark.parametrize("fraction", [0., .415])
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 6])
def test_miss_compute_is_independent_of_cache_admission(device, fetch, fraction, limit):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(3902)
    def make():
        return OffloadMoeCache(4, 16, 20, torch.device(device),
            quant_format="ds_fp4", decode_target="hybrid",
            hybrid_max_fetch=fetch, hybrid_fetch_fraction=fraction)
    control, candidate = make(), make()
    for step in range(96):
        layer = step % 4
        # Duplicate ids and alternating layers force actual cache eviction.
        raw = torch.randint(16, (1, 6), dtype=torch.int32, device=device)
        prior = candidate.slot_for_id[layer, raw.long()].clone()
        a, b = raw.clone(), raw.clone()
        control.ensure_experts_hybrid(layer, a)
        candidate.ensure_experts_hybrid(layer, b, compute_misses_on_cpu=True, cpu_miss_limit=limit)
        expected = prior if limit == 0 or int(candidate.num_missing_full.item()) <= limit else a
        assert torch.equal(b, expected)
        for name in ("slot_for_id", "id_of_slot", "usage", "step",
                     "expert_recency", "num_indices", "num_missing_full"):
            assert torch.equal(getattr(control, name), getattr(candidate, name)), name
        count = int(candidate.num_indices.item())
        for name in ("evict_slots", "src_indices"):
            assert torch.equal(getattr(control, name)[:count], getattr(candidate, name)[:count])


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 6])
def test_gpu_hybrid_cpu_reference_agrees_for_background_routes(limit):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(43903)
    gpu = OffloadMoeCache(4, 16, 20, torch.device("cuda"), decode_target="hybrid")
    cpu = OffloadMoeCache(4, 16, 20, torch.device("cpu"), decode_target="hybrid")
    for step in range(128):
        ids = torch.randint(16, (1, 6), dtype=torch.int32)
        g, c = ids.cuda(), ids.clone()
        gpu.ensure_experts_hybrid(step % 4, g, compute_misses_on_cpu=True, cpu_miss_limit=limit)
        cpu.ensure_experts_hybrid(step % 4, c, compute_misses_on_cpu=True, cpu_miss_limit=limit)
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), cpu.slot_for_id)


@pytest.mark.parametrize("limit", [1, 2, 3, 6])
def test_copy_split_matches_admission_limit_under_changing_graph_replay(limit):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from freetoken.moe.background_admission import _split_miss_copy
    missing, fetched = [torch.zeros(1, dtype=torch.int32, device="cuda") for _ in range(2)]
    sync, background = [torch.empty_like(missing) for _ in range(2)]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _split_miss_copy[(1,)](missing, fetched, sync, background, MAX_MISSES=limit, num_warps=1)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _split_miss_copy[(1,)](missing, fetched, sync, background, MAX_MISSES=limit, num_warps=1)
    torch.cuda.current_stream().wait_stream(stream)
    for count in [0, 1, 2, 3, 6, 7, 2, 0, 3, 1]:
        for amount in range(min(count, 3) + 1):
            missing.fill_(count)
            fetched.fill_(amount)
            graph.replay()
            a, b = int(sync.item()), int(background.item())
            assert a + b == amount
            assert (a, b) == ((0, amount) if 0 < count <= limit else (amount, 0))


@pytest.mark.parametrize("fails", [False, True])
def test_background_stream_lifetime_and_empty_routes(monkeypatch, fails):
    from freetoken.moe.background_admission import BackgroundAdmission
    from freetoken.moe import fused_ds_fp4
    events = []
    class Stream:
        def __init__(self, name): self.name = name
        def wait_stream(self, other): events.append((self.name, "wait", other.name))
    main, copy = Stream("main"), Stream("copy")
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: copy)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: main)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    def ensure(layer, ids, **kwargs):
        assert kwargs == {"compute_misses_on_cpu": True}
        events.append("ensure")
        ids.copy_(torch.tensor([[3, -1, -1]], dtype=torch.int32))
    def submit(layer, hidden, weights, ids):
        events.append("submit")
        assert ids.tolist() == [[-1, 7, -1]]
        return "pending"
    def sync(pending):
        events.append("sync")
        return torch.tensor([[[0., 0.], [3., 4.], [0., 0.]]])
    def gemm(*args, **kwargs):
        events.append("gemm")
        assert kwargs["skip_zero_weight"] is True
        assert args[2].tolist() == [[1., 0., 0.]]
        if fails:
            raise RuntimeError("injected")
        return torch.tensor([[[1., 2.], [0., 0.], [0., 0.]]])
    monkeypatch.setattr(fused_ds_fp4, "routed_experts_fp4", gemm)
    executor = SimpleNamespace(dsfp4_gpu_order=True, return_route_outputs=True,
        decode_submit=submit, decode_sync=sync)
    cache = SimpleNamespace(device=torch.device("cpu"), quant_format="ds_fp4",
        decode_target="hybrid", cpu_executor=executor, num_layers=1,
        ensure_experts_hybrid=ensure, copy_missing=lambda: events.append("copy"),
        bank_views=lambda: ())
    state = BackgroundAdmission(cache)
    layer = SimpleNamespace(layer_id=0, return_route_outputs=True,
        swiglu_limit=10., dsfp4_activation_block_size=32)
    args = (layer, torch.zeros(1, 2), torch.tensor([[1., .5, 0.]]),
            torch.tensor([[2, 7, 8]], dtype=torch.int32))
    if fails:
        with pytest.raises(RuntimeError, match="injected"):
            state.decode(*args)
        assert not state.pending
        assert events[-1] == ("main", "wait", "copy")
    else:
        actual = state.decode(*args)
        assert actual.tolist() == [[[1., 2.], [3., 4.], [0., 0.]]]
        assert state.pending
        assert events == ["ensure", "submit", ("copy", "wait", "main"), "copy", "gemm", "sync"]
        state.join()
        assert events[-1] == ("main", "wait", "copy")
        assert not state.pending


@pytest.mark.parametrize("operation", ["ensure", "hybrid", "sparse", "materialize", "reset", "prefill"])
def test_cache_operation_drains_before_reusing_plan(monkeypatch, operation):
    from freetoken.moe import offload_kernels
    cache = OffloadMoeCache(2, 4, 8, torch.device("cpu"), decode_target="hybrid")
    events = []
    cache._dsv41_background_admission = SimpleNamespace(join=lambda: events.append("join"))
    names = {"ensure": "ensure_experts", "hybrid": "ensure_experts_hybrid",
             "sparse": "ensure_experts_sparse_prefill", "materialize": "materialize_layer",
             "reset": "reset_cache"}
    if operation in names:
        monkeypatch.setattr(offload_kernels, names[operation], lambda *a, **k: events.append("mutate"))
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    if operation == "ensure": cache.ensure_experts(0, ids)
    elif operation == "hybrid": cache.ensure_experts_hybrid(0, ids)
    elif operation == "sparse": cache.ensure_sparse_prefill_experts(0, ids)
    elif operation == "materialize": cache.materialize_layer(0)
    elif operation == "reset": cache.reset()
    else: cache.begin_prefill()
    assert events == (["join"] if operation == "prefill" else ["join", "mutate"])


def test_background_stats_count_cpu_computation_and_admission_independently():
    cache = OffloadMoeCache(2, 4, 8, torch.device("cpu"), decode_target="hybrid")
    cache.stat_missing.copy_(torch.tensor(5))
    cache.stat_fetched.copy_(torch.tensor(3))
    cache.stat_active.copy_(torch.tensor(8))
    cache.stat_calls.copy_(torch.tensor(2))
    cache.stat_missing_layer.copy_(torch.tensor([3, 2]))
    cache.stat_fetched_layer.copy_(torch.tensor([2, 1]))
    cache.stat_active_layer.copy_(torch.tensor([4, 4]))
    cache.stat_steps_layer.fill_(1)
    admitted = torch.tensor([1, 0])
    cache._dsv41_background_admission = SimpleNamespace(
        join=lambda: None, stat_cpu_admitted=admitted, reset_stats=lambda: admitted.zero_())
    assert cache.decode_miss_stats()["cpu_per_layer"] == 1.5
    assert cache.decode_miss_stats()["fetched_per_layer"] == 1.5
    assert [r["cpu_per_step"] for r in cache.decode_miss_stats_per_layer()["per_layer"]] == [2., 1.]
    cache.reset_stats()
    assert cache.decode_miss_stats()["cpu_per_layer"] == 0.
    assert admitted.tolist() == [0, 0]
