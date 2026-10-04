"""Empty EP ranks must neither admit dummy experts nor refresh their recency."""
import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.partition import cache_safe_route_ids


def make(device, mode, experts=16, slots=20):
    return OffloadMoeCache(4, experts, slots, torch.device(device), quant_format="ds_fp4",
        decode_target="gpu" if mode == "gpu" else "hybrid", hybrid_max_fetch=1)


def ensure(cache, mode, layer, ids, weights=None):
    kwargs = {"route_weights": weights} if weights is not None else {}
    if mode == "gpu":
        cache.ensure_experts(layer, ids, **kwargs)
    else:
        cache.ensure_experts_hybrid(layer, ids, **kwargs,
            compute_misses_on_cpu=mode == "background", cpu_miss_limit=1)


@pytest.mark.parametrize("mode", ["gpu", "hybrid", "background"])
def test_empty_rank_does_not_admit_or_touch_dummy(mode):
    cache = make("cpu", mode)
    # Include a resident dummy so the warm case checks recency, not just copies.
    cache.slot_for_id[2, 0] = 3
    cache.id_of_slot[3] = 32
    cache.usage[3] = 7
    cache.step.fill_(9)
    cache.expert_recency[2, 0] = 6
    cache.num_indices.fill_(1)
    cache.num_missing_full.fill_(1)
    cache.active_mask.fill_(1)
    frozen = {k: getattr(cache, k).clone() for k in
              ("slot_for_id", "id_of_slot", "usage", "expert_recency")}
    for layer in (0, 2, 1, 2):
        weights = torch.zeros(1, 6)
        ids = cache_safe_route_ids(weights, torch.full((1, 6), 16, dtype=torch.int32))
        ensure(cache, mode, layer, ids, weights)
        assert cache.num_indices.item() == 0
        assert torch.count_nonzero(ids) == 0
        for k, value in frozen.items():
            assert torch.equal(getattr(cache, k), value), k
        if mode != "gpu":
            assert cache.num_missing_full.item() == 0
            assert torch.count_nonzero(cache.active_mask) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["gpu", "hybrid", "background"])
@pytest.mark.parametrize("collect_stats", [False, True])
@pytest.mark.parametrize("experts,slots", [(16, 20), (128, 256), (256, 1472), (192, 2350)])
def test_empty_and_nonempty_changing_graph_cache_state(mode, collect_stats, experts, slots):
    # Cover eviction with a tiny bank and the exact production launch geometries.
    candidate, reference = make("cuda", mode, experts, slots), make("cuda", mode, experts, slots)
    candidate.collect_stats = reference.collect_stats = collect_stats
    ids = torch.zeros(1, 6, dtype=torch.int32, device="cuda")
    weights = torch.ones(1, 6, device="cuda")
    for layer in range(4):
        ensure(candidate, mode, layer, ids, weights)
    torch.cuda.synchronize()
    graphs = []
    for layer in range(4):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ensure(candidate, mode, layer, ids, weights)
        graphs.append(graph)
    candidate.reset()
    reference.reset()
    candidate.reset_stats()
    reference.reset_stats()
    for n in range(96):
        layer = n % 4
        torch.manual_seed(n + 854)
        raw = torch.randint(experts, (1, 6), dtype=torch.int32, device="cuda")
        weights.copy_(torch.tensor([[0.] * 6 if n % 3 == 0 else [0., .2, .3, 0., .1, 0.]], device="cuda"))
        raw = cache_safe_route_ids(weights, raw)
        expected = raw.clone()
        ids.copy_(raw)
        graphs[layer].replay()
        if n % 3:
            ensure(reference, mode, layer, expected)
        else:
            reference.step.add_(1)
            reference.num_indices.zero_()
            expected.zero_()
            if mode == "gpu" and collect_stats:
                reference.lru_stats[layer, 2].add_(1)
            elif mode != "gpu":
                reference.num_missing_full.zero_()
                reference.active_mask.zero_()
        torch.cuda.synchronize()
        assert torch.equal(ids, expected)
        names = ["slot_for_id", "id_of_slot", "usage", "step", "num_indices", "expert_recency"]
        names += ["lru_stats"] if mode == "gpu" else ["num_missing_full", "active_mask"]
        for name in names:
            assert torch.equal(getattr(candidate, name), getattr(reference, name)), (n, name)
        count = int(candidate.num_indices.item())
        for name in ("src_indices", "evict_slots"):
            assert torch.equal(getattr(candidate, name)[:count], getattr(reference, name)[:count])
