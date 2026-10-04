import pytest
from types import SimpleNamespace

from freetoken.moe import cpu_executor as cpu


def test_disjoint_rank_pools_reserve_coordinators(monkeypatch):
    monkeypatch.setattr(cpu, "physical_core_cpus", lambda: list(range(56)))
    first, coord1 = cpu.partition_rank_affinity(1, (1, 2), 24)
    second, coord2 = cpu.partition_rank_affinity(2, (1, 2), 24)
    assert first == list(range(24)) and coord1 == 27
    assert second == list(range(28, 52)) and coord2 == 55
    assert not (set(first + [coord1]) & set(second + [coord2]))


def test_auto_divides_uneven_physical_pool(monkeypatch):
    monkeypatch.setattr(cpu, "physical_core_cpus", lambda: list(range(11)))
    pools = [cpu.partition_rank_affinity(rank, (0, 1, 2), 0) for rank in range(3)]
    assigned = [core for workers, coord in pools for core in workers + [coord]]
    assert sorted(assigned) == list(range(11))
    assert len(set(assigned)) == len(assigned)


@pytest.mark.parametrize("ranks,threads", [((1, 1), 0), ((2,), 0), ((1, 2), 28)])
def test_invalid_rank_allocation_rejected(monkeypatch, ranks, threads):
    monkeypatch.setattr(cpu, "physical_core_cpus", lambda: list(range(56)))
    with pytest.raises(ValueError):
        cpu.partition_rank_affinity(1, ranks, threads)


def test_hybrid_batch_ceiling():
    from freetoken.moe.offload_cache import OffloadMoeCache
    cache = object.__new__(OffloadMoeCache)
    cache.decode_target, cache.hybrid_max_bs = "hybrid", 1
    assert cache.uses_hybrid(1)
    assert not cache.uses_hybrid(5)
    cache.hybrid_max_bs = 0
    assert cache.uses_hybrid(5)
    cache.decode_target = "gpu"
    assert not cache.uses_hybrid(1)


@pytest.mark.parametrize("fmt", ["ds_fp4", "bf16"])
def test_gpu_order_rejects_stale_or_wrong_format_extension(monkeypatch, fmt):
    import torch
    import freetoken.kernel

    monkeypatch.setattr(freetoken.kernel, "_cpu_moe",
                        SimpleNamespace(CpuMoeExecutor=object), raising=False)
    with pytest.raises(RuntimeError, match="GPU-order.*rebuilt"):
        cpu.CpuMoeExecutor(
            SimpleNamespace(quant_format=fmt), top_k=6, activation="silu",
            apply_router_weight_on_input=False, num_threads=2, max_tokens=1,
            device=torch.device("cpu"), dsfp4_gpu_order=True,
        )
