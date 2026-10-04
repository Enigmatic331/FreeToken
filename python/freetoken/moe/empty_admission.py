"""One-row empty-route guard around the existing cache admission kernels.

An empty rank still participates in EP collectives, but has no expert to admit or
touch. Slot zero is only an output placeholder: callers MUST skip zero-weight
GPU reads. Nonempty calls execute the existing LRU implementation unchanged.
"""
import torch
import triton
import triton.language as tl

from flashlib.kernels.slot_cache.triton.lru_ensure import _lru_ensure_kernel
from freetoken.moe.offload_kernels import _ensure_experts_hybrid_kernel


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _guard_lru(weights, query, slots, owners, usage, step, src, dst, count,
               stats, K, num_cached, id_base,
               BK: tl.constexpr, BC: tl.constexpr, STATS: tl.constexpr):
    k = tl.arange(0, BK)
    active = tl.sum((tl.load(weights + k, k < K, other=0.) != 0.).to(tl.int32))
    if active > 0:
        _lru_ensure_kernel(query, slots, owners, usage, step, query, src, dst,
            count, stats, K, num_cached, id_base, BK, BC, 9223372036854775807, STATS)
    else:
        tl.store(step, tl.load(step) + 1)
        tl.store(count, 0)
        tl.store(query + k, 0, k < K)
        if STATS:
            si = tl.arange(0, 4)
            tl.atomic_add(stats + si, (si == 2).to(tl.int64), si < 3, sem="relaxed")


@triton.jit(do_not_specialize=["layer", "K", "max_fetch", "fraction"])
def _guard_hybrid(weights, ids, slots, owners, usage, step, active_mask,
                  dst, src, count, missing, recency, layer, K, max_fetch,
                  fraction, E: tl.constexpr, C: tl.constexpr,
                  BK: tl.constexpr, BE: tl.constexpr, BC: tl.constexpr,
                  RECENCY: tl.constexpr, CPU: tl.constexpr, LIMIT: tl.constexpr):
    k = tl.arange(0, BK)
    active = tl.sum((tl.load(weights + k, k < K, other=0.) != 0.).to(tl.int32))
    if active > 0:
        _ensure_experts_hybrid_kernel(ids, slots, owners, usage, step,
            active_mask, dst, src, count, missing, recency, layer, K,
            max_fetch, fraction, E, C, BE, BC, RECENCY, CPU, LIMIT)
    else:
        tl.store(step, tl.load(step) + 1)
        tl.store(count, 0)
        tl.store(missing, 0)
        e = tl.arange(0, BE)
        tl.store(active_mask + e, 0, e < E)
        tl.store(ids + k, 0, k < K)


def ensure_nonempty(cache, layer, ids, weights, *, hybrid=False,
                    compute_misses_on_cpu=False, cpu_miss_limit=0, stats=None):
    if ids.ndim != 2 or ids.shape[0] != 1 or weights.shape != ids.shape:
        raise ValueError("empty-route admission requires one row with matching weights")
    if not weights.is_contiguous() or not ids.is_contiguous():
        raise ValueError("empty-route admission requires contiguous routes")
    if ids.device != weights.device:
        raise ValueError("route ids and weights must share a device")
    from freetoken.moe import offload_kernels as kernels

    if not ids.is_cuda:
        if not bool(torch.any(weights != 0)):
            cache.step.add_(1)
            cache.num_indices.zero_()
            ids.zero_()
            if hybrid:
                cache.num_missing_full.zero_()
                cache.active_mask.zero_()
            elif stats is not None:
                stats[2].add_(1)
            return
        if hybrid:
            return kernels.ensure_experts_hybrid(cache, layer, ids,
                cache.hybrid_max_fetch, cache.hybrid_fetch_fraction,
                compute_misses_on_cpu=compute_misses_on_cpu, cpu_miss_limit=cpu_miss_limit)
        return kernels.ensure_experts(cache, layer, ids, stats=stats)

    k, c = ids.numel(), cache.cache_size
    bk, bc = triton.next_power_of_2(k), triton.next_power_of_2(c)
    warps = max(8 if bc >= 2048 else 4, min(triton.next_power_of_2(max(bc // 1024, 1)), 32))
    if hybrid:
        fraction = min(1 << 16, max(0, round(cache.hybrid_fetch_fraction * (1 << 16))))
        _guard_hybrid[(1,)](weights, ids, cache.slot_for_id, cache.id_of_slot,
            cache.usage, cache.step, cache.active_mask, cache.evict_slots,
            cache.src_indices, cache.num_indices, cache.num_missing_full,
            cache.expert_recency, layer, k, cache.hybrid_max_fetch, fraction,
            cache.num_experts, c, bk, triton.next_power_of_2(cache.num_experts),
            bc, kernels._HYBRID_FETCH_BY_RECENCY, compute_misses_on_cpu,
            cpu_miss_limit, num_warps=warps)
    else:
        # The qualified V4.1 caches use FlashLib's sequential strategy. Keep the
        # dependency explicit instead of silently changing strategy for larger banks.
        if c >= 40000:
            raise ValueError("empty-route guard is qualified for cache sizes below 40000")
        _guard_lru[(1,)](weights, ids, cache.slot_for_id, cache.id_of_slot,
            cache.usage, cache.step, cache.src_indices, cache.evict_slots,
            cache.num_indices, stats, k, c, layer * cache.num_experts,
            bk, bc, stats is not None, num_warps=warps)
