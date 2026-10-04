"""Experimental CPU miss computation with deferred GPU cache admission.

Not enabled by any serving default. A caller owns the lifetime of this object,
must join before *any* other cache operation, and must join before returning from
a captured model forward. One in-flight copy uses the cache's existing plan;
there is no extra expert bank, host polling loop, or speculative cache lookup.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _split_miss_copy(full_count, fetch_count, sync_count, async_count,
                     MAX_MISSES: tl.constexpr):
    missing = tl.load(full_count)
    fetched = tl.load(fetch_count)
    selected = (missing > 0) & (missing <= MAX_MISSES)
    tl.store(async_count, tl.where(selected, fetched, 0))
    tl.store(sync_count, tl.where(selected, 0, fetched))


class BackgroundAdmission:
    def __init__(self, cache, *, single_miss_only=False, max_cpu_misses=None):
        if cache.quant_format != "ds_fp4" or cache.decode_target != "hybrid":
            raise ValueError("background admission requires native DS-FP4 hybrid decode")
        self.cache = cache
        self.stream = torch.cuda.Stream(device=cache.device)
        self.pending = False
        if max_cpu_misses is None:
            max_cpu_misses = 1 if single_miss_only else 0
        if not isinstance(max_cpu_misses, int) or isinstance(max_cpu_misses, bool) or max_cpu_misses < 0:
            raise ValueError("background CPU miss limit must be a nonnegative integer")
        self.max_cpu_misses = max_cpu_misses
        self.single_miss_only = max_cpu_misses == 1
        self.stat_cpu_admitted = torch.zeros(cache.num_layers, dtype=torch.int64, device=cache.device)
        if max_cpu_misses:
            # Persistent counts: side-stream copy must not outlive a temporary
            # produced/freed on the main stream. Reuse only after join().
            self.sync_count = torch.empty_like(cache.num_indices)
            self.async_count = torch.empty_like(cache.num_indices)

    def join(self):
        """Order future cache reads/writes after the copy, without a host wait."""
        if self.pending:
            torch.cuda.current_stream(self.cache.device).wait_stream(self.stream)
            self.pending = False

    def reset_stats(self):
        self.stat_cpu_admitted.zero_()

    def decode(self, layer, hidden, weights, ids):
        """GPU computes hits; selected misses compute on CPU while their rows fill.

        The admission policy is unchanged (typically one fill). Zero-weight
        routes must skip their memory reads: their dummy slot may be uninitialized
        or be a copy destination. Preexisting hit slots are protected by LRU.
        """
        cache = self.cache
        executor = cache.cpu_executor
        if (hidden.shape[0] != 1 or executor is None
                or not executor.dsfp4_gpu_order or not executor.return_route_outputs):
            raise ValueError("background admission requires exact-order one-row CPU routes")
        self.join()
        raw = ids.clone()
        kwargs = {"cpu_miss_limit": self.max_cpu_misses} if self.max_cpu_misses else {}
        if getattr(layer, "skip_empty_cache", False):
            kwargs["route_weights"] = weights
        cache.ensure_experts_hybrid(layer.layer_id, ids, compute_misses_on_cpu=True, **kwargs)
        if getattr(cache, "collect_stats", False):
            cache.record_decode_stats_hybrid(layer.layer_id)
        on_gpu = ids >= 0
        cpu_ids = torch.where(on_gpu | (weights == 0), raw.new_full((), -1), raw).contiguous()
        pending = executor.decode_submit(layer.layer_id, hidden, weights, cpu_ids)
        if self.max_cpu_misses:
            _split_miss_copy[(1,)](cache.num_missing_full, cache.num_indices,
                self.sync_count, self.async_count, MAX_MISSES=self.max_cpu_misses, num_warps=1)
            # Beyond the selected limit, retain the qualified one-fetch plus
            # CPU-overflow policy. Admission and eviction decisions never change.
            cache.copy_missing(num_indices=self.sync_count)
        if getattr(cache, "collect_stats", False):
            admitted = self.async_count if self.max_cpu_misses else cache.num_indices
            self.stat_cpu_admitted[layer.layer_id] += admitted.sum()
        self.stream.wait_stream(torch.cuda.current_stream(cache.device))
        with torch.cuda.stream(self.stream):
            if self.max_cpu_misses:
                cache.copy_missing(num_indices=self.async_count)
            else:
                cache.copy_missing()
        self.pending = True
        try:
            from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

            gpu_weights = torch.where(on_gpu, weights, weights.new_zeros(())).contiguous()
            gpu_routes = routed_experts_fp4(
                hidden, ids.clamp_min(0), gpu_weights, *cache.bank_views(),
                layer.swiglu_limit, return_route_outputs=True,
                activation_block_size=layer.dsfp4_activation_block_size,
                skip_zero_weight=True,
            )
            cpu_routes = executor.decode_sync(pending)
            merged = gpu_routes + cpu_routes
            if not getattr(layer, "return_route_outputs", False):
                return merged.sum(dim=1)
            return merged
        except BaseException:
            self.join()
            raise
