"""DeepSeek-V4.1 specialization of the DSV4 paged sparse backend."""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

from .dsv4_sparse import DSV4SparseAttnBackend


class DSV41SparseAttnBackend(DSV4SparseAttnBackend):
    """Route each consumer layer to its latest source-owned CSA2 slabs.

    The page-table snapshot and 128-token window currency are identical to
    DSV4.  V4.1 differs in physical ownership: compressed KV and index keys
    exist only at layers 2/8/14/20 and are shared by the following consumers.
    """

    def __init__(self, config):
        self.config = config
        self.device = get_global_ctx().kv_cache.device
        self.window_size = config.dsv41_args.window_size
        self.capture = None
        self.capture_bs = []
        self.max_graph_bs = 0
        self._window_ar = torch.arange(self.window_size, device=self.device)

    def compress_pool(self, layer_id: int, tier: str) -> torch.Tensor:
        if tier == "attn":
            return self.pool.compressed_pool_for(layer_id)
        if tier == "idx":
            return self.pool.index_pool_for(layer_id)
        raise ValueError(f"unknown V4.1 compressed tier: {tier}")

    def compress_state_ring(self, layer_id: int, tier: str):
        if tier != "attn":
            raise ValueError("V4.1 has no separate indexer compressor ring")
        source = self.pool.source_layer_of(layer_id)
        return self.pool.state_ring[source]

    def compress_scratch_base(self, layer_id: int, tier: str) -> int:
        source = self.pool.source_layer_of(layer_id)
        if tier == "attn":
            return self.pool.cmp_scratch_base[source]
        if tier == "idx":
            return self.pool.idx_scratch_base[source]
        raise ValueError(f"unknown V4.1 compressed tier: {tier}")

    def attend(
        self,
        q: torch.Tensor,
        layer_id: int,
        topk_idxs: torch.Tensor,
        n_window: int,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        cmp_counts: torch.Tensor | None = None,
        has_compression: bool = True,
    ) -> torch.Tensor:
        pool = self.pool
        cmp = (
            pool.compressed_pool_for(layer_id)
            if has_compression
            else pool.window_pool[layer_id]
        )
        return sparse_attn_paged(
            q,
            pool.window_pool[layer_id],
            cmp,
            attn_sink,
            topk_idxs.int(),
            n_window,
            softmax_scale,
            cmp_counts=cmp_counts,
        )


__all__ = ["DSV41SparseAttnBackend"]
