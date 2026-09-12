from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.dsv41_cost_model import (
    dsv41_kv_unit_bytes,
    dsv41_pool_bytes,
    dsv41_pool_sizes,
    dsv41_solve_num_pages,
    dsv41_window_unit_bytes,
)
from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
from freetoken.models.deepseek_v41.args import load_args


MODEL_PATH = "/home/enigmatic331/models/DeepSeek-V4.1-Flash"


def _engine_config(args, *, num_page_override=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(dsv41_args=args),
        page_size=args.window_size,
        max_running_req=2,
        max_seq_len=4096,
        cache_type="swa_radix",
        swa_full_tokens_ratio=0.5,
        swa_num_pages_override=None,
        num_page_override=num_page_override,
        memory_ratio=1.0,
    )


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="official checkpoint absent")
def test_source_only_pool_matches_exact_cost_on_cpu():
    args = load_args(MODEL_PATH)
    sizes = dsv41_pool_sizes(4, args, 0.5, n_win_pages=2)
    pool = DSV41PagedKVCache(
        sizes, args, torch.device("cpu"), P=128, n_scratch=2
    )
    assert pool.total_bytes() == dsv41_pool_bytes(sizes, args, n_scratch=2)
    assert [i for i, value in enumerate(pool.cmp_pool) if value is not None] == [2, 8, 14, 20]
    assert [i for i, value in enumerate(pool.idx_pool) if value is not None] == [2, 8, 14, 20]
    assert [i for i, value in enumerate(pool.state_ring) if value is not None] == [2, 8, 14]
    assert pool.source_layer_of(7) == 2
    assert pool.source_layer_of(13) == 8
    assert pool.source_layer_of(19) == 14
    assert pool.source_layer_of(39) == 20
    assert pool.compressed_pool_for(7) is pool.cmp_pool[2]
    assert pool.index_pool_for(27) is pool.idx_pool[20]


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="official checkpoint absent")
def test_solver_is_maximal_and_unit_costs_cover_only_physical_sources():
    args = load_args(MODEL_PATH)
    budget = 128 << 20
    sizes = dsv41_solve_num_pages(
        budget, args, 0.25, floor_win_pages=2, n_scratch=3
    )
    assert dsv41_pool_bytes(sizes, args, 3) <= budget
    next_full = sizes.full_token + 128
    next_window = max(2, (round(0.25 * next_full) + 127) // 128)
    next_sizes = dsv41_pool_sizes(
        sizes.full_token // 128 + 1,
        args,
        0.25,
        n_win_pages=next_window,
    )
    assert dsv41_pool_bytes(next_sizes, args, 3) > budget
    assert dsv41_kv_unit_bytes(args) == 3_208
    assert dsv41_window_unit_bytes(args) == 41_152


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="official checkpoint absent")
def test_engine_sizing_and_rebuild_surface_use_v41_costs():
    from freetoken.kvcache.dsv4_cost_model import _dsv4_window_floor_pages
    from freetoken.kvcache.dsv41_cost_model import _dsv41_pool_sizes

    args = load_args(MODEL_PATH)
    config = _engine_config(args)
    floor = _dsv4_window_floor_pages(config, args.window_size)
    per_page, fixed, P, reserve = DSV41PagedKVCache.kv_cost(config)
    assert P == args.window_size
    assert per_page > 0 and fixed >= 0 and reserve >= floor * P

    with pytest.raises(ValueError, match="working-set floor"):
        DSV41PagedKVCache.solve_num_pages(
            _engine_config(args, num_page_override=floor - 1), 0
        )

    sizes = _dsv41_pool_sizes(config, floor + 6)
    budget = dsv41_pool_bytes(sizes, args, config.max_running_req + 1)
    pages = DSV41PagedKVCache.solve_num_pages(config, budget)
    assert pages > 1
    assert dsv41_pool_bytes(
        _dsv41_pool_sizes(config, pages + 1),
        args,
        config.max_running_req + 1,
    ) <= budget


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="official checkpoint absent")
def test_attention_backend_routes_consumers_to_source_owned_slabs():
    from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    args = load_args(MODEL_PATH)
    sizes = dsv41_pool_sizes(4, args, 0.5, n_win_pages=2)
    pool = DSV41PagedKVCache(sizes, args, torch.device("cpu"), P=128, n_scratch=2)
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=128)
        set_global_ctx(ctx)
    ctx.kv_cache = pool
    backend = DSV41SparseAttnBackend(SimpleNamespace(dsv41_args=args))

    assert backend.compress_pool(7, "attn") is pool.cmp_pool[2]
    assert backend.compress_pool(13, "idx") is pool.idx_pool[8]
    assert backend.compress_pool(19, "attn") is pool.cmp_pool[14]
    assert backend.compress_pool(39, "idx") is pool.idx_pool[20]
    assert backend.compress_state_ring(7, "attn") is pool.state_ring[2]
