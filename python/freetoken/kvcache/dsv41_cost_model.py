"""Exact source-only paged-cache sizing for DeepSeek-V4.1 CSA2."""

from __future__ import annotations

import os

from .dsv4_cost_model import DSV4PoolSizes, dsv4_reserved_window_pages

_BF16 = 2
_FP32 = 4
_INT64 = 8
_PAIR_RING = 2
_AUTO_SLACK = 2 << 30


def _sources(args) -> tuple[tuple[int, int], ...]:
    result = []
    for layer in args.kv_source_layers:
        ratio = int(args.compress_ratios[layer])
        if ratio not in (1, 2):
            raise ValueError(f"V4.1 KV source {layer} has unsupported ratio {ratio}")
        result.append((int(layer), ratio))
    return tuple(result)


def dsv41_pool_sizes(
    num_pages: int,
    args,
    swa_ratio: float,
    P: int = 128,
    n_win_pages: int | None = None,
) -> DSV4PoolSizes:
    full_token = num_pages * P
    if n_win_pages is None:
        n_win_pages = (round(swa_ratio * full_token) + P - 1) // P
    n_win_pages = min(n_win_pages, num_pages)
    n_layers = int(args.n_layers)
    cmp_blocks: list[int | None] = [None] * n_layers
    idx_blocks: list[int | None] = [None] * n_layers
    state_slots: list[int | None] = [None] * n_layers
    ring_sizes: list[int | None] = [None] * n_layers
    idx_state_slots: list[int | None] = [None] * n_layers
    for layer, ratio in _sources(args):
        if P % ratio:
            raise ValueError(f"page size {P} is not divisible by ratio {ratio}")
        cmp_blocks[layer] = full_token // ratio
        idx_blocks[layer] = full_token // ratio
        if ratio == 2:
            state_slots[layer] = n_win_pages * _PAIR_RING
            ring_sizes[layer] = _PAIR_RING
    return DSV4PoolSizes(
        P=P,
        swa_ratio=swa_ratio,
        full_token=full_token,
        n_win_slots=n_win_pages * P,
        n_win_pages=n_win_pages,
        cmp_blocks=cmp_blocks,
        idx_blocks=idx_blocks,
        state_slots=state_slots,
        ring_sizes=ring_sizes,
        idx_state_slots=idx_state_slots,
    )


def dsv41_pool_bytes(sizes: DSV4PoolSizes, args, n_scratch: int = 1) -> int:
    kv_bytes = int(args.head_dim) * _BF16
    index_bytes = int(args.index_head_dim) * _BF16
    total = int(args.n_layers) * sizes.n_win_slots * kv_bytes
    total += (sizes.full_token + 1) * _INT64
    for layer, ratio in _sources(args):
        total += (sizes.cmp_blocks[layer] + n_scratch) * kv_bytes
        total += (sizes.idx_blocks[layer] + n_scratch) * index_bytes
        if ratio == 2:
            # One ring row is kv|score, each head_dim-wide FP32.
            total += (sizes.state_slots[layer] + 1) * 2 * int(args.head_dim) * _FP32
    return int(total)


def dsv41_cache_per_page(args, swa_ratio: float, P: int = 128) -> int:
    kv_bytes = int(args.head_dim) * _BF16
    index_bytes = int(args.index_head_dim) * _BF16
    total = int(args.n_layers) * round(swa_ratio * P) * kv_bytes
    for _layer, ratio in _sources(args):
        total += (P // ratio) * (kv_bytes + index_bytes)
        if ratio == 2:
            total += round(swa_ratio * _PAIR_RING) * 2 * int(args.head_dim) * _FP32
    return int(total)


def dsv41_kv_unit_bytes(args, P: int = 128) -> int:
    per_page = P * _INT64
    for _layer, ratio in _sources(args):
        per_page += (P // ratio) * (
            int(args.head_dim) * _BF16 + int(args.index_head_dim) * _BF16
        )
    return -(-per_page // P)


def dsv41_window_unit_bytes(args, P: int = 128) -> int:
    per_page = int(args.n_layers) * P * int(args.head_dim) * _BF16
    for _layer, ratio in _sources(args):
        if ratio == 2:
            per_page += _PAIR_RING * 2 * int(args.head_dim) * _FP32
    return -(-per_page // P)


def dsv41_solve_num_pages(
    available_bytes: int,
    args,
    swa_ratio: float,
    floor_win_pages: int,
    P: int = 128,
    n_scratch: int = 1,
) -> DSV4PoolSizes:
    def sized(pages: int) -> DSV4PoolSizes:
        window = max(floor_win_pages, (round(swa_ratio * pages * P) + P - 1) // P)
        return dsv41_pool_sizes(pages, args, swa_ratio, P=P, n_win_pages=window)

    lo = max(floor_win_pages, 2)
    if dsv41_pool_bytes(sized(lo), args, n_scratch) > available_bytes:
        raise ValueError("V4.1 KV budget cannot fit the minimum window working set")
    hi = max(lo + 1, available_bytes // max(1, dsv41_cache_per_page(args, 0, P)))
    while dsv41_pool_bytes(sized(hi), args, n_scratch) <= available_bytes:
        hi *= 2
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if dsv41_pool_bytes(sized(mid), args, n_scratch) <= available_bytes:
            lo = mid
        else:
            hi = mid
    return sized(lo)


def dsv41_auto_cost_model(args, swa_ratio, floor_win_pages, P=128, n_scratch=1):
    per_page = dsv41_cache_per_page(args, swa_ratio, P) + P * _INT64
    pages = max(floor_win_pages, 2)
    window = max(floor_win_pages, (round(swa_ratio * pages * P) + P - 1) // P)
    base = dsv41_pool_bytes(
        dsv41_pool_sizes(pages, args, swa_ratio, P=P, n_win_pages=window),
        args,
        n_scratch,
    )
    slack_pages = -(-_AUTO_SLACK // per_page)
    return per_page, max(0, base - pages * per_page), (pages + slack_pages) * P


def dsv41_reserved_window_pages(max_running_req: int, radix: bool) -> int:
    return dsv4_reserved_window_pages(max_running_req, radix)


def _dsv41_pool_sizes(config, num_pages: int, num_swa_pages: int | None = None):
    """Resolve engine configuration into the exact V4.1 source-only geometry.

    ``num_pages`` is physical and includes the generic dummy page.  An explicit
    rebuild target wins over the persistent override, matching the DSV4 pool
    contract used by the generic cache manager.
    """

    from .dsv4_cost_model import _dsv4_swa_ratio, _dsv4_window_floor_pages

    args = config.model_config.dsv41_args
    P = int(args.window_size)
    ratio = _dsv4_swa_ratio(config)
    override = os.environ.get("DSV41_FORCE_SMALL_POOL")
    if override:
        num_pages = max(2, int(override))
        return dsv41_pool_sizes(num_pages, args, ratio, P=P)

    floor = _dsv4_window_floor_pages(config, P)
    target = (
        num_swa_pages
        if num_swa_pages is not None
        else config.swa_num_pages_override
    )
    if target is not None:
        window = min(num_pages, max(floor, int(target) + 1))
    else:
        window = max(floor, (round(ratio * num_pages * P) + P - 1) // P)
    return dsv41_pool_sizes(
        num_pages, args, ratio, P=P, n_win_pages=window
    )


__all__ = [
    "dsv41_auto_cost_model",
    "dsv41_cache_per_page",
    "dsv41_kv_unit_bytes",
    "dsv41_pool_bytes",
    "dsv41_pool_sizes",
    "dsv41_reserved_window_pages",
    "dsv41_solve_num_pages",
    "dsv41_window_unit_bytes",
    "_dsv41_pool_sizes",
]
