"""C2 r2: the synthetic hybrid decode warmup routes inside the CPU layout interval."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import _hybrid_warmup_local_ids

TOP_K, LOCAL = 6, 176


def _old_ids(top_k, num_experts):
    return torch.arange(top_k, dtype=torch.int32).remainder(int(num_experts))


@pytest.mark.parametrize("top_k,num_experts", [(6, 176), (6, 144), (6, 8), (6, 4), (1, 1), (8, 256)])
def test_flag_off_ids_are_exactly_the_previous_expression(top_k, num_experts):
    ids = _hybrid_warmup_local_ids(top_k, num_experts, None, torch.device("cpu"))
    assert ids.dtype == torch.int32
    assert torch.equal(ids, _old_ids(top_k, num_experts))


@pytest.mark.parametrize("interval,expected", [((0, 144), list(range(0, 6))), ((32, 176), list(range(32, 38)))])
def test_production_rank_intervals_route_wholly_inside_the_layout(interval, expected):
    ids = _hybrid_warmup_local_ids(TOP_K, LOCAL, interval, torch.device("cpu"))
    assert ids.dtype == torch.int32 and ids.tolist() == expected
    assert all(interval[0] <= e < interval[1] for e in ids.tolist())


def test_exactly_sized_interval_at_the_top_does_not_wrap():
    assert _hybrid_warmup_local_ids(TOP_K, LOCAL, (170, 176), torch.device("cpu")).tolist() == list(range(170, 176))


@pytest.mark.parametrize("interval", [
    (171, 176),        # undersized at the top: modulo would escape to 0
    (32, 37),          # undersized
    (0, 0), (40, 32),  # empty / reversed
    (-1, 144),         # below the local range
    (32, 177),         # above the local range
    (0.0, 144.0), (True, 144), (0, "144"),
])
def test_undersized_or_bad_intervals_fail_loudly(interval):
    with pytest.raises(ValueError, match="cannot cover"):
        _hybrid_warmup_local_ids(TOP_K, LOCAL, interval, torch.device("cpu"))


def _run_warmup(monkeypatch, cache_extra):
    from freetoken.engine.engine import Engine
    import freetoken.kernel.triton.dsv41 as kernels

    seen = []
    cache = SimpleNamespace(
        num_experts=LOCAL, is_cpu_layer=lambda _: False, uses_hybrid=lambda rows: True,
        reset=lambda: None, ensure_experts=lambda *_a, **_k: None, copy_missing=lambda: None,
        bank_views=lambda: (), alphas_for_slots=lambda _: None, **cache_extra)
    layer = SimpleNamespace(
        layer_id=0, top_k=TOP_K, skip_empty_cache=False, _expert_gemm=lambda _c, hidden, *_a, **_k: hidden,
        _decode_hybrid=lambda _c, _h, _w, ids: seen.append(ids.clone()))
    engine = SimpleNamespace(
        moe_offload_cache=cache, model=SimpleNamespace(_iter_offload_moe_layers=lambda: iter([layer])),
        dtype=torch.bfloat16, device=torch.device("cpu"), stream=None,
        _execution_plan=SimpleNamespace(enabled=False), tp_cpu_group=object())
    monkeypatch.setattr(kernels, "fused_localize_cache_safe_routes", lambda weights, ids, **_: (weights, ids))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda **_: SimpleNamespace(
        record=lambda *_: None, elapsed_time=lambda _: 0.0))
    Engine._warmup_dspark_moe_kernels(engine, SimpleNamespace(
        speculative_dspark=False, max_running_req=1, cuda_graph_max_bs=1, cuda_graph_bs=[1, 2],
        model_config=SimpleNamespace(dsv41_args=SimpleNamespace(dspark_block_size=5, dim=4))))
    return seen


@pytest.mark.parametrize("cache_extra", [{}, {"cpu_executor": None},
                                         {"cpu_executor": SimpleNamespace(layout_experts=None)}])
def test_engine_warmup_without_layout_keeps_ids_0_to_top_k(monkeypatch, cache_extra):
    seen = _run_warmup(monkeypatch, cache_extra)
    assert [ids.shape for ids in seen] == [(1, TOP_K), (2, TOP_K)]
    assert all(ids.tolist() == [list(range(TOP_K))] * ids.shape[0] for ids in seen)


@pytest.mark.parametrize("interval", [(0, 144), (32, 176)])
def test_engine_warmup_routes_every_row_inside_the_executor_layout(monkeypatch, interval):
    seen = _run_warmup(monkeypatch, {"cpu_executor": SimpleNamespace(layout_experts=interval)})
    assert [ids.shape for ids in seen] == [(1, TOP_K), (2, TOP_K)]
    for ids in seen:
        assert ids.dtype == torch.int32
        assert ids.tolist() == [list(range(interval[0], interval[0] + TOP_K))] * ids.shape[0]


def test_engine_warmup_refuses_an_undersized_layout(monkeypatch):
    with pytest.raises(ValueError, match="cannot cover"):
        _run_warmup(monkeypatch, {"cpu_executor": SimpleNamespace(layout_experts=(172, 176))})
