"""Target-only hybrid startup loads local kernels before collective traffic."""
from types import SimpleNamespace
import inspect

import pytest
import torch


@pytest.mark.parametrize("hybrid,graph_sizes,max_rows", [
    (False, None, 1), (True, None, 1), (True, [1, 2, 4], 4),
])
@pytest.mark.parametrize("skip_empty_cache", [False, True])
def test_target_only_local_warmup_and_all_rank_barrier(
    monkeypatch, hybrid, graph_sizes, max_rows, skip_empty_cache
):
    from freetoken.engine.engine import Engine
    import freetoken.kernel.triton.dsv41 as kernels

    events, gpu_rows, cpu_rows, admissions = [], [], [], []

    def admit(_layer, ids, **kwargs):
        admissions.append((ids.shape[0], kwargs))
        events.append("admit")

    cache = SimpleNamespace(
        num_experts=8, is_cpu_layer=lambda _: False, uses_hybrid=lambda rows: hybrid,
        reset=lambda: events.append("reset"), ensure_experts=admit,
        copy_missing=lambda: None, bank_views=lambda: (), alphas_for_slots=lambda _: None)

    def gemm(_cache, hidden, *_args, **_kwargs):
        gpu_rows.append(hidden.shape[0])
        return hidden

    def overlap(_cache, hidden, *_args):
        cpu_rows.append(hidden.shape[0])
        events.append("cpu")
        return hidden

    layer = SimpleNamespace(layer_id=0, top_k=6, _expert_gemm=gemm,
                            _decode_hybrid=overlap, skip_empty_cache=skip_empty_cache)
    engine = SimpleNamespace(
        moe_offload_cache=cache, model=SimpleNamespace(_iter_offload_moe_layers=lambda: iter([layer])),
        dtype=torch.bfloat16, device=torch.device("cpu"), stream=None,
        _execution_plan=SimpleNamespace(enabled=True), tp_cpu_group=object())
    monkeypatch.setattr(kernels, "fused_localize_cache_safe_routes", lambda weights, ids, **_: (weights, ids))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "Event", lambda **_: SimpleNamespace(
        record=lambda *_: None, elapsed_time=lambda _: 0.0))
    monkeypatch.setattr(torch.distributed, "barrier", lambda **_: events.append("barrier"))
    Engine._warmup_dspark_moe_kernels(engine, SimpleNamespace(
        speculative_dspark=False, max_running_req=1, cuda_graph_max_bs=1, cuda_graph_bs=graph_sizes,
        model_config=SimpleNamespace(dsv41_args=SimpleNamespace(dspark_block_size=5, dim=4))))
    assert gpu_rows == list(range(1, max_rows + 1))
    assert cpu_rows == (gpu_rows if hybrid else [])
    assert [rows for rows, _ in admissions] == gpu_rows
    for rows, kwargs in admissions:
        if skip_empty_cache and rows == 1:
            assert set(kwargs) == {"route_weights"}
            assert torch.equal(kwargs["route_weights"], torch.tensor([[1., 0., 0., 0., 0., 0.]]))
        else:
            assert kwargs == {}
    assert events[-1] == "barrier" and events.count("barrier") == 1
    if hybrid:
        last_cpu = max(i for i, event in enumerate(events) if event == "cpu")
        assert events[last_cpu + 1:][:3] == ["sync", "reset", "sync"]


def test_target_only_hybrid_warmup_runs_before_kv_and_graphs():
    from freetoken.engine.engine import Engine

    source = inspect.getsource(Engine.__init__)
    guard = source[source.index("if config.speculative_dspark or (") :]
    assert 'self._dsv41_plan is not None and config.moe_backend == "hybrid"' in guard
    assert guard.index("self._warmup_dspark_moe_kernels(config)") < guard.index("KV cache initialization")
