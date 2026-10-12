from types import SimpleNamespace as NS

import pytest
import torch

import freetoken.moe.phase_priority as boundary


def make():
    order = []
    cache = NS(quant_format="ds_fp4", decode_target="hybrid", num_layers=40,
               num_experts=208, id_of_slot=torch.tensor([0, 32, 175, 176, -1], dtype=torch.int32),
               usage=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64),
               finish_background_admission=lambda: order.append("fence"),
               prefill_copy_stream=None, device=torch.device("cpu"))
    return cache, order


def batch(prefill=False, speculative=False, size=1):
    return NS(is_prefill=prefill, speculative=speculative, reqs=[object()] * size)


def test_exactly_once_after_last_prefill_chunk_and_after_fence(monkeypatch):
    cache, order = make()
    primitive = boundary.demote
    def checked(*args, **kwargs):
        assert order[-1] == "fence"
        order.append("demote")
        return primitive(*args, **kwargs)
    monkeypatch.setattr(boundary, "demote", checked)
    phase = boundary.PhaseCachePriority(cache, start=32, count=144)
    before = cache.usage.clone()
    for _ in range(5):
        phase.before(batch(True))
        assert torch.equal(cache.usage, before) and order == []
    phase.before(batch())
    assert order == ["fence", "demote"]
    assert cache.usage.tolist() == [0, 11, 12, 0, 14]
    cache.usage.fill_(20)
    for _ in range(100):
        phase.before(batch())
    assert order == ["fence", "demote"] and cache.usage.tolist() == [20] * 5
    phase.before(batch(True))
    phase.before(batch())
    assert order == ["fence", "demote", "fence", "demote"]


def test_decode_without_prefill_and_cancelled_prefill_next_request():
    cache, order = make()
    phase = boundary.PhaseCachePriority(cache, start=32, count=144)
    phase.before(batch())
    assert order == []
    phase.before(batch(True))  # Cancelled before a decode.
    phase.before(batch(True))  # Different request starts with prefill.
    phase.before(batch())
    assert order == ["fence"] and not phase.pending


def test_failure_does_not_clear_pending(monkeypatch):
    cache, _ = make()
    phase = boundary.PhaseCachePriority(cache, start=32, count=144)
    phase.before(batch(True))
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic primitive failure")
    monkeypatch.setattr(boundary, "demote", fail)
    with pytest.raises(RuntimeError):
        phase.before(batch())
    assert phase.pending


def test_warm_uses_separate_tensors_and_preserves_pending():
    cache, order = make()
    phase = boundary.PhaseCachePriority(cache, start=32, count=144)
    phase.pending = True
    owners, usage = cache.id_of_slot.clone(), cache.usage.clone()
    phase.warm(None)
    assert phase.pending and order == []
    assert torch.equal(cache.id_of_slot, owners) and torch.equal(cache.usage, usage)


@pytest.mark.parametrize("speculative,size", [(True, 1), (False, 0), (False, 2)])
def test_reject_unsupported_batches(speculative, size):
    cache, order = make()
    phase = boundary.PhaseCachePriority(cache, start=32, count=144)
    with pytest.raises(ValueError):
        phase.before(batch(True, speculative, size))
    assert not phase.pending and order == []


def engine():
    cache, _ = make()
    return NS(config=NS(max_running_req=1, speculative_dspark=False, cuda_graph_max_bs=1,
                        tp_info=NS(size=4), model_config=NS(dsv41_args=NS(n_routed_experts=384))),
              moe_offload_cache=cache, stream=None,
              _dsv41_plan=NS(phase_aware=True,
                  storage_partition=lambda total: NS(global_offset=144, local_count=208),
                  partition=lambda total, prefill: NS(global_offset=176, local_count=144)))


def test_factory_derives_storage_local_range_and_warms_without_cache_mutation():
    e = engine()
    before = e.moe_offload_cache.usage.clone()
    obj = boundary.PhaseCachePriority.from_engine(e)
    assert (obj.start, obj.count) == (32, 144) and not obj.pending
    assert torch.equal(e.moe_offload_cache.usage, before)


@pytest.mark.parametrize("damage", ["plan", "phase", "concurrency", "spec", "graph", "ranks", "layers", "storage"])
def test_factory_rejects_unqualified_runtime(damage):
    e = engine()
    if damage == "plan":
        e._dsv41_plan = None
    elif damage == "phase":
        e._dsv41_plan.phase_aware = False
    elif damage == "concurrency":
        e.config.max_running_req = 2
    elif damage == "spec":
        e.config.speculative_dspark = True
    elif damage == "graph":
        e.config.cuda_graph_max_bs = 2
    elif damage == "ranks":
        e.config.tp_info.size = 3
    elif damage == "layers":
        e.moe_offload_cache.num_layers = 1
    else:
        e.moe_offload_cache.num_experts = 144
    with pytest.raises(ValueError):
        boundary.PhaseCachePriority.from_engine(e)


# ---- DeepSeek-V4.1 gated C2: one-request prefills, decode batches of one or two rows.


def c2_engine():
    e = engine()
    e.config.max_running_req = e.config.cuda_graph_max_bs = 2
    e.config.model_config.prefill_batch_size_limit = 1
    return e


def test_factory_accepts_the_gated_c2_geometry():
    obj = boundary.PhaseCachePriority.from_engine(c2_engine())
    assert (obj.start, obj.count, obj.max_decode_rows) == (32, 144, 2)
    assert boundary.PhaseCachePriority.from_engine(engine()).max_decode_rows == 1


@pytest.mark.parametrize("damage", ["graph1", "graph0", "ragged_prefill", "three"])
def test_factory_rejects_unqualified_c2(damage):
    e = c2_engine()
    if damage == "graph1":
        e.config.cuda_graph_max_bs = 1
    elif damage == "graph0":
        e.config.cuda_graph_max_bs = 0
    elif damage == "ragged_prefill":
        e.config.model_config.prefill_batch_size_limit = None
    else:
        e.config.max_running_req = e.config.cuda_graph_max_bs = 3
    with pytest.raises(ValueError):
        boundary.PhaseCachePriority.from_engine(e)


def test_c2_demotes_once_at_the_first_decode_of_one_or_two_rows(monkeypatch):
    cache, order = make()
    primitive = boundary.demote

    def checked(*args, **kwargs):
        assert order[-1] == "fence"
        order.append("demote")
        return primitive(*args, **kwargs)

    monkeypatch.setattr(boundary, "demote", checked)
    phase = boundary.PhaseCachePriority(cache, start=32, count=144, max_decode_rows=2)
    phase.before(batch(True))  # one-request prefill (the other request waits)
    phase.before(batch(size=2))  # the first decode after it holds both rows
    assert order == ["fence", "demote"] and cache.usage.tolist() == [0, 11, 12, 0, 14]
    for size in (2, 1, 2):
        phase.before(batch(size=size))
    assert order == ["fence", "demote"]  # no CUDA work on later decode steps
    phase.before(batch(True))
    phase.before(batch(size=1))
    assert order == ["fence", "demote", "fence", "demote"]


@pytest.mark.parametrize("prefill,size", [(True, 2), (False, 3), (False, 0)])
def test_c2_still_rejects_two_request_prefills_and_wider_decodes(prefill, size):
    cache, order = make()
    phase = boundary.PhaseCachePriority(cache, start=32, count=144, max_decode_rows=2)
    with pytest.raises(ValueError):
        phase.before(batch(prefill, size=size))
    assert not phase.pending and order == []


def test_one_request_priority_still_rejects_two_row_decodes():
    cache, order = make()
    phase = boundary.PhaseCachePriority(cache, start=32, count=144)
    phase.before(batch(True))
    with pytest.raises(ValueError):
        phase.before(batch(size=2))
    assert phase.pending and order == []


@pytest.mark.parametrize("rows", [0, 3, True])
def test_decode_row_limit_is_one_or_two(rows):
    cache, _ = make()
    with pytest.raises(ValueError):
        boundary.PhaseCachePriority(cache, start=32, count=144, max_decode_rows=rows)
