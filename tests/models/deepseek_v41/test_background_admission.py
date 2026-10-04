"""The experiment is worker/BS1 only and always drains at forward boundaries."""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_policy_scope(monkeypatch, worker, flag):
    from freetoken.models.deepseek_v41.moe import _background_admission_enabled
    if flag is None:
        monkeypatch.delenv("FREETOKEN_DSV41_BACKGROUND_ADMISSION", raising=False)
    else:
        monkeypatch.setenv("FREETOKEN_DSV41_BACKGROUND_ADMISSION", flag)
    assert _background_admission_enabled(SimpleNamespace(is_expert_worker=worker)) == (
        worker and flag == "1")


@pytest.mark.parametrize("enabled,rows,expected", [(False, 1, "control"),
    (True, 2, "control"), (True, 1, "background")])
def test_worker_decode_scope(monkeypatch, enabled, rows, expected):
    from freetoken.models.deepseek_v41.moe import RoutedExperts
    from freetoken.layers import ExpertParallelOffloadMoELayer
    from freetoken.moe import background_admission
    created = []
    class State:
        def __init__(self, cache, *, max_cpu_misses):
            assert max_cpu_misses == 1
            created.append(cache)
        def decode(self, *args): return "background"
    monkeypatch.setattr(background_admission, "BackgroundAdmission", State)
    monkeypatch.setattr(ExpertParallelOffloadMoELayer, "_decode_hybrid", lambda *a: "control")
    layer = object.__new__(RoutedExperts)
    layer.background_admission_enabled = enabled
    layer.background_admission_max_misses = 1
    cache = SimpleNamespace()
    for _ in range(2):
        assert layer._decode_hybrid(cache, torch.zeros(rows, 4), None, None) == expected
    assert len(created) == (1 if expected == "background" else 0)


@pytest.mark.parametrize("value", ["1", "2", "3", "6"])
def test_background_miss_limit_is_explicit_and_bounded(monkeypatch, value):
    from freetoken.models.deepseek_v41.moe import _background_admission_max_misses
    monkeypatch.setenv("FREETOKEN_DSV41_BACKGROUND_MAX_MISSES", value)
    assert _background_admission_max_misses() == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "7", "bad"])
def test_invalid_background_miss_limit_fails_before_capture(monkeypatch, value):
    from freetoken.models.deepseek_v41.moe import _background_admission_max_misses
    monkeypatch.setenv("FREETOKEN_DSV41_BACKGROUND_MAX_MISSES", value)
    with pytest.raises(ValueError):
        _background_admission_max_misses()


@pytest.mark.parametrize("fails", [False, True])
def test_model_forward_drains_even_on_failure(fails):
    from freetoken.models.deepseek_v41.model import DeepseekV41ExpertWorkerModel
    events = []
    state = SimpleNamespace(join=lambda: events.append("join"))
    cache = SimpleNamespace(_dsv41_background_admission=state)
    def forward(*args):
        events.append("work")
        if fails:
            raise RuntimeError("injected")
    worker = object.__new__(DeepseekV41ExpertWorkerModel)
    worker.args = SimpleNamespace(dim=4, engram_layer_ids=())
    worker.engram_coordinator = SimpleNamespace(execution=SimpleNamespace(phase_aware=False))
    worker.layers = [SimpleNamespace(worker_forward=forward,
        experts=SimpleNamespace(offload_cache=cache))]
    if fails:
        with pytest.raises(RuntimeError, match="injected"):
            worker.forward(torch.tensor([1]))
    else:
        assert worker.forward(torch.tensor([1])).shape == (1, 1)
    assert events == ["work", "join"]
