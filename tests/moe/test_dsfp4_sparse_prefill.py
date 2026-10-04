"""Sparse DS-FP4 prefill must only change weight movement, not arithmetic."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.moe import ExpertParallelOffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slots", [16, 1472, 2350])
def test_sparse_prefill_matches_full_banks_through_eviction_and_phase_changes(slots):
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4_prefill

    # Both the GEMV (<768 routes) and grouped-GEMM branches; realistic cache
    # dimensions with small experts keep this test below 600 MiB of bank VRAM.
    E, H, I, K, L = 8, 512, 256, 6, 3
    dev = torch.device("cuda")
    rng = torch.Generator().manual_seed(7429)
    sources = {}
    for name, shape, low, high in (
        ("gate_up_packed", (E, 2*I, H//2), 0, 256),
        ("gate_up_scale", (E, 2*I, H//32), 119, 124),
        ("down_packed", (E, H, I//2), 0, 256),
        ("down_scale", (E, H, I//32), 119, 124),
    ):
        sources[name] = [torch.randint(low, high, shape, generator=rng,
                                      dtype=torch.uint8).pin_memory() for _ in range(L)]
    cache = OffloadMoeCache(L, E, slots, dev, prefill_overlap=True,
                           sparse_prefill_max_tokens=256, quant_format="ds_fp4")
    cache.set_bank_sources(sources)
    layer = object.__new__(ExpertParallelOffloadMoELayer)
    layer.offload_cache = cache
    layer.num_experts = E
    layer.swiglu_limit = 7.0
    layer.dsfp4_activation_block_size = 32
    layer.skip_inactive_prefill_routes = False
    for T in (1, 16, 127, 128, 129, 170, 171, 256, 257, 512, 16, 128):
        for layer_id in range(L):
            layer.layer_id = layer_id
            x = (torch.randn(T, H, generator=rng) * .1).to(dev, torch.bfloat16)
            weights = torch.rand(T, K, generator=rng).to(dev)
            weights[:, 1::2] = 0
            weights[0] = 0  # no local route on this token
            ids = torch.randint(E, (T, K), generator=rng, dtype=torch.int32).to(dev)
            ids[weights == 0] = E  # normal EP out-of-range sentinel
            safe = layer._cache_safe_route_ids(weights, ids)
            banks = [sources[name][layer_id].to(dev) for name in cache.bank_schema]
            reference = routed_experts_fp4_prefill(
                x, safe, weights, *banks, layer.swiglu_limit, E,
                return_route_outputs=True, activation_block_size=32)
            actual = layer._prefill_routed(x, weights, ids)
            torch.cuda.synchronize()
            if T <= 256:
                for expert in range(E):
                    slot = int(cache.slot_for_id[layer_id, expert])
                    if slot >= 0:
                        for name in cache.bank_schema:
                            assert torch.equal(cache.bank_caches[name][slot].cpu(),
                                               sources[name][layer_id][expert]), (name, layer_id, expert, slot)
            assert torch.isfinite(actual).all()
            assert torch.equal(actual, reference), (slots, T, layer_id,
                (actual.float() - reference.float()).abs().max().item())


@pytest.mark.parametrize("unpinned,fmt", [(True, "ds_fp4"), (False, "nvfp4")])
def test_sparse_prefill_does_not_admit_unsupported_host_or_format(monkeypatch, unpinned, fmt):
    from freetoken.layers.moe import OffloadMoELayer

    calls = []
    monkeypatch.setattr(OffloadMoELayer, "_prefill_routed",
                        lambda self, x, weights, ids: calls.append(ids) or x)
    layer = object.__new__(ExpertParallelOffloadMoELayer)
    layer.layer_id = 0
    layer.offload_cache = SimpleNamespace(sparse_prefill_max_tokens=256,
        quant_format=fmt, is_unpinned_layer=lambda _: unpinned)
    x = torch.ones(2, 4)
    weights = torch.tensor([[.5, 0], [.5, 0]])
    ids = torch.tensor([[1, 4], [1, 4]], dtype=torch.int32)
    assert layer._prefill_routed(x, weights, ids) is x
    assert len(calls) == 1


def test_sparse_prefill_flag_does_not_change_mtp_dispatch(monkeypatch):
    import freetoken.layers.moe as module

    layer = object.__new__(ExpertParallelOffloadMoELayer)
    layer.offload_cache = SimpleNamespace(sparse_prefill_max_tokens=256)
    monkeypatch.setattr(module, "get_global_ctx", lambda: SimpleNamespace(
        batch=SimpleNamespace(is_prefill=True, is_moe_prefill=False, speculative=True)))
    calls = []
    layer._decode_routed = lambda x, weights, ids: calls.append("decode") or x
    layer._maybe_all_reduce = lambda x: x
    layer._prefill_routed = lambda *_: pytest.fail("MTP must not enter ordinary short prefill")
    x = torch.zeros(5, 4)
    assert layer.routed_forward(x, torch.ones(5, 2), torch.zeros(5, 2, dtype=torch.int32)) is x
    assert calls == ["decode"]
