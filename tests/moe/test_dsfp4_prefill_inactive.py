"""Default-off grouped prefill skipping preserves active arithmetic and writes zeros."""
import pytest
import torch

from freetoken.moe import fused_ds_fp4 as fp4


def test_switch_default_off():
    import os
    assert fp4._PREFILL_SKIP_INACTIVE == (os.environ.get("FREETOKEN_DSV41_PREFILL_SKIP_INACTIVE", "0") == "1")


@pytest.mark.parametrize("experts,skip,expected", [(8,False,"native"),(112,False,"native"),
    (1024,False,"triton"),(8,True,"triton"),(112,True,"triton"),(2350,True,"triton")])
def test_sorter_contract_selection(monkeypatch, experts, skip, expected):
    from freetoken.kernel.triton import moe_align
    slots = object()
    def native(ids, block, count):
        assert ids is slots and block == 64 and count == experts
        return "native"
    def explicit(ids, block, count, *, include_sentinel):
        assert ids is slots and block == 64 and count == experts and include_sentinel
        return "triton"
    monkeypatch.setattr(fp4, "moe_align_block_size", native)
    monkeypatch.setattr(moe_align, "moe_align_block_size", explicit)
    assert fp4._align_prefill_routes(slots, 64, experts, skip) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("experts", [8,32,96,112,128,208,240,1024,2350])
@pytest.mark.parametrize("tokens", [130,1106,4096])
def test_explicit_sentinel_covers_every_route_once(experts, tokens):
    torch.manual_seed(6106 + experts + tokens)
    ids = torch.randint(experts, (tokens,6), dtype=torch.int32, device="cuda")
    for active in (6,0,1,3):
        masked = ids.clone()
        masked[:,active:] = experts
        masked[::3] = experts
        si, ei, nt = fp4._align_prefill_routes(masked, 64, experts, True)
        count = int(nt.item())
        assert count <= si.numel() and count % 64 == 0
        routes = si[:count]
        assert torch.all((routes >= 0) & (routes <= ids.numel()))
        assigned = ei[:count//64].repeat_interleave(64)
        assert torch.all((assigned >= 0) & (assigned <= experts))
        valid = routes < ids.numel()
        assert torch.equal(routes[valid].sort().values, torch.arange(ids.numel(), device="cuda", dtype=torch.int32))
        assert torch.equal(assigned[valid], masked.flatten()[routes[valid].long()])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("experts", [8, 112, 208, 1024])
@pytest.mark.parametrize("tokens", [130,4096])
def test_exact_routes_and_replay(monkeypatch, experts, tokens):
    torch.manual_seed(6106)
    t, h, i, k = tokens, 128, 64, 6
    banks = [torch.randint(0, 256, s, dtype=torch.uint8, device="cuda") for s in
             [(experts, 2*i, h//2), (experts, h, i//2)]]
    scales = [torch.full(s, 122, dtype=torch.uint8, device="cuda") for s in
              [(experts, 2*i, h//32), (experts, h, i//32)]]
    x = torch.randn(t, h, dtype=torch.bfloat16, device="cuda") * .25
    ids = torch.randint(experts, (t, k), dtype=torch.int32, device="cuda")
    w = torch.ones(t, k*2, device="cuda")[:, ::2]

    def run(enabled):
        monkeypatch.setattr(fp4, "_PREFILL_SKIP_INACTIVE", enabled)
        return fp4.routed_experts_fp4_prefill(x, ids, w, banks[0], scales[0], banks[1], scales[1],
            10., experts, return_route_outputs=True, activation_block_size=32)

    for _ in range(3):
        run(False); run(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    for active in (6, 0, 1, 5, 0, 3, 6):
        x.normal_().mul_(.25)
        ids.copy_((ids + 1) % experts)
        w.zero_()
        w[:, :active] = .75
        # Cover empty tokens interleaved with active ones and signed zero.
        w[::3] = -0.0
        expected, actual = run(False), run(True)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(expected).all()
        assert torch.equal(actual, expected)
        assert torch.equal(captured, expected)
        assert torch.count_nonzero(actual[w == 0]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_guard_clears_poisoned_outputs_before_bank_load():
    t, k, e, h, n = 130, 6, 8, 128, 128
    ids = torch.full((t, k), e, dtype=torch.int32, device="cuda")
    si, ei, ntpp = fp4._align_prefill_routes(ids, 64, e, True)
    x = torch.full((t, h), float("nan"), dtype=torch.bfloat16, device="cuda")
    weights = torch.zeros(t*k, device="cuda")
    packed = torch.full((e, n, h//2), 255, dtype=torch.uint8, device="cuda")
    scales = torch.full((e, n, h//32), 255, dtype=torch.uint8, device="cuda")
    out = torch.full((t, k, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    cfg = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
               num_warps=8, num_stages=1)
    fp4._grouped_prefill(x, packed, scales, out, weights, si, ei, ntpp, t*k, k, False, cfg,
                         inactive_expert=e)
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
