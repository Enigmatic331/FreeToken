"""Opt-in zero local routes must not change active-route arithmetic or graph replay."""
import pytest
import torch

from freetoken.moe import fused_ds_fp4 as fp4


@pytest.mark.parametrize("enabled,tokens,block,kind,capability,expected", [
    (False, 1, 32, "cuda", (12, 0), False),
    (True, 2, 32, "cuda", (12, 0), False),
    (True, 1, 128, "cuda", (12, 0), False),
    (True, 1, 32, "cpu", (12, 0), False),
    (True, 1, 32, "cuda", (8, 0), False),
    (True, 1, 32, "cuda", (9, 0), False),
    (True, 1, 32, "cuda", (12, 0), True),
    (True, 1, 32, "cuda", (8, 9), True),
])
def test_zero_route_policy(monkeypatch, enabled, tokens, block, kind, capability, expected):
    calls = []
    device = torch.device(kind, 2) if kind == "cuda" else torch.device("cpu")
    monkeypatch.setattr(fp4, "_SKIP_ZERO_ROUTES", enabled)
    def probe(actual):
        calls.append(actual)
        return capability
    monkeypatch.setattr(torch.cuda, "get_device_capability", probe)
    fp4._zero_route_device_supported.cache_clear()
    try:
        for _ in range(2):
            assert fp4._use_zero_route_skip(tokens, block, device) is expected
        assert calls == ([device] if enabled and tokens == 1 and block == 32 and kind == "cuda" else [])
    finally:
        fp4._zero_route_device_supported.cache_clear()


@pytest.mark.parametrize("enabled,capability,sms,expected", [
    (False, (8, 9), 128, (16, 128, 1)),
    (True, (12, 0), 128, (16, 128, 1)),
    (True, (8, 9), 80, (16, 128, 1)),
    (True, (8, 9), 128, (8, 128, 1)),
])
def test_decode_launch_geometry(monkeypatch, enabled, capability, sms, expected):
    calls = []
    device = torch.device("cuda", 2)
    monkeypatch.setattr(fp4, "_SM89_WIDE_GRID", enabled)

    def capability_probe(actual):
        calls.append(("capability", actual))
        return capability

    class Properties:
        multi_processor_count = sms

    def properties_probe(actual):
        calls.append(("properties", actual))
        return Properties()

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability_probe)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties_probe)
    fp4._decode_launch_geometry.cache_clear()
    try:
        assert fp4._decode_launch_geometry(device) == expected
        assert fp4._decode_launch_geometry(device) == expected
        if not enabled:
            assert calls == []
        elif capability != (8, 9):
            assert calls == [("capability", device)]
        else:
            assert calls == [("capability", device), ("properties", device)]
    finally:
        fp4._decode_launch_geometry.cache_clear()


@pytest.mark.parametrize("selected,kind,capability,expected", [
    ("", "cuda", (8, 0), (64, 64, 64, 8, 8, 1)),
    ("m32n64w4s2", "cpu", (8, 0), (64, 64, 64, 8, 8, 1)),
    ("m32n64w4s2", "cuda", (8, 9), (64, 64, 64, 8, 8, 1)),
    ("m32n64w4s2", "cuda", (8, 0), (32, 64, 64, 8, 4, 2)),
    ("m32n64w8s1", "cuda", (8, 0), (32, 64, 64, 8, 8, 1)),
    ("m64n32w4s2", "cuda", (8, 0), (64, 32, 64, 8, 4, 2)),
])
def test_prefill_launch_geometry(monkeypatch, selected, kind, capability, expected):
    calls = []
    device = torch.device(kind, 2) if kind == "cuda" else torch.device("cpu")
    monkeypatch.setattr(fp4, "_SM80_PREFILL_CONFIG", selected)

    def probe(actual):
        calls.append(actual)
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", probe)
    fp4._prefill_launch_geometry.cache_clear()
    try:
        assert fp4._prefill_launch_geometry(device) == expected
        assert fp4._prefill_launch_geometry(device) == expected
        assert calls == ([device] if selected and kind == "cuda" else [])
    finally:
        fp4._prefill_launch_geometry.cache_clear()


def test_prefill_launch_geometry_rejects_unknown(monkeypatch):
    monkeypatch.setattr(fp4, "_SM80_PREFILL_CONFIG", "mystery")
    fp4._prefill_launch_geometry.cache_clear()
    try:
        with pytest.raises(ValueError, match="unknown FREETOKEN_DSV41_SM80"):
            fp4._prefill_launch_geometry(torch.device("cuda", 0))
    finally:
        fp4._prefill_launch_geometry.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("route_outputs", [False, True])
def test_zero_routes_exact_replay(monkeypatch, dtype, route_outputs):
    if torch.cuda.get_device_capability() not in ((12, 0), (8, 9)):
        pytest.skip("qualified architectures only")
    torch.manual_seed(4103)
    device = torch.device("cuda", 0)
    h, i, experts, topk = 1024, 512, 8, 6
    banks = [torch.randint(0, 256, shape, dtype=torch.uint8, device=device) for shape in
             [(experts, 2*i, h//2), (experts, h, i//2)]]
    scales = [torch.randint(119, 123, shape, dtype=torch.uint8, device=device) for shape in
              [(experts, 2*i, h//32), (experts, h, i//32)]]
    x = torch.randn((1, h), dtype=dtype, device=device) * .3
    ids = torch.arange(topk, device=device, dtype=torch.int32).unsqueeze(0)
    # Non-contiguous route weights exercise actual weight strides in gate/up too.
    weights = torch.ones((1, 2*topk), device=device)[:, ::2]
    def run(enabled):
        monkeypatch.setattr(fp4, "_SKIP_ZERO_ROUTES", enabled)
        return fp4.routed_experts_fp4(x, ids, weights, banks[0], scales[0], banks[1], scales[1],
                                     10., return_route_outputs=route_outputs, activation_block_size=32)
    for _ in range(3):
        run(False); run(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    # Both transitions matter: nonzero->zero must clear stale output lanes,
    # zero->nonzero must not keep a captured mask or input.
    for active in (6, 0, 1, 5, 2, 0, 3, 6):
        x.normal_().mul_(.3)
        ids.copy_((ids+1) % experts)
        weights.zero_()
        weights[:, :active] = .75
        expected = run(False)
        actual = run(True)
        graph.replay()
        assert torch.isfinite(expected).all()
        assert torch.equal(expected, actual)
        assert torch.equal(expected, captured)
