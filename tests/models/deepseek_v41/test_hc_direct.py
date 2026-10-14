"""Exactness and boundary tests for opt-in direct HC pre-combine inputs."""
import pytest
import torch
from types import SimpleNamespace
from freetoken.models.deepseek_v41 import hc


@pytest.mark.parametrize("enabled", [False,True])
def test_cpu_path_unchanged(monkeypatch, enabled):
    monkeypatch.setattr(hc, "_DIRECT_PRE", enabled)
    torch.manual_seed(6106)
    x=torch.randn(2,3,4,64,dtype=torch.bfloat16)
    pre=torch.randn(2,3,4,dtype=torch.float32)
    expected=(pre.unsqueeze(-1)*x.float()).sum(-2).to(x.dtype)
    assert torch.equal(hc.hc_pre(x,pre),expected)


@pytest.mark.parametrize("enabled,device", [(False,"cuda"),(True,"cpu")])
def test_inactive_warmup_does_not_touch_cuda(monkeypatch, enabled, device):
    monkeypatch.setattr(hc, "_DIRECT_PRE", enabled)
    def fail(): raise AssertionError("Unexpected CUDA query")
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",fail)
    hc.warmup_direct_pre(torch.device(device),5120,4)


def test_enabled_warmup_rejects_graph_capture(monkeypatch):
    monkeypatch.setattr(hc, "_DIRECT_PRE", True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="precede graph capture"):
        hc.warmup_direct_pre(torch.device("cuda"), 5120, 4)


def test_enabled_warmup_covers_row_specializations_without_collectives(monkeypatch):
    monkeypatch.setattr(hc, "_DIRECT_PRE", True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    device = torch.device("cuda:0")
    allocations, calls, syncs, phases = [], [], [], []
    def zeros(shape, *, device, dtype):
        value = SimpleNamespace(shape=shape, device=device, dtype=dtype)
        allocations.append(value)
        return value
    monkeypatch.setattr(torch, "zeros", zeros)
    def record(x, pre, **phase):
        calls.append((x, pre))
        phases.append(phase)
    monkeypatch.setattr(hc, "hc_pre", record)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: syncs.append(d))
    hc.warmup_direct_pre(device, 5120, 4)
    assert len(allocations) == 6 and syncs == [device]
    assert [x.shape for x, _ in calls] == [(r, 4, 5120) for r in (1, 2, 16)]
    assert [p.shape for _, p in calls] == [(r, 4) for r in (1, 2, 16)]
    assert all(x.device == p.device == device for x, p in calls)
    assert all(x.dtype == torch.bfloat16 and p.dtype == torch.float32 for x, p in calls)
    # The warmup must select the direct variant in the prefill-only mode too.
    assert phases == [{"prefill": True}] * 3


def test_runtime_warms_hc_before_attention_bind(monkeypatch):
    from freetoken.models.deepseek_v41.model import Transformer
    calls = []
    device = torch.device("cuda:0")
    monkeypatch.setattr(hc, "warmup_direct_pre", lambda d, dim, mult: calls.append(("warm", d, dim, mult)))
    model = SimpleNamespace(
        _attach_routed_experts=lambda: calls.append(("experts",)),
        args=SimpleNamespace(dim=5120, hc_mult=4),
        layers=[SimpleNamespace(attn=SimpleNamespace(
            bind=lambda d: calls.append(("bind", d)), indexer=None))],
    )
    Transformer.attach_runtime(model, device)
    assert calls == [("experts",), ("warm", device, 5120, 4), ("bind", device)]


@pytest.mark.skipif(not torch.cuda.is_available(),reason="needs CUDA")
@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16,torch.float32])
@pytest.mark.parametrize("shape",[(1,4,5120),(2,3,4,64),(130,4,5120)])
@pytest.mark.parametrize("strided",[False,True])
def test_direct_matches_staged_and_graph(monkeypatch,dtype,shape,strided):
    torch.manual_seed(6106)
    storage_shape=(*shape[:-1],shape[-1]*(2 if strided else 1))
    storage=torch.randn(storage_shape,dtype=dtype,device="cuda")
    x=storage[...,::2] if strided else storage
    pre=torch.randn(shape[:-1],dtype=torch.float32,device="cuda")
    def run(enabled):
        monkeypatch.setattr(hc,"_DIRECT_PRE",enabled)
        return hc.hc_pre(x,pre)
    for _ in range(3): run(False);run(True)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): captured=run(True)
    for case in range(8):
        x.normal_().mul_(2.**(case-3));pre.normal_()
        if case==0: x.zero_()
        if case==1: pre.zero_();pre[...,0]=1
        if case==2: x[...,1,:]=-x[...,0,:];pre.fill_(1)
        if case==3:
            x.fill_(torch.finfo(dtype).tiny / 2)
            x[...,1::2,:].neg_()
            pre.fill_(1)
        if case==4:
            x.zero_();x[...,1::2,:].neg_();pre.fill_(1)
        expected,actual=run(False),run(True)
        graph.replay();torch.cuda.synchronize()
        assert torch.isfinite(expected).all()
        # Integer views include signed zero, unlike floating-point equality.
        assert torch.equal(expected.view(torch.int8), actual.view(torch.int8))
        assert torch.equal(expected.view(torch.int8), captured.view(torch.int8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_direct_wrapper_full_8192_chunk(monkeypatch):
    """Exercise the actual wrapper at the production chunk size, not just its kernel."""
    torch.manual_seed(6106)
    x = torch.randn(8192, 4, 5120, dtype=torch.bfloat16, device="cuda")
    pre = torch.randn(8192, 4, dtype=torch.float32, device="cuda")
    monkeypatch.setattr(hc, "_DIRECT_PRE", False)
    expected = hc.hc_pre(x, pre)
    monkeypatch.setattr(hc, "_DIRECT_PRE", True)
    actual = hc.hc_pre(x, pre)
    torch.cuda.synchronize()
    assert torch.isfinite(expected).all()
    assert torch.equal(expected.view(torch.int8), actual.view(torch.int8))
