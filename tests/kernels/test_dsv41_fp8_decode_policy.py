"""Host-only guards for the opt-in native-quant decoder."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.kernel.triton.dsv4.fp8_linear import _decode_cfg
from freetoken.kernel.triton.dsv41 import fp8_decode as tuned
from freetoken.kernel.triton.dsv41 import fp8_linear as wrapper


def tensor(shape, device="cuda:7", cuda=True):
    size = 1
    for n in shape:
        size *= n
    return SimpleNamespace(shape=shape, ndim=len(shape), is_cuda=cuda,
                           device=device, numel=lambda: size)


@pytest.mark.parametrize("shape,cfg", tuned.SM120_CONFIGS.items())
def test_preserves_splitk(shape, cfg):
    n,k = shape
    assert cfg[1] == _decode_cfg(n,k,32)[1]
    assert n % 32 == k % (cfg[1]*32) == 0
    assert cfg[3] in (1,2)


@pytest.mark.parametrize("case", ["disabled","emulated","cpu","prefill","unknown","sm89","sm80"])
def test_fallbacks_do_not_route_to_tuned_kernel(monkeypatch, case):
    monkeypatch.setattr(tuned,"ENABLED",case != "disabled")
    monkeypatch.setattr(tuned,"FORCE_EMU",case == "emulated")
    x = tensor((7 if case == "prefill" else 1,5120),cuda=case != "cpu")
    w = tensor((512 if case != "unknown" else 256,5120))
    queried = []
    def capability(device):
        queried.append(device)
        return {"sm89":(8,9),"sm80":(8,0)}.get(case,(12,0))
    monkeypatch.setattr(torch.cuda,"get_device_capability",capability)
    assert not tuned.use_tuned_decode(x,w)
    assert queried == (["cuda:7"] if case in ("sm89","sm80") else [])


def test_tensor_local_device_and_single_row_leading_dims(monkeypatch):
    monkeypatch.setattr(tuned,"ENABLED",True)
    monkeypatch.setattr(tuned,"FORCE_EMU",False)
    queried = []
    def capability(device):
        queried.append(device)
        return (12,0)
    monkeypatch.setattr(torch.cuda,"get_device_capability",capability)
    assert tuned.use_tuned_decode(tensor((1,1,5120)),tensor((512,5120)))
    assert queried == ["cuda:7"]


@pytest.mark.parametrize("selected", [False,True])
def test_wrapper_preserves_original_fallback_and_bias(monkeypatch,selected):
    x,w,s,b = object(),object(),object(),object()
    expected = object()
    monkeypatch.setattr(wrapper,"use_tuned_decode",lambda a,c: selected)
    calls=[]
    def original(*args,**kwargs):
        calls.append(("original",args,kwargs)); return expected
    def candidate(*args,**kwargs):
        calls.append(("tuned",args,kwargs)); return expected
    monkeypatch.setattr(wrapper,"block_fp8_linear",original)
    monkeypatch.setattr(wrapper,"tuned_fp8_linear_32",candidate)
    assert wrapper.block_fp8_linear_32(x,w,s,b) is expected
    assert calls == [("tuned" if selected else "original",(x,w,s,b),{} if selected else {"block_size":32})]
