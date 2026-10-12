from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear
from freetoken.kernel.triton.dsv41 import fp8_decode as tuned
from freetoken.kernel.triton.dsv41.fp8_linear import block_fp8_linear_32

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16,torch.float32])
@pytest.mark.parametrize("leading",[(),(1,),(1,1)])
def test_tuned_decode_preserves_dtype_shape_bias_and_graph(monkeypatch,dtype,leading):
    if torch.cuda.get_device_capability() != (12,0):
        pytest.skip("SM120-only experimental kernel")
    monkeypatch.setattr(tuned,"ENABLED",True)
    torch.manual_seed(371)
    n,k=512,5120
    weight=torch.randn(n,k,device="cuda").to(torch.float8_e4m3fn)
    scale=torch.randint(124,130,(n//32,k//32),device="cuda",dtype=torch.uint8)
    bias=torch.randn(n,device="cuda",dtype=torch.float32)
    x=torch.randn((*leading,k),device="cuda",dtype=dtype)
    assert tuned.use_tuned_decode(x,weight)
    expected=block_fp8_linear(x,weight,scale,bias,block_size=32)
    actual=block_fp8_linear_32(x,weight,scale,bias)
    assert actual.shape == (*leading,n) and actual.dtype == dtype
    assert torch.equal(actual,expected)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured=block_fp8_linear_32(x,weight,scale,bias)
    for _ in range(4):
        x.normal_(); graph.replay()
        assert torch.equal(captured,block_fp8_linear(x,weight,scale,bias,block_size=32))


@pytest.mark.parametrize("leading", [(2,), (1, 2)])
def test_two_row_decode_matches_independent_c1_rows_and_graph(monkeypatch, leading):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120-only experimental kernel")
    monkeypatch.setattr(tuned, "ENABLED", True)
    torch.manual_seed(372)
    n, k = 512, 5120
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.randint(124, 130, (n // 32, k // 32), device="cuda", dtype=torch.uint8)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    x = torch.randn((*leading, k), device="cuda", dtype=torch.bfloat16)

    def independent_rows():
        flat = x.reshape(2, k)
        return torch.cat([
            block_fp8_linear(flat[i : i + 1], weight, scale, bias, block_size=32)
            for i in range(2)
        ]).reshape(*leading, n)

    assert tuned.use_tuned_decode(x, weight)
    actual = block_fp8_linear_32(x, weight, scale, bias)
    assert actual.shape == (*leading, n) and actual.dtype == x.dtype
    assert torch.equal(actual, independent_rows())

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = block_fp8_linear_32(x, weight, scale, bias)
    for _ in range(4):
        x.normal_()
        graph.replay()
        assert torch.equal(captured, independent_rows())
