"""G3 graph-pool and memo-epoch qualification scaffolding."""

from __future__ import annotations

import pytest
import torch

from freetoken.models.deepseek_v41 import authority_fusion as fusion
from freetoken.kernel.triton.dsv41 import hc_prepare_mix


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="G3 needs CUDA")


def _capture(rows, pool=None):
    x = torch.randn(rows, 1, 4, 5120, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        hc_prepare_mix(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        flat, square = hc_prepare_mix(x)
    return graph, x, flat, square


def test_shared_pool_bs2_bs1_bs2_new_inputs():
    graph2, x2, flat2, square2 = _capture(2)
    graph1, x1, flat1, square1 = _capture(1, graph2.pool())
    for graph, x, flat, square in (
        (graph2, x2, flat2, square2),
        (graph1, x1, flat1, square1),
        (graph2, x2, flat2, square2),
    ):
        x.normal_()
        expected = x.flatten(-2).float()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(flat.view(torch.int32), expected.view(torch.int32))
        assert torch.equal(square.view(torch.int32), expected.square().view(torch.int32))


def test_forward_epoch_never_reuses_warmup_memo():
    memo = fusion.ForwardMemo()
    table = torch.arange(64, device="cuda", dtype=torch.float32).reshape(16, 4)
    positions = torch.tensor([1, 3], device="cuda")
    first_epoch = memo.begin()
    first = memo.frequencies(table, positions)
    memo.clear()
    second_epoch = memo.begin()
    second = memo.frequencies(table, positions)
    assert second_epoch == first_epoch + 1 and second is not first
    memo.clear()
