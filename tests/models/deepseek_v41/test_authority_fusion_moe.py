"""G2/G3 authority-route and prefilled-wire qualification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.deepseek_v41 import authority_fusion as fusion
from freetoken.models.deepseek_v41 import moe


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="G2/G3 need CUDA")


@pytest.mark.parametrize("rows", [1, 2])
def test_real_wire_views_have_frozen_offsets_and_alignment(rows):
    hidden = torch.zeros(rows, 5120, dtype=torch.bfloat16, device="cuda")
    wire, wire_hidden, weights, ids = fusion._dispatch_wire(hidden, 6)
    assert wire_hidden.shape == hidden.shape
    assert weights.shape == ids.shape == (rows, 6)
    assert ids.data_ptr() - wire.data_ptr() == (10264 if rows == 1 else 20528)
    assert ids.data_ptr() % (8 if rows == 1 else 16) == 0


@pytest.mark.parametrize("rows", [1, 2])
def test_prefilled_wire_is_broadcast_once_without_copy(rows):
    hidden = torch.randn(rows, 5120, dtype=torch.bfloat16, device="cuda")
    wire, wire_hidden, weights, ids = fusion._dispatch_wire(hidden, 6)
    wire_hidden.copy_(hidden)
    weights.normal_()
    ids.copy_(torch.arange(6, device="cuda", dtype=torch.int32).repeat(rows, 1))
    before = wire.clone()
    calls = []

    class FakeComm:
        def broadcast(self, value, root):
            calls.append((value.data_ptr(), root))
            return value

    out = moe._broadcast_prefilled_decode_dispatch(
        FakeComm(), SimpleNamespace(backbone_rank=0, rank=0), wire, tuple(hidden.shape), 6
    )
    assert calls == [(wire.data_ptr(), 0)]
    assert torch.equal(wire, before)
    assert out[0].data_ptr() == wire_hidden.data_ptr()


def test_route_scope_cleans_up_after_cuda_error():
    experts = SimpleNamespace(_authority_route_passthrough=False)
    with pytest.raises(RuntimeError):
        with fusion._authority_route_passthrough(experts):
            raise RuntimeError("planted route failure")
    assert not experts._authority_route_passthrough
