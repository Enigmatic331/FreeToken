from __future__ import annotations

import pytest
import torch
from torch import nn

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.engram_layer import Engram


DIM = 64
HC = 4
HASHES = 4
HEAD_DIM = 32
EPS = 1e-6


class HostTable:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def lookup(self, row_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self.weight.index_select(0, row_ids.flatten()).view(*row_ids.shape, -1)

    def prefetch(self, row_ids: torch.Tensor) -> None:
        pass


def _module() -> Engram:
    args = DeepseekV41Args(
        dim=DIM,
        n_layers=2,
        n_heads=4,
        compress_ratios=(0, 0),
        engram_layer_ids=(1,),
        engram_num_embeddings=(16,),
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=HEAD_DIM,
        norm_eps=EPS,
    )
    assert args.hc_mult == HC and args.engram_hashes_per_token == HASHES
    module = Engram(args, 1)
    torch.manual_seed(41)
    # The FP8 projection needs CUDA; a float projection exercises the same shape contract.
    module.wkv = nn.Linear(HASHES * HEAD_DIM, DIM * (HC + 1), bias=False)
    nn.init.normal_(module.wkv.weight, std=0.1)
    module.q_weight.data.copy_(torch.randn(HC, DIM).bfloat16())
    module.k_weight.data.copy_(torch.randn(HC, DIM).bfloat16())
    module.attach_table(HostTable(torch.randn(16, HEAD_DIM)))
    return module


def _reference(module: Engram, x: torch.Tensor, row_ids: torch.Tensor, mask) -> torch.Tensor:
    """Per-token gate on flat [T, HC, DIM] tokens; no cross-token broadcasting possible."""

    flat_x = x.reshape(-1, HC, DIM)
    flat_mask = None if mask is None else mask.reshape(-1)
    rows = module.table.lookup(row_ids)
    product = module.q_weight.float() * module.k_weight.float()
    out = []
    for t in range(flat_x.shape[0]):
        kv = module.wkv(rows[t].flatten())
        key, value = kv.split([HC * DIM, DIM])
        key = key.float().view(HC, DIM)
        h = flat_x[t].float()
        rstd = torch.rsqrt(h.square().mean(-1) + EPS) * torch.rsqrt(
            key.square().mean(-1) + EPS
        )
        dot = (h * product * key).sum(-1) * rstd * DIM**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if flat_mask is not None and not bool(flat_mask[t]):
            gate = torch.zeros_like(gate)
        out.append((h + gate.unsqueeze(-1) * value.float()).to(x.dtype))
    return torch.stack(out).view_as(x)


@pytest.mark.parametrize(
    ("token_shape", "with_mask"),
    [
        ((3,), True),  # legacy token-flat hidden
        ((3,), False),
        ((1, 5), True),  # single-request prefill / verify
        ((1, 1), False),  # batch-1 decode
        ((2, 1), False),  # batch-2 decode (the C2 crash)
        ((2, 1), True),
        ((4, 1), True),  # padded graph batch
    ],
)
def test_engram_preserves_hidden_shape_per_token(token_shape, with_mask):
    module = _module()
    tokens = int(torch.tensor(token_shape).prod())
    row_ids = torch.randint(0, 16, (tokens, HASHES))
    x = torch.randn(*token_shape, HC, DIM, dtype=torch.bfloat16)
    mask = None
    if with_mask:
        mask = (torch.arange(tokens) % 2 == 0).view(token_shape)
    with torch.no_grad():
        got = module(x, row_ids, mask)
        want = _reference(module, x, row_ids, mask)
    assert got.shape == x.shape and got.dtype == x.dtype
    torch.testing.assert_close(got, want)
    if mask is not None:
        torch.testing.assert_close(got[~mask], x[~mask])


def test_engram_batch_decode_rows_are_independent():
    module = _module()
    row_ids = torch.randint(0, 16, (2, HASHES))
    x = torch.randn(2, 1, HC, DIM, dtype=torch.bfloat16)
    with torch.no_grad():
        batched = module(x, row_ids)
        singles = torch.cat([module(x[b : b + 1], row_ids[b : b + 1]) for b in range(2)])
    torch.testing.assert_close(batched, singles)


def test_engram_rejects_row_count_mismatch():
    module = _module()
    x = torch.randn(2, 1, HC, DIM, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        module(x, torch.randint(0, 16, (3, HASHES)))
