from __future__ import annotations

import pytest
import torch

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.engram_layer import Engram


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class DeviceTable:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.prefetched = None

    def lookup(self, row_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self.weight.index_select(0, row_ids.flatten()).view(*row_ids.shape, -1)

    def prefetch(self, row_ids: torch.Tensor) -> None:
        self.prefetched = row_ids


def test_engram_layer_matches_explicit_gate_and_masks_non_text_tokens():
    args = DeepseekV41Args(
        dim=64,
        n_layers=2,
        n_heads=4,
        compress_ratios=(0, 0),
        engram_layer_ids=(1,),
        engram_num_embeddings=(16,),
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=32,
    )
    module = Engram(args, 1).cuda()
    torch.manual_seed(41)
    module.wkv.weight.data.copy_(
        (torch.randn_like(module.wkv.weight, dtype=torch.float32) * 0.5).to(
            torch.float8_e4m3fn
        )
    )
    module.wkv.scale.data.view(torch.uint8).fill_(127)
    module.q_weight.data.copy_(torch.randn_like(module.q_weight.float()).bfloat16())
    module.k_weight.data.copy_(torch.randn_like(module.k_weight.float()).bfloat16())
    table = DeviceTable(torch.randn(16, 32, device="cuda", dtype=torch.bfloat16))
    module.attach_table(table)

    row_ids = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], device="cuda"
    )
    hidden = torch.randn(3, 4, 64, device="cuda", dtype=torch.bfloat16)
    token_mask = torch.tensor([True, False, True], device="cuda")
    module.prefetch(row_ids)
    assert table.prefetched is row_ids
    got = module(hidden, row_ids, token_mask)

    rows = table.lookup(row_ids).flatten(-2)
    key, value = module.wkv(rows).split([4 * 64, 64], dim=-1)
    key = key.float().view(3, 4, 64)
    h = hidden.float()
    product = module.q_weight.float() * module.k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + args.norm_eps)
    rstd *= torch.rsqrt(key.square().mean(-1) + args.norm_eps)
    dot = (h * product * key).sum(-1) * rstd * 64**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    gate[~token_mask] = 0
    want = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).bfloat16()
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got[1], hidden[1])
