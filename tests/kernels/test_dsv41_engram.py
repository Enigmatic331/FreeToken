from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _fixture(rows: int, dim: int, device):
    values = ((torch.arange(rows * dim, dtype=torch.float32).reshape(rows, dim) % 15) - 7) / 2
    weight = values.to(torch.float8_e4m3fn).to(device)
    codes = (torch.arange(rows * (dim // 32), dtype=torch.uint8).reshape(rows, dim // 32) % 5) + 125
    scale = codes.contiguous().view(torch.float8_e8m0fnu).to(device)
    return weight, scale


def _reference(weight, scale, ids, row_lo, row_hi):
    out = torch.zeros(ids.numel(), weight.shape[1], dtype=torch.bfloat16, device=ids.device)
    owned = (ids >= row_lo) & (ids < row_hi)
    if owned.any():
        local = ids[owned] - row_lo
        values = weight[local].float().unflatten(-1, (-1, 32))
        factors = scale[local].view(torch.uint8).float().sub(127).exp2().unsqueeze(-1)
        out[owned] = (values * factors).flatten(-2).to(torch.bfloat16)
    return out


def test_device_table_gather_matches_torch_and_zeroes_unowned_rows():
    from freetoken.kernel.triton.dsv41 import engram_gather_rows

    rows, dim, row_lo = 9, 256, 17
    weight, scale = _fixture(rows, dim, "cuda")
    ids = torch.tensor([16, 17, 20, 25, 26, -1], device="cuda", dtype=torch.int64)
    out = torch.empty(ids.numel(), dim, device="cuda", dtype=torch.bfloat16)
    engram_gather_rows(
        weight.data_ptr(),
        scale.data_ptr(),
        ids,
        out,
        dim=dim,
        row_lo=row_lo,
        row_hi=row_lo + rows,
    )
    assert torch.equal(out, _reference(weight, scale, ids, row_lo, row_lo + rows))


def test_pinned_host_shard_matches_device_oracle():
    from freetoken.models.deepseek_v41.engram import EngramHostTable, EngramShardPlan

    num_rows, dim = 9, 256
    plan = EngramShardPlan.build(num_rows, dim, rank=1, world_size=2)
    table = EngramHostTable(plan, device=torch.device("cuda"), prefetch=False)
    full_weight, full_scale = _fixture(num_rows, dim, "cpu")
    table.weight.copy_(full_weight[plan.row_start : plan.row_end])
    table.scale.copy_(full_scale[plan.row_start : plan.row_end])
    table.finish_load(pin=True, collapse=False)

    ids = torch.tensor([0, plan.row_start, num_rows - 1, num_rows], device="cuda")
    got = table.lookup(ids, reduce=False)
    want = _reference(
        full_weight[plan.row_start : plan.row_end].to("cuda"),
        full_scale[plan.row_start : plan.row_end].to("cuda"),
        ids,
        plan.row_start,
        plan.row_end,
    )
    assert torch.equal(got, want)
