"""Gathered Lightning-Indexer candidate logits against the dense kernel.

On CUDA every comparison is bitwise; on the SM120 authority that is the
CSA2-S qualification gate.  Under ``TRITON_INTERPRET=1`` on the host the
kernels run through numpy, whose matmul/sum rounding depends on tile shape and
column position, so those runs use exactly representable inputs (integer
Q/K, power-of-two weights: every summation order yields the same bits) and
check random floats only to a tolerance.  The interpreter also mishandles
bf16, so it uses fp16/fp32 inputs.
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.kernel.triton.dsv4.indexer import indexer_gathered_logits, indexer_logits

CUDA = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(
    not (CUDA or INTERPRET), reason="needs CUDA or TRITON_INTERPRET=1"
)
DEVICE = "cuda" if CUDA else "cpu"
DTYPE = torch.bfloat16 if CUDA else torch.float16


def _exact(rows, width, *, heads=32, dim=128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randint(-2, 3, (rows, heads, dim), generator=gen)
    keys = torch.randint(-2, 3, (width, dim), generator=gen)
    sign = torch.randint(0, 2, (rows, heads), generator=gen) * 2 - 1
    w = sign * 2.0 ** torch.randint(-2, 3, (rows, heads), generator=gen)
    return (q.to(DEVICE, DTYPE), keys.to(DEVICE, DTYPE), w.to(DEVICE, DTYPE))


def _dense(q, keys, w):
    return indexer_logits(q[None], keys[None], w[None])[0]


def _columns(blocks, width, block_size):
    cols = torch.zeros(blocks.shape[0], width, dtype=torch.bool)
    for row, ids in enumerate(blocks.tolist()):
        for block in ids:
            if 0 <= block * block_size < width:
                cols[row, block * block_size:(block + 1) * block_size] = True
    return cols.to(blocks.device)


def _blocks(rows, width, block_size, cands, *, seed):
    gen = torch.Generator().manual_seed(seed)
    n = -(-width // block_size)
    out = []
    for row in range(rows):
        ids = torch.randperm(n, generator=gen)[:cands]
        ids = torch.cat([ids, torch.full((max(0, cands - n),), -1)])
        out.append(ids)
    blocks = torch.stack(out).int()
    blocks[0, : min(3, cands)] = n - 1            # duplicate final (partial) block
    if rows > 1:
        blocks[1, ::3] = -1                        # padding in the middle
        blocks[1, -1] = n + 5                      # out of range: must not store
    if rows > 2:
        blocks[2, 0] = -(2**31)                    # most negative id
        if cands > 1:
            blocks[2, 1] = 2**31 - 1               # would wrap if scaled before the check
    return blocks.to(DEVICE)


def _assert_bits(got, expected):
    assert got.dtype == expected.dtype == torch.float32
    assert torch.equal(got.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("width", [1, 7, 8, 9, 127, 128, 129, 1000])
@pytest.mark.parametrize("block_size", [8, 4, 16])
@pytest.mark.parametrize("cands", [1, 16, 17, 40])
def test_exact_inputs_store_dense_bits_only_at_candidates(width, block_size, cands):
    rows = 3
    q, keys, w = _exact(rows, width, seed=width + cands)
    blocks = _blocks(rows, width, block_size, cands, seed=block_size)
    sentinel = torch.full((rows, width), torch.nan, device=DEVICE)
    got = indexer_gathered_logits(q, keys, w, blocks, block_size, out=sentinel.clone())
    cols = _columns(blocks, width, block_size)
    expected = torch.where(cols, _dense(q, keys, w), sentinel)
    _assert_bits(got, expected)
    fresh = indexer_gathered_logits(q, keys, w, blocks, block_size)
    _assert_bits(fresh, expected.masked_fill(~cols, -torch.inf))


def test_every_block_in_any_order_reproduces_the_dense_tile():
    width, block_size = 1027, 8
    q, keys, w = _exact(2, width, seed=5)
    n = -(-width // block_size)
    gen = torch.Generator().manual_seed(6)
    blocks = torch.stack([torch.randperm(n, generator=gen) for _ in range(2)]).int()
    got = indexer_gathered_logits(q, keys, w, blocks.to(DEVICE), block_size)
    _assert_bits(got, _dense(q, keys, w))


def test_zero_scores_tie_and_empty_candidates_store_nothing():
    q, keys, w = _exact(2, 300, seed=7)
    q.zero_()
    blocks = torch.tensor([[0, 5, 37], [-1, -1, -1]], dtype=torch.int32, device=DEVICE)
    got = indexer_gathered_logits(q, keys, w, blocks, 8)
    cols = _columns(blocks, 300, 8)
    _assert_bits(got, _dense(q, keys, w).masked_fill(~cols, -torch.inf))  # incl. zero signs
    assert (got[cols] == 0).all() and not cols[1].any()


def test_repeated_calls_and_changing_inputs_are_stable():
    q, keys, w = _exact(2, 513, seed=8)
    blocks = _blocks(2, 513, 8, 20, seed=9)
    first = indexer_gathered_logits(q, keys, w, blocks, 8)
    _assert_bits(indexer_gathered_logits(q, keys, w, blocks, 8), first)
    q2, keys2, w2 = _exact(2, 513, seed=10)
    changed = indexer_gathered_logits(q2, keys2, w2, blocks, 8)
    cols = _columns(blocks, 513, 8)
    _assert_bits(changed, _dense(q2, keys2, w2).masked_fill(~cols, -torch.inf))


@pytest.mark.skipif(CUDA, reason="interpreter numerics only")
def test_interpreter_random_floats_match_to_tolerance():
    gen = torch.Generator().manual_seed(11)
    q = torch.randn(3, 32, 128, generator=gen)
    keys = torch.randn(700, 128, generator=gen)
    w = torch.randn(3, 32, generator=gen)
    blocks = _blocks(3, 700, 8, 30, seed=12)
    got = indexer_gathered_logits(q, keys, w, blocks, 8)
    cols = _columns(blocks, 700, 8)
    torch.testing.assert_close(got[cols], _dense(q, keys, w)[cols], atol=1e-3, rtol=1e-4)
    assert torch.isneginf(got[~cols]).all()


# --- SM120 authority gate: random bf16, bitwise --------------------------------

@pytest.mark.skipif(not CUDA, reason="needs an idle CUDA GPU")
@pytest.mark.parametrize("rows,width", [
    (1, 16385), (129, 16385), (129, 65543), (129, 262147), (129, 520000), (2, 524288),
])
@pytest.mark.parametrize("pattern", ["random", "newest", "clustered", "duplicates", "published"])
def test_cuda_random_bf16_candidate_logits_are_bitwise_dense(rows, width, pattern):
    gen = torch.Generator(device="cuda").manual_seed(width + rows)
    q = torch.randn(rows, 32, 128, device="cuda", generator=gen).bfloat16()
    keys = torch.randn(width, 128, device="cuda", generator=gen).bfloat16()
    w = torch.randn(rows, 32, device="cuda", generator=gen).bfloat16()
    n = -(-width // 8)
    if pattern == "random":
        blocks = torch.rand(rows, n, device="cuda", generator=gen).argsort(-1)[:, :2048]
    elif pattern == "newest":
        blocks = torch.arange(n - 2048, n, device="cuda").expand(rows, -1)
    elif pattern == "clustered":
        start = torch.randint(0, n - 2048, (rows, 1), device="cuda", generator=gen)
        blocks = start + torch.arange(2048, device="cuda")
    elif pattern == "duplicates":
        blocks = torch.randint(0, n, (rows, 2048), device="cuda", generator=gen)
    else:  # the serving source's own publication: score order, never ascending
        from freetoken.models.deepseek_v41.indexer import select_candidate_block_mask

        blocks = torch.empty(rows, 2048, dtype=torch.int32, device="cuda")
        select_candidate_block_mask(_dense(q, keys, w), width, 2048, 8, ids_out=blocks)
    blocks = blocks.int().contiguous()
    got = indexer_gathered_logits(q, keys, w, blocks, 8)
    cols = _columns(blocks.cpu(), width, 8).cuda()
    _assert_bits(got, _dense(q, keys, w).masked_fill(~cols, -torch.inf))
