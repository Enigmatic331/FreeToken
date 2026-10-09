"""CSA2-S sparse consumer logits through the serving ``_index_prefill`` path.

The host tests replace the Triton primitives with references.  Integer
queries/keys and power-of-two weights make every logit exactly representable,
so dense and gathered references agree bitwise regardless of summation order
and produce heavy score ties.  They prove the integration (the source's id
publication, ``-inf`` fill, tiling, fallbacks, and a bitwise-identical
selector input), not the kernel's GPU parity; the CUDA tests at the end are
that gate and need an idle SM120.
"""
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.dsv4_indexer import IndexerBackendMixin
from freetoken.kernel.triton.dsv4.indexer import indexer_gathered_logits
from freetoken.models.deepseek_v41 import model
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.indexer import (
    expand_candidate_block_mask,
    select_candidate_block_mask,
    select_index_topk,
    sparse_consumer_logits_enabled,
    warmup_sparse_consumer,
)

TOPK, TOPK_BLOCKS, BLOCK = 512, 2048, 8
CONSUMERS = (24, 28, 32, 36)
FIRST, MIDDLE, FINAL = 16385, 262147, 524288  # n_blocks just past the dense cutover


@pytest.fixture
def warned(monkeypatch):
    """FreeToken loggers do not propagate, so record warnings directly."""
    messages = []
    monkeypatch.setattr(model.logger, "warning", lambda msg, *a: messages.append(msg % a))
    return messages


def logits_ref(q, keys, weights):
    score = torch.einsum("shd,td->sht", q.double(), keys.double()).relu()
    return (score * weights.double().unsqueeze(-1)).sum(1).float()


class ReferenceBackend:
    """Dense reference; optionally the gathered reference computing only candidates."""

    def __init__(self, keys, *, gathered=True, from_dense=False):
        self.keys = keys
        self.calls = 0
        self.from_dense = from_dense
        if gathered:
            self.indexer_prefill_gathered_logits = self._gathered

    def indexer_keys(self, table_idx, n_blocks, ratio, layer_id, bsz):
        assert n_blocks == self.keys.shape[0]
        return self.keys.unsqueeze(0)

    def indexer_prefill_logits(self, q, keys, weights):
        return logits_ref(q[0], keys[0], weights[0]).unsqueeze(0)

    def _gathered(self, q, keys, weights, ids, block_size, out):
        self.calls += 1
        assert ids.dtype == torch.int32 and out.dtype == torch.float32
        width = keys.shape[0]
        for row in range(q.shape[0]):
            blocks = ids[row][ids[row] >= 0].long()
            cols = (blocks[:, None] * block_size + torch.arange(block_size)).flatten()
            cols = cols[cols < width]
            if self.from_dense:  # random floats: assume an exact primitive
                value = logits_ref(q[row:row + 1], keys, weights[row:row + 1])[0, cols]
            else:
                value = logits_ref(q[row:row + 1], keys[cols], weights[row:row + 1])[0]
            out[row, cols] = value
        return out

    @staticmethod
    def blocks_to_global(selected, *args, **kwargs):
        return selected


def inputs(rows, n_blocks, *, seed, heads=2, dim=8, kind="exact"):
    gen = torch.Generator().manual_seed(seed)
    if kind == "exact":
        q = torch.randint(-2, 3, (rows, heads, dim), generator=gen)
        sign = torch.randint(0, 2, (rows, heads), generator=gen) * 2 - 1
        w = sign * 2.0 ** torch.randint(-2, 3, (rows, heads), generator=gen)
        keys = torch.randint(-2, 3, (n_blocks, dim), generator=gen)
    elif kind == "zero":  # every live candidate ties at zero
        q = torch.zeros(rows, heads, dim)
        w = torch.ones(rows, heads)
        keys = torch.randint(-2, 3, (n_blocks, dim), generator=gen)
    else:
        q = torch.randn(rows, heads, dim, generator=gen)
        w = torch.randn(rows, heads, generator=gen)
        keys = torch.randn(n_blocks, dim, generator=gen)
    return q.bfloat16(), w.bfloat16(), keys.bfloat16()


def invoke(runtime, monkeypatch, layer_id, q, w, backend, *, ratio=1, tile_rows=129,
           sparse=False, compact=True, packed=False, inputs_seen=None):
    """One ``_index_prefill``; ``inputs_seen`` collects each tile's final selector input."""
    monkeypatch.setattr(model, "shared_attention", runtime)
    rows, n_blocks = q.shape[0], backend.keys.shape[0]
    start = n_blocks * ratio - rows
    source = layer_id == 20

    def select(x_, live, **kw):
        picked = select_index_topk(x_, live, TOPK, **kw)
        if inputs_seen is not None:
            assert kw.get("inplace")  # so x_ now holds exactly what topk consumed
            inputs_seen.append(x_.clone())
        return picked

    indexer = SimpleNamespace(
        prefill_max_logits_bytes=tile_rows * n_blocks * 4,
        compact_prefill_candidates=compact, bitpacked_prefill_candidates=packed,
        sparse_consumer_logits=sparse and not source, publish_candidate_ids=sparse and source,
        index_topk=TOPK, candidate_topk_blocks=TOPK_BLOCKS, candidate_block_size=BLOCK,
        queries=lambda q_, freqs: q_, head_weights=lambda x_: w, select=select,
    )
    attention = SimpleNamespace(
        plan=SimpleNamespace(owns_index=True, compress_ratio=ratio,
                             is_candidate_source=source, uses_candidates=not source),
        indexer=indexer, layer_id=layer_id, attn=backend,
        freqs_cis=torch.arange(start + rows, device=q.device),
    )
    attention._sparse_consumer_ids = partial(model.Attention._sparse_consumer_ids, attention)
    x = torch.zeros(1, rows, 1, device=q.device)
    return model.Attention._index_prefill(attention, x, q.unsqueeze(0), start, 0)[0]


def ids_of(mask):
    """Test-only: a hand-built publication's ids, ``-1`` padded (any order is valid)."""
    ids = torch.full((mask.shape[0], TOPK_BLOCKS), -1, dtype=torch.int32)
    for row in range(mask.shape[0]):
        found = mask[row].nonzero().flatten()
        assert found.numel() <= TOPK_BLOCKS
        ids[row, :found.numel()] = found.flip(0).int()  # deliberately not ascending
    return ids


def assert_ids_match_mask(ids, mask):
    assert ids.dtype == torch.int32 and ids.shape == (mask.shape[0], TOPK_BLOCKS)
    valid = ids >= 0
    assert ((ids < mask.shape[-1]) | ~valid).all()
    width = mask.shape[-1]
    rebuilt = torch.zeros(mask.shape[0], width + 1, dtype=torch.bool, device=mask.device)
    rebuilt.scatter_(-1, torch.where(valid, ids, width).long(), True)  # pads hit column width
    assert torch.equal(rebuilt[:, :width], mask)
    assert torch.equal(valid.sum(-1), mask.sum(-1))  # no duplicate ids


def assert_bits(a, b):
    assert len(a) == len(b) and all(x.dtype == torch.float32 for x in (*a, *b))
    for x, y in zip(a, b):
        assert torch.equal(x.view(torch.int32), y.view(torch.int32))


def run_pair(monkeypatch, rows, n_blocks, *, seed, kind="exact", ratio=1, tile_rows=129,
             publication=None):
    """Source then all four consumers, OFF and ON, on separate runtimes."""
    q20, w20, keys = inputs(rows, n_blocks // 1, seed=seed, kind=kind)
    off_rt, on_rt = model.SharedAttentionRuntime(), model.SharedAttentionRuntime()
    dense = ReferenceBackend(keys, gathered=False)
    gathered = ReferenceBackend(keys, from_dense=kind == "random")
    for runtime, backend, sparse in ((off_rt, dense, False), (on_rt, gathered, True)):
        runtime.reset()
        invoke(runtime, monkeypatch, 20, q20, w20, backend, ratio=ratio, tile_rows=tile_rows,
               sparse=sparse)
        if publication is not None:
            runtime.candidates.prefill_block_mask = publication.clone()
            if sparse:
                runtime.candidates.prefill_block_ids = ids_of(publication)
    assert torch.equal(off_rt.candidates.prefill_block_mask, on_rt.candidates.prefill_block_mask)
    assert off_rt.candidates.prefill_block_ids is None
    assert_ids_match_mask(on_rt.candidates.prefill_block_ids, on_rt.candidates.prefill_block_mask)
    assert gathered.calls == 0  # the source itself always scores densely
    for offset, layer_id in enumerate(CONSUMERS):
        q, w, _ = inputs(rows, 1, seed=seed * 100 + offset + 1, kind=kind)
        seen_off, seen_on = [], []
        off = invoke(off_rt, monkeypatch, layer_id, q, w, dense, ratio=ratio,
                     tile_rows=tile_rows, inputs_seen=seen_off)
        calls = gathered.calls
        on = invoke(on_rt, monkeypatch, layer_id, q, w, gathered, ratio=ratio,
                    tile_rows=tile_rows, sparse=True, inputs_seen=seen_on)
        assert on.dtype == off.dtype == torch.int32
        assert_bits(seen_on, seen_off)  # the selector consumed identical bits
        assert torch.equal(on, off), (layer_id, (on != off).sum())
        yield layer_id, gathered.calls - calls, off, off_rt.candidates.prefill_block_mask


@pytest.mark.parametrize("n_blocks,rows,tile_rows", [
    (FIRST, 5, 129), (FIRST + 3, 130, 129), (16392, 259, 129),
    (MIDDLE, 4, 129), (FINAL, 3, 2),
])
@pytest.mark.parametrize("kind", ["exact", "zero", "random"])
def test_all_consumers_select_bitwise_identical_rows(monkeypatch, n_blocks, rows, tile_rows, kind):
    tiles = -(-rows // tile_rows)
    for layer_id, calls, selected, published in run_pair(
            monkeypatch, rows, n_blocks, seed=n_blocks % 997, kind=kind, tile_rows=tile_rows):
        assert calls == tiles, layer_id  # the sparse branch really ran per tile
        assert selected.shape == (rows, TOPK)
        assert published.shape == (rows, -(-n_blocks // BLOCK))


def test_ratio_two_and_adversarial_publications(monkeypatch):
    rows, n_blocks = 6, FIRST + 3
    width = -(-n_blocks // BLOCK)
    gen = torch.Generator().manual_seed(9)
    mask = torch.zeros(rows, width, dtype=torch.bool)
    mask[0] = False                                   # no candidates at all
    mask[1, :TOPK_BLOCKS] = True                      # oldest blocks, exactly full
    mask[2, -TOPK_BLOCKS:] = True                     # newest incl. partial and non-causal
    mask[3, torch.randperm(width, generator=gen)[:TOPK_BLOCKS]] = True
    mask[4, ::width // 7] = True                      # sparse, -1 padded
    mask[5, width - 1] = True                         # only the 3-key partial tail
    for ratio in (1, 2):
        for _, calls, selected, _ in run_pair(monkeypatch, rows, n_blocks, seed=ratio,
                                               ratio=ratio, publication=mask):
            assert calls == 1
            assert (selected[0] == -1).all()
            assert (selected[5] >= 0).sum() <= n_blocks % BLOCK


def test_repeated_and_changing_inputs_reuse_no_stale_state(monkeypatch):
    first = [s.clone() for _, _, s, _ in run_pair(monkeypatch, 7, FIRST, seed=1)]
    again = [s for _, _, s, _ in run_pair(monkeypatch, 7, FIRST, seed=1)]
    changed = [s for _, _, s, _ in run_pair(monkeypatch, 7, FIRST, seed=2)]
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert any(not torch.equal(a, b) for a, b in zip(first, changed))


def _consumer(monkeypatch, rows=4, n_blocks=FIRST, *, ids=None, dtype=torch.bfloat16,
              backend_gathered=True, **kw):
    q, w, keys = inputs(rows, n_blocks, seed=3)
    q, w, keys = q.to(dtype), w.to(dtype), keys.to(dtype)
    runtime = model.SharedAttentionRuntime()
    backend = ReferenceBackend(keys, gathered=backend_gathered)
    invoke(runtime, monkeypatch, 20, q, w, ReferenceBackend(keys, gathered=False), sparse=True)
    if ids is not None:  # rewrite the source's publication
        runtime.candidates.prefill_block_ids = ids(runtime.candidates.prefill_block_ids)
    mask = runtime.candidates.prefill_block_mask
    dense = invoke(runtime, monkeypatch, 24, q, w, ReferenceBackend(keys, gathered=False))
    got = invoke(runtime, monkeypatch, 24, q, w, backend, **kw)
    expected = select_index_topk(
        logits_ref(q, keys, w), (n_blocks - rows + torch.arange(rows) + 1), TOPK,
        candidate_mask=expand_candidate_block_mask(mask, n_blocks, BLOCK))
    assert torch.equal(dense, expected)
    return got, expected, backend


def test_flag_off_never_touches_the_gathered_path(monkeypatch):
    got, expected, backend = _consumer(monkeypatch, sparse=False)
    assert torch.equal(got, expected) and backend.calls == 0


def test_dense_cutover_and_missing_backend_stay_dense(monkeypatch, warned):
    got, expected, backend = _consumer(monkeypatch, n_blocks=16384, sparse=True)
    assert torch.equal(got, expected) and backend.calls == 0  # 2048 blocks: all candidates
    assert model.shared_attention.candidates.prefill_block_ids is None  # nothing published
    got, expected, backend = _consumer(monkeypatch, sparse=True, backend_gathered=False)
    assert torch.equal(got, expected) and backend.calls == 0
    assert warned == ["layer 24 keeps dense consumer logits: backend lacks gathered logits"]


@pytest.mark.parametrize("case,reason", [
    ("fp32", "not contiguous bf16"),
    ("missing", "source published no candidate ids"),
    ("int64", "not contiguous int32 [rows, capacity]"),
    ("narrow", "not contiguous int32 [rows, capacity]"),
    ("strided", "not contiguous int32 [rows, capacity]"),
])
def test_unexpected_inputs_fail_closed_to_dense(monkeypatch, warned, case, reason):
    rewrite = {
        "fp32": None,
        "missing": lambda published: None,
        "int64": lambda published: published.long(),
        "narrow": lambda published: published[:, :-1],
        "strided": lambda published: published.repeat(1, 2)[:, ::2],
    }[case]
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    got, expected, backend = _consumer(monkeypatch, ids=rewrite, dtype=dtype, sparse=True)
    assert torch.equal(got, expected) and backend.calls == 0
    assert len(warned) == 1 and warned[0].startswith("layer 24 keeps dense consumer logits")
    assert reason in warned[0]


def test_consumers_never_read_the_mask_on_the_sparse_path(monkeypatch):
    """The mask stays the dense fallback's input; ON selection depends only on the ids."""
    rows, n_blocks = 5, FIRST + 3
    q, w, keys = inputs(rows, n_blocks, seed=4)
    runtime = model.SharedAttentionRuntime()
    invoke(runtime, monkeypatch, 20, q, w, ReferenceBackend(keys, gathered=False), sparse=True)
    mask = runtime.candidates.prefill_block_mask
    want = invoke(runtime, monkeypatch, 24, q, w, ReferenceBackend(keys), sparse=True)
    runtime.candidates.prefill_block_mask = ~mask  # same shape, wrong bits
    got = invoke(runtime, monkeypatch, 24, q, w, ReferenceBackend(keys), sparse=True)
    assert torch.equal(got, want)


def _source_scores(rows, blocks, *, seed, kind):
    gen = torch.Generator().manual_seed(seed)
    width = blocks * BLOCK - 3                                   # partial final block
    if kind == "ties":
        scores = torch.randint(-2, 3, (rows, width), generator=gen).float()
    elif kind == "zeros":
        scores = torch.zeros(rows, width)
        scores[:, ::5] = -0.0
    else:
        scores = torch.randn(rows, width, generator=gen)
    scores[0, :BLOCK * 3] = torch.nan                             # NaN blocks are never kept
    scores[1, BLOCK:] = -torch.inf                                # one finite block + newest
    return scores, width


@pytest.mark.parametrize("kind", ["random", "ties", "zeros"])
@pytest.mark.parametrize("blocks", [TOPK_BLOCKS + 1, 3 * TOPK_BLOCKS + 5])
def test_published_ids_are_exactly_the_mask_from_the_same_topk(kind, blocks):
    scores, width = _source_scores(4, blocks, seed=blocks, kind=kind)
    live = torch.tensor([[width], [width], [width // 2], [1]])
    want = select_candidate_block_mask(scores.clone(), live, TOPK_BLOCKS, BLOCK)
    ids = torch.full((4, TOPK_BLOCKS), 7, dtype=torch.int32)
    got = select_candidate_block_mask(scores.clone(), live, TOPK_BLOCKS, BLOCK, ids_out=ids)
    assert torch.equal(got, want)  # the mask bits do not depend on publishing ids
    assert_ids_match_mask(ids, want)
    assert not want[0, :3].any() and want[1].sum() == 2


def test_published_ids_reject_a_mismatched_buffer():
    scores, width = _source_scores(2, TOPK_BLOCKS + 1, seed=1, kind="random")
    for ids in (torch.empty(2, TOPK_BLOCKS, dtype=torch.int64),
                torch.empty(2, TOPK_BLOCKS - 1, dtype=torch.int32)):
        with pytest.raises(ValueError, match="candidate ids"):
            select_candidate_block_mask(scores, width, TOPK_BLOCKS, BLOCK, ids_out=ids)


def test_source_flag_off_publishes_no_ids(monkeypatch):
    rows, n_blocks = 4, FIRST + 3
    q, w, keys = inputs(rows, n_blocks, seed=6)
    off = model.SharedAttentionRuntime()
    invoke(off, monkeypatch, 20, q, w, ReferenceBackend(keys, gathered=False))
    assert off.candidates.prefill_block_ids is None
    on = model.SharedAttentionRuntime()
    invoke(on, monkeypatch, 20, q, w, ReferenceBackend(keys, gathered=False), sparse=True)
    assert torch.equal(on.candidates.prefill_block_mask, off.candidates.prefill_block_mask)
    assert on.candidates.prefill_block_ids is not None
    on.reset()
    assert on.candidates.prefill_block_ids is None and on.candidates.prefill_block_mask is None


@pytest.mark.parametrize("flag,expected", [(None, False), ("0", False), ("1", True)])
def test_flag_is_explicit_opt_in(monkeypatch, flag, expected):
    name = "FREETOKEN_DSV41_SPARSE_CONSUMER_LOGITS"
    if flag is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, flag)
    assert sparse_consumer_logits_enabled() is expected


@pytest.mark.parametrize("flag", ["", "true", "2", "on"])
def test_flag_rejects_invalid_values(monkeypatch, flag):
    monkeypatch.setenv("FREETOKEN_DSV41_SPARSE_CONSUMER_LOGITS", flag)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        sparse_consumer_logits_enabled()


def _args():
    return DeepseekV41Args(
        dim=32, head_dim=32, rope_head_dim=16, q_lora_rank=32, index_n_heads=2,
        index_head_dim=32, n_layers=37, compress_ratios=(0,) + (1,) * 36,
        engram_layer_ids=(), engram_num_embeddings=(),
    )


@pytest.mark.parametrize("compact,packed,expected", [
    ("1", "0", True), ("0", "0", False), ("1", "1", False),
])
def test_flag_freezes_at_construction_for_consumers_only(monkeypatch, compact, packed, expected):
    monkeypatch.setenv("FREETOKEN_DSV41_SPARSE_CONSUMER_LOGITS", "1")
    monkeypatch.setenv("FREETOKEN_DSV41_COMPACT_PREFILL_CANDIDATES", compact)
    monkeypatch.setenv("FREETOKEN_DSV41_BITPACKED_PREFILL_CANDIDATES", packed)
    args = _args()
    indexers = {layer_id: model.Indexer(args, layer_id) for layer_id in (2, 20, *CONSUMERS)}
    monkeypatch.setenv("FREETOKEN_DSV41_SPARSE_CONSUMER_LOGITS", "0")
    assert not indexers[2].sparse_consumer_logits
    assert not indexers[20].sparse_consumer_logits
    assert all(indexers[layer_id].sparse_consumer_logits is expected for layer_id in CONSUMERS)
    assert indexers[20].publish_candidate_ids is expected
    assert not any(indexers[layer_id].publish_candidate_ids for layer_id in (2, *CONSUMERS))


def test_default_construction_is_off(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DSV41_SPARSE_CONSUMER_LOGITS", raising=False)
    monkeypatch.setenv("FREETOKEN_DSV41_COMPACT_PREFILL_CANDIDATES", "1")
    assert not any(model.Indexer(_args(), layer_id).sparse_consumer_logits
                   for layer_id in CONSUMERS)
    assert not model.Indexer(_args(), 20).publish_candidate_ids


def test_warmup_is_a_host_noop():
    assert warmup_sparse_consumer(torch.device("cpu"), heads=2, dim=8) is None


def _kernel_inputs(rows=2, width=17, heads=2, dim=8, cands=3, device="cpu"):
    return (torch.zeros(rows, heads, dim, dtype=torch.bfloat16, device=device),
            torch.zeros(width, dim, dtype=torch.bfloat16, device=device),
            torch.zeros(rows, heads, dtype=torch.bfloat16, device=device),
            torch.zeros(rows, cands, dtype=torch.int32, device=device))


@pytest.mark.parametrize("case", [
    "ids_int64", "ids_rows", "q_ndim", "k_dim", "w_shape", "block3", "block0",
    "out_dtype", "out_shape", "dim_pow2",
])
def test_gathered_wrapper_guards_reject_before_launch(case):
    q, k, w, ids = _kernel_inputs()
    out, block = None, 8
    if case == "ids_int64":
        ids = ids.long()
    elif case == "ids_rows":
        ids = ids[:1]
    elif case == "q_ndim":
        q = q[None]
    elif case == "k_dim":
        k = k[:, :4]
    elif case == "w_shape":
        w = w[:1]
    elif case == "block3":
        block = 3
    elif case == "block0":
        block = 0
    elif case == "out_dtype":
        out = torch.zeros(2, 17, dtype=torch.bfloat16)
    elif case == "out_shape":
        out = torch.zeros(2, 16)
    else:
        q, k = q[..., :6], k[:, :6]
    with pytest.raises(ValueError):
        indexer_gathered_logits(q, k, w, ids, block, out=out)


def test_gathered_wrapper_rejects_int32_overflowing_tiles():
    q, k, w, ids = _kernel_inputs(rows=8192, width=2**19, device="meta")
    with pytest.raises(ValueError, match="int32"):
        indexer_gathered_logits(q, k, w, ids, 8, out=torch.empty(
            8192, 2**19, device="meta"))


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="needs CUDA")
def test_gathered_wrapper_rejects_mixed_devices():
    q, k, w, ids = _kernel_inputs()
    with pytest.raises(ValueError, match="span devices"):
        indexer_gathered_logits(q, k.cuda(), w, ids, 8)


# --- SM120 authority gate: real kernels, bitwise ------------------------------

class KernelBackend(IndexerBackendMixin):
    def __init__(self, keys):
        self.keys = keys

    def indexer_keys(self, table_idx, n_blocks, ratio, layer_id, bsz):
        return self.keys.unsqueeze(0)

    @staticmethod
    def blocks_to_global(selected, *args, **kwargs):
        return selected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an idle CUDA GPU")
@pytest.mark.parametrize("n_blocks,rows,tile_rows", [
    (FIRST, 129, 129), (FIRST + 7, 130, 129), (65537, 258, 258), (MIDDLE, 129, 129),
    (520000, 129, 129), (FINAL, 2, 1),
])
def test_cuda_real_kernels_select_bitwise_identical_rows(monkeypatch, n_blocks, rows, tile_rows):
    gen = torch.Generator(device="cuda").manual_seed(n_blocks)
    keys = torch.randn(n_blocks, 128, device="cuda", generator=gen).bfloat16()
    off_rt, on_rt = model.SharedAttentionRuntime(), model.SharedAttentionRuntime()
    backend = KernelBackend(keys)
    for layer_id in (20, *CONSUMERS):
        q = torch.randn(rows, 32, 128, device="cuda", generator=gen).bfloat16()
        w = torch.randn(rows, 32, device="cuda", generator=gen).bfloat16()
        if layer_id == 24:
            w[: rows // 2] = 0  # zero-score ties across every candidate
        seen_off, seen_on = [], []
        off = invoke(off_rt, monkeypatch, layer_id, q, w, backend, tile_rows=tile_rows,
                     inputs_seen=seen_off)
        on = invoke(on_rt, monkeypatch, layer_id, q, w, backend, tile_rows=tile_rows,
                    sparse=True, inputs_seen=seen_on)
        assert_bits(seen_on, seen_off)
        assert torch.equal(on, off), (layer_id, (on != off).sum().item())
        if layer_id == 20:
            assert torch.equal(on_rt.candidates.prefill_block_mask,
                               off_rt.candidates.prefill_block_mask)
            assert_ids_match_mask(on_rt.candidates.prefill_block_ids.cpu(),
                                  on_rt.candidates.prefill_block_mask.cpu())
