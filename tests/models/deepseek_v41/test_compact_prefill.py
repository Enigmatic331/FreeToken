"""Exercise the serving prefill publication path with synthetic logits/pages.

This covers CPU and CUDA mask/top-k operations, not the real indexer-logits
kernel or full-model equivalence; those remain separate qualification gates.
"""
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.deepseek_v41 import model
from freetoken.models.deepseek_v41.indexer import (
    expand_candidate_block_mask,
    select_candidate_blocks,
    select_index_topk,
)


def invoke(scores, ratio, tile, source, compact, *, owns_index=True):
    rows, width = scores.shape
    start = width * ratio - rows
    topk = min(512, width)
    topk_blocks = min(2048, max(1, width // 32))
    indexer = SimpleNamespace(
        prefill_max_logits_bytes=tile * width * 4,
        compact_prefill_candidates=compact,
        index_topk=topk, candidate_topk_blocks=topk_blocks, candidate_block_size=8,
        queries=lambda q, f: q, head_weights=lambda x: torch.zeros_like(x),
        select_candidate_mask=lambda x, live: select_candidate_blocks(x, live, topk_blocks, 8),
        select=lambda x, live, **kw: select_index_topk(x, live, topk, **kw),
    )
    backend = SimpleNamespace(
        indexer_keys=lambda *a: torch.zeros(1, width, 1, device=scores.device),
        indexer_prefill_logits=lambda q, k, w: scores.index_select(
            0, q[0, :, 0].long()).unsqueeze(0),
        blocks_to_global=lambda selected, *a, **kw: selected,
    )
    attention = SimpleNamespace(
        plan=SimpleNamespace(owns_index=owns_index, compress_ratio=ratio,
                             is_candidate_source=source, uses_candidates=not source),
        indexer=indexer, layer_id=20 if source else 24, attn=backend,
        freqs_cis=torch.arange(width * ratio, device=scores.device),
    )
    return model.Attention._index_prefill(
        attention, torch.zeros(1, rows, 1, device=scores.device),
        torch.arange(rows, device=scores.device).reshape(1, rows, 1), start, 0)


def reference_mask(scores, lengths):
    """Frozen pre-experiment algorithm, independent of the new helper."""
    width = scores.shape[-1]
    values = torch.nn.functional.pad(scores, (0, -width % 8), value=-torch.inf)
    values = values.unflatten(-1, (-1, 8)).amax(-1)
    newest = (lengths[:, None] - 1) // 8
    values = values.masked_fill(
        torch.arange(values.shape[-1], device=scores.device) == newest, torch.inf)
    top = values.topk(min(2048, max(1, width // 32), values.shape[-1]), dim=-1)
    keep = torch.zeros_like(values, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(8, -1)[..., :width]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("width,rows,tile", [(9, 5, 1), (65, 7, 3), (520000, 5, 2)])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tied", [False, True])
def test_serving_prefill_source_consumers_reuse_and_reset(monkeypatch, device, width, rows, tile, ratio, tied):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(4105)
    publications = [model.SharedAttentionRuntime(), model.SharedAttentionRuntime()]
    for _ in range(2):
        for runtime in publications:
            runtime.reset()
        mask = None
        for source in (True, False, False):
            scores = torch.randn(rows, width, device=device)
            if tied:
                scores = scores.round()
            live = (width * ratio - rows + torch.arange(rows, device=device) + 1) // ratio
            scores.masked_fill_(torch.arange(width, device=device)[None, :] >= live[:, None], -torch.inf)
            if source:
                mask = reference_mask(scores, live)
            expected = select_index_topk(scores, live, min(512, width), candidate_mask=mask)
            for compact, runtime in zip((False, True), publications):
                monkeypatch.setattr(model, "shared_attention", runtime)
                got = invoke(scores, ratio, tile, source, compact)
                assert torch.equal(got[0], expected)
                assert invoke(scores, ratio, tile, False, compact, owns_index=False) is got
                if source:
                    if compact:
                        assert runtime.candidates.mask is None
                        assert torch.equal(expand_candidate_block_mask(
                            runtime.candidates.prefill_block_mask, width, 8), mask)
                    else:
                        assert runtime.candidates.prefill_block_mask is None
                        assert torch.equal(runtime.candidates.mask, mask)


@pytest.mark.parametrize("compact", [False, True])
def test_missing_source_and_wrong_shape_fail_closed(monkeypatch, compact):
    runtime = model.SharedAttentionRuntime()
    monkeypatch.setattr(model, "shared_attention", runtime)
    scores = torch.randn(5, 65)
    with pytest.raises(RuntimeError, match="before the source"):
        invoke(scores, 1, 2, False, compact)
    wrong = torch.ones(5, 3, dtype=torch.bool)
    if compact:
        runtime.candidates.prefill_block_mask = wrong
    else:
        runtime.candidates.mask = wrong
    with pytest.raises(ValueError, match="expected publication"):
        invoke(scores, 1, 2, False, compact)


def test_empty_compressed_prefix_preserves_empty_output(monkeypatch):
    runtime = model.SharedAttentionRuntime()
    monkeypatch.setattr(model, "shared_attention", runtime)
    attention = SimpleNamespace(
        plan=SimpleNamespace(owns_index=True, compress_ratio=2),
        indexer=object(), attn=SimpleNamespace(blocks_to_global=lambda x, *a, **k: x))
    got = model.Attention._index_prefill(attention, torch.zeros(1, 1, 1), None, 0, 0)
    assert got.shape == (1, 1, 0)
    assert runtime.candidates.mask is runtime.candidates.prefill_block_mask is None
