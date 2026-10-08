from __future__ import annotations

import pytest
import torch

from freetoken.models.deepseek_v41.indexer import (
    CandidateRuntime,
    compact_prefill_candidates_enabled,
    expand_candidate_block_mask,
    indexer_prefill_chunk_rows,
    indexer_prefill_max_logits_bytes,
    select_candidate_block_mask,
    select_candidate_blocks,
    select_index_topk,
    visible_compressed_lengths,
)


def test_prefill_chunk_rows_caps_fp32_logits_and_handles_indivisible_row():
    mib = 1024 * 1024
    assert indexer_prefill_chunk_rows(4096, 4096, 512 * mib) == 4096
    assert indexer_prefill_chunk_rows(32768, 32768, 512 * mib) == 4096
    assert indexer_prefill_chunk_rows(9, 1_000_000, 1 * mib) == 1
    assert indexer_prefill_chunk_rows(0, 4096, 1 * mib) == 0
    assert indexer_prefill_chunk_rows(9, 0, 1 * mib) == 9


def test_prefill_logits_cap_environment(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DSV41_INDEXER_MAX_LOGITS_MB", "64")
    assert indexer_prefill_max_logits_bytes() == 64 * 1024 * 1024


def _candidate_reference(logits, lengths, topk_blocks, block_size):
    width = logits.shape[-1]
    padded = torch.nn.functional.pad(
        logits, (0, -width % block_size), value=-torch.inf
    )
    scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
    newest = (lengths - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(scores.shape[-1], device=logits.device) == newest, torch.inf
    )
    top = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, -1)[..., :width]


def test_candidate_blocks_match_official_algorithm_and_pin_partial_tail():
    logits = torch.tensor(
        [
            [9.0, 8.0, 7.0, 6.0, 1.0, 1.0, -torch.inf, -torch.inf],
            [9.0, 8.0, 7.0, 6.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    lengths = torch.tensor([[6], [8]])
    got = select_candidate_blocks(logits, lengths, topk_blocks=1, block_size=4)
    expected = _candidate_reference(logits, lengths, 1, 4)
    assert torch.equal(got, expected)
    assert got[0, 4:6].all()  # newest partial block is pinned despite lower scores
    assert got[1, 4:8].all()


def test_hierarchical_topk_consumes_source_mask_and_sorts_by_position():
    source = torch.tensor([[9.0, 8.0, 1.0, 0.0, 7.0, 6.0, 3.0, 2.0]])
    consumer = torch.tensor([[1.0, 2.0, 99.0, 98.0, 3.0, 4.0, 97.0, 96.0]])
    runtime = CandidateRuntime()
    mask = runtime.publish(source, 8, topk_blocks=2, block_size=2)
    masked = runtime.consume(consumer)
    assert torch.equal(masked == -torch.inf, ~mask)
    selected = select_index_topk(masked, 8, 6, candidate_mask=mask, offset=128)
    assert selected.dtype == torch.int32
    assert selected.tolist() == [[128, 129, 134, 135, -1, -1]]


def test_topk_respects_ratio_visibility_and_pads_unreachable_columns():
    positions = torch.arange(6)
    assert visible_compressed_lengths(positions, 2).tolist() == [0, 1, 1, 2, 2, 3]
    scores = torch.arange(18, dtype=torch.float32).view(6, 3)
    got = select_index_topk(scores, visible_compressed_lengths(positions, 2), 3)
    assert got.tolist() == [
        [-1, -1, -1],
        [0, -1, -1],
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, -1],
        [0, 1, 2],
    ]


def test_topk_inplace_matches_default_and_reuses_disposable_logits():
    logits = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]])
    original = logits.clone()
    candidate_mask = torch.tensor([[True, False, True, True, True, True]])
    expected = select_index_topk(
        logits,
        4,
        4,
        candidate_mask=candidate_mask,
    )
    assert torch.equal(logits, original)

    got = select_index_topk(
        logits,
        4,
        4,
        candidate_mask=candidate_mask,
        inplace=True,
    )
    assert torch.equal(got, expected)
    assert torch.equal(
        torch.isneginf(logits),
        torch.tensor([[False, True, False, False, True, True]]),
    )


def test_candidate_consumer_requires_source_publication():
    runtime = CandidateRuntime()
    try:
        runtime.consume(torch.zeros(1, 4))
    except RuntimeError as exc:
        assert "before the source" in str(exc)
    else:
        raise AssertionError("missing publication must fail closed")


@pytest.mark.parametrize("flag,expected", [(None, False), ("0", False), ("1", True)])
def test_compact_prefill_is_explicit_opt_in(monkeypatch, flag, expected):
    name = "FREETOKEN_DSV41_COMPACT_PREFILL_CANDIDATES"
    if flag is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, flag)
    assert compact_prefill_candidates_enabled() is expected


@pytest.mark.parametrize("flag", ["", "true", "2", "-1"])
def test_compact_prefill_rejects_invalid_flag(monkeypatch, flag):
    monkeypatch.setenv("FREETOKEN_DSV41_COMPACT_PREFILL_CANDIDATES", flag)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        compact_prefill_candidates_enabled()


@pytest.mark.parametrize("width", [0, 1, 7, 8, 9, 31, 32, 65])
@pytest.mark.parametrize("block_size", [1, 2, 8])
@pytest.mark.parametrize("topk_blocks", [1, 2048])
def test_compact_mask_preserves_every_expanded_bit(width, block_size, topk_blocks):
    torch.manual_seed(41)
    logits = torch.randn(3, width)
    lengths = torch.tensor([[0], [max(0, width - 3)], [width]])
    logits.masked_fill_(torch.arange(width)[None, :] >= lengths, -torch.inf)
    keep = select_candidate_block_mask(logits, lengths, topk_blocks, block_size)
    assert keep.shape == (3, (width + block_size - 1) // block_size)
    expected = (_candidate_reference(logits, lengths, topk_blocks, block_size)
                if width else torch.zeros_like(logits, dtype=torch.bool))
    assert torch.equal(expand_candidate_block_mask(keep, width, block_size), expected)
    assert torch.equal(select_candidate_blocks(logits, lengths, topk_blocks, block_size), expected)


def test_native_prefill_publication_shape_and_bounded_expansion():
    native = torch.empty(8192, 520000, device="meta")
    keep = select_candidate_block_mask(native, 520000, 2048, 8)
    assert keep.shape == (8192, 65000) and keep.dtype == torch.bool
    assert keep.numel() == 532480000
    assert native.numel() - keep.numel() == 3727360000
    assert indexer_prefill_chunk_rows(8192, 520000, 256 * 1024 * 1024) == 129
    assert 129 * 520000 < 64 * 1024 * 1024


def test_reset_clears_both_publications_and_preserves_decode_interface():
    runtime = CandidateRuntime()
    runtime.mask = torch.ones(1, 8, dtype=torch.bool)
    runtime.prefill_block_mask = torch.ones(1, 1, dtype=torch.bool)
    runtime.reset()
    assert runtime.mask is runtime.prefill_block_mask is None
    logits = torch.randn(1, 8)
    mask = runtime.publish(logits, 8, 1, 8)
    assert mask.shape == logits.shape and runtime.prefill_block_mask is None
    assert torch.equal(runtime.consume(logits), logits.masked_fill(~mask, -torch.inf))


@pytest.mark.parametrize("width,block_size", [(-1, 8), (8, 0), (9, 8)])
def test_compact_expansion_rejects_invalid_geometry(width, block_size):
    with pytest.raises(ValueError):
        expand_candidate_block_mask(torch.ones(1, 1, dtype=torch.bool), width, block_size)
