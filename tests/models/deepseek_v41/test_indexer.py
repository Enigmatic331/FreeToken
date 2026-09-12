from __future__ import annotations

import torch

from freetoken.models.deepseek_v41.indexer import (
    CandidateRuntime,
    indexer_prefill_chunk_rows,
    indexer_prefill_max_logits_bytes,
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
        torch.arange(scores.shape[-1]) == newest, torch.inf
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


def test_candidate_consumer_requires_source_publication():
    runtime = CandidateRuntime()
    try:
        runtime.consume(torch.zeros(1, 4))
    except RuntimeError as exc:
        assert "before the source" in str(exc)
    else:
        raise AssertionError("missing publication must fail closed")
