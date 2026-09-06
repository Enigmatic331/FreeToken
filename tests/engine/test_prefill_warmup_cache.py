"""CPU-only policy checks for finalizing the synthetic prefill warmup."""

from __future__ import annotations

from unittest.mock import Mock

from freetoken.engine.engine import _finalize_moe_prefill_warmup, _needs_prefill_warmup


def test_sparse_prefill_requests_warmup_with_non_triton_attention():
    cache = Mock(sparse_prefill_max_tokens=256)

    assert _needs_prefill_warmup("qsa_sparse", cache)


def test_triton_attention_preserves_existing_warmup_behavior():
    cache = Mock(sparse_prefill_max_tokens=0)

    assert _needs_prefill_warmup("triton,flashinfer", cache)


def test_non_triton_dense_path_does_not_add_startup_work():
    cache = Mock(sparse_prefill_max_tokens=0)

    assert not _needs_prefill_warmup("qsa_sparse", cache)


def test_sparse_prefill_keeps_warmed_experts_and_clears_stats():
    cache = Mock(sparse_prefill_max_tokens=256)

    _finalize_moe_prefill_warmup(cache)

    cache.reset_stats.assert_called_once_with()
    cache.reset.assert_not_called()


def test_dense_prefill_preserves_cold_reset_behavior():
    cache = Mock(sparse_prefill_max_tokens=0)

    _finalize_moe_prefill_warmup(cache)

    cache.reset.assert_called_once_with()
    cache.reset_stats.assert_not_called()


def test_prefill_warmup_without_moe_cache_is_a_noop():
    _finalize_moe_prefill_warmup(None)
