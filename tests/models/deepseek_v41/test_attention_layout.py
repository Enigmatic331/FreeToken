from __future__ import annotations

import os

import pytest

from freetoken.models.deepseek_v41.args import load_args
from freetoken.models.deepseek_v41.attention_layout import AttentionLayout


MODEL_PATH = "/home/enigmatic331/models/DeepSeek-V4.1-Flash"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="official checkpoint absent")
def test_official_csa2_source_reuse_and_cache_arithmetic():
    args = load_args(MODEL_PATH)
    layout = AttentionLayout(args)
    assert layout[0].kv_source is None
    assert (layout[2].kv_source, layout[7].kv_source) == (2, 2)
    assert (layout[8].kv_source, layout[13].kv_source) == (8, 8)
    assert (layout[14].kv_source, layout[19].kv_source) == (14, 14)
    assert (layout[20].kv_source, layout[39].kv_source) == (20, 20)
    assert (layout[24].index_source, layout[27].index_source) == (24, 24)
    assert layout[20].is_candidate_source
    assert layout[24].uses_candidates
    assert sum(layer.owns_kv for layer in layout.layers) == 4
    assert sum(layer.owns_index for layer in layout.layers) == 8

    cache = layout.cache_bytes(65_536)
    assert cache.window == 5_242_880
    assert cache.compressed_kv == 167_772_160
    assert cache.index_keys == 41_943_040
    assert cache.compressor_state == 24_576
    assert cache.total == 214_982_656
