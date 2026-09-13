from types import SimpleNamespace

import torch

from freetoken.engine.graph import (
    _dsv41_graph_stage_ranges,
    _dsv41_local_n_heads,
)


def _consumer_blackwell_split_count(_b, _m, _h, topk, _device):
    # V4.1 bs=1, H=64 on a 170-SM RTX 5090: occupancy does not clamp these
    # widths, leaving ceil(topk / (4 * 32)).
    splits = (topk + 127) // 128
    return splits if splits > 1 else 0


def test_dsv41_graph_ranges_preserve_each_sparse_attention_topology():
    args = SimpleNamespace(
        compress_ratios=(0, 0, 2, 2, 1, 1),
        window_size=128,
        index_topk=512,
    )
    ranges = _dsv41_graph_stage_ranges(
        args,
        batch_size=1,
        n_heads=64,
        max_seq_len=32768,
        device=torch.device("cuda"),
        split_counter=_consumer_blackwell_split_count,
    )

    assert ranges == (
        (1, 127, 127),
        (128, 255, 255),
        (256, 256, 256),
        (257, 383, 383),
        (384, 512, 512),
        (513, 768, 768),
        (769, 32767, 32767),
    )

    ratios = (1, 2)
    for lo, hi, cap in ranges:
        captured = tuple(
            _consumer_blackwell_split_count(
                1, 1, 64, 128 + min(512, (cap + 1) // ratio), None
            )
            for ratio in ratios
        )
        for position in {lo, hi, (lo + hi) // 2}:
            live = tuple(
                _consumer_blackwell_split_count(
                    1,
                    1,
                    64,
                    128 + min(512, (position + 1) // ratio),
                    None,
                )
                for ratio in ratios
            )
            assert live == captured


def test_dsv41_graph_final_range_keeps_full_history_ceiling():
    args = SimpleNamespace(
        compress_ratios=(1, 2), window_size=128, index_topk=512
    )
    ranges = _dsv41_graph_stage_ranges(
        args,
        batch_size=1,
        n_heads=64,
        max_seq_len=4096,
        device=torch.device("cuda"),
        split_counter=_consumer_blackwell_split_count,
    )

    assert ranges[-1] == (769, 4095, 4095)


def test_dsv41_local_heads_tolerates_expert_only_rank():
    args = SimpleNamespace(n_heads=64)
    worker = SimpleNamespace(_model=SimpleNamespace(layers=[SimpleNamespace()]))
    assert _dsv41_local_n_heads(worker, args) == 64


def test_dsv41_local_heads_observes_attention_sharding():
    args = SimpleNamespace(n_heads=64)
    attention = SimpleNamespace(
        n_heads=32, plan=SimpleNamespace(compress_ratio=2)
    )
    model = SimpleNamespace(
        _model=SimpleNamespace(layers=[SimpleNamespace(attn=attention)])
    )
    assert _dsv41_local_n_heads(model, args) == 32
