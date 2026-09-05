from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.scheduler import _make_rope_positions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires pinned-memory CUDA path")
def test_scheduler_preserves_prompt_mrope_and_offsets_decode_tokens():
    prompt_positions = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 7, 11],
            [2, 7, 12],
            [2, 8, 11],
        ],
        dtype=torch.int32,
    )
    multimodal = SimpleNamespace(
        cached_len=3,
        device_len=7,
        extend_len=4,
        prompt_rope_positions=prompt_positions,
        mrope_position_delta=-2,
    )
    text_padding = SimpleNamespace(
        cached_len=4,
        device_len=6,
        extend_len=2,
        prompt_rope_positions=None,
        mrope_position_delta=0,
    )
    batch = SimpleNamespace(padded_reqs=[multimodal, text_padding])

    actual = _make_rope_positions(batch, torch.device("cuda"))

    expected = torch.tensor(
        [
            [2, 7, 12],
            [2, 8, 11],
            [3, 3, 3],
            [4, 4, 4],
            [4, 4, 4],
            [5, 5, 5],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(actual, expected)
