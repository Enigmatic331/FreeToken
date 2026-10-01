from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import torch
from PIL import Image

from freetoken.models.deepseek_v41.image_processor import (
    image_token_types,
    process_image,
)
from freetoken.multimodal.deepseek_v41 import DeepseekV41Processor
from freetoken.scheduler.multimodal_cache import MultimodalCacheKeyRegistry


def _data_url(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (4, 4), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _args():
    return SimpleNamespace(
        image_token_id=99,
        vision_patch_size=2,
        vision_downsample_ratio=2,
        vision_max_n_token=64,
        vision_min_pixels=1,
        vision_max_wh_ratio=None,
    )


def test_native_image_preprocessing_has_reference_patch_and_span_layout():
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = process_image(
        Image.new("RGB", (4, 4), (255, 0, 127)), _args()
    )
    assert patches.shape == (4, 3, 2, 2)
    assert (n_vit_h, n_vit_w, n_llm_h, n_llm_w) == (2, 2, 1, 1)
    assert image_token_types(n_llm_h, n_llm_w).tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(patches[0, 0], torch.ones((2, 2)))
    torch.testing.assert_close(patches[0, 1], -torch.ones((2, 2)))


def test_processor_preserves_adjacent_image_boundaries_for_radix_identity():
    class Tokenizer:
        @staticmethod
        def encode(_prompt, add_special_tokens=False):
            assert not add_special_tokens
            return [7, 99, 99, 8]

    processor = object.__new__(DeepseekV41Processor)
    processor.args = _args()
    processor.tokenizer = Tokenizer()
    first, second = _data_url((1, 2, 3)), _data_url((4, 5, 6))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": first},
                {"type": "image_url", "image_url": second},
            ],
        }
    ]

    encoded = processor.process("unused", messages)
    assert encoded.image_token_spans == [(1, 5), (5, 9)]
    assert encoded.image_grid_thw.tolist() == [[2, 2, 4], [2, 2, 4]]
    assert encoded.pixel_values.shape == (8, 3, 2, 2)
    assert encoded.input_ids.tolist() == [7] + [99] * 8 + [8]
    assert len(encoded.image_cache_keys) == 2
    assert encoded.image_cache_keys[0] != encoded.image_cache_keys[1]

    cache_ids = MultimodalCacheKeyRegistry().cache_ids(
        encoded.input_ids,
        99,
        encoded.image_cache_keys,
        encoded.image_token_spans,
    )
    assert cache_ids[1] < 0 and cache_ids[5] < 0
    assert cache_ids[1] != cache_ids[5]
