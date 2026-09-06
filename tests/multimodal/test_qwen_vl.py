from __future__ import annotations

import base64
import io

import torch
from PIL import Image

from freetoken.multimodal.qwen_vl import (
    image_cache_keys,
    image_sources,
    load_image,
    qwen_vl_mrope_positions,
)


def test_image_sources_and_data_url_decode():
    image = Image.new("RGB", (3, 2), (12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]

    assert image_sources(messages) == [url]
    decoded = load_image(url)
    assert decoded.mode == "RGB" and decoded.size == (3, 2)
    assert decoded.getpixel((0, 0)) == (12, 34, 56)


def test_qwen_image_mrope_positions_and_decode_delta():
    # text(2), one merged 1x2x3 image grid, text(2)
    input_ids = torch.arange(10, dtype=torch.int32)
    token_types = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int32)
    positions, delta = qwen_vl_mrope_positions(
        input_ids,
        token_types,
        torch.tensor([[1, 4, 6]], dtype=torch.int32),
        spatial_merge_size=2,
    )

    expected = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 3],
            [2, 2, 4],
            [2, 3, 2],
            [2, 3, 3],
            [2, 3, 4],
            [5, 5, 5],
            [6, 6, 6],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(positions, expected)
    assert delta == -3


def test_image_cache_keys_are_content_and_geometry_addressed():
    image_a = Image.new("RGB", (3, 2), (12, 34, 56))
    image_a_copy = image_a.copy()
    image_b = image_a.copy()
    image_b.putpixel((2, 1), (12, 34, 57))
    grid = torch.tensor([[1, 4, 6]], dtype=torch.int32)

    key_a = image_cache_keys([image_a], grid)[0]
    assert len(key_a) == 32
    assert image_cache_keys([image_a_copy], grid)[0] == key_a
    assert image_cache_keys([image_b], grid)[0] != key_a
    assert image_cache_keys([image_a], torch.tensor([[1, 6, 6]]))[0] != key_a
