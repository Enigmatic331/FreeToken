"""Tokenizer-side native DeepSeek-V4.1 image-span construction."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import torch

from freetoken.models.deepseek_v41.args import load_args
from freetoken.models.deepseek_v41.image_processor import (
    MAX_REQUEST_IMAGES,
    image_token_types,
    load_image,
    process_image,
)

from .qwen_vl import TokenizedMultimodalPrompt, image_sources


def _processed_image_key(
    patches: torch.Tensor,
    n_vit_h: int,
    n_vit_w: int,
    types: torch.Tensor,
) -> bytes:
    canonical = patches.to(dtype=torch.bfloat16, device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"FreeToken/DeepSeekV41/patches-bf16/v1\0")
    digest.update(struct.pack("<3I", n_vit_h, n_vit_w, types.numel()))
    digest.update(bytes(types.to(torch.uint8).tolist()))
    digest.update(canonical.view(torch.uint8).numpy().tobytes())
    return digest.digest()


class DeepseekV41Processor:
    def __init__(self, model_path: str) -> None:
        from freetoken.utils import load_tokenizer

        self.model_path = model_path
        self.args = load_args(model_path)
        self.tokenizer = load_tokenizer(model_path)

    def _images(self, sources: list[Any]):
        if len(sources) > MAX_REQUEST_IMAGES:
            raise ValueError(f"at most {MAX_REQUEST_IMAGES} images are supported per request")
        results = []
        for source in sources:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = process_image(
                load_image(source), self.args
            )
            types = image_token_types(n_llm_h, n_llm_w)
            patches = patches.to(dtype=torch.bfloat16).contiguous().cpu()
            results.append((patches, n_vit_h, n_vit_w, types))
        return results

    def process(self, prompt: str, messages: Any) -> TokenizedMultimodalPrompt:
        sources = image_sources(messages)
        if not sources:
            raise ValueError("multimodal processing requested without an image")
        images = self._images(sources)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        placeholder_count = sum(
            token == self.args.image_token_id for token in prompt_ids
        )
        if placeholder_count != len(images):
            raise ValueError(
                f"DeepSeek prompt contains {placeholder_count} image placeholders "
                f"for {len(images)} images"
            )

        ids: list[int] = []
        spans: list[tuple[int, int]] = []
        grids: list[list[int]] = []
        keys: list[bytes] = []
        image_iter = iter(images)
        for token in prompt_ids:
            if token != self.args.image_token_id:
                ids.append(token)
                continue
            patches, n_vit_h, n_vit_w, types = next(image_iter)
            start = len(ids)
            ids.extend([self.args.image_token_id] * types.numel())
            spans.append((start, len(ids)))
            grids.append([n_vit_h, n_vit_w, types.numel()])
            keys.append(
                _processed_image_key(patches, n_vit_h, n_vit_w, types)
            )

        input_ids = torch.tensor(ids, dtype=torch.int32)
        positions = torch.arange(input_ids.numel(), dtype=torch.int32)
        return TokenizedMultimodalPrompt(
            input_ids=input_ids,
            pixel_values=torch.cat([image[0] for image in images], dim=0),
            image_grid_thw=torch.tensor(grids, dtype=torch.int32),
            rope_positions=positions[:, None].expand(-1, 3).contiguous(),
            image_cache_keys=keys,
            image_token_spans=spans,
            image_inputs=(
                list(sources)
                if all(isinstance(source, (str, bytes)) for source in sources)
                else None
            ),
        )

    def process_images(
        self, sources: list[str | bytes]
    ) -> tuple[torch.Tensor, torch.Tensor, list[bytes]]:
        images = self._images(sources)
        grids = []
        keys = []
        for patches, n_vit_h, n_vit_w, types in images:
            grids.append([n_vit_h, n_vit_w, types.numel()])
            keys.append(_processed_image_key(patches, n_vit_h, n_vit_w, types))
        return (
            torch.cat([image[0] for image in images], dim=0),
            torch.tensor(grids, dtype=torch.int32),
            keys,
        )


__all__ = ["DeepseekV41Processor"]
