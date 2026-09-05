"""Qwen-VL image preprocessing and text-decoder MRoPE coordinates.

Qwen3.8 reuses the Qwen3-VL visual frontend.  This module deliberately owns the
processor boundary rather than placing it in ``models.qwen4_exp`` so later Qwen-VL
families can share it.  Media stays as CPU data here; the scheduler's backbone rank
owns the visual encoder and chooses its CUDA device.
"""

from __future__ import annotations

import base64
import binascii
import io
import itertools
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

import torch


_MAX_IMAGE_BYTES = 64 << 20


@dataclass(frozen=True)
class TokenizedMultimodalPrompt:
    input_ids: torch.Tensor                 # CPU [N] int32
    pixel_values: torch.Tensor | None = None  # CPU [patches, patch_width] float32
    image_grid_thw: torch.Tensor | None = None  # CPU [images, 3] int32
    rope_positions: torch.Tensor | None = None  # CPU [N, 3] int32 (T/H/W)
    mrope_position_delta: int = 0

    @property
    def is_multimodal(self) -> bool:
        return self.pixel_values is not None


def _content_parts(messages: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    yield part


def image_sources(messages: Any) -> list[Any]:
    """Image payloads in chat-template order (OpenAI/Responses normalized shape)."""
    result: list[Any] = []
    for part in _content_parts(messages):
        ptype = part.get("type")
        if ptype not in ("image", "image_url", "input_image") and not (
            "image" in part or "image_url" in part
        ):
            continue
        source = part.get("image", part.get("image_url"))
        if isinstance(source, dict):
            source = source.get("url", source.get("data"))
        if source is None:
            raise ValueError("image content part is missing image_url")
        result.append(source)
    return result


def _bounded_read(response) -> bytes:
    data = response.read(_MAX_IMAGE_BYTES + 1)
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES >> 20} MiB limit")
    return data


def load_image(source: Any):
    """Decode a PIL image from bytes, a data URL, or an HTTP(S) URL."""
    from PIL import Image

    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, bytes):
        data = source
    elif isinstance(source, str) and source.startswith("data:"):
        try:
            header, encoded = source.split(",", 1)
        except ValueError as exc:
            raise ValueError("invalid image data URL") from exc
        if ";base64" not in header.lower():
            raise ValueError("image data URL must use base64 encoding")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid base64 image data") from exc
        if len(data) > _MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES >> 20} MiB limit")
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "FreeToken/vision"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = _bounded_read(response)
        except Exception as exc:  # noqa: BLE001 -- becomes a per-request tokenizer error
            raise ValueError(f"could not fetch image: {exc}") from exc
    else:
        raise ValueError("image_url must be an http(s) URL or a base64 data URL")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 -- Pillow has several decode exception classes
        raise ValueError(f"could not decode image: {exc}") from exc


def qwen_vl_mrope_positions(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    """Return engine-layout ``[N,3]`` T/H/W positions and the decode delta.

    This is the image-only subset of Qwen3-VL's ``get_rope_index``. Text spans
    advance normally; an image span advances by the larger merged spatial axis,
    while its tokens receive their temporal/row/column grid coordinates.
    """
    ids = input_ids.view(-1)
    types = mm_token_type_ids.view(-1)
    if ids.numel() != types.numel():
        raise ValueError("input_ids and mm_token_type_ids must have the same length")
    grids = iter(image_grid_thw.view(-1, 3).tolist())
    pieces: list[torch.Tensor] = []
    current = 0
    device = ids.device

    for modality, group in itertools.groupby(enumerate(types.tolist()), lambda item: item[1]):
        grouped = list(group)
        length = grouped[-1][0] - grouped[0][0] + 1
        if modality == 0:
            pos = torch.arange(current, current + length, dtype=torch.int64, device=device)
            pieces.append(pos[:, None].expand(-1, 3))
            current += length
            continue
        if modality != 1:
            raise ValueError("video MRoPE is not implemented yet")
        try:
            grid_t, grid_h, grid_w = (int(v) for v in next(grids))
        except StopIteration as exc:
            raise ValueError("missing image_grid_thw row") from exc
        if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
            raise ValueError("image grid is not divisible by spatial_merge_size")
        llm_t = grid_t
        llm_h = grid_h // spatial_merge_size
        llm_w = grid_w // spatial_merge_size
        if length != llm_t * llm_h * llm_w:
            raise ValueError(
                f"image token span ({length}) does not match grid ({llm_t}x{llm_h}x{llm_w})"
            )
        t, h, w = torch.meshgrid(
            torch.arange(llm_t, device=device),
            torch.arange(llm_h, device=device),
            torch.arange(llm_w, device=device),
            indexing="ij",
        )
        image_pos = torch.stack((t, h, w), dim=-1).reshape(-1, 3).to(torch.int64)
        image_pos[:, 0] += current
        image_pos[:, 1:] += current
        pieces.append(image_pos)
        current += max(llm_h, llm_w)

    try:
        next(grids)
    except StopIteration:
        pass
    else:
        raise ValueError("unused image_grid_thw row")
    positions = torch.cat(pieces, dim=0) if pieces else torch.empty((0, 3), dtype=torch.int64)
    delta = int(positions.max().item() + 1 - ids.numel()) if positions.numel() else 0
    return positions.to(torch.int32), delta


class QwenVLProcessor:
    """Lazy Hugging Face Qwen3-VL processor wrapper for Qwen3.8 image prompts."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            # Qwen3.8 advertises Qwen3VLProcessor in preprocessor_config.json. The PIL
            # backend is deterministic and avoids an unnecessary GPU context in workers.
            # ``backend=\"pil\"`` is forwarded to the video processor too in
            # Transformers 5.15, whose read-only backend property then raises.  The
            # deprecated spelling remains the only component-scoped selector there.
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True, use_fast=False
            )
        return self._processor

    def process(self, prompt: str, messages: Any) -> TokenizedMultimodalPrompt:
        sources = image_sources(messages)
        if not sources:
            raise ValueError("multimodal processing requested without an image")
        images = [load_image(source) for source in sources]
        processor = self._get_processor()
        encoded = processor(
            text=[prompt], images=images, return_tensors="pt", padding=False
        )
        input_ids = encoded["input_ids"].view(-1).to(torch.int32).cpu()
        pixel_values = encoded["pixel_values"].contiguous().cpu()
        grid = encoded["image_grid_thw"].to(torch.int32).contiguous().cpu()
        token_types = encoded["mm_token_type_ids"].view(-1)
        positions, delta = qwen_vl_mrope_positions(
            input_ids,
            token_types,
            grid,
            spatial_merge_size=int(processor.image_processor.merge_size),
        )
        return TokenizedMultimodalPrompt(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            rope_positions=positions.cpu(),
            mrope_position_delta=delta,
        )


__all__ = [
    "QwenVLProcessor",
    "TokenizedMultimodalPrompt",
    "image_sources",
    "load_image",
    "qwen_vl_mrope_positions",
]
