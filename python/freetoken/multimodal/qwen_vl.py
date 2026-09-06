"""Qwen-VL image preprocessing and text-decoder MRoPE coordinates.

Qwen3.8 reuses the Qwen3-VL visual frontend.  This module deliberately owns the
processor boundary rather than placing it in ``models.qwen4_exp`` so later Qwen-VL
families can share it.  Media stays as CPU data here; the scheduler's backbone rank
owns the visual encoder and chooses its CUDA device.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import itertools
import urllib.request
from collections import OrderedDict
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
    # One exact, content-addressed identity per image. The scheduler maps these
    # full digests to collision-free radix symbols for the lifetime of its KV cache.
    image_cache_keys: list[bytes] | None = None
    # Original compressed inputs let the scheduler safely reconstruct pixels after
    # a vision-feature LRU eviction. Online inputs are bytes/URLs; PIL objects stay local.
    image_inputs: list[str | bytes] | None = None

    @property
    def is_multimodal(self) -> bool:
        return self.pixel_values is not None or bool(self.image_cache_keys)


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


def _image_content_keys(images: list[Any]) -> list[bytes]:
    result: list[bytes] = []
    for image in images:
        digest = hashlib.sha256()
        digest.update(b"FreeToken/QwenVL/RGB/v1\0")
        digest.update(int(image.width).to_bytes(8, "little"))
        digest.update(int(image.height).to_bytes(8, "little"))
        digest.update(image.mode.encode("ascii"))
        digest.update(b"\0")
        digest.update(image.tobytes())
        result.append(digest.digest())
    return result


def image_cache_keys(
    images: list[Any],
    image_grid_thw: torch.Tensor,
    *,
    content_keys: list[bytes] | None = None,
) -> list[bytes]:
    """Hash the exact RGB content and processor geometry of each image."""
    grids = image_grid_thw.view(-1, 3)
    if len(images) != grids.shape[0]:
        raise ValueError("image count does not match image_grid_thw")
    content_keys = content_keys or _image_content_keys(images)
    if len(content_keys) != len(images):
        raise ValueError("image content-key count does not match images")
    result: list[bytes] = []
    for content_key, grid in zip(content_keys, grids, strict=True):
        digest = hashlib.sha256()
        digest.update(b"FreeToken/QwenVL/image-cache/v1\0")
        digest.update(content_key)
        digest.update(grid.to(torch.int32).contiguous().numpy().tobytes())
        result.append(digest.digest())
    return result


class QwenVLProcessor:
    """Lazy Hugging Face Qwen3-VL processor wrapper for Qwen3.8 image prompts."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._processor = None
        # Geometry and replacement strings are tiny compared with the float32 patch
        # tensor. A per-tokenizer LRU lets repeat turns skip image preprocessing and
        # the large tokenizer->scheduler pixel transfer.
        self._image_metadata: OrderedDict[
            tuple[bytes, ...], tuple[torch.Tensor, list[bytes]]
        ] = OrderedDict()
        self._image_metadata_entries = 128

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
        content_keys = _image_content_keys(images)
        metadata_key = tuple(content_keys)
        processor = self._get_processor()
        cached = self._image_metadata.get(metadata_key)
        can_omit_pixels = cached is not None and all(
            isinstance(source, (str, bytes)) for source in sources
        )
        if can_omit_pixels:
            self._image_metadata.move_to_end(metadata_key)
            grid, keys = cached
            merge = int(processor.image_processor.merge_size)
            replacements = [
                processor.image_token * (int(row.to(torch.int64).prod().item()) // (merge * merge))
                for row in grid
            ]
            text, _ = processor.get_text_with_replacements(
                [prompt], replacements, [], []
            )
            merged = processor._merge_kwargs(
                processor.valid_processor_kwargs,
                tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
                return_tensors="pt",
                padding=False,
            )
            text_kwargs = dict(merged["text_kwargs"])
            text_kwargs.pop("return_mm_token_type_ids", None)
            text_kwargs.pop("return_text_replacement_offsets", None)
            text_inputs = processor.tokenizer(text, **text_kwargs)
            processor._check_special_mm_tokens(
                text, text_inputs, modalities=["image", "video", "audio"]
            )
            input_ids = text_inputs["input_ids"].view(-1).to(torch.int32).cpu()
            token_types = torch.tensor(
                processor.create_mm_token_type_ids(text_inputs["input_ids"]),
                dtype=torch.int32,
            ).view(-1)
            pixel_values = None
        else:
            encoded = processor(
                text=[prompt], images=images, return_tensors="pt", padding=False
            )
            input_ids = encoded["input_ids"].view(-1).to(torch.int32).cpu()
            pixel_values = encoded["pixel_values"].contiguous().cpu()
            grid = encoded["image_grid_thw"].to(torch.int32).contiguous().cpu()
            token_types = encoded["mm_token_type_ids"].view(-1)
            keys = image_cache_keys(images, grid, content_keys=content_keys)
            self._image_metadata[metadata_key] = (grid.clone(), keys)
            self._image_metadata.move_to_end(metadata_key)
            while len(self._image_metadata) > self._image_metadata_entries:
                self._image_metadata.popitem(last=False)
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
            image_cache_keys=keys,
            image_inputs=(
                list(sources)
                if can_omit_pixels
                else None
            ),
        )

    def process_images(
        self, sources: list[str | bytes]
    ) -> tuple[torch.Tensor, torch.Tensor, list[bytes]]:
        """Reconstruct processor pixels for a scheduler-side feature-cache miss."""
        images = [load_image(source) for source in sources]
        encoded = self._get_processor()(images=images, return_tensors="pt")
        pixels = encoded["pixel_values"].contiguous().cpu()
        grid = encoded["image_grid_thw"].to(torch.int32).contiguous().cpu()
        return pixels, grid, image_cache_keys(images, grid)


__all__ = [
    "QwenVLProcessor",
    "TokenizedMultimodalPrompt",
    "image_sources",
    "image_cache_keys",
    "load_image",
    "qwen_vl_mrope_positions",
]
