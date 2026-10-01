"""Native DeepSeek-V4.1 image preprocessing.

The resize, patch ordering, normalization, and soft-token layout follow the
checkpoint reference implementation.  Network inputs are bounded and every
redirect target is revalidated before it is fetched.
"""

from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import math
import socket
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import torch
from PIL import Image, ImageOps


TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
MAX_IMAGE_BYTES = 32 << 20
MAX_REQUEST_IMAGES = 16
MAX_IMAGE_PIXELS = 64 * 1024 * 1024


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int]:
    return (
        math.ceil((best_height // patch_size) / downsample_ratio),
        math.ceil((best_width // patch_size) / downsample_ratio),
    )


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int]:
    ratio = height / width
    max_w = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h = max_w * ratio
    cell = patch_size * downsample_ratio
    if max_w < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w) * cell / width,
        math.floor(max_h) * cell / height,
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    n_llm_h, n_llm_w = llm_grid(
        best_height, best_width, patch_size, downsample_ratio
    )
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height,
            width,
            patch_size,
            downsample_ratio,
            max_n_token,
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
            raise RuntimeError("DeepSeek image resize did not satisfy its token budget")
    return n_llm_h, n_llm_w, best_height, best_width


def _validate_image_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("images require an HTTP(S) URL or a base64 data URL")
    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
    except OSError as exc:
        raise ValueError("could not resolve image URL") from exc
    if not addresses or any(
        not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses
    ):
        raise ValueError("image URLs must resolve to public addresses")


class _ImageRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _decode_base64(data: str) -> bytes:
    if len(data) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ValueError("image exceeds the 32 MiB upload limit")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise ValueError("invalid base64 image") from exc
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the 32 MiB upload limit")
    return decoded


def load_image_bytes(source) -> bytes:
    """Load bounded API image data without accepting local filesystem paths."""

    if isinstance(source, bytes):
        if len(source) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds the 32 MiB upload limit")
        return source
    if not isinstance(source, str):
        raise ValueError("image_url must be bytes, an HTTP(S) URL, or a base64 data URL")
    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("image data URL must use base64 encoding")
        return _decode_base64(payload)
    if source.startswith(("http://", "https://")):
        _validate_image_url(source)
        request = Request(
            source, headers={"User-Agent": "FreeToken/vision"}
        )
        with build_opener(_ImageRedirectHandler()).open(request, timeout=30) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds the 32 MiB download limit")
        return data
    raise ValueError("images require an HTTP(S) URL or a base64 data URL")


def load_image(source):
    try:
        with Image.open(io.BytesIO(load_image_bytes(source))) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError("image exceeds the 64 megapixel limit")
            opened.load()
            return opened.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:  # Pillow exposes several format-specific exceptions.
        raise ValueError(f"could not decode image: {exc}") from exc


def plan_image_grid(width: int, height: int, args) -> tuple[int, int, int, int]:
    patch = args.vision_patch_size
    if (
        width <= 0
        or height <= 0
        or patch <= 0
        or args.vision_downsample_ratio <= 0
        or args.vision_max_n_token < 4
    ):
        raise ValueError("invalid image dimensions or vision configuration")
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = int(height * args.vision_max_wh_ratio)
    if width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / patch) * patch
    best_height = math.ceil(height / patch) * patch
    return safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch,
        args.vision_downsample_ratio,
        args.vision_max_n_token,
    )


def process_image(image: Image.Image, args):
    patch = args.vision_patch_size
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("image exceeds the 64 megapixel limit")
    image = image.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(
        image.width, image.height, args
    )
    n_vit_h, n_vit_w = best_height // patch, best_width // patch
    if (
        args.vision_max_wh_ratio is not None
        and image.width >= args.vision_max_wh_ratio * image.height
    ):
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    values = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
    values = values.permute(2, 0, 1) / 255
    values = (values - 0.5) / 0.5
    patches = (
        values.reshape(3, n_vit_h, patch, n_vit_w, patch)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch, patch)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


__all__ = [
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_START",
    "MAX_REQUEST_IMAGES",
    "image_token_types",
    "load_image",
    "num_image_tokens",
    "plan_image_grid",
    "process_image",
]
