"""Bounded, content-addressed caches for online multimodal requests."""

from __future__ import annotations

import os
from collections import OrderedDict

import torch


_MAX_REGISTERED_IMAGES = 65_536
_DEFAULT_VISION_CACHE_BYTES = 256 << 20


class MultimodalCacheKeyRegistry:
    """Map full image digests to process-lifetime int32 radix symbols.

    A single radix symbol cannot carry a 256-bit digest. Assigning symbols from a
    scheduler-local registry makes collisions impossible while the corresponding
    prefix cache is alive. Every EP rank sees the same messages in the same order,
    so the independently maintained registries stay lock-step without another
    collective.
    """

    def __init__(self, max_images: int = _MAX_REGISTERED_IMAGES) -> None:
        self.max_images = max_images
        self._symbols: dict[bytes, int] = {}
        self._next_symbol = -1

    def cache_ids(
        self,
        input_ids: torch.Tensor,
        image_token_id: int,
        image_keys: list[bytes],
    ) -> torch.Tensor | None:
        """Return token-length radix keys, or ``None`` when safe reuse is unavailable."""
        if not image_keys:
            return None
        if not input_ids.is_cpu or input_ids.dtype != torch.int32 or input_ids.ndim != 1:
            raise ValueError("multimodal input_ids must be a CPU 1D int32 tensor")

        mask = input_ids == image_token_id
        starts = mask & ~torch.cat((torch.tensor([False]), mask[:-1]))
        span_starts = starts.nonzero(as_tuple=False).view(-1).tolist()
        if len(span_starts) != len(image_keys):
            raise ValueError(
                f"image cache keys ({len(image_keys)}) do not match image-token spans "
                f"({len(span_starts)})"
            )

        new_keys = [key for key in dict.fromkeys(image_keys) if key not in self._symbols]
        if len(self._symbols) + len(new_keys) > self.max_images:
            return None
        for key in new_keys:
            if not isinstance(key, bytes) or len(key) != 32:
                raise ValueError("image cache keys must be 32-byte SHA-256 digests")
            self._symbols[key] = self._next_symbol
            self._next_symbol -= 1

        result = input_ids.clone()
        for index, start in enumerate(span_starts):
            end = start
            while end < mask.numel() and bool(mask[end]):
                end += 1
            result[start:end] = self._symbols[image_keys[index]]
        return result


class VisionFeatureCache:
    """Byte-bounded LRU of CPU vision features keyed by exact image identities."""

    def __init__(self, max_bytes: int | None = None) -> None:
        if max_bytes is None:
            raw = os.environ.get("FREETOKEN_VISION_CACHE_BYTES", str(_DEFAULT_VISION_CACHE_BYTES))
            try:
                max_bytes = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"FREETOKEN_VISION_CACHE_BYTES must be an integer, got {raw!r}"
                ) from exc
        self.max_bytes = max(0, max_bytes)
        self.current_bytes = 0
        self._entries: OrderedDict[tuple[bytes, ...], torch.Tensor] = OrderedDict()

    @staticmethod
    def _nbytes(value: torch.Tensor) -> int:
        return value.numel() * value.element_size()

    def get(self, key: tuple[bytes, ...]) -> torch.Tensor | None:
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: tuple[bytes, ...], value: torch.Tensor) -> bool:
        if self.max_bytes == 0:
            return False
        stored = value.detach().to(device="cpu").contiguous()
        nbytes = self._nbytes(stored)
        if nbytes > self.max_bytes:
            return False
        old = self._entries.pop(key, None)
        if old is not None:
            self.current_bytes -= self._nbytes(old)
        while self._entries and self.current_bytes + nbytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self.current_bytes -= self._nbytes(evicted)
        self._entries[key] = stored
        self.current_bytes += nbytes
        return True

    def clear(self) -> None:
        self._entries.clear()
        self.current_bytes = 0

