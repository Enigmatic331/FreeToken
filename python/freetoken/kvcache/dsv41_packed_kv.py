"""Packed FP4 storage for DeepSeek-V4.1's compressed main KV (``FREETOKEN_DSV41_PACKED_KV=1``).

V4.1 rounds every compressed KV row to FP4 before caching it: E2M1 values with one E4M3
scale per 16 channels (``kernel/triton/dsv41/rope_fp4.py``). FreeToken used to store the
rounded values as BF16, 1,024 bytes per 512-channel row. A product of an E2M1 value (2
significant bits) and an E4M3 scale (at most 4) is exact in BF16, so the same row packs
into 256 bytes of nibbles plus 32 scale bytes and unpacks to the identical BF16 bits. That
is lossless: outputs do not change, the cache shrinks 3.6x.

Nibble layout: bit 3 is the sign (kept for zeros, so a -0.0 stays -0.0), bits 0-2 the
E2M1 magnitude code (0, 0.5, 1, 1.5, 2, 3, 4, 6). Storage is 16-bit words: nibble word ``w``
holds channels ``w``, ``w + D/4``, ``w + D/2`` and ``w + 3D/4`` in bits 0-3, 4-7, 8-11 and
12-15; scale word ``g`` holds the raw E4M3 scales of groups ``2g`` (low byte) and ``2g + 1``.
Each quarter of a row therefore unpacks 1:1 from one contiguous 16-bit load, which keeps
the attention kernel's MMA operands on the same layout (``kWidth=2``) as a BF16 tile: from
8-bit loads Triton picks ``kWidth=4``, which changes the accumulation order and the bits.

The flag is read once at startup: the representation cannot change under live pages or
captured graphs.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch

GROUP = 16  # channels per E4M3 scale


@functools.lru_cache(maxsize=1)
def _packed_source_spec() -> frozenset[int] | None:
    """Resolve the startup-only storage policy.

    ``FREETOKEN_DSV41_PACKED_KV=1`` retains the original all-source switch.
    Otherwise ``FREETOKEN_DSV41_PACKED_KV_SOURCES=2,8,14`` may pack only selected
    physical source slabs.  ``None`` means all sources; an empty set means off.
    Source selection is useful because V4.1's ratio-2 sources save 60% of the
    available bytes while serving fewer than half of its attention layers.
    """
    if os.environ.get("FREETOKEN_DSV41_PACKED_KV", "0") == "1":
        return None
    raw = os.environ.get("FREETOKEN_DSV41_PACKED_KV_SOURCES", "").strip()
    if not raw:
        return frozenset()
    try:
        result = frozenset(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(
            "FREETOKEN_DSV41_PACKED_KV_SOURCES must be comma-separated layer ids"
        ) from exc
    if any(layer < 0 for layer in result):
        raise ValueError("packed KV source layer ids must be non-negative")
    return result


def packed_kv_enabled() -> bool:
    """Whether at least one source slab uses packed storage."""
    sources = _packed_source_spec()
    return sources is None or bool(sources)


def packed_kv_enabled_for_source(source_layer: int) -> bool:
    """Whether one physical source slab uses packed storage."""
    sources = _packed_source_spec()
    return sources is None or int(source_layer) in sources


# Preserve the old test/operator reset surface now that the cached value is the policy.
packed_kv_enabled.cache_clear = _packed_source_spec.cache_clear
packed_kv_enabled_for_source.cache_clear = _packed_source_spec.cache_clear


@functools.lru_cache(maxsize=1)
def packed_prefill_gluon_enabled() -> bool:
    """``FREETOKEN_DSV41_PACKED_PREFILL_GLUON=1`` (startup-only, default off): prefill
    (``m > 1``) over a packed pool runs the experimental Gluon kernel
    (``kernel/triton/dsv41/sparse_attn_gluon.py``) instead of the Triton packed kernel.
    Both must reproduce the BF16 pool's output bits; enable only once qualified."""
    return os.environ.get("FREETOKEN_DSV41_PACKED_PREFILL_GLUON", "0") == "1"


@functools.lru_cache(maxsize=1)
def packed_decode_stage_enabled() -> bool:
    """Use transient exact-BF16 staging for graph-captured ``m == 1`` packed attention."""
    return os.environ.get("FREETOKEN_DSV41_PACKED_DECODE_STAGE", "0") == "1"


def packed_row_bytes(head_dim: int) -> int:
    return head_dim // 2 + head_dim // GROUP


@dataclass
class PackedRows:
    """``q`` int16 ``[n, D/4]`` nibble words and ``s`` int16 ``[n, D/32]`` scale words."""

    q: torch.Tensor
    s: torch.Tensor

    @classmethod
    def zeros(cls, n: int, head_dim: int, device) -> PackedRows:
        if head_dim % (4 * 2 * GROUP):
            raise ValueError(f"packed KV needs head_dim divisible by {8 * GROUP}, got {head_dim}")
        return cls(
            torch.zeros(n, head_dim // 4, dtype=torch.int16, device=device),
            torch.zeros(n, head_dim // (2 * GROUP), dtype=torch.int16, device=device),
        )

    @classmethod
    def empty(cls, n: int, head_dim: int, device) -> PackedRows:
        """Kernel outputs: every element gets written."""
        return cls(
            torch.empty(n, head_dim // 4, dtype=torch.int16, device=device),
            torch.empty(n, head_dim // (2 * GROUP), dtype=torch.int16, device=device),
        )

    @property
    def head_dim(self) -> int:
        return self.q.shape[1] * 4

    @property
    def device(self) -> torch.device:
        return self.q.device

    def __len__(self) -> int:
        return self.q.shape[0]

    def numel_bytes(self) -> int:
        return self.q.numel() * self.q.element_size() + self.s.numel() * self.s.element_size()

    def index_copy_(self, dim: int, rows: torch.Tensor, value: PackedRows) -> PackedRows:
        if not isinstance(value, PackedRows):
            raise TypeError("a packed pool takes PackedRows (from rope_fp4_pack), not BF16 rows")
        self.q.index_copy_(dim, rows, value.q)
        self.s.index_copy_(dim, rows, value.s)
        return self

    def unpack(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        """BF16 rows, bit-identical to what the BF16 pool would hold (debug and tests)."""
        q = self.q if rows is None else self.q.index_select(0, rows)
        s = self.s if rows is None else self.s.index_select(0, rows)
        return unpack_reference(q, s)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def pack_reference(rounded: torch.Tensor, scale: torch.Tensor, negative: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch mirror of the packing kernel for tests: ``rounded`` holds the E2M1 grid values
    (before scaling) ``[n, D]``, ``scale`` the E4M3-representable scales ``[n, D/16]``,
    ``negative`` the pre-rounding sign ``[n, D]``."""
    magnitude = rounded.abs().float()
    code = torch.where(
        magnitude < 2.0, magnitude * 2.0, torch.where(magnitude < 4.0, magnitude + 2.0, magnitude * 0.5 + 4.0)
    ).to(torch.int32)
    nibble = code | (negative.to(torch.int32) << 3)
    quarters = nibble.view(nibble.shape[0], 4, -1)
    words = quarters[:, 0] | (quarters[:, 1] << 4) | (quarters[:, 2] << 8) | (quarters[:, 3] << 12)
    s8 = scale.float().to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32).view(scale.shape[0], -1, 2)
    s16 = s8[..., 0] | (s8[..., 1] << 8)
    return _as_int16(words), _as_int16(s16)


def _as_int16(x: torch.Tensor) -> torch.Tensor:
    """Low 16 bits of an int32 tensor as int16 (two's complement)."""
    return torch.where(x >= 1 << 15, x - (1 << 16), x).to(torch.int16).contiguous()


def unpack_reference(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Torch mirror of the in-kernel dequantization: BF16 ``[n, D]``."""
    table = torch.tensor(_E2M1, dtype=torch.float32, device=q.device)
    words = q.to(torch.int32) & 0xFFFF
    nibbles = torch.cat([(words >> (4 * k)) & 0xF for k in range(4)], dim=-1).long()
    value = table[nibbles & 7]
    value = torch.where((nibbles >> 3).bool(), -value, value)
    sw = s.to(torch.int32) & 0xFFFF
    scale_bytes = torch.stack((sw & 0xFF, sw >> 8), dim=-1).reshape(s.shape[0], -1).to(torch.uint8)
    scale = scale_bytes.view(torch.float8_e4m3fn).float()
    out = value.view(q.shape[0], -1, GROUP) * scale[..., None]
    return out.reshape(q.shape[0], -1).to(torch.bfloat16)


__all__ = [
    "GROUP",
    "PackedRows",
    "pack_reference",
    "packed_kv_enabled",
    "packed_kv_enabled_for_source",
    "packed_decode_stage_enabled",
    "packed_prefill_gluon_enabled",
    "packed_row_bytes",
    "unpack_reference",
]
