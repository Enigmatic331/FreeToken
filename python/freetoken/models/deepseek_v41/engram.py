# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Hash/layout arithmetic and the resident-table design are adapted from SGLang's
# DeepSeek-V4.1 implementation.  The table backend builds on FreeToken's Qwen PLE
# UVA path and O_DIRECT host-bank loader.
"""DeepSeek-V4.1 Engram hashing and resident, row-sharded host tables.

Unlike Qwen4Exp PLE, DeepSeek uses two independent tables, 24 hashes per token,
FP8 rows with one E8M0 scale per 32 values, and no PLE convolution.  Each EP/TP
rank therefore holds only its global row interval in an anonymous host mapping.
The GPU gathers owned rows over UVA; a 12 KiB/token BF16 all-reduce reconstructs
the complete 24x256 lookup at each of the two Engram layers.
"""

from __future__ import annotations

import ctypes
import json
import math
import mmap
import os
import re
import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from freetoken.distributed import DistributedCommunicator
from freetoken.kernel.pinned import device_ptr
from freetoken.moe.host_banks import HostBank, HostResidency, read_range_into
from freetoken.utils.progress import byte_bar

from .profile import profile_range

_DEFAULT_BLOCK_SIZE = 32
_MADV_COLLAPSE = 25  # Linux >= 6.1; Python's mmap module does not expose it.
_THP_DIR = "/sys/kernel/mm/transparent_hugepage"


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    value = start + 1
    while not _is_prime(value) or value in seen:
        value += 1
    return value


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Return training-compatible token-id normalization and its vocabulary size."""

    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_id: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        compressed = key_to_id.get(key)
        if compressed is None:
            compressed = len(key_to_id)
            key_to_id[key] = compressed
        lookup[token_id] = compressed
    return lookup, len(key_to_id)


def compute_hash_multipliers(
    layer_ids: Sequence[int], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """Reproduce the official odd int64 multiplier for every layer/lookback."""

    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(0, bound, size=max_ngram_size, dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1, dtype=torch.int64))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @property
    def hashes_per_token(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def build(
        cls,
        *,
        layer_ids: Sequence[int],
        num_embeddings: Sequence[int],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
    ) -> "EngramLayout":
        layers = tuple(int(v) for v in layer_ids)
        rows = tuple(int(v) for v in num_embeddings)
        if len(layers) != len(rows):
            raise ValueError("Engram layer_ids and num_embeddings must have equal length")
        primes: list[tuple[tuple[int, ...], ...]] = []
        seen: set[int] = set()
        for _ in layers:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], vocab_size - 1
                for _ in range(n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        result = cls(
            max_ngram_size=int(max_ngram_size),
            layer_ids=layers,
            num_embeddings=rows,
            primes=tuple(primes),
            n_heads=int(n_heads),
            head_dim=int(head_dim),
        )
        # The checkpoint's table row count is the sum of its disjoint prime buckets.
        for layer, expected in zip(result.primes, result.num_embeddings):
            actual = sum(p for ngram in layer for p in ngram)
            if actual != expected:
                raise ValueError(f"Engram bucket layout has {actual} rows, checkpoint declares {expected}")
        return result

    @classmethod
    def from_config(cls, config: Any) -> "EngramLayout":
        text = getattr(config, "text_config", config)
        return cls.build(
            layer_ids=text.engram_layer_ids,
            num_embeddings=text.engram_num_embeddings,
            max_ngram_size=text.engram_max_ngram_size,
            n_heads=text.engram_n_heads,
            head_dim=text.engram_head_dim,
            vocab_size=text.engram_vocab_size,
        )


def _layout_tensors(
    layout: EngramLayout, compressed_vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = [[p for ngram in layer for p in ngram] for layer in layout.primes]
    offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
    return (
        compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, compressed_vocab_size),
        torch.tensor(layout.primes, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
    )


def compute_engram_hash_ids(
    tokens: torch.Tensor,
    blocked: torch.Tensor,
    *,
    pad_id: int,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Hash ``tokens [T,N]`` (current first) into ``[T,layers,hashes]`` rows."""

    if tokens.shape != blocked.shape:
        raise ValueError(f"tokens/blocked shape mismatch: {tokens.shape} != {blocked.shape}")
    compressed = torch.where(blocked, pad_id, token_map[tokens.long()])
    products = compressed.unsqueeze(1) * multipliers
    rolling, hashes = products[..., 0], []
    for shift in range(1, tokens.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., shift])
        hashes.append(rolling.unsqueeze(-1) % primes[:, shift - 1])
    return torch.cat(hashes, dim=-1) + offsets


class EngramHasher:
    """Exact V4.1 hash arithmetic plus ragged prefill/decode window construction."""

    def __init__(self, layout: EngramLayout, tokenizer, *, pad_token_id: int, compressed_vocab_size: int):
        token_map, actual = build_compressed_token_map(tokenizer)
        if actual != compressed_vocab_size:
            raise ValueError(
                f"tokenizer compresses to {actual} ids, checkpoint expects {compressed_vocab_size}"
            )
        self.layout = layout
        self.pad_id = token_map[pad_token_id]
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.multipliers, self.primes, self.offsets = _layout_tensors(
            layout, compressed_vocab_size
        )

    def to(self, device: torch.device | str) -> "EngramHasher":
        for name in ("token_map", "multipliers", "primes", "offsets"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def row_ids(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """Hash a ragged forward.

        ``history [B,N-1]`` is oldest-first and contains the raw tokens immediately
        before this forward.  ``positions [T]`` are absolute per-request positions;
        they prevent a fresh sequence from consuming placeholder history.
        """

        n = self.layout.max_ngram_size
        if history.ndim != 2 or history.shape[1] != n - 1:
            raise ValueError(f"history must be [B,{n - 1}], got {tuple(history.shape)}")
        cu = cu_seqlens.long()
        if cu.numel() != history.shape[0] + 1:
            raise ValueError("cu_seqlens does not describe input_ids/history")
        if cu.device.type == "cpu" and int(cu[-1]) != input_ids.numel():
            raise ValueError("cu_seqlens does not describe input_ids/history")
        token_row = torch.arange(input_ids.numel(), device=input_ids.device)
        req = torch.searchsorted(cu[1:], token_row, right=True)
        offset = token_row - cu[req]
        shifts = torch.arange(n, device=input_ids.device)
        from_batch = shifts.unsqueeze(0) <= offset.unsqueeze(1)
        batch_index = (token_row.unsqueeze(1) - shifts).clamp_min(0)
        from_input = input_ids.long()[batch_index]
        hist_col = (n - 2 - (shifts.unsqueeze(0) - offset.unsqueeze(1) - 1)).clamp(0, n - 2)
        from_history = history.long()[req].gather(1, hist_col)
        tokens = torch.where(from_batch, from_input, from_history)
        blocked = positions.long().unsqueeze(1) < shifts
        return compute_engram_hash_ids(
            tokens,
            blocked,
            pad_id=self.pad_id,
            token_map=self.token_map,
            multipliers=self.multipliers,
            primes=self.primes,
            offsets=self.offsets,
        )


@dataclass(frozen=True)
class EngramShardPlan:
    num_embeddings: int
    dim: int
    block_size: int
    rank: int
    world_size: int
    row_start: int
    row_end: int

    @classmethod
    def build(
        cls, num_embeddings: int, dim: int, *, rank: int, world_size: int, block_size: int = 32
    ) -> "EngramShardPlan":
        if not 0 <= rank < world_size:
            raise ValueError(f"invalid Engram shard {rank=}, {world_size=}")
        if dim % block_size:
            raise ValueError(f"Engram dim {dim} is not divisible by block size {block_size}")
        return cls(
            num_embeddings=int(num_embeddings),
            dim=int(dim),
            block_size=int(block_size),
            rank=int(rank),
            world_size=int(world_size),
            row_start=num_embeddings * rank // world_size,
            row_end=num_embeddings * (rank + 1) // world_size,
        )

    @property
    def rows(self) -> int:
        return self.row_end - self.row_start

    @property
    def weight_bytes(self) -> int:
        return self.rows * self.dim

    @property
    def scale_bytes(self) -> int:
        return self.rows * (self.dim // self.block_size)

    @property
    def nbytes(self) -> int:
        return self.weight_bytes + self.scale_bytes


def _advise_huge_pages(bank: HostBank) -> None:
    obj = bank.memoryview().obj
    if isinstance(obj, mmap.mmap) and hasattr(mmap, "MADV_HUGEPAGE"):
        obj.madvise(mmap.MADV_HUGEPAGE)


def _collapse_huge_pages(bank: HostBank, tries: int = 3) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    for attempt in range(tries):
        if libc.madvise(ctypes.c_void_p(bank.addr), ctypes.c_size_t(bank.nbytes), _MADV_COLLAPSE) == 0:
            return 0
        error = ctypes.get_errno()
        if error != 11 or attempt + 1 == tries:  # EAGAIN is worth retrying.
            return error
        time.sleep(1.0)
    return 0


class EngramHostTable:
    """One rank's contiguous resident Engram shard: weights followed by scales."""

    def __init__(
        self,
        plan: EngramShardPlan,
        *,
        device: torch.device | None = None,
        prefetch: bool = True,
    ) -> None:
        self.plan = plan
        self.bank = HostBank((max(1, plan.nbytes),), torch.uint8, backing="mmap")
        _advise_huge_pages(self.bank)  # before the first page fault
        raw = self.bank.tensor[: plan.nbytes]
        self.weight = raw[: plan.weight_bytes].view(torch.float8_e4m3fn).view(plan.rows, plan.dim)
        self.scale = raw[plan.weight_bytes :].view(torch.float8_e8m0fnu).view(
            plan.rows, plan.dim // plan.block_size
        )
        self._device = device
        self._stream = None
        if device is not None and prefetch:
            self._stream = torch.cuda.Stream(device=device)
        self._staging: torch.Tensor | None = None
        self._graph_staging: dict[int, torch.Tensor] = {}
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def nbytes(self) -> int:
        return self.plan.nbytes

    def finish_load(self, *, pin: bool = True, collapse: bool = True) -> int:
        """Collapse base pages when possible, then register the filled mapping."""

        collapse_errno = _collapse_huge_pages(self.bank) if collapse and self.nbytes else 0
        if pin and torch.cuda.is_available():
            self.bank.pin()
        return collapse_errno

    def _stage(self, rows: int, device: torch.device) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            result = self._graph_staging.get(rows)
            if result is None:
                result = torch.empty((rows, self.plan.dim), dtype=torch.bfloat16, device=device)
                self._graph_staging[rows] = result
            return result
        if self._staging is None or self._staging.device != device or self._staging.shape[0] < rows:
            self._staging = torch.empty((rows, self.plan.dim), dtype=torch.bfloat16, device=device)
        return self._staging[:rows]

    def _gather(self, row_ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self.bank.residency is not HostResidency.PINNED:
            raise RuntimeError("Engram host table must be filled and pinned before GPU lookup")
        from freetoken.kernel.triton.dsv41.engram_gather import engram_gather_rows

        base = device_ptr(self.bank.tensor)
        return engram_gather_rows(
            base,
            base + self.plan.weight_bytes,
            row_ids.reshape(-1),
            out,
            dim=self.plan.dim,
            block_size=self.plan.block_size,
            row_lo=self.plan.row_start,
            row_hi=self.plan.row_end,
        )

    def prefetch(self, row_ids: torch.Tensor) -> None:
        if self._stream is None or row_ids.numel() == 0:
            return
        out = self._stage(row_ids.numel(), row_ids.device)
        self._stream.wait_stream(torch.cuda.current_stream(row_ids.device))
        if not torch.cuda.is_current_stream_capturing():
            row_ids.record_stream(self._stream)
        with torch.cuda.stream(self._stream):
            self._gather(row_ids, out)
        self._pending = (row_ids, out)

    def lookup(
        self,
        row_ids: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        reduce: bool = True,
        communicator: DistributedCommunicator | None = None,
    ) -> torch.Tensor:
        pending, self._pending = self._pending, None
        if pending is not None:
            torch.cuda.current_stream(row_ids.device).wait_stream(self._stream)
        if pending is not None and pending[0] is row_ids:
            rows = pending[1]
        else:
            with profile_range("DSV41/Engram/GatherUVA"):
                rows = self._gather(row_ids, self._stage(row_ids.numel(), row_ids.device))
        if reduce and self.plan.world_size > 1:
            with profile_range("DSV41/Engram/AllReduce"):
                rows = (communicator or DistributedCommunicator()).all_reduce(rows)
        rows = rows.view(*row_ids.shape, self.plan.dim)
        if out is None:
            return rows
        out.copy_(rows)
        return out


def _safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size)), 8 + size


def _checkpoint_weight_map(model_path: str) -> dict[str, str]:
    with open(os.path.join(model_path, "model.safetensors.index.json"), encoding="utf-8") as handle:
        return json.load(handle)["weight_map"]


def _tensor_range(path: str, name: str, dtype: str, shape: tuple[int, ...]) -> tuple[int, int]:
    header, base = _safetensors_header(path)
    try:
        meta = header[name]
    except KeyError as exc:
        raise KeyError(f"{name} is absent from {path}") from exc
    if meta["dtype"] != dtype or tuple(meta["shape"]) != shape:
        raise ValueError(f"{name}: expected {dtype} {shape}, got {meta['dtype']} {tuple(meta['shape'])}")
    begin, end = meta["data_offsets"]
    return base + begin, end - begin


def load_engram_host_table(
    model_path: str,
    *,
    layer_id: int,
    num_embeddings: int,
    dim: int,
    rank: int,
    world_size: int,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    pin: bool = True,
    collapse: bool = True,
    workers: int = 8,
    chunk: int = 8 << 20,
    reader: Callable[..., int] = read_range_into,
) -> EngramHostTable:
    """Read only this rank's row interval from an official safetensors checkpoint."""

    plan = EngramShardPlan.build(
        num_embeddings, dim, rank=rank, world_size=world_size, block_size=block_size
    )
    table = EngramHostTable(plan)
    weight_name = f"layers.{layer_id}.engram.embed.weight"
    scale_name = f"layers.{layer_id}.engram.embed.scale"
    weight_map = _checkpoint_weight_map(model_path)
    try:
        weight_path = os.path.join(model_path, weight_map[weight_name])
        scale_path = os.path.join(model_path, weight_map[scale_name])
    except KeyError as exc:
        raise KeyError(f"checkpoint has no complete Engram table for layer {layer_id}") from exc

    weight_offset, weight_bytes = _tensor_range(
        weight_path, weight_name, "F8_E4M3", (num_embeddings, dim)
    )
    scale_cols = dim // block_size
    scale_offset, scale_bytes = _tensor_range(
        scale_path, scale_name, "F8_E8M0", (num_embeddings, scale_cols)
    )
    assert weight_bytes == num_embeddings * dim
    assert scale_bytes == num_embeddings * scale_cols
    bar = byte_bar(plan.nbytes, f"Loading Engram layer {layer_id} rank {rank}/{world_size}")
    try:
        target = table.bank.memoryview()
        reader(
            target,
            weight_path,
            file_offset=weight_offset + plan.row_start * dim,
            nbytes=plan.weight_bytes,
            dest_offset=0,
            workers=workers,
            chunk=chunk,
        )
        bar.update(plan.weight_bytes)
        reader(
            target,
            scale_path,
            file_offset=scale_offset + plan.row_start * scale_cols,
            nbytes=plan.scale_bytes,
            dest_offset=plan.weight_bytes,
            workers=workers,
            chunk=chunk,
        )
        bar.update(plan.scale_bytes)
    finally:
        bar.close()
    table.finish_load(pin=pin, collapse=collapse)
    return table


__all__ = [
    "EngramHasher",
    "EngramHostTable",
    "EngramLayout",
    "EngramShardPlan",
    "build_compressed_token_map",
    "compute_engram_hash_ids",
    "compute_hash_multipliers",
    "load_engram_host_table",
]
