"""Opt-in, bounded, content-free MoE admission trace for offline eviction-policy replay.

``FREETOKEN_MOE_ADMISSION_TRACE_DIR=/abs/existing/dir`` enables it; unset adds no work anywhere.
Each rank appends one fixed-width int32 record per admission call to a device ring. The cursor
and a forward sequence number live in device memory, so captured decode graphs record the real
routes of every replay (the recorder kernels must therefore exist before graph capture; the
engine attaches the trace right after building the cache). At the first scheduler idle epoch the
trace arms (state snapshot, recording enabled); every later idle epoch -- or a forward whose
predicted records would overflow the ring -- flushes the ring to ``rank<R>-flush<NNNN>.npz`` plus
a JSON sidecar holding the host forward log (sequence, phase, rows, request uids, chunk sizes,
prefill-buffer invalidations, phase demotions) and an admission-state snapshot, so an offline
replay can be validated end to end.

Content: layer ids, storage-local expert ids, cache slots, counts and request uids only -- never
token ids, text, hidden states or routing scores. Bounds: ``..._RECORDS`` ring records per rank
(default 131072) and ``..._MAX_FILES`` flushes (default 256), after which recording disarms.

Record (int32, width ``8 + 3 * rk``; ``rk = max(max_rows * top_k, ceil(experts / 32))``):

    0 seq | 1 kind | 2 layer | 3 n_ids | 4 flags | 5 nonzero | 6 n_missing | 7 n_fetch |
    ids[rk] | fill_slots[rk] | fill_experts[rk]

kind 1 = ``ensure_experts`` (GPU-target LRU), 2 = ``ensure_experts_hybrid``, 3 = sparse prefill
(``ids`` holds the active-expert bitmask, 32 experts per word; no fill plan). flags bit 0 =
route weights supplied (the empty-route guard), bit 1 = CPU computes misses (background
admission), bits 8..15 = its miss limit. ``nonzero`` counts nonzero route weights (-1 without).

The CUDA recorders are NOT serving-qualified until the campaign's GPU lab passes
(dsv41-flash-results/2026-10-13-four-lever-campaign/item2_eviction/gpu_lab_admission_trace.py).
``from_env`` compiles and loads every recorder variant while disabled (``warm``), at engine
construction -- before any collective is queued or graph captured -- so no first-use module
load can land behind an in-flight NCCL kernel.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

try:  # Triton is only needed for the CUDA recorders.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = tl = None

TRACE_DIR_ENV = "FREETOKEN_MOE_ADMISSION_TRACE_DIR"
TRACE_RECORDS_ENV = "FREETOKEN_MOE_ADMISSION_TRACE_RECORDS"
TRACE_FILES_ENV = "FREETOKEN_MOE_ADMISSION_TRACE_MAX_FILES"
KIND_LRU, KIND_HYBRID, KIND_SPARSE = 1, 2, 3
HEADER = 8
STATE_FIELDS = ("slot_for_id", "id_of_slot", "usage", "step", "expert_recency")


def trace_settings() -> dict | None:
    """Strict parse of the opt-in environment; None when disabled."""
    directory = os.environ.get(TRACE_DIR_ENV)
    if directory is None or directory == "":
        return None
    path = Path(directory)
    if not path.is_absolute() or not path.is_dir() or not os.access(path, os.W_OK):
        raise ValueError(f"{TRACE_DIR_ENV} must be an existing writable absolute directory")

    def bounded(name, default, lo, hi):
        raw = os.environ.get(name)
        value = default if raw is None else int(raw)
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
        return value

    return dict(directory=path, capacity=bounded(TRACE_RECORDS_ENV, 131072, 1024, 1 << 20),
                max_files=bounded(TRACE_FILES_ENV, 256, 1, 100000))


if triton is not None:
    @triton.jit(do_not_specialize=["layer", "kind", "n_ids", "flags"])
    def _trace_pre(ids_ptr, w_ptr, ring_ptr, cursor_ptr, cur_ptr, seq_ptr, overflow_ptr, enabled_ptr,
                   layer, kind, n_ids, flags, CAP: tl.constexpr, W: tl.constexpr, RK: tl.constexpr,
                   BLOCK: tl.constexpr, HAS_W: tl.constexpr):
        if tl.load(enabled_ptr) != 0:
            idx = tl.load(cursor_ptr)
            tl.store(cur_ptr, idx)
            tl.store(cursor_ptr, idx + 1)
            if idx < CAP:
                base = ring_ptr + idx.to(tl.int64) * W
                k = tl.arange(0, BLOCK)
                lane = k < RK
                live = k < n_ids
                ids = tl.load(ids_ptr + k, mask=live, other=-1)
                nonzero = -1
                if HAS_W:
                    w = tl.load(w_ptr + k, mask=live, other=0.0)
                    nonzero = tl.sum((live & (w != 0.0)).to(tl.int32))
                tl.store(base + 0, tl.load(seq_ptr))
                tl.store(base + 1, kind)
                tl.store(base + 2, layer)
                tl.store(base + 3, n_ids)
                tl.store(base + 4, flags)
                tl.store(base + 5, nonzero)
                tl.store(base + 6, -1)
                tl.store(base + 7, -1)
                # Record fields start after the 8-word header (HEADER).
                tl.store(base + 8 + k, ids.to(tl.int32), mask=lane)
                tl.store(base + 8 + RK + k, -1, mask=lane)
                tl.store(base + 8 + 2 * RK + k, -1, mask=lane)
            else:
                tl.store(overflow_ptr, tl.load(overflow_ptr) + 1)

    @triton.jit(do_not_specialize=["layer", "n_routes"])
    def _trace_sparse_pre(routes_ptr, ring_ptr, cursor_ptr, cur_ptr, seq_ptr, overflow_ptr, enabled_ptr,
                          layer, n_routes, E: tl.constexpr, CAP: tl.constexpr, W: tl.constexpr,
                          RK: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK: tl.constexpr):
        if tl.load(enabled_ptr) != 0:
            idx = tl.load(cursor_ptr)
            tl.store(cur_ptr, idx)
            tl.store(cursor_ptr, idx + 1)
            if idx < CAP:
                base = ring_ptr + idx.to(tl.int64) * W
                expert = tl.arange(0, BLOCK_E)
                present = tl.zeros((BLOCK_E,), dtype=tl.int1)
                for r in tl.range(n_routes):
                    present = present | (expert == tl.load(routes_ptr + r))
                present = present & (expert < E)
                one = tl.full((BLOCK_E,), 1, tl.int64)
                for j in tl.static_range(RK):
                    bits = tl.where(present & (expert // 32 == j), one << (expert % 32).to(tl.int64), 0)
                    tl.store(base + 8 + j, tl.sum(bits).to(tl.int32))  # two's-complement word
                tl.store(base + 0, tl.load(seq_ptr))
                tl.store(base + 1, 3)
                tl.store(base + 2, layer)
                tl.store(base + 3, E)
                tl.store(base + 4, 0)
                tl.store(base + 5, -1)
                tl.store(base + 6, -1)
                tl.store(base + 7, -1)
                k = tl.arange(0, BLOCK)
                lane = k < RK
                tl.store(base + 8 + RK + k, -1, mask=lane)
                tl.store(base + 8 + 2 * RK + k, -1, mask=lane)
            else:
                tl.store(overflow_ptr, tl.load(overflow_ptr) + 1)

    @triton.jit
    def _trace_post(ring_ptr, cur_ptr, missing_ptr, count_ptr, evict_ptr, src_ptr, enabled_ptr,
                    CAP: tl.constexpr, W: tl.constexpr, RK: tl.constexpr, BLOCK: tl.constexpr,
                    HYBRID: tl.constexpr, PLAN: tl.constexpr):
        if tl.load(enabled_ptr) != 0:
            idx = tl.load(cur_ptr)
            if idx < CAP:
                base = ring_ptr + idx.to(tl.int64) * W
                n_fetch = tl.load(count_ptr).to(tl.int32)
                if HYBRID:
                    n_missing = tl.load(missing_ptr).to(tl.int32)
                else:
                    n_missing = n_fetch
                tl.store(base + 6, n_missing)
                tl.store(base + 7, n_fetch)
                if PLAN:
                    k = tl.arange(0, BLOCK)
                    plan = (k < RK) & (k < n_fetch)
                    tl.store(base + 8 + RK + k, tl.load(evict_ptr + k, mask=plan, other=-1), mask=plan)
                    tl.store(base + 8 + 2 * RK + k, tl.load(src_ptr + k, mask=plan, other=-1), mask=plan)


class AdmissionTrace:
    """Per-rank recorder bound to one OffloadMoeCache (see the module docstring)."""

    def __init__(self, cache, *, directory: Path, rank: int, top_k: int, max_rows: int,
                 capacity: int, max_files: int) -> None:
        self.cache, self.rank, self.directory = cache, int(rank), Path(directory)
        self.top_k, self.max_rows = int(top_k), int(max_rows)
        self.rk = max(self.max_rows * self.top_k, math.ceil(cache.num_experts / 32))
        self.width = HEADER + 3 * self.rk
        self.capacity, self.max_files = int(capacity), int(max_files)
        device = cache.id_of_slot.device
        self.ring = torch.full((self.capacity, self.width), -1, dtype=torch.int32, device=device)
        self.cursor = torch.zeros((1,), dtype=torch.int32, device=device)
        self.cur = torch.zeros((1,), dtype=torch.int32, device=device)
        self.seq_dev = torch.zeros((1,), dtype=torch.int32, device=device)
        self.overflow = torch.zeros((1,), dtype=torch.int32, device=device)
        self.enabled = torch.zeros((1,), dtype=torch.int32, device=device)
        self.armed = False
        self.seq = 0
        self.files = 0
        self.predicted = 0
        self.forwards: list[dict] = []
        self.snapshot_start: dict | None = None

    @classmethod
    def from_env(cls, cache, *, rank: int, top_k: int, max_rows: int):
        settings = trace_settings()
        if settings is None:
            return None
        trace = cls(cache, directory=settings["directory"], rank=rank, top_k=top_k,
                    max_rows=max_rows, capacity=settings["capacity"], max_files=settings["max_files"])
        trace.warm()
        return trace

    def warm(self) -> None:
        """Compile/load every CUDA recorder variant while disabled; writes no record.

        Startup only (host sync), before any collective or graph capture."""
        if not self.ring.is_cuda:
            return
        if self.armed or int(self.enabled.item()):
            raise RuntimeError("warm the admission trace before arming it")
        device = self.ring.device
        ids = torch.zeros((1, self.top_k), dtype=torch.int32, device=device)
        for kind in (KIND_LRU, KIND_HYBRID):
            for weights in (None, torch.ones(ids.shape, dtype=torch.float32, device=device),
                            torch.ones(ids.shape, dtype=torch.bfloat16, device=device)):
                self.record_pre(kind, 0, ids, weights)
            self.record_post(kind)
        self.record_sparse_pre(0, ids)
        self.record_post(KIND_SPARSE)
        torch.cuda.synchronize(device)

    # -- device-side recording (graph-capturable; inert while not armed)
    def record_pre(self, kind: int, layer: int, ids: torch.Tensor, route_weights=None, *,
                   compute_misses_on_cpu: bool = False, cpu_miss_limit: int = 0) -> None:
        n_ids = ids.numel()
        if n_ids > self.rk:
            raise ValueError(f"admission call with {n_ids} ids exceeds the trace width {self.rk}")
        flags = int(route_weights is not None) | (int(compute_misses_on_cpu) << 1) | (int(cpu_miss_limit) << 8)
        if ids.is_cuda:
            _trace_pre[(1,)](ids, route_weights if route_weights is not None else ids, self.ring,
                             self.cursor, self.cur, self.seq_dev, self.overflow, self.enabled,
                             int(layer), int(kind), n_ids, flags, self.capacity, self.width, self.rk,
                             triton.next_power_of_2(self.rk), route_weights is not None, num_warps=1)
            return
        if not int(self.enabled.item()):
            return
        idx = int(self.cursor.item())
        self.cur.fill_(idx)
        self.cursor.fill_(idx + 1)
        if idx >= self.capacity:
            self.overflow += 1
            return
        row = self.ring[idx]
        row.fill_(-1)
        flat = ids.reshape(-1).to(torch.int32)
        row[HEADER:HEADER + n_ids] = flat
        nonzero = -1 if route_weights is None else int((route_weights.reshape(-1) != 0).sum())
        row[:6] = torch.tensor([self.seq_dev.item(), kind, layer, n_ids, flags, nonzero], dtype=torch.int32)

    def record_sparse_pre(self, layer: int, routes: torch.Tensor) -> None:
        experts = self.cache.num_experts
        if routes.is_cuda:
            _trace_sparse_pre[(1,)](routes, self.ring, self.cursor, self.cur, self.seq_dev, self.overflow,
                                    self.enabled, int(layer), routes.numel(), experts, self.capacity,
                                    self.width, self.rk, triton.next_power_of_2(experts),
                                    triton.next_power_of_2(self.rk), num_warps=4)
            return
        if not int(self.enabled.item()):
            return
        idx = int(self.cursor.item())
        self.cur.fill_(idx)
        self.cursor.fill_(idx + 1)
        if idx >= self.capacity:
            self.overflow += 1
            return
        row = self.ring[idx]
        row.fill_(-1)
        words = [0] * self.rk
        for e in set(int(x) for x in routes.reshape(-1).tolist()):
            if 0 <= e < experts:
                words[e // 32] |= 1 << (e % 32)
        row[HEADER:HEADER + self.rk] = torch.tensor(
            [w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32)
        row[:6] = torch.tensor([self.seq_dev.item(), KIND_SPARSE, layer, experts, 0, -1], dtype=torch.int32)

    def record_post(self, kind: int) -> None:
        cache = self.cache
        hybrid = kind == KIND_HYBRID
        plan = kind != KIND_SPARSE
        if cache.num_indices.is_cuda:
            _trace_post[(1,)](self.ring, self.cur, cache.num_missing_full, cache.num_indices,
                              cache.evict_slots, cache.src_indices, self.enabled, self.capacity,
                              self.width, self.rk, triton.next_power_of_2(self.rk), hybrid, plan,
                              num_warps=1)
            return
        if not int(self.enabled.item()):
            return
        idx = int(self.cur.item())
        if idx >= self.capacity:
            return
        row = self.ring[idx]
        n_fetch = int(cache.num_indices.reshape(-1)[0])
        row[6] = int(cache.num_missing_full.reshape(-1)[0]) if hybrid else n_fetch
        row[7] = n_fetch
        if plan:
            n = min(n_fetch, self.rk)
            row[HEADER + self.rk:HEADER + self.rk + n] = cache.evict_slots[:n].to(torch.int32)
            row[HEADER + 2 * self.rk:HEADER + 2 * self.rk + n] = cache.src_indices[:n].to(torch.int32)

    # -- host-side boundaries (never inside graph capture)
    def _snapshot(self) -> dict:
        cache = self.cache
        cache.finish_background_admission()
        if cache.id_of_slot.is_cuda:
            torch.cuda.synchronize(cache.id_of_slot.device)
        return {name: getattr(cache, name).detach().to("cpu").clone() for name in STATE_FIELDS}

    def arm(self) -> None:
        if self.armed or self.files >= self.max_files:
            return
        self.snapshot_start = self._snapshot()
        self.cursor.zero_()
        self.overflow.zero_()
        self.forwards = []
        self.predicted = 0
        self.armed = True
        self.enabled.fill_(1)

    def begin_forward(self, batch) -> None:
        if not self.armed:
            return
        layers = self.cache.num_layers
        expected = 2 * layers  # decode, or sparse prefill; doubled for shadow verification
        if self.predicted + expected + 4 * layers > self.capacity:
            self.flush("capacity")
            if not self.armed:
                return
        self.seq += 1
        self.seq_dev.fill_(self.seq)
        self.predicted += expected
        self.forwards.append(dict(
            seq=self.seq, phase="prefill" if batch.is_prefill else "decode",
            speculative=bool(getattr(batch, "speculative", False)), rows=len(batch.reqs),
            uids=[int(req.uid) for req in batch.reqs],
            new_tokens=int(getattr(batch, "log_new_tokens", 0) or 0),
            paired=getattr(batch, "paired_head", None) is not None,
            cone=getattr(batch, "dsv41_cone", None) is not None,
            invalidations=[], demoted=None))

    def note_invalidate(self, buffer_id: int) -> None:
        if self.armed and self.forwards:
            self.forwards[-1]["invalidations"].append(int(buffer_id))

    def note_materialize(self, layer: int) -> None:
        """Non-overlap prefill: the whole layer occupies slots [0, experts) (clock +1)."""
        if self.armed and self.forwards:
            self.forwards[-1].setdefault("materialized", []).append(int(layer))

    def note_demote(self, start: int, count: int) -> None:
        if self.armed and self.forwards:
            self.forwards[-1]["demoted"] = [int(start), int(count)]

    def flush(self, reason: str) -> Path | None:
        if not self.armed:
            return None
        if self.cache.id_of_slot.is_cuda:
            torch.cuda.synchronize(self.cache.id_of_slot.device)
        count = int(self.cursor.item())
        if count == 0 and not self.forwards:
            return None  # nothing happened since the last flush
        overflow = int(self.overflow.item())
        records = self.ring[: min(count, self.capacity)].to("cpu").numpy().copy()
        end = self._snapshot()
        index = self.files
        self.files += 1
        stem = self.directory / f"rank{self.rank}-flush{index:04d}"
        import numpy as np

        tmp = stem.with_suffix(".npz.tmp")
        with open(tmp, "wb") as handle:
            np.savez(handle, records=records,
                     **{f"start_{k}": v.numpy() for k, v in self.snapshot_start.items()},
                     **{f"end_{k}": v.numpy() for k, v in end.items()})
        os.replace(tmp, stem.with_suffix(".npz"))
        cache = self.cache
        from freetoken.moe import offload_kernels

        meta = dict(rank=self.rank, flush=index, reason=reason, wall_time=time.time(),
                    records=count, overflow=overflow, valid=overflow == 0 and count <= self.capacity,
                    width=self.width, rk=self.rk, top_k=self.top_k, max_rows=self.max_rows,
                    layers=cache.num_layers, experts=cache.num_experts, slots=cache.cache_size,
                    decode_target=cache.decode_target, hybrid_max_fetch=cache.hybrid_max_fetch,
                    hybrid_fetch_fraction=cache.hybrid_fetch_fraction,
                    hybrid_fetch_by_recency=bool(offload_kernels._HYBRID_FETCH_BY_RECENCY),
                    prefill_overlap=bool(cache.prefill_overlap),
                    prefill_expert_start=cache.prefill_expert_start,
                    prefill_expert_count=cache.prefill_expert_count,
                    sparse_prefill_max_tokens=cache.sparse_prefill_max_tokens,
                    forwards=self.forwards)
        tmp = stem.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, separators=(",", ":")) + "\n")
        os.replace(tmp, stem.with_suffix(".json"))
        # Continue from this exact state: the next file's start snapshot is this end snapshot.
        self.snapshot_start = end
        self.cursor.zero_()
        self.overflow.zero_()
        self.forwards = []
        self.predicted = 0
        if self.files >= self.max_files:
            self.enabled.zero_()
            self.armed = False
        return stem


__all__ = ["AdmissionTrace", "KIND_HYBRID", "KIND_LRU", "KIND_SPARSE", "TRACE_DIR_ENV",
           "TRACE_FILES_ENV", "TRACE_RECORDS_ENV", "trace_settings"]
