"""CPU-only isolation proof for the default-off DS-FP4 inactive prefill skip.

``FREETOKEN_DSV41_PREFILL_SKIP_INACTIVE`` (``fused_ds_fp4._PREFILL_SKIP_INACTIVE``)
may only change which grouped-GEMM rows the prefill kernel computes.  These tests
prove, without CUDA, that toggling it leaves every cache/admission observable of
the sparse (<=256-token slot-cache) and bulk (streamed full-layer, tiled and
untiled) prefill paths unchanged, and that no decode or graph code reads it.

The CUDA kernels are replaced by slot-dependent CPU references and the CUDA-only
LRU admission by a deterministic CPU stand-in that is identical in both modes;
real-kernel numerics and real admission are the maintenance GPU harness's job
(``dsv41-flash-results/lab_prefill_inactive_state.py``).
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import freetoken.layers.moe as moe_module
from freetoken.layers.moe import ExpertParallelOffloadMoELayer
from freetoken.moe import fused_ds_fp4 as fp4
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.partition import (
    ExpertPartition,
    ExpertStorageRange,
    localize_expert_routes_to_storage,
)

PACKAGE = Path(fp4.__file__).resolve().parents[1]
FLAG = "_PREFILL_SKIP_INACTIVE"
ENV = "FREETOKEN_DSV41_PREFILL_SKIP_INACTIVE"

# Small DS-FP4 geometry: 24 global experts, this rank stores [0, 12) and owns the
# prefill interval [4, 12), mirroring a worker whose storage starts below its
# prefill range.  2E = 24 slots are borrowed by the prefill double buffer.
H, I, K, GLOBAL, STORED, CACHE = 64, 32, 6, 24, 12, 40
PREFILL_START, PREFILL_COUNT = 4, 8
STORAGE = ExpertStorageRange(GLOBAL, global_offset=0, local_count=STORED)
OWNERSHIP = ExpertPartition(GLOBAL, world_size=3, rank=1, shard_counts=(4, 8, 12))


def _references(path: Path, names: set[str]) -> list[str]:
    """Enclosing qualified scopes of every load/store/import of ``names``."""
    found: list[str] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, scope + (child.name,))
                continue
            if (
                (isinstance(child, ast.Name) and child.id in names)
                or (isinstance(child, ast.Attribute) and child.attr in names)
                or (isinstance(child, ast.alias) and child.name in names)
                or (isinstance(child, ast.arg) and child.arg in names)
                or (isinstance(child, ast.keyword) and child.arg in names)
                or (isinstance(child, ast.Constant) and child.value in names)
            ):
                found.append(".".join(scope) or "<module>")
            visit(child, scope)

    visit(ast.parse(path.read_text(), filename=str(path)), ())
    return found


def _consumers(names: set[str]) -> dict[str, list[str]]:
    # Guard against inspecting another checkout's installed package.
    assert PACKAGE == Path(__file__).resolve().parents[2] / "python" / "freetoken", PACKAGE
    consumers = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        refs = _references(path, names)
        if refs:
            consumers[path.relative_to(PACKAGE).as_posix()] = sorted(set(refs))
    return consumers


def test_flag_is_read_only_by_the_prefill_wrapper_and_its_startup_warmup():
    # Decode (routed_experts_fp4/_grouped_decode/_decode_dsfp4_moe_kernel), CUDA
    # graph capture, background admission and phase priority must not appear.
    assert _consumers({FLAG, ENV}) == {
        "engine/engine.py": ["Engine._warmup_sparse_prefill_moe_kernels"],
        "moe/fused_ds_fp4.py": ["<module>", "routed_experts_fp4_prefill"],
    }


def test_inactive_sentinel_reaches_only_the_grouped_prefill_kernel():
    assert _consumers({"INACTIVE_EXPERT"}) == {
        "kernel/triton/dsv4/fused_moe.py": ["_prefill_dsfp4_moe_kernel"],
        "moe/fused_ds_fp4.py": ["_grouped_prefill"],
    }
    assert _consumers({"inactive_expert"}) == {
        "moe/fused_ds_fp4.py": ["_grouped_prefill", "routed_experts_fp4_prefill"],
    }


def test_flag_is_resolved_per_call_so_same_process_toggling_is_valid():
    # A default-argument or closure capture would make an OFF->ON toggle inside
    # one process (the GPU harness) silently test only one mode.
    assert FLAG in fp4.routed_experts_fp4_prefill.__code__.co_names
    assert FLAG not in fp4.routed_experts_fp4.__code__.co_names
    assert FLAG not in fp4._grouped_decode.__code__.co_names


class _RecordingCache(OffloadMoeCache):
    """Real CPU cache; only the two CUDA-only movement/admission calls are replaced."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.log: list[tuple] = []

    def _record(self, *event) -> None:
        self.log.append(event)

    def begin_prefill(self) -> None:
        self._record("begin_prefill")
        super().begin_prefill()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        self._record("prefetch_prefill_layer", layer_id)
        super().prefetch_prefill_layer(layer_id)

    def release_prefill_layer(self, layer_id: int) -> None:
        self._record("release_prefill_layer", layer_id)
        super().release_prefill_layer(layer_id)

    def materialize_layer(self, layer_id: int) -> None:  # pragma: no cover - guarded
        raise AssertionError("prefill overlap is enabled; materialize must not run")

    def ensure_sparse_prefill_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        # Deterministic stand-in for the CUDA LRU: identical inputs give identical
        # state, which is all this mode-isolation test relies on.
        self._record("ensure_sparse_prefill_experts", layer_id, expert_ids.clone())
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        self.step += 1
        flat = expert_ids.reshape(-1)
        for expert in sorted(set(flat.tolist())):
            slot = int(self.slot_for_id[layer_id, expert])
            if slot < 0:
                slot = int(torch.argmin(self.usage))
                old = int(self.id_of_slot[slot])
                if old >= 0:
                    self.slot_for_id.view(-1)[old] = -1
                self.id_of_slot[slot] = layer_id * self.num_experts + expert
                self.slot_for_id[layer_id, expert] = slot
            self.usage[slot] = int(self.step)
        flat.copy_(self.slot_for_id[layer_id, flat.long()])

    def copy_missing(self, *, num_indices=None) -> None:
        self._record("copy_missing", self._pending_src_layer, self._pending_whole_layer)


def _snapshot(cache: _RecordingCache) -> dict:
    state = {
        name: getattr(cache, name).clone()
        for name in (
            "slot_for_id", "id_of_slot", "usage", "step", "expert_recency",
            "lru_stats", "sparse_prefill_stats", "num_indices",
        )
    }
    state.update(
        prefill_hit_rows=cache.prefill_hit_rows,
        prefill_total_rows=cache.prefill_total_rows,
        buffer_layer=list(cache._prefill_buffer_layer),
        buffer_released=list(cache._prefill_buffer_released),
        pending=(cache._pending_src_layer, cache._pending_whole_layer),
        banks={name: bank.clone() for name, bank in cache.bank_caches.items()},
    )
    return state


def _assert_same(a, b, where: str) -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype, where
        assert torch.equal(a, b), where
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for key in a:
            _assert_same(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for index, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{where}[{index}]")
    else:
        assert a == b, where


def _cpu_align(slots, block, num_rows, include_sentinel):
    flat = slots.reshape(-1).long()
    numel = flat.numel()
    if not include_sentinel:
        assert torch.all((flat >= 0) & (flat < num_rows)), "OFF path needs valid ids"
    sorted_ids, expert_ids = [], []
    for expert in range(num_rows + 1 if include_sentinel else num_rows):
        routes = torch.nonzero(flat == expert).flatten().tolist()
        if routes:
            routes += [numel] * ((-len(routes)) % block)
            sorted_ids += routes
            expert_ids += [expert] * (len(routes) // block)
    assert sorted(r for r in sorted_ids if r < numel) == list(range(numel))
    return (torch.tensor(sorted_ids, dtype=torch.int32),
            torch.tensor(expert_ids, dtype=torch.int32),
            torch.tensor([len(sorted_ids)], dtype=torch.int64))


def _cpu_grouped(a, packed, c, tw, sorted_ids, expert_ids, num_valid, top_k, mul,
                 block, inactive_expert):
    """Slot-dependent reference: a wrong slot or a missed zero-write changes bits."""
    out = c.view(-1, c.shape[-1])
    for index, expert in enumerate(expert_ids.tolist()):
        for route in sorted_ids[index * block:(index + 1) * block].tolist():
            if route >= num_valid:
                continue
            if expert == inactive_expert:
                out[route] = 0
                continue
            weight = packed[expert].float() / 255
            value = weight @ a[route // top_k, ::2].float()
            if mul:
                value = value * tw[route]
            out[route] = value.to(out.dtype)


@pytest.fixture
def cpu_kernels(monkeypatch):
    trace: list[tuple] = []
    original = fp4.routed_experts_fp4_prefill

    def wrapper(x, slots, weights, *args, **kwargs):
        trace.append(("prefill_inputs", slots.clone(), weights.clone(), args[5]))
        return original(x, slots, weights, *args, **kwargs)

    def align(slots, block, num_rows, skip_inactive):
        trace.append(("align", slots.clone(), num_rows, skip_inactive))
        return _cpu_align(slots, block, num_rows, skip_inactive)

    def grouped(a, packed, _scale, c, tw, sorted_ids, expert_ids, _ntpp, num_valid,
                top_k, mul, cfg, inactive_expert=-1):
        trace.append(("grouped", inactive_expert))
        _cpu_grouped(a, packed, c, tw, sorted_ids, expert_ids, num_valid, top_k, mul,
                     cfg["BLOCK_SIZE_M"], inactive_expert)

    def swiglu(gate_up, limit):
        gate, up = gate_up.float().chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate.clamp(max=limit))
                * up.clamp(-limit, limit)).to(gate_up.dtype)

    def gemv(x, slots, weights, gate_up, _gus, down, _ds, limit,
             return_route_outputs=False, activation_block_size=128, **_):
        # The small-route fallback must not consult the flag; record its inputs.
        trace.append(("gemv", slots.clone()))
        T, top_k = slots.shape
        routes = x.new_empty((T, top_k, 2 * I))
        si, ei, _ = _cpu_align(slots, 1, gate_up.shape[0], False)
        _cpu_grouped(x, gate_up, routes, weights.reshape(-1), si, ei, T * top_k, top_k,
                     False, 1, -1)
        act = swiglu(routes, limit).reshape(T * top_k, I)
        down_out = x.new_empty((T, top_k, H))
        _cpu_grouped(act, down, down_out, weights.reshape(-1), si, ei, T * top_k, 1,
                     True, 1, -1)
        return down_out if return_route_outputs else down_out.sum(dim=1)

    monkeypatch.setattr(fp4, "routed_experts_fp4_prefill", wrapper)
    monkeypatch.setattr(fp4, "_align_prefill_routes", align)
    monkeypatch.setattr(fp4, "_grouped_prefill", grouped)
    monkeypatch.setattr(fp4, "act_quant_fp8_roundtrip", lambda x, _block: x)
    monkeypatch.setattr(fp4, "act_quant_fp8_inplace", lambda _x, _block: None)
    monkeypatch.setattr(fp4, "fused_swiglu", swiglu)
    monkeypatch.setattr(fp4, "routed_experts_fp4", gemv)
    monkeypatch.setattr(moe_module, "get_tp_info", lambda: SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(moe_module, "get_global_ctx", lambda: SimpleNamespace(
        batch=SimpleNamespace(is_prefill=True, is_moe_prefill=True, speculative=False)))
    # Bulk tiles of 256 rows stand in for production's 4096-token route tile.
    monkeypatch.setattr(moe_module, "_EP_PREFILL_ROUTE_TILE_TOKENS", 256)
    return trace


def _inputs(seed: int, tokens: int):
    generator = torch.Generator().manual_seed(seed)
    global_ids = torch.rand(tokens, GLOBAL, generator=generator).topk(K).indices.int()
    global_ids[0] = torch.arange(GLOBAL - K, GLOBAL, dtype=torch.int32)  # all foreign
    weights = torch.rand(tokens, K, generator=generator) + 0.05
    weights, ids = localize_expert_routes_to_storage(weights, global_ids, OWNERSHIP, STORAGE)
    hidden = (torch.randn(tokens, H, generator=generator) * 0.25).to(torch.bfloat16)
    return hidden, weights.contiguous(), ids.to(torch.int32).contiguous()


# (label, layer, tokens): sparse grouped, sparse GEMV fallback, untiled bulk on
# both layers (layer 0 begins the overlap prefill), tiled bulk, then sparse again
# after bulk invalidated the borrowed double-buffer slots.
STEPS = (
    ("sparse-grouped", 0, 200), ("sparse-gemv", 1, 100),
    ("bulk", 0, 300), ("bulk", 1, 300),
    ("bulk-tiled", 0, 600), ("bulk-tiled", 1, 600),
    ("sparse-after-bulk", 1, 200),
)


def _run_mode(monkeypatch, trace, enabled: bool) -> dict:
    monkeypatch.setattr(fp4, FLAG, enabled)
    trace.clear()
    generator = torch.Generator().manual_seed(7106)
    sources = {
        name: [torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
               for _ in range(2)]
        for name, shape in (
            ("gate_up_packed", (STORED, 2 * I, H // 2)),
            ("gate_up_scale", (STORED, 2 * I, H // 32)),
            ("down_packed", (STORED, H, I // 2)),
            ("down_scale", (STORED, H, I // 32)),
        )
    }
    cache = _RecordingCache(2, STORED, CACHE, torch.device("cpu"), prefill_overlap=True,
                            sparse_prefill_max_tokens=256, quant_format="ds_fp4")
    cache.set_bank_sources(sources)
    # Slot caches are torch.empty; never-written slots would otherwise differ
    # between the two fresh caches for reasons unrelated to the flag.
    for bank in cache.bank_caches.values():
        bank.zero_()
    cache.set_prefill_expert_range(PREFILL_START, PREFILL_COUNT)
    captured: list[torch.Tensor] = []
    layer = object.__new__(ExpertParallelOffloadMoELayer)
    layer.offload_cache = cache
    layer.num_experts = STORED
    layer.top_k = K
    layer.hidden_size = H
    layer.swiglu_limit = 7.0
    layer.dsfp4_activation_block_size = 32
    layer.skip_inactive_prefill_routes = False  # production EP flag stays off
    layer.packed_prefill_root = None
    layer._maybe_all_reduce = lambda routes: captured.append(routes.clone()) or routes
    result = {"steps": [], "log": cache.log}
    for index, (label, layer_id, tokens) in enumerate(STEPS):
        layer.layer_id = layer_id
        hidden, weights, ids = _inputs(610 + index, tokens)
        pristine = (hidden.clone(), weights.clone(), ids.clone())
        captured.clear()
        trace_start = len(trace)
        layer.routed_forward(hidden, weights, ids)
        _assert_same((hidden, weights, ids), pristine, f"{label}: inputs mutated")
        result["steps"].append(dict(
            label=label, weights=weights, routes=list(captured),
            trace=trace[trace_start:], state=_snapshot(cache)))
    return result


def test_toggle_preserves_cache_admission_and_streaming_bookkeeping(monkeypatch, cpu_kernels):
    off = _run_mode(monkeypatch, cpu_kernels, False)
    on = _run_mode(monkeypatch, cpu_kernels, True)

    _assert_same(off["log"], on["log"], "cache call log")
    exercised = 0
    for a, b in zip(off["steps"], on["steps"], strict=True):
        label = a["label"]
        _assert_same(a["state"], b["state"], f"{label}: cache state")
        kinds = [event[0] for event in a["trace"]]
        assert kinds == [event[0] for event in b["trace"]], label
        for x, y in zip(a["trace"], b["trace"]):
            if x[0] in ("prefill_inputs", "gemv"):
                # Identical ids reach the kernel wrapper in both modes: the layer's
                # admission/remap never sees the flag.
                _assert_same(x, y, f"{label}: kernel inputs")
        # Pair each grouped-path kernel call with its aligner and two GEMM
        # launches; the small-route GEMV fallback must never reach them.
        inputs = [e for e in a["trace"] if e[0] == "prefill_inputs"]
        grouped_inputs = [e for e in inputs if e[1].numel() >= fp4._GROUPED_MIN_ROUTES]
        for mode in (a, b):
            assert [e[0] for e in mode["trace"] if e[0] in ("align", "grouped")] == (
                ["align", "grouped", "grouped"] * len(grouped_inputs)
            ), label
        off_align = [e for e in a["trace"] if e[0] == "align"]
        on_align = [e for e in b["trace"] if e[0] == "align"]
        off_gemm = [e[1] for e in a["trace"] if e[0] == "grouped"]
        on_gemm = [e[1] for e in b["trace"] if e[0] == "grouped"]
        for index, (_, slots, weights, rows) in enumerate(grouped_inputs):
            assert off_align[index][3] is False and on_align[index][3] is True, label
            assert off_align[index][2] == on_align[index][2] == rows, label
            _assert_same(off_align[index][1], slots, f"{label}: OFF aligner input")
            _assert_same(on_align[index][1], torch.where(weights == 0, rows, slots),
                         f"{label}: ON sentinel bucket")
            assert off_gemm[2 * index:2 * index + 2] == [-1, -1], label
            assert on_gemm[2 * index:2 * index + 2] == [rows, rows], label
            exercised += int(bool((weights == 0).any() and (weights != 0).any()))
        assert len(a["routes"]) == len(b["routes"]) >= 1, label
        tiles = a["weights"].split(a["routes"][0].shape[0])
        for weights, r_off, r_on in zip(tiles, a["routes"], b["routes"], strict=True):
            owned = weights != 0
            _assert_same(r_off[owned], r_on[owned], f"{label}: owned route lanes")
            assert torch.all(r_on[~owned] == 0), f"{label}: ON inactive lanes not zeroed"
            assert torch.all(r_off[~owned] == 0), f"{label}: OFF inert lanes nonzero"
    # The ON branch must really have run on mixed active/inactive inputs.
    assert exercised >= 5


def test_sparse_admission_query_uses_cache_safe_ids_in_both_modes(monkeypatch, cpu_kernels):
    for enabled in (False, True):
        result = _run_mode(monkeypatch, cpu_kernels, enabled)
        queries = [event for event in result["log"] if event[0] == "ensure_sparse_prefill_experts"]
        # sparse-grouped, sparse-gemv, sparse-after-bulk; bulk never admits.
        assert [event[1] for event in queries] == [0, 1, 1]
        for _, _, ids in queries:
            assert torch.all((ids >= 0) & (ids < STORED)), "sentinel leaked into LRU query"
        assert not any(event[0] == "materialize_layer" for event in result["log"])
