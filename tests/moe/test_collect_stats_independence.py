"""``--moe-collect-stats`` is telemetry only: no admission, eviction or routing reads it.

Pins the audited set of readers/writers (a new reference fails until it is audited), proves
every stats-guarded block only writes counters, checks that FlashLib's ``COLLECT_STATS``
specialization only gates a final counter atomic, and runs the real CPU hybrid decode
orchestration with statistics on and off over identical changing routes.
"""
from __future__ import annotations

import ast
import inspect
import random
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import freetoken
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache

ROOT = Path(freetoken.__file__).resolve().parent
NAMES = {"collect_stats", "moe_collect_stats"}

# (file, enclosing scope) -> why the reference is telemetry-only.  Audited 2026-10-13.
AUDITED = {
    ("engine/config.py", "EngineConfig"): "config field (default False)",
    ("engine/engine.py", "Engine._init_offload_moe_cache"): "copies the flag onto the cache before warmups/capture",
    ("layers/moe.py", "OffloadMoELayer._decode_hybrid"): "guards record_decode_stats_hybrid only",
    ("models/deepseek_v4/moe.py", "DSV4OffloadMoELayer._prefill_routed"): "guards the no-op record_decode_stats (V4, not V4.1)",
    ("moe/background_admission.py", "BackgroundAdmission.decode"): "guards record_decode_stats_hybrid and stat_cpu_admitted",
    ("moe/offload_cache.py", "OffloadMoeCache.__post_init__"): "initialises the flag to False",
    ("moe/offload_cache.py", "OffloadMoeCache.ensure_experts"): "selects the lru_stats row passed as stats=",
    ("moe/offload_cache.py", "OffloadMoeCache.ensure_sparse_prefill_experts"): "selects the sparse_prefill_stats row",
    ("scheduler/scheduler.py", "Scheduler.__init__"): "resets counters at startup",
    ("scheduler/scheduler.py", "Scheduler._log_moe_cache_stats"): "idle-only log of the counters",
    ("server/args.py", "parse_args"): "CLI flag definition",
}


def _references() -> set[tuple[str, str]]:
    found = set()
    for path in sorted(ROOT.rglob("*.py")):
        source = path.read_text()
        if "collect_stats" not in source:
            continue
        stack: list[str] = []

        class Visitor(ast.NodeVisitor):
            def _scoped(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

            def _hit(self):
                found.add((str(path.relative_to(ROOT)), ".".join(stack)))

            def visit_Attribute(self, node):
                if node.attr in NAMES:
                    self._hit()
                self.generic_visit(node)

            def visit_Name(self, node):
                if node.id in NAMES:
                    self._hit()

            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value in NAMES | {"--moe-collect-stats"}:
                    self._hit()

        Visitor().visit(ast.parse(source))
    return found


def test_every_collect_stats_reference_is_audited():
    assert _references() == set(AUDITED)


def _function(module_path: str, qualname: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / module_path).read_text())
    cls_name, fn_name = qualname.split(".")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)


def _mentions_stats(test: ast.expr) -> bool:
    return any((isinstance(n, ast.Attribute) and n.attr in NAMES)
               or (isinstance(n, ast.Constant) and n.value in NAMES)
               for n in ast.walk(test))


def _guarded_bodies(fn: ast.FunctionDef) -> list[list[ast.stmt]]:
    return [n.body for n in ast.walk(fn) if isinstance(n, ast.If) and _mentions_stats(n.test)]


def _only_counter_writes(body: list[ast.stmt]) -> bool:
    """Each statement records a counter: a record_decode_stats* call, a local temporary,
    or an augmented assignment into a ``stat_*`` tensor."""
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if not (isinstance(func, ast.Attribute) and func.attr.startswith("record_decode_stats")):
                return False
        elif isinstance(stmt, ast.Assign):
            if not all(isinstance(t, ast.Name) for t in stmt.targets):
                return False
        elif isinstance(stmt, ast.AugAssign):
            target = stmt.target.value if isinstance(stmt.target, ast.Subscript) else stmt.target
            if not (isinstance(target, ast.Attribute) and target.attr.startswith("stat_")):
                return False
        else:
            return False
    return True


@pytest.mark.parametrize("module_path,qualname,blocks", [
    ("layers/moe.py", "OffloadMoELayer._decode_hybrid", 1),
    ("moe/background_admission.py", "BackgroundAdmission.decode", 2),
    ("models/deepseek_v4/moe.py", "DSV4OffloadMoELayer._prefill_routed", 1),
])
def test_stats_guarded_blocks_only_record_counters(module_path, qualname, blocks):
    bodies = _guarded_bodies(_function(module_path, qualname))
    assert len(bodies) == blocks
    assert all(_only_counter_writes(body) for body in bodies)


def test_background_admission_local_is_not_reused_outside_its_stats_block():
    fn = _function("moe/background_admission.py", "BackgroundAdmission.decode")
    guarded = {id(n) for body in _guarded_bodies(fn) for stmt in body for n in ast.walk(stmt)}
    outside = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "admitted"
               and id(n) not in guarded]
    assert outside == []


def test_hybrid_recorder_writes_only_counters():
    fn = _function("moe/offload_cache.py", "OffloadMoeCache.record_decode_stats_hybrid")
    for stmt in fn.body:
        if isinstance(stmt, (ast.Assert, ast.Expr)) and not isinstance(getattr(stmt, "value", None), ast.Call):
            continue  # assert / docstring
        if isinstance(stmt, ast.Assign):
            assert all(isinstance(t, ast.Name) for t in stmt.targets)  # locals only
            continue
        assert isinstance(stmt, ast.AugAssign)
        target = stmt.target.value if isinstance(stmt.target, ast.Subscript) else stmt.target
        assert isinstance(target, ast.Attribute) and target.attr.startswith("stat_")


def test_ensure_paths_pass_statistics_only_as_the_stats_argument():
    for qualname in ("OffloadMoeCache.ensure_experts", "OffloadMoeCache.ensure_sparse_prefill_experts"):
        fn = _function("moe/offload_cache.py", qualname)
        uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "stats"
                and isinstance(n.ctx, ast.Load)]
        keywords = [k for n in ast.walk(fn) if isinstance(n, ast.Call) for k in n.keywords
                    if k.arg == "stats" and isinstance(k.value, ast.Name) and k.value.id == "stats"]
        assert uses and len(uses) == len(keywords)


def test_flashlib_collect_stats_only_gates_the_final_counter_atomic():
    import importlib

    # The package re-exports the lru_ensure *function*; inspect the kernel module itself.
    module = importlib.import_module("flashlib.kernels.slot_cache.triton.lru_ensure")

    tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("_lru_ensure_kernel", "_lru_ensure_insert_kernel"):
        fn = functions[name]
        gated = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == "COLLECT_STATS"]
        assert len(gated) == 1 and gated[0] is fn.body[-1]  # last statement of the kernel
        call = gated[0].body[0].value
        assert len(gated[0].body) == 1 and call.func.id == "_stats"
        stats_reads = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "stats_ptr"]
        # The parameter (an ast.arg) is read exactly once: inside the gated _stats call.
        assert len(stats_reads) == 1 and any(n is stats_reads[0] for n in ast.walk(gated[0]))
    stats = functions["_stats"]
    stores = [n for n in ast.walk(stats) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr in ("store", "atomic_add")]
    assert [c.func.attr for c in stores] == ["atomic_add"]
    assert isinstance(stores[0].args[0], ast.BinOp) and stores[0].args[0].left.id == "stats_ptr"


def test_empty_route_guard_statistics_only_touch_the_counter():
    from freetoken.moe import empty_admission

    fn = next(n for n in ast.parse(inspect.getsource(empty_admission)).body
              if isinstance(n, ast.FunctionDef) and n.name == "_guard_lru")
    gated = [n for n in ast.walk(fn) if isinstance(n, ast.If)
             and isinstance(n.test, ast.Name) and n.test.id == "STATS"]
    assert len(gated) == 1
    calls = [n for n in ast.walk(gated[0]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr in ("store", "atomic_add")]
    assert [c.func.attr for c in calls] == ["atomic_add"]
    assert calls[0].args[0].left.id == "stats"


# ------------------------------------------------------------------ CPU differential run

LAYERS, EXPERTS, SLOTS, TOPK, HIDDEN, INTER = 3, 16, 20, 4, 8, 4


def _cache(collect: bool) -> OffloadMoeCache:
    cache = OffloadMoeCache(LAYERS, EXPERTS, SLOTS, torch.device("cpu"), quant_format="bf16",
                            decode_target="hybrid", hybrid_max_fetch=1)
    gen = torch.Generator().manual_seed(11)
    sources = {
        "gate_up": [torch.randn(EXPERTS, 2 * INTER, HIDDEN, generator=gen).bfloat16()
                    for _ in range(LAYERS)],
        "down": [torch.randn(EXPERTS, HIDDEN, INTER, generator=gen).bfloat16()
                 for _ in range(LAYERS)],
    }
    cache.set_bank_sources(sources)
    cache.collect_stats = collect

    def copy_missing(*, num_indices=None):  # CPU stand-in for the CUDA row copy
        layer = cache._pending_src_layer
        n = int((cache.num_indices if num_indices is None else num_indices).reshape(-1)[0])
        for i in range(n):
            slot, row = int(cache.evict_slots[i]), int(cache.src_indices[i])
            for per_layer, bank in cache.banks:
                bank[slot].copy_(per_layer[layer][row])

    cache.copy_missing = copy_missing
    return cache


class _Executor:
    return_route_outputs = False

    def __init__(self):
        self.calls = []

    def decode_submit(self, layer_id, hidden, weights, cpu_ids):
        self.calls.append((layer_id, cpu_ids.clone()))
        return (layer_id, cpu_ids.clone(), weights.clone(), hidden.clone())

    def decode_sync(self, pending):
        layer_id, ids, weights, hidden = pending
        return (hidden * (ids.clamp_min(0).float().sum(-1, keepdim=True) + 1)
                * weights.sum(-1, keepdim=True) * (layer_id + 1))


def _layer(layer_id: int, skip_empty: bool):
    def gemm(cache, hidden, weights, slots, *, views, **_):
        gate_up = views[0][slots.long()].float()  # [rows, k, 2I, H]: reads the resident rows
        return (gate_up.sum(dim=(-1, -2)) * weights).sum(-1, keepdim=True) * hidden
    return SimpleNamespace(layer_id=layer_id, skip_empty_cache=skip_empty, _expert_gemm=gemm)


STATE = ("slot_for_id", "id_of_slot", "usage", "step", "expert_recency", "num_indices",
         "num_missing_full", "active_mask")


@pytest.mark.parametrize("rows,skip_empty", [(1, False), (1, True), (2, False)])
def test_hybrid_decode_orchestration_is_identical_with_statistics_on_and_off(rows, skip_empty):
    caches = {flag: _cache(flag) for flag in (False, True)}
    executors = {flag: _Executor() for flag in (False, True)}
    for flag in caches:
        caches[flag].set_cpu_executor(executors[flag])
    rng = random.Random(2026_10_13 + rows + 7 * skip_empty)
    for step in range(60):
        for layer_id in range(LAYERS):
            ids = torch.tensor([[rng.randrange(EXPERTS) for _ in range(TOPK)] for _ in range(rows)],
                               dtype=torch.int32)
            weights = torch.tensor([[rng.choice((0.0, 0.25, 0.5, 1.0)) for _ in range(TOPK)]
                                    for _ in range(rows)])
            if step % 9 == 4:
                weights.zero_()  # an empty rank call
            hidden = torch.randn(rows, HIDDEN, generator=torch.Generator().manual_seed(step))
            outputs = {}
            for flag, cache in caches.items():
                outputs[flag] = OffloadMoELayer._decode_hybrid(
                    _layer(layer_id, skip_empty), cache, hidden.clone(), weights.clone(), ids.clone())
            assert torch.equal(outputs[False], outputs[True])
            for name in STATE:
                assert torch.equal(getattr(caches[False], name), getattr(caches[True], name)), name
            assert all(torch.equal(a, b) for (_, a), (_, b) in
                       zip(executors[False].calls, executors[True].calls))
    for name in ("stat_missing_layer", "stat_active_layer", "stat_fetched_layer", "stat_steps_layer"):
        assert not getattr(caches[False], name).any(), name  # off: counters never written
    assert int(caches[True].stat_steps_layer.sum()) == 60 * LAYERS
    assert int(caches[True].stat_active.item()) > 0


def test_reset_stats_leaves_admission_state_untouched():
    cache = _cache(True)
    cache.set_cpu_executor(_Executor())
    for step in range(5):
        ids = torch.tensor([[step, step + 1, step + 2, 3]], dtype=torch.int32)
        OffloadMoELayer._decode_hybrid(_layer(0, False), cache, torch.ones(1, HIDDEN),
                                       torch.ones(1, TOPK), ids)
    before = {name: getattr(cache, name).clone() for name in STATE}
    cache.reset_stats()
    assert all(torch.equal(before[name], getattr(cache, name)) for name in STATE)
    assert not cache.stat_steps_layer.any() and not cache.lru_stats.any()
