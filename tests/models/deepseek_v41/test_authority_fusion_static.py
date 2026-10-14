"""CPU/static integration and planted-defect controls for Package A."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.deepseek_v41 import authority_fusion as fusion
from freetoken.models.deepseek_v41 import moe


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "python/freetoken/models/deepseek_v41/model.py"
MOE = ROOT / "python/freetoken/models/deepseek_v41/moe.py"
AUTHORITY = ROOT / "python/freetoken/models/deepseek_v41/authority_fusion.py"
KERNEL = ROOT / "python/freetoken/kernel/triton/dsv41/authority_fused.py"
OVERLAY = "1be8860bfd52547e47b9ecec6f81a3bb911c0df1"


def _class_method(source: str, class_name: str, method_name: str):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _current(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _overlay(path: Path) -> str:
    relative = path.relative_to(ROOT)
    return subprocess.check_output(
        ["git", "show", f"{OVERLAY}:{relative}"], cwd=ROOT, text=True
    )


def _module_function(source: str, name: str):
    return next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _dump(node) -> str:
    return ast.dump(node, include_attributes=False)


@pytest.mark.parametrize(
    "class_name, method_name",
    [
        ("Block", "prefill_single"),
        ("Block", "verify_single"),
        ("Transformer", "prefill_single"),
        ("Transformer", "verify"),
        ("Transformer", "prefill_tail_pair"),
    ],
)
def test_prefill_and_verify_ast_unchanged(class_name, method_name):
    current = _class_method(_current(MODEL), class_name, method_name)
    baseline = _class_method(_overlay(MODEL), class_name, method_name)
    assert ast.dump(current, include_attributes=False) == ast.dump(
        baseline, include_attributes=False
    )


def test_worker_forward_ast_unchanged():
    current = _class_method(_current(MOE), "MoE", "worker_forward")
    baseline = _class_method(_overlay(MOE), "MoE", "worker_forward")
    assert ast.dump(current, include_attributes=False) == ast.dump(
        baseline, include_attributes=False
    )


@pytest.mark.parametrize(
    "path,class_name,method_name",
    [
        (MODEL, "Attention", "decode_step"),
        (MODEL, "Attention", "prefill_single"),
        (MODEL, "Block", "_mixes"),
        (MODEL, "Block", "decode_step"),
        (MODEL, "MoEState", "forward"),
        (MODEL, "Transformer", "decode"),
        (MOE, "MoE", "forward"),
        (MOE, "MoE", "worker_forward"),
    ],
)
def test_literal_old_path_method_bodies_match_overlay(path, class_name, method_name):
    assert _dump(_class_method(_current(path), class_name, method_name)) == _dump(
        _class_method(_overlay(path), class_name, method_name)
    )


@pytest.mark.parametrize(
    "name",
    ["identity_pre_mix", "hc_mixes", "hc_pre", "hc_post", "warmup_direct_pre"],
)
def test_legacy_hc_functions_match_overlay(name):
    current = ROOT / "python/freetoken/models/deepseek_v41/hc.py"
    assert _dump(_module_function(_current(current), name)) == _dump(
        _module_function(_overlay(current), name)
    )


def test_legacy_broadcast_function_matches_overlay():
    assert _dump(_module_function(_current(MOE), "_broadcast_decode_dispatch")) == _dump(
        _module_function(_overlay(MOE), "_broadcast_decode_dispatch")
    )


def test_selector_contains_literal_old_decode_else_path():
    method = _class_method(_current(MODEL), "DeepseekV41ForCausalLM", "forward")
    selectors = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "authority_fusion_eligible"
    ]
    assert len(selectors) == 1
    old_calls = [
        call
        for node in selectors[0].orelse
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "decode"
    ]
    assert len(old_calls) == 1
    baseline = _class_method(
        _overlay(MODEL), "DeepseekV41ForCausalLM", "forward"
    )
    baseline_assign = next(
        node for node in ast.walk(baseline)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "decode"
    )
    assert _dump(selectors[0].orelse[0]) == _dump(baseline_assign)


def test_forward_outside_selector_matches_overlay_exactly():
    current = _class_method(
        _current(MODEL), "DeepseekV41ForCausalLM", "forward"
    )

    class RemoveSelector(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "authority_fusion":
                return None
            return node

        def visit_If(self, node):
            if (
                isinstance(node.test, ast.Call)
                and isinstance(node.test.func, ast.Name)
                and node.test.func.id == "authority_fusion_eligible"
            ):
                return [self.visit(item) for item in node.orelse]
            return self.generic_visit(node)

    normalized = RemoveSelector().visit(current)
    baseline = _class_method(
        _overlay(MODEL), "DeepseekV41ForCausalLM", "forward"
    )
    assert _dump(normalized) == _dump(baseline)


def test_attach_runtime_diff_is_only_authority_warmup_hook():
    current = _class_method(_current(MODEL), "Transformer", "attach_runtime")

    class RemoveWarmup(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "authority_fusion":
                return None
            return node

        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "warmup_authority_fusion"
            ):
                return None
            return self.generic_visit(node)

    normalized = RemoveWarmup().visit(current)
    baseline = _class_method(_overlay(MODEL), "Transformer", "attach_runtime")
    assert _dump(normalized) == _dump(baseline)


def test_authority_control_flow_is_free_functions_not_model_methods():
    tree = ast.parse(_current(AUTHORITY))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"decode_authority_fused", "authority_block_decode", "authority_attention_decode", "authority_moe_forward"} <= functions
    model_tree = ast.parse(_current(MODEL))
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "authority" in node.name
        for node in ast.walk(model_tree)
    )


def test_kernel_surface_and_separate_buffers_are_static():
    tree = ast.parse(_current(KERNEL))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "hc_prepare_mix",
        "hc_post_mixprep",
        "hc_sinkhorn_rmsprep",
        "router_scores_exact",
        "router_wire_tail",
    } <= names
    source = _current(KERNEL)
    assert "flat = torch.empty" in source and "square = torch.empty_like(flat)" in source
    assert ".square_(" not in source
    assert "enable_fp_fusion=False" in source
    assert "enable_reflect_ftz=False" in source
    assert "tl.div_rn" in source and "tl.sqrt_rn" in source


def test_pinned_triton_exposes_non_ftz_codegen_option():
    from triton.backends.nvidia.compiler import CUDAOptions

    assert "enable_reflect_ftz" in CUDAOptions.__dataclass_fields__
    assert "enable_fp_fusion" in CUDAOptions.__dataclass_fields__


def test_route_passthrough_preserves_three_dimensions(monkeypatch):
    class FakeCommunicator:
        def all_reduce(self, value):
            return value + 1

    monkeypatch.setattr(moe, "DistributedCommunicator", FakeCommunicator)
    experts = moe.RoutedExperts.__new__(moe.RoutedExperts)
    experts._authority_route_passthrough = True
    routes = torch.zeros(2, 6, 8, dtype=torch.bfloat16)
    result = experts._maybe_all_reduce(routes)
    assert result.shape == routes.shape and torch.equal(result, routes + 1)


def test_route_passthrough_default_uses_legacy_sum(monkeypatch):
    from freetoken.layers import ExpertParallelOffloadMoELayer

    monkeypatch.setattr(
        ExpertParallelOffloadMoELayer,
        "_maybe_all_reduce",
        lambda self, value: value.sum(1),
    )
    experts = moe.RoutedExperts.__new__(moe.RoutedExperts)
    experts._authority_route_passthrough = False
    routes = torch.ones(2, 6, 8, dtype=torch.bfloat16)
    assert experts._maybe_all_reduce(routes).shape == (2, 8)


def test_route_passthrough_scope_restores_after_exception():
    experts = SimpleNamespace(_authority_route_passthrough=False)
    with pytest.raises(RuntimeError, match="planted"):
        with fusion._authority_route_passthrough(experts):
            assert experts._authority_route_passthrough is True
            raise RuntimeError("planted defect")
    assert experts._authority_route_passthrough is False


def test_route_passthrough_reentry_is_planted_defect():
    experts = SimpleNamespace(_authority_route_passthrough=True)
    with pytest.raises(RuntimeError, match="re-entered"):
        with fusion._authority_route_passthrough(experts):
            pass


def test_prefilled_wire_root_only_and_single_collective():
    calls = []

    class FakeCommunicator:
        def broadcast(self, value, root):
            calls.append((value, root))
            return value

    execution = SimpleNamespace(backbone_rank=0, rank=0)
    hidden_shape = (1, 8)
    wire = torch.arange(4 + 12, dtype=torch.int32)
    hidden, weights, ids = moe._broadcast_prefilled_decode_dispatch(
        FakeCommunicator(), execution, wire, hidden_shape, 6
    )
    assert hidden.shape == hidden_shape
    assert weights.shape == ids.shape == (1, 6)
    assert calls == [(wire, 0)]
    with pytest.raises(RuntimeError, match="root only"):
        moe._broadcast_prefilled_decode_dispatch(
            FakeCommunicator(), SimpleNamespace(backbone_rank=0, rank=1), wire, hidden_shape, 6
        )


def test_authority_wire_bytes_decode_identically_on_worker_cpu():
    hidden = torch.arange(8, dtype=torch.bfloat16).view(1, 8)
    wire, wire_hidden, weights, ids = fusion._dispatch_wire(hidden, 6)
    wire_hidden.copy_(hidden)
    weights.copy_(torch.arange(6, dtype=torch.float32).view(1, 6))
    ids.copy_(torch.arange(6, dtype=torch.int32).view(1, 6))

    class WorkerComm:
        def broadcast(self, target, root):
            target.copy_(wire)
            return target

    got = moe._broadcast_decode_dispatch(
        WorkerComm(), SimpleNamespace(backbone_rank=0, rank=1), torch.empty_like(hidden), 6
    )
    assert torch.equal(got[0], hidden)
    assert torch.equal(got[1], weights)
    assert torch.equal(got[2], ids)


def test_memo_and_side_stream_have_finally_guards():
    tree = ast.parse(_current(AUTHORITY))
    decode = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_authority_fused")
    attention = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "authority_attention_decode")
    decode_finally = [node for node in ast.walk(decode) if isinstance(node, ast.Try) and node.finalbody]
    attention_finally = [node for node in ast.walk(attention) if isinstance(node, ast.Try) and node.finalbody]
    assert any(
        any("memo.clear()" in ast.unparse(stmt) for stmt in node.finalbody)
        for node in decode_finally
    )
    assert any(
        any("current.wait_stream(_KV_STREAM)" in ast.unparse(stmt) for stmt in node.finalbody)
        for node in attention_finally
    )
    # Planted defects: deleting either finally action must make the oracle false.
    defective = _current(AUTHORITY).replace("memo.clear()", "pass", 1).replace(
        "current.wait_stream(_KV_STREAM)", "pass", 1
    )
    tree = ast.parse(defective)
    broken_decode = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_authority_fused")
    broken_attention = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "authority_attention_decode")
    assert not any(
        "memo.clear()" in ast.unparse(stmt)
        for node in ast.walk(broken_decode) if isinstance(node, ast.Try)
        for stmt in node.finalbody
    )
    assert not any(
        "current.wait_stream(_KV_STREAM)" in ast.unparse(stmt)
        for node in ast.walk(broken_attention) if isinstance(node, ast.Try)
        for stmt in node.finalbody
    )


def test_attention_side_window_contains_no_collective():
    method = next(
        node
        for node in ast.parse(_current(AUTHORITY)).body
        if isinstance(node, ast.FunctionDef) and node.name == "authority_attention_decode"
    )
    source = ast.unparse(method)
    assert "all_reduce" not in source and ".broadcast(" not in source


def test_hc_prepared_cpu_matches_reference_and_alias_defect():
    from freetoken.models.deepseek_v41.hc import hc_mixes, hc_mixes_from_prepared

    torch.manual_seed(13)
    hidden = torch.randn(2, 1, 4, 8, dtype=torch.bfloat16)
    weight = torch.randn(24, 32, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(24, dtype=torch.float32)
    expected = hc_mixes(
        hidden.clone(), weight, scale, base,
        hc_mult=4, sinkhorn_iters=3, eps=1e-6, norm_eps=1e-20,
    )
    flat = hidden.flatten(-2).float()
    square = flat.square()
    actual = hc_mixes_from_prepared(
        flat, square, weight, scale, base,
        hc_mult=4, sinkhorn_iters=3, eps=1e-6, norm_eps=1e-20,
    )
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    with pytest.raises(ValueError, match="must not alias"):
        hc_mixes_from_prepared(
            flat, flat, weight, scale, base,
            hc_mult=4, sinkhorn_iters=3, eps=1e-6, norm_eps=1e-20,
        )
