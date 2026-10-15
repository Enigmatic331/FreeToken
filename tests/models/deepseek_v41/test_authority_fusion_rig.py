"""Offline contract and mutation tests for the Package-A authority rig."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from .authority_fusion_rig import (
    AuthorityRigInfrastructureError,
    AUTHORITY_UUID,
    BoundaryRecorder,
    CapturedAuthorityDecode,
    CUDA_NODE_COUNT,
    CUDA_NODE_INVENTORY_SHA256,
    ExpertComputeStub,
    FROZEN_PROTOCOL,
    LAYER_CLASSES,
    LAYER_WEIGHTS,
    LayerStateSnapshot,
    MarkerCommunicator,
    REPRESENTATIVE_LAYERS,
    TensorSnapshot,
    cudagraph_marker_sidecar,
    current_qualification_rig,
    disjoint_full_loc_map,
    expected_wire_layout,
    installed_qualification_rig,
    protocol_receipt,
    validate_live_candidate_runtime,
    validate_live_environment,
    validate_junit_inventory,
)
from .authority_fusion_rig_cli import failure_class


class _Event:
    def __init__(self, serial):
        self.serial = serial
        self.recorded = False

    def record(self):
        self.recorded = True


def test_failure_class_is_type_based_and_fail_closed():
    assert failure_class(AuthorityRigInfrastructureError("graph dump failed")) == (
        "harness_infrastructure"
    )
    assert failure_class(AssertionError("output mismatch")) == "correctness"
    assert failure_class(RuntimeError("CUDA graph debug dump absent/empty")) == "unknown"


class _Events:
    def __init__(self):
        self.serial = 0

    def __call__(self):
        self.serial += 1
        return _Event(self.serial)


def _complete_token(recorder: BoundaryRecorder):
    recorder.begin_token()
    for layer in range(40):
        recorder.copy_scope_begin(layer)
        recorder.broadcast_start(layer)
        recorder.all_reduce_end(layer)
    recorder.end_token()


def test_frozen_protocol_and_layer_classes_are_exact():
    FROZEN_PROTOCOL.validate()
    assert REPRESENTATIVE_LAYERS == (0, 1, 2, 5, 13, 14, 20, 24, 39)
    assert tuple(LAYER_CLASSES) == REPRESENTATIVE_LAYERS
    assert sum(LAYER_WEIGHTS.values()) == 40
    assert FROZEN_PROTOCOL.aa_null_pairs == 12
    assert (FROZEN_PROTOCOL.full_pages, FROZEN_PROTOCOL.window_pages) == (12, 12)
    receipt = protocol_receipt()
    assert receipt["sha256"] == hashlib.sha256(
        __import__("json").dumps(
            receipt["protocol"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    "rows,weights_offset,ids_offset,total,alignment",
    [(1, 10240, 10264, 10288, 8), (2, 20480, 20528, 20576, 16)],
)
def test_real_m1_m2_wire_offsets_are_frozen(rows, weights_offset, ids_offset, total, alignment):
    layout = expected_wire_layout(rows)
    assert (layout.weights_offset, layout.ids_offset, layout.total_bytes) == (
        weights_offset,
        ids_offset,
        total,
    )
    assert layout.ids_alignment == alignment


def test_wire_mutations_are_not_accepted():
    one = expected_wire_layout(1)
    assert one.ids_offset != 10240  # hidden-only is not a valid route-id offset
    with pytest.raises(ValueError):
        expected_wire_layout(3)


def test_pre_boundary_definition_is_mechanical():
    recorder = BoundaryRecorder(event_factory=_Events())
    _complete_token(recorder)
    pairs = recorder.pre_event_pairs()
    assert tuple(pairs) == tuple(range(40))
    # PRE0 = token-start -> BC0.
    assert (pairs[0][0].serial, pairs[0][1].serial) == (1, 3)
    # PRE(l) = AR(l-1)-end -> BC(l)-start.
    assert (pairs[1][0].serial, pairs[1][1].serial) == (4, 6)
    assert (pairs[39][0].serial, pairs[39][1].serial) == (118, 120)
    assert len(recorder.marker_receipt()) == 122
    assert all(event.recorded for pair in pairs.values() for event in pair)


def test_wire_destination_receipt_is_exact_and_ordered():
    recorder = BoundaryRecorder(event_factory=_Events())
    wire = torch.empty(5120 + 24, dtype=torch.int32)
    wire_hidden = wire[:5120].view(torch.bfloat16).view(2, 5120)
    with recorder.require_wire_destinations():
        recorder.begin_token()
        for layer in range(40):
            recorder.copy_scope_begin(layer)
            recorder.wire_destination(layer, wire_hidden)
            recorder.broadcast_start(layer)
            recorder.all_reduce_end(layer)
        recorder.end_token()
    receipt = recorder.wire_destination_receipt()
    assert [item["layer"] for item in receipt] == list(range(40))
    assert {item["base_ptr"] for item in receipt} == {wire_hidden.data_ptr()}
    assert {item["nbytes"] for item in receipt} == {20480}
    assert {item["tail_base_ptr"] for item in receipt} == {wire_hidden.data_ptr() + 20480}
    assert {item["tail_nbytes"] for item in receipt} == {96}
    assert {item["wire_nbytes"] for item in receipt} == {20576}


def test_cudagraph_sidecar_pins_d2d_destinations_and_dot_grammar(tmp_path):
    dot = tmp_path / "full.dot"
    dot.write_text('digraph "cuda" { a -> b; }\n')
    destinations = tuple(
        {"layer": layer, "base_ptr": 0x100000 + layer * 0x10000, "nbytes": 10240,
         "wire_base_ptr": 0x100000 + layer * 0x10000, "wire_nbytes": 10288,
         "tail_base_ptr": 0x100000 + layer * 0x10000 + 10240, "tail_nbytes": 48}
        for layer in range(40)
    )
    captured = SimpleNamespace(marker_events=({"kind": "token_start", "layer": 0},), wire_destinations=destinations)
    value = cudagraph_marker_sidecar(
        captured=captured, rows=1, arm="full", dot_path=dot
    )
    assert value["schema"] == "authority-cudagraph-marker-events-v2"
    assert value["copy_scope"]["required_memcpy_kind"] == "Device-to-Device"
    assert value["copy_destinations"]["wire_hidden"] == tuple(
        {"layer": row["layer"], "base_ptr": row["base_ptr"], "nbytes": row["nbytes"]}
        for row in destinations
    )
    assert value["copy_destinations"]["wire_tail"] == tuple(
        {"layer": row["layer"], "base_ptr": row["tail_base_ptr"], "nbytes": row["tail_nbytes"]}
        for row in destinations
    )
    assert value["dot_receipt"]["producer"] == "torch.cuda.CUDAGraph.debug_dump"
    assert value["validation_status"] == "RAW_UNEVALUATED"
    assert len(value["dot_sha256"]) == 64


def test_cuda_inventory_pin_is_exact():
    assert CUDA_NODE_COUNT == 10
    assert CUDA_NODE_INVENTORY_SHA256 == (
        "601bb852124f7ad39e59f1e5ce236f813ae86bf72315a6b7066148cd274c3c50"
    )


def test_same_process_qualification_injection_has_consumption_receipt():
    rig = object()
    with installed_qualification_rig(rig) as receipt:
        assert current_qualification_rig() is rig
        assert receipt["consumed"] is True and receipt["rig_id"] == id(rig)
    assert current_qualification_rig() is None


def test_default_cuda_event_is_external_for_graph_capture(monkeypatch):
    calls = []
    events = _Events()

    def event(**kwargs):
        calls.append(kwargs)
        return events()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", event)
    recorder = BoundaryRecorder()
    recorder.begin_token()
    assert calls == [{"enable_timing": True, "external": True}]
    assert recorder.records[0].event.recorded is True


def test_live_environment_is_shared_and_fail_closed(monkeypatch):
    values = {
        "FREETOKEN_AUTHORITY_RIG_LIVE": "1",
        "FREETOKEN_DSV41_AUTHORITY_FUSION": "1",
        "FREETOKEN_DSV41_FUSED_DECODE_DISPATCH": "1",
        "CUDA_VISIBLE_DEVICES": AUTHORITY_UUID,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TRITON_DEFAULT_FP_FUSION", raising=False)
    assert validate_live_environment()["FREETOKEN_DSV41_FUSED_DECODE_DISPATCH"] == "1"
    monkeypatch.setenv("FREETOKEN_DSV41_FUSED_DECODE_DISPATCH", "0")
    with pytest.raises(RuntimeError, match="FUSED_DECODE_DISPATCH"):
        validate_live_environment()


def test_g4_retains_and_instantiates_debug_graphs_outside_timing():
    capture = inspect.getsource(__import__(
        "tests.models.deepseek_v41.authority_fusion_rig",
        fromlist=["RealWeightAuthorityRig"],
    ).RealWeightAuthorityRig.capture_complete_token)
    matrix = inspect.getsource(__import__(
        "tests.models.deepseek_v41.authority_fusion_rig",
        fromlist=["RealWeightAuthorityRig"],
    ).RealWeightAuthorityRig.run_g4_matrix)
    assert "keep_graph: bool = False" in capture
    assert "CUDAGraph(keep_graph=keep_graph)" in capture
    assert "if keep_graph:" in capture and "graph.instantiate()" in capture
    assert "keep_graph=graph_dir is not None" in matrix
    assert "enable_debug_mode" not in matrix


def test_live_candidate_runtime_checks_clean_tree_cache_and_uuid(monkeypatch, tmp_path):
    commit, tree = "a" * 40, "b" * 40
    run_dir = tmp_path / "run"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(run_dir / "triton"))
    monkeypatch.setenv("FREETOKEN_KERNEL_CACHE_DIR", str(run_dir / "freetoken"))

    def clean_run(command, **_kwargs):
        if command[0] == "git":
            if command[-2:] == ["rev-parse", "HEAD"]:
                return SimpleNamespace(stdout=commit + "\n")
            if command[-2:] == ["rev-parse", "HEAD^{tree}"]:
                return SimpleNamespace(stdout=tree + "\n")
            if command[-3:] == ["status", "--porcelain=v1", "--untracked-files=all"]:
                return SimpleNamespace(stdout="")
        raise AssertionError(command)

    monkeypatch.setattr("subprocess.run", clean_run)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(uuid=AUTHORITY_UUID.removeprefix("GPU-")),
    )
    receipt = validate_live_candidate_runtime(
        candidate_root=tmp_path,
        expected_commit=commit,
        expected_tree=tree,
        run_dir=run_dir,
    )
    assert receipt["candidate"]["clean"] is True
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "shared"))
    with pytest.raises(RuntimeError, match="TRITON_CACHE_DIR"):
        validate_live_candidate_runtime(
            candidate_root=tmp_path,
            expected_commit=commit,
            expected_tree=tree,
            run_dir=run_dir,
        )


def test_live_candidate_runtime_rejects_more_than_one_process_visible_gpu(
    monkeypatch, tmp_path
):
    commit, tree = "a" * 40, "b" * 40
    run_dir = tmp_path / "run"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(run_dir / "triton"))
    monkeypatch.setenv("FREETOKEN_KERNEL_CACHE_DIR", str(run_dir / "freetoken"))

    def clean_run(command, **_kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=commit + "\n")
        if command[-2:] == ["rev-parse", "HEAD^{tree}"]:
            return SimpleNamespace(stdout=tree + "\n")
        if command[-3:] == ["status", "--porcelain=v1", "--untracked-files=all"]:
            return SimpleNamespace(stdout="")
        raise AssertionError(command)

    monkeypatch.setattr("subprocess.run", clean_run)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="torch sees 2 devices"):
        validate_live_candidate_runtime(
            candidate_root=tmp_path,
            expected_commit=commit,
            expected_tree=tree,
            run_dir=run_dir,
        )


def test_captured_replay_resets_before_pre_reset_digest():
    source = inspect.getsource(CapturedAuthorityDecode.replay)
    assert source.index("self.rig.reset()") < source.index("pre_reset_sha256 =")


def test_pre_chain_dot_is_derived_from_exact_captured_structure(tmp_path):
    from .authority_fusion_rig_cli import _marker_manifest, _write_graph

    structure = [("token_start", 0)]
    for layer in range(40):
        structure.extend((("broadcast_start", layer), ("all_reduce_end", layer)))
    structure.append(("token_end", 39))
    path = tmp_path / "pre-chain.dot"
    _write_graph(path, 2, tuple(structure))
    text = path.read_text()
    assert text.count("[label=") == 122  # 82 nodes + 40 PRE edges.
    assert 'label="PRE(0)"' in text and 'label="PRE(39)"' in text
    with pytest.raises(RuntimeError, match="82-node"):
        _write_graph(path, 2, tuple(structure[:-1]))
    marker_events = (
        {"kind": "token_start", "communicator_scope": "authority-boundary"},
        {"kind": "broadcast_start", "communicator_scope": "MoEState._comm"},
    )
    marker = _marker_manifest(2, marker_events)
    assert marker["communicator_scope"] == "MoEState._comm"
    assert marker["engram_collectives_observed"] == 0
    assert marker["marker_event_count"] == len(marker_events)
    assert len(marker["boundaries"]) == 40
    assert marker["boundaries"][0]["start"] == {"kind": "token_start", "layer": 0}
    assert marker["boundaries"][39]["start"] == {
        "kind": "all_reduce_end",
        "layer": 38,
    }


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda r: r.broadcast_start(1), "broadcast order"),
        (lambda r: r.all_reduce_end(0), "all-reduce order"),
    ],
)
def test_pre_boundary_order_mutations_fail(mutator, match):
    recorder = BoundaryRecorder(event_factory=_Events())
    recorder.begin_token()
    with pytest.raises(RuntimeError, match=match):
        mutator(recorder)


def test_incomplete_token_and_open_token_refuse_segmentation():
    recorder = BoundaryRecorder(event_factory=_Events())
    recorder.begin_token()
    recorder.copy_scope_begin(0)
    recorder.broadcast_start(0)
    recorder.all_reduce_end(0)
    with pytest.raises(RuntimeError, match="open token"):
        recorder.pre_event_pairs()
    with pytest.raises(RuntimeError, match="incomplete 40-layer"):
        recorder.end_token()


def test_marker_transport_labels_only_its_frozen_layer():
    recorder = BoundaryRecorder(event_factory=_Events())
    recorder.begin_token()
    recorder.copy_scope_begin(0)
    comm = MarkerCommunicator(recorder, 0)
    tensor = torch.zeros(1)
    assert comm.broadcast(tensor, 0) is tensor
    assert comm.all_reduce(tensor) is tensor
    with pytest.raises(RuntimeError, match="broadcast root drift"):
        comm.broadcast(tensor, 1)


def test_expert_stub_calls_real_maybe_all_reduce(monkeypatch):
    calls = []

    def real(self, routes):
        calls.append((self.layer_id, tuple(routes.shape), self._authority_route_passthrough))
        return routes

    from freetoken.models.deepseek_v41 import moe

    monkeypatch.setattr(moe.RoutedExperts, "_maybe_all_reduce", real)
    recorder = BoundaryRecorder(event_factory=_Events())
    comm = MarkerCommunicator(recorder, 7)
    stub = ExpertComputeStub(layer=7, top_k=6, communicator=comm)
    stub._authority_route_passthrough = True
    hidden = torch.arange(16, dtype=torch.float32).view(2, 8)
    weights = torch.ones(2, 6)
    out = stub.routed_forward(hidden, weights, torch.zeros(2, 6, dtype=torch.int32))
    assert out.shape == (2, 6, 8)
    assert calls == [(7, (2, 6, 8), True)]


def test_expert_stub_literal_arm_enters_production_base_reduction(monkeypatch):
    calls = []

    def base(self, routes):
        calls.append((self.layer_id, tuple(routes.shape)))
        return routes.sum(dim=1)

    from freetoken.layers.moe import ExpertParallelOffloadMoELayer

    monkeypatch.setattr(ExpertParallelOffloadMoELayer, "_maybe_all_reduce", base)
    recorder = BoundaryRecorder(event_factory=_Events())
    comm = MarkerCommunicator(recorder, 7)
    stub = ExpertComputeStub(layer=7, top_k=6, communicator=comm)
    hidden = torch.arange(16, dtype=torch.float32).view(2, 8)
    weights = torch.ones(2, 6)
    out = stub.routed_forward(hidden, weights, torch.zeros(2, 6, dtype=torch.int32))
    assert out.shape == (2, 8)
    assert calls == [(7, (2, 6, 8))]


def test_disjoint_full_location_rows_cover_frozen_geometry():
    mapping, bindings = disjoint_full_loc_map(FROZEN_PROTOCOL, device=torch.device("cpu"))
    assert mapping.shape == (2, 768)
    assert mapping[0, 767].item() == 767
    assert mapping[1, 0].item() == 768
    assert mapping[1, 767].item() == 1535
    assert len(bindings) == 12 and len(set(bindings)) == 12
    assert not torch.isin(mapping[0], mapping[1]).any()


def test_batch_inputs_use_real_engram_row_end_and_disjoint_snapshots():
    from .authority_fusion_rig import RealWeightAuthorityRig

    mapping, _ = disjoint_full_loc_map(FROZEN_PROTOCOL, device=torch.device("cpu"))
    table = SimpleNamespace(plan=SimpleNamespace(row_start=100, row_end=200))
    layer = SimpleNamespace(
        engram=SimpleNamespace(
            table=SimpleNamespace(coordinator=SimpleNamespace(tables={1: table}))
        )
    )
    rig = RealWeightAuthorityRig.__new__(RealWeightAuthorityRig)
    rig.protocol = FROZEN_PROTOCOL
    rig.pool = SimpleNamespace(device=torch.device("cpu"), full_loc_map=mapping)
    rig.transformer = SimpleNamespace(
        args=SimpleNamespace(
            vocab_size=320,
            engram_layer_ids=(1,),
            engram_hashes_per_token=4,
        ),
        layers=(SimpleNamespace(), layer),
    )
    ids, positions, cap, engram, batch = rig.batch_inputs(2, seed=17)
    assert ids.shape == (2, 1)
    assert tuple(positions.tolist()) == (255, 383) and cap == 383
    assert engram[1].shape == (2, 4)
    assert int(engram[1].min()) >= 100 and int(engram[1].max()) < 200
    assert not torch.equal(batch.attn_metadata.full_snap[0], batch.attn_metadata.full_snap[1])


def test_tensor_snapshot_restores_and_detects_inventory_mutation():
    state = {"window": torch.arange(8), "ring": torch.arange(4, dtype=torch.float32)}
    snapshot = TensorSnapshot.capture(state)
    digest = snapshot.digest()
    state["window"].fill_(99)
    snapshot.restore(state)
    assert torch.equal(state["window"], torch.arange(8))
    assert snapshot.digest() == digest
    with pytest.raises(RuntimeError, match="inventory drift"):
        snapshot.restore({"window": state["window"]})


def test_layer_snapshot_exactness_and_mutations_are_fail_closed():
    pool = type("Pool", (), {})()
    pool.full_to_window = torch.arange(4)
    pool.full_loc_map = torch.arange(8).view(2, 4)
    pool.window_pool = [torch.arange(4, dtype=torch.float32)]
    pool.cmp_pool = [None]
    pool.idx_pool = [None]
    pool.state_ring = [None]
    pool.indexer_state_ring = [None]
    hidden = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    incoming = torch.arange(4, dtype=torch.float32).view(1, 1, 4)
    first = LayerStateSnapshot.capture(hidden, incoming, pool)
    second = LayerStateSnapshot.capture(hidden, incoming, pool)
    first.assert_equal(second, layer=5)
    second.hidden.view(-1)[0] += 1
    with pytest.raises(AssertionError, match="hidden bytes mismatch"):
        first.assert_equal(second, layer=5)
    second = LayerStateSnapshot.capture(hidden, incoming, pool)
    second.state.tensors["window_pool.0"][0] += 1
    with pytest.raises(AssertionError, match="window_pool.0"):
        first.assert_equal(second, layer=5)


def test_junit_semantics_reject_skip_xfail_error_deselection_and_hash_drift(tmp_path):
    expected = tmp_path / "expected.txt"
    collected = tmp_path / "collected.txt"
    junit = tmp_path / "junit.xml"
    nodes = ["tests/a.py::test_one", "tests/a.py::test_two"]
    expected.write_text("\n".join(nodes) + "\n")
    collected.write_text(expected.read_text())
    digest = hashlib.sha256(expected.read_bytes()).hexdigest()
    clean = (
        '<testsuite><testcase file="tests/a.py" name="test_one"/>'
        '<testcase file="tests/a.py" name="test_two"/></testsuite>'
    )
    junit.write_text(clean)
    assert validate_junit_inventory(
        expected=expected, collected=collected, junit=junit, expected_sha256=digest
    )["node_count"] == 2
    for tag in ("failure", "error", "skipped"):
        junit.write_text(
            '<testsuite><testcase file="tests/a.py" name="test_one"/>'
            f'<testcase file="tests/a.py" name="test_two"><{tag}/></testcase></testsuite>'
        )
        with pytest.raises(RuntimeError, match="failure/error/skip/xfail"):
            validate_junit_inventory(
                expected=expected, collected=collected, junit=junit, expected_sha256=digest
            )
    junit.write_text(clean)
    collected.write_text(nodes[0] + "\n")
    with pytest.raises(RuntimeError, match="deselection"):
        validate_junit_inventory(
            expected=expected, collected=collected, junit=junit, expected_sha256=digest
        )
    collected.write_text(expected.read_text())
    with pytest.raises(RuntimeError, match="hash drift"):
        validate_junit_inventory(
            expected=expected, collected=collected, junit=junit, expected_sha256="0" * 64
        )


def test_real_rig_loader_is_cuda_refusing_offline():
    from .authority_fusion_rig import RealWeightAuthorityRig

    with pytest.raises(RuntimeError, match="authorized CUDA"):
        RealWeightAuthorityRig.load(
            model_path=Path("/definitely/not/used"), device=torch.device("cpu")
        )
