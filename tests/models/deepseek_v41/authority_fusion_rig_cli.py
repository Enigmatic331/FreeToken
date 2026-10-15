"""Fail-closed CLI for the real-weight Package-A authority qualification rig.

The module is inert unless an explicit live receipt supplies the exact candidate
commit/tree, authority GPU UUID, output directory and private cache roots.  The
``collect`` command is the only offline command; it performs collection only.
For a live G2-G4 window, ``qualify`` is the normative command because it keeps
G2, G3 and G4 in one PID/model residency.  The individual gate commands exist
only for bounded diagnosis and must not be used as successor-gate evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

# Refuse candidate-tree bytecode side effects even before live preflight.  The
# launcher must still provide PYTHONDONTWRITEBYTECODE=1 as a sealed receipt.
sys.dont_write_bytecode = True

import torch

from .authority_fusion_rig import (
    APPROVED_EXTENSION_HASHES,
    CUDA_NODE_COUNT,
    CUDA_NODE_INVENTORY_SHA256,
    RealWeightAuthorityRig,
    assert_real_wire_layout,
    checkpoint_receipt,
    installed_qualification_rig,
    process_identity_receipt,
    protocol_receipt,
    validate_junit_inventory,
    validate_live_candidate_runtime,
    validate_live_environment,
)


SCHEMA = "authority-fusion-real-weight-rig-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    path.write_text(payload)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _live_preflight(args) -> tuple[Path, dict[str, Any]]:
    environment = validate_live_environment()
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory must be new/empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = validate_live_candidate_runtime(
        candidate_root=args.candidate_root,
        expected_commit=args.expected_commit,
        expected_tree=args.expected_tree,
        run_dir=run_dir,
    )
    prefixes = ("FREETOKEN_", "TRITON_", "CUDA_", "TORCH_", "PYTORCH_", "NCCL_")
    environment_dump = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(prefixes)
    }
    receipt = {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "invocation_id": str(uuid.uuid4()),
        "argv": sys.argv,
        "authority_gpu_uuid": runtime["authority_gpu_uuid"],
        "candidate": runtime["candidate"],
        "checkpoint": checkpoint_receipt(args.model_path),
        "protocol": protocol_receipt(),
        "environment": environment,
        "environment_dump": environment_dump,
        "private_caches": runtime["private_caches"],
    }
    _write_json(run_dir / "preflight.json", receipt)
    return run_dir, receipt


def _process_identity(args, run_dir: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    identity = process_identity_receipt(args.candidate_root, APPROVED_EXTENSION_HASHES)
    value = {"schema": SCHEMA, "preflight": preflight, "process": identity}
    _write_json(run_dir / "identity" / "process.json", value)
    return value


def _load_rig(args) -> RealWeightAuthorityRig:
    torch.cuda.set_device(0)
    return RealWeightAuthorityRig.load(
        model_path=args.model_path,
        device=torch.device("cuda:0"),
    )


def _run_g2(rig: RealWeightAuthorityRig, run_dir: Path) -> dict[str, Any]:
    rows = {}
    for count in (1, 2):
        hidden = torch.zeros(count, 5120, dtype=torch.bfloat16, device=rig.pool.device)
        rows[str(count)] = assert_real_wire_layout(hidden)
    value = {
        "schema": SCHEMA,
        "gate": "G2",
        "wire": rows,
        "raw_receipts_file": "exactness.jsonl",
        "verdict": "PASS",
    }
    _append_jsonl(run_dir / "exactness.jsonl", value)
    return value


def _run_g3(rig: RealWeightAuthorityRig, run_dir: Path) -> dict[str, Any]:
    receipts = []
    for rows in (1, 2):
        value = rig.compare_generated_states(rows, states=256)
        receipt = {
            "schema": SCHEMA,
            "gate": "G3",
            "mode": "eager",
            **value,
        }
        _append_jsonl(run_dir / "exactness.jsonl", receipt)
        receipts.append(receipt)
        captured = rig.compare_captured_control_fused(rows, states=256)
        capture_receipt = {
            "schema": SCHEMA,
            "gate": "G3",
            "mode": "captured_graph",
            **captured,
        }
        _append_jsonl(run_dir / "exactness.jsonl", capture_receipt)
        receipts.append(capture_receipt)
    mutation = rig.compare_partner_mutation()
    crossing = rig.compare_page_crossing_sequence()
    shared_pool = rig.compare_shared_pool_graph_schedule()
    _append_jsonl(
        run_dir / "exactness.jsonl",
        {"schema": SCHEMA, "gate": "G3", "oracle": "M2-partner-mutation", **mutation},
    )
    _append_jsonl(
        run_dir / "exactness.jsonl",
        {"schema": SCHEMA, "gate": "G3", "oracle": "C2-cross-cancel-recovery", **crossing},
    )
    _append_jsonl(
        run_dir / "exactness.jsonl",
        {"schema": SCHEMA, "gate": "G3", "oracle": "shared-pool-bs2-bs1-bs2", **shared_pool},
    )
    return {
        "schema": SCHEMA,
        "gate": "G3",
        "verdict": "PASS",
        "receipts": len(receipts),
        "partner_mutation": mutation,
        "crossing": crossing,
        "shared_pool": shared_pool,
        "raw_receipts_file": "exactness.jsonl",
    }


def _marker_manifest(rows: int, marker_events: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    boundaries = [
        {
            "pre_layer": layer,
            "start": (
                {"kind": "token_start", "layer": 0}
                if layer == 0
                else {"kind": "all_reduce_end", "layer": layer - 1}
            ),
            "end": {"kind": "broadcast_start", "layer": layer},
        }
        for layer in range(40)
    ]
    engram_observed = sum(
        event.get("communicator_scope") == "EngramCoordinator"
        for event in marker_events
    )
    return {
        "schema": SCHEMA,
        "m": rows,
        "communicator_scope": "MoEState._comm",
        "engram_collectives_observed": engram_observed,
        "marker_event_count": len(marker_events),
        "boundaries": boundaries,
        "copy_scope": "ffn_norm.output_external_cuda_event -> moe.broadcast_start(layer)",
    }


def _write_graph(path: Path, rows: int, structure: tuple[tuple[str, int], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(structure) != 82 or structure[0] != ("token_start", 0) or structure[-1] != (
        "token_end",
        39,
    ):
        raise RuntimeError("captured PRE structure is not the exact 82-node chain")
    lines = [f'digraph "authority-M{rows}" {{']
    names = []
    for index, (kind, layer) in enumerate(structure):
        name = f"n{index}"
        names.append(name)
        lines.append(f'  {name} [label="{kind}:{layer}"];')
        if index:
            edge = f' [label="PRE({layer})"]' if kind == "broadcast_start" else ""
            lines.append(f"  {names[index - 1]} -> {name}{edge};")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def _run_g4(rig: RealWeightAuthorityRig, run_dir: Path) -> dict[str, Any]:
    summaries = []
    graph_files: dict[str, str] = {}
    graph_marker_files: dict[str, str] = {}
    pre_marker_files: dict[str, str] = {}
    for rows in (1, 2):
        label = f"M{rows}"
        matrix = rig.run_g4_matrix(rows, graph_dir=run_dir / "graphs" / label)
        marker = _marker_manifest(rows, matrix["control_marker_events"])
        marker["captured_structure"] = matrix["structure"]
        marker["arm_graphs"] = matrix["arm_graphs"]
        pre_path = run_dir / "markers" / label / "pre-boundaries.json"
        _write_json(pre_path, marker)
        pre_marker_files[str(rows)] = str(pre_path.relative_to(run_dir))
        _write_graph(
            run_dir / "graphs" / label / "pre-chain.dot",
            rows,
            matrix["structure"],
        )
        for arm in rig.protocol.m2_arm_order:
            key = f"{rows}/{arm}"
            graph_files[key] = f"graphs/{label}/{arm}.dot"
            graph_marker_files[key] = f"graphs/{label}/{arm}.markers.json"
        for item in matrix["aa_null"]:
            _append_jsonl(run_dir / "aa.jsonl", {"schema": SCHEMA, "m": rows, **item})
        for item in matrix["measurements"]:
            for replay, raw in enumerate(item["replays"]):
                _append_jsonl(
                    run_dir / "timing.jsonl",
                    {
                        "schema": SCHEMA,
                        "m": rows,
                        "round": item["round"],
                        "slot": item["position"],
                        "arm": item["arm"],
                        "replay": replay,
                        "pre_ms": [raw["pre_ms"][layer] for layer in range(40)],
                        "measurement_start_monotonic_ns": raw[
                            "measurement_start_monotonic_ns"
                        ],
                        "measurement_end_monotonic_ns": raw[
                            "measurement_end_monotonic_ns"
                        ],
                        "pid": raw["pid"],
                        "invocation_id": raw["invocation_id"],
                        "model_load_id": raw["model_load_id"],
                    },
                )
                _append_jsonl(
                    run_dir / "exactness.jsonl",
                    {
                        "schema": SCHEMA,
                        "gate": "G4",
                        "m": rows,
                        "round": item["round"],
                        "arm": item["arm"],
                        "replay": replay,
                        "logits_sha256": raw["logits_sha256"],
                        "touched_state_sha256": raw["touched_state_sha256"],
                        "pre_reset_sha256": raw["pre_reset_sha256"],
                    },
                )
        summaries.append(
            {
                "m": rows,
                "arm_graphs": matrix["arm_graphs"],
                "measurement_groups": len(matrix["measurements"]),
            }
        )
    return {
        "schema": SCHEMA,
        "gate": "G4",
        "raw_complete": True,
        "verdict": "RAW_UNEVALUATED",
        "timing_file": "timing.jsonl",
        "exactness_file": "exactness.jsonl",
        "aa_file": "aa.jsonl",
        "graph_files": graph_files,
        "graph_marker_files": graph_marker_files,
        "pre_marker_files": pre_marker_files,
        "summaries": summaries,
    }


def _collect(args) -> int:
    root = args.candidate_root.resolve()
    target = "tests/models/deepseek_v41/test_authority_fusion_rig_cuda.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "python")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            str(args.python),
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            target,
        ],
        cwd=root,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    nodes = [line.strip() for line in result.stdout.splitlines() if "::" in line]
    if len(nodes) != CUDA_NODE_COUNT or len(nodes) != len(set(nodes)):
        raise RuntimeError(
            f"CUDA node collection must contain exactly {CUDA_NODE_COUNT} unique nodes"
        )
    payload = "\n".join(nodes) + "\n"
    if hashlib.sha256(payload.encode()).hexdigest() != CUDA_NODE_INVENTORY_SHA256:
        raise RuntimeError("CUDA node collection does not match the source-pinned inventory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(json.dumps({"nodes": len(nodes), "sha256": _sha256(args.output)}, sort_keys=True))
    return 0


class _CollectedInventory:
    """Persist the exact nodes pytest actually collected in this process."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination

    def pytest_collection_finish(self, session) -> None:
        nodes = [item.nodeid for item in session.items]
        self.destination.write_text("\n".join(nodes) + "\n")


def _run_pinned_cuda_inventory(
    rig: RealWeightAuthorityRig, args, run_dir: Path
) -> dict[str, Any]:
    """Execute the pinned CUDA nodes in the current PID/model residency."""

    import pytest

    if args.expected_inventory is None or args.expected_inventory_sha256 is None:
        raise RuntimeError("qualify requires a pinned CUDA inventory and SHA256")
    if args.expected_inventory_sha256 != CUDA_NODE_INVENTORY_SHA256:
        raise RuntimeError("qualify inventory SHA256 is not the source-pinned constant")
    inventory_dir = run_dir / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    expected = inventory_dir / "expected.txt"
    expected.write_bytes(args.expected_inventory.read_bytes())
    observed_inventory_sha256 = _sha256(expected)
    if observed_inventory_sha256 != args.expected_inventory_sha256:
        raise RuntimeError(
            "pinned CUDA inventory hash drift: "
            f"{observed_inventory_sha256} != {args.expected_inventory_sha256}"
        )
    nodes = [line.strip() for line in expected.read_text().splitlines() if "::" in line]
    if len(nodes) != CUDA_NODE_COUNT or len(nodes) != len(set(nodes)):
        raise RuntimeError(
            f"pinned CUDA inventory must contain exactly {CUDA_NODE_COUNT} unique nodes"
        )
    collected = inventory_dir / "collected.txt"
    junit = inventory_dir / "junit.xml"
    plugin = _CollectedInventory(collected)
    pinned_environment = {
        "FREETOKEN_AUTHORITY_RIG_CANDIDATE_ROOT": str(args.candidate_root.resolve()),
        "FREETOKEN_AUTHORITY_RIG_MODEL_PATH": str(args.model_path.resolve()),
        "FREETOKEN_AUTHORITY_RIG_EXPECTED_COMMIT": args.expected_commit,
        "FREETOKEN_AUTHORITY_RIG_EXPECTED_TREE": args.expected_tree,
        "FREETOKEN_AUTHORITY_RIG_RUN_DIR": str(run_dir.resolve()),
        "FREETOKEN_AUTHORITY_RIG_REQUIRE_INJECTED": "1",
    }
    old_environment = {name: os.environ.get(name) for name in pinned_environment}
    os.environ.update(pinned_environment)
    try:
        with installed_qualification_rig(rig) as injection:
            status = pytest.main(
                [
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "-o",
                    "junit_family=legacy",
                    f"--junitxml={junit}",
                    *nodes,
                ],
                plugins=[plugin],
            )
        if not injection["consumed"] or injection["rig_id"] != id(rig):
            raise RuntimeError("pinned CUDA pytest did not consume the loaded qualification rig")
    finally:
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if int(status) != int(pytest.ExitCode.OK):
        raise RuntimeError(f"pinned CUDA pytest failed with exit code {int(status)}")
    validation = validate_junit_inventory(
        expected=expected,
        collected=collected,
        junit=junit,
        expected_sha256=args.expected_inventory_sha256,
    )
    receipt = {
        **validation,
        "same_pid_model_residency": True,
        "pid": os.getpid(),
        "rig_id": id(rig),
        "expected": str(expected.relative_to(run_dir)),
        "collected": str(collected.relative_to(run_dir)),
        "junit": str(junit.relative_to(run_dir)),
    }
    _write_json(inventory_dir / "validation.json", receipt)
    return receipt


def _seal_run(run_dir: Path, result: dict[str, Any]) -> None:
    _write_json(run_dir / "result.json", result)
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{_sha256(path)}  {path.relative_to(run_dir)}")
    (run_dir / "SHA256SUMS").write_text("\n".join(entries) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--candidate-root", type=Path, required=True)
    collect.add_argument("--python", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    for name in ("identity", "g2", "g3", "g4", "qualify"):
        gate = sub.add_parser(name)
        gate.add_argument("--candidate-root", type=Path, required=True)
        gate.add_argument("--expected-commit", required=True)
        gate.add_argument("--expected-tree", required=True)
        gate.add_argument("--model-path", type=Path, required=True)
        gate.add_argument("--run-dir", type=Path, required=True)
        gate.add_argument("--expected-inventory", type=Path)
        gate.add_argument("--expected-inventory-sha256")
    args = parser.parse_args(argv)
    if args.command == "collect":
        return _collect(args)
    run_dir, preflight = _live_preflight(args)
    try:
        if args.command == "identity":
            identity = _process_identity(args, run_dir, preflight)
            result = {"schema": SCHEMA, "command": "identity", "identity": identity}
        else:
            if args.command in ("g4", "qualify"):
                from tests.models.deepseek_v41.authority_fusion_rig import (
                    validate_cudagraph_debug_dump_capability,
                )

                preflight["cudagraph_debug_dump"] = validate_cudagraph_debug_dump_capability(
                    run_dir / "capability" / "cudagraph-debug.dot"
                )
                _write_json(run_dir / "preflight.json", preflight)
            rig = _load_rig(args)
            rig.invocation_id = preflight["invocation_id"]
            identity = _process_identity(args, run_dir, preflight)
            stages: dict[str, Any] = {}
            # Run the exact pytest inventory in this loaded rig/PID before the
            # G4 timing matrix. Retained debug graphs are scoped to G4 only.
            inventory = (
                _run_pinned_cuda_inventory(rig, args, run_dir)
                if args.command == "qualify"
                else None
            )
            if args.command in ("g2", "qualify"):
                stages["g2"] = _run_g2(rig, run_dir)
            if args.command in ("g3", "qualify"):
                stages["g3"] = _run_g3(rig, run_dir)
            if args.command in ("g4", "qualify"):
                stages["g4"] = _run_g4(rig, run_dir)
            result = {
                "schema": SCHEMA,
                "command": args.command,
                "same_pid_model_residency": args.command == "qualify",
                "pid": os.getpid(),
                "invocation_id": rig.invocation_id,
                "model_load_id": rig.model_load_id,
                "stages": stages,
                "inventory": inventory,
            }
    except BaseException as exc:
        _seal_run(
            run_dir,
            {
                "schema": SCHEMA,
                "command": args.command,
                "verdict": "FAIL_CLOSED",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        )
        raise
    else:
        _seal_run(run_dir, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
