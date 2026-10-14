"""Live G2/G3 nodes for the real-weight authority rig.

Collection is safe offline.  Execution is allowed only by the successor site
adapter, which sets the explicit live receipt and isolates the authority GPU.
The live controller must hash this exact collected inventory and parse legacy
JUnit with zero skipped/xfail/deselected/error outcomes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from .authority_fusion_rig import (
    APPROVED_EXTENSION_HASHES,
    RealWeightAuthorityRig,
    assert_real_wire_layout,
    checkpoint_receipt,
    current_qualification_rig,
    process_identity_receipt,
    protocol_receipt,
    validate_live_candidate_runtime,
    validate_live_environment,
)


@pytest.fixture(scope="session")
def authority_rig():
    injected = current_qualification_rig()
    if injected is not None:
        yield injected
        return
    if os.getenv("FREETOKEN_AUTHORITY_RIG_REQUIRE_INJECTED") == "1":
        raise RuntimeError("qualify requires the already-loaded same-PID authority rig")
    validate_live_environment()
    candidate = Path(os.environ["FREETOKEN_AUTHORITY_RIG_CANDIDATE_ROOT"]).resolve()
    expected_commit = os.environ["FREETOKEN_AUTHORITY_RIG_EXPECTED_COMMIT"]
    expected_tree = os.environ["FREETOKEN_AUTHORITY_RIG_EXPECTED_TREE"]
    validate_live_candidate_runtime(
        candidate_root=candidate,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
        run_dir=Path(os.environ["FREETOKEN_AUTHORITY_RIG_RUN_DIR"]),
    )
    model_path = Path(os.environ["FREETOKEN_AUTHORITY_RIG_MODEL_PATH"])
    rig = RealWeightAuthorityRig.load(model_path=model_path, device=torch.device("cuda:0"))
    yield rig


def test_g2_process_import_extension_checkpoint_protocol_identity(authority_rig):
    candidate = Path(os.environ["FREETOKEN_AUTHORITY_RIG_CANDIDATE_ROOT"])
    identity = process_identity_receipt(candidate, APPROVED_EXTENSION_HASHES)
    checkpoint = checkpoint_receipt(Path(os.environ["FREETOKEN_AUTHORITY_RIG_MODEL_PATH"]))
    assert identity["pid"] == os.getpid()
    assert checkpoint["shards"]
    assert len(protocol_receipt()["sha256"]) == 64


def test_g2_real_m1_wire_offsets_alignment(authority_rig):
    hidden = torch.zeros(1, 5120, dtype=torch.bfloat16, device=authority_rig.pool.device)
    observed = assert_real_wire_layout(hidden)
    assert observed == {
        "hidden_offset": 0,
        "weights_offset": 10240,
        "ids_offset": 10264,
        "total_bytes": 10288,
        "ids_alignment": 8,
    }


def test_g2_real_m2_wire_offsets_alignment(authority_rig):
    hidden = torch.zeros(2, 5120, dtype=torch.bfloat16, device=authority_rig.pool.device)
    observed = assert_real_wire_layout(hidden)
    assert observed == {
        "hidden_offset": 0,
        "weights_offset": 20480,
        "ids_offset": 20528,
        "total_bytes": 20576,
        "ids_alignment": 16,
    }


def test_g3_eager_m1_256_generated_states_exact(authority_rig):
    result = authority_rig.compare_generated_states(1, states=256)
    assert result["states"] == 256 and len(result["raw_states"]) == 256


def test_g3_eager_m2_256_generated_states_exact(authority_rig):
    result = authority_rig.compare_generated_states(2, states=256)
    assert result["states"] == 256 and len(result["raw_states"]) == 256


def test_g3_captured_m1_256_generated_states_exact(authority_rig):
    result = authority_rig.compare_captured_control_fused(1, states=256)
    assert result["states"] == 256 and len(result["raw_states"]) == 256


def test_g3_captured_m2_256_generated_states_exact(authority_rig):
    result = authority_rig.compare_captured_control_fused(2, states=256)
    assert result["states"] == 256 and len(result["raw_states"]) == 256


def test_g3_shared_pool_bs2_bs1_bs2_changing_caps(authority_rig):
    result = authority_rig.compare_shared_pool_graph_schedule()
    assert result["order"] == (2, 1, 2) and result["shared_pool"] is True


def test_g3_m2_partner_payload_row_isolation(authority_rig):
    result = authority_rig.compare_partner_mutation()
    assert result["row_a_before_sha256"] == result["row_a_after_sha256"]
    assert result["row_b_before_sha256"] != result["row_b_after_sha256"]


def test_g3_c2_page_cross_cancel_recovery_exact(authority_rig):
    result = authority_rig.compare_page_crossing_sequence()
    assert result["page_crossing"] == ((255, 256), (383, 384))
    assert result["cancel_sequence"] == ("M2", "M2-cross", "M1", "M2-recovery")
