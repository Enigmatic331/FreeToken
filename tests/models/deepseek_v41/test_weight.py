from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import freetoken.distributed.info as distributed_info
from freetoken.distributed import DistributedInfo
from freetoken.models.deepseek_v41.execution import (
    _reset_execution_for_tests,
    configure_execution,
)
from freetoken.models.deepseek_v41.weight import (
    _tp2_ep2_resident_slice,
    expected_resident_specs,
    inspect_checkpoint,
    iter_weights,
    load_dsfp4_expert_sources,
    plan_expert_shard,
    validate_resident_checkpoint,
)


MODEL_PATH = "/home/enigmatic331/models/DeepSeek-V4.1-Flash"


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_ep2_resident_slices_match_projection_partition(rank):
    _reset_execution_for_tests()
    distributed_info._TP_INFO = DistributedInfo(rank, 2)
    configure_execution(0, tp2_ep2=True)
    try:
        matrix = torch.arange(32, dtype=torch.float32).view(4, 8)
        column = _tp2_ep2_resident_slice(
            "layers.0.attn.wq_b.weight", matrix
        )
        row = _tp2_ep2_resident_slice(
            "layers.0.attn.wo_b.weight", matrix
        )
        groups = _tp2_ep2_resident_slice("layers.0.attn.wo_a", matrix)
        assert column.shape == groups.shape == (2, 8)
        assert row.shape == (4, 4)
        torch.testing.assert_close(column, matrix.chunk(2, 0)[rank])
        torch.testing.assert_close(row, matrix.chunk(2, 1)[rank])
        torch.testing.assert_close(groups, matrix.chunk(2, 0)[rank])
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


@pytest.mark.parametrize("rank", [0, 1])
def test_attention_tp2_ep2_slices_attention_and_replicates_shared_expert(rank):
    _reset_execution_for_tests()
    distributed_info._TP_INFO = DistributedInfo(rank, 2)
    configure_execution(0, attention_tp2_ep2=True)
    try:
        matrix = torch.arange(32, dtype=torch.float32).view(4, 8)
        attention = _tp2_ep2_resident_slice(
            "layers.0.attn.wq_b.weight", matrix
        )
        shared = _tp2_ep2_resident_slice(
            "layers.0.ffn.shared_experts.w1.weight", matrix
        )
        torch.testing.assert_close(attention, matrix.chunk(2, 0)[rank])
        torch.testing.assert_close(shared, matrix)
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_ftw_conversion_rejects_missing_engram_representation():
    from freetoken.checkpoint.convert import _validate_ftw_conversion_supported

    with pytest.raises(SystemExit, match="rank-sharded Engram-table entries"):
        _validate_ftw_conversion_supported(SimpleNamespace(dsv41_args=object()))

    _validate_ftw_conversion_supported(SimpleNamespace(dsv41_args=None))


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="official checkpoint absent")
def test_official_checkpoint_payload_plan_is_exact():
    plan = inspect_checkpoint(MODEL_PATH)
    assert plan.bytes("text_resident") == 9_846_748_608
    assert plan.bytes("routed_experts") == 288_777_830_400
    assert plan.bytes("engram_table") == 202_758_032_400
    assert plan.bytes("mtp") == 7_932_874_632
    assert plan.bytes("vision") == 970_536_960
    assert sum(tensor.nbytes for tensor in plan.tensors) == 510_286_023_000
    assert plan.baseline_resident_bytes < plan.bytes("text_resident")

    args = __import__("freetoken.models.deepseek_v41.args", fromlist=["load_args"]).load_args(
        MODEL_PATH
    )
    specs = expected_resident_specs(args)
    assert len(specs) == 1_214
    validate_resident_checkpoint(plan, args)
    assert plan.baseline_resident_bytes == 9_846_687_168
    assert plan.baseline_device_bytes == 12_542_838_208
    rank0 = plan_expert_shard(plan, args, rank=0, world_size=2)
    rank1 = plan_expert_shard(plan, args, rank=1, world_size=2)
    assert (rank0.global_offset, rank0.local_count) == (0, 192)
    assert (rank1.global_offset, rank1.local_count) == (192, 192)
    assert rank0.tensor_count == rank1.tensor_count == 40 * 192 * 6
    assert rank0.source_bytes == rank1.source_bytes == 144_388_915_200


def test_text_stream_never_materializes_engram_experts_mtp_or_vision(tmp_path):
    tensors = {
        "head.weight": torch.arange(8, dtype=torch.bfloat16).view(2, 4),
        "layers.0.attn_norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "layers.0.ffn.gate.bias_vl": torch.zeros(2),
        "layers.0.ffn.experts.0.w1.weight": torch.zeros(2, 2, dtype=torch.uint8),
        "layers.1.engram.embed.weight": torch.zeros(3, 4, dtype=torch.float8_e4m3fn),
        "mtp.0.attn_norm.weight": torch.zeros(4, dtype=torch.bfloat16),
        "vision.pos_embed": torch.zeros(4, dtype=torch.bfloat16),
    }
    shard = "model.safetensors"
    save_file(tensors, tmp_path / shard)
    index = {"weight_map": {name: shard for name in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    loaded = dict(
        iter_weights(
            str(tmp_path),
            torch.device("cpu"),
            include_moe_experts=False,
        )
    )
    assert set(loaded) == {"head", "layers.0.attn_norm.weight"}
    torch.testing.assert_close(loaded["head"], tensors["head.weight"])


def test_attention_tp2_peer_stream_skips_root_owned_router_and_shared(tmp_path):
    tensors = {
        "head.weight": torch.arange(8, dtype=torch.bfloat16).view(2, 4),
        "layers.0.attn.wq_b.weight": torch.arange(
            16, dtype=torch.bfloat16
        ).view(4, 4),
        "layers.0.ffn.gate.weight": torch.ones(4, 4, dtype=torch.bfloat16),
        "layers.0.ffn.shared_experts.w1.weight": torch.ones(
            4, 4, dtype=torch.bfloat16
        ),
    }
    shard = "model.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )

    _reset_execution_for_tests()
    distributed_info._TP_INFO = DistributedInfo(1, 2)
    configure_execution(0, attention_tp2_ep2=True)
    try:
        loaded = dict(
            iter_weights(
                str(tmp_path),
                torch.device("cpu"),
                include_moe_experts=False,
            )
        )
        assert set(loaded) == {"head", "layers.0.attn.wq_b.weight"}
        assert loaded["layers.0.attn.wq_b.weight"].shape == (2, 4)
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_expert_loader_places_only_owned_global_rows(tmp_path):
    args = SimpleNamespace(n_layers=2, n_routed_experts=4, dim=64, moe_inter_dim=32)
    tensors = {}
    for layer in range(args.n_layers):
        for expert in range(args.n_routed_experts):
            value = 10 * layer + expert
            base = f"layers.{layer}.ffn.experts.{expert}"
            for projection in ("w1", "w3"):
                tensors[f"{base}.{projection}.weight"] = torch.full(
                    (32, 32), value, dtype=torch.uint8
                )
                tensors[f"{base}.{projection}.scale"] = torch.full(
                    (32, 2), value, dtype=torch.uint8
                ).view(torch.float8_e8m0fnu)
            tensors[f"{base}.w2.weight"] = torch.full((64, 16), value, dtype=torch.uint8)
            tensors[f"{base}.w2.scale"] = torch.full(
                (64, 1), value, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
    shard = "experts.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )

    completed = []
    banks = load_dsfp4_expert_sources(
        str(tmp_path),
        args,
        rank=1,
        world_size=2,
        layer_sink=lambda layer, _banks: completed.append(layer),
    )
    assert sorted(completed) == [0, 1]
    assert banks["gate_up_packed"][0].shape == (2, 64, 32)
    assert banks["down_packed"][1].shape == (2, 64, 16)
    assert banks["gate_up_packed"][0][0, 0, 0].item() == 2
    assert banks["gate_up_packed"][1][1, -1, -1].item() == 13


def test_dummy_dsfp4_expert_payloads_are_deterministic_zero(monkeypatch):
    import freetoken.moe.host_banks as host_banks
    from freetoken.models.deepseek_v4.weight import dummy_dsfp4_expert_sources

    monkeypatch.setattr(host_banks, "pin_banks", lambda _banks: None)
    args = SimpleNamespace(n_layers=1, n_routed_experts=2, dim=64, moe_inter_dim=32)

    banks = dummy_dsfp4_expert_sources(args)

    assert not torch.count_nonzero(banks["gate_up_packed"][0])
    assert not torch.count_nonzero(banks["down_packed"][0])
