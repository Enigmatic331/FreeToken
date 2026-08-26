from __future__ import annotations

import json
import re
from types import SimpleNamespace

import safetensors.torch
import torch

from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.moe.partition import ExpertPartition


def test_glm_partition_balances_ep3(monkeypatch):
    from freetoken.models.glm_moe_dsa import config as glm_config

    monkeypatch.setattr(
        "freetoken.distributed.try_get_tp_info",
        lambda: SimpleNamespace(rank=2, size=3),
    )

    partition = glm_config.ep_partition(256)

    assert (partition.global_offset, partition.local_count) == (171, 85)


def test_glm_sparse_block_keeps_global_router_and_local_expert_bank(monkeypatch):
    from freetoken.models.glm_moe_dsa import config as glm_config
    from freetoken.models.glm_moe_dsa.moe import GlmMoeDsaSparseBlock
    import freetoken.layers.moe as moe_layers

    partition = ExpertPartition(8, world_size=3, rank=1)  # owns [3, 6)
    monkeypatch.setattr(glm_config, "ep_partition", lambda total: partition)
    monkeypatch.setattr(
        moe_layers, "get_tp_info", lambda: SimpleNamespace(rank=1, size=3)
    )
    cfg = SimpleNamespace(
        glm_dsa_args=SimpleNamespace(num_experts=8),
        num_experts=3,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        hidden_size=16,
        moe_intermediate_size=16,
        first_k_dense_replace=0,
        n_shared_experts=1,
        dense_quant="none",
        moe_backend="offload",
    )

    with torch.device("meta"):
        block = GlmMoeDsaSparseBlock(cfg, layer_id=0)

    assert block.num_experts == 8
    assert block.gate.weight.shape == (8, 16)
    assert block.experts.num_experts == 3
    assert block.partition == partition


def test_nvfp4_loader_filters_and_localizes_partitioned_experts(tmp_path):
    """A rank owning global experts 2/3 allocates two rows and maps them to 0/1."""
    H = I = 16
    tensors: dict[str, torch.Tensor] = {}
    for expert in range(4):
        for proj, out_dim, in_dim in (
            ("gate_proj", I, H),
            ("up_proj", I, H),
            ("down_proj", H, I),
        ):
            prefix = f"model.layers.0.mlp.experts.{expert}.{proj}"
            tensors[f"{prefix}.weight"] = torch.full(
                (out_dim, in_dim // 2), expert, dtype=torch.uint8
            )
            tensors[f"{prefix}.weight_scale"] = torch.full(
                (out_dim, in_dim // 16), float(expert + 1), dtype=torch.float8_e4m3fn
            )
            tensors[f"{prefix}.weight_scale_2"] = torch.tensor(
                float(10 + expert), dtype=torch.float16
            )

    shard = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )
    spec = Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(
            r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
            r"(?P<proj>gate_proj|up_proj|down_proj)\."
            r"(?P<kind>weight|weight_scale|weight_scale_2)$"
        ),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,
        desc="test experts",
    )
    cfg = SimpleNamespace(
        num_experts=2,
        hidden_size=H,
        moe_intermediate_size=I,
        num_layers=1,
        first_k_dense_replace=0,
    )
    completed: list[int] = []

    banks = load_nvfp4_expert_source_banks(
        str(tmp_path),
        cfg,
        spec,
        drop_page_cache=lambda path: None,
        primary=False,
        layer_sink=lambda layer, layer_banks: completed.append(layer),
        partition=ExpertPartition(4, world_size=2, rank=1),
    )

    assert completed == [0]
    assert banks["gate_up_packed"][0][:, 0, 0].tolist() == [2, 3]
    assert banks["down_packed"][0][:, 0, 0].tolist() == [2, 3]
    assert banks["gate_up_global"][0][:, 0].tolist() == [12.0, 13.0]
