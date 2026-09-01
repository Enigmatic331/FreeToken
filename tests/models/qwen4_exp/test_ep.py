import freetoken.distributed.info as distributed_info
import torch
from freetoken.distributed import DistributedInfo, get_tp_info
from freetoken.engine.config import EngineConfig
from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.models.qwen4_exp.execution import Qwen4ExpExecutionPlan


def test_execution_roles_and_model_tp_context(monkeypatch):
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(1, 2))
    plan = Qwen4ExpExecutionPlan(rank=1, world_size=2, backbone_rank=0)
    assert plan.enabled
    assert plan.is_expert_worker
    assert not plan.is_backbone

    with plan.model_tp_context():
        assert get_tp_info() == DistributedInfo(0, 1)
    assert get_tp_info() == DistributedInfo(1, 2)


def test_execution_backbone_role():
    plan = Qwen4ExpExecutionPlan(rank=0, world_size=2, backbone_rank=0)
    assert plan.is_backbone
    assert not plan.is_expert_worker
    partition = plan.partition(512)
    assert partition.local_count == 256


def test_ep_model_state_uses_tp1_geometry(monkeypatch):
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(1, 2))
    plan = Qwen4ExpExecutionPlan(rank=1, world_size=2, backbone_rank=0)
    config = EngineConfig(
        model_path="unused",
        tp_info=DistributedInfo(1, 2),
        dtype=torch.bfloat16,
        qwen4_exp_backbone_rank=0,
    )
    assert config.model_tp_size == 1
    with plan.model_tp_context():
        pool = MHAKVCache(
            num_kv_heads=2,
            num_layers=1,
            head_dim=256,
            num_pages=2,
            page_size=1,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
    assert pool.k_cache(0).shape == (2, 1, 2, 256)
