from types import SimpleNamespace

import pytest
import torch

import freetoken.distributed.info as distributed_info
from freetoken.distributed import DistributedInfo
from freetoken.models.config import DSV4AttentionGroupConfig, ModelConfig, RotaryConfig
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.execution import (
    _reset_execution_for_tests,
    configure_execution,
)
from freetoken.models.deepseek_v41.model import (
    DeepseekV41ExpertWorkerModel,
    DeepseekV41ForCausalLM,
    Transformer,
)
from freetoken.server.args import ServerArgs


def _tiny_args() -> DeepseekV41Args:
    return DeepseekV41Args(
        vocab_size=64,
        dim=32,
        moe_inter_dim=32,
        n_layers=2,
        n_heads=2,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=16,
        compress_ratios=(0, 1),
        kv_source_layers=(1,),
        index_source_layers=(1,),
        candidate_source_layer=-1,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        engram_layer_ids=(),
        engram_num_embeddings=(),
        n_routed_experts=8,
        n_activated_experts=2,
    )


def _build(
    rank: int, *, tp2_ep2: bool = False, attention_tp2_ep2: bool = False
):
    _reset_execution_for_tests()
    distributed_info._TP_INFO = DistributedInfo(rank, 2)
    plan = configure_execution(
        0, tp2_ep2=tp2_ep2, attention_tp2_ep2=attention_tp2_ep2
    )
    with plan.model_tp_context(), torch.device("meta"):
        model = DeepseekV41ForCausalLM(SimpleNamespace(dsv41_args=_tiny_args()))
    return plan, model


def _engine_model_config(args: DeepseekV41Args) -> ModelConfig:
    return ModelConfig(
        num_layers=args.n_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,
        head_dim=args.head_dim,
        hidden_size=args.dim,
        vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim,
        rms_norm_eps=args.norm_eps,
        rotary_config=RotaryConfig(
            head_dim=args.head_dim,
            rotary_dim=args.rope_head_dim,
            max_position=1024,
            base=args.rope_theta,
            scaling=None,
        ),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=args.norm_topk_prob,
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        moe_enabled=True,
        expert_quant="ds_fp4",
        dsv4_args=args,
        dsv41_args=args,
        single_stream_only=True,
        attention_groups=(
            DSV4AttentionGroupConfig(
                name="dsv41",
                layer_ids=tuple(range(args.n_layers)),
                num_kv_heads=1,
                head_dim=args.head_dim,
                sliding_window=args.window_size,
            ),
        ),
    )


def _engine_config(
    *,
    world_size: int = 2,
    backbone_rank=0,
    expert_shards: tuple[int, ...] | None = None,
    prefill_expert_shards: tuple[int, ...] | None = None,
    expert_storage_ranges: tuple[tuple[int, int], ...] | None = None,
    engram_ranks: tuple[int, ...] | None = None,
    tp2_ep2: bool = False,
    attention_tp2_ep2: bool = False,
) -> ServerArgs:
    args = _tiny_args()
    config = ServerArgs(
        model_path="unused",
        tp_info=DistributedInfo(0, world_size),
        dtype=torch.bfloat16,
        attention_backend="dsv4_sparse",
        moe_backend="offload",
        moe_cache_auto=True,
        dsv41_backbone_rank=backbone_rank,
        dsv41_expert_shards=expert_shards,
        dsv41_prefill_expert_shards=prefill_expert_shards,
        dsv41_expert_storage_ranges=expert_storage_ranges,
        dsv41_engram_ranks=engram_ranks,
        dsv41_tp2_ep2=tp2_ep2,
        dsv41_attention_tp2_ep2=attention_tp2_ep2,
        max_running_req=4,
        cuda_graph_bs=[1, 2, 4],
        cuda_graph_max_bs=4,
    )
    config.__dict__["model_config"] = _engine_model_config(args)
    return config


def test_authority_adapter_owns_dense_state_and_rank_local_expert_shells():
    plan, model = _build(0)
    try:
        assert isinstance(model._model, Transformer)
        assert model.state_dict()
        experts = list(model._iter_offload_moe_layers())
        assert len(experts) == 2
        assert all(layer.num_experts == 4 for layer in experts)
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_worker_adapter_has_no_dense_state_but_joins_every_expert_layer():
    plan, model = _build(1)
    try:
        assert plan.is_expert_worker
        assert isinstance(model._model, DeepseekV41ExpertWorkerModel)
        assert model.state_dict() == {}
        experts = list(model._iter_offload_moe_layers())
        assert len(experts) == 2
        assert all(layer.num_experts == 4 for layer in experts)
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_ep2_builds_sharded_dense_backbone_on_both_ranks(rank):
    plan, model = _build(rank, tp2_ep2=True)
    try:
        assert plan.tp2_ep2
        assert not plan.is_expert_worker
        assert isinstance(model._model, Transformer)
        layer = model._model.layers[0]
        assert layer.attn.n_heads == 1
        assert layer.attn.n_groups == 1
        assert layer.attn.wq_b.weight.shape == (32, 32)
        assert layer.attn.wo_a.shape == (16, 32)
        assert layer.attn.wo_b.weight.shape == (32, 16)
        assert layer.ffn.shared_experts.w1.weight.shape == (16, 32)
        assert layer.ffn.shared_experts.w2.weight.shape == (32, 16)
        assert layer.ffn.shared_experts.w3.weight.shape == (16, 32)
        assert layer.ffn.experts.num_experts == 4
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


@pytest.mark.parametrize("rank", [0, 1])
def test_attention_tp2_ep2_shards_attention_but_keeps_shared_expert_whole(rank):
    plan, model = _build(rank, attention_tp2_ep2=True)
    try:
        assert plan.attention_parallel
        assert not plan.shared_expert_parallel
        assert not plan.is_expert_worker
        assert isinstance(model._model, Transformer)
        layer = model._model.layers[0]
        assert layer.attn.n_heads == 1
        assert layer.attn.n_groups == 1
        assert layer.attn.wq_b.weight.shape == (32, 32)
        assert layer.attn.wo_a.shape == (16, 32)
        assert layer.attn.wo_b.weight.shape == (32, 16)
        if rank == 0:
            assert layer.ffn.gate is not None
            assert layer.ffn.shared_experts.w1.weight.shape == (32, 32)
            assert layer.ffn.shared_experts.w2.weight.shape == (32, 32)
            assert layer.ffn.shared_experts.w3.weight.shape == (32, 32)
        else:
            assert layer.ffn.gate is None
            assert layer.ffn.shared_experts is None
        assert layer.ffn.experts.num_experts == 4
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engine_config_resolves_ep2_single_stream_and_safe_eager_default():
    from freetoken.engine.engine import _adjust_config

    config = _engine_config()
    try:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = config.tp_info
        configure_execution(config.dsv41_backbone_rank)
        _adjust_config(config)
        assert config.model_tp_size == 1
        assert config.model_config.num_experts == 4
        assert config.max_running_req == 1
        assert config.cuda_graph_bs == []
        assert config.cuda_graph_max_bs == 0
        assert config.page_size == 128
        assert config.distributed_timeout == 1800.0
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engine_config_exposes_true_model_tp_width_for_tp2_ep2():
    from freetoken.engine.engine import _adjust_config

    config = _engine_config(tp2_ep2=True)
    try:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = config.tp_info
        configure_execution(
            config.dsv41_backbone_rank,
            tp2_ep2=config.dsv41_tp2_ep2,
        )
        _adjust_config(config)
        assert config.model_tp_size == 2
        assert config.model_config.num_experts == 4
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engine_config_exposes_true_model_tp_width_for_attention_tp2_ep2():
    from freetoken.engine.engine import _adjust_config

    config = _engine_config(attention_tp2_ep2=True)
    try:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = config.tp_info
        configure_execution(
            config.dsv41_backbone_rank,
            attention_tp2_ep2=config.dsv41_attention_tp2_ep2,
        )
        _adjust_config(config)
        assert config.model_tp_size == 2
        assert config.model_config.num_experts == 4
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engine_config_rejects_both_dense_parallel_modes():
    from freetoken.engine.engine import _adjust_config

    config = _engine_config(tp2_ep2=True, attention_tp2_ep2=True)
    try:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = config.tp_info
        with pytest.raises(ValueError, match="mutually exclusive"):
            _adjust_config(config)
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engine_config_can_opt_in_to_ep2_graph_decode(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    monkeypatch.setenv("FREETOKEN_DSV41_CUDA_GRAPH", "1")
    config = _engine_config()
    try:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = config.tp_info
        configure_execution(config.dsv41_backbone_rank)
        _adjust_config(config)
        assert config.cuda_graph_bs == [1]
        assert config.cuda_graph_max_bs == 1
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engine_config_requires_ep_and_backbone_rank():
    from freetoken.engine.engine import _adjust_config

    for config, message in (
        (_engine_config(backbone_rank=None), "requires row-sharded Engram/EP"),
        (_engine_config(world_size=1), "requires --tensor-parallel-size > 1"),
    ):
        try:
            _reset_execution_for_tests()
            distributed_info._TP_INFO = config.tp_info
            configure_execution(config.dsv41_backbone_rank)
            with pytest.raises(ValueError, match=message):
                _adjust_config(config)
        finally:
            _reset_execution_for_tests()
            distributed_info._TP_INFO = None


def test_engine_config_accepts_asymmetric_ep3():
    from freetoken.engine.engine import _adjust_config

    config = _engine_config(world_size=3, expert_shards=(3, 3, 2))
    try:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = config.tp_info
        plan = configure_execution(
            config.dsv41_backbone_rank,
            config.dsv41_expert_shards,
            engram_ranks=config.dsv41_engram_ranks,
        )
        _adjust_config(config)
        assert plan.partition(8).local_count == 3
        assert not plan.supports_packed_prefill
        assert config.model_tp_size == 1
        assert config.model_config.num_experts == 3
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engine_config_accepts_ep3_with_engram_on_two_ranks():
    from freetoken.engine.engine import _adjust_config

    config = _engine_config(
        world_size=3, expert_shards=(3, 3, 2), engram_ranks=(0, 1)
    )
    try:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = config.tp_info
        plan = configure_execution(
            config.dsv41_backbone_rank,
            config.dsv41_expert_shards,
            engram_ranks=config.dsv41_engram_ranks,
        )
        _adjust_config(config)
        assert plan.resolved_engram_ranks == (0, 1)
        assert plan.participates_in_engram
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engine_config_accepts_decode_only_auxiliary_prefill_split():
    from freetoken.engine.engine import _adjust_config

    config = _engine_config(
        world_size=3,
        expert_shards=(3, 3, 2),
        prefill_expert_shards=(4, 4, 0),
        expert_storage_ranges=((0, 4), (3, 5), (6, 2)),
        engram_ranks=(0, 1),
    )
    try:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = config.tp_info
        plan = configure_execution(
            backbone_rank=config.dsv41_backbone_rank,
            expert_shards=config.dsv41_expert_shards,
            prefill_expert_shards=config.dsv41_prefill_expert_shards,
            expert_storage_ranges=config.dsv41_expert_storage_ranges,
            engram_ranks=config.dsv41_engram_ranks,
        )
        _adjust_config(config)
        assert plan.phase_aware
        assert plan.prefill_active_ranks == (0, 1)
        assert config.model_config.num_experts == 4
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_decode_only_phase_rank_does_not_reserve_prefill_overlap_slots():
    from freetoken.engine.engine import _rank_moe_prefill_overlap_enabled

    try:
        for rank, expected in ((0, True), (1, True), (2, False)):
            _reset_execution_for_tests()
            distributed_info._TP_INFO = DistributedInfo(rank, 3)
            plan = configure_execution(
                backbone_rank=0,
                expert_shards=(3, 3, 2),
                prefill_expert_shards=(4, 4, 0),
                expert_storage_ranges=((0, 4), (3, 5), (6, 2)),
                engram_ranks=(0, 1),
            )
            assert _rank_moe_prefill_overlap_enabled(True, plan, 8) is expected
            assert not _rank_moe_prefill_overlap_enabled(False, plan, 8)
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engine_config_accepts_phase_aware_ep4_with_two_decode_only_ranks():
    from freetoken.engine.engine import _adjust_config

    try:
        for rank, stored_experts in enumerate((4, 6, 2, 2)):
            config = _engine_config(
                world_size=4,
                expert_shards=(2, 2, 2, 2),
                prefill_expert_shards=(4, 4, 0, 0),
                expert_storage_ranges=((0, 4), (2, 6), (4, 2), (6, 2)),
                engram_ranks=(0, 1),
            )
            object.__setattr__(config, "tp_info", DistributedInfo(rank, 4))
            _reset_execution_for_tests()
            distributed_info._TP_INFO = config.tp_info
            plan = configure_execution(
                backbone_rank=config.dsv41_backbone_rank,
                expert_shards=config.dsv41_expert_shards,
                prefill_expert_shards=config.dsv41_prefill_expert_shards,
                expert_storage_ranges=config.dsv41_expert_storage_ranges,
                engram_ranks=config.dsv41_engram_ranks,
            )
            _adjust_config(config)
            assert plan.phase_aware
            assert plan.prefill_active_ranks == (0, 1)
            assert plan.supports_packed_prefill
            assert config.model_config.num_experts == stored_experts
    finally:
        _reset_execution_for_tests()
        distributed_info._TP_INFO = None


def test_engram_history_comes_from_tokens_immediately_before_the_forward():
    class CapturingHasher:
        def row_ids(self, input_ids, positions, cu, history):
            self.args = (input_ids.clone(), positions.clone(), cu.clone(), history.clone())
            return torch.zeros(input_ids.numel(), 2, 24, dtype=torch.int64)

    model = object.__new__(DeepseekV41ForCausalLM)
    model._args = SimpleNamespace(
        engram_max_ngram_size=4,
        engram_pad_id=2,
        engram_layer_ids=(1, 14),
    )
    model._engram_hasher = CapturingHasher()
    req = SimpleNamespace(
        input_ids=torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.int32),
        cached_len=4,
    )
    batch = SimpleNamespace(
        padded_reqs=[req], positions=torch.tensor([4, 5], dtype=torch.int32)
    )
    rows = model._engram_rows(batch, torch.tensor([9, 10], dtype=torch.int64))
    _, _, cu, history = model._engram_hasher.args
    assert cu.tolist() == [0, 2]
    assert history.tolist() == [[6, 7, 8]]
    assert set(rows) == {1, 14}
    assert rows[1].shape == (2, 24)


def test_engram_history_uses_address_stable_graph_inputs_when_present():
    class CapturingHasher:
        def row_ids(self, input_ids, positions, cu, history):
            self.args = (input_ids.clone(), positions.clone(), cu.clone(), history.clone())
            return torch.zeros(input_ids.numel(), 2, 24, dtype=torch.int64)

    model = object.__new__(DeepseekV41ForCausalLM)
    model._args = SimpleNamespace(
        engram_max_ngram_size=4,
        engram_pad_id=2,
        engram_layer_ids=(1, 14),
    )
    model._engram_hasher = CapturingHasher()
    batch = SimpleNamespace(
        padded_reqs=[SimpleNamespace(input_ids=torch.tensor([99]), cached_len=1)],
        positions=torch.tensor([9], dtype=torch.int32),
        engram_history=torch.tensor([[6, 7, 8]], dtype=torch.int64),
        engram_cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
    )
    model._engram_rows(batch, torch.tensor([10], dtype=torch.int64))
    _, _, cu, history = model._engram_hasher.args
    assert cu.tolist() == [0, 1]
    assert history.tolist() == [[6, 7, 8]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_buffer_stages_fresh_host_engram_history_each_replay():
    from freetoken.engine.graph import GraphCaptureBuffer

    buffer = GraphCaptureBuffer.init(
        1,
        1,
        torch.device("cuda"),
        engram_history_width=3,
        engram_pad_id=2,
    )
    captured_batch = SimpleNamespace(padded_size=1)
    buffer.set_batch(captured_batch)
    result = torch.empty((), dtype=torch.int64, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result.copy_(
            captured_batch.engram_history.sum()
            + captured_batch.input_ids.long().sum()
            + captured_batch.positions.long().sum()
        )

    req = SimpleNamespace(
        input_ids=torch.tensor([3, 5, 7, 11], dtype=torch.int32), cached_len=3
    )
    replay_batch = SimpleNamespace(
        padded_size=1,
        padded_reqs=[req],
        input_ids=torch.tensor([11], dtype=torch.int32, device="cuda"),
        out_loc=torch.tensor([9], dtype=torch.int32, device="cuda"),
        positions=torch.tensor([3], dtype=torch.int32, device="cuda"),
        rope_positions=None,
        linear_table_idx=None,
    )
    buffer.copy_from(replay_batch)
    graph.replay()
    assert result.item() == 3 + 5 + 7 + 11 + 3

    req.input_ids = torch.tensor([13, 17, 19, 23, 29], dtype=torch.int32)
    req.cached_len = 4
    replay_batch.input_ids.fill_(29)
    replay_batch.positions.fill_(4)
    buffer.copy_from(replay_batch)
    graph.replay()
    assert result.item() == 17 + 19 + 23 + 29 + 4


def test_worker_enters_engram_collective_at_the_matching_layer_boundary():
    events = []

    class Layer:
        def __init__(self, layer_id):
            self.layer_id = layer_id

        def worker_forward(self, hidden_shape, device):
            events.append(("moe", self.layer_id, hidden_shape, device.type))

    class Coordinator:
        execution = SimpleNamespace(
            participates_in_engram=True,
            phase_aware=False,
        )

        def worker_lookup(self, layer_id, **kwargs):
            events.append(("engram", layer_id, kwargs["num_tokens"], kwargs["hashes_per_token"]))

    worker = object.__new__(DeepseekV41ExpertWorkerModel)
    worker.args = SimpleNamespace(
        n_layers=3,
        dim=32,
        engram_layer_ids=(1,),
        engram_hashes_per_token=24,
    )
    worker.layers = [Layer(i) for i in range(3)]
    worker.engram_coordinator = Coordinator()
    output = worker.forward(torch.tensor([3, 4], dtype=torch.int64))
    assert events == [
        ("moe", 0, (2, 32), "cpu"),
        ("engram", 1, 2, 24),
        ("moe", 1, (2, 32), "cpu"),
        ("moe", 2, (2, 32), "cpu"),
    ]
    assert output.shape == (1, 1)
