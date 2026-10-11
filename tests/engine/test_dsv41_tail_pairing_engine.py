"""Engine side of DSV4.1 tail pairing: the paired forward_batch branch and startup checks (CPU)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import freetoken.engine.engine as engine_mod
import freetoken.models.deepseek_v41.moe as dsv41_moe
from freetoken.core import Batch, Context, Req, SamplingParams
from freetoken.distributed import DistributedInfo
from freetoken.engine.engine import Engine, _validate_tail_pairing
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.moe import prefill_trace
from freetoken.scheduler import SchedulerConfig
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.tail_pairing import TAIL_PAIRING_ENV, TailPairPlan
from freetoken.server.args import ServerArgs, parse_args

UID = 7


class _Event:
    def record(self, stream):
        self.stream = stream


@pytest.fixture
def cpu_cuda(monkeypatch):
    stream = object()
    monkeypatch.setattr(engine_mod.torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(engine_mod.torch.cuda, "Event", _Event)
    return stream


def _pair():
    plan = TailPairPlan(UID, 0, 8192, 8192, 1106)
    ids = torch.arange(plan.prompt_end, dtype=torch.int32) % 90 + 1
    head_req = ChunkedReq(input_ids=ids[: plan.tail_start], table_idx=0, cached_len=8192,
                          output_len=4, uid=UID, sampling_params=SamplingParams(),
                          cache_handle=None)
    tail_req = Req(input_ids=ids, table_idx=0, cached_len=plan.tail_start, output_len=4, uid=UID,
                   sampling_params=SamplingParams(), cache_handle=None)
    head, tail = Batch(reqs=[head_req], phase="prefill"), Batch(reqs=[tail_req], phase="prefill")
    head.tail_pair_role, tail.tail_pair_role = "head", "tail"
    head.tail_pair_plan = tail.tail_pair_plan = plan
    tail.paired_head = SimpleNamespace(batch=head, input_tuple=None)
    return head, tail


def _engine(stream, *, worker=False, enabled=False):
    engine = object.__new__(Engine)
    engine.stream = stream
    engine.ctx = Context(page_size=128)
    engine._dsv41_phase_priority = None
    engine.cpu_moe_executor = None
    engine.config = SimpleNamespace(speculative_dspark=False)
    engine._execution_plan = SimpleNamespace(is_expert_worker=worker, enabled=enabled,
                                             backbone_rank=0)
    engine.device = torch.device("cpu")
    log = []

    class Model:
        def forward(self):
            raise AssertionError("an unpaired forward ran for a paired batch")

        def forward_tail_pair(self, head_batch):
            batch = engine.ctx.batch
            log.append(("forward_tail_pair", head_batch, batch))
            return torch.tensor([[0.0, 3.0, 1.0]])

        def dspark_target_features(self):
            return None

    engine.model = Model()
    engine.graph_runner = SimpleNamespace(
        can_use_cuda_graph=lambda batch: (_ for _ in ()).throw(AssertionError("graph probed")))
    engine.sampler = SimpleNamespace(
        sample=lambda logits, args: log.append(("sample", logits.shape)) or logits.argmax(-1))
    return engine, log


def _trace_completion(batch, log, name):
    req = batch.reqs[0]
    original = req.complete_one

    def traced():
        log.append(("complete", name))
        original()

    req.complete_one = traced


def test_paired_forward_completes_head_then_tail_and_samples_once(cpu_cuda):
    head, tail = _pair()
    engine, log = _engine(cpu_cuda)
    _trace_completion(head, log, "head")
    _trace_completion(tail, log, "tail")
    out = engine.forward_batch(tail, None)
    assert log[0] == ("forward_tail_pair", head, tail)  # the outer context batch is the tail
    assert [e for e in log if e[0] == "complete"] == [("complete", "head"), ("complete", "tail")]
    assert [e for e in log if e[0] == "sample"] == [("sample", torch.Size([1, 3]))]
    assert out.next_tokens_gpu.tolist() == [1]
    assert (head.reqs[0].cached_len, tail.reqs[0].cached_len) == (16_384, 17_490)
    assert engine.ctx._batch is None


def test_paired_forward_broadcasts_the_tail_token_once_on_workers(cpu_cuda, monkeypatch):
    head, tail = _pair()
    engine, log = _engine(cpu_cuda, worker=True, enabled=True)
    broadcasts = []

    class Communicator:
        def broadcast(self, tensor, src):
            broadcasts.append((tuple(tensor.shape), src))
            return tensor

    monkeypatch.setattr(engine_mod, "DistributedCommunicator", Communicator)
    engine.forward_batch(tail, None)
    assert broadcasts == [((1,), 0)]
    assert not [e for e in log if e[0] == "sample"]  # workers never sample


def test_unpaired_batches_never_take_the_paired_branch(cpu_cuda):
    engine, log = _engine(cpu_cuda)
    req = Req(input_ids=torch.arange(10, dtype=torch.int32), table_idx=0, cached_len=0,
              output_len=2, uid=UID, sampling_params=SamplingParams(), cache_handle=None)
    batch = Batch(reqs=[req], phase="prefill")
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda b: False)
    engine.model.forward = lambda: torch.tensor([[1.0, 0.0]])
    engine.forward_batch(batch, None)
    assert not [e for e in log if e[0] == "forward_tail_pair"]


def _config(**overrides):
    values = dict(max_running_req=1, speculative_dspark=False, moe_backend="hybrid",
                  moe_prefill_overlap=True, tp_info=SimpleNamespace(rank=0),
                  max_extend_tokens=8192, moe_sparse_prefill_max_tokens=256, page_size=128)
    values.update(overrides)
    return SimpleNamespace(**values)


def _plan(**overrides):
    values = dict(uses_authority_transport=True, participates_in_prefill=True)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def moe_flag_seen(monkeypatch):
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", True)


def _validate(config, plan):
    """``_validate_tail_pairing`` for a config whose request ``_adjust_config`` left unchanged."""
    _validate_tail_pairing(config, plan, requested_max_running_req=config.max_running_req)


def test_startup_validation_accepts_the_production_geometry(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DSV41_PREFILL_EARLY_REFILL", raising=False)
    monkeypatch.setattr(prefill_trace, "enabled", lambda: False)
    _validate(_config(), _plan())
    _validate(_config(moe_backend="offload"), _plan())
    # a decode-only rank needs no prefill overlap reserve
    _validate(_config(moe_prefill_overlap=False), _plan(participates_in_prefill=False))


@pytest.mark.parametrize(
    ("config", "plan", "env", "trace", "message"),
    [
        (_config(), None, None, False, "authority-EP"),
        (_config(), _plan(uses_authority_transport=False), None, False, "authority-EP"),
        (_config(max_running_req=2), _plan(), None, False, "max-running-requests"),
        (_config(speculative_dspark=True), _plan(), None, False, "MTP"),
        (_config(), _plan(), "1", False, "EARLY_REFILL"),
        (_config(), _plan(), None, True, "PREFILL_TRACE"),
        (_config(moe_prefill_overlap=False), _plan(), None, False, "prefill overlap"),
        (_config(moe_backend="fused"), _plan(), None, False, "offload MoE backend"),
        # The qualified chunk geometry, also refused before the weight load.
        (_config(max_extend_tokens=16_384), _plan(), None, False, "--max-extend-tokens 8192"),
        (_config(max_extend_tokens=4096), _plan(), None, False, "--max-extend-tokens 8192"),
        (_config(moe_sparse_prefill_max_tokens=4096), _plan(), None, False,
         "--moe-sparse-prefill-max-tokens below 4096"),
    ],
)
def test_startup_validation_refuses_unsafe_execution(monkeypatch, config, plan, env, trace,
                                                     message):
    if env is None:
        monkeypatch.delenv("FREETOKEN_DSV41_PREFILL_EARLY_REFILL", raising=False)
    else:
        monkeypatch.setenv("FREETOKEN_DSV41_PREFILL_EARLY_REFILL", env)
    monkeypatch.setattr(prefill_trace, "enabled", lambda: trace)
    with pytest.raises(ValueError, match=message) as excinfo:
        _validate(config, plan)
    assert TAIL_PAIRING_ENV in str(excinfo.value)


@pytest.mark.parametrize(
    ("requested", "effective", "message"),
    [
        # DeepSeek-V4.1 is single-stream: _adjust_config has set 1 before the check runs.
        (2, 1, "--max-running-requests 1 (got 2)"),
        (4, 1, "--max-running-requests 1 (got 4)"),  # the --max-running-requests default
        (0, 1, "--max-running-requests 1 (got 0)"),
        # The value the runtime will use is still checked as well.
        (1, 2, "--max-running-requests 1 (effective 2)"),
    ],
)
def test_startup_validation_checks_the_requested_concurrency_not_only_the_clamped_one(
    monkeypatch, requested, effective, message
):
    monkeypatch.delenv("FREETOKEN_DSV41_PREFILL_EARLY_REFILL", raising=False)
    monkeypatch.setattr(prefill_trace, "enabled", lambda: False)
    with pytest.raises(ValueError) as excinfo:
        _validate_tail_pairing(_config(max_running_req=effective), _plan(),
                               requested_max_running_req=requested)
    assert str(excinfo.value) == f"{TAIL_PAIRING_ENV}=1 requires: {message}"


def test_startup_validation_refuses_a_moe_layer_flag_mismatch(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DSV41_PREFILL_EARLY_REFILL", raising=False)
    monkeypatch.setattr(prefill_trace, "enabled", lambda: False)
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", False)
    with pytest.raises(ValueError, match="MoE layers"):
        _validate(_config(), _plan())


def _ready_engine(*, prefill=True, **cache):
    engine = object.__new__(Engine)
    engine._dsv41_plan = SimpleNamespace(participates_in_prefill=prefill)
    values = dict(prefill_overlap=True, prefill_bank_buffers=[object()],
                  sparse_prefill_max_tokens=256)
    values.update(cache)
    engine.moe_offload_cache = SimpleNamespace(**values)
    return engine


def test_rank_readiness_requires_a_streaming_double_buffer():
    plan = TailPairPlan(UID, 0, 8192, 8192, 1106)
    assert _ready_engine().tail_pair_unready(plan) is None
    assert _ready_engine(sparse_prefill_max_tokens=0).tail_pair_unready(plan) is None
    assert "sparse path" in _ready_engine(sparse_prefill_max_tokens=1106).tail_pair_unready(plan)
    assert "double buffer" in _ready_engine(prefill_overlap=False).tail_pair_unready(plan)
    assert "double buffer" in _ready_engine(prefill_bank_buffers=[]).tail_pair_unready(plan)
    no_cache = _ready_engine()
    no_cache.moe_offload_cache = None
    assert "double buffer" in no_cache.tail_pair_unready(plan)
    # A decode-only rank runs no prefill MoE work for either chunk.
    assert _ready_engine(prefill=False, prefill_overlap=False).tail_pair_unready(plan) is None
    no_cache._dsv41_plan = None
    assert "DeepSeek-V4.1" in no_cache.tail_pair_unready(plan)


def test_engine_reads_the_flag_without_importing_the_scheduler_when_unset():
    import inspect

    source = inspect.getsource(Engine.__init__)
    guard = 'if os.environ.get("FREETOKEN_DSV41_TAIL_PAIRING", "0") != "0":'
    assert guard in source
    assert source.index(guard) < source.index("from freetoken.scheduler.tail_pairing import")
    # validated before the dense weight load and the expert-bank load, not after them
    assert source.index(guard) < source.index("self._load_weight_state_dict(config)")
    assert source.index(guard) < source.index("self._init_offload_moe_cache(config)")


def test_moe_flag_mirrors_the_scheduler_flag_name():
    import inspect

    source = inspect.getsource(dsv41_moe)
    assert f'os.getenv("{TAIL_PAIRING_ENV}", "0") == "1"' in source


# --------------------------------------------- startup through the real Engine.__init__ (G0)


class _PastStartupChecks(Exception):
    """Raised by the first statement after Engine.__init__'s pre-load configuration checks."""


def _tiny_dsv41_args() -> DeepseekV41Args:
    return DeepseekV41Args(
        vocab_size=64, dim=32, moe_inter_dim=32, n_layers=2, n_heads=2, q_lora_rank=32,
        head_dim=32, rope_head_dim=16, o_groups=2, o_lora_rank=16, compress_ratios=(0, 1),
        kv_source_layers=(1,), index_source_layers=(1,), candidate_source_layer=-1,
        index_n_heads=2, index_head_dim=32, index_topk=4, engram_layer_ids=(),
        engram_num_embeddings=(), n_routed_experts=8, n_activated_experts=2,
    )


# A two-rank stand-in for the production serve command, without --max-running-requests. The gate
# launcher appends its extra arguments last, so a repeated flag takes the last value.
_SERVE_ARGS = (
    "--model", "/models/anonymous", "--dtype", "bfloat16", "--tool-call-parser", "llama3",
    "--reasoning-parser", "off", "--tp-size", "2", "--dsv41-backbone-rank", "0",
    "--max-prefill-length", "8192", "--moe-backend", "hybrid", "--moe-hybrid-ranks", "1",
    "--moe-sparse-prefill-max-tokens", "256", "--attention-backend", "dsv4_sparse",
    "--cuda-graph-max-bs", "1",
)
_PRODUCTION_CONCURRENCY = ("--max-running-requests", "1")


def _serve_config(*extra: str) -> ServerArgs:
    config, _ = parse_args([*_SERVE_ARGS, *extra])
    return config


def _direct_config(cls, **overrides):
    values = dict(model_path="/models/anonymous", tp_info=DistributedInfo(0, 2),
                  dtype=torch.bfloat16, dsv41_backbone_rank=0, moe_backend="hybrid",
                  moe_hybrid_ranks=(1,), moe_cache_auto=True, moe_sparse_prefill_max_tokens=256,
                  attention_backend="dsv4_sparse")
    values.update(overrides)
    return cls(**values)


def _hf_config():
    return SimpleNamespace(architectures=["DeepseekV41ForCausalLM"],
                           _name_or_path="/models/anonymous", max_position_embeddings=1024)


@pytest.fixture
def startup(monkeypatch):
    """Run the real ``Engine.__init__`` on a real DeepSeek-V4.1 config up to the weight load.

    The model registry's own ``parse_config`` resolves the model config, so the single-stream
    declaration behind the max_running_req clamp is the shipped one; only the checkpoint read is
    replaced by tiny args. ``_adjust_config`` and the execution plan are real. The first statement
    after the pre-load checks raises ``_PastStartupChecks``; communication, the memory probe, model
    construction and the weight load fail the test if reached. Yields ``(start, validations)``:
    the keyword arguments of every ``_validate_tail_pairing`` call.
    """
    from freetoken import gpu_select
    from freetoken.distributed import info as distributed_info
    from freetoken.models.deepseek_v41 import config as dsv41_config
    from freetoken.models.deepseek_v41.execution import _reset_execution_for_tests
    from freetoken.models.qwen4_exp.execution import _reset_qwen4_exp_execution_for_tests

    def forbidden(name):
        def call(*_args, **_kwargs):
            raise AssertionError(f"{name} ran: startup went past the pre-load checks")

        return call

    def past_checks(*_args, **_kwargs):
        raise _PastStartupChecks

    monkeypatch.setattr(dsv41_config, "load_args", lambda path, **_: _tiny_dsv41_args())
    monkeypatch.setattr(gpu_select, "bind_assigned_gpu", lambda rank=0: torch.device("cpu"))
    monkeypatch.setattr(engine_mod, "_ensure_expandable_segments", lambda: None)
    monkeypatch.setattr(engine_mod.torch, "manual_seed", past_checks)
    for name in ("_init_communication", "_sync_get_memory", "_load_weight_state_dict"):
        monkeypatch.setattr(Engine, name, forbidden(name))
    monkeypatch.setattr(engine_mod, "create_model", forbidden("create_model"))
    validations = []
    validate = engine_mod._validate_tail_pairing

    def recording_validate(config, plan, **kwargs):
        validations.append(kwargs)
        return validate(config, plan, **kwargs)

    monkeypatch.setattr(engine_mod, "_validate_tail_pairing", recording_validate)
    for name in ("FREETOKEN_DSV41_PREFILL_EARLY_REFILL", "FREETOKEN_DSV41_CUDA_GRAPH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(prefill_trace, "enabled", lambda: False)

    def reset():
        _reset_execution_for_tests()
        _reset_qwen4_exp_execution_for_tests()
        distributed_info._TP_INFO = None

    def start(config):
        config.__dict__["hf_config"] = _hf_config()
        Engine(config)

    reset()
    try:
        yield start, validations
    finally:
        reset()


def _refusal(start, config) -> str:
    with pytest.raises(ValueError) as excinfo:
        start(config)
    return str(excinfo.value)


def test_requested_concurrency_2_is_refused_before_the_weight_load(monkeypatch, startup):
    start, _ = startup
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    config = _serve_config(*_PRODUCTION_CONCURRENCY, "--max-running-requests", "2")
    assert config.max_running_req == 2
    assert _refusal(start, config) == (
        f"{TAIL_PAIRING_ENV}=1 requires: --max-running-requests 1 (got 2)")
    # _adjust_config had already clamped the effective value; the refusal is of the request.
    assert config.max_running_req == 1


def test_requested_concurrency_1_passes_the_check(monkeypatch, startup):
    start, validations = startup
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    config = _serve_config(*_PRODUCTION_CONCURRENCY)
    with pytest.raises(_PastStartupChecks):
        start(config)
    assert validations == [{"requested_max_running_req": 1}]
    assert config.max_running_req == 1


@pytest.mark.parametrize("flag", [None, "0"], ids=["unset", "0"])
@pytest.mark.parametrize(
    "requested",
    [(), ("--max-running-requests", "1"), ("--max-running-requests", "2")],
    ids=["omitted", "1", "2"],
)
def test_flag_off_keeps_the_dsv41_single_stream_clamp(monkeypatch, startup, flag, requested):
    start, validations = startup
    if flag is None:
        monkeypatch.delenv(TAIL_PAIRING_ENV, raising=False)
    else:
        monkeypatch.setenv(TAIL_PAIRING_ENV, flag)
    config = _serve_config(*requested)
    with pytest.raises(_PastStartupChecks):
        start(config)
    assert not validations
    assert config.max_running_req == 1
    assert (config.cuda_graph_bs, config.cuda_graph_max_bs) == ([], 0)


@pytest.mark.parametrize(
    ("build", "requested"),
    [
        (lambda: _serve_config("--max-running-requests", "2"), 2),
        (lambda: _serve_config(), 4),  # flag omitted: the --max-running-requests default
        (lambda: _direct_config(ServerArgs, max_running_req=2), 2),
        (lambda: _direct_config(SchedulerConfig, max_running_req=2), 2),  # LLM(...)'s config
        (lambda: _direct_config(SchedulerConfig), 4),
        (lambda: _direct_config(SchedulerConfig, max_running_req=0), 0),
        # the launcher's per-rank copy (launch._apply_rank_local_cuda_visibility)
        (lambda: replace(_serve_config("--max-running-requests", "2"), vision_device=None), 2),
    ],
    ids=["cli", "cli-default", "server-args", "scheduler-config", "scheduler-config-default",
         "scheduler-config-zero", "launcher-replace"],
)
def test_no_construction_path_bypasses_the_requested_concurrency_check(
    monkeypatch, startup, build, requested
):
    start, _ = startup
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    config = build()
    assert config.max_running_req == requested
    assert _refusal(start, config) == (
        f"{TAIL_PAIRING_ENV}=1 requires: --max-running-requests 1 (got {requested})")


def test_the_offline_llm_api_refuses_dsv41_before_the_weight_load(monkeypatch, startup):
    from freetoken.engine import config as engine_config
    from freetoken.llm import LLM

    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path: _hf_config())
    # One process cannot serve DeepSeek-V4.1 at all, at any concurrency.
    with pytest.raises(ValueError, match="--tensor-parallel-size > 1"):
        LLM("/models/anonymous", dsv41_backbone_rank=0, moe_backend="hybrid", max_running_req=2)


_G0 = f"{TAIL_PAIRING_ENV}=1 requires: "


@pytest.mark.parametrize(
    ("env", "trace", "extra", "requirement"),
    [
        ({"FREETOKEN_DSV41_PREFILL_EARLY_REFILL": "1"}, False, (),
         "FREETOKEN_DSV41_PREFILL_EARLY_REFILL off"),
        ({}, True, (), "FREETOKEN_DSV41_PREFILL_TRACE off"),
        ({}, False, ("--max-running-requests", "2"), "--max-running-requests 1"),
        ({TAIL_PAIRING_ENV: "true"}, False, (), None),
        ({}, False, ("--max-prefill-length", "16384"),
         "--max-extend-tokens 8192, the qualified head chunk (got 16384)"),
        ({}, False, ("--max-prefill-length", "4096"),
         "--max-extend-tokens 8192, the qualified head chunk (got 4096)"),
        ({}, False, ("--moe-sparse-prefill-max-tokens", "4096"),
         "--moe-sparse-prefill-max-tokens below 4096, so that tails stream the bank (got 4096)"),
    ],
    ids=["early_refill", "prefill_trace", "max_running_2", "flag_true", "head_16384", "head_4096",
         "sparse_4096"],
)
def test_the_live_gate_g0_launches_are_refused_before_the_weight_load(
    monkeypatch, startup, env, trace, extra, requirement
):
    """The live gate's seven G0 launches (production argv + extra arguments last), on the host,
    judged as livegate_controller.py judges them."""
    start, _ = startup
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(prefill_trace, "enabled", lambda: trace)
    message = _refusal(start, _serve_config(*_PRODUCTION_CONCURRENCY, *extra))
    if requirement is None:
        assert message == f"{TAIL_PAIRING_ENV} must be '0' or '1', got 'true'"
    else:
        assert message.startswith(_G0) and requirement in message.split(_G0, 1)[1]
