"""DeepSeek-V4.1 layer-major tail pairing: scheduler-side host tests (CPU, no model, no GPU).

Covers the single eligibility decision, the plan wire format, the fail-closed cross-rank
agreement, paired-batch validation, and complete request lifecycles driven through the real
``Scheduler`` scheduling/forward/drain methods over a real DSV4.1 page pool and SWA radix tree --
paired against unpaired -- including cancellation and multimodal spans.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

import freetoken.scheduler.scheduler as scheduler_mod
from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.engine.engine import ForwardOutput
from freetoken.kvcache.dsv41_cost_model import dsv41_pool_sizes
from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
from freetoken.message import AbortBackendMsg, DetokenizeMsg, PromptAdmittedMsg
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.tail_pairing import (
    MAX_PROMPT_END,
    MAX_TAIL_TOKENS,
    TAIL_PAIRING_ENV,
    TailPairLimits,
    TailPairMismatch,
    TailPairPlan,
    agree_tail_pair,
    plan_tail_pair,
    tail_pair_limits,
    tail_pairing_enabled,
    validate_paired_batches,
)
from freetoken.scheduler.utils import PendingReq

DEVICE = torch.device("cpu")
P = 128
CHUNK = 8192
SPARSE = 256
UID = 7
IMAGE = 99


def _limits(**overrides) -> TailPairLimits:
    return TailPairLimits(head_tokens=CHUNK, sparse_max_tokens=SPARSE, page_size=P, **overrides)


# ----------------------------------------------------------------------------- flag


def test_flag_is_strict_and_default_off(monkeypatch):
    monkeypatch.delenv(TAIL_PAIRING_ENV, raising=False)
    assert tail_pairing_enabled() is False
    monkeypatch.setenv(TAIL_PAIRING_ENV, "0")
    assert tail_pairing_enabled() is False
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    assert tail_pairing_enabled() is True
    for bad in ("true", "on", "2", " 1", ""):
        monkeypatch.setenv(TAIL_PAIRING_ENV, bad)
        with pytest.raises(ValueError, match=TAIL_PAIRING_ENV):
            tail_pairing_enabled()


def test_limits_reject_unusable_geometry():
    assert _limits().max_tail_tokens == MAX_TAIL_TOKENS == 4096
    assert _limits().max_prompt_end == MAX_PROMPT_END == 131_072
    with pytest.raises(ValueError):
        TailPairLimits(head_tokens=8000, sparse_max_tokens=SPARSE, page_size=P)  # not page multiple
    with pytest.raises(ValueError):
        TailPairLimits(head_tokens=2048, sparse_max_tokens=SPARSE, page_size=P)  # tail > head
    with pytest.raises(ValueError):
        _limits(max_tail_tokens=SPARSE)
    with pytest.raises(TypeError):
        TailPairLimits(head_tokens=CHUNK, sparse_max_tokens=True, page_size=P)


def test_config_limits_require_the_qualified_head_and_a_streaming_sparse_threshold():
    config = SimpleNamespace(max_extend_tokens=CHUNK, moe_sparse_prefill_max_tokens=SPARSE,
                             page_size=P)
    assert tail_pair_limits(config) == _limits()
    no_sparse = SimpleNamespace(**{**vars(config), "moe_sparse_prefill_max_tokens": 0})
    assert tail_pair_limits(no_sparse).sparse_max_tokens == 0
    for head in (4096, 16_384, None):
        with pytest.raises(ValueError, match="--max-extend-tokens 8192"):
            tail_pair_limits(SimpleNamespace(**{**vars(config), "max_extend_tokens": head}))
    for sparse in (MAX_TAIL_TOKENS, CHUNK, -1):
        with pytest.raises(ValueError, match="--moe-sparse-prefill-max-tokens below 4096"):
            tail_pair_limits(
                SimpleNamespace(**{**vars(config), "moe_sparse_prefill_max_tokens": sparse}))


# ----------------------------------------------------------------------------- eligibility


def _req(cls, prompt_len, start, end, *, uid=UID, table_idx=0, max_tokens=8):
    return cls(
        input_ids=torch.arange(prompt_len, dtype=torch.int32)[:end],
        table_idx=table_idx,
        cached_len=start,
        output_len=max_tokens,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        cache_handle=SimpleNamespace(cached_len=0),
    )


def _scheduled_chunk(prompt_len, cached=0):
    """The batch the real chunker schedules just before the prompt's final chunk (or the
    single chunk): full 8K chunks from ``cached``, so the last full chunk is the head."""
    remain = prompt_len - cached
    if remain <= CHUNK:
        return Batch(reqs=[_req(Req, prompt_len, cached, prompt_len)], phase="prefill")
    full = (remain - 1) // CHUNK  # the final chunk takes the remainder (1..CHUNK)
    start = cached + (full - 1) * CHUNK
    return Batch(reqs=[_req(ChunkedReq, prompt_len, start, start + CHUNK)], phase="prefill")


def _decide(batch, prompt_len, **overrides):
    kwargs = dict(
        limits=_limits(),
        prompt_len=prompt_len,
        prefill_budget=CHUNK,
        decode_running=False,
        pending_matches=True,
        swa_free_tokens=10**9,
    )
    kwargs.update(overrides)
    return plan_tail_pair(batch, **kwargs)


@pytest.mark.parametrize(
    ("prompt_len", "cached", "expected"),
    [
        (17_490, 0, (8192, 8192, 1106)),  # the motivating fixture: chunks 8192 + [8192 + 1106]
        (9_298, 0, (0, 8192, 1106)),  # the pair is the request's first chunk
        (8_449, 0, (0, 8192, 257)),  # smallest tail that streams the bank
        (8_448, 0, None),  # 256-token tail takes the sparse slot path
        (12_288, 0, (0, 8192, 4096)),  # largest eligible tail
        (12_289, 0, None),  # 4,097
        (16_384, 0, None),  # exact multiple: the last chunk is full and final
        (131_072, 0, None),  # exact multiple at the depth bound
        (8_192, 0, None),  # single full chunk
        (5_000, 0, None),  # single short chunk
        (126_976, 0, (114_688, 8192, 4096)),  # deepest eligible from 8K-aligned chunks
        (132_178, 0, None),  # 16 x 8192 + 1,106: past the depth bound
        (14_288, 4_096, (4096, 8192, 2000)),  # page-aligned prefix-cache hit
        (131_072, 4_096, (118_784, 8192, 4096)),  # deepest eligible, with a prefix hit
    ],
)
def test_eligibility_boundaries(prompt_len, cached, expected):
    plan = _decide(_scheduled_chunk(prompt_len, cached), prompt_len)
    if expected is None:
        assert plan is None
    else:
        head_start, head_len, tail_len = expected
        assert plan == TailPairPlan(UID, 0, head_start, head_len, tail_len)
        assert plan.prompt_end == prompt_len
        assert plan.tail_start == head_start + head_len


def test_eligibility_refuses_every_unsafe_state():
    batch = _scheduled_chunk(17_490)
    assert _decide(batch, 17_490) is not None
    assert _decide(batch, 17_490, decode_running=True) is None  # concurrent decode
    assert _decide(batch, 17_490, pending_matches=False) is None  # stale queue
    assert _decide(batch, 17_490, prefill_budget=CHUNK - P) is None  # window-capped budget
    two = Batch(reqs=[batch.reqs[0], _req(Req, 100, 0, 100, uid=8, table_idx=1)], phase="prefill")
    assert _decide(two, 17_490) is None  # multiple requests
    decode = Batch(reqs=batch.reqs, phase="decode")
    assert _decide(decode, 17_490) is None
    batch.speculative = True
    assert _decide(batch, 17_490) is None
    shrunk = Batch(reqs=[_req(ChunkedReq, 17_490, 8192, 8192 + 8064)], phase="prefill")
    assert _decide(shrunk, 17_490) is None  # SWA-capped head
    unaligned = Batch(reqs=[_req(ChunkedReq, 17_490, 100, 100 + CHUNK)], phase="prefill")
    assert _decide(unaligned, 17_490) is None


def test_eligibility_requires_free_window_pages_without_eviction():
    batch = _scheduled_chunk(17_490)
    # tail [16384, 17490) spans 9 pages; one more page for the first decode step
    needed = (9 + 1) * P
    assert _decide(batch, 17_490, swa_free_tokens=needed) is not None
    assert _decide(batch, 17_490, swa_free_tokens=needed - P) is None
    assert _decide(batch, 17_490, swa_free_tokens=None) is not None  # no paged window pool


# ----------------------------------------------------------------------------- plan wire


def test_plan_wire_roundtrip_and_validation():
    plan = TailPairPlan(UID, 3, 8192, 8192, 1106)
    assert TailPairPlan.decode(plan.encode()) == plan
    assert TailPairPlan.decode(TailPairPlan.encode_none()) is None
    with pytest.raises(TypeError):
        TailPairPlan(UID, 3, 8192, 8192, True)
    for bad in (dict(head_len=0), dict(tail_len=0), dict(table_idx=-1), dict(head_start=-1)):
        fields = dict(uid=UID, table_idx=3, head_start=8192, head_len=8192, tail_len=1106)
        fields.update(bad)
        with pytest.raises(ValueError):
            TailPairPlan(**fields)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda w: w.to(torch.int32),
        lambda w: w[:7],
        lambda w: w.view(2, 4),
        lambda w: torch.cat([w[:1] + 1, w[1:]]),  # magic
        lambda w: torch.cat([w[:1], w[1:2] + 1, w[2:]]),  # version
        lambda w: torch.cat([w[:2], w.new_tensor([2]), w[3:]]),  # present flag
        lambda w: torch.cat([w[:6], w.new_tensor([0]), w[7:]]),  # empty head
        lambda w: torch.cat([w[:4], w.new_tensor([-1]), w[5:]]),  # negative row
    ],
)
def test_malformed_plan_wire_fails_closed(mutate):
    wire = mutate(TailPairPlan(UID, 3, 8192, 8192, 1106).encode())
    with pytest.raises(TailPairMismatch):
        TailPairPlan.decode(wire)


def test_absent_plan_wire_must_be_empty():
    wire = TailPairPlan.encode_none()
    wire[5] = 8192
    with pytest.raises(TailPairMismatch):
        TailPairPlan.decode(wire)


# ----------------------------------------------------------------------------- agreement


class _ThreadGroup:
    """In-process stand-in for the Gloo CPU group: real rendezvous across rank threads."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.calls = [[] for _ in range(size)]

    def broadcast(self, rank, tensor):
        self.calls[rank].append("broadcast")
        if rank == 0:
            self.slots[0] = tensor.clone()
        self.barrier.wait()
        out = self.slots[0].clone()
        self.barrier.wait()
        return out

    def all_reduce_sum(self, rank, tensor):
        self.calls[rank].append("all_reduce_sum")
        self.slots[rank] = tensor.clone()
        self.barrier.wait()
        out = sum(self.slots)
        self.barrier.wait()
        return out


def _agree_on_ranks(plans, unready=None):
    group = _ThreadGroup(len(plans))
    results = [None] * len(plans)

    def run(rank):
        try:
            results[rank] = agree_tail_pair(
                plans[rank],
                rank=rank,
                world_size=len(plans),
                broadcast=lambda t: group.broadcast(rank, t),
                all_reduce_sum=lambda t: group.all_reduce_sum(rank, t),
                unready=(unready or {}).get(rank),
            )
        except Exception as exc:  # noqa: BLE001
            results[rank] = exc

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(len(plans))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "agreement deadlocked"
    return results, group


def test_agreement_returns_the_common_plan_on_every_rank():
    plan = TailPairPlan(UID, 0, 8192, 8192, 1106)
    results, group = _agree_on_ranks([plan] * 4)
    assert results == [plan] * 4
    assert all(calls == ["broadcast", "all_reduce_sum"] for calls in group.calls)
    results, _ = _agree_on_ranks([None] * 4)
    assert results == [None] * 4


@pytest.mark.parametrize(
    "worker_plan",
    [None, TailPairPlan(UID, 0, 8192, 8192, 1105), TailPairPlan(UID + 1, 0, 8192, 8192, 1106)],
)
def test_one_divergent_rank_makes_every_rank_fail_closed(worker_plan):
    plan = TailPairPlan(UID, 0, 8192, 8192, 1106)
    results, _ = _agree_on_ranks([plan, plan, worker_plan, plan])
    assert all(isinstance(result, TailPairMismatch) for result in results)
    results, _ = _agree_on_ranks([None, None, plan, None])  # authority declines, a worker pairs
    assert all(isinstance(result, TailPairMismatch) for result in results)


@pytest.mark.parametrize("unready_rank", [0, 2])
def test_a_rank_that_cannot_execute_the_plan_makes_every_rank_fail_closed(unready_rank):
    plan = TailPairPlan(UID, 0, 8192, 8192, 1106)
    reason = "no MoE prefill double buffer on this rank"
    results, group = _agree_on_ranks([plan] * 4, unready={unready_rank: reason})
    assert all(isinstance(result, TailPairMismatch) for result in results)
    assert f"unready: {reason}" in str(results[unready_rank])
    # Folded into the existing agreement: still exactly one broadcast and one all-reduce.
    assert all(calls == ["broadcast", "all_reduce_sum"] for calls in group.calls)


def test_single_rank_agreement_issues_no_collective():
    def forbidden(_t):
        raise AssertionError("no collective at world size one")

    plan = TailPairPlan(UID, 0, 8192, 8192, 1106)
    assert agree_tail_pair(plan, rank=0, world_size=1, broadcast=forbidden,
                           all_reduce_sum=forbidden) is plan
    with pytest.raises(TailPairMismatch, match="cannot execute paired prefill: no buffer"):
        agree_tail_pair(plan, rank=0, world_size=1, broadcast=forbidden,
                        all_reduce_sum=forbidden, unready="no buffer")


# ----------------------------------------------------------------------------- validation


def _paired_batches(plan=None):
    plan = plan or TailPairPlan(UID, 0, 8192, 8192, 1106)
    head = Batch(reqs=[_req(ChunkedReq, plan.prompt_end, plan.head_start, plan.tail_start)],
                 phase="prefill")
    tail = Batch(reqs=[_req(Req, plan.prompt_end, plan.tail_start, plan.prompt_end)],
                 phase="prefill")
    head.input_ids = torch.zeros(plan.head_len, dtype=torch.int32)
    tail.input_ids = torch.zeros(plan.tail_len, dtype=torch.int32)
    head.tail_pair_role, tail.tail_pair_role = "head", "tail"
    head.tail_pair_plan = tail.tail_pair_plan = plan
    return head, tail


def test_validate_accepts_the_agreed_pair():
    head, tail = _paired_batches()
    assert validate_paired_batches(head, tail) is tail.tail_pair_plan


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda h, t: setattr(t, "tail_pair_plan", None),
        lambda h, t: setattr(h, "tail_pair_plan", TailPairPlan(UID, 0, 8192, 8192, 1106)),
        lambda h, t: setattr(h, "tail_pair_role", "tail"),
        lambda h, t: setattr(t, "tail_pair_role", None),
        lambda h, t: setattr(t, "input_ids", torch.zeros(1105, dtype=torch.int32)),
        lambda h, t: setattr(h, "input_ids", torch.zeros(8191, dtype=torch.int32)),
        lambda h, t: setattr(t.reqs[0], "uid", UID + 1),
        lambda h, t: setattr(t.reqs[0], "table_idx", 1),
        lambda h, t: setattr(t.reqs[0], "cached_len", 8192),
        lambda h, t: setattr(h.reqs[0], "device_len", 16_383),
        lambda h, t: t.reqs.append(t.reqs[0]),
        lambda h, t: setattr(h, "speculative", True),
        lambda h, t: setattr(t, "phase", "decode"),
        # Same geometry, wrong roles: a decoding head, or a tail that would not decode.
        lambda h, t: h.reqs.__setitem__(0, _req(Req, 17_490, 8192, 16_384)),
        lambda h, t: setattr(t.reqs[0], "max_device_len", t.reqs[0].device_len),
    ],
)
def test_validate_rejects_mismatched_metadata(corrupt):
    head, tail = _paired_batches()
    corrupt(head, tail)
    with pytest.raises(TailPairMismatch):
        validate_paired_batches(head, tail)


# ----------------------------------------------------------------------------- lifecycles


def _dsv41_args():
    return DeepseekV41Args(
        max_batch_size=2, max_seq_len=32768, vocab_size=128, dim=64, moe_inter_dim=32,
        n_layers=3, n_heads=2, n_routed_experts=4, n_activated_experts=2, q_lora_rank=64,
        head_dim=64, rope_head_dim=16, o_groups=1, o_lora_rank=32, window_size=P,
        compress_ratios=(0, 2, 2), kv_source_layers=(1,), index_source_layers=(1,),
        index_n_heads=2, index_head_dim=32, index_topk=2, candidate_source_layer=1,
        candidate_topk_blocks=1, candidate_block_size=2, engram_layer_ids=(),
        engram_num_embeddings=(),
    )


def _cpu_positions(batch, device):
    return torch.cat([
        torch.arange(r.cached_len, r.device_len, dtype=torch.int32) for r in batch.padded_reqs
    ]).to(device)


def _cpu_input_tuple(batch, device):
    rows = torch.cat([
        torch.full((r.extend_len,), r.table_idx, dtype=torch.int64) for r in batch.padded_reqs
    ])
    return rows.to(device), batch.positions.to(torch.int64)


def _cpu_rope_positions(batch, device):
    assert all(r.prompt_rope_positions is None for r in batch.padded_reqs)
    return torch.cat([
        torch.arange(r.cached_len, r.device_len, dtype=torch.int32)[:, None].expand(-1, 3)
        for r in batch.padded_reqs
    ]).to(device)


def _cpu_write_tuple(batch, device):
    rows = torch.tensor([r.table_idx for r in batch.reqs], dtype=torch.int64)
    write = torch.tensor([r.device_len if r.can_decode else -1 for r in batch.reqs],
                         dtype=torch.int64)
    return rows.to(device), write.to(device)


@pytest.fixture
def cpu_scheduler_helpers(monkeypatch):
    """The scheduler's pinned-memory helpers, minus the pinning (no CUDA here)."""
    monkeypatch.setattr(scheduler_mod, "_make_positions", _cpu_positions)
    monkeypatch.setattr(scheduler_mod, "_make_input_tuple", _cpu_input_tuple)
    monkeypatch.setattr(scheduler_mod, "_make_write_tuple", _cpu_write_tuple)
    monkeypatch.setattr(scheduler_mod, "_make_rope_positions", _cpu_rope_positions)
    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=P))


class _StubEngine:
    """The engine surface the scheduler touches; forward bookkeeping as Engine.forward_batch."""

    def __init__(self, page_table):
        self.page_table = page_table
        self.linear_state_pool = None
        self.graph_runner = SimpleNamespace(pad_batch=self._pad)
        self.attn_backend = SimpleNamespace(prepare_metadata=lambda batch: None)
        self.sampler = SimpleNamespace(prepare=lambda batch: None)
        self.forwards = []
        self.unready = None  # Engine.tail_pair_unready's verdict for every plan

    @staticmethod
    def _pad(batch):
        batch.padded_reqs = batch.reqs

    def tail_pair_unready(self, plan):
        return self.unready

    def forward_batch(self, batch, _args):
        paired = getattr(batch, "paired_head", None)
        record = dict(phase=batch.phase, spans=[], tokens=[])
        for part in ([paired.batch] if paired is not None else []) + [batch]:
            req = part.reqs[0]
            record["spans"].append((req.cached_len, req.device_len))
            record["tokens"].append(part.input_ids.tolist())
        self.forwards.append(record)
        if paired is not None:
            validate_paired_batches(paired.batch, batch)
            for req in paired.batch.reqs:
                req.complete_one()
        for req in batch.reqs:
            req.complete_one()
        tokens = torch.full((batch.size,), 5, dtype=torch.int32)
        return ForwardOutput(tokens, tokens, SimpleNamespace(synchronize=lambda: None))


def _scheduler(*, pairing, num_pages=256, win_pages=160, max_running=1):
    args = _dsv41_args()
    sizes = dsv41_pool_sizes(num_pages + 1, args, 1.0, P=P, n_win_pages=win_pages)
    pool = DSV41PagedKVCache(sizes, args, DEVICE, P=P, n_scratch=max_running + 1)
    pool._init_paged_state(max_running, True)
    page_table = torch.zeros((max_running + 1, num_pages * P), dtype=torch.int32)
    page_table[max_running].fill_(num_pages * P)
    pool.full_loc_map = page_table
    cm = CacheManager(num_pages=num_pages, page_size=P, page_table=page_table, type="swa_radix",
                      swa_pool=pool, sliding_window_size=P)
    tm = TableManager(max_running_reqs=max_running, page_table=page_table)
    dm = DecodeManager(page_size=P)
    sched = object.__new__(Scheduler)
    sent = []
    sched.engine = _StubEngine(page_table)
    sched.cache_manager, sched.table_manager, sched.decode_manager = cm, tm, dm
    sched.prefill_manager = PrefillManager(cm, tm, dm)
    sched.token_pool = tm.token_pool
    sched.prefill_budget = min(CHUNK, cm.prefill_chunk_budget)
    sched.device = DEVICE
    sched.config = SimpleNamespace(
        tp_info=SimpleNamespace(rank=0, size=1, is_primary=lambda: True),
        model_config=SimpleNamespace(image_token_id=IMAGE),
        page_size=P,
    )
    sched._forward_iter = 0
    sched._speculative = None
    sched._tail_pairing = _limits() if pairing else None
    sched.finished_reqs = set()
    sched.eos_token_ids = set()
    sched.toolcall_anchor_id = None
    sched.status_reporter = SimpleNamespace(report_batch=lambda batch, **_: sent.append(
        ("report", batch.log_new_tokens, batch.log_cached_tokens)))
    sched.send_result = sent.extend
    sched._pending_abort_acks = set()
    sched._abort_tombstones = {}
    sched._last_data = None
    sched.sent = sent
    return sched, pool


def _submit(sched, prompt_len, *, uid=UID, max_tokens=6, mm=None):
    ids = torch.arange(prompt_len, dtype=torch.int32) % 90 + 1  # never the image token
    is_mm = mm is not None
    if is_mm:
        ids = ids.clone()
        ids[torch.tensor(mm[0], dtype=torch.long)] = IMAGE
    sched.prefill_manager.pending_list.append(PendingReq(
        uid=uid, input_ids=ids, sampling_params=SamplingParams(max_tokens=max_tokens),
        mm_embeds=mm[1] if is_mm else None, is_multimodal=is_mm,
    ))
    return ids


def _run(sched, *, abort_after_forward=None, mm_capture=None):
    """The overlap loop in miniature: schedule+launch N+1, then drain N."""
    last = None
    launched = 0
    while True:
        sched._last_data = last
        if abort_after_forward is not None and launched == abort_after_forward:
            sched._process_one_msg(AbortBackendMsg(uid=UID))
        forward_input = sched._schedule_next_batch()
        ongoing = None
        if forward_input is not None:
            if mm_capture is not None:
                paired = getattr(forward_input.batch, "paired_head", None)
                for part in ([paired.batch] if paired else []) + [forward_input.batch]:
                    if part.is_prefill:
                        req = part.reqs[0]
                        mm_capture.append(((req.cached_len, req.device_len), part.mm_embeds))
            ongoing = (forward_input, sched._forward(forward_input))
            launched += 1
        if last is not None:
            sched._process_last_data(last)
            sched._flush_abort_acks()
        last = ongoing
        if forward_input is None and last is None:
            break
    sched._last_data = None
    sched._flush_abort_acks()


def _tree_state(cm, pool, prompt):
    m = cm.prefix_cache.match_prefix(prompt[: len(prompt) - 1])
    pc = cm.prefix_cache
    return dict(
        free_pages=sorted(cm.free_slots.tolist()),
        swa_free=pool._win_alloc.available(),
        matched=m.cached_len,
        matched_full=m.kv_indices.tolist(),
        full_evictable=pc.full_evictable,
        full_protected=pc.full_protected,
        swa_evictable=pc.swa_evictable,
        swa_protected=pc.swa_protected,
    )


def _replies(sched):
    return [(m.uid, m.next_token, m.finished, m.finish_reason)
            for m in sched.sent if isinstance(m, DetokenizeMsg)]


@pytest.mark.usefixtures("cpu_scheduler_helpers")
@pytest.mark.parametrize("prompt_len", [17_490, 9_298, 8_449, 12_288, 16_384, 8_448, 5_000])
def test_paired_lifecycle_matches_unpaired_state(prompt_len):
    states = {}
    for pairing in (False, True):
        sched, pool = _scheduler(pairing=pairing)
        prompt = _submit(sched, prompt_len)
        first_reply = []
        send = sched.send_result

        def snapshot(msgs, sched=sched, pool=pool, send=send, first_reply=first_reply):
            if not first_reply and any(isinstance(m, DetokenizeMsg) for m in msgs):
                pc = sched.cache_manager.prefix_cache  # just after the final chunk's commit
                first_reply.append((pool._win_alloc.available(), pc.swa_evictable,
                                    pc.swa_protected, pc.full_evictable, pc.full_protected,
                                    sorted(sched.cache_manager.free_slots.tolist())))
            send(msgs)

        sched.send_result = snapshot
        _run(sched)
        sched.cache_manager.check_integrity()
        assert sched.table_manager.available_size == 1
        states[pairing] = dict(
            first_reply=first_reply,
            tree=_tree_state(sched.cache_manager, pool, prompt),
            replies=_replies(sched),
            admissions=[(m.uid, m.prompt_tokens, m.cached_tokens)
                        for m in sched.sent if isinstance(m, PromptAdmittedMsg)],
            spans=[f["spans"] for f in sched.engine.forwards if f["phase"] == "prefill"],
            tokens=[t for f in sched.engine.forwards if f["phase"] == "prefill"
                    for t in f["tokens"]],
            reports=[r for r in sched.sent if isinstance(r, tuple) and r[0] == "report"
                     and r[1] > 1],
        )
    off, on = states[False], states[True]
    assert on["first_reply"] == off["first_reply"] and len(on["first_reply"]) == 1
    assert on["tree"] == off["tree"]
    assert on["reports"] == off["reports"] if on["spans"] == off["spans"] else (
        sum(r[1] for r in on["reports"]) == sum(r[1] for r in off["reports"]))
    assert on["replies"] == off["replies"] and on["replies"][-1][2]  # finished once
    assert on["admissions"] == off["admissions"] == [(UID, prompt_len, 0)]
    # identical logical chunks and token ids, only the execution grouping differs
    flat = lambda spans: [span for group in spans for span in group]  # noqa: E731
    assert flat(on["spans"]) == flat(off["spans"])
    assert on["tokens"] == off["tokens"]
    tail = prompt_len % CHUNK
    pairs = [spans for spans in on["spans"] if len(spans) == 2]
    if prompt_len > CHUNK and SPARSE < tail <= MAX_TAIL_TOKENS:
        assert pairs == [[(prompt_len - tail - CHUNK, prompt_len - tail),
                          (prompt_len - tail, prompt_len)]]
    else:
        assert pairs == []
        assert on["spans"] == off["spans"]


@pytest.mark.usefixtures("cpu_scheduler_helpers")
def test_paired_drain_releases_head_window_before_the_radix_commit(monkeypatch):
    sched, pool = _scheduler(pairing=True)
    _submit(sched, 17_490)
    events = []
    release = sched.cache_manager.free_swa_out_of_window_before
    commit = sched.cache_manager.cache_req

    def traced_release(req, cached_len):
        events.append(("release", cached_len, req.swa_evicted_seqlen))
        release(req, cached_len)

    def traced_commit(req, *, finished):
        events.append(("commit", finished, req.swa_evicted_seqlen))
        commit(req, finished=finished)

    monkeypatch.setattr(sched.cache_manager, "free_swa_out_of_window_before", traced_release)
    monkeypatch.setattr(sched.cache_manager, "cache_req", traced_commit)
    _run(sched)
    # the tail's deferred extend-free (frontier = its schedule-time start 16,384) precedes the
    # prefix commit, so the commit sees the unpaired watermark align_down(16384 - 256) = 16128
    pair_release = [e for e in events if e[0] == "release" and e[1] == 16_384]
    assert len(pair_release) == 1
    assert events.index(pair_release[0]) < events.index(("commit", False, 16_128))


@pytest.mark.usefixtures("cpu_scheduler_helpers")
def test_paired_window_footprint_and_no_tree_eviction(monkeypatch):
    sched, pool = _scheduler(pairing=True)
    _submit(sched, 17_490)
    evictions = []
    real = sched.cache_manager.prefix_cache.evict_swa
    monkeypatch.setattr(sched.cache_manager.prefix_cache, "evict_swa",
                        lambda n: evictions.append(n) or real(n))
    seen = []
    forward = sched.engine.forward_batch

    def observe(batch, args):
        paired = getattr(batch, "paired_head", None)
        seen.append((paired is not None,
                     pool.swa_num_tokens - 1 - pool._win_alloc.available()))
        if paired is not None:
            # Every window slot either chunk's attention can read -- the head's own positions
            # and the window before it, then the tail -- is still bound when the pair runs.
            head = paired.batch.reqs[0]
            row = sched.cache_manager.page_table[head.table_idx]
            readable = row[max(0, head.cached_len - (P - 1)) : batch.reqs[0].device_len]
            assert bool((pool.translate_full_to_window(readable) >= 0).all())
        return forward(batch, args)

    sched.engine.forward_batch = observe
    _run(sched)
    assert evictions == []
    paired_live = [live for paired, live in seen if paired]
    # head's early pages [7936, 16128) are still bound while both chunks run: 2 + 64 + 9 pages
    assert paired_live == [(2 + 64 + 9) * P]


@pytest.mark.usefixtures("cpu_scheduler_helpers")
def test_window_pool_pressure_falls_back_to_unpaired():
    # Another owner holds 90 of the 159 window pages: after the head only 3 stay free, fewer
    # than the tail's 9 + 1. Pairing must decline instead of evicting; unpaired still fits,
    # because the head's early pages return before the tail allocates.
    sched, pool = _scheduler(pairing=True)
    held = pool._win_alloc.alloc(90)
    assert sched.prefill_budget == CHUNK
    _submit(sched, 17_490)
    _run(sched)
    assert [len(f["spans"]) for f in sched.engine.forwards if f["phase"] == "prefill"] == [1, 1, 1]
    assert _replies(sched)[-1][2]
    pool._win_alloc.free(held)
    sched.cache_manager.check_integrity()


@pytest.mark.usefixtures("cpu_scheduler_helpers")
def test_a_rank_that_cannot_execute_the_pair_fails_closed_before_scheduling_the_tail():
    sched, _ = _scheduler(pairing=True)
    sched.engine.unready = "no MoE prefill double buffer on this rank"
    _submit(sched, 17_490)
    with pytest.raises(TailPairMismatch, match="no MoE prefill double buffer"):
        _run(sched)
    # Chunk 0 ran; the head was prepared but never launched, and no tail was scheduled.
    assert [f["spans"] for f in sched.engine.forwards] == [[(0, 8192)]]
    pending = sched.prefill_manager.pending_list
    assert len(pending) == 1 and isinstance(pending[0].chunked_req, ChunkedReq)
    assert pending[0].chunked_req.device_len == 16_384


@pytest.mark.usefixtures("cpu_scheduler_helpers")
def test_pairing_off_issues_no_agreement_and_sets_no_pair_state(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("pairing machinery reached with the flag off")

    monkeypatch.setattr(scheduler_mod, "plan_tail_pair", forbidden)
    monkeypatch.setattr(scheduler_mod, "agree_tail_pair", forbidden)
    sched, _ = _scheduler(pairing=False)
    _submit(sched, 17_490)
    batches = []
    forward = sched.engine.forward_batch
    sched.engine.forward_batch = lambda batch, args: batches.append(batch) or forward(batch, args)
    _run(sched)
    assert all(b.paired_head is None and b.tail_pair_role is None and b.tail_pair_plan is None
               for b in batches)


def test_scheduler_resolves_limits_only_for_dsv41_single_stream(monkeypatch):
    config = SimpleNamespace(
        model_config=SimpleNamespace(dsv41_args=object()), max_running_req=1,
        max_extend_tokens=CHUNK, moe_sparse_prefill_max_tokens=SPARSE, page_size=P,
    )
    monkeypatch.delenv(TAIL_PAIRING_ENV, raising=False)
    assert scheduler_mod._tail_pairing_limits(config) is None
    monkeypatch.setenv(TAIL_PAIRING_ENV, "1")
    assert scheduler_mod._tail_pairing_limits(config) == _limits()
    assert scheduler_mod._tail_pairing_limits(
        SimpleNamespace(**{**vars(config), "max_running_req": 2})
    ) is None
    with pytest.raises(ValueError, match="DeepSeek-V4.1"):
        scheduler_mod._tail_pairing_limits(
            SimpleNamespace(**{**vars(config), "model_config": SimpleNamespace(dsv41_args=None)}))
    with pytest.raises(ValueError, match=f"{TAIL_PAIRING_ENV}=1 requires --max-extend-tokens"):
        scheduler_mod._tail_pairing_limits(
            SimpleNamespace(**{**vars(config), "max_extend_tokens": 16_384}))
    with pytest.raises(ValueError, match="--moe-sparse-prefill-max-tokens below"):
        scheduler_mod._tail_pairing_limits(
            SimpleNamespace(**{**vars(config), "moe_sparse_prefill_max_tokens": MAX_TAIL_TOKENS}))


# ----------------------------------------------------------------------------- cancellation


def _abort_run(pairing, abort_after_forward):
    sched, pool = _scheduler(pairing=pairing)
    prompt = _submit(sched, 17_490)
    _run(sched, abort_after_forward=abort_after_forward)
    sched.cache_manager.check_integrity()
    return sched, pool, prompt


@pytest.mark.usefixtures("cpu_scheduler_helpers")
@pytest.mark.parametrize("abort_after_forward", [0, 1, 2, 3, 5])
def test_cancellation_before_between_and_after_sub_chunks(abort_after_forward):
    """Forward 1 is chunk 0 and forward 2 is the paired chunk (ON) or chunk 1 (OFF); the abort
    lands before anything runs, while chunk 0 runs, while the pair runs, or during decode."""
    for pairing in (False, True):
        sched, _pool, _ = _abort_run(pairing, abort_after_forward)
        acks = [m for m in sched.sent if getattr(m, "error", None) == "request aborted"]
        assert [m.uid for m in acks] == [UID]  # exactly one terminal acknowledgement
        replies = _replies(sched)
        assert all(not finished for _uid, _tok, finished, _ in replies)  # never a finish reply
        assert sched.table_manager.available_size == 1  # row returned exactly once
        assert not sched.decode_manager.runnable and not sched.prefill_manager.runnable
        if pairing and abort_after_forward == 2:
            # the abort lands while the paired forward is in flight: both chunks finish on
            # the device, the drain frees once, and no token is published
            assert [len(f["spans"]) for f in sched.engine.forwards] == [1, 2]
            assert replies == []


@pytest.mark.usefixtures("cpu_scheduler_helpers")
def test_inflight_abort_marks_the_tail_and_the_drain_frees_once():
    sched, pool = _scheduler(pairing=True)
    _submit(sched, 17_490)
    first = sched._schedule_next_batch()
    first_out = (first, sched._forward(first))  # chunk 0
    sched._last_data = first_out
    paired = sched._schedule_next_batch()
    assert paired.batch.paired_head is not None
    paired_out = (paired, sched._forward(paired))
    sched._process_last_data(first_out)
    sched._last_data = paired_out
    tail = paired.batch.reqs[0]
    sched._process_one_msg(AbortBackendMsg(uid=UID))
    assert tail.aborted and tail.table_idx != -1  # marked, never freed under the forward
    assert sched._schedule_next_batch() is None
    sched._process_last_data(paired_out)
    assert tail.table_idx == -1
    assert tail.swa_evicted_seqlen == 16_128  # the deferred release still ran before the free
    sched._flush_abort_acks()
    assert [m.uid for m in sched.sent if getattr(m, "error", None) == "request aborted"] == [UID]
    assert _replies(sched) == []
    sched.cache_manager.check_integrity()  # free + tree == capacity: nothing leaked or doubled


# ----------------------------------------------------------------------------- multimodal


@pytest.mark.usefixtures("cpu_scheduler_helpers")
def test_multimodal_spans_crossing_pair_boundaries_slice_identically():
    prompt_len = 17_490
    # image spans crossing a = 8192 (head start), b = 16384 (head/tail), inside the tail, and
    # ending at c (the prompt's last token)
    positions = (list(range(8100, 8300)) + list(range(16_300, 16_500))
                 + list(range(17_000, 17_100)) + list(range(17_480, 17_490)))
    features = torch.arange(len(positions) * 4, dtype=torch.float32).view(-1, 4)
    captured = {}
    for pairing in (False, True):
        sched, _ = _scheduler(pairing=pairing)
        _submit(sched, prompt_len, mm=(positions, features))
        capture = []
        _run(sched, mm_capture=capture)
        sched.cache_manager.check_integrity()
        captured[pairing] = capture
        spans = [f["spans"] for f in sched.engine.forwards if f["phase"] == "prefill"]
        assert [len(s) for s in spans] == ([1, 2] if pairing else [1, 1, 1])
    assert [span for span, _ in captured[True]] == [span for span, _ in captured[False]]
    for (span, on), (_, off) in zip(captured[True], captured[False]):
        assert on is not None and off is not None and torch.equal(on, off), span
    assert [span for span, _ in captured[True]] == [(0, 8192), (8192, 16_384), (16_384, 17_490)]
    torch.testing.assert_close(torch.cat([f for _, f in captured[True]]), features)
