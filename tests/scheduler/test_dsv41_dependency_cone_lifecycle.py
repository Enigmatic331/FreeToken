"""DeepSeek-V4.1 dependency-cone prefill through the real ``Scheduler`` (CPU, no model, no GPU).

Complete request lifecycles run through the real scheduling, forward-bookkeeping and drain
methods, over a real DSV4.1 page pool and SWA radix tree, with the cone on and off. The checks:

* every forward carries the agreed segment, and the plain run carries none;
* the cache after finish equals the plain run's, and so does the next turn's prefix hit;
* cancellation at every point frees exactly once and publishes nothing inexact;
* with the flag off, no cone code or collective runs; a divergent rank fails closed before the
  forward.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.scheduler.scheduler as scheduler_mod
from freetoken.core import Context, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.engine.engine import ForwardOutput
from freetoken.kvcache.dsv41_cost_model import dsv41_pool_sizes
from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
from freetoken.message import AbortBackendMsg, DetokenizeMsg
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.scheduler import cache as cache_mod
from freetoken.scheduler import dependency_cone as dc
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

DEVICE = torch.device("cpu")
P = 128
CHUNK = 8192
UID = 9
REAL_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)


def _geometry():
    """The checkpoint's cone: 40 layers, decoder from layer 20, window 128."""
    return dc.ConeGeometry.from_args(DeepseekV41Args(compress_ratios=REAL_RATIOS), page_size=P,
                                     retain_gap=cache_mod._SWA_RETAIN_GAP)


def _pool_args():
    # The page pool only needs the window and one source; the cone's geometry is the real one.
    return DeepseekV41Args(
        max_batch_size=2, max_seq_len=32768, vocab_size=128, dim=64, moe_inter_dim=32,
        n_layers=3, n_heads=2, n_routed_experts=4, n_activated_experts=2, q_lora_rank=64,
        head_dim=64, rope_head_dim=16, o_groups=1, o_lora_rank=32, window_size=P,
        compress_ratios=(0, 2, 2), kv_source_layers=(1,), index_source_layers=(1,),
        index_n_heads=2, index_head_dim=32, index_topk=2, candidate_source_layer=1,
        candidate_topk_blocks=1, candidate_block_size=2, engram_layer_ids=(),
        engram_num_embeddings=(),
    )


@pytest.fixture(autouse=True)
def cpu_scheduler_helpers(monkeypatch):
    """The scheduler's pinned-memory helpers, minus the pinning (no CUDA here)."""

    def positions(batch, device):
        return torch.cat([torch.arange(r.cached_len, r.device_len, dtype=torch.int32)
                          for r in batch.padded_reqs]).to(device)

    def input_tuple(batch, device):
        rows = torch.cat([torch.full((r.extend_len,), r.table_idx, dtype=torch.int64)
                          for r in batch.padded_reqs])
        return rows.to(device), batch.positions.to(torch.int64)

    def write_tuple(batch, device):
        rows = torch.tensor([r.table_idx for r in batch.reqs], dtype=torch.int64)
        write = torch.tensor([r.device_len if r.can_decode else -1 for r in batch.reqs],
                             dtype=torch.int64)
        return rows.to(device), write.to(device)

    monkeypatch.setattr(scheduler_mod, "_make_positions", positions)
    monkeypatch.setattr(scheduler_mod, "_make_input_tuple", input_tuple)
    monkeypatch.setattr(scheduler_mod, "_make_write_tuple", write_tuple)
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

    @staticmethod
    def _pad(batch):
        batch.padded_reqs = batch.reqs

    def forward_batch(self, batch, _args):
        req = batch.reqs[0]
        self.forwards.append((batch.phase, (req.cached_len, req.device_len), batch.dsv41_cone))
        for r in batch.reqs:
            r.complete_one()
        tokens = torch.full((batch.size,), 5, dtype=torch.int32)
        return ForwardOutput(tokens, tokens, SimpleNamespace(synchronize=lambda: None))


def _scheduler(*, cone, num_pages=256, win_pages=160, rank=0, size=1):
    args = _pool_args()
    sizes = dsv41_pool_sizes(num_pages + 1, args, 1.0, P=P, n_win_pages=win_pages)
    pool = DSV41PagedKVCache(sizes, args, DEVICE, P=P, n_scratch=2)
    pool._init_paged_state(1, True)
    page_table = torch.zeros((2, num_pages * P), dtype=torch.int32)
    page_table[1].fill_(num_pages * P)
    pool.full_loc_map = page_table
    cm = CacheManager(num_pages=num_pages, page_size=P, page_table=page_table, type="swa_radix",
                      swa_pool=pool, sliding_window_size=P)
    tm = TableManager(max_running_reqs=1, page_table=page_table)
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
        tp_info=SimpleNamespace(rank=rank, size=size, is_primary=lambda: rank == 0),
        model_config=SimpleNamespace(image_token_id=None), page_size=P,
    )
    sched._forward_iter = 0
    sched._speculative = None
    sched._tail_pairing = None
    sched._dependency_cone = _geometry() if cone else None
    sched.finished_reqs = set()
    sched.eos_token_ids = set()
    sched.toolcall_anchor_id = None
    sched.status_reporter = SimpleNamespace(report_batch=lambda batch, **_: None)
    sched.send_result = sent.extend
    sched._pending_abort_acks = set()
    sched._abort_tombstones = {}
    sched._last_data = None
    sched.sent = sent
    return sched, pool


def _submit(sched, prompt):
    sched.prefill_manager.pending_list.append(PendingReq(
        uid=UID, input_ids=prompt, sampling_params=SamplingParams(max_tokens=6)))


def _prompt(n, base=1):
    return (torch.arange(n, dtype=torch.int32) + base) % 90 + 1


def _run(sched, *, abort_after_forward=None):
    """The overlap loop in miniature: schedule + launch N+1, then drain N."""
    last, launched = None, 0
    while True:
        sched._last_data = last
        if abort_after_forward is not None and launched == abort_after_forward:
            sched._process_one_msg(AbortBackendMsg(uid=UID))
        forward_input = sched._schedule_next_batch()
        ongoing = None
        if forward_input is not None:
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


def _tree(sched, pool, prompt):
    cm, pc = sched.cache_manager, sched.cache_manager.prefix_cache
    return dict(
        free_pages=len(cm.free_slots),
        swa_free=pool._win_alloc.available(),
        matched=pc.match_prefix(prompt[: len(prompt) - 1]).cached_len,
        full=(pc.full_evictable, pc.full_protected),
        swa=(pc.swa_evictable, pc.swa_protected),
    )


def _replies(sched):
    return [(m.uid, m.next_token, m.finished, m.finish_reason)
            for m in sched.sent if isinstance(m, DetokenizeMsg)]


@pytest.mark.parametrize("prompt_len", [17_490, 26_000, 9_000])
def test_lifecycle_matches_the_plain_prefill(prompt_len):
    g = _geometry()
    latest = g.latest_decoder_start(prompt_len)
    results = {}
    for cone in (False, True):
        sched, pool = _scheduler(cone=cone)
        prompt = _prompt(prompt_len)
        _submit(sched, prompt)
        _run(sched)
        sched.cache_manager.check_integrity()
        results[cone] = (sched.engine.forwards, _tree(sched, pool, prompt), _replies(sched))
    (plain_fw, plain_tree, plain_replies), (cone_fw, cone_tree, cone_replies) = (
        results[False], results[True])
    assert [f[:2] for f in cone_fw] == [f[:2] for f in plain_fw]  # same chunks, same steps
    assert all(f[2] is None for f in plain_fw)
    prefill = [f for f in cone_fw if f[0] == "prefill"]
    if prefill[0][1][1] > latest:  # the first chunk crosses R*: the plain prefill
        assert all(f[2] is None for f in prefill)
    else:
        segments = [f[2] for f in prefill]
        r0 = next(s.start for s in segments if not s.is_encoder)
        for (start, end), seg in zip((f[1] for f in prefill), segments):
            assert (seg.start, seg.end) == (start, end)
            assert seg.is_encoder == (end <= latest)
            assert seg.is_encoder or seg.decoder_start == r0
    assert all(f[2] is None for f in cone_fw if f[0] == "decode")
    assert cone_tree == plain_tree  # after finish: the same reusable cache
    assert cone_replies == plain_replies


def test_the_next_turn_hits_the_same_prefix():
    """Turn two (prompt + reply + new message) resumes at the same point with the cone on."""
    hits = {}
    for cone in (False, True):
        sched, _pool = _scheduler(cone=cone)
        prompt = _prompt(17_490)
        _submit(sched, prompt)
        _run(sched)
        turn2 = torch.cat([prompt, torch.full((6,), 5, dtype=torch.int32), _prompt(300, 7)])
        _submit(sched, turn2)
        admitted = []
        report = sched._report_prompt_admissions
        sched._report_prompt_admissions = lambda batch: (
            admitted.extend(batch.prompt_admissions), report(batch))
        _run(sched)
        sched.cache_manager.check_integrity()
        hits[cone] = admitted
    assert hits[True] == hits[False] and hits[True][0][2] > 17_000


@pytest.mark.parametrize("abort_after_forward", [0, 1, 2, 3, 5])
def test_cancellation_frees_once_and_publishes_nothing_inexact(abort_after_forward):
    """Forward 1 is chunk [0, 8192) (encoder-only), forward 2 the first decoder chunk, forward 3
    the final chunk; the abort lands before anything runs, mid-prefill, or during decode."""
    g = _geometry()
    sched, _pool = _scheduler(cone=True)
    prompt = _prompt(17_490)
    _submit(sched, prompt)
    _run(sched, abort_after_forward=abort_after_forward)
    sched.cache_manager.check_integrity()
    acks = [m for m in sched.sent if getattr(m, "error", None) == "request aborted"]
    assert [m.uid for m in acks] == [UID]
    assert all(not finished for _u, _t, finished, _r in _replies(sched))
    assert sched.table_manager.available_size == 1
    assert not sched.decode_manager.runnable and not sched.prefill_manager.runnable
    segments = [f[2] for f in sched.engine.forwards if f[0] == "prefill"]
    r0 = next((s.start for s in segments if not s.is_encoder), None)
    pc = sched.cache_manager.prefix_cache
    for m in range(P, len(prompt) + 1, P):
        got = pc.match_prefix(prompt[:m]).cached_len
        if got:  # anything reusable lies a full live window past the frontier
            assert r0 is not None and got - P >= g.tombstone_frontier(r0), (m, got)


def test_flag_off_runs_no_cone_code(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("dependency-cone machinery reached with the flag off")

    monkeypatch.setattr(scheduler_mod, "plan_prefill_cone", forbidden)
    monkeypatch.setattr(scheduler_mod, "agree_cone_segment", forbidden)
    monkeypatch.setattr(CacheManager, "_honor_dependency_cone", forbidden)
    sched, _pool = _scheduler(cone=False)
    _submit(sched, _prompt(26_000))
    _run(sched)
    assert all(f[2] is None for f in sched.engine.forwards)
    sched.cache_manager.check_integrity()


@pytest.mark.parametrize("cone_segment, pairs", [(object(), False), (None, True)])
def test_tail_pairing_only_runs_when_the_cone_declines(monkeypatch, cone_segment, pairs):
    """The two exact optimisations share one service but never one forward."""
    sched = object.__new__(Scheduler)
    batch = SimpleNamespace(is_prefill=True, dsv41_cone=None)
    prepared = SimpleNamespace(batch=batch)
    sched.prefill_manager = SimpleNamespace(schedule_next_batch=lambda _budget: batch)
    sched.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None)
    sched.prefill_budget = CHUNK
    sched._dependency_cone = _geometry()
    sched._tail_pairing = object()
    sched._maybe_make_speculative = lambda _batch: None
    sched._plan_dependency_cone = lambda target: setattr(target, "dsv41_cone", cone_segment)
    sched._prepare_batch = lambda target: prepared
    sched._report_prompt_admissions = lambda _batch: None
    calls = []
    sched._maybe_pair_tail = lambda value: calls.append(value) or value
    assert sched._schedule_next_batch() is prepared
    assert calls == ([prepared] if pairs else [])


@pytest.mark.parametrize("peer_flags, rank", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_ranks_agree_on_every_prefill_batch_or_fail_closed(monkeypatch, peer_flags, rank):
    """Every prefill batch (cone or plain) runs one broadcast + one all-reduce; a mismatch
    anywhere raises on this rank before the batch is prepared or launched."""
    sched, _pool = _scheduler(cone=True, rank=rank, size=4)
    calls, planned = [], []
    plan = scheduler_mod.plan_prefill_cone
    monkeypatch.setattr(scheduler_mod, "plan_prefill_cone",
                        lambda batch, g: planned.append(plan(batch, g)) or planned[-1])

    def broadcast(wire, label):
        calls.append(label)
        if rank != 0:
            assert not wire.any()  # a non-root rank contributes nothing
        seg = planned[-1]  # rank 0 planned the same segment from its replicated state
        return seg.encode() if seg is not None else dc.ConeSegment.encode_none()

    def all_reduce(flag, label):
        calls.append(label)
        return flag + peer_flags  # the other ranks' mismatch flags

    sched._cpu_broadcast, sched._cpu_all_reduce_sum = broadcast, all_reduce
    _submit(sched, _prompt(17_490))
    if peer_flags:
        with pytest.raises(dc.DependencyConeMismatch):
            sched._schedule_next_batch()
        assert sched.engine.forwards == []
        assert calls == ["dependency_cone", "dependency_cone_mismatch"]
        return
    _run(sched)
    n_prefill = sum(1 for f in sched.engine.forwards if f[0] == "prefill")
    assert calls == ["dependency_cone", "dependency_cone_mismatch"] * n_prefill
    assert [f[2] for f in sched.engine.forwards if f[0] == "prefill"] == planned
