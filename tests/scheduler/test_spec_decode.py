"""MTP speculative-decoding driver (scheduler/spec.py) -- CPU, no model.

Covers the pieces real serving exercises but a unit harness can pin exactly:
the accept / reject / finish state machine (counters, emits, rewind), the
reject-path page cleanup against the REAL DSV4 cache stack (conservation via
check_integrity), prompt hc ingestion across chunk boundaries (the carry-row
alignment), spec-batch write/row plumbing, and exit-to-decode / abort.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.sample import Sampler
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.scheduler import _make_write_tuple
from freetoken.scheduler.spec import SpecDecodeManager

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- fakes
class FakeModel:
    def __init__(self):
        self.hc = None

    def take_spec_hc(self):
        h, self.hc = self.hc, None
        return h


class FakeSidecar:
    """Records extend() calls as (n_states, tokens, pos_before); returns queued drafts."""

    def __init__(self, drafts=()):
        self.drafts = list(drafts)
        self.pos = 0
        self.calls = []
        self.device = DEVICE

    def reset(self):
        self.pos = 0

    def extend(self, hc, tokens):
        toks = [int(t) for t in tokens]
        assert hc.shape[0] == len(toks)
        self.calls.append((hc.shape[0], toks, self.pos))
        self.pos += hc.shape[0]
        return self.drafts.pop(0) if self.drafts else 777


class FakePrefillManager:
    def __init__(self):
        self.pending = []

    @property
    def runnable(self):
        return bool(self.pending)


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(tp_info=SimpleNamespace(rank=0, size=1))
        self.engine = SimpleNamespace(model=FakeModel())
        self.decode_manager = DecodeManager(page_size=1)
        self.prefill_manager = FakePrefillManager()
        self.token_pool = torch.zeros(4, 256, dtype=torch.int32)
        self.eos_token_ids = frozenset({2})
        self.freed = []
        self.free_tail_calls = []
        self.cache_manager = SimpleNamespace(
            free_spec_tail=lambda req, keep_len: self.free_tail_calls.append(keep_len)
        )

    def _match_stop_str(self, req):
        return None

    def _free_req_resources(self, req):
        self.freed.append(req)
        req.table_idx = -1


def _mgr(drafts=()):
    sched = FakeScheduler()
    return SpecDecodeManager(sched, FakeSidecar(drafts)), sched


def _boundary_req(committed=8, budget=16, uid=1):
    """A request at a spec cycle boundary: ids == device_len tokens, one pending input
    (extend_len == 1) -- indistinguishable from a normal mid-decode request."""
    ids = torch.arange(100, 100 + committed, dtype=torch.int32)
    req = Req(
        input_ids=ids, table_idx=0, cached_len=committed - 1, output_len=budget,
        uid=uid, sampling_params=SamplingParams(), cache_handle=None,
    )
    req.spec_active = True
    return req


def _adopt(mgr, req, draft):
    mgr.req = req
    mgr.pending_draft = draft


def _run_cycle(mgr, sched, samples):
    """schedule -> (engine: complete_one) -> drain, like normal_loop without a GPU."""
    batch = mgr.schedule_next_batch()
    assert batch is not None and batch.spec_verify
    T = batch.rows_per_req[0]
    for r in batch.reqs:
        r.complete_one()
    sched.engine.model.hc = torch.zeros(T, 2, 3)
    reply = mgr.process_drained(batch, torch.tensor(samples[:T], dtype=torch.int32))
    return batch, reply


# ------------------------------------------------------------------- state machine
def test_accept_cycle_emits_two_and_stays_at_boundary():
    mgr, sched = _mgr(drafts=[61])
    req = _boundary_req(committed=8)
    _adopt(mgr, req, draft=50)

    batch, reply = _run_cycle(mgr, sched, [50, 60])  # s1 == draft -> accept
    assert batch.rows_per_req == [2]
    assert [m.next_token for m in reply] == [50, 60]
    assert not any(m.finished for m in reply)
    # boundary invariant: ids == device_len, extend 1, KV through device_len-1
    assert req.device_len == 10 and req.cached_len == 9
    assert req.input_ids.tolist()[-2:] == [50, 60]
    # sidecar got BOTH committed pairs, contiguously
    assert mgr.sidecar.calls[-1] == (2, [50, 60], 0)
    assert mgr.pending_draft == 61
    assert mgr.cycles == 1 and mgr.accepts == 1
    # the staged verify chunk reached the token pool
    assert sched.token_pool[0, 7].item() == 107 and sched.token_pool[0, 8].item() == 50


def test_reject_cycle_rewinds_and_still_drafts():
    mgr, sched = _mgr(drafts=[62])
    req = _boundary_req(committed=8)
    _adopt(mgr, req, draft=50)

    _, reply = _run_cycle(mgr, sched, [55, 60])  # s1 != draft -> reject
    assert [m.next_token for m in reply] == [55]
    # rewound: draft position re-extends next cycle
    assert req.device_len == 9 and req.cached_len == 8
    assert req.input_ids.tolist()[-1] == 55
    # the stale tail page was offered for cleanup BEFORE the rewind
    assert sched.free_tail_calls == [8]
    # only the committed pair fed; a NEW draft exists (pair (state@7, 55) predicts pos 9)
    assert mgr.sidecar.calls[-1] == (1, [55], 0)
    assert mgr.pending_draft == 62
    assert mgr.accepts == 0 and mgr.cycles == 1

    # next cycle re-extends the rewound position with the fresh draft
    batch2, _ = _run_cycle(mgr, sched, [62, 70])
    assert batch2.rows_per_req == [2]
    assert req.input_ids.tolist()[-2:] == [62, 70]


def test_eos_on_accepted_draft_drops_bonus_token():
    mgr, sched = _mgr()
    req = _boundary_req(committed=8)
    _adopt(mgr, req, draft=2)  # draft IS the eos token

    _, reply = _run_cycle(mgr, sched, [2, 60])  # accepted and eos
    assert [(m.next_token, m.finished) for m in reply] == [(2, True)]
    assert reply[0].finish_reason == "stop"
    assert req.device_len == 9 and len(req.input_ids) == 9  # s2 never committed
    assert sched.freed == [req] and mgr.req is None


def test_eos_on_bonus_token_finishes_after_two():
    mgr, sched = _mgr()
    req = _boundary_req(committed=8)
    _adopt(mgr, req, draft=50)

    _, reply = _run_cycle(mgr, sched, [50, 2])
    assert [(m.next_token, m.finished) for m in reply] == [(50, False), (2, True)]
    assert sched.freed == [req] and mgr.req is None


def test_eos_on_rejection_sample_finishes():
    mgr, sched = _mgr()
    req = _boundary_req(committed=8)
    _adopt(mgr, req, draft=50)

    _, reply = _run_cycle(mgr, sched, [2, 60])  # reject with eos truth
    assert [(m.next_token, m.finished) for m in reply] == [(2, True)]
    assert sched.free_tail_calls == [8]  # cleanup still ran
    assert sched.freed == [req]


def test_budget_edge_runs_draftless_final_step():
    mgr, sched = _mgr()
    # remain_len == 1: no room for the draft plus the accept-path bonus token
    req = _boundary_req(committed=8, budget=1)
    _adopt(mgr, req, draft=50)

    batch, reply = _run_cycle(mgr, sched, [90])
    assert batch.rows_per_req == [1]  # draft was discarded
    assert [(m.next_token, m.finished) for m in reply] == [(90, True)]
    assert reply[0].finish_reason == "length"
    assert sched.freed == [req]


def test_prefill_arrival_exits_to_decode_at_boundary():
    mgr, sched = _mgr()
    req = _boundary_req(committed=8)
    _adopt(mgr, req, draft=50)

    sched.prefill_manager.pending.append(object())
    mgr.before_schedule()
    assert mgr.req is None and mgr.pending_draft is None
    assert req.spec_active is False and req.can_decode
    assert sched.decode_manager.running_reqs == {req}
    assert mgr.schedule_next_batch() is None


def test_abort_returns_request_and_clears_state():
    mgr, sched = _mgr()
    req = _boundary_req(committed=8, uid=42)
    _adopt(mgr, req, draft=50)

    assert mgr.abort_req(7) is None
    got = mgr.abort_req(42)
    assert got is req and mgr.req is None and req.spec_active is False


# --------------------------------------------------------------- prompt ingestion
def _prefill_batch(req):
    b = Batch(reqs=[req], phase="prefill")
    b.padded_reqs = [req]
    return b


def test_single_chunk_prompt_adoption_alignment():
    mgr, sched = _mgr(drafts=[71])
    prompt = torch.arange(100, 105, dtype=torch.int32)  # 5 tokens
    req = Req(input_ids=prompt, table_idx=0, cached_len=0, output_len=16, uid=9,
              sampling_params=SamplingParams(), cache_handle=SimpleNamespace(cached_len=0))
    # engine: forward + complete_one; generic drain: append s0
    req.complete_one()
    req.append_host(torch.tensor([200], dtype=torch.int32))
    sched.decode_manager.running_reqs = {req}
    sched.engine.model.hc = torch.zeros(5, 2, 3)

    mgr.observe_batch(_prefill_batch(req), torch.tensor([200]), set())
    assert mgr.req is req and req.spec_active
    assert mgr.pending_draft == 71
    # pairs: (state@i -> prompt[i+1]) for i<4, (state@4 -> s0), from ring pos 0
    assert mgr.sidecar.calls == [(5, [101, 102, 103, 104, 200], 0)]
    assert sched.decode_manager.running_reqs == set()


def test_out_of_scope_prompts_are_not_adopted():
    prompt = torch.arange(100, 105, dtype=torch.int32)

    def _final_req(**kw):
        d = dict(input_ids=prompt, table_idx=0, cached_len=0, output_len=16, uid=9,
                 sampling_params=SamplingParams(),
                 cache_handle=SimpleNamespace(cached_len=0))
        d.update(kw)
        r = Req(**d)
        r.complete_one()
        r.append_host(torch.tensor([200], dtype=torch.int32))
        return r

    # chunked-prompt continuation chunk (ChunkedReq): no bulk-ingest path
    mgr, sched = _mgr()
    chunk = ChunkedReq(input_ids=prompt[:3], table_idx=0, cached_len=0, output_len=16,
                       uid=9, sampling_params=SamplingParams(),
                       cache_handle=SimpleNamespace(cached_len=0))
    chunk.complete_one()
    sched.engine.model.hc = torch.zeros(3, 2, 3)
    mgr.observe_batch(_prefill_batch(chunk), torch.tensor([0]), set())
    assert mgr.req is None and mgr.sidecar.calls == []

    # prefix-cache hit: the cached positions have no hc states
    mgr, sched = _mgr()
    req = _final_req(cached_len=2, cache_handle=SimpleNamespace(cached_len=2))
    sched.decode_manager.running_reqs = {req}
    sched.engine.model.hc = torch.zeros(3, 2, 3)  # only 3 fresh positions were forwarded
    mgr.observe_batch(_prefill_batch(req), torch.tensor([200]), set())
    assert mgr.req is None and mgr.sidecar.calls == []

    # prompt above the one-shot ingest bound
    mgr, sched = _mgr()
    mgr.max_prompt = 4
    req = _final_req()
    sched.decode_manager.running_reqs = {req}
    sched.engine.model.hc = torch.zeros(5, 2, 3)
    mgr.observe_batch(_prefill_batch(req), torch.tensor([200]), set())
    assert mgr.req is None and mgr.sidecar.calls == []

    # a second live request
    mgr, sched = _mgr()
    req = _final_req()
    sched.decode_manager.running_reqs = {req, object()}
    sched.engine.model.hc = torch.zeros(5, 2, 3)
    mgr.observe_batch(_prefill_batch(req), torch.tensor([200]), set())
    assert mgr.req is None and mgr.sidecar.calls == []


def test_sidecar_failure_latches_disabled_and_keeps_serving():
    mgr, sched = _mgr()
    req = _boundary_req(committed=8)
    _adopt(mgr, req, draft=50)

    def boom(hc, tokens):
        raise RuntimeError("sidecar OOM")

    mgr.sidecar.extend = boom
    _, reply = _run_cycle(mgr, sched, [50, 60])
    # the cycle's tokens still shipped; speculation latched off; request back on decode
    assert [m.next_token for m in reply] == [50, 60]
    assert mgr.disabled and mgr.req is None
    assert sched.decode_manager.running_reqs == {req} and req.can_decode


# ------------------------------------------------------------------ spec plumbing
def test_spec_write_tuple_targets_row_positions():
    req = _boundary_req(committed=8, budget=16)
    req.append_host(req.input_ids.new_tensor([50]))
    req.device_len += 1  # staged draft, extend_len == 2
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    batch.spec_verify = True
    batch.rows_per_req = [2]
    mapping, write = _make_write_tuple(batch, DEVICE)
    assert mapping.tolist() == [0, 0]
    # row t's sample lands at position cached_len + 1 + t
    assert write.tolist() == [8, 9]


def test_spec_write_tuple_stays_inside_budget():
    # Final draft-less step: the write position is device_len, which the scheduling guard
    # keeps < max_device_len (remain_len >= 1 while schedulable) -- the -1 clamp in
    # _make_write_tuple is defense-in-depth only.
    req = _boundary_req(committed=8, budget=1)  # max_device_len == 9
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    batch.spec_verify = True
    batch.rows_per_req = [1]
    _, write = _make_write_tuple(batch, DEVICE)
    assert write.tolist() == [8]  # in range: 8 < max_device_len == 9


def test_sampler_expands_params_per_logit_row():
    req = _boundary_req(committed=8)
    req.sampling_params = SamplingParams(temperature=0.7)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    batch.rows_per_req = [2]
    args = Sampler(device=DEVICE, vocab_size=100).prepare(batch)
    assert args.temperatures is not None and args.temperatures.shape[0] == 2

    req.sampling_params = SamplingParams(temperature=0.0)
    args = Sampler(device=DEVICE, vocab_size=100).prepare(batch)
    assert args.temperatures is None  # greedy stays greedy (argmax handles [T, vocab])


def test_num_logit_rows_property():
    req = _boundary_req(committed=8)
    batch = Batch(reqs=[req], phase="prefill")
    assert batch.num_logit_rows == 1
    batch.rows_per_req = [2]
    assert batch.num_logit_rows == 2


# ----------------------------------------------- reject cleanup on the real DSV4 stack
def test_free_spec_tail_conserves_pages_across_boundary_reject():
    """A reject whose verify extend crossed a 128-page boundary must return the freshly
    allocated page (full + window tiers) before the re-extend re-allocates it; conservation
    is the stack's own check_integrity after the request finishes."""
    from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    from freetoken.models.deepseek_v4.args import DeepseekV4Args
    from freetoken.scheduler.cache import CacheManager

    P, MRR, num_pages = 128, 4, 32
    args = DeepseekV4Args(
        n_layers=8, compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0), max_seq_len=8192,
        head_dim=512, index_head_dim=128, window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=num_pages + 1, args=args, swa_ratio=1.0, P=P)
    pool = DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE, P=P, n_scratch=MRR + 1)
    pool._init_paged_state(MRR, True)
    pt = torch.zeros(MRR + 1, args.max_seq_len, dtype=torch.int32)
    pt[MRR].fill_(num_pages * P)
    pool.full_loc_map = pt
    cm = CacheManager(num_pages=num_pages, page_size=P, page_table=pt, type="swa_radix",
                      swa_pool=pool, sliding_window_size=P)

    ids = (torch.arange(200) % 97).to(torch.int32)
    req = Req(input_ids=ids, table_idx=0, cached_len=0, output_len=256, uid=1,
              sampling_params=SamplingParams(), cache_handle=None)
    req.input_len = len(ids)
    h = cm.match_req(req).cuda_handle
    req.cache_handle = h
    cm.lock(h)
    cm.allocate_paged([req])
    req.complete_one()

    # decode to the cycle boundary just before a page edge: pending token at position 255
    while req.device_len < 256:
        req.append_host(torch.tensor([9], dtype=torch.int32))
        cm.allocate_paged([req])
        req.complete_one()
    assert req.cached_len == 255 and req.device_len == 256

    free_before = len(cm.free_slots)
    # verify extend [tok@255, draft@256] allocates the page for position 256
    req.append_host(torch.tensor([50], dtype=torch.int32))
    req.device_len += 1
    cm.allocate_paged([req])
    req.complete_one()
    assert len(cm.free_slots) == free_before - 1

    # reject: cleanup then rewind; the fresh page must come back
    cm.free_spec_tail(req, keep_len=256)
    req.spec_rewind(256, 55)
    assert len(cm.free_slots) == free_before

    # re-extend the same boundary with the next draft, accept both tokens
    req.append_host(torch.tensor([60], dtype=torch.int32))
    req.device_len += 1
    cm.allocate_paged([req])
    req.complete_one()
    req.append_host(torch.tensor([70], dtype=torch.int32))
    assert req.cached_len == 257 and len(req.input_ids) == 258

    # a mid-page reject frees nothing (partial page is reused in place)
    req.append_host(torch.tensor([80], dtype=torch.int32))
    req.device_len += 1
    cm.allocate_paged([req])
    req.complete_one()
    free_mid = len(cm.free_slots)
    cm.free_spec_tail(req, keep_len=258)
    req.spec_rewind(259, 85)
    assert len(cm.free_slots) == free_mid

    cm.cache_req(req, finished=True)
    cm.check_integrity()


# ------------------------------------------------------------------ tp>1 lockstep
class FakeCpuGroup:
    """Stands in for the scheduler's gloo tp_cpu_group: rank 0's value (queued here)
    lands in every rank's tensor."""

    def __init__(self, values):
        self.values = list(values)

    def broadcast(self, t, root=0):
        t[0] = self.values.pop(0)
        return SimpleNamespace(wait=lambda: None)


def _follower_mgr(shared_values):
    sched = FakeScheduler()
    sched.config = SimpleNamespace(tp_info=SimpleNamespace(rank=1, size=2))
    sched.engine.tp_cpu_group = FakeCpuGroup(shared_values)
    return SpecDecodeManager(sched, None), sched


def test_follower_receives_draft_without_sidecar():
    mgr, _ = _follower_mgr([123])
    assert not mgr.is_drafter and mgr.sidecar is None
    hc = torch.zeros(1, 1, 4)
    assert mgr._draft_step(hc, [5]) == 123
    assert not mgr.disabled


def test_follower_latches_on_disable_sentinel():
    from freetoken.scheduler.spec import _SPEC_DISABLED

    mgr, _ = _follower_mgr([_SPEC_DISABLED])
    assert mgr._draft_step(torch.zeros(1, 1, 4), [5]) is None
    assert mgr.disabled


def test_drafter_broadcasts_its_draft():
    sched = FakeScheduler()
    sched.config = SimpleNamespace(tp_info=SimpleNamespace(rank=0, size=2))
    captured = []

    class CaptureGroup:
        def broadcast(self, t, root=0):
            captured.append(int(t[0]))
            return SimpleNamespace(wait=lambda: None)

    sched.engine.tp_cpu_group = CaptureGroup()
    mgr = SpecDecodeManager(sched, FakeSidecar([321]))
    assert mgr._draft_step(torch.zeros(1, 1, 4), [5]) == 321
    assert captured == [321]
