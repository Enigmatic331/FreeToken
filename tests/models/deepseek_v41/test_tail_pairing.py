"""DeepSeek-V4.1 layer-major tail pairing: model-side host tests (CPU, no CUDA kernels).

The real ``Transformer.prefill_single``/``prefill_tail_pair`` and expert-worker loops run over
stateful stand-in blocks that model the properties exactness depends on: a per-layer KV store
the next chunk reads, shared-attention publications later layers consume, the context batch the
MoE layers consult, and per-layer Engram/MoE collectives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import freetoken.models.deepseek_v41.model as model_mod
from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.models.deepseek_v41.layers import RMSNorm
from freetoken.models.deepseek_v41.model import (
    DeepseekV41ExpertWorkerModel,
    DeepseekV41ForCausalLM,
    SharedAttentionRuntime,
    Transformer,
    _active_segment,
    _PrefillSegment,
)
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.tail_pairing import TailPairMismatch, TailPairPlan

UID = 7
IMAGE = 99
DIM = 8
HC = 4
VOCAB = 128
WINDOW = 3


@pytest.fixture(autouse=True)
def global_ctx():
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=128)
        set_global_ctx(ctx)
    assert ctx._batch is None
    yield ctx
    ctx._batch = None


def _plan(head_start=96, head_len=64, tail_len=24):
    return TailPairPlan(UID, 0, head_start, head_len, tail_len)


def _batches(plan, *, ids=None, paired=True):
    ids = ids if ids is not None else (torch.arange(plan.prompt_end, dtype=torch.int32) % 90) + 1
    head_req = ChunkedReq(input_ids=ids[: plan.tail_start], table_idx=plan.table_idx,
                          cached_len=plan.head_start, output_len=4, uid=plan.uid,
                          sampling_params=SamplingParams(), cache_handle=None)
    tail_req = Req(input_ids=ids[: plan.prompt_end], table_idx=plan.table_idx,
                   cached_len=plan.tail_start, output_len=4, uid=plan.uid,
                   sampling_params=SamplingParams(), cache_handle=None)
    head, tail = Batch(reqs=[head_req], phase="prefill"), Batch(reqs=[tail_req], phase="prefill")
    for batch, req in ((head, head_req), (tail, tail_req)):
        batch.padded_reqs = batch.reqs
        batch.input_ids = ids[req.cached_len : req.device_len].clone()
        batch.positions = torch.arange(req.cached_len, req.device_len, dtype=torch.int32)
    if paired:
        head.tail_pair_role, tail.tail_pair_role = "head", "tail"
        head.tail_pair_plan = tail.tail_pair_plan = plan
    return head, tail


# ----------------------------------------------------------------------------- authority


def _earlier_chunk_kv(layer, position):
    # KV of chunks before the head, identical for both schedules.
    return torch.tensor(float(position % 7) / 10 + layer)


class _FakeBlock(nn.Module):
    """Deterministic stand-in for ``Block.prefill_single`` with the dependencies that matter:

    * attention-like reads of the previous ``WINDOW`` positions' per-layer KV (written by an
      earlier chunk at this same layer) and its own KV writes;
    * index-source layers publish ``shared_attention.topk_rows`` and candidates; consumers read
      them and require this chunk's own row count;
    * the context batch must be this chunk's batch (the MoE layers consult it);
    * Engram rows and the image/token masks feed the output.
    """

    def __init__(self, layer_id, kv, log, *, source):
        super().__init__()
        self.layer_id = layer_id
        self.kv = kv
        self.log = log
        self.source = source

    def prefill_single(self, hidden, incoming, *, start_pos, table_idx, engram_rows=None,
                       token_mask=None, image_mask=None):
        batch = get_global_ctx().batch
        req = batch.reqs[0]
        n = hidden.shape[1]
        assert (req.cached_len, req.extend_len) == (start_pos, n), "context batch is not this chunk"
        runtime = model_mod.shared_attention
        self.log.append((self.layer_id, start_pos, getattr(batch, "tail_pair_role", None),
                         id(runtime)))
        prior = [self.kv.get((self.layer_id, p), _earlier_chunk_kv(self.layer_id, p))
                 for p in range(max(0, start_pos - WINDOW), start_pos)]
        prior = torch.stack(prior).mean() if prior else torch.zeros(())
        if self.source:
            runtime.topk_rows = hidden.float().mean(dim=(2, 3)) + self.layer_id
            runtime.candidates.prefill_block_ids = torch.arange(n, dtype=torch.int32) + start_pos
        else:
            published = runtime.topk_rows
            assert published is not None and published.shape == (1, n), "publication leaked"
            assert torch.equal(runtime.candidates.prefill_block_ids,
                               torch.arange(n, dtype=torch.int32) + start_pos)
        extra = torch.zeros(())
        if engram_rows is not None:
            extra = engram_rows.float().mean() * 1e-3
        mask_term = torch.zeros(())
        if token_mask is not None:
            mask_term = mask_term + token_mask.float().mean() * 1e-2
        if image_mask is not None:
            mask_term = mask_term + image_mask.float().mean() * 1e-2
        for p in range(n):
            self.kv[(self.layer_id, start_pos + p)] = hidden[0, p].float().mean()
        mix = runtime.topk_rows.unsqueeze(-1).unsqueeze(-1)
        out = hidden.float() * 0.5 + prior + 0.25 * mix + extra + mask_term + self.layer_id * 1e-2
        return out.to(hidden.dtype), incoming + 1e-3 * (self.layer_id + 1)


def _transformer(kv, log, n_layers=4, sources=(0, 2)):
    torch.manual_seed(4101)
    model = object.__new__(Transformer)
    nn.Module.__init__(model)
    model.args = SimpleNamespace(hc_mult=HC, dim=DIM, image_token_id=IMAGE)
    model.embed = nn.Embedding(VOCAB, DIM, dtype=torch.bfloat16)
    model.layers = nn.ModuleList(
        [_FakeBlock(i, kv, log, source=i in sources) for i in range(n_layers)]
    )
    model.norm = RMSNorm(DIM, 1e-6)
    model.head = nn.Parameter(torch.randn(VOCAB, DIM), requires_grad=False)
    model._dspark_hidden = None
    return model


def _engram_rows(batch, n_layers=4):
    n = batch.input_ids.numel()
    base = batch.reqs[0].cached_len
    return {1: (torch.arange(n * 24).view(n, 24) + base) % 1000}


def _segment(batch, mm=None):
    return _PrefillSegment(
        batch=batch, input_ids=batch.input_ids.long().view(1, -1),
        start_pos=batch.reqs[0].cached_len, table_idx=batch.reqs[0].table_idx,
        engram_rows=_engram_rows(batch), mm_embeds=mm, runtime=SharedAttentionRuntime(),
    )


def _unpaired(ctx, plan, ids=None, mm=(None, None)):
    kv, log = {}, []
    model = _transformer(kv, log)
    head, tail = _batches(plan, ids=ids, paired=False)
    with ctx.forward_batch(head):
        model.prefill_single(head.input_ids.long().view(1, -1), start_pos=plan.head_start,
                             table_idx=0, engram_rows=_engram_rows(head), mm_embeds=mm[0])
    with ctx.forward_batch(tail):
        logits = model.prefill_single(tail.input_ids.long().view(1, -1), start_pos=plan.tail_start,
                                      table_idx=0, engram_rows=_engram_rows(tail), mm_embeds=mm[1])
    return logits, kv, log


def _paired(ctx, plan, ids=None, mm=(None, None)):
    kv, log = {}, []
    model = _transformer(kv, log)
    head, tail = _batches(plan, ids=ids)
    outer_runtime = model_mod.shared_attention
    with ctx.forward_batch(tail):
        logits = model.prefill_tail_pair(_segment(head, mm[0]), _segment(tail, mm[1]))
        assert get_global_ctx().batch is tail
    assert model_mod.shared_attention is outer_runtime
    return logits, kv, log


@pytest.mark.parametrize("plan", [_plan(), _plan(head_start=0, head_len=40, tail_len=9)])
def test_paired_forward_is_bitwise_the_sequential_forwards(global_ctx, plan):
    ref_logits, ref_kv, ref_log = _unpaired(global_ctx, plan)
    logits, kv, log = _paired(global_ctx, plan)
    assert logits.dtype == torch.float32 and logits.shape == (1, VOCAB)
    assert torch.equal(logits, ref_logits)
    assert kv.keys() == ref_kv.keys()
    assert all(torch.equal(kv[k], ref_kv[k]) for k in kv)
    # Same per-chunk layer sequence; paired interleaves head/tail at every layer.
    per_chunk = lambda entries, start: [e[0] for e in entries if e[1] == start]  # noqa: E731
    for start in (plan.head_start, plan.tail_start):
        assert per_chunk(log, start) == per_chunk(ref_log, start) == list(range(4))
    assert [(e[0], e[2]) for e in log] == [(layer, role) for layer in range(4)
                                           for role in ("head", "tail")]
    # Each chunk used one private runtime throughout, distinct from the module default.
    head_ids = {e[3] for e in log if e[2] == "head"}
    tail_ids = {e[3] for e in log if e[2] == "tail"}
    assert len(head_ids) == len(tail_ids) == 1 and head_ids != tail_ids
    assert id(model_mod.shared_attention) not in head_ids | tail_ids


def test_shared_publications_would_leak_without_isolation(global_ctx, monkeypatch):
    """Negative control: with one shared runtime, the head's consumer layer reads the tail's
    publication -- exactly the failure the per-chunk runtimes prevent."""
    plan = _plan()
    common = SharedAttentionRuntime()
    original = model_mod._active_segment

    def leaky(segment):
        object.__setattr__(segment, "runtime", common)
        return original(segment)

    monkeypatch.setattr(model_mod, "_active_segment", leaky)
    with pytest.raises(AssertionError, match="publication leaked"):
        _paired(global_ctx, plan)


def test_paired_forward_frees_the_previous_forwards_publications_at_entry(global_ctx,
                                                                          monkeypatch):
    """Like ``prefill_single``, the paired forward resets the module-global runtime first.

    The global still holds the preceding unpaired chunk's publications (its ``topk_rows``,
    candidate mask and CSA2-S ids). ``_active_segment`` restores that object after every block,
    so without the reset those tensors stay alive for the whole paired forward: up to 212 MiB
    on the authority at the deepest eligible pair.
    """
    import weakref

    plan = _plan()
    stale = SharedAttentionRuntime()
    monkeypatch.setattr(model_mod, "shared_attention", stale)
    rows = plan.head_len
    stale.topk_rows = torch.zeros(1, rows, 4, dtype=torch.int64)
    stale.candidates.mask = torch.zeros(rows, 16, dtype=torch.bool)
    stale.candidates.prefill_block_mask = torch.zeros(rows, 2, dtype=torch.bool)
    stale.candidates.prefill_packed_mask = torch.zeros(rows, 1, dtype=torch.uint8)
    stale.candidates.prefill_block_ids = torch.zeros(rows, 2, dtype=torch.int32)
    candidates = stale.candidates
    published = [
        weakref.ref(tensor)
        for tensor in (stale.topk_rows, candidates.mask, candidates.prefill_block_mask,
                       candidates.prefill_packed_mask, candidates.prefill_block_ids)
    ]

    def stale_state():
        alive = [ref() is not None for ref in published]
        held = [stale.topk_rows is not None,
                *(value is not None for value in vars(candidates).values())]
        return any(alive), any(held)

    assert stale_state() == (True, True)  # live, and referenced only by the module global
    kv, log = {}, []
    model = _transformer(kv, log)
    seen = []
    embed = model._prefill_inputs

    def recording_embed(*args, **kwargs):
        seen.append(("embed", None, stale_state()))
        return embed(*args, **kwargs)

    model._prefill_inputs = recording_embed
    for block in model.layers:
        def recording_block(*args, _run=block.prefill_single, _layer=block.layer_id, **kwargs):
            seen.append((_layer, get_global_ctx().batch.tail_pair_role, stale_state()))
            return _run(*args, **kwargs)

        block.prefill_single = recording_block
    head, tail = _batches(plan)
    with global_ctx.forward_batch(tail):
        model.prefill_tail_pair(_segment(head), _segment(tail))
    first_block = next(entry for entry in seen if entry[0] != "embed")
    assert first_block == (0, "head", (False, False))
    assert seen[0] == ("embed", None, (False, False))  # freed before the first allocation
    assert all(state == (False, False) for _, _, state in seen)
    assert len(seen) == 2 + 2 * len(model.layers)
    # After the forward the global is the same outer object, and it is clean.
    assert model_mod.shared_attention is stale
    assert stale_state() == (False, False)


def test_multimodal_spans_across_the_pair_boundary(global_ctx):
    plan = _plan()
    ids = (torch.arange(plan.prompt_end, dtype=torch.int32) % 90) + 1
    image_positions = list(range(150, 170)) + list(range(plan.prompt_end - 3, plan.prompt_end))
    ids[torch.tensor(image_positions)] = IMAGE
    head_count = sum(plan.head_start <= p < plan.tail_start for p in image_positions)
    features = torch.randn(len(image_positions), DIM, dtype=torch.bfloat16)
    mm = (features[:head_count], features[head_count:])
    assert 0 < head_count < len(image_positions)
    ref_logits, ref_kv, _ = _unpaired(global_ctx, plan, ids=ids, mm=mm)
    logits, kv, _ = _paired(global_ctx, plan, ids=ids, mm=mm)
    assert torch.equal(logits, ref_logits)
    assert all(torch.equal(kv[k], ref_kv[k]) for k in ref_kv)
    with pytest.raises(ValueError, match="placeholder"):
        _paired(global_ctx, plan, ids=ids, mm=(features, features[head_count:]))


def test_failure_inside_the_pair_restores_context_and_publications(global_ctx):
    plan = _plan()
    kv, log = {}, []
    model = _transformer(kv, log)
    head, tail = _batches(plan)

    def boom(*_args, **_kwargs):
        raise RuntimeError("device fault in the tail")

    model.layers[2].prefill_single = boom
    outer = model_mod.shared_attention
    with global_ctx.forward_batch(tail):
        with pytest.raises(RuntimeError, match="device fault"):
            model.prefill_tail_pair(_segment(head), _segment(tail))
        assert get_global_ctx().batch is tail
    assert model_mod.shared_attention is outer


def test_active_segment_requires_an_active_forward(global_ctx):
    head, _ = _batches(_plan())
    with pytest.raises(AssertionError, match="active forward_batch"):
        with _active_segment(_segment(head)):
            pass


# ----------------------------------------------------------------------------- worker


def _worker(events, n_layers=3, engram_layers=(1,)):
    class Layer:
        def __init__(self, layer_id):
            self.layer_id = layer_id

        def worker_forward(self, hidden_shape, device):
            batch = get_global_ctx().batch
            events.append(("moe", self.layer_id, hidden_shape[0], batch.tail_pair_role))

    class Coordinator:
        execution = SimpleNamespace(participates_in_engram=True, phase_aware=True,
                                    participates_in_prefill=True)

        def worker_lookup(self, layer_id, **kwargs):
            events.append(("engram", layer_id, kwargs["num_tokens"],
                           get_global_ctx().batch.tail_pair_role, kwargs["is_prefill"]))

    worker = object.__new__(DeepseekV41ExpertWorkerModel)
    worker.args = SimpleNamespace(n_layers=n_layers, dim=DIM, engram_layer_ids=engram_layers,
                                  engram_hashes_per_token=24)
    worker.layers = [Layer(i) for i in range(n_layers)]
    worker.engram_coordinator = Coordinator()
    worker.joined = 0

    def join():
        worker.joined += 1

    worker._join_background_admission = join
    return worker


def test_worker_enters_each_chunks_collectives_in_the_authority_order(global_ctx):
    events = []
    worker = _worker(events)
    head, tail = _batches(_plan())
    with global_ctx.forward_batch(tail):
        out = worker.forward_tail_pair(head, tail)
        assert get_global_ctx().batch is tail
    assert out.shape == (1, 1)
    assert events == [
        ("moe", 0, 64, "head"), ("moe", 0, 24, "tail"),
        ("engram", 1, 64, "head", True), ("moe", 1, 64, "head"),
        ("engram", 1, 24, "tail", True), ("moe", 1, 24, "tail"),
        ("moe", 2, 64, "head"), ("moe", 2, 24, "tail"),
    ]
    assert worker.joined == 1


def test_worker_rejoins_side_streams_when_a_collective_fails(global_ctx):
    events = []
    worker = _worker(events)

    def broken(layer_id, **kwargs):
        raise RuntimeError("NCCL error")

    worker.engram_coordinator.worker_lookup = broken
    head, tail = _batches(_plan())
    with global_ctx.forward_batch(tail):
        with pytest.raises(RuntimeError, match="NCCL"):
            worker.forward_tail_pair(head, tail)
        assert get_global_ctx().batch is tail
    assert worker.joined == 1


def test_inactive_prefill_worker_skips_the_pair(global_ctx):
    events = []
    worker = _worker(events)
    worker.engram_coordinator.execution = SimpleNamespace(
        participates_in_engram=False, phase_aware=True, participates_in_prefill=False)
    head, tail = _batches(_plan())
    with global_ctx.forward_batch(tail):
        assert worker.forward_tail_pair(head, tail).shape == (1, 1)
    assert events == []


# ----------------------------------------------------------------------------- adapter


class _CapturingHasher:
    def __init__(self):
        self.calls = []

    def row_ids(self, input_ids, positions, cu, history, **_):
        self.calls.append((positions[0].item(), input_ids.numel(), history.tolist()))
        return torch.zeros(input_ids.numel(), 2, 24, dtype=torch.int64)


def _adapter(*, worker=False, subgroup=False):
    model = object.__new__(DeepseekV41ForCausalLM)
    model._execution = SimpleNamespace(is_expert_worker=worker, uses_prefill_subgroup=subgroup)
    model._args = SimpleNamespace(engram_max_ngram_size=4, engram_pad_id=2,
                                  engram_layer_ids=(1, 14))
    model._engram_hasher = _CapturingHasher()
    model._phase_barrier_group = None
    model._dspark_target_features = "stale"
    calls = []
    model._model = SimpleNamespace(
        prefill_tail_pair=lambda head, tail: calls.append(("authority", head, tail))
        or torch.ones(1, VOCAB),
        forward_tail_pair=lambda head, tail: calls.append(("worker", head, tail))
        or torch.zeros(1, 1),
    )
    model.prepare_for_runtime = lambda: calls.append(("prepare",))
    return model, calls


def test_adapter_builds_one_segment_per_chunk_with_its_own_engram_history(global_ctx):
    plan = _plan()
    model, calls = _adapter()
    head, tail = _batches(plan)
    with global_ctx.forward_batch(tail):
        logits = model.forward_tail_pair(head)
    assert torch.equal(logits, torch.ones(1, VOCAB))
    assert calls[0] == ("prepare",)
    _, head_seg, tail_seg = calls[1]
    assert (head_seg.batch, head_seg.start_pos, head_seg.input_ids.shape) == (head, 96, (1, 64))
    assert (tail_seg.batch, tail_seg.start_pos, tail_seg.input_ids.shape) == (tail, 160, (1, 24))
    assert head_seg.runtime is not tail_seg.runtime
    ids = tail.reqs[0].input_ids
    # Each chunk hashes with the three tokens immediately before its own start.
    assert model._engram_hasher.calls == [
        (96, 64, [ids[93:96].tolist()]),
        (160, 24, [ids[157:160].tolist()]),
    ]
    assert model._dspark_target_features is None


def test_adapter_routes_workers_to_the_paired_worker_loop(global_ctx):
    model, calls = _adapter(worker=True)
    head, tail = _batches(_plan())
    with global_ctx.forward_batch(tail):
        out = model.forward_tail_pair(head)
    assert out.shape == (1, 1)
    assert calls == [("worker", head, tail)]


def test_adapter_fails_closed_before_any_collective_on_bad_metadata(global_ctx):
    model, calls = _adapter()
    head, tail = _batches(_plan())
    tail.input_ids = tail.input_ids[:-1]
    with global_ctx.forward_batch(tail):
        with pytest.raises(TailPairMismatch):
            model.forward_tail_pair(head)
    assert calls == []
    worker, worker_calls = _adapter(worker=True)
    head, tail = _batches(_plan())
    head.tail_pair_plan = _plan(tail_len=23)
    with global_ctx.forward_batch(tail):
        with pytest.raises(TailPairMismatch):
            worker.forward_tail_pair(head)
    assert worker_calls == []


def test_phase_barrier_is_shared_by_paired_and_unpaired_forwards(global_ctx, monkeypatch):
    barriers = []
    monkeypatch.setattr(model_mod.torch.distributed, "barrier",
                        lambda group=None: barriers.append(group))
    model, _ = _adapter(subgroup=True)
    model._phase_barrier_group = "gloo-phase"
    head, tail = _batches(_plan())
    with global_ctx.forward_batch(tail):
        model.forward_tail_pair(head)
    assert barriers == ["gloo-phase"]
    model._phase_barrier_group = None
    with global_ctx.forward_batch(tail):
        with pytest.raises(RuntimeError, match="phase-aware prefill barrier"):
            model.forward_tail_pair(head)
