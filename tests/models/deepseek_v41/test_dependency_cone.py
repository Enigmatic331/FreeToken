"""DeepSeek-V4.1 dependency-cone prefill: model-side host tests (CPU, no CUDA kernels).

The real ``Transformer.prefill_single``, expert-worker loop, ``Attention`` window code and
adapter routing run over stand-ins for everything that needs CUDA. The checks cover the
properties exactness depends on:

* an encoder-only chunk runs the encoder layers unchanged, then the source layer's
  publication, and nothing past it on either side of the EP transport;
* a decoder chunk clamps windows only above the source layer;
* with the flag off, every call is the unchanged one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import freetoken.models.deepseek_v41.model as model_mod
from freetoken.attention.dsv4_sparse import DSV4SparseAttnBackend
from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.layers import RMSNorm
from freetoken.models.deepseek_v41.model import (
    Attention,
    DeepseekV41ExpertWorkerModel,
    DeepseekV41ForCausalLM,
    Transformer,
)
from freetoken.models.deepseek_v41.moe import RoutedExperts
from freetoken.scheduler.dependency_cone import (
    DECODER,
    ENCODER,
    ConeSegment,
    DependencyConeMismatch,
)
from freetoken.scheduler.prefill import ChunkedReq

N_LAYERS, SOURCE = 6, 3
DIM, HC, VOCAB = 8, 4, 32
UID = 3


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


def _args(**kw) -> DeepseekV41Args:
    ratios = (0,) + (2,) * (SOURCE - 1) + (1,) * (N_LAYERS - SOURCE)
    base = dict(
        max_batch_size=1, max_seq_len=4096, vocab_size=VOCAB, dim=64, moe_inter_dim=32,
        n_layers=N_LAYERS, n_heads=2, n_routed_experts=4, n_activated_experts=2,
        q_lora_rank=64, head_dim=64, rope_head_dim=16, o_groups=1, o_lora_rank=32,
        window_size=8, compress_ratios=ratios, kv_source_layers=(1, SOURCE),
        index_source_layers=(1, SOURCE, SOURCE + 1), index_n_heads=2, index_head_dim=32,
        index_topk=2, candidate_source_layer=SOURCE, candidate_topk_blocks=1,
        candidate_block_size=2, engram_layer_ids=(1,), engram_num_embeddings=(1000,),
    )
    base.update(kw)
    return DeepseekV41Args(**base)


def _encoder(start=0, end=64):
    return ConeSegment(UID, start, end, ENCODER, -1, SOURCE, N_LAYERS)


def _decoder(start=64, end=96, r0=64):
    return ConeSegment(UID, start, end, DECODER, r0, SOURCE, N_LAYERS)


# ----------------------------------------------------------------------------- authority loop
class _Block(nn.Module):
    def __init__(self, layer_id, log):
        super().__init__()
        self.layer_id, self.log = layer_id, log

    def prefill_single(self, hidden, incoming, *, start_pos, table_idx, engram_rows=None,
                       token_mask=None, image_mask=None, **extra):
        self.log.append(("layer", self.layer_id, start_pos, dict(extra)))
        return hidden + 1, incoming

    def publish_prefill_source(self, hidden, incoming, *, start_pos, table_idx):
        self.log.append(("publish", self.layer_id, start_pos, float(hidden.float().mean())))


def _transformer(log):
    torch.manual_seed(7)
    model = object.__new__(Transformer)
    nn.Module.__init__(model)
    model.args = SimpleNamespace(hc_mult=HC, dim=DIM, image_token_id=VOCAB - 1,
                                 dspark_target_layer_ids=(N_LAYERS - 1,))
    model.embed = nn.Embedding(VOCAB, DIM, dtype=torch.bfloat16)
    model.embed.weight.requires_grad_(False)
    model.layers = nn.ModuleList([_Block(i, log) for i in range(N_LAYERS)])
    model.norm = RMSNorm(DIM, 1e-6)
    model.head = nn.Parameter(torch.randn(VOCAB, DIM), requires_grad=False)
    model._dspark_hidden = None
    return model


def _prefill(model, start, n, **kw):
    ids = (torch.arange(n) % (VOCAB - 2)).view(1, -1)
    return model.prefill_single(ids, start_pos=start, table_idx=0, **kw)


def test_encoder_chunk_runs_the_encoder_then_only_the_source_publication():
    log = []
    logits = _prefill(_transformer(log), 0, 64, cone=_encoder())
    assert [entry[:2] for entry in log] == (
        [("layer", i) for i in range(SOURCE)] + [("publish", SOURCE)]
    )
    assert all(entry[3] == {} for entry in log if entry[0] == "layer")  # plain encoder calls
    # The publication sees the source layer's input: the encoder's output (three +1 layers).
    assert log[-1][3] > 0
    assert logits.dtype == torch.float32 and logits.shape == (1, VOCAB)
    assert not logits.any()


def test_decoder_chunk_clamps_windows_only_above_the_source_layer():
    log = []
    logits = _prefill(_transformer(log), 64, 32, cone=_decoder())
    assert [entry[:2] for entry in log] == [("layer", i) for i in range(N_LAYERS)]
    for _, layer, start, extra in log:
        assert start == 64
        assert extra == ({"window_floor": 64} if layer > SOURCE else {})
    assert logits.shape == (1, VOCAB) and logits.any()


def test_flag_off_calls_are_unchanged():
    log = []
    plain = _prefill(_transformer(log), 64, 32)
    assert [entry[:2] for entry in log] == [("layer", i) for i in range(N_LAYERS)]
    assert all(entry[3] == {} for entry in log)
    # The stand-in blocks ignore the clamp, so this shows the loop adds nothing else.
    again = _prefill(_transformer([]), 64, 32, cone=_decoder())
    assert torch.equal(plain, again)


@pytest.mark.parametrize("kw", [dict(full_logits=True), dict(capture_dspark=True)])
def test_cone_refuses_full_logits_and_dspark_taps(kw):
    with pytest.raises(RuntimeError, match="dependency-cone"):
        _prefill(_transformer([]), 0, 64, cone=_encoder(), **kw)


# ----------------------------------------------------------------------------- expert worker
def _worker(events):
    class Layer:
        def __init__(self, layer_id):
            self.layer_id = layer_id

        def worker_forward(self, hidden_shape, device):
            events.append(("moe", self.layer_id, hidden_shape))

    class Coordinator:
        execution = SimpleNamespace(participates_in_engram=True, phase_aware=False)

        def worker_lookup(self, layer_id, **kwargs):
            events.append(("engram", layer_id, kwargs["num_tokens"]))

    worker = object.__new__(DeepseekV41ExpertWorkerModel)
    worker.args = SimpleNamespace(n_layers=N_LAYERS, dim=DIM, engram_layer_ids=(1,),
                                  engram_hashes_per_token=24)
    worker.layers = [Layer(i) for i in range(N_LAYERS)]
    worker.engram_coordinator = Coordinator()
    return worker


def test_worker_enters_only_the_encoder_collectives_of_an_encoder_chunk(monkeypatch):
    monkeypatch.setattr(model_mod, "get_global_ctx",
                        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=True)))
    ids = torch.zeros(64, dtype=torch.int64)
    events = []
    _worker(events).forward(ids, cone=_encoder())
    assert events == [("moe", 0, (64, DIM)), ("engram", 1, 64), ("moe", 1, (64, DIM)),
                      ("moe", 2, (64, DIM))]
    for cone in (None, _decoder(0, 64, 0)):
        events = []
        _worker(events).forward(ids, **({} if cone is None else {"cone": cone}))
        assert [e for e in events if e[0] == "moe"] == [("moe", i, (64, DIM))
                                                        for i in range(N_LAYERS)]


def test_encoder_chunk_does_not_prefetch_the_source_expert_bank(global_ctx, monkeypatch):
    calls = []
    monkeypatch.setenv("FREETOKEN_DSV41_DEPENDENCY_CONE", "1")

    class Cache:
        def begin_prefill(self):
            calls.append(("begin",))

        def prefetch_prefill_layer(self, layer):
            calls.append(("prefetch", layer))

        def wait_prefill_layer(self, layer):
            calls.append(("wait", layer))
            return (layer,)

    expert = object.__new__(RoutedExperts)
    expert.layer_id = SOURCE - 1
    expert.prefill_early_refill = None
    monkeypatch.setattr(model_mod.prefill_trace, "active", lambda: False)

    batch = Batch(reqs=[_req(0, 64, final=False)], phase="prefill")
    batch.dsv41_cone = _encoder()
    with global_ctx.forward_batch(batch):
        assert expert._wait_prefill_overlap(Cache()) == (SOURCE - 1,)
    assert calls == [("prefetch", SOURCE - 1), ("wait", SOURCE - 1)]

    calls.clear()
    batch.dsv41_cone = _decoder(0, 64, 0)
    with global_ctx.forward_batch(batch):
        assert expert._wait_prefill_overlap(Cache()) == (SOURCE - 1,)
    assert calls == [
        ("prefetch", SOURCE - 1),
        ("prefetch", SOURCE),
        ("wait", SOURCE - 1),
    ]


# ----------------------------------------------------------------------------- attention window
class _Backend:
    """Window addressing of the real DSV4 backend over an identity-like slot map."""

    win_cols_to_global = DSV4SparseAttnBackend.win_cols_to_global

    def __init__(self):
        self.stored, self.attended = [], []

    def window_slots_of(self, ti, lo, hi):
        return 1000 + torch.arange(lo, hi)

    def store_window(self, kv, layer_id, slots):
        self.stored.append((layer_id, kv.clone(), slots.clone()))

    def attend(self, q, layer_id, topk, n_window, sink, scale, cmp_counts=None,
               has_compression=True):
        self.attended.append((topk.clone(), n_window))
        return torch.zeros_like(q)


WINDOW, HEADS, HEAD_DIM, ROPE = 8, 2, 16, 4


def _attention(global_ctx, *, owns_kv=False, published=None):
    backend = _Backend()
    global_ctx.attn_backend = backend
    attn = object.__new__(Attention)
    nn.Module.__init__(attn)
    attn.layer_id, attn.window_size, attn.rope_head_dim = 5, WINDOW, ROPE
    attn.plan = SimpleNamespace(owns_kv=owns_kv, compress_ratio=1, owns_index=False)
    angles = torch.arange(512, dtype=torch.float64)[:, None] * torch.ones(ROPE // 2)
    attn.freqs_cis = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)
    attn.attn_sink = torch.zeros(HEADS)
    attn.softmax_scale = 1.0

    def project_q(x, freqs):
        return x, torch.ones(1, x.shape[1], HEADS, HEAD_DIM)

    def project_window_kv(x, freqs):
        return torch.arange(x.shape[1] * HEAD_DIM, dtype=torch.float32).view(1, -1, HEAD_DIM)

    def publish(x, start_pos, table_idx, slots):
        published.append((start_pos, table_idx, slots.clone()))

    attn._project_q = project_q
    attn._project_window_kv = project_window_kv
    attn._index_prefill = lambda x, q_lora, s, t: torch.full((1, x.shape[1], 2), 7)
    attn._project_output = lambda output: output
    attn._publish_source_prefill = publish
    return attn, backend


def _window_positions(start, n, floor):
    rows = []
    for q in range(start, start + n):
        lo = max(0, q - WINDOW + 1, floor)
        cols = [1000 + p for p in range(lo, q + 1)]
        rows.append(cols + [-1] * (WINDOW - len(cols)))
    return torch.tensor(rows)


@pytest.mark.parametrize("start, n, floor", [(64, 24, 64), (64, 24, 60), (70, 9, 64),
                                             (64, 24, 0), (64, 24, 57)])
def test_attention_window_floor(global_ctx, start, n, floor):
    x = torch.zeros(1, n, 4)
    attn, backend = _attention(global_ctx)
    attn.prefill_single(x, start, 0)
    plain = backend.attended[-1][0][0, :, :WINDOW]
    assert torch.equal(plain, _window_positions(start, n, 0))
    attn.prefill_single(x, start, 0, window_floor=floor)
    clamped, n_window = backend.attended[-1]
    clamped = clamped[0, :, :WINDOW]
    assert n_window == WINDOW
    assert torch.equal(clamped, _window_positions(start, n, floor))
    # Rows whose window starts at or after the floor are untouched.
    free = torch.arange(start, start + n) - WINDOW + 1 >= floor
    assert torch.equal(clamped[free], plain[free])
    # Both calls stored the same window KV in the same slots.
    (_, kv0, s0), (_, kv1, s1) = backend.stored[-2:]
    assert torch.equal(kv0, kv1) and torch.equal(s0, s1)
    assert torch.equal(s0, 1000 + torch.arange(start, start + n))


def test_attention_window_floor_validation(global_ctx):
    attn, _ = _attention(global_ctx)
    for floor in (-1, 65):
        with pytest.raises(ValueError, match="window floor"):
            attn.prefill_single(torch.zeros(1, 4, 4), 64, 0, window_floor=floor)


def test_source_publication_matches_prefill_single(global_ctx):
    published = []
    attn, backend = _attention(global_ctx, owns_kv=True, published=published)
    x = torch.zeros(1, 24, 4)
    attn.prefill_single(x, 64, 2)
    attended = len(backend.attended)
    attn.publish_prefill_source(x, 64, 2)
    assert len(backend.attended) == attended  # no attention
    (_, kv0, s0), (_, kv1, s1) = backend.stored[-2:]
    assert torch.equal(kv0, kv1) and torch.equal(s0, s1)
    assert len(published) == 2
    assert published[0][:2] == published[1][:2] == (64, 2)
    assert torch.equal(published[0][2], published[1][2])
    attn.plan = SimpleNamespace(owns_kv=False)
    with pytest.raises(RuntimeError, match="publishes no compressed KV"):
        attn.publish_prefill_source(x, 64, 2)


# ----------------------------------------------------------------------------- adapter routing
def _req(start, end, *, final):
    cls = Req if final else ChunkedReq
    return cls(input_ids=torch.arange(end, dtype=torch.int32), table_idx=0, cached_len=start,
               output_len=4, uid=UID, sampling_params=SamplingParams(), cache_handle=None)


def _batch(req, cone):
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = req.input_ids[req.cached_len:req.device_len].clone()
    batch.positions = torch.arange(req.cached_len, req.device_len, dtype=torch.int32)
    batch.dsv41_cone = cone
    return batch


def _adapter(*, worker, calls):
    model = object.__new__(DeepseekV41ForCausalLM)
    model._args = _args()
    model._execution = SimpleNamespace(is_expert_worker=worker, rank=0, enabled=True,
                                       uses_prefill_subgroup=False)
    model.prepare_for_runtime = lambda: None
    model._engram_rows = lambda batch, ids: {}

    class Inner:
        _dspark_hidden = None

        def forward(self, input_ids, **kw):
            calls.append(("worker", input_ids.numel(), kw))
            return torch.zeros(1, 1)

        def prefill_single(self, input_ids, **kw):
            calls.append(("authority", input_ids.numel(), kw))
            return torch.zeros(1, VOCAB)

    model._model = Inner()
    return model


@pytest.mark.parametrize("worker", [False, True])
def test_adapter_routes_the_segment_and_leaves_the_plain_call_alone(global_ctx, worker):
    for req, cone in ((_req(0, 64, final=False), _encoder()),
                      (_req(64, 96, final=True), _decoder()),
                      (_req(64, 96, final=True), None)):
        calls = []
        model = _adapter(worker=worker, calls=calls)
        with global_ctx.forward_batch(_batch(req, cone)):
            model.forward()
        (role, tokens, kw), = calls
        assert role == ("worker" if worker else "authority") and tokens == req.extend_len
        assert kw.get("cone") is cone
        assert ("cone" in kw) == (cone is not None)


@pytest.mark.parametrize("req, cone, error", [
    (_req(0, 64, final=True), _encoder(), "would be sampled"),
    (_req(0, 64, final=False), _encoder(0, 72), "does not describe"),
    (_req(0, 64, final=False), ConeSegment(UID + 1, 0, 64, ENCODER, -1, SOURCE, N_LAYERS),
     "does not describe"),
    (_req(0, 64, final=False), ConeSegment(UID, 0, 64, ENCODER, -1, SOURCE - 1, N_LAYERS),
     "decoder source layer"),
])
def test_adapter_refuses_a_segment_that_does_not_fit(global_ctx, req, cone, error):
    model = _adapter(worker=False, calls=[])
    with global_ctx.forward_batch(_batch(req, cone)):
        with pytest.raises(DependencyConeMismatch, match=error):
            model.forward()


def test_adapter_refuses_dspark_and_non_prefill(global_ctx):
    model = _adapter(worker=False, calls=[])
    model._args = _args(dspark_enabled=True)
    with global_ctx.forward_batch(_batch(_req(0, 64, final=False), _encoder())):
        with pytest.raises(RuntimeError, match="MTP/DSpark off"):
            model.forward()
    model = _adapter(worker=True, calls=[])
    batch = _batch(_req(0, 64, final=False), _encoder())
    batch.phase = "decode"
    with global_ctx.forward_batch(batch):
        with pytest.raises(RuntimeError, match="not one bulk prefill"):
            model.forward()
