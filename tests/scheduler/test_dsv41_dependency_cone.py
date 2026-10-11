"""DeepSeek-V4.1 dependency-cone prefill: the scheduler side (CPU-only, no model).

* the layer-plan check and the cone bounds, on the checkpoint's own layer plan;
* chunk classification for every chunking the PrefillAdder can produce: fresh, resumed,
  uneven and page-capped;
* the cross-rank wire and its fail-closed agreement;
* the SWA-radix commit. Inexact decoder window KV never becomes matchable, no slot leaks, and
  after finish the tree serves the same matches as a plain prefill of the same prompt.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Context, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.attention_layout import decoder_source_layer
from freetoken.scheduler import cache as cache_mod
from freetoken.scheduler import dependency_cone as dc
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq
from freetoken.utils import align_down

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)

# The released checkpoint's layer plan (config.json: 40 layers + 3 MTP entries).
REAL_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)
GAP = cache_mod._SWA_RETAIN_GAP
UID = 11


def _real_args() -> DeepseekV41Args:
    return DeepseekV41Args(compress_ratios=REAL_RATIOS)


def _small_args(n_layers=6, source=3, window=8, engram=(1,), kv_sources=None):
    ratios = (0,) + (2,) * (source - 1) + (1,) * (n_layers - source)
    kv = kv_sources if kv_sources is not None else (1, source)
    return DeepseekV41Args(
        max_batch_size=1, max_seq_len=4096, vocab_size=64, dim=64, moe_inter_dim=32,
        n_layers=n_layers, n_heads=2, n_routed_experts=4, n_activated_experts=2,
        q_lora_rank=64, head_dim=64, rope_head_dim=16, o_groups=1, o_lora_rank=32,
        window_size=window, compress_ratios=ratios, kv_source_layers=kv,
        index_source_layers=tuple(sorted(set(kv) | {source + 1})), index_n_heads=2,
        index_head_dim=32, index_topk=2, candidate_source_layer=source,
        candidate_topk_blocks=1, candidate_block_size=2, engram_layer_ids=engram,
        engram_num_embeddings=tuple(1000 for _ in engram),
    )


def _real_geometry() -> dc.ConeGeometry:
    return dc.ConeGeometry.from_args(_real_args(), page_size=128, retain_gap=GAP)


def _small_geometry(page=8) -> dc.ConeGeometry:
    return dc.ConeGeometry.from_args(_small_args(window=page), page_size=page, retain_gap=GAP)


# --------------------------------------------------------------------------- flag and layer plan
def test_flag_is_strict_and_default_off(monkeypatch):
    monkeypatch.delenv(dc.DEPENDENCY_CONE_ENV, raising=False)
    assert not dc.dependency_cone_enabled()
    monkeypatch.setenv(dc.DEPENDENCY_CONE_ENV, "0")
    assert not dc.dependency_cone_enabled()
    monkeypatch.setenv(dc.DEPENDENCY_CONE_ENV, "1")
    assert dc.dependency_cone_enabled()
    for bad in ("true", "on", "2", " 1", ""):
        monkeypatch.setenv(dc.DEPENDENCY_CONE_ENV, bad)
        with pytest.raises(ValueError, match=dc.DEPENDENCY_CONE_ENV):
            dc.dependency_cone_enabled()


def test_minimum_uncached_tokens_is_strict(monkeypatch):
    monkeypatch.delenv(dc.DEPENDENCY_CONE_MIN_TOKENS_ENV, raising=False)
    assert dc.dependency_cone_min_tokens() == 0
    for value, expected in (("0", 0), ("65536", 65_536)):
        monkeypatch.setenv(dc.DEPENDENCY_CONE_MIN_TOKENS_ENV, value)
        assert dc.dependency_cone_min_tokens() == expected
    for bad in ("", "-1", "+1", "01", "1.0", " 1"):
        monkeypatch.setenv(dc.DEPENDENCY_CONE_MIN_TOKENS_ENV, bad)
        with pytest.raises(ValueError, match=dc.DEPENDENCY_CONE_MIN_TOKENS_ENV):
            dc.dependency_cone_min_tokens()


def test_real_layer_plan_bounds():
    """V4.1: decoder from layer 20 (the last KV source; 20-39 all read it), window 128."""
    assert decoder_source_layer(_real_args()) == 20
    g = _real_geometry()
    assert (g.n_layers, g.source_layer, g.window, g.page_size) == (40, 20, 128, 128)
    assert g.logits_depth == 19 * 127 == 2413
    assert g.window_depth == 18 * 127 == 2286
    # 32K: the finish-time retention bound binds; F lands exactly on the retention start.
    assert g.retention_start(32768) == 32512
    assert g.latest_decoder_start(32768) == 32512 - 2286 == 30226
    assert g.tombstone_frontier(30226) == 32512
    # The maximum context.
    assert g.latest_decoder_start(524288) == 524032 - 2286
    # Shorter than one 8K chunk plus the cone: no encoder-only chunk can exist.
    assert g.latest_decoder_start(10_000) < 8192 < g.latest_decoder_start(10_880)


def test_layer_plan_rejections():
    # Engram in the decoder: an encoder-only chunk would skip its lookup collective.
    with pytest.raises(ValueError, match="Engram layers \\[4\\]"):
        decoder_source_layer(_small_args(engram=(1, 4)))
    # No compressed-KV source at all.
    args = _small_args()
    args.kv_source_layers = ()
    with pytest.raises(ValueError, match="needs a compressed-KV source"):
        decoder_source_layer(args)
    # The last source is the first layer: there is no encoder to run alone.
    args = _small_args()
    args.compress_ratios = (1,) * args.n_layers
    args.kv_source_layers = args.index_source_layers = (0,)
    args.candidate_source_layer = 0
    with pytest.raises(ValueError, match="leaves no encoder"):
        decoder_source_layer(args)


def test_layer_plan_rejects_a_window_only_decoder_layer():
    args = _small_args()
    args.compress_ratios = args.compress_ratios[:-1] + (0,)  # last layer reads no compressed KV
    with pytest.raises(ValueError, match="does not read the last KV source"):
        decoder_source_layer(args)


# --------------------------------------------------------------------------- the bounds
@pytest.mark.parametrize("geometry", ["real", "small"])
def test_bounds_hold_and_are_tight_for_every_prompt_length(geometry):
    g = _real_geometry() if geometry == "real" else _small_geometry()
    P, w = g.page_size, g.window
    lengths = range(1, 40_000, 37) if geometry == "real" else range(1, 2_000)
    for n in lengths:
        r0 = g.latest_decoder_start(n)
        frontier = g.tombstone_frontier(r0)
        # The final logits, the decode window, and every published window are exact.
        assert n - 1 - r0 >= g.logits_depth
        assert n - w + 1 >= r0 + g.window_depth
        assert frontier % P == 0 and frontier >= r0 + g.window_depth
        assert frontier <= align_down(n, P) - P  # a commit keeps a live, exact last page
        assert frontier <= g.retention_start(n)  # finish-time retention is unchanged
        # Tight: one row later breaks the binding bound.
        late = r0 + 1
        assert (n - 1 - late < g.logits_depth
                or g.tombstone_frontier(late) > align_down(n, P) - P
                or g.tombstone_frontier(late) > g.retention_start(n))


# --------------------------------------------------------------------------- classification
def _chunks(start, end, size):
    out, a = [], start
    while a < end:
        b = min(a + size, end)
        out.append((a, b))
        a = b
    return out


def _classify(g, n, chunks, admitted=0):
    state = dc.ConeRequestState(g, UID, n, admitted, g.latest_decoder_start(n))
    return state, [state.classify(a, b) for a, b in chunks]


def test_max_context_8k_chunks():
    g = _real_geometry()
    n = 524_288
    latest = g.latest_decoder_start(n)
    state, segments = _classify(g, n, _chunks(0, n, 8192))
    encoder = [s for s in segments if s.is_encoder]
    decoder = [s for s in segments if not s.is_encoder]
    assert all(s.end <= latest for s in encoder) and encoder
    assert all(s.end > latest for s in decoder[:1])
    r0 = decoder[0].start
    assert state.decoder_start == r0 == align_down(latest, 8192) <= latest
    assert [s.start for s in decoder] == list(range(r0, n, 8192))
    assert all(s.decoder_start == r0 for s in decoder)
    # Decoder rows: at most one chunk plus the cone, out of the whole context.
    assert n - r0 <= 8192 + (n - latest)
    assert (len(encoder), n - r0) == (63, 8192)  # 63 of 64 chunks skip the decoder
    for s in encoder:
        assert s.executed_layers == 20 and s.decoder_start == -1
        assert all(s.window_floor(layer) is None for layer in range(40))
    for s in decoder:
        assert s.executed_layers == 40
        assert [s.window_floor(layer) for layer in (0, 19, 20)] == [None, None, None]
        assert all(s.window_floor(layer) == r0 for layer in range(21, 40))


@pytest.mark.parametrize("admitted", [0, 2048, 4096])
@pytest.mark.parametrize("sizes", [(8192,), (128, 8192, 640, 8192), (4096, 1024, 256)])
def test_uneven_and_resumed_chunkings(admitted, sizes):
    g = _real_geometry()
    n = 41_000
    latest = g.latest_decoder_start(n)
    chunks, a, i = [], admitted, 0
    while a < n:
        b = min(a + sizes[i % len(sizes)], n)
        chunks.append((a, b))
        a, i = b, i + 1
    if chunks[0][1] > latest:
        pytest.skip("first chunk crosses R*: the plain prefill")
    state, segments = _classify(g, n, chunks, admitted)
    kinds = [s.kind for s in segments]
    first_dec = kinds.index(dc.DECODER)
    assert kinds == [dc.ENCODER] * first_dec + [dc.DECODER] * (len(kinds) - first_dec)
    assert segments[first_dec].start == state.decoder_start <= latest < segments[first_dec].end
    assert segments[-1].end == n and not segments[-1].is_encoder


def test_classification_rejects_gaps_and_rewinds():
    g = _real_geometry()
    n = 30_000
    assert g.latest_decoder_start(n) == 27_410
    state = dc.ConeRequestState(g, UID, n, 0, g.latest_decoder_start(n))
    assert state.classify(0, 8192).is_encoder
    with pytest.raises(ValueError, match="does not continue"):
        state.classify(8320, 16384)
    assert state.classify(8192, 16384).is_encoder
    assert state.classify(16384, 24576).is_encoder
    first = state.classify(24576, n)
    assert (first.kind, first.decoder_start, state.decoder_start) == (dc.DECODER, 24576, 24576)
    with pytest.raises(ValueError, match="precedes"):
        state.classify(0, 128)
    fresh = dc.ConeRequestState(g, UID, n, 0, g.latest_decoder_start(n))
    with pytest.raises(ValueError, match="not encoder-only"):
        fresh.classify(0, n)
    with pytest.raises(ValueError, match="outside prompt"):
        fresh.classify(0, n + 1)


# --------------------------------------------------------------------------- wire / agreement
def _segment(kind=dc.DECODER, start=8192, end=16384, r0=8192):
    return dc.ConeSegment(UID, start, end, kind, r0 if kind == dc.DECODER else -1, 20, 40)


def test_wire_roundtrip_and_rejections():
    for seg in (_segment(), _segment(dc.ENCODER, 0, 8192)):
        assert dc.ConeSegment.decode(seg.encode()) == seg
    assert dc.ConeSegment.decode(dc.ConeSegment.encode_none()) is None
    good = _segment().encode()
    for bad in (good[:-1], good.to(torch.int32), good.clone().fill_(0)):
        with pytest.raises(dc.DependencyConeMismatch):
            dc.ConeSegment.decode(bad)
    corrupt = good.clone()
    corrupt[6] = 9  # unknown kind
    with pytest.raises(dc.DependencyConeMismatch):
        dc.ConeSegment.decode(corrupt)
    late = good.clone()
    late[7] = 9000  # decoder start after the chunk start
    with pytest.raises(dc.DependencyConeMismatch):
        dc.ConeSegment.decode(late)
    absent_with_data = dc.ConeSegment.encode_none()
    absent_with_data[4] = 1
    with pytest.raises(dc.DependencyConeMismatch):
        dc.ConeSegment.decode(absent_with_data)


def _two_ranks(*locals_):
    """Every rank's verdict under simulated CPU collectives: rank 0's wire is broadcast and the
    per-rank mismatch flags are summed."""
    wires = [seg.encode() if seg else dc.ConeSegment.encode_none() for seg in locals_]
    total = sum(int(not torch.equal(wires[0], wire)) for wire in wires)
    verdicts = []
    for rank, local in enumerate(locals_):
        try:
            verdicts.append(dc.agree_cone_segment(
                local, rank=rank, world_size=len(locals_),
                broadcast=lambda t: wires[0].clone(),
                all_reduce_sum=lambda flag: torch.tensor([total], dtype=torch.int64)))
        except dc.DependencyConeMismatch as exc:
            verdicts.append(exc)
    return verdicts


def test_agreement_proceeds_together_or_fails_together():
    seg = _segment()
    assert _two_ranks(seg, seg, seg, seg) == [seg] * 4
    assert _two_ranks(None, None) == [None, None]
    for ranks in ((seg, None), (None, seg), (seg, seg, _segment(start=8320, r0=8320), seg)):
        assert all(isinstance(v, dc.DependencyConeMismatch) for v in _two_ranks(*ranks))
    assert dc.agree_cone_segment(seg, rank=0, world_size=1, broadcast=None,
                                 all_reduce_sum=None) is seg


# --------------------------------------------------------------------------- startup checks
def _config(**kw):
    model_config = SimpleNamespace(dsv41_args=kw.pop("args", _real_args()))
    base = dict(model_config=model_config, page_size=128, max_running_req=1,
                speculative_dspark=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _plan(authority=True):
    return SimpleNamespace(uses_authority_transport=authority)


def test_startup_problems(monkeypatch):
    for env in (dc.DEPENDENCY_CONE_ENV, "FREETOKEN_DSV41_TAIL_PAIRING",
                "FREETOKEN_DSV41_PREFILL_EARLY_REFILL"):
        monkeypatch.delenv(env, raising=False)
    ok = dc.dependency_cone_problems(_config(), _plan(), requested_max_running_req=1)
    assert ok == []
    assert dc.dependency_cone_problems(
        SimpleNamespace(model_config=SimpleNamespace(dsv41_args=None)), _plan(),
        requested_max_running_req=1) == ["DeepSeek-V4.1"]
    problems = dc.dependency_cone_problems(
        _config(speculative_dspark=True, max_running_req=1), _plan(False),
        requested_max_running_req=2)
    assert any("authority-EP" in p for p in problems)
    assert any("--max-running-requests 1 (got 2)" in p for p in problems)
    assert any("MTP/DSpark off" in p for p in problems)
    monkeypatch.setenv("FREETOKEN_DSV41_TAIL_PAIRING", "1")
    monkeypatch.setenv("FREETOKEN_DSV41_PREFILL_EARLY_REFILL", "1")
    problems = dc.dependency_cone_problems(_config(), _plan(), requested_max_running_req=1)
    assert problems == ["FREETOKEN_DSV41_PREFILL_EARLY_REFILL off"]
    monkeypatch.setenv(dc.DEPENDENCY_CONE_MIN_TOKENS_ENV, "32k")
    problems = dc.dependency_cone_problems(_config(), _plan(), requested_max_running_req=1)
    assert any(dc.DEPENDENCY_CONE_MIN_TOKENS_ENV in problem for problem in problems)


def test_scheduler_geometry(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DSV41_TAIL_PAIRING", raising=False)
    monkeypatch.delenv(dc.DEPENDENCY_CONE_ENV, raising=False)
    monkeypatch.delenv(dc.DEPENDENCY_CONE_MIN_TOKENS_ENV, raising=False)
    assert dc.dependency_cone_geometry(_config()) is None  # off: nothing is built
    monkeypatch.setenv(dc.DEPENDENCY_CONE_ENV, "1")
    assert dc.dependency_cone_geometry(_config()) == _real_geometry()
    with pytest.raises(ValueError, match="only supported for DeepSeek-V4.1"):
        dc.dependency_cone_geometry(SimpleNamespace(model_config=SimpleNamespace(), page_size=1))
    monkeypatch.setenv("FREETOKEN_DSV41_TAIL_PAIRING", "1")
    monkeypatch.setenv(dc.DEPENDENCY_CONE_MIN_TOKENS_ENV, "65536")
    assert dc.dependency_cone_geometry(_config()) == replace(
        _real_geometry(), min_uncached_tokens=65_536
    )


def test_cache_window_must_match_cone_geometry():
    geometry = _real_geometry()
    dc.validate_dependency_cone_cache_window(None, None)
    dc.validate_dependency_cone_cache_window(geometry, geometry.window)
    with pytest.raises(ValueError, match="scheduler/cache sliding window"):
        dc.validate_dependency_cone_cache_window(geometry, geometry.window * 2)


# --------------------------------------------------------------------------- scheduler + cache
PAGE = 8  # the small geometry: window == page == 8, decoder from layer 3 of 6
MAX_RUNNING = 1


def _managers(num_swa_tokens=4096 + 1, num_pages=4096, width=4096):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=PAGE))
    groups = (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8,
                         sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8,
                         sliding_window=PAGE),
    )
    pool = HybridSWAKVCache(groups=groups, num_layers=2, num_full_pages=num_pages,
                            page_size=PAGE, dtype=torch.bfloat16, device=torch.device("cpu"),
                            num_swa_tokens=num_swa_tokens)
    pt = torch.zeros((MAX_RUNNING + 1, width), dtype=torch.int32)
    cm = CacheManager(num_pages=num_pages, page_size=PAGE, page_table=pt, type="swa_radix",
                      swa_pool=pool, sliding_window_size=PAGE)
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    return cm, tm, PrefillManager(cm, tm, DecodeManager(page_size=PAGE))


def _ids(n, base=1):
    return torch.arange(base, base + n, dtype=torch.int32)


def _prefill(cm, pm, ids, geometry, budget, *, n_decode=4, stop_after=None):
    """Real chunked prefill in the scheduler's overlap order: plan + prepare chunk N+1, then
    drain chunk N (only the final chunk commits). Returns (final req or last chunk, segments)."""
    pm.pending_list = [PendingReq(uid=UID, input_ids=ids,
                                  sampling_params=SamplingParams(max_tokens=n_decode))]
    final = last_batch = None
    segments = []
    while pm.runnable or last_batch is not None:
        batch = pm.schedule_next_batch(budget) if pm.runnable else None
        if batch is not None:
            segments.append(dc.plan_prefill_cone(batch, geometry) if geometry else None)
            cm.free_swa_out_of_window_extend(batch.reqs)
            cm.allocate_paged(batch.reqs)
            for r in batch.reqs:
                r.complete_one()
        if last_batch is not None:
            for r in last_batch.reqs:
                if not isinstance(r, ChunkedReq):
                    cm.cache_req(r, finished=False)
                    final = r
        last_batch = batch
        if stop_after is not None and len(segments) == stop_after:
            return batch.reqs[0], segments
    return final, segments


def _decode(cm, req, n):
    for i in range(n):
        req.append_host(torch.tensor([9999], dtype=torch.int32))
        req.decode_batch_idx = i + 1
        cm.maybe_free_swa_out_of_window([req], forward_iter=i + 1)
        cm.allocate_paged([req])
        req.complete_one()


def _finish(cm, tm, req):
    cm.cache_req(req, finished=True)  # Scheduler._free_req_resources
    tm.free(req.table_idx)
    req.table_idx = -1


def _matches(cm, ids, step=PAGE):
    return [cm.prefix_cache.match_prefix(ids[:m]).cached_len for m in range(0, len(ids) + 1, step)]


@pytest.fixture(autouse=True)
def _every_rank_runs_the_same_eviction(monkeypatch):
    monkeypatch.setattr(cache_mod, "_SWA_EVICTION_INTERVAL", 1)


def test_planner_on_the_real_prefill_manager():
    g = _small_geometry()
    n = 400
    cm, tm, pm = _managers()
    final, segments = _prefill(cm, pm, _ids(n), g, budget=64)
    state = final.dsv41_cone
    assert isinstance(state, dc.ConeRequestState) and state.prompt_len == n
    kinds = [s.kind if s else None for s in segments]
    first_dec = kinds.index(dc.DECODER)
    assert kinds == [dc.ENCODER] * first_dec + [dc.DECODER] * (len(kinds) - first_dec)
    assert state.decoder_start == segments[first_dec].start <= state.latest_decoder_start
    # A short prompt never builds a state; neither does one whose first chunk crosses R*.
    for prompt, budget in ((40, 64), (400, 512)):
        cm2, _tm2, pm2 = _managers()
        final2, segments2 = _prefill(cm2, pm2, _ids(prompt), g, budget=budget)
        assert segments2 == [None] * len(segments2) and final2.dsv41_cone is None


def test_admission_threshold_counts_only_uncached_tokens():
    g = _small_geometry()
    n = 400
    for minimum, engages in ((n, True), (n + 1, False)):
        cm, _tm, pm = _managers()
        policy = replace(g, min_uncached_tokens=minimum)
        final, segments = _prefill(cm, pm, _ids(n), policy, budget=64)
        assert (final.dsv41_cone is not None) is engages
        assert any(segments) is engages

    # The total prompt is long, but only the suffix after a cache hit is new work.  The second
    # request must therefore stay below a threshold larger than that uncached suffix.
    cm, tm, pm = _managers()
    first = _ids(n)
    final, _ = _prefill(cm, pm, first, None, budget=64, n_decode=2)
    _decode(cm, final, 2)
    prefix = final.input_ids.clone()
    _finish(cm, tm, final)
    suffix = _ids(200, base=5000)
    second = torch.cat([prefix, suffix])
    resumed, segments = _prefill(
        cm, pm, second, replace(g, min_uncached_tokens=len(suffix) + 16), budget=64
    )
    assert resumed.cached_len > 0
    assert resumed.dsv41_cone is None
    assert all(segment is None for segment in segments)


@pytest.mark.parametrize("n, hook_binds", [(380, True), (400, False)])
def test_commit_tombstones_the_inexact_region_and_keeps_an_exact_live_tail(n, hook_binds):
    """The commit tombstones every position below max(F, the extend-free watermark).

    At 380 the decoder starts in the final chunk. The extend-free watermark (final start - 16)
    then lies below F, and only the cone's frontier keeps [304, 328) unmatchable. At 400 the
    decoder started one chunk earlier, and the watermark already covers F.
    """
    g = _small_geometry()
    ids = _ids(n)
    cm, tm, pm = _managers()
    final, segments = _prefill(cm, pm, ids, g, budget=64)
    frontier = g.tombstone_frontier(final.dsv41_cone.decoder_start)
    watermark = align_down(segments[-1].start - 2 * PAGE, PAGE)  # the final chunk's extend-free
    assert (frontier > watermark) == hook_binds
    assert final.swa_evicted_seqlen == max(frontier, watermark)
    insert_len = align_down(n, PAGE)
    for m in range(PAGE, insert_len + 1, PAGE):
        got = cm.prefix_cache.match_prefix(ids[:m]).cached_len
        if m - PAGE >= max(frontier, watermark):  # the window before m is exact and live
            assert got == m, m
        else:                                     # never a match ending in the inexact region
            assert got == 0, m
    cm.prefix_cache.check_integrity()
    _decode(cm, final, 4)
    _finish(cm, tm, final)
    cm.check_integrity()  # idle: free + tree == capacity in both currencies


@pytest.mark.parametrize("n", [380, 400, 409, 1000])
def test_after_finish_the_tree_serves_the_plain_prefills_matches(n):
    """Cache semantics: once the request finishes, the cone's tree and the plain prefill's tree
    reuse exactly the same prefixes (the next turn's resume near the prompt end included)."""
    g = _small_geometry()
    ids = _ids(n)
    trees = []
    for geometry in (None, g):
        cm, tm, pm = _managers()
        final, segments = _prefill(cm, pm, ids, geometry, budget=64, n_decode=6)
        assert (final.dsv41_cone is None) == (geometry is None)
        _decode(cm, final, 6)
        tokens = final.input_ids.clone()
        _finish(cm, tm, final)
        cm.check_integrity()
        trees.append(_matches(cm, tokens))
    assert trees[0] == trees[1]
    assert trees[0][-1] > 0  # the resume point near the end is live in both


def test_abort_before_the_decoder_publishes_nothing_and_leaks_nothing():
    g = _small_geometry()
    ids = _ids(400)
    cm, tm, pm = _managers()
    chunk, segments = _prefill(cm, pm, ids, g, budget=64, stop_after=2)
    assert all(s.is_encoder for s in segments)
    pm.abort_req(UID)
    _finish(cm, tm, chunk)
    assert _matches(cm, ids) == [0] * len(_matches(cm, ids))
    cm.check_integrity()


def test_unfinished_commit_without_an_exact_frontier_fails_closed():
    g = _small_geometry()
    cm, _tm, pm = _managers()
    chunk, segments = _prefill(cm, pm, _ids(400), g, budget=64, stop_after=2)
    assert all(segment.is_encoder for segment in segments)
    with pytest.raises(RuntimeError, match="before its exact window frontier"):
        cm.cache_req(chunk, finished=False)


@pytest.mark.parametrize("stop_after", [5, 6, 7])
def test_abort_after_the_decoder_started(stop_after):
    g = _small_geometry()
    ids = _ids(400)
    cm, tm, pm = _managers()
    chunk, segments = _prefill(cm, pm, ids, g, budget=64, stop_after=stop_after)
    pm.abort_req(UID)
    _finish(cm, tm, chunk)
    state = chunk.dsv41_cone
    frontier = state.swa_commit_frontier(align_down(chunk.cached_len, PAGE))
    for m, got in zip(range(0, len(ids) + 1, PAGE), _matches(cm, ids)):
        if got:
            assert frontier is not None and got - PAGE >= frontier
    cm.check_integrity()


def test_resumed_prompt_runs_its_own_cone_from_the_matched_prefix():
    g = _small_geometry()
    first = _ids(400)
    cm, tm, pm = _managers()
    final, _ = _prefill(cm, pm, first, g, budget=64, n_decode=2)
    _decode(cm, final, 2)
    tokens = final.input_ids.clone()
    _finish(cm, tm, final)
    second = torch.cat([tokens, _ids(600, base=5000)])
    resumed, segments = _prefill(cm, pm, second, g, budget=64, n_decode=2)
    state = resumed.dsv41_cone
    assert state is not None and state.admitted_len > 0
    assert state.admitted_len == segments[0].start
    assert state.decoder_start > state.admitted_len
    _decode(cm, resumed, 2)
    _finish(cm, tm, resumed)
    cm.check_integrity()
