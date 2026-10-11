"""DSV4.1 tail pairing reuses each layer's prefill double buffer (CPU OffloadMoeCache).

Drives the real ``RoutedExperts._wait_prefill_overlap`` -> ``OffloadMoELayer._prefill_routed``
-> ``release_prefill_layer`` chain over every layer, in the unpaired order (head forward, then a
fresh tail forward) and the paired layer-major order (head L, tail L, head L+1, ...). Only the
expert GEMM is replaced, by a probe that records the exact bank bytes each chunk reads. The
CUDA gate test at the end repeats the paired order on a real CUDA cache with a slowed tail.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

import freetoken.models.deepseek_v41.moe as dsv41_moe
from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.moe import ExpertParallelOffloadMoELayer
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.moe.offload_cache import OffloadMoeCache

NUM_LAYERS = 5
NUM_EXPERTS = 4
HIDDEN = 8
INTER = 16


@pytest.fixture
def ctx():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    try:
        context = get_global_ctx()
    except AssertionError:
        context = Context(page_size=128)
        set_global_ctx(context)
    yield context
    context._batch = None


def _args(num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS):
    return DeepseekV41Args(
        vocab_size=64, dim=HIDDEN, moe_inter_dim=INTER, n_layers=num_layers, n_heads=2,
        q_lora_rank=32, head_dim=32, rope_head_dim=16, o_groups=2, o_lora_rank=16,
        compress_ratios=(0,) * num_layers, kv_source_layers=(), index_source_layers=(),
        candidate_source_layer=-1, index_n_heads=2, index_head_dim=32, index_topk=4,
        engram_layer_ids=(), engram_num_embeddings=(), n_routed_experts=num_experts,
        n_activated_experts=2,
    )


def _stack(monkeypatch):
    cache = OffloadMoeCache(num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS,
                            cache_size=2 * NUM_EXPERTS + 2, device=torch.device("cpu"),
                            prefill_overlap=True)
    gate_up = list(torch.arange(NUM_LAYERS * NUM_EXPERTS * 2 * INTER * HIDDEN,
                                dtype=torch.float32).reshape(-1, 2 * INTER, HIDDEN)
                   .split(NUM_EXPERTS))
    down = list((-torch.arange(NUM_LAYERS * NUM_EXPERTS * HIDDEN * INTER, dtype=torch.float32))
                .reshape(-1, HIDDEN, INTER).split(NUM_EXPERTS))
    cache.set_bank_sources({"gate_up": gate_up, "down": down})
    layers = [dsv41_moe.RoutedExperts(layer, _args(), NUM_EXPERTS) for layer in range(NUM_LAYERS)]
    for layer in layers:
        layer.offload_cache = cache
    copies = []
    invalidate = cache._invalidate_prefill_buffer
    monkeypatch.setattr(cache, "_invalidate_prefill_buffer",
                        lambda buffer_id: copies.append(buffer_id) or invalidate(buffer_id))
    reads = []

    def probe(hidden, w1, w2, weights, ids, activation, apply_router_weight_on_input):
        batch = get_global_ctx().batch
        reads.append((batch.reqs[0].cached_len, w1.clone(), w2.clone()))
        return hidden + w1.sum() * 0 + len(reads)

    monkeypatch.setattr("freetoken.layers.moe.fused_experts_impl", probe)
    return cache, layers, (gate_up, down), copies, reads


def _batch(start, n, role=None):
    req = Req(input_ids=torch.zeros(start + n, dtype=torch.int32), table_idx=0, cached_len=start,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=None)
    batch = Batch(reqs=[req], phase="prefill")
    batch.tail_pair_role = role
    return batch


def _moe(layer, n, device=None):
    hidden = torch.ones(n, HIDDEN, device=device)
    weights = torch.full((n, 2), 0.5, device=device)
    ids = torch.tensor([[0, 1]] * n, dtype=torch.int32, device=device)
    return ExpertParallelOffloadMoELayer._prefill_routed(layer, hidden, weights, ids)


def test_paired_tail_reads_each_layers_bank_without_restreaming(ctx, monkeypatch):
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", True)
    head, tail = _batch(8192, 64, "head"), _batch(8256, 24, "tail")

    cache, layers, banks, copies, reads = _stack(monkeypatch)
    for layer in layers:  # unpaired: two forwards, two prefill sessions
        with ctx.forward_batch(_batch(8192, 64)):
            _moe(layer, 64)
    for layer in layers:
        with ctx.forward_batch(_batch(8256, 24)):
            _moe(layer, 24)
    unpaired_copies, unpaired_reads = list(copies), list(reads)

    cache, layers, banks, copies, reads = _stack(monkeypatch)
    for layer in layers:  # paired: one session, head then tail at every layer
        for batch, n in ((head, 64), (tail, 24)):
            with ctx.forward_batch(batch):
                _moe(layer, n)
    assert len(unpaired_copies) == 2 * NUM_LAYERS
    assert len(copies) == NUM_LAYERS  # every layer streamed exactly once for both chunks
    gate_up, down = banks
    for reads_, order in ((unpaired_reads, "unpaired"), (reads, "paired")):
        by_chunk = {start: [(w1, w2) for s, w1, w2 in reads_ if s == start]
                    for start in (8192, 8256)}
        for start, seen in by_chunk.items():
            assert len(seen) == NUM_LAYERS, (order, start)
            for layer_id, (w1, w2) in enumerate(seen):
                assert torch.equal(w1, gate_up[layer_id]), (order, start, layer_id)
                assert torch.equal(w2, down[layer_id]), (order, start, layer_id)
    assert [s for s, _, _ in reads] == [8192, 8256] * NUM_LAYERS


def test_reuse_refuses_a_layer_that_is_not_resident(ctx, monkeypatch):
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", True)
    cache, layers, _, copies, _ = _stack(monkeypatch)
    with ctx.forward_batch(_batch(8256, 24, "tail")):
        with pytest.raises(RuntimeError, match="not resident"):
            _moe(layers[0], 24)  # no head streamed layer 0 in this session
    with ctx.forward_batch(_batch(8192, 64, "head")):
        _moe(layers[0], 64)
        _moe(layers[1], 64)
        _moe(layers[2], 64)  # layer 2 now owns buffer 0
    with ctx.forward_batch(_batch(8256, 24, "tail")):
        with pytest.raises(RuntimeError, match="not resident"):
            _moe(layers[0], 24)  # layer 0's slot was overwritten
    with pytest.raises(RuntimeError, match="not resident"):
        cache.reuse_prefill_layer(NUM_LAYERS)
    assert copies  # the refusals themselves never copied


def test_a_reused_buffer_stays_unreleased_until_the_tail_releases_it(ctx, monkeypatch):
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", True)
    cache, layers, _, _, _ = _stack(monkeypatch)
    with ctx.forward_batch(_batch(8192, 64, "head")):
        _moe(layers[0], 64)  # streams layers 0 and 1, then releases layer 0
    assert cache._prefill_buffer_layer == [0, 1]
    cache.reuse_prefill_layer(0)  # the tail now reads buffer 0
    with pytest.raises(AssertionError, match="reused before release"):
        cache.prefetch_prefill_layer(2)  # layer 2 would land under the tail's GEMMs
    cache.release_prefill_layer(0)  # the tail's own release
    cache.prefetch_prefill_layer(2)
    assert cache._prefill_buffer_layer == [2, 1]


def test_reuse_requires_prefill_overlap(ctx):
    cache = OffloadMoeCache(num_layers=2, num_experts=2, cache_size=4,
                            device=torch.device("cpu"), prefill_overlap=False)
    with pytest.raises(RuntimeError, match="prefill overlap"):
        cache.reuse_prefill_layer(0)


def test_flag_off_never_consults_the_batch(ctx, monkeypatch):
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", False)
    cache, layers, _, copies, reads = _stack(monkeypatch)
    with ctx.forward_batch(_batch(8192, 64)):
        monkeypatch.setattr(dsv41_moe, "get_global_ctx",
                            lambda: (_ for _ in ()).throw(AssertionError("batch consulted")))
        for layer in layers:
            layer._wait_prefill_overlap(cache)
            cache.release_prefill_layer(layer.layer_id)
    assert len(copies) == NUM_LAYERS


def test_tail_role_is_ignored_when_the_flag_is_off(ctx, monkeypatch):
    """A role on the batch alone never selects reuse; the flag gates the behaviour."""
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", False)
    cache, layers, _, copies, _ = _stack(monkeypatch)
    with ctx.forward_batch(_batch(8256, 24, "tail")):
        for layer in layers:
            _moe(layer, 24)
    assert len(copies) == NUM_LAYERS  # streamed normally


# ----------------------------------------------------------------------------- CUDA gate test
#
# Runs on the isolated GPUs before G2 (qualification-plan.md). The CPU cache above copies
# synchronously, so it cannot show that layer L+2's copy waits for the TAIL's GEMMs: only the
# tail's re-recorded release event orders it, and a missing re-record is a silent race. Here the
# tail's GEMM is a probe that stalls the compute stream and then snapshots its bank views on
# that stream, so an unordered copy lands first. The negative control must observe that.

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
GATE_LAYERS = 6
GATE_EXPERTS = 8
RESIDENT = (1, 5)  # hit-region experts of every layer; the other six are misses
TAIL_SLEEP_CYCLES = 50_000_000  # tens of milliseconds, far longer than one layer's copy


def _require_hit_d2d_runtime():
    """The JIT and driver prerequisites of tests/moe/test_prefill_hit_d2d.py."""
    if os.getenv("FREETOKEN_DISABLE_JIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        pytest.skip("batch_memcpy has no AOT prebuild; needs runtime JIT")
    version = torch.version.cuda
    if version is None or tuple(int(x) for x in version.split(".")[:2]) < (13, 0):
        pytest.skip("the cudaMemcpyBatchAsync binding needs CUDA >= 13.0")


def _gate_stack(monkeypatch, device, *, hit_d2d):
    """A real cache under the real RoutedExperts chain; the GEMM probe snapshots its views.

    ``gate_up`` rows are 256 KiB, so under hit-D2D its resident experts are gathered on the
    compute stream and only its misses cross PCIe on the copy stream. ``down`` is a small bank,
    copied whole-layer on the copy stream either way.
    """
    generator = torch.Generator().manual_seed(4101)

    def bank(*shape):
        layers = [torch.randn(GATE_EXPERTS, *shape, generator=generator).to(torch.bfloat16)
                  for _ in range(GATE_LAYERS)]
        return [layer.pin_memory() for layer in layers] if device.type == "cuda" else layers

    sources = {"gate_up": bank(512, 256), "down": bank(8, 16)}
    cache = OffloadMoeCache(
        num_layers=GATE_LAYERS, num_experts=GATE_EXPERTS,
        cache_size=2 * GATE_EXPERTS + len(RESIDENT) * GATE_LAYERS, device=device,
        prefill_overlap=True, prefill_hit_d2d=hit_d2d,
    )
    cache.set_bank_sources(sources)
    slot = 2 * GATE_EXPERTS  # the first slot past the double buffers
    for layer_id in range(GATE_LAYERS):
        for expert_id in RESIDENT:
            cache.slot_for_id[layer_id, expert_id] = slot
            cache.id_of_slot[slot] = layer_id * GATE_EXPERTS + expert_id
            for name, per_layer in sources.items():
                cache.bank_caches[name][slot].copy_(per_layer[layer_id][expert_id])
            slot += 1
    layers = [dsv41_moe.RoutedExperts(layer_id, _args(GATE_LAYERS, GATE_EXPERTS), GATE_EXPERTS)
              for layer_id in range(GATE_LAYERS)]
    for layer in layers:
        layer.offload_cache = cache
    snapshots = []
    tail_read_done = {}
    race_windows = []

    prefetch = cache.prefetch_prefill_layer

    def observe_prefetch(layer_id):
        # Head L+1 queues the L+2 copy that can overwrite tail L's parity buffer.  Query the
        # event recorded after tail L's snapshots before issuing that copy: if it has already
        # completed, this harness cannot exercise the ordering race and must not qualify it.
        # Only the call that issues the copy counts: the base class also re-requests the
        # resident layer (prefetch(L) and wait_prefill_layer(L)) and asks for layer num_layers.
        source_layer = layer_id - 2
        issues_copy = (
            layer_id < cache.num_layers and cache._prefill_buffer_layer[layer_id % 2] != layer_id
        )
        if device.type == "cuda" and issues_copy and source_layer in tail_read_done:
            race_windows.append((source_layer, layer_id, tail_read_done[source_layer].query()))
        return prefetch(layer_id)

    monkeypatch.setattr(cache, "prefetch_prefill_layer", observe_prefetch)

    def probe(hidden, w1, w2, weights, ids, activation, apply_router_weight_on_input):
        role = get_global_ctx().batch.tail_pair_role
        layer_id = len(snapshots) // 2
        if role == "tail" and device.type == "cuda":
            torch.cuda._sleep(TAIL_SLEEP_CYCLES)  # a slow tail GEMM
        snapshots.append((role, w1.clone(), w2.clone()))  # read on the compute stream
        if role == "tail" and device.type == "cuda":
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(device))
            tail_read_done[layer_id] = done
        return hidden

    monkeypatch.setattr("freetoken.layers.moe.fused_experts_impl", probe)
    return cache, layers, sources, snapshots, race_windows


def _run_paired_layers(ctx, layers, device):
    head, tail = _batch(8192, 64, "head"), _batch(8256, 24, "tail")
    # Prepare every input before timing-sensitive work begins.  Constructing
    # torch.tensor(list, device="cuda") inside the loop performs a blocking H2D copy and waits
    # for the compute stream, which made the old negative control incapable of creating a race.
    prepared = []
    for _ in layers:
        per_layer = []
        for n in (64, 24):
            hidden = torch.ones(n, HIDDEN, device=device)
            weights = torch.full((n, 2), 0.5, device=device)
            ids = torch.empty((n, 2), dtype=torch.int32, device=device)
            ids[:, 0].zero_()
            ids[:, 1].fill_(1)
            per_layer.append((hidden, weights, ids))
        prepared.append(per_layer)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for layer_id, layer in enumerate(layers):
        for role_id, batch in enumerate((head, tail)):
            hidden, weights, ids = prepared[layer_id][role_id]
            with ctx.forward_batch(batch):
                ExpertParallelOffloadMoELayer._prefill_routed(
                    layer, hidden, weights, ids
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _misread_layers(snapshots, sources):
    """(role, layer) of every snapshot that is not exactly its own layer's bank bytes."""
    misread = []
    for index, (role, gate_up, down) in enumerate(snapshots):
        layer_id = index // 2
        if not (torch.equal(gate_up.cpu(), sources["gate_up"][layer_id])
                and torch.equal(down.cpu(), sources["down"][layer_id])):
            misread.append((role, layer_id))
    return misread


@pytest.mark.parametrize(
    ("device", "hit_d2d"),
    [("cpu", False), pytest.param("cuda", False, marks=CUDA),
     pytest.param("cuda", True, marks=CUDA)],
)
def test_gate_slow_tail_reads_exactly_its_layer(ctx, monkeypatch, device, hit_d2d):
    """The CPU case checks this harness itself; the CUDA cases are the gate's reuse test."""
    device = torch.device(device)
    if hit_d2d:
        _require_hit_d2d_runtime()
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", True)
    cache, layers, sources, snapshots, race_windows = _gate_stack(
        monkeypatch, device, hit_d2d=hit_d2d
    )
    _run_paired_layers(ctx, layers, device)
    assert [role for role, _, _ in snapshots] == ["head", "tail"] * GATE_LAYERS
    assert _misread_layers(snapshots, sources) == []
    if device.type == "cuda":
        assert race_windows == [
            (layer, layer + 2, False) for layer in range(GATE_LAYERS - 2)
        ], "the L+2 copies were not issued while their tails were still reading"
    if hit_d2d:
        assert cache._prefill_hit_d2d_active  # a silent legacy fallback would test less
        assert cache.prefill_hit_rows == len(RESIDENT) * GATE_LAYERS
        assert cache.prefill_total_rows == GATE_EXPERTS * GATE_LAYERS  # each layer once


@CUDA
@pytest.mark.parametrize("hit_d2d", [False, True])
def test_gate_negative_control_tail_skips_release_record(ctx, monkeypatch, hit_d2d):
    """Mutation ``tail_skips_release_record``: the tail marks its buffer released without
    re-recording the event, so L+2's copy waits only for the head. It must be caught."""
    device = torch.device("cuda")
    if hit_d2d:
        _require_hit_d2d_runtime()
    monkeypatch.setattr(dsv41_moe, "_TAIL_PAIRING_ENABLED", True)
    cache, layers, sources, snapshots, race_windows = _gate_stack(
        monkeypatch, device, hit_d2d=hit_d2d
    )
    release = OffloadMoeCache.release_prefill_layer

    def tail_skips_release_record(self, layer_id):
        if get_global_ctx().batch.tail_pair_role == "tail":
            self._prefill_buffer_released[layer_id % 2] = True
            return None
        return release(self, layer_id)

    monkeypatch.setattr(OffloadMoeCache, "release_prefill_layer", tail_skips_release_record)
    _run_paired_layers(ctx, layers, device)
    assert race_windows == [
        (layer, layer + 2, False) for layer in range(GATE_LAYERS - 2)
    ], "the L+2 copies were not issued while their tails were still reading"
    misread = _misread_layers(snapshots, sources)
    assert misread, "the slowed tail read intact banks without its release record"
    # Only the tail is exposed: layers L < GATE_LAYERS - 2 are overwritten by L + 2.
    assert {role for role, _ in misread} == {"tail"}
    assert {layer for _, layer in misread} <= set(range(GATE_LAYERS - 2))
