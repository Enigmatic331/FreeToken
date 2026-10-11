"""Exactness of the DSV4.1 dependency-cone schedule on a CPU twin of V4.1's row coupling.

The twin keeps exactly the cross-row couplings the cone argument relies on, and nothing that
would hide a mistake in it:

* layer j's window attention reads layer j's own window KV, a function of its input, at rows
  [max(floor, t - w + 1), t];
* every layer from the source layer D on also reads one global KV that D publishes from its
  input. Encoder layers read their own source's global KV;
* everything else (norm, residual, MLP) is per row.

Rows are computed one at a time in float64, so a row's output is bitwise a function of its
inputs. FreeToken's kernels must have the same property at fixed chunk shapes; a GPU gate
checks that. Window entries a schedule never wrote are NaN, so a schedule that reads one
cannot match. The schedule, frontier and publication rule come from the real planner
(scheduler/dependency_cone.py).
"""
from __future__ import annotations

import math

import pytest
import torch

from freetoken.scheduler import dependency_cone as dc
from freetoken.utils import align_down

D_MODEL = 8
VOCAB = 61
L, D, W, P, GAP = 6, 3, 8, 8, 16  # 6 layers, decoder from layer 3, window == page == 8
UID = 5
CAPACITY = 1024  # rows of every twin cache: prompt, follow-up and decode


class Twin:
    def __init__(self, n_layers=L, source=D, window=W, *, enc_source=1, seed=0):
        gen = torch.Generator().manual_seed(seed)

        def rnd(*shape):
            return torch.randn(*shape, generator=gen, dtype=torch.float64) * 0.5

        self.n_layers, self.source, self.window = n_layers, source, window
        self.enc_source = enc_source
        self.embed = rnd(VOCAB, D_MODEL)
        self.layers = [
            {k: rnd(D_MODEL, D_MODEL) for k in ("q", "kv", "o", "up", "down")}
            for _ in range(n_layers)
        ]
        self.sources = {s: (rnd(D_MODEL, D_MODEL), rnd(D_MODEL, D_MODEL))
                        for s in (enc_source, source)}
        self.head = rnd(VOCAB, D_MODEL)

    def global_source(self, layer):
        if layer >= self.source:
            return self.source
        return self.enc_source if layer >= self.enc_source else None

    def new_cache(self, capacity):
        def nan(*shape):
            return torch.full(shape, float("nan"), dtype=torch.float64)

        return {"window": nan(self.n_layers, capacity, D_MODEL),
                "global": {s: nan(capacity, 2, D_MODEL) for s in self.sources}}

    @staticmethod
    def _norm(x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

    def _publish(self, layer, cache, t, h):
        cache["window"][layer, t] = torch.tanh(h @ self.layers[layer]["kv"])
        if layer in self.sources:
            k, v = self.sources[layer]
            cache["global"][layer][t, 0] = h @ k
            cache["global"][layer][t, 1] = h @ v

    def _attend(self, layer, cache, t, h, floor):
        keys = values = cache["window"][layer, max(floor, t - self.window + 1): t + 1]
        src = self.global_source(layer)
        if src is not None:
            g = cache["global"][src][: t + 1]
            keys, values = torch.cat([keys, g[:, 0]]), torch.cat([values, g[:, 1]])
        q = h @ self.layers[layer]["q"]
        return torch.softmax(keys @ q / math.sqrt(D_MODEL), 0) @ values

    def layer(self, layer, cache, x, start, floor=0):
        """One layer over a chunk: publish every row first, then attend row by row."""
        h = [self._norm(x[i]) for i in range(x.shape[0])]
        for i, hi in enumerate(h):
            self._publish(layer, cache, start + i, hi)
        p = self.layers[layer]
        out = torch.empty_like(x)
        for i, hi in enumerate(h):
            y = x[i] + torch.tanh(self._attend(layer, cache, start + i, hi, floor) @ p["o"])
            out[i] = y + torch.tanh(self._norm(y) @ p["up"]) @ p["down"]
        return out

    def chunk(self, cache, ids, start, end, segment=None):
        """A prefill chunk: plain, or as its cone segment says. Returns the last row's logits
        (None for an encoder-only chunk, whose logits nothing reads)."""
        x = self.embed[ids[start:end]]
        if segment is not None and segment.is_encoder:
            for layer in range(self.source):
                x = self.layer(layer, cache, x, start)
            for i in range(x.shape[0]):  # the source layer's publication only
                self._publish(self.source, cache, start + i, self._norm(x[i]))
            return None
        for layer in range(self.n_layers):
            floor = None if segment is None else segment.window_floor(layer)
            x = self.layer(layer, cache, x, start, 0 if floor is None else floor)
        return self.head @ self._norm(x[-1])

    def decode(self, cache, token, pos):
        x = self.embed[[token]]
        for layer in range(self.n_layers):
            x = self.layer(layer, cache, x, pos)
        return self.head @ self._norm(x[-1])


def _geometry():
    return dc.ConeGeometry(n_layers=L, source_layer=D, window=W, page_size=P, retain_gap=GAP)


def _prompt(n, seed=1):
    return torch.randint(0, VOCAB, (n,), generator=torch.Generator().manual_seed(seed))


def _chunking(start, end, sizes):
    out, a, i = [], start, 0
    while a < end:
        b = min(a + sizes[i % len(sizes)], end)
        out.append((a, b))
        a, i = b, i + 1
    return out


def _run(twin, ids, chunks, *, cone, cache=None, admitted=0):
    """Prefill ``ids`` from ``admitted`` in ``chunks``; with ``cone`` the real planner picks
    each chunk's segment. Returns (cache, last logits, state)."""
    cache = cache if cache is not None else twin.new_cache(CAPACITY)
    state = None
    if cone:
        g = _geometry()
        state = dc.ConeRequestState(g, UID, len(ids), admitted, g.latest_decoder_start(len(ids)))
    logits = None
    for a, b in chunks:
        segment = state.classify(a, b) if state is not None else None
        out = twin.chunk(cache, ids, a, b, segment)
        logits = out if out is not None else logits
    return cache, logits, state


def _decode_tail(twin, cache, n, steps=5):
    tokens = _prompt(steps, seed=99)
    return torch.stack([twin.decode(cache, int(tokens[i]), n + i) for i in range(steps)])


def _same(a, b):
    return a.shape == b.shape and torch.equal(a, b)


CASES = [
    (300, (40,), 0),
    (257, (64,), 0),
    (300, (13, 50, 7, 33), 0),
    (420, (48,), 96),      # resumed: a cached prefix of 96 rows
    (333, (24, 64), 40),   # resumed at a non-page row, uneven chunks
]


@pytest.mark.parametrize("n, sizes, admitted", CASES)
def test_cone_schedule_reproduces_the_plain_prefill_bit_for_bit(n, sizes, admitted):
    twin, ids = Twin(), _prompt(n)
    prefix = _chunking(0, admitted, (64,))
    chunks = _chunking(admitted, n, sizes)
    plain, plain_logits, _ = _run(twin, ids, prefix + chunks, cone=False)
    cone_cache = twin.new_cache(CAPACITY)
    if prefix:  # the cached prefix both schedules resume from: a plain prefill
        _run(twin, ids, prefix, cone=False, cache=cone_cache)
    cone_cache, cone_logits, state = _run(twin, ids, chunks, cone=True, cache=cone_cache,
                                          admitted=admitted)
    r0 = state.decoder_start
    assert state.encoded_until == r0 > admitted  # some rows really skipped the decoder
    assert _same(cone_logits, plain_logits)
    # Compressed (global) KV of every source and every row, and the encoder's window KV.
    for s in twin.sources:
        assert _same(cone_cache["global"][s][:n], plain["global"][s][:n])
    assert _same(cone_cache["window"][: D + 1, :n], plain["window"][: D + 1, :n])
    # Decoder window KV: exact from the frontier on, and never written below r0.
    frontier = state.geometry.tombstone_frontier(r0)
    assert _same(cone_cache["window"][:, frontier:n], plain["window"][:, frontier:n])
    assert torch.isnan(cone_cache["window"][D + 1:, admitted:r0]).all()
    # Decode continues bit for bit.
    assert _same(_decode_tail(twin, cone_cache, n), _decode_tail(twin, plain, n))


@pytest.mark.parametrize("n, sizes, admitted", CASES[:3])
def test_resume_from_every_published_prefix_matches_a_fresh_prefill(n, sizes, admitted):
    """Model the SWA tree: the commit publishes only if the planner says so; positions below the
    frontier are tombstones (NaN here), and a match must end one full live window past it.
    Each such resume equals a fresh plain prefill of the follow-up prompt, decode included."""
    twin, ids = Twin(), _prompt(n)
    cache, _, state = _run(twin, ids, _chunking(admitted, n, sizes), cone=True)
    insert_len = align_down(n, P)
    frontier = state.swa_commit_frontier(insert_len)
    assert frontier is not None and frontier <= state.geometry.retention_start(n)
    published = {k: v.clone() for k, v in cache.items() if k == "window"}
    published["window"][:, :frontier] = float("nan")  # tombstoned
    published["window"][:, insert_len:] = float("nan")  # never inserted
    published["global"] = {s: g.clone() for s, g in cache["global"].items()}
    for s in published["global"]:
        published["global"][s][insert_len:] = float("nan")
    matches = range(frontier + W, insert_len + 1, P)
    assert len(matches) >= 2
    for m in matches:
        follow = torch.cat([ids[:m], _prompt(90, seed=m)])
        fresh, fresh_logits, _ = _run(twin, follow, _chunking(0, len(follow), (64,)), cone=False)
        resumed = {"window": published["window"].clone(),
                   "global": {s: g.clone() for s, g in published["global"].items()}}
        resumed, resumed_logits, _ = _run(twin, follow, _chunking(m, len(follow), (32,)),
                                          cone=False, cache=resumed)
        assert _same(resumed_logits, fresh_logits), m
        assert _same(_decode_tail(twin, resumed, len(follow), 3),
                     _decode_tail(twin, fresh, len(follow), 3)), m


def _manual(twin, ids, r0, *, floor=True):
    """Encoder-only chunks up to ``r0``, then one decoder chunk from ``r0`` (any decoder start,
    not just the planner's)."""
    n = len(ids)
    cache = twin.new_cache(CAPACITY)
    logits = None
    for a, b in _chunking(0, r0, (32,)):
        twin.chunk(cache, ids, a, b, dc.ConeSegment(UID, a, b, dc.ENCODER, -1, D, L))
    segment = dc.ConeSegment(UID, r0, n, dc.DECODER, r0 if floor else 0, D, L)
    logits = twin.chunk(cache, ids, r0, n, segment)
    return cache, logits


def test_the_depth_bounds_are_tight():
    twin, n = Twin(seed=3), 200
    ids = _prompt(n, seed=3)
    plain, plain_logits, _ = _run(twin, ids, _chunking(0, n, (64,)), cone=False)
    g = _geometry()
    exact = n - 1 - g.logits_depth  # the latest decoder start for the final logits
    cache, logits = _manual(twin, ids, exact)
    assert _same(logits, plain_logits)
    _, late_logits = _manual(twin, ids, exact + 1)
    assert not torch.equal(late_logits, plain_logits)
    # Window KV: exact from r0 + window_depth on in every layer; the row before is not.
    r0 = 96
    cache, _ = _manual(twin, ids, r0)
    first = r0 + g.window_depth
    assert _same(cache["window"][:, first:n], plain["window"][:, first:n])
    assert not torch.equal(cache["window"][L - 1, first - 1], plain["window"][L - 1, first - 1])


def test_without_the_window_floor_edge_rows_read_never_written_kv():
    """The clamp is what keeps every computed row a function of written state. Without it the
    edge rows read never-written window KV. The kept outputs are still exact, because the edge
    is never read past the cone, but those rows compute garbage. On a GPU that garbage reaches
    expert routing and every collective."""
    twin, n = Twin(seed=4), 200
    ids = _prompt(n, seed=4)
    _, plain_logits, _ = _run(twin, ids, _chunking(0, n, (64,)), cone=False)
    r0 = 96
    clamped, logits = _manual(twin, ids, r0)
    unclamped, loose_logits = _manual(twin, ids, r0, floor=False)
    assert not torch.isnan(clamped["window"][:, r0:n]).any()
    assert torch.isnan(unclamped["window"][D + 2, r0:r0 + W]).any()
    assert _same(logits, plain_logits) and _same(loose_logits, plain_logits)


def test_plain_schedule_is_chunking_invariant():
    """Sanity of the twin itself: per-row arithmetic makes the plain prefill chunk-invariant."""
    twin, n = Twin(seed=5), 230
    ids = _prompt(n, seed=5)
    a, la, _ = _run(twin, ids, _chunking(0, n, (64,)), cone=False)
    b, lb, _ = _run(twin, ids, _chunking(0, n, (7, 41, 19)), cone=False)
    assert _same(la, lb)
    assert _same(a["window"][:, :n], b["window"][:, :n])
