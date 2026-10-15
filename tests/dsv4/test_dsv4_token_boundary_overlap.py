"""Opt-in DSV4 token-boundary overlap (``FREETOKEN_DSV41_TOKEN_BOUNDARY_OVERLAP=1``).

``prepare_metadata`` runs on the scheduler stream after the scheduler has made that stream wait
on the in-flight forward. The original decode ``last_indices`` was ``torch.tensor(...,
device=cuda).cumsum_(0) - 1``: a pageable H2D, so a ``cudaStreamSynchronize`` that held the host
until the previous token's forward finished. With the flag on, decode takes a view of one cached,
never-written ``arange`` instead. These CPU checks pin three things:

* the values (and dtype) are exactly what the original expression produces, for decode bs=1/2
  with the engine's dummy padding row, while prefill segments are untouched;
* the optimized decode path issues no tensor construction, copy, scan or host read -- a
  TorchFunctionMode spy sees every torch call, and is positively controlled against the
  original path, which it must catch;
* anything outside the proven decode contract falls back to the original construction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.overrides import TorchFunctionMode

from freetoken.attention import dsv4_sparse
from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx

ENV = dsv4_sparse.TOKEN_BOUNDARY_OVERLAP_ENV
MRR = 4  # the engine's dummy row is table_idx == max_running_req


class _Pool(SimpleNamespace):
    """Just enough pool for prepare_metadata / capture staging on CPU."""


def _ctx_with_pool():
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=128)
        set_global_ctx(ctx)
    pool = _Pool(device=torch.device("cpu"))
    pool.full_loc_map = torch.arange((MRR + 1) * 256, dtype=torch.int32).view(MRR + 1, 256)
    ctx.kv_cache = pool
    return ctx


def _backend(monkeypatch, enabled: bool, cls=None):
    if enabled:
        monkeypatch.setenv(ENV, "1")
    else:
        monkeypatch.delenv(ENV, raising=False)
    _ctx_with_pool()
    if cls is None:
        return dsv4_sparse.DSV4SparseAttnBackend(
            SimpleNamespace(dsv4_args=SimpleNamespace(window_size=128))
        )
    return cls(SimpleNamespace(dsv41_args=SimpleNamespace(window_size=128)))


def _req(uid, table_idx, prompt_len, cached_len=0):
    return Req(input_ids=torch.zeros(prompt_len, dtype=torch.int32), table_idx=table_idx,
               cached_len=cached_len, output_len=64, uid=uid,
               sampling_params=SamplingParams(), cache_handle=None)


def _decoding(uid, table_idx, prompt_len=7, decode_steps=0):
    """A request in the state the engine leaves it: prompt forwarded (complete_one), then
    ``decode_steps`` decode forwards (complete_one each)."""
    r = _req(uid, table_idx, prompt_len)
    r.complete_one()  # the prompt's prefill forward
    for _ in range(decode_steps):
        r.complete_one()
    return r


def _dummy():
    # engine.py: the persistent padding request (one prompt token, never advanced).
    return Req(input_ids=torch.tensor([0], dtype=torch.int32), table_idx=MRR, cached_len=0,
               output_len=1, uid=-1, sampling_params=None, cache_handle=None)


def _decode_batch(reqs, padded_size=None):
    batch = Batch(reqs=list(reqs), phase="decode")
    pad = (padded_size or len(reqs)) - len(reqs)
    batch.padded_reqs = batch.reqs + [_dummy()] * pad
    batch.active_table_idx = torch.tensor([r.table_idx for r in batch.padded_reqs],
                                          dtype=torch.int64)
    batch.positions = torch.tensor([r.cached_len for r in batch.padded_reqs], dtype=torch.int32)
    return batch


def _original_last(batch):
    """The production expression, verbatim, on the host."""
    return torch.tensor(
        [r.extend_len for r in batch.padded_reqs], dtype=torch.int32
    ).cumsum_(0) - 1


class _Spy(TorchFunctionMode):
    """Records every torch function/method call made while active."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.calls.append((getattr(func, "__name__", str(func)), args, kwargs))
        return func(*args, **kwargs)


# Construction, copies, scans and host reads: any of these on the decode boundary is the
# blocking pattern this change removes (or a new one).
_FORBIDDEN = {"tensor", "as_tensor", "cumsum", "cumsum_", "copy_", "cpu", "cuda", "item",
              "tolist", "arange", "full", "empty", "zeros", "ones", "synchronize", "nonzero",
              "equal", "index_select", "__bool__"}


def _assert_no_device_traffic(calls):
    names = [c[0] for c in calls]
    bad = [n for n in names if n in _FORBIDDEN]
    assert not bad, f"optimized decode prepare_metadata issued {bad}"
    for name, args, kwargs in calls:
        if name == "to":
            # Only the pre-existing dtype-only cast of active_table_idx (_table_rows).
            assert "device" not in kwargs and not any(
                isinstance(a, (torch.device, str)) for a in args[1:]
            ), f"device move in optimized decode path: {args[1:]} {kwargs}"


# ----- values ---------------------------------------------------------------------------------

@pytest.mark.parametrize("enabled", [False, True], ids=["flag-off", "flag-on"])
@pytest.mark.parametrize(
    "rows,padded",
    [((2,), 1), ((2,), 2), ((2, 3), 2), ((0, 1, 3), 3)],
    ids=["bs1", "bs1-padded-to-2", "bs2", "bs3-eager"],
)
def test_decode_last_indices_are_exact(monkeypatch, enabled, rows, padded):
    backend = _backend(monkeypatch, enabled)
    reqs = [_decoding(i, t, prompt_len=5 + i, decode_steps=i) for i, t in enumerate(rows)]
    batch = _decode_batch(reqs, padded)
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    expected = _original_last(batch)
    assert torch.equal(expected, torch.arange(padded, dtype=torch.int32))
    assert md.last_indices.dtype == torch.int32 and md.last_indices.shape == (padded,)
    assert torch.equal(md.last_indices, expected)
    assert torch.equal(md.get_last_indices(len(rows)), expected[: len(rows)])
    assert md.full_snap is None and md.segments is None
    assert torch.equal(md.table_rows, batch.active_table_idx)
    if enabled:
        cache = backend._decode_last
        assert md.last_indices.data_ptr() == cache.data_ptr()   # a view, not a new tensor
    else:
        assert backend._decode_last is None


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-off", "flag-on"])
def test_prefill_segments_and_last_indices_unchanged(monkeypatch, enabled):
    backend = _backend(monkeypatch, enabled)
    reqs = [
        _req(0, 2, 300, cached_len=256),   # radix hit / chunk continuation
        _req(1, 1, 5),                     # cold
        _req(2, 3, 129, cached_len=1),
    ]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    spy = _Spy()
    with spy:
        backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert md.segments == [(0, 44, 2, 256), (44, 5, 1, 0), (49, 128, 3, 1)]
    assert torch.equal(md.last_indices, torch.tensor([43, 48, 176], dtype=torch.int32))
    assert torch.equal(md.last_indices, _original_last(batch))
    assert md.table_rows is None and md.full_snap is None
    # Prefill keeps the original construction in both modes.
    assert "tensor" in [c[0] for c in spy.calls]
    if enabled:
        assert md.last_indices.data_ptr() != backend._decode_last.data_ptr()


# ----- no blocking construction on the optimized decode path -----------------------------------

def test_spy_catches_the_original_blocking_construction(monkeypatch):
    """Positive control: the flag-off decode path is the pattern the trace attributed (pageable
    torch.tensor(..., device=) + cumsum_). The spy must see it, or the next test proves nothing."""
    backend = _backend(monkeypatch, False)
    batch = _decode_batch([_decoding(0, 2)], 2)
    spy = _Spy()
    with spy:
        backend.prepare_metadata(batch)
    names = [c[0] for c in spy.calls]
    assert "tensor" in names and "cumsum_" in names
    tensor_call = next(c for c in spy.calls if c[0] == "tensor")
    assert tensor_call[2].get("device") == backend.device
    with pytest.raises(AssertionError, match="optimized decode prepare_metadata issued"):
        _assert_no_device_traffic(spy.calls)


@pytest.mark.parametrize("rows,padded", [((2,), 1), ((2,), 2), ((2, 3), 2)],
                         ids=["bs1", "bs1-padded-to-2", "bs2"])
def test_optimized_decode_issues_no_construction_copy_or_sync(monkeypatch, rows, padded):
    backend = _backend(monkeypatch, True)
    batch = _decode_batch([_decoding(i, t) for i, t in enumerate(rows)], padded)
    spy = _Spy()
    with spy:
        backend.prepare_metadata(batch)
    _assert_no_device_traffic(spy.calls)


# ----- cache stability, fallbacks ----------------------------------------------------------------

def test_cache_is_address_stable_and_never_written(monkeypatch):
    backend = _backend(monkeypatch, True)
    cache = backend._decode_last
    ptr, version = cache.data_ptr(), cache._version
    reference = torch.arange(cache.shape[0], dtype=torch.int32)
    reqs = [_decoding(0, 1), _decoding(1, 2)]
    for step in range(300):  # bs 1 <-> 2 churn: EOS / abort / admission reshape the batch
        live = reqs[: 1 + step % 2]
        batch = _decode_batch(live, 2 if step % 3 else len(live))
        backend.prepare_metadata(batch)
        assert torch.equal(batch.attn_metadata.last_indices,
                           torch.arange(batch.padded_size, dtype=torch.int32))
        for r in live:
            r.complete_one()
    assert backend._decode_last is cache and cache.data_ptr() == ptr
    assert cache._version == version and torch.equal(cache, reference)


def test_non_unit_extend_row_falls_back_to_the_original_values(monkeypatch):
    """Not reachable in decode today (speculative rollback re-arms extend_len 1); if it ever
    were, the values must still be the original cumsum, not an arange."""
    backend = _backend(monkeypatch, True)
    odd = _decoding(0, 1)
    odd.complete_n(3)            # cached_len -> device_len, device_len += 3: extend_len 3
    assert odd.extend_len == 3
    batch = _decode_batch([odd, _decoding(1, 2)], 2)
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert torch.equal(md.last_indices, torch.tensor([2, 3], dtype=torch.int32))
    assert md.last_indices.data_ptr() != backend._decode_last.data_ptr()


def test_batch_wider_than_the_cache_falls_back(monkeypatch):
    backend = _backend(monkeypatch, True)
    n = backend._decode_last.shape[0] + 1
    batch = _decode_batch([_decoding(i, 0) for i in range(n)], n)
    backend.prepare_metadata(batch)
    assert torch.equal(batch.attn_metadata.last_indices, torch.arange(n, dtype=torch.int32))
    assert batch.attn_metadata.last_indices.data_ptr() != backend._decode_last.data_ptr()


@pytest.mark.parametrize("value", ["", "0", "false", "off", "2"])
def test_flag_parsing_defaults_off(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert dsv4_sparse.decode_last_cache(torch.device("cpu")) is None


# ----- graph replay / capture and the V4.1 subclass ----------------------------------------------

def test_replay_and_capture_still_use_the_captured_buffer(monkeypatch):
    """Graph replay replaces the prepare_metadata metadata with the capture buffer's; the flag
    must not leak the cached view into capture/replay addressing."""
    backend = _backend(monkeypatch, True)
    backend.init_capture_graph(max_seq_len=256, bs_list=[1, 2])
    cap = backend.capture
    batch = _decode_batch([_decoding(0, 2)], 2)
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.last_indices.data_ptr() == backend._decode_last.data_ptr()
    backend.prepare_for_replay(batch)
    md = batch.attn_metadata
    assert md.last_indices.data_ptr() == cap.last_indices.data_ptr()
    assert torch.equal(md.last_indices, torch.arange(2, dtype=torch.int32))
    pool = get_global_ctx().kv_cache
    assert torch.equal(md.full_snap[0], pool.full_loc_map[2].to(torch.int64))
    assert torch.equal(md.full_snap[1], pool.full_loc_map[MRR].to(torch.int64))

    capture_batch = Batch(reqs=[_dummy()] * 2, phase="decode")
    capture_batch.padded_reqs = capture_batch.reqs
    backend.prepare_for_capture(capture_batch)
    assert capture_batch.attn_metadata.last_indices.data_ptr() == cap.last_indices.data_ptr()


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-off", "flag-on"])
def test_v41_backend_shares_the_decode_path(monkeypatch, enabled):
    from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend

    backend = _backend(monkeypatch, enabled, cls=DSV41SparseAttnBackend)
    assert (backend._decode_last is not None) is enabled
    batch = _decode_batch([_decoding(0, 1), _decoding(1, 3)], 2)
    spy = _Spy()
    with spy:
        backend.prepare_metadata(batch)
    assert torch.equal(batch.attn_metadata.last_indices, _original_last(batch))
    if enabled:
        _assert_no_device_traffic(spy.calls)
    else:
        assert "tensor" in [c[0] for c in spy.calls]


def test_eager_decode_snapshot_is_unaffected(monkeypatch):
    """A graph-ineligible decode keeps prepare_metadata's metadata; its lazy snapshot still
    copies the live rows."""
    backend = _backend(monkeypatch, True)
    batch = _decode_batch([_decoding(0, 2), _decoding(1, 3)], 2)
    backend.prepare_metadata(batch)
    pool = get_global_ctx().kv_cache
    with get_global_ctx().forward_batch(batch):
        snap = backend.snapshot()
    assert torch.equal(snap, pool.full_loc_map.index_select(0, torch.tensor([2, 3])).long())
