"""Token-boundary overlap under the real ``overlap_loop``: run-ahead lifecycle is unchanged.

With ``FREETOKEN_DSV41_TOKEN_BOUNDARY_OVERLAP=1`` the DSV4 decode ``prepare_metadata`` no longer
blocks the host on the in-flight forward, so batch N+1 is genuinely queued behind batch N. The
host-side order was already run-ahead (N+1 is chosen, allocated and prepared before N drains);
only the GPU idle gap goes away. These tests drive the real ``overlap_loop`` /
``_process_last_data`` / abort handler with the real DSV4 backend ``prepare_metadata`` on every
scheduled batch, through C2 admission (bs 1 -> 2), EOS, abort mid-decode, abort mid-prefill,
table-index reuse and padded/eager batch shapes, and require the flag-on run to be identical to
the flag-off run: same launches, drains, publishes, acks, table indices, stream waits and
``last_indices`` values, with clean cache integrity at the end. The flag-on run additionally
asserts that no decode ``prepare_metadata`` call constructed, copied or synchronized anything.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.overrides import TorchFunctionMode

from freetoken.attention import dsv4_sparse
from freetoken.core import Req, get_global_ctx
from freetoken.message import AbortBackendMsg

from .test_overlap_early_publish import MAX_RUNNING, A, B, C, _Harness, _token

D = 4
GRAPH_BS = (1, 2)  # production decode graph buckets under C2
# Same forbidden set as tests/dsv4/test_dsv4_token_boundary_overlap.py.
_FORBIDDEN = {"tensor", "as_tensor", "cumsum", "cumsum_", "copy_", "cpu", "cuda", "item",
              "tolist", "arange", "full", "empty", "zeros", "ones", "synchronize", "nonzero",
              "equal", "index_select", "__bool__"}


class _Spy(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.calls.append((getattr(func, "__name__", str(func)), args, kwargs))
        return func(*args, **kwargs)


def _assert_no_device_traffic(calls):
    assert not [c[0] for c in calls if c[0] in _FORBIDDEN], calls
    for name, args, kwargs in calls:
        if name == "to":
            assert "device" not in kwargs and not any(
                isinstance(a, (torch.device, str)) for a in args[1:])


class _BackendHarness(_Harness):
    """The overlap harness, plus the DSV4 backend's prepare_metadata on every batch."""

    def __init__(self, backend, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.dummy = Req(input_ids=torch.tensor([0], dtype=torch.int32), table_idx=MAX_RUNNING,
                         cached_len=0, output_len=1, uid=-1, sampling_params=None,
                         cache_handle=None)
        self.metadata: list[tuple] = []
        self.tables: list[tuple] = []

    def _schedule(self):
        forward_input = super()._schedule()
        if forward_input is None:
            return None
        batch = forward_input.batch
        size = batch.size
        padded = (next(b for b in GRAPH_BS if b >= size)
                  if batch.is_decode and size <= GRAPH_BS[-1] else size)
        batch.padded_reqs = batch.reqs + [self.dummy] * (padded - size)
        if batch.is_decode:
            batch.active_table_idx = torch.tensor(
                [r.table_idx for r in batch.padded_reqs], dtype=torch.int64)
        spy = _Spy()
        with spy:
            self.backend.prepare_metadata(batch)
        md = batch.attn_metadata
        cache = getattr(self.backend, "_decode_last", None)
        if batch.is_decode:
            assert torch.equal(md.last_indices, torch.arange(padded, dtype=torch.int32))
            if cache is not None:
                assert md.last_indices.data_ptr() == cache.data_ptr()
                _assert_no_device_traffic(spy.calls)
        expected = torch.tensor([r.extend_len for r in batch.padded_reqs],
                                dtype=torch.int32).cumsum_(0) - 1
        assert torch.equal(md.last_indices, expected)
        self.metadata.append((batch.phase, padded, tuple(md.last_indices.tolist()),
                              md.segments))
        self.tables.append(tuple((r.uid, r.table_idx) for r in batch.reqs))
        return forward_input


def _run(monkeypatch, enabled: bool):
    if enabled:
        monkeypatch.setenv(dsv4_sparse.TOKEN_BOUNDARY_OVERLAP_ENV, "1")
    else:
        monkeypatch.delenv(dsv4_sparse.TOKEN_BOUNDARY_OVERLAP_ENV, raising=False)
    h = _BackendHarness.__new__(_BackendHarness)
    _Harness.__init__(h, prefill_budget=8, eos=(_token(C, 3),))
    ctx = get_global_ctx()
    saved = getattr(ctx, "kv_cache", None)
    ctx.kv_cache = SimpleNamespace(device=torch.device("cpu"))
    try:
        backend = dsv4_sparse.DSV4SparseAttnBackend(
            SimpleNamespace(dsv4_args=SimpleNamespace(window_size=128)))
        _BackendHarness.__init__(h, backend, prefill_budget=8, eos=(_token(C, 3),))
        assert (backend._decode_last is not None) is enabled

        data = h.step(None, ("user", A, 8, 14))       # A: one-chunk prefill, then bs=1 decode
        for _ in range(3):
            data = h.step(data)
        data = h.step(data, ("user", B, 8, 10))       # C2 admission -> bs=2 decode
        for _ in range(3):
            data = h.step(data)
        b_table = h.reqs[B].table_idx
        data = h.step(data, AbortBackendMsg(uid=B))   # abort mid-decode (B in flight)
        data = h.step(data, ("user", C, 8, 16))       # C reuses B's table row; EOS at token 3
        for _ in range(4):
            data = h.step(data)
        data = h.step(data, ("user", D, 24, 4))       # chunked prefill (3 chunks of 8)
        data = h.step(data)
        data = h.step(data, AbortBackendMsg(uid=D))   # abort mid-prefill (chunk in flight)
        h.run_to_idle(data)
        return h, b_table
    finally:
        ctx.kv_cache = saved


def test_runahead_lifecycle_is_identical_with_the_flag(monkeypatch):
    off, b_table_off = _run(monkeypatch, False)
    on, b_table_on = _run(monkeypatch, True)

    # The flag changes nothing the host can observe: order, replies, waits, tables, metadata.
    assert on.log == off.log
    assert on.wait_calls_per_iter == off.wait_calls_per_iter
    assert on.tables == off.tables and b_table_on == b_table_off
    assert on.metadata == off.metadata

    for h in (off, on):
        h.assert_clean_finish({A: (14, "length"), C: (3, "stop")})
        assert [e for e in h.log if e[0] == "ack"] == [("ack", B), ("ack", D)]
        assert h.reqs[B].table_idx == -1 and h.reqs[D].table_idx == -1
        # C was admitted onto B's freed table row while A kept decoding.
        c_tables = {t for batch in h.tables for uid, t in batch if uid == C}
        assert c_tables == {b_table_on}
        # Every decode batch was bs 1 or 2 (graph-padded); bs=2 genuinely occurred.
        decode = [m for m in h.metadata if m[0] == "decode"]
        assert {m[1] for m in decode} == {1, 2}
        assert all(m[2] == tuple(range(m[1])) for m in decode)
        # Nothing published for B after its abort, nor for D at all.
        ack_b = h.index(lambda e: e == ("ack", B))
        assert not [e for e in h.log[ack_b:] if e[0] == "publish" and e[1] == B]
        assert not h.publishes(D)


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-off", "flag-on"])
def test_decode_only_overlap_keeps_one_batch_in_flight(monkeypatch, enabled):
    """Steady bs=2 decode: N+1 launches before N drains, exactly one batch overlapped, and the
    scheduler's two stream-ordering waits per iteration are unchanged."""
    if enabled:
        monkeypatch.setenv(dsv4_sparse.TOKEN_BOUNDARY_OVERLAP_ENV, "1")
    else:
        monkeypatch.delenv(dsv4_sparse.TOKEN_BOUNDARY_OVERLAP_ENV, raising=False)
    h = _BackendHarness.__new__(_BackendHarness)
    _Harness.__init__(h)
    ctx = get_global_ctx()
    saved = getattr(ctx, "kv_cache", None)
    ctx.kv_cache = SimpleNamespace(device=torch.device("cpu"))
    try:
        backend = dsv4_sparse.DSV4SparseAttnBackend(
            SimpleNamespace(dsv4_args=SimpleNamespace(window_size=128)))
        _BackendHarness.__init__(h, backend)
        data = h.step(None, ("user", A, 16, 12), ("user", B, 16, 12))
        data = h.step(data)
        start, start_iter = len(h.log), len(h.wait_calls_per_iter)
        for _ in range(6):
            data = h.step(data)
        events = [e for e in h.log[start:] if e[0] in ("launch", "drain")]
        assert [e[0] for e in events] == ["launch", "drain"] * 6
        assert all(e[3] == 1 for e in events if e[0] == "launch")
        assert h.wait_calls_per_iter[start_iter:] == [2] * 6
        h.run_to_idle(data)
        h.assert_clean_finish({A: (12, "length"), B: (12, "length")})
    finally:
        ctx.kv_cache = saved
