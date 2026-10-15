"""GPU proof that the opt-in decode ``prepare_metadata`` does not wait on the in-flight forward.

Reproduces the scheduler's ordering: the engine stream is busy (a long ``_sleep``), the
scheduler stream ``wait_stream``s it, then ``prepare_metadata`` runs on the scheduler stream.
Flag off, the pageable H2D synchronizes the scheduler stream, so the engine stream has drained by
the time the call returns (positive control). Flag on, the call returns while the engine stream is
still busy. Opt-in only (``FREETOKEN_TOKEN_BOUNDARY_CUDA_TEST=1``): it needs an idle GPU.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import dsv4_sparse
from freetoken.core import Batch, Context, Req, get_global_ctx, set_global_ctx

pytestmark = pytest.mark.skipif(
    os.getenv("FREETOKEN_TOKEN_BOUNDARY_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="opt-in GPU test (FREETOKEN_TOKEN_BOUNDARY_CUDA_TEST=1)",
)

_SLEEP_CYCLES = 2_000_000_000  # ~1 s at GPU clocks: far longer than a host-side call


def _batch(device):
    reqs = []
    for uid, table_idx in enumerate((0, 1)):
        r = Req(input_ids=torch.zeros(5, dtype=torch.int32), table_idx=table_idx, cached_len=0,
                output_len=8, uid=uid, sampling_params=None, cache_handle=None)
        r.complete_one()
        reqs.append(r)
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.active_table_idx = torch.tensor([0, 1], dtype=torch.int64, device=device)
    return batch


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-off-control", "flag-on"])
def test_decode_prepare_metadata_does_not_wait_for_the_engine_stream(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv(dsv4_sparse.TOKEN_BOUNDARY_OVERLAP_ENV, "1")
    else:
        monkeypatch.delenv(dsv4_sparse.TOKEN_BOUNDARY_OVERLAP_ENV, raising=False)
    device = torch.device("cuda")
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=128)
        set_global_ctx(ctx)
    saved = getattr(ctx, "kv_cache", None)
    ctx.kv_cache = SimpleNamespace(device=device)
    try:
        backend = dsv4_sparse.DSV4SparseAttnBackend(
            SimpleNamespace(dsv4_args=SimpleNamespace(window_size=128)))
        batch = _batch(device)
        torch.cuda.synchronize()
        engine, scheduler = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(engine):
            torch.cuda._sleep(_SLEEP_CYCLES)      # the in-flight forward
        scheduler.wait_stream(engine)             # scheduler.py: before _schedule_next_batch
        with torch.cuda.stream(scheduler):
            backend.prepare_metadata(batch)
        engine_busy = not engine.query()
        torch.cuda.synchronize()
        assert torch.equal(batch.attn_metadata.last_indices.cpu(),
                           torch.arange(2, dtype=torch.int32))
        assert engine_busy is enabled
    finally:
        ctx.kv_cache = saved
