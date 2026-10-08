"""C2: optional pre-transposed CPU layout bank (FREETOKEN_DSV41_CPU_LAYOUT_BANK).

Pure ownership/memory/ledger tests run anywhere.  Extension tests need the rebuilt
_cpu_moe (``set_dsfp4_layout``/``dsfp4_layout_copy``) on an AVX2 host and use eager
CPU tasks only; the final mapped-handoff graph test additionally needs CUDA.
"""
import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe import cpu_executor as cx

GIB = 2**30
R = lambda offset, count: SimpleNamespace(global_offset=offset, local_count=count)  # noqa: E731
# EP4: storage 0:112, 32:208, 144:208, 288:96; decode 32/144/144/64; prefill 112/128/112/32.
RANKS = {
    0: (R(0, 112), R(0, 32), R(0, 112)),
    1: (R(32, 208), R(32, 144), R(112, 128)),
    2: (R(144, 208), R(176, 144), R(240, 112)),
    3: (R(288, 96), R(320, 64), R(352, 32)),
}
PER_EXPERT_ALL_LAYERS = 40 * 18_800_640


def _pack(codes):
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous().to(torch.uint8)


def _dsfp4_cache(L, E, H=5120, I=2304, seed=0):
    torch.manual_seed(seed)

    def rows(out, inp):
        packed = _pack(torch.randint(0, 16, (L * E, out, inp), dtype=torch.uint8))
        scale = torch.randint(122, 128, (L * E, out, inp // 32), dtype=torch.uint8)
        return packed, scale.view(torch.float8_e8m0fnu).contiguous()

    gup, gus = rows(2 * I, H)
    dnp, dns = rows(H, I)
    return SimpleNamespace(
        quant_format="ds_fp4", num_layers=L, num_experts=E, decode_target="cpu", cpu_executor=None,
        bank_sources={"gate_up_packed": list(gup.split(E)), "gate_up_scale": list(gus.split(E)),
                      "down_packed": list(dnp.split(E)), "down_scale": list(dns.split(E))})


# ---------------------------------------------------------------- ownership and memory

@pytest.mark.parametrize("rank,decode,decode_only", [
    (1, (0, 144), (0, 80)), (2, (32, 176), (32, 96)), (3, (32, 96), (32, 64)), (0, (0, 32), None)])
def test_layout_ranges_follow_current_ep4_ownership(rank, decode, decode_only):
    storage, dec, pre = RANKS[rank]
    assert cx.dsv41_cpu_layout_range("0", storage, dec, pre) is None
    assert cx.dsv41_cpu_layout_range("decode", storage, dec, pre) == decode
    assert cx.dsv41_cpu_layout_range("decode_only", storage, dec, pre) == decode_only


def test_layout_range_rejects_bad_modes_and_split_subsets():
    with pytest.raises(ValueError):
        cx.dsv41_cpu_layout_range("1", R(0, 10), R(0, 10), R(0, 0))
    with pytest.raises(ValueError, match="not one interval"):
        cx.dsv41_cpu_layout_range("decode_only", R(0, 30), R(0, 30), R(10, 5))
    with pytest.raises(ValueError, match="outside storage"):
        cx.dsv41_cpu_layout_range("decode", R(10, 5), R(0, 30), R(0, 0))


def test_whole_plan_and_projected_bytes_match_the_audit():
    plan = cx.dsv41_cpu_layout_plan("decode", {r: RANKS[r] for r in (1, 2)})
    assert plan == {1: (0, 144), 2: (32, 176)}
    total = sum(stop - start for start, stop in plan.values()) * PER_EXPERT_ALL_LAYERS
    assert total / GIB == pytest.approx(201.71, abs=0.01)
    only = cx.dsv41_cpu_layout_plan("decode_only", {r: RANKS[r] for r in (1, 2)})
    assert sum(b - a for a, b in only.values()) * PER_EXPERT_ALL_LAYERS / GIB == pytest.approx(100.85, abs=0.01)


def test_layout_bytes_are_exact_v41_rows():
    E, L = 6, 2
    shapes = {"gate_up_packed": (4608, 2560), "gate_up_scale": (4608, 160),
              "down_packed": (5120, 1152), "down_scale": (5120, 72)}
    banks = {n: [torch.empty((E, *s), dtype=torch.uint8) for _ in range(L)] for n, s in shapes.items()}
    assert cx.dsfp4_layout_bytes(banks, 1, 4) == 3 * L * 18_800_640


def test_floor_is_safe_in_either_build_order_without_spurious_refusal():
    own = peer = 144 * PER_EXPERT_ALL_LAYERS          # full decode, 100.85 GiB per rank
    loaded = int(331.1 * GIB)                          # loaded-state MemAvailable before C2
    floor = 64 * GIB
    # First builder charges the peer's whole unbuilt plan; the second sees the
    # first copy already inside MemAvailable and charges nothing extra for it.
    assert cx.layout_floor_violation(loaded, own, peer, 0, floor) is None
    assert cx.layout_floor_violation(loaded - peer, own, peer, peer, floor) is None
    # The old per-rank "need x ranks" rule would have refused the second builder.
    assert loaded - peer - 2 * own < floor
    # A genuinely short host still refuses, before any allocation.
    assert cx.layout_floor_violation(int(250 * GIB), own, peer, 0, floor)
    assert cx.layout_floor_violation(loaded - peer, own, peer, 0, floor)  # peer built but unregistered


def test_ledger_counts_only_live_peers(tmp_path, monkeypatch):
    ledger = cx._LayoutLedger(str(tmp_path))
    with ledger:
        assert ledger.peers_built() == 0
        ledger.record(123)
        assert ledger.peers_built() == 0  # our own entry is never a peer
        parent = os.getppid()
        (tmp_path / f"{parent}-{cx._process_start(parent)}.bytes").write_text("456")
        (tmp_path / f"{parent}-1.bytes").write_text("999")        # pid reused: start differs
        (tmp_path / "999999999-5.bytes").write_text("777")        # dead pid
        (tmp_path / "junk.bytes").write_text("1")
        assert ledger.peers_built() == 456


# ---------------------------------------------------------------- extension

def _extension():
    from freetoken.kernel import _cpu_moe

    if not hasattr(_cpu_moe, "dsfp4_layout_copy") or not hasattr(_cpu_moe.CpuMoeExecutor, "set_dsfp4_layout"):
        pytest.skip("requires a _cpu_moe rebuilt with the C2 layout bank")
    return _cpu_moe


@pytest.mark.parametrize("rows,row_bytes", [(8, 16), (16, 2560), (4608, 160), (24, 1152), (8, 72)])
def test_layout_copy_is_the_eight_row_byte_transpose(rows, row_bytes):
    ext = _extension()
    src = torch.randint(0, 256, (rows, row_bytes), dtype=torch.uint8)
    dst = torch.empty_like(src)
    ext.dsfp4_layout_copy(dst.data_ptr(), src.data_ptr(), rows, row_bytes, 4)
    expected = src.view(rows // 8, 8, row_bytes).transpose(1, 2).contiguous().view(rows, row_bytes)
    assert torch.equal(dst, expected)
    with pytest.raises(Exception):
        ext.dsfp4_layout_copy(dst.data_ptr(), src.data_ptr(), rows - 1 if rows > 8 else 7, row_bytes, 1)


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_LAYOUT_LEDGER", str(tmp_path / "ledger"))
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_LAYOUT_MIN_FREE_GIB", "0")
    return tmp_path / "ledger"


def _executor(cache, layout=None, device="cpu", **kw):
    return cx.CpuMoeExecutor(
        cache, top_k=6, activation="silu", apply_router_weight_on_input=False, num_threads=2,
        max_tokens=1, device=torch.device(device), swiglu_limit=10., dsfp4_activation_block_size=32,
        return_route_outputs=True, dsfp4_gpu_order=True, layout_experts=layout, **kw)


_CPU_TASKS: dict = {}


def _run(ex, layer, x, ids, w):
    """Eager task on plain CPU buffers (no CUDA pinned allocation needed)."""
    key = (id(ex), layer)
    if key not in _CPU_TASKS:
        io = dict(x=torch.zeros(1, ex.H, dtype=torch.bfloat16), ids=torch.zeros(1, ex.top_k, dtype=torch.int32),
                  w=torch.zeros(1, ex.top_k), y=torch.zeros(1, ex.top_k, ex.H, dtype=torch.bfloat16))
        task = ex._ext.create_task(layer, 1, io["x"].data_ptr(), io["ids"].data_ptr(),
                                   io["w"].data_ptr(), io["y"].data_ptr())
        _CPU_TASKS[key] = (ex, io, task)  # keeps buffers and executor alive with the task
    _, io, task = _CPU_TASKS[key]
    io["x"].copy_(x)
    io["ids"].copy_(ids)
    io["w"].copy_(w)
    ex._ext.run_task(task)
    return io["y"].clone()


def test_layout_tasks_are_bitwise_native_and_partial_coverage_falls_back(ledger_dir, capfd):
    _extension()
    cache = _dsfp4_cache(2, 4, seed=2026)
    native, layout = _executor(cache), _executor(cache, layout=(1, 3))  # experts 1,2 covered
    assert layout.layout_experts == (1, 3)
    generator = torch.Generator().manual_seed(1008)
    cases = []  # (ids, expected class: None = no valid route, True = layout, False = fallback)
    for n_routes in (0, 1, 2, 3):
        ids = torch.full((1, 6), -1, dtype=torch.int32)
        ids[0, :n_routes] = torch.tensor([1, 2, 1][:n_routes], dtype=torch.int32)
        cases.append((ids, None if n_routes == 0 else True, None))
    for partial in ([1, 0], [3, 2, 1], [2, -1, 3]):
        ids = torch.full((1, 6), -1, dtype=torch.int32)
        ids[0, :len(partial)] = torch.tensor(partial, dtype=torch.int32)
        cases.append((ids, False, None))
    zero = torch.full((1, 6), -1, dtype=torch.int32)
    zero[0, :2] = torch.tensor([1, 0], dtype=torch.int32)
    cases.append((zero, False, 1))           # uncovered expert 0 carries weight 0: still computed
    inert = torch.full((1, 6), -1, dtype=torch.int32)
    inert[0, :2] = torch.tensor([1, 2], dtype=torch.int32)
    cases.append((inert, None, "all"))       # valid ids, all weights 0: empty task, not counted
    expected = [0, 0]
    for layer in (0, 1):
        for index, (ids, kind, zero_at) in enumerate(cases):
            x = (torch.randn(1, 5120, generator=generator) * .3).to(torch.bfloat16)
            w = torch.rand(1, 6, generator=generator) * .25 + .01
            w[ids < 0] = 0
            if zero_at == "all":
                w.zero_()
            elif zero_at is not None:
                w[0, zero_at] = 0
            reference = _run(native, layer, x, ids, w)
            actual = _run(layout, layer, x, ids, w)
            assert torch.equal(actual.view(torch.int16), reference.view(torch.int16)), (layer, index)
            assert torch.isfinite(actual).all()
            if kind is not None:
                expected[0 if kind else 1] += 1
    assert layout.layout_task_counts() == tuple(expected)
    assert native.layout_task_counts() == (0, 0)
    err = capfd.readouterr().err
    assert err.count("[cpu-layout] first task served from layout bank") == 1
    assert err.count("[cpu-layout] first native fallback") == 1


def test_no_valid_route_tasks_never_mark_served(ledger_dir, capfd):
    _extension()
    cache = _dsfp4_cache(1, 4, seed=5)
    layout = _executor(cache, layout=(0, 4))
    for _ in range(3):
        _run(layout, 0, torch.zeros(1, 5120, dtype=torch.bfloat16),
             torch.full((1, 6), -1, dtype=torch.int32), torch.zeros(1, 6))
    assert layout.layout_task_counts() == (0, 0)
    assert "[cpu-layout]" not in capfd.readouterr().err


def test_layout_preconditions_and_memory_floor(ledger_dir, monkeypatch):
    _extension()
    cache = _dsfp4_cache(1, 4, seed=7)
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_LAYOUT_MIN_FREE_GIB", "64")
    monkeypatch.setattr(cx, "_mem_available_bytes", lambda: 1 * GIB)
    with pytest.raises(RuntimeError, match="MemAvailable"):
        _executor(cache, layout=(0, 4))
    monkeypatch.setattr(cx, "_mem_available_bytes", lambda: 1 << 50)
    with pytest.raises(RuntimeError, match="still planned"):
        _executor(cache, layout=(0, 4), layout_peer_experts=1 << 40)
    with pytest.raises(ValueError, match="outside"):
        _executor(cache, layout=(2, 5))
    with pytest.raises(RuntimeError, match="GPU-order"):
        cx.CpuMoeExecutor(cache, top_k=6, activation="silu", apply_router_weight_on_input=False,
                          num_threads=2, max_tokens=1, device=torch.device("cpu"), swiglu_limit=10.,
                          dsfp4_activation_block_size=32, return_route_outputs=True,
                          dsfp4_gpu_order=False, layout_experts=(0, 4))
    ex = _executor(cache)
    with pytest.raises(Exception, match="all set or all zero"):
        ex._ext.set_dsfp4_layout(1, 0, 0, 0)


def test_layout_bank_is_default_off(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DSV41_CPU_LAYOUT_BANK", raising=False)
    import inspect

    from freetoken.engine.engine import Engine

    source = inspect.getsource(Engine._init_cpu_moe_executor)
    assert 'os.getenv("FREETOKEN_DSV41_CPU_LAYOUT_BANK", "0")' in source
    assert "layout_experts=layout_experts" in source
    assert cx.dsv41_cpu_layout_range("0", *RANKS[1]) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_layout_mapped_graph_replay_matches_gpu_oracle(ledger_dir, monkeypatch):
    """Production handoff (mapped flags, CUDA graph, fused prep) with every expert covered."""
    _extension()
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    monkeypatch.setenv("FREETOKEN_CPU_MOE_MAPPED_HANDOFF", "1")
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_FUSE_PREP", "1")
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_ROUTE_PARALLEL", "0")
    E, top_k = 4, 6
    cache = _dsfp4_cache(1, E, seed=7319)
    ex = cx.CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False, num_threads=4,
        max_tokens=1, device=torch.device("cuda"), swiglu_limit=10., dsfp4_activation_block_size=32,
        return_route_outputs=True, dsfp4_gpu_order=True, layout_experts=(0, E))
    if not ex._mapped_handoff:
        pytest.skip("mapped handoff requires CUDA stream memory operations")
    banks = [cache.bank_sources[k][0].cuda() for k in cx.LAYOUT_BANKS]
    x = torch.zeros(1, 5120, device="cuda", dtype=torch.bfloat16)
    ids = torch.full((1, top_k), -1, device="cuda", dtype=torch.int32)
    w = torch.zeros(1, top_k, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ex.decode_sync(ex.decode_submit(0, x, w, ids))
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = ex.decode_sync(ex.decode_submit(0, x, w, ids))
    torch.cuda.current_stream().wait_stream(stream)
    covered = 0
    for n in (1, 0, 4, 2, 0, 3, 6, 1):
        x.copy_(torch.randn_like(x) * .3)
        rid = torch.randint(E, ids.shape, device="cuda", dtype=torch.int32)
        ids.copy_(rid)
        w.copy_(torch.rand_like(w) * .25)
        w[:, n:] = 0
        reference = routed_experts_fp4(x, rid, w, *banks, 10., activation_block_size=32,
                                       return_route_outputs=True)
        graph.replay()
        torch.cuda.synchronize()
        if n:
            assert torch.equal(output.view(torch.int16), reference.view(torch.int16)), n
        else:
            # Empty task: the unchanged early return writes +0 while the GPU oracle
            # multiplies by zero weights (signed zeros).  Same as the existing
            # mapped-replay test's value-equality contract, and as the old binary.
            assert torch.equal(output, reference) and not output.any()
        covered += n > 0
        ex.raise_if_unhealthy()
    served, fallback = ex.layout_task_counts()
    assert fallback == 0 and served == covered
