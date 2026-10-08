"""C3: two-tile DS-FP4 CPU kernel (FREETOKEN_DSV41_CPU_X2) and bounded MWAITX warm
wait (FREETOKEN_DSV41_CPU_WARM_WAIT_US).  Both default off.

Extension tests need a _cpu_moe built with ``set_dsfp4_x2``/``set_warm_wait_us`` on
an AVX2 host and use eager CPU tasks only (no CUDA).  Every comparison is bytewise
against the unchanged rows8 path of the same binary, including NaN payloads,
infinities, signed zeros and subnormals.
"""
import gc
import os
import random
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe import cpu_executor as cx

H, I, TOPK = 5120, 2304, 6


def _pack(codes):
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous().to(torch.uint8)


def _dsfp4_cache(L, E, seed=0, scale_range=(122, 128)):
    torch.manual_seed(seed)

    def rows(out, inp):
        packed = _pack(torch.randint(0, 16, (L * E, out, inp), dtype=torch.uint8))
        scale = torch.randint(*scale_range, (L * E, out, inp // 32), dtype=torch.uint8)
        return packed, scale.view(torch.float8_e8m0fnu).contiguous()

    gup, gus = rows(2 * I, H)
    dnp, dns = rows(H, I)
    return SimpleNamespace(
        quant_format="ds_fp4", num_layers=L, num_experts=E, decode_target="cpu", cpu_executor=None,
        bank_sources={"gate_up_packed": list(gup.split(E)), "gate_up_scale": list(gus.split(E)),
                      "down_packed": list(dnp.split(E)), "down_scale": list(dns.split(E))})


def _extension():
    from freetoken.kernel import _cpu_moe

    if not hasattr(_cpu_moe.CpuMoeExecutor, "set_dsfp4_x2") or \
            not hasattr(_cpu_moe.CpuMoeExecutor, "set_warm_wait_us"):
        pytest.skip("requires a _cpu_moe rebuilt with C3")
    return _cpu_moe


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_LAYOUT_LEDGER", str(tmp_path / "ledger"))
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_LAYOUT_MIN_FREE_GIB", "0")
    monkeypatch.delenv("FREETOKEN_DSV41_CPU_X2", raising=False)
    monkeypatch.delenv("FREETOKEN_DSV41_CPU_WARM_WAIT_US", raising=False)
    return tmp_path / "ledger"


def _executor(cache, layout=None, threads=3, **kw):
    return cx.CpuMoeExecutor(
        cache, top_k=TOPK, activation="silu", apply_router_weight_on_input=False, num_threads=threads,
        max_tokens=1, device=torch.device("cpu"), swiglu_limit=10., dsfp4_activation_block_size=32,
        return_route_outputs=True, dsfp4_gpu_order=True, layout_experts=layout, **kw)


class Task:
    """Eager task on plain CPU buffers; owns its buffers and executor."""

    def __init__(self, ex, layer=0):
        self.ex = ex
        self.io = dict(x=torch.zeros(1, H, dtype=torch.bfloat16), ids=torch.zeros(1, TOPK, dtype=torch.int32),
                       w=torch.zeros(1, TOPK), y=torch.zeros(1, TOPK, H, dtype=torch.bfloat16))
        self.task = ex._ext.create_task(layer, 1, *[self.io[k].data_ptr() for k in ("x", "ids", "w", "y")])

    def run(self, x, ids, w):
        self.io["x"].copy_(x)
        self.io["ids"].copy_(ids)
        self.io["w"].copy_(w)
        self.io["y"].fill_(1.0)  # every output element must be rewritten
        self.ex._ext.run_task(self.task)
        return self.io["y"].clone()


def _bits(t):
    return t.view(torch.int16)


def _adversarial_x(generator, step):
    x = torch.randn(1, H, generator=generator) * (10.0 ** ((step * 7) % 41 - 20))
    specials = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, -1e-40, 3e38, -3e38])
    mask = torch.rand(1, H, generator=generator) < (0.0, 0.002, 0.05, 0.5)[step % 4]
    x[mask] = specials[torch.randint(len(specials), (int(mask.sum()),), generator=generator)]
    xb = x.to(torch.bfloat16)
    if step % 3 == 0:  # explicit BF16 NaN payloads of both signs
        pick = torch.randint(H, (6,), generator=generator)
        _bits(xb)[0, pick] = torch.tensor([0x7fc1, -0x3e, 0x7f81, -0x7f, 0x7fff, 0x7fa5], dtype=torch.int16)
    return xb


def _cases(E, generator):
    """(ids, weights, raw-activation flag) covering empty, partial, duplicate, invalid and NaN routes."""
    out = []
    for n in range(TOPK + 1):
        ids = torch.full((1, TOPK), -1, dtype=torch.int32)
        ids[0, :n] = torch.randint(E, (n,), generator=generator, dtype=torch.int32)
        out.append(ids)
    dup = torch.tensor([[1, 1, 2, -1, 1, 0]], dtype=torch.int32)
    bad = torch.tensor([[E, -5, 2, 1 << 20, -1, 3]], dtype=torch.int32)
    out += [dup, bad]
    cases = []
    for k, ids in enumerate(out):
        w = torch.rand(1, TOPK, generator=generator) * .3 + .01
        cases.append((ids, w, False))
        w2 = w.clone()
        w2[0, k % TOPK] = 0.0
        w2[0, (k + 1) % TOPK] = float("nan") if k % 2 else -0.0
        cases.append((ids, w2, True))
    return cases


# ---------------------------------------------------------------- C3a: two-tile kernel

@pytest.mark.parametrize("layout", [None, (0, 4), (1, 3)], ids=["native", "layout", "partial-layout"])
def test_x2_tasks_are_bitwise_rows8(ledger_dir, monkeypatch, capfd, layout):
    _extension()
    cache = _dsfp4_cache(2, 4, seed=3001)
    reference = Task(_executor(cache))
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_X2", "1")
    ex = _executor(cache, layout=layout)
    assert ex.x2 and ex.warm_wait_us == 0
    candidates = {layer: Task(ex, layer) for layer in (0, 1)}
    references = {0: reference, 1: Task(reference.ex, 1)}
    generator = torch.Generator().manual_seed(1009)
    active_tasks = 0
    for layer in (0, 1):
        for step, (ids, w, raw) in enumerate(_cases(4, generator)):
            x = _adversarial_x(generator, step) if raw else (torch.randn(1, H, generator=generator) * .3).to(
                torch.bfloat16)
            expected = references[layer].run(x, ids, w)
            actual = candidates[layer].run(x, ids, w)
            assert torch.equal(_bits(actual), _bits(expected)), (layout, layer, step)
            valid = (ids >= 0) & (ids < 4)
            active_tasks += bool((valid & (w != 0)).any())
    assert ex.x2_task_count() == active_tasks > 0
    assert reference.ex.x2_task_count() == 0
    err = capfd.readouterr().err
    assert err.count("[cpu-x2] first task served by two-tile kernel") == 1


def test_x2_extreme_scales_and_every_nibble(ledger_dir, monkeypatch):
    """Scale codes 0 (subnormal 2^-127), 1, 254 and 255 (clamped) and all 256 byte values."""
    _extension()
    cache = _dsfp4_cache(1, 3, seed=77, scale_range=(0, 256))
    for name in ("gate_up_scale", "down_scale"):
        s = cache.bank_sources[name][0].view(torch.uint8)
        s.view(-1)[:4096] = torch.tensor([0, 1, 254, 255] * 1024, dtype=torch.uint8)
    reference = Task(_executor(cache))
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_X2", "1")
    native, layout = Task(_executor(cache)), Task(_executor(cache, layout=(0, 3)))
    generator = torch.Generator().manual_seed(5)
    for step in range(12):
        ids = torch.tensor([[0, 1, 2, step % 3, -1, (step + 1) % 3]], dtype=torch.int32)
        w = torch.rand(1, TOPK, generator=generator) + 0.5
        x = _adversarial_x(generator, step) if step % 2 else (torch.randn(1, H, generator=generator)).to(
            torch.bfloat16)
        expected = reference.run(x, ids, w)
        assert torch.equal(_bits(native.run(x, ids, w)), _bits(expected)), step
        assert torch.equal(_bits(layout.run(x, ids, w)), _bits(expected)), step


def test_x2_signed_zero_rows(ledger_dir, monkeypatch):
    """All-(-0) activations against all-positive codes: every product is -0, so only an
    identical first-product/accumulation sequence reproduces the -0 outputs."""
    _extension()
    cache = _dsfp4_cache(1, 2, seed=23)
    for name in ("gate_up_packed", "down_packed"):
        cache.bank_sources[name][0][0].fill_(0x21)  # expert 0: codes 1 and 2 (+0.5, +1.0)
    reference = Task(_executor(cache))
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_X2", "1")
    native, layout = Task(_executor(cache)), Task(_executor(cache, layout=(0, 2)))
    ids = torch.tensor([[0, -1, -1, -1, -1, -1]], dtype=torch.int32)
    for sign in (-0.0, 0.0):
        x = torch.full((1, H), sign, dtype=torch.bfloat16)
        for w in (torch.full((1, TOPK), .5), torch.full((1, TOPK), -.5)):
            expected = reference.run(x, ids, w)
            assert not expected[0, 0].any()  # an all-zero route; only the zero signs vary
            assert torch.equal(_bits(native.run(x, ids, w)), _bits(expected)), (sign, float(w[0, 0]))
            assert torch.equal(_bits(layout.run(x, ids, w)), _bits(expected)), (sign, float(w[0, 0]))


def test_x2_preconditions(ledger_dir, monkeypatch):
    _extension()
    cache = _dsfp4_cache(1, 2, seed=9)
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_X2", "1")
    with pytest.raises(RuntimeError, match="GPU-order"):
        cx.CpuMoeExecutor(cache, top_k=TOPK, activation="silu", apply_router_weight_on_input=False,
                          num_threads=2, max_tokens=1, device=torch.device("cpu"), swiglu_limit=10.,
                          dsfp4_activation_block_size=32, return_route_outputs=True, dsfp4_gpu_order=False)
    monkeypatch.delenv("FREETOKEN_DSV41_CPU_X2")
    ex = _executor(cache)
    ex._ext.set_dsfp4_route_outputs(False)
    with pytest.raises(Exception, match="GPU-order route outputs"):
        ex._ext.set_dsfp4_x2(True)


def test_c3_is_default_off(ledger_dir):
    _extension()
    ex = _executor(_dsfp4_cache(1, 2, seed=11))
    assert ex.x2 is False and ex.warm_wait_us == 0
    assert ex._ext.warm_wait_cycles() == 0
    task = Task(ex)
    task.run(torch.ones(1, H, dtype=torch.bfloat16), torch.tensor([[0, 1, -1, -1, -1, -1]], dtype=torch.int32),
             torch.full((1, TOPK), .1))
    assert ex.x2_task_count() == 0


# ---------------------------------------------------------------- C3b: warm wait

def _run_with_deadline(fn, seconds=10.0):
    box = {}
    thread = threading.Thread(target=lambda: box.setdefault("value", fn()), daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "task hung"
    return box["value"]


def test_warm_wait_is_bitwise_across_window_boundaries(ledger_dir, monkeypatch):
    """Gaps inside, at and beyond a 2 ms window: warm doorbell, condvar fallback and the
    transitions between them must all produce the cold executor's bytes, without hangs."""
    _extension()
    cache = _dsfp4_cache(1, 4, seed=4242)
    cold = Task(_executor(cache, threads=4))
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_WARM_WAIT_US", "2000")
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_X2", "1")
    warm = Task(_executor(cache, threads=4))
    assert warm.ex.warm_wait_us == 2000 and warm.ex._ext.warm_wait_cycles() > 0
    rng = random.Random(77)
    generator = torch.Generator().manual_seed(78)
    for step in range(240):
        ids = torch.full((1, TOPK), -1, dtype=torch.int32)
        for j in range(rng.randrange(0, 4)):
            ids[0, j] = rng.randrange(4)
        w = torch.rand(1, TOPK, generator=generator) * .2 + .01
        x = (torch.randn(1, H, generator=generator) * .3).to(torch.bfloat16)
        expected = cold.run(x, ids, w)
        time.sleep(rng.choice((0.0, 0.0, 5e-5, 1.5e-3, 1.9e-3, 2.1e-3, 6e-3)))
        actual = _run_with_deadline(lambda: warm.run(x, ids, w))
        assert torch.equal(_bits(actual), _bits(expected)), step


def test_warm_wait_validation(ledger_dir):
    ext = _extension()
    ex = _executor(_dsfp4_cache(1, 2, seed=13))
    for bad in (-1, 100001):
        with pytest.raises(Exception, match="0..100000"):
            ex._ext.set_warm_wait_us(bad)
    ex._ext.set_warm_wait_us(0)
    ex._ext.set_warm_wait_us(1000)
    assert ex._ext.warm_wait_cycles() > 0
    task = Task(ex)
    task.run(torch.ones(1, H, dtype=torch.bfloat16), torch.tensor([[0, -1, -1, -1, -1, -1]], dtype=torch.int32),
             torch.full((1, TOPK), .1))
    with pytest.raises(Exception, match="before the first task"):
        ex._ext.set_warm_wait_us(500)
    assert hasattr(ext, "CpuMoeExecutor")


def test_warm_teardown_is_prompt(ledger_dir, monkeypatch):
    """Destroying an executor whose workers are warm-waiting (100 ms window) returns quickly."""
    _extension()
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_WARM_WAIT_US", "100000")
    task = Task(_executor(_dsfp4_cache(1, 2, seed=17), threads=4))
    task.run(torch.ones(1, H, dtype=torch.bfloat16), torch.tensor([[0, 1, -1, -1, -1, -1]], dtype=torch.int32),
             torch.full((1, TOPK), .1))
    started = time.monotonic()
    del task
    gc.collect()
    assert time.monotonic() - started < 2.0


def test_c3_env_refuses_a_stale_extension(ledger_dir, monkeypatch):
    """Either flag on a binary without the C3 setters fails loudly, never silently off."""
    ext = _extension()
    real = ext.CpuMoeExecutor
    hidden = ("set_warm_wait_us", "set_dsfp4_x2")

    class Stale:
        def __init__(self, *a, **k):
            object.__setattr__(self, "_inner", real(*a, **k))

        def __getattribute__(self, name):
            if name in hidden:
                raise AttributeError(name)
            return getattr(object.__getattribute__(self, "_inner"), name)

    for name in dir(real):  # class-level capability probes still see the other setters
        if not name.startswith("__") and name not in hidden:
            setattr(Stale, name, True)
    monkeypatch.setattr(ext, "CpuMoeExecutor", Stale)
    cache = _dsfp4_cache(1, 2, seed=19)
    for key, value in (("FREETOKEN_DSV41_CPU_WARM_WAIT_US", "1000"), ("FREETOKEN_DSV41_CPU_X2", "1")):
        monkeypatch.setenv(key, value)
        with pytest.raises(RuntimeError, match="rebuilt _cpu_moe"):
            _executor(cache)
        monkeypatch.delenv(key)
