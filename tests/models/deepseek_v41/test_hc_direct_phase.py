"""Prefill-only HC direct-pre: gated by explicit execution phase, never by row count.

CPU/fake-CUDA coverage of the ``FREETOKEN_DSV41_HC_DIRECT_PRE`` modes. A fake CUDA tensor
records whether the FP32 staging copy happens and what the (stubbed) combine kernel receives,
so the routing matrix is checked without a GPU. GPU exactness lives in test_hc_direct.py.
"""
from __future__ import annotations

import ast
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import freetoken
from freetoken import core
from freetoken.kernel.triton.dsv4 import hc as kernel
from freetoken.models.deepseek_v41 import hc

MODEL = Path(freetoken.__file__).resolve().parent / "models/deepseek_v41/model.py"
DSPARK = Path(freetoken.__file__).resolve().parent / "models/deepseek_v41/dspark.py"


# ------------------------------------------------------------------ mode parsing

@pytest.mark.parametrize("value,mode", [(None, "off"), ("0", "off"), ("1", "all"),
                                        ("prefill", "prefill")])
def test_mode_parse(monkeypatch, value, mode):
    if value is None:
        monkeypatch.delenv("FREETOKEN_DSV41_HC_DIRECT_PRE", raising=False)
    else:
        monkeypatch.setenv("FREETOKEN_DSV41_HC_DIRECT_PRE", value)
    assert hc._direct_pre_mode() == mode


@pytest.mark.parametrize("value", ["", "2", "true", "on", "Prefill", "decode", " prefill"])
def test_unknown_modes_are_refused(monkeypatch, value):
    monkeypatch.setenv("FREETOKEN_DSV41_HC_DIRECT_PRE", value)
    with pytest.raises(ValueError, match="FREETOKEN_DSV41_HC_DIRECT_PRE"):
        hc._direct_pre_mode()


def test_default_import_state_is_off():
    # The test process does not set the variable: nothing is enabled.
    assert hc._DIRECT_PRE_MODE == "off" and not hc._DIRECT_PRE and not hc._DIRECT_PRE_PREFILL


# ------------------------------------------------------------------ fake CUDA routing

class _FakeStreams:
    def __init__(self, dtype, rows, log):
        self.dtype, self.rows, self.log = dtype, rows, log

    def float(self):
        self.log.append("staged-fp32-copy")
        return _FakeStreams(torch.float32, self.rows, self.log)


class _FakeCudaX:
    is_cuda = True

    def __init__(self, rows, dtype, log, hc_mult=4, dim=8):
        self.shape = (rows, hc_mult, dim)
        self.dtype, self.rows, self.log = dtype, rows, log

    def reshape(self, *shape):
        assert shape == (-1, self.shape[1], self.shape[2])
        return _FakeStreams(self.dtype, self.rows, self.log)


class _FakePre:
    def reshape(self, *shape):
        return self


class _FakeResult:
    def view(self, *shape):
        return ("result", shape)


def _route(monkeypatch, *, mode, prefill, dtype, rows):
    monkeypatch.setattr(hc, "_DIRECT_PRE", mode == "all")
    monkeypatch.setattr(hc, "_DIRECT_PRE_PREFILL", mode == "prefill")
    log, seen = [], []

    def combine(streams, pre, out_dtype):
        seen.append(streams.dtype)
        assert out_dtype == dtype
        return _FakeResult()

    monkeypatch.setattr(kernel, "hc_pre_combine", combine)
    result = hc.hc_pre(_FakeCudaX(rows, dtype, log), _FakePre(), prefill=prefill)
    assert result == ("result", (rows, 8))
    return ("staged" if log else "direct"), seen[0]


MATRIX = list(itertools.product(("off", "all", "prefill"), (False, True),
                                (torch.bfloat16, torch.float16, torch.float32), (1, 2, 16, 8192)))


@pytest.mark.parametrize("mode,prefill,dtype,rows", MATRIX)
def test_direct_input_depends_only_on_mode_phase_and_dtype(monkeypatch, mode, prefill, dtype, rows):
    path, kernel_dtype = _route(monkeypatch, mode=mode, prefill=prefill, dtype=dtype, rows=rows)
    half = dtype in (torch.bfloat16, torch.float16)
    expect_direct = half and (mode == "all" or (mode == "prefill" and prefill))
    assert path == ("direct" if expect_direct else "staged")
    assert kernel_dtype == (dtype if expect_direct else torch.float32)


@pytest.mark.parametrize("rows", [1, 2])
def test_c2_two_row_decode_never_takes_the_prefill_path(monkeypatch, rows):
    # Decode call sites never pass prefill=True; row count is irrelevant.
    assert _route(monkeypatch, mode="prefill", prefill=False, dtype=torch.bfloat16,
                  rows=rows)[0] == "staged"


@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_path_ignores_every_mode(monkeypatch, enabled):
    monkeypatch.setattr(hc, "_DIRECT_PRE", enabled)
    monkeypatch.setattr(hc, "_DIRECT_PRE_PREFILL", enabled)
    torch.manual_seed(6113)
    x = torch.randn(3, 4, 16, dtype=torch.bfloat16)
    pre = torch.randn(3, 4, dtype=torch.float32)
    expected = (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)
    for phase in (False, True):
        assert torch.equal(hc.hc_pre(x, pre, prefill=phase), expected)


# ------------------------------------------------------------------ defensive phase check

@pytest.mark.parametrize("batch,raises", [
    (None, False),
    (SimpleNamespace(is_prefill=True), False),
    (SimpleNamespace(is_prefill=False), True),
])
def test_prefill_direct_refuses_inside_a_decode_forward(monkeypatch, batch, raises):
    monkeypatch.setattr(core, "_GLOBAL_CTX", SimpleNamespace(_batch=batch))
    if raises:
        with pytest.raises(RuntimeError, match="non-prefill forward"):
            _route(monkeypatch, mode="prefill", prefill=True, dtype=torch.bfloat16, rows=2)
    else:
        assert _route(monkeypatch, mode="prefill", prefill=True, dtype=torch.bfloat16,
                      rows=2)[0] == "direct"


def test_staged_paths_never_consult_the_forward_phase(monkeypatch):
    class Exploding:
        @property
        def _batch(self):
            raise AssertionError("phase consulted on a staged path")

    monkeypatch.setattr(core, "_GLOBAL_CTX", Exploding())
    for mode, prefill in (("off", True), ("prefill", False), ("off", False)):
        assert _route(monkeypatch, mode=mode, prefill=prefill, dtype=torch.bfloat16,
                      rows=2)[0] == "staged"


def test_no_global_context_is_allowed(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    assert _route(monkeypatch, mode="prefill", prefill=True, dtype=torch.bfloat16,
                  rows=1)[0] == "direct"


# ------------------------------------------------------------------ warmup

def test_prefill_mode_warmup_loads_the_direct_variant(monkeypatch):
    monkeypatch.setattr(hc, "_DIRECT_PRE", False)
    monkeypatch.setattr(hc, "_DIRECT_PRE_PREFILL", True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch, "zeros", lambda shape, *, device, dtype: SimpleNamespace(shape=shape))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: None)
    calls = []
    monkeypatch.setattr(hc, "hc_pre", lambda x, pre, **kw: calls.append((x.shape, kw)))
    hc.warmup_direct_pre(torch.device("cuda:0"), 5120, 4)
    assert calls == [((r, 4, 5120), {"prefill": True}) for r in (1, 2, 16)]


def test_off_mode_warmup_does_nothing(monkeypatch):
    monkeypatch.setattr(hc, "_DIRECT_PRE", False)
    monkeypatch.setattr(hc, "_DIRECT_PRE_PREFILL", False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: (_ for _ in ()).throw(AssertionError("CUDA queried")))
    hc.warmup_direct_pre(torch.device("cuda:0"), 5120, 4)


# ------------------------------------------------------------------ call-site inventory

PREFILL_SITES = {"Block.prefill_single": 2, "Block.publish_prefill_source": 1,
                 "Transformer.prefill_single": 1, "Transformer.prefill_tail_pair": 1}
OTHER_SITES = {"Block.decode_step": 2, "Block.verify_single": 2,
               "Transformer.decode": 1, "Transformer.verify": 1}


def _hc_pre_calls(path: Path) -> dict[str, list[ast.Call]]:
    tree = ast.parse(path.read_text())
    out: dict[str, list[ast.Call]] = {}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                     and isinstance(c.func, ast.Name) and c.func.id == "hc_pre"]
            if calls:
                out[f"{cls.name}.{fn.name}"] = calls
    return out


def _phase(call: ast.Call):
    kw = {k.arg: k.value for k in call.keywords}
    if "prefill" not in kw:
        return None
    assert isinstance(kw["prefill"], ast.Constant)
    return kw["prefill"].value


def test_model_call_sites_declare_their_phase_explicitly():
    calls = _hc_pre_calls(MODEL)
    assert {k: len(v) for k, v in calls.items()} == {**PREFILL_SITES, **OTHER_SITES}
    for site in PREFILL_SITES:
        assert [_phase(c) for c in calls[site]] == [True] * PREFILL_SITES[site], site
    for site in OTHER_SITES:
        assert [_phase(c) for c in calls[site]] == [None] * OTHER_SITES[site], site
    for site_calls in calls.values():
        for call in site_calls:
            # Phase is a literal at the call site: never derived from shapes or row counts.
            assert len(call.args) == 2


def test_speculative_draft_paths_stay_on_the_staged_default():
    calls = _hc_pre_calls(DSPARK)
    assert calls and all(_phase(c) is None for site in calls.values() for c in site)


# ------------------------------------------------------------------ functional block test

def _block(monkeypatch, phases):
    from freetoken.models.deepseek_v41.model import Block

    block = Block.__new__(Block)
    nn.Module.__init__(block)
    block.layer_id = 3
    block.engram = None
    block._mixes = lambda hidden, sub: ("pre-" + sub, "post-" + sub, "comb-" + sub)
    block.attn_norm = block.ffn_norm = lambda x: x
    block.attn = SimpleNamespace(
        prefill_single=lambda x, *a, **k: x, decode_step=lambda x, *a: x,
        verify_single=lambda x, *a: x, publish_prefill_source=lambda x, *a: None)
    block.ffn = lambda x, *a: x

    def record(hidden, incoming, *, prefill=False):
        phases.append((tuple(hidden.shape), prefill))
        return hidden[..., 0, :]

    monkeypatch.setattr(hc, "hc_pre", record)
    monkeypatch.setattr(hc, "hc_post", lambda out, res, post, comb: res)
    return block


@pytest.mark.parametrize("rows", [1, 2])
def test_block_phases_reach_hc_pre(monkeypatch, rows):
    phases = []
    block = _block(monkeypatch, phases)
    hidden = torch.zeros(rows, 4, 8)
    block.decode_step(hidden, "pre", positions=None, rows=None, cmp_stage_cap=0, window_ctx=None)
    block.verify_single(hidden, "pre", positions=None, cmp_stage_cap=0, window_ctx=None)
    assert phases == [((rows, 4, 8), False)] * 4
    phases.clear()
    block.prefill_single(hidden, "pre", start_pos=0, table_idx=0)
    block.publish_prefill_source(hidden, "pre", start_pos=0, table_idx=0)
    assert phases == [((rows, 4, 8), True)] * 3


# ------------------------------------------------------------------ GPU exactness (lab only)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mode(monkeypatch, mode):
    monkeypatch.setattr(hc, "_DIRECT_PRE", mode == "all")
    monkeypatch.setattr(hc, "_DIRECT_PRE_PREFILL", mode == "prefill")


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape,strided", [((1, 4, 5120), False), ((2, 4, 5120), False),
                                            ((17, 4, 5120), True), ((8192, 4, 5120), False),
                                            ((1106, 4, 5120), True)])
def test_gpu_prefill_mode_is_bitwise_equal_to_staged(monkeypatch, dtype, shape, strided):
    torch.manual_seed(6114)
    storage = torch.randn((*shape[:-1], shape[-1] * (2 if strided else 1)), dtype=dtype, device="cuda")
    x = storage[..., ::2] if strided else storage
    pre = torch.randn(shape[:-1], dtype=torch.float32, device="cuda")
    _mode(monkeypatch, "off")
    expected = hc.hc_pre(x, pre)
    _mode(monkeypatch, "prefill")
    direct = hc.hc_pre(x, pre, prefill=True)
    torch.cuda.synchronize()
    assert torch.isfinite(expected).all()
    assert torch.equal(expected.view(torch.int8), direct.view(torch.int8))


@needs_cuda
def test_gpu_prefill_mode_decode_graph_still_stages_fp32(monkeypatch):
    seen = []
    original = kernel.hc_pre_combine

    def spy(streams, pre, out_dtype):
        seen.append(streams.dtype)
        return original(streams, pre, out_dtype)

    monkeypatch.setattr(kernel, "hc_pre_combine", spy)
    _mode(monkeypatch, "prefill")
    x = torch.randn(2, 4, 5120, dtype=torch.bfloat16, device="cuda")
    pre = torch.randn(2, 4, dtype=torch.float32, device="cuda")
    hc.hc_pre(x, pre)  # warm the staged variant outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = hc.hc_pre(x, pre)  # a two-row C2 decode call site: no prefill keyword
    graph.replay()
    torch.cuda.synchronize()
    assert seen == [torch.float32, torch.float32]
    _mode(monkeypatch, "off")
    assert torch.equal(captured.view(torch.int8), hc.hc_pre(x, pre).view(torch.int8))
