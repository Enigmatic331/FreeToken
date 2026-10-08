"""CPU MoE executor (``CpuMoeExecutor``) tests.

Part 1 -- numerical alignment: the CPU SwiGLU MoE GEMV vs FreeToken's production
GPU decode kernels (bf16/nvfp4/mxfp4/ds_fp4) on identical banks and routing across
batch sizes (fp32-accumulate, so the only spread is reduction order -> tight tol).

Part 2 -- CUDA-graph capture/replay: the cudaLaunchHostFunc submit/sync host nodes
end to end (eager decode, capture, replay with *new* data -> result tracks the new
routing, i.e. the dependency flows through the pinned buffers, not baked-in).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as Fn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_cache(L, E, H, I, scale=0.1):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    gate_up = alloc_pinned_tensor(L * E, 2 * I, H, dtype=torch.bfloat16)
    down = alloc_pinned_tensor(L * E, H, I, dtype=torch.bfloat16)
    gate_up.copy_(torch.randn(L * E, 2 * I, H) * scale)
    down.copy_(torch.randn(L * E, H, I) * scale)
    return SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": list(gate_up.split(E)), "down": list(down.split(E))},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 2, 5, 16])
def test_cpu_decode_matches_gpu_decode_kernel(bs):
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(bs)
    L, E, H, I, top_k = 4, 16, 1024, 512, 4
    layer = 2
    dev = torch.device("cuda")
    cache = _make_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    gate_up_layer = cache.bank_sources["gate_up"][layer].to(dev)
    down_layer = cache.bank_sources["down"][layer].to(dev)
    gpu_out = fused_experts_decode_impl(
        hidden, gate_up_layer, down_layer, w, ids.clone(), "silu", False
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 2e-2, f"bs={bs} rel err {rel.item()}"


def _pack_nvfp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack an [..., IN] uint8 tensor of 4-bit codes to [..., IN//2], low nibble first."""
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).contiguous().to(torch.uint8)


def _make_nvfp4_cache(L, E, H, I, seed=0):
    """Random but valid NVFP4 banks (native ModelOpt schema) for the cpu backend."""
    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, IN):
        codes = torch.randint(0, 16, (S, OUT, IN), dtype=torch.uint8)
        packed = _pack_nvfp4(codes)  # [S, OUT, IN//2]
        # block scales in fp8-e4m3 (1 byte), ~[0.5, 1.5]; row globals in fp16, small.
        scale = (0.5 + torch.rand(S, OUT, IN // 16)).to(torch.float8_e4m3fn)
        glob = (0.02 + 0.01 * torch.rand(S, OUT)).to(torch.float16)
        return packed, scale.contiguous(), glob.contiguous()

    gup, gus, gug = rows(2 * I, H)
    dnp, dns, dng = rows(H, I)
    return SimpleNamespace(
        quant_format="nvfp4",
        bank_sources={
            "gate_up_packed": list(gup.split(E)), "gate_up_scale": list(gus.split(E)),
            "gate_up_global": list(gug.split(E)), "down_packed": list(dnp.split(E)),
            "down_scale": list(dns.split(E)), "down_global": list(dng.split(E)),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 3, 8])
def test_cpu_decode_nvfp4_matches_dequant_then_gpu(bs):
    """CPU inline-dequant NVFP4 GEMV vs. canonical dequant_nvfp4 + bf16 GPU decode.

    Both paths read byte-identical quantized banks; the CPU dequantizes inside the
    K-loop (fp32), the reference materializes bf16 weights first, so the only spread
    is weight bf16-rounding + reduction order -> tight relative tolerance.
    """
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(100 + bs)
    L, E, H, I, top_k = 3, 16, 1024, 512, 4
    layer = 1
    dev = torch.device("cuda")
    cache = _make_nvfp4_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    slots = torch.arange(E, device=dev, dtype=torch.int32)  # layer-local expert rows
    gate_up_layer = dequant_nvfp4(
        b["gate_up_packed"][layer].to(dev), b["gate_up_scale"][layer].to(dev),
        b["gate_up_global"][layer].to(dev), slots, dtype=torch.bfloat16,
    )  # [E, 2I, H]
    down_layer = dequant_nvfp4(
        b["down_packed"][layer].to(dev), b["down_scale"][layer].to(dev),
        b["down_global"][layer].to(dev), slots, dtype=torch.bfloat16,
    )  # [E, H, I]
    gpu_out = fused_experts_decode_impl(
        hidden, gate_up_layer, down_layer, w, ids.clone(), "silu", False
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 3e-2, f"nvfp4 bs={bs} rel err {rel.item()}"


@pytest.mark.parametrize("bs", [1, 4])
def test_cpu_decode_nvfp4_swigluoai_matches_dequant_reference(bs):
    """MiniMax-M3's swigluoai routed experts on the CPU executor (ActKind 3 through
    the generic NVFP4 GEMV epilogue, with the layer's alpha/limit scalars) vs. a
    pure-torch dequant reference."""
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(300 + bs)
    L, E, H, I, top_k = 2, 16, 1024, 512, 4
    layer = 1
    alpha, limit = 1.702, 7.0
    dev = torch.device("cuda")
    cache = _make_nvfp4_cache(L, E, H, I, seed=17)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="swigluoai",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
        swiglu_alpha=alpha,
        swiglu_limit=limit,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    slots = torch.arange(E, device=dev, dtype=torch.int32)
    gate_up_layer = dequant_nvfp4(
        b["gate_up_packed"][layer].to(dev), b["gate_up_scale"][layer].to(dev),
        b["gate_up_global"][layer].to(dev), slots, dtype=torch.bfloat16,
    ).float()  # [E, 2I, H]
    down_layer = dequant_nvfp4(
        b["down_packed"][layer].to(dev), b["down_scale"][layer].to(dev),
        b["down_global"][layer].to(dev), slots, dtype=torch.bfloat16,
    ).float()  # [E, H, I]

    ref = torch.zeros(bs, H, device=dev, dtype=torch.float32)
    x = hidden.float()
    for t in range(bs):
        for k in range(top_k):
            e = int(ids[t, k])
            h = gate_up_layer[e] @ x[t]
            gate = h[:I].clamp(max=limit)
            up = h[I:].clamp(-limit, limit)
            act = gate * torch.sigmoid(gate * alpha) * (up + 1.0)
            ref[t] += float(w[t, k]) * (down_layer[e] @ act)

    rel = (cpu_out - ref).abs().max() / (ref.abs().max() + 1e-6)
    assert rel < 3e-2, f"nvfp4 swigluoai bs={bs} rel err {rel.item()}"


def test_stale_extension_rejected_for_swigluoai(monkeypatch):
    """A prebuilt _cpu_moe.so from before ACT_SWIGLUOAI accepts act id 3 without
    error and silently computes the wrong activation in the generic epilogue.
    The executor probes the extension's `max_generic_act_id` marker -- absent on
    stale builds -- and must fail loudly with the rebuild instruction."""
    from freetoken.kernel import _cpu_moe
    from freetoken.moe.cpu_executor import CpuMoeExecutor, compiled_extension_supports

    # raising=False: on a GENUINELY stale extension the attribute is already
    # absent, and this test must still run (it is the test for that case).
    monkeypatch.delattr(_cpu_moe, "max_generic_act_id", raising=False)
    assert not compiled_extension_supports("swigluoai")
    assert compiled_extension_supports("silu")  # silu family predates the marker
    cache = _make_nvfp4_cache(1, 4, 256, 128)
    with pytest.raises(RuntimeError, match="rebuild"):
        CpuMoeExecutor(
            cache, top_k=2, activation="swigluoai",
            apply_router_weight_on_input=False, num_threads=0, max_tokens=1,
            device=torch.device("cuda"), swiglu_alpha=1.702, swiglu_limit=7.0,
        )
    # silu predates the marker and must keep working on a stale build.
    CpuMoeExecutor(
        cache, top_k=2, activation="silu",
        apply_router_weight_on_input=False, num_threads=0, max_tokens=1,
        device=torch.device("cuda"),
    )


def _pack_mxfp4_transposed(codes: torch.Tensor) -> torch.Tensor:
    """Pack [S, K, N] codes -> transposed blocks [S, K//2, N] (low nibble = even K)."""
    lo = codes[:, 0::2, :]
    hi = codes[:, 1::2, :]
    return (lo | (hi << 4)).contiguous().to(torch.uint8)


def _make_mxfp4_cache(L, E, H, I, seed=0):
    """Random gpt-oss ``mxfp4_triton`` banks (transposed split-K layout)."""
    torch.manual_seed(seed)
    S = L * E

    def gu_rows():  # gate_up: K=H, N=2I
        codes = torch.randint(0, 16, (S, H, 2 * I), dtype=torch.uint8)
        blocks = _pack_mxfp4_transposed(codes)  # [S, H//2, 2I]
        scales = torch.randint(122, 128, (S, H // 32, 2 * I), dtype=torch.uint8)
        bias = (0.01 * torch.randn(S, 2 * I)).to(torch.bfloat16)
        return blocks, scales, bias

    def dn_rows():  # down: K=I, N=H
        codes = torch.randint(0, 16, (S, I, H), dtype=torch.uint8)
        blocks = _pack_mxfp4_transposed(codes)  # [S, I//2, H]
        scales = torch.randint(122, 128, (S, I // 32, H), dtype=torch.uint8)
        bias = (0.01 * torch.randn(S, H)).to(torch.bfloat16)
        return blocks, scales, bias

    gub, gus, gob = gu_rows()
    dnb, dns, dob = dn_rows()
    return SimpleNamespace(
        quant_format="mxfp4_triton",
        bank_sources={
            "gate_up_blocks": list(gub.split(E)), "gate_up_scales": list(gus.split(E)),
            "gate_up_bias": list(gob.split(E)), "down_blocks": list(dnb.split(E)),
            "down_scales": list(dns.split(E)), "down_bias": list(dob.split(E)),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 4, 8])
def test_cpu_decode_mxfp4_matches_gpu_splitk(bs):
    """CPU mxfp4 N-accumulator GEMV vs. gpt-oss production split-K decode, on
    byte-identical transposed banks (same dequant, clamped swiglu, bias, router wt)."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts as _run_mxfp4_splitk_decode_experts

    torch.manual_seed(200 + bs)
    L, E, H, I, top_k = 2, 8, 256, 256, 2
    layer = 1
    alpha, limit = 1.702, 7.0
    dev = torch.device("cuda")
    cache = _make_mxfp4_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="gpt_oss_swiglu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
        swiglu_alpha=alpha,
        swiglu_limit=limit,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gpu_out = _run_mxfp4_splitk_decode_experts(
        hidden, w, ids.clone(),
        b["gate_up_blocks"][layer].to(dev), b["gate_up_scales"][layer].to(dev), b["gate_up_bias"][layer].to(dev),
        b["down_blocks"][layer].to(dev), b["down_scales"][layer].to(dev), b["down_bias"][layer].to(dev),
        top_k=top_k, hidden_act_alpha=alpha, swiglu_limit=limit,
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 4e-2, f"mxfp4 bs={bs} rel err {rel.item()}"


def _make_dsfp4_cache(L, E, H, I, seed=0):
    """Random DeepSeek-V4 ``ds_fp4`` banks (row-major e2m1 + e8m0/32, no global)."""
    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, IN):
        codes = torch.randint(0, 16, (S, OUT, IN), dtype=torch.uint8)
        packed = _pack_nvfp4(codes)  # [S, OUT, IN//2], low nibble = even K
        # e8m0 codes ~ 2^-5..2^0; stored as float8_e8m0fnu (raw byte == code).
        scale = torch.randint(122, 128, (S, OUT, IN // 32), dtype=torch.uint8)
        scale = scale.view(torch.float8_e8m0fnu).contiguous()
        return packed, scale

    gup, gus = rows(2 * I, H)
    dnp, dns = rows(H, I)
    return SimpleNamespace(
        quant_format="ds_fp4",
        bank_sources={
            "gate_up_packed": list(gup.split(E)), "gate_up_scale": list(gus.split(E)),
            "down_packed": list(dnp.split(E)), "down_scale": list(dns.split(E)),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 3, 8])
@pytest.mark.parametrize("activation_block_size", [32, 128])
def test_cpu_decode_dsfp4_matches_gpu(bs, activation_block_size):
    """CPU ds_fp4 W4A8 GEMV vs. DeepSeek-V4 production ``routed_experts_fp4`` on
    byte-identical banks: same e2m1/e8m0 dequant, FP8-e4m3 activation round-trips
    (V4 block 128 or V4.1 block 32), silu-swiglu with clamp, router weight on output."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    torch.manual_seed(300 + bs)
    L, E, H, I, top_k = 2, 8, 256, 256, 4
    layer = 1
    swiglu_limit = 7.0
    dev = torch.device("cuda")
    cache = _make_dsfp4_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
        swiglu_limit=swiglu_limit,
        dsfp4_activation_block_size=activation_block_size,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gpu_out = routed_experts_fp4(
        hidden, ids.clone(), w.contiguous(),
        b["gate_up_packed"][layer].to(dev), b["gate_up_scale"][layer].to(dev),
        b["down_packed"][layer].to(dev), b["down_scale"][layer].to(dev),
        swiglu_limit,
        activation_block_size=activation_block_size,
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 4e-2, f"ds_fp4 bs={bs} rel err {rel.item()}"


# --------------------------------------------------------------------------------
# Part 2: CUDA-graph capture/replay (cudaLaunchHostFunc submit/sync host nodes).
# --------------------------------------------------------------------------------


def _reference(cache, layer, hidden, ids, w, act="silu", apply_in=False):
    gu = cache.bank_sources["gate_up"][layer].float()
    dn = cache.bank_sources["down"][layer].float()
    I = gu.shape[1] // 2
    H = gu.shape[2]
    bs, top_k = ids.shape
    xf = hidden.float().cpu()
    out = torch.zeros(bs, H, dtype=torch.float32)
    for t in range(bs):
        for k in range(top_k):
            e = int(ids[t, k])
            if e < 0:
                continue
            wi = float(w[t, k]) if apply_in else 1.0
            wo = 1.0 if apply_in else float(w[t, k])
            gate = (gu[e, :I, :] @ xf[t]) * wi
            up = (gu[e, I : 2 * I, :] @ xf[t]) * wi
            g = (Fn.silu(gate) * up).bfloat16().float()
            out[t] += wo * (dn[e] @ g)
    return out


def test_cpu_moe_decode_cuda_graph_replay():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(0)
    L, E, H, I, top_k = 3, 8, 512, 256, 2
    layer = 1
    bs = 4
    cache = _make_cache(L, E, H, I)

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=8,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    # Eager forward: also materializes the per-bs pinned buffers + per-(layer,bs) task.
    out_eager = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()
    ref_eager = _reference(cache, layer, hidden, ids.cpu(), w.cpu())
    rel = (out_eager.float().cpu() - ref_eager).abs().max() / ref_eager.abs().max()
    assert rel < 2e-2, f"eager rel err {rel}"

    # Capture.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    # Replay several times with NEW data; the captured host nodes must read the
    # freshly written pinned buffers, not the capture-time values.
    for it in range(3):
        torch.manual_seed(100 + it)
        new_hidden = torch.randn(bs, H, dtype=torch.bfloat16)
        new_ids = torch.randint(0, E, (bs, top_k), dtype=torch.int32)
        new_w = torch.rand(bs, top_k, dtype=torch.float32)
        hidden.copy_(new_hidden)
        ids.copy_(new_ids)
        w.copy_(new_w)
        g.replay()
        torch.cuda.synchronize()
        ref = _reference(cache, layer, new_hidden, new_ids, new_w)
        rel = (out_cap.float().cpu() - ref).abs().max() / ref.abs().max()
        assert rel < 2e-2, f"replay {it} rel err {rel}"

    # When the flag handshake is available (stream memops probed OK), the decodes above
    # must actually have exercised it -- guard against a silent fallback to the
    # host-func path making this test vacuous for the capture-embedded memops.
    if ex._flag_sync:
        assert (layer, bs) in ex._flag_slots, "flag slot expected for the decode task"
        slot = ex._flag_slots[(layer, bs)]
        assert ex._ext.flag_served_count(slot) >= 4, "1 eager + 3 replay dispatches expected"
        assert int(ex._done[slot]) == 1 and int(ex._ready[slot]) == 0, "handshake at rest"
        assert int(ex._err.sum()) == 0, "watchdog must not fire in normal operation"
        ex.raise_if_unhealthy()

    print("cpu moe cuda graph replay OK")


def test_cpu_moe_decode_cuda_graph_replay_mxfp4():
    """gpt-oss mxfp4 path under capture/replay: the host nodes must recompute the
    clamped-swiglu+bias GEMV from the freshly written pinned routing on each replay."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts as _run_mxfp4_splitk_decode_experts

    torch.manual_seed(7)
    L, E, H, I, top_k = 2, 8, 256, 256, 2
    layer, bs, alpha, limit = 1, 4, 1.702, 7.0
    cache = _make_mxfp4_cache(L, E, H, I)

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="gpt_oss_swiglu",
        apply_router_weight_on_input=False, num_threads=8, max_tokens=bs, device=dev,
        swiglu_alpha=alpha, swiglu_limit=limit,
    )
    b = cache.bank_sources
    layer_banks = [b[n][layer].to(dev) for n in (
        "gate_up_blocks", "gate_up_scales", "gate_up_bias",
        "down_blocks", "down_scales", "down_bias")]

    def reference(hidden, ids, w):
        return _run_mxfp4_splitk_decode_experts(
            hidden, w, ids.clone(), *layer_banks,
            top_k=top_k, hidden_act_alpha=alpha, swiglu_limit=limit,
        ).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    ex.decode(layer, hidden, w, ids)  # eager warmup: materialize buffers + task
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    for it in range(3):
        torch.manual_seed(300 + it)
        hidden.copy_(torch.randn(bs, H, dtype=torch.bfloat16))
        ids.copy_(torch.randint(0, E, (bs, top_k), dtype=torch.int32))
        w.copy_(torch.rand(bs, top_k, dtype=torch.float32))
        g.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, ids, w)
        rel = (out_cap.float() - ref).abs().max() / (ref.abs().max() + 1e-6)
        assert rel < 4e-2, f"mxfp4 replay {it} rel err {rel.item()}"

    print("cpu moe cuda graph replay (mxfp4) OK")


@pytest.mark.parametrize("activation_block_size", [32, 128])
def test_cpu_moe_decode_cuda_graph_replay_dsfp4(activation_block_size):
    """DeepSeek-V4 ds_fp4 path under capture/replay: the 4-phase pipeline (input
    FP8 round-trip -> gate_up -> intermediate FP8 round-trip -> down) must recompute
    from the freshly written pinned routing on each replay."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    torch.manual_seed(11)
    L, E, H, I, top_k = 2, 8, 256, 256, 4
    layer, bs, limit = 1, 4, 7.0
    cache = _make_dsfp4_cache(L, E, H, I)

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu",
        apply_router_weight_on_input=False, num_threads=8, max_tokens=bs, device=dev,
        swiglu_limit=limit,
        dsfp4_activation_block_size=activation_block_size,
    )
    b = cache.bank_sources
    layer_banks = [b[n][layer].to(dev) for n in (
        "gate_up_packed", "gate_up_scale", "down_packed", "down_scale")]

    def reference(hidden, ids, w):
        return routed_experts_fp4(
            hidden, ids.clone(), w.contiguous(), *layer_banks, limit,
            activation_block_size=activation_block_size,
        ).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    ex.decode(layer, hidden, w, ids)  # eager warmup: materialize buffers + task
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    for it in range(3):
        torch.manual_seed(500 + it)
        hidden.copy_(torch.randn(bs, H, dtype=torch.bfloat16))
        ids.copy_(torch.randint(0, E, (bs, top_k), dtype=torch.int32))
        w.copy_(torch.rand(bs, top_k, dtype=torch.float32))
        g.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, ids, w)
        rel = (out_cap.float() - ref).abs().max() / (ref.abs().max() + 1e-6)
        assert rel < 4e-2, f"ds_fp4 replay {it} rel err {rel.item()}"

    print("cpu moe cuda graph replay (ds_fp4) OK")


if __name__ == "__main__":
    for bs in (1, 2, 5, 16):
        test_cpu_decode_matches_gpu_decode_kernel(bs)
        print(f"bf16 bs={bs} OK")
    for bs in (1, 3, 8):
        test_cpu_decode_nvfp4_matches_dequant_then_gpu(bs)
        print(f"nvfp4 bs={bs} OK")
    for bs in (1, 4, 8):
        test_cpu_decode_mxfp4_matches_gpu_splitk(bs)
        print(f"mxfp4 bs={bs} OK")
    for bs in (1, 3, 8):
        test_cpu_decode_dsfp4_matches_gpu(bs, 128)
        print(f"ds_fp4 bs={bs} OK")
    test_cpu_moe_decode_cuda_graph_replay()
    test_cpu_moe_decode_cuda_graph_replay_mxfp4()
    test_cpu_moe_decode_cuda_graph_replay_dsfp4(128)


def test_cpu_moe_executor_is_collectable():
    """The flag watchdog must not pin the executor against GC (it holds a weakref and
    re-derefs per tick): dropping the last reference collects the executor -- with its
    C++ worker pool and pinned banks -- and the watchdog daemon exits by itself."""
    import gc
    import time
    import weakref

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    cache = _make_cache(2, 4, 64, 32)
    ex = CpuMoeExecutor(
        cache, top_k=2, activation="silu", apply_router_weight_on_input=False,
        num_threads=2, max_tokens=1, device=torch.device("cuda"),
    )
    watchdog = ex._watchdog if ex._flag_sync else None
    ref = weakref.ref(ex)
    del ex
    gc.collect()
    assert ref() is None, "executor must be collectable once the last reference drops"
    if watchdog is not None:
        watchdog.join(timeout=5.0)  # exits on the first tick after the weakref dies
        assert not watchdog.is_alive(), "watchdog thread must exit after executor GC"


@pytest.mark.parametrize("route_outputs", [False, True])
def test_dsfp4_empty_routes_then_active_graph_replay(route_outputs):
    """Empty overflow must clear stale output without advancing the worker generation."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    cache = _make_dsfp4_cache(1, 8, 256, 256, seed=29)
    ex = CpuMoeExecutor(
        cache, top_k=4, activation="silu", apply_router_weight_on_input=False,
        num_threads=4, max_tokens=1, device=torch.device("cuda"),
        swiglu_limit=10.0, dsfp4_activation_block_size=32,
        return_route_outputs=route_outputs,
    )
    # Also exercise scratch growth from max_tokens=1 to a verification batch.
    x = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.full((5, 4), -1, device="cuda", dtype=torch.int32)
    weights = torch.ones(5, 4, device="cuda")
    ex.decode(0, x, weights, ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = ex.decode(0, x, weights, ids)
    for active in (True, False, True, False):
        ids.fill_(0 if active else -1)
        graph.replay()
        torch.cuda.synchronize()
        if active:
            assert torch.isfinite(output).all() and output.abs().max() > 0
        else:
            assert torch.count_nonzero(output) == 0


def test_dsfp4_per_route_cpu_gpu_merge():
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    cache = _make_dsfp4_cache(1, 8, 256, 256, seed=30)
    ex = CpuMoeExecutor(
        cache, top_k=4, activation="silu", apply_router_weight_on_input=False,
        num_threads=4, max_tokens=1, device=torch.device("cuda"),
        swiglu_limit=10.0, dsfp4_activation_block_size=32, return_route_outputs=True,
    )
    banks = [cache.bank_sources[k][0].cuda() for k in (
        "gate_up_packed", "gate_up_scale", "down_packed", "down_scale")]
    x = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.tensor([[0, 1, 2, 3]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.2, 0.3, 0.1, 0.4]], device="cuda")
    reference = routed_experts_fp4(x, ids, weights, *banks, 10.0, activation_block_size=32)
    for split in (0, 1, 3, 4):
        cpu_ids = ids.clone()
        cpu_ids[:, split:] = -1
        gpu_w = weights.clone()
        gpu_w[:, :split] = 0
        pending = ex.decode_submit(0, x, weights, cpu_ids)
        gpu = routed_experts_fp4(
            x, ids, gpu_w, *banks, 10.0, activation_block_size=32,
            return_route_outputs=True,
        )
        cpu = ex.decode_sync(pending)
        assert cpu.shape == (1, 4, 256)
        merged = (cpu + gpu).sum(dim=1)
        torch.cuda.synchronize()
        relative = (merged.float() - reference.float()).abs().max() / reference.abs().max()
        assert relative < 0.04
        assert torch.count_nonzero(cpu[:, split:]) == 0


def test_dsfp4_gpu_order_rejects_unqualified_geometry():
    from freetoken.kernel import _cpu_moe
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    if not hasattr(_cpu_moe.CpuMoeExecutor, "set_dsfp4_gpu_order"):
        pytest.skip("requires rebuilt GPU-order CPU extension")
    with pytest.raises(ValueError, match="qualified only for V4.1"):
        CpuMoeExecutor(
            _make_dsfp4_cache(1, 2, 256, 256), top_k=2, activation="silu",
            apply_router_weight_on_input=False, num_threads=2, max_tokens=1,
            device=torch.device("cuda"), dsfp4_gpu_order=True,
        )


@pytest.mark.parametrize("bs", [1, 3, 5, 6])
@pytest.mark.parametrize("route_parallel", [False, True])
@pytest.mark.parametrize("fuse_prep", [False, True])
def test_dsfp4_gpu_order_mapped_changing_graph_replay(monkeypatch, bs, route_parallel, fuse_prep):
    """BS1 mapped and larger-batch DMA fallbacks preserve fresh route outputs."""
    from freetoken.kernel import _cpu_moe
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_FUSE_PREP", "1" if fuse_prep else "0")
    if fuse_prep and not hasattr(_cpu_moe.CpuMoeExecutor, "set_dsfp4_fuse_prep"):
        pytest.skip("Requires rebuilt fused-preparation extension")
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_ROUTE_PARALLEL", "1" if route_parallel else "0")
    if route_parallel and not hasattr(_cpu_moe.CpuMoeExecutor, "set_dsfp4_route_parallel"):
        pytest.skip("Requires rebuilt route-parallel extension")
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    if not hasattr(_cpu_moe.CpuMoeExecutor, "set_dsfp4_gpu_order"):
        pytest.skip("requires rebuilt GPU-order CPU extension")
    monkeypatch.setenv("FREETOKEN_CPU_MOE_MAPPED_HANDOFF", "1")
    H, I, E, top_k = 5120, 2304, 4, 4
    cache = _make_dsfp4_cache(1, E, H, I, seed=7319)
    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=4, max_tokens=bs, device=torch.device("cuda"),
        swiglu_limit=10., dsfp4_activation_block_size=32,
        return_route_outputs=True, dsfp4_gpu_order=True,
    )
    assert ex.dsfp4_route_parallel == route_parallel
    assert ex.dsfp4_fuse_prep == fuse_prep
    if not ex._mapped_handoff:
        pytest.skip("mapped handoff requires CUDA stream memory operations")
    banks = [cache.bank_sources[k][0].cuda() for k in (
        "gate_up_packed", "gate_up_scale", "down_packed", "down_scale")]
    x = torch.zeros(bs, H, device="cuda", dtype=torch.bfloat16)
    ids = torch.full((bs, top_k), -1, device="cuda", dtype=torch.int32)
    w = torch.zeros(bs, top_k, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ex.decode_sync(ex.decode_submit(0, x, w, ids))
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = ex.decode_sync(ex.decode_submit(0, x, w, ids))
    torch.cuda.current_stream().wait_stream(stream)
    slot = ex._flag_slots[(0, bs)]
    for n in (1, 0, 4, 2, 0, 3, 0, 4):
        x.copy_(torch.randn_like(x) * .3)
        rid = torch.randint(E, ids.shape, device="cuda", dtype=torch.int32)
        ids.copy_(rid)
        w.copy_(torch.rand_like(w) * .25)
        w[:, n:] = 0
        before = ex._ext.flag_served_count(slot)
        reference = routed_experts_fp4(x, rid, w, *banks, 10.,
            activation_block_size=32, return_route_outputs=True)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, reference)
        assert torch.isfinite(output).all()
        if bs == 1 and n == 0:
            assert ex._ext.flag_served_count(slot) == before
        ex.raise_if_unhealthy()


def test_dsfp4_gpu_order_compacts_only_active_first_stage_routes(monkeypatch):
    """Sparse CPU routes keep their original output slots across graph replays."""
    from freetoken.kernel import _cpu_moe
    if not hasattr(_cpu_moe.CpuMoeExecutor, "set_dsfp4_compact_active_routes"):
        pytest.skip("Requires rebuilt active-route compaction extension")
    monkeypatch.setenv("FREETOKEN_CPU_MOE_MAPPED_HANDOFF", "1")
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_FUSE_PREP", "1")
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_ROUTE_PARALLEL", "0")
    monkeypatch.setenv("FREETOKEN_DSV41_CPU_COMPACT_ACTIVE_ROUTES", "1")
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    H, I, E, top_k = 5120, 2304, 6, 6
    cache = _make_dsfp4_cache(1, E, H, I, seed=8127)
    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=4, max_tokens=1, device=torch.device("cuda"),
        swiglu_limit=10., dsfp4_activation_block_size=32,
        return_route_outputs=True, dsfp4_gpu_order=True,
    )
    assert ex.dsfp4_compact_active_routes
    assert not ex.dsfp4_route_parallel
    if not ex._mapped_handoff:
        pytest.skip("mapped handoff requires CUDA stream memory operations")
    banks = [cache.bank_sources[k][0].cuda() for k in (
        "gate_up_packed", "gate_up_scale", "down_packed", "down_scale")]
    x = torch.zeros(1, H, device="cuda", dtype=torch.bfloat16)
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
    slot = ex._flag_slots[(0, 1)]
    patterns = (
        [-1, 2, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [5, 0, 5, 1, 4, 2],
        [-1, 2, -1, 2, -1, -1],
        [1, -1, 0, 1, -1, -1],
    )
    for route_ids in patterns:
        x.copy_(torch.randn_like(x) * .3)
        host_ids = torch.tensor([route_ids], device="cuda", dtype=torch.int32)
        ids.copy_(host_ids)
        weights = torch.rand_like(w) * .25
        weights.masked_fill_(host_ids < 0, 0)
        w.copy_(weights)
        safe_ids = host_ids.clamp_min(0)
        reference = routed_experts_fp4(
            x, safe_ids, weights, *banks, 10., activation_block_size=32,
            return_route_outputs=True,
        )
        before = ex._ext.flag_served_count(slot)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, reference)
        assert torch.isfinite(output).all()
        active = sum(route >= 0 for route in route_ids)
        if active:
            assert ex._ext.flag_served_count(slot) == before + 1
            assert ex._ext.active_route_count() == active
            assert ex._ext.first_stage_job_count() == active * (I // 32)
        else:
            assert ex._ext.flag_served_count(slot) == before
        ex.raise_if_unhealthy()
