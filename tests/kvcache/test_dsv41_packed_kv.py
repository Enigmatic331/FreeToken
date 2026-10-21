"""Packed FP4 compressed KV must hold exactly the values the BF16 pool holds."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache import dsv41_packed_kv
from freetoken.kvcache.dsv41_packed_kv import PackedRows, pack_reference, unpack_reference
from freetoken.models.deepseek_v41.quant import fake_quant_compressed_kv


def _rows(n=257, d=512, seed=0):
    """Random rows plus the edge cases: zeros, tiny blocks clamped to the 2^-9 scale,
    values that round to -0.0, large magnitudes."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g) * torch.logspace(-4, 1, n)[:, None]
    x[0] = 0.0
    x[1] = torch.randn(d, generator=g) * 1e-5  # below the minimum scale
    x[2, :16] = torch.tensor([6.0, -1e-3, 1e-3, -2e-3] + [0.0] * 12)  # small negatives round to -0.0
    x[3] = torch.randn(d, generator=g) * 400.0
    return x.to(torch.bfloat16)


def _kernel_parts(x):
    """Torch mirror of the rounding in kernel/triton/dsv41/rope_fp4.py (compressed KV)."""
    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(-1)
    scale = (amax / 6.0).clamp(2**-9, 448.0).to(torch.float8_e4m3fn).float()
    scaled = (blocks / scale[..., None]).clamp(-6.0, 6.0)
    magnitude = scaled.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    grid = torch.round(magnitude / step) * step
    return grid.flatten(-2), scale, (scaled < 0).flatten(-2)


def test_kernel_mirror_matches_the_reference_rounding():
    x = _rows()
    grid, scale, negative = _kernel_parts(x)
    sign = torch.where(negative, -1.0, 1.0)
    rounded = (grid * sign).unflatten(-1, (-1, 16)) * scale[..., None]
    expect = fake_quant_compressed_kv(x)
    assert torch.equal(rounded.flatten(-2).to(torch.bfloat16).view(torch.int16), expect.view(torch.int16))


def test_pack_unpack_is_bit_exact_including_negative_zero():
    x = _rows()
    grid, scale, negative = _kernel_parts(x)
    q, s = pack_reference(grid, scale, negative)
    assert q.shape == (x.shape[0], 128) and s.shape == (x.shape[0], 16) and q.dtype == torch.int16
    out = unpack_reference(q, s)
    expect = fake_quant_compressed_kv(x)
    assert torch.equal(out.view(torch.int16), expect.view(torch.int16))  # bitwise, so -0.0 counts
    assert (expect[2, 1].view(torch.int16) == torch.tensor(-0.0, dtype=torch.bfloat16).view(torch.int16)).item()


def test_packed_rows_copy_and_unpack():
    x = _rows(n=8)
    grid, scale, negative = _kernel_parts(x)
    q, s = pack_reference(grid, scale, negative)
    pool = PackedRows.zeros(16, 512, "cpu")
    rows = torch.tensor([3, 9, 0, 15, 1, 2, 7, 4])
    pool.index_copy_(0, rows, PackedRows(q, s))
    assert torch.equal(pool.unpack(rows).view(torch.int16), fake_quant_compressed_kv(x).view(torch.int16))
    assert pool.numel_bytes() == 16 * (256 + 32)
    with pytest.raises(TypeError):
        pool.index_copy_(0, rows, torch.zeros(8, 512, dtype=torch.bfloat16))


def test_packed_storage_shrinks_the_cost_model(monkeypatch):
    from freetoken.kvcache.dsv41_cost_model import dsv41_cache_per_page, dsv41_kv_unit_bytes

    args = SimpleNamespace(n_layers=4, head_dim=512, index_head_dim=128, compress_ratios=(0, 1, 0, 2),
                           kv_source_layers=(1, 3), window_size=128)
    monkeypatch.delenv("FREETOKEN_DSV41_PACKED_KV_SOURCES", raising=False)
    dsv41_packed_kv.packed_kv_enabled.cache_clear()
    monkeypatch.setenv("FREETOKEN_DSV41_PACKED_KV", "0")
    bf16_unit, bf16_page = dsv41_kv_unit_bytes(args), dsv41_cache_per_page(args, 0.25)
    dsv41_packed_kv.packed_kv_enabled.cache_clear()
    monkeypatch.setenv("FREETOKEN_DSV41_PACKED_KV", "1")
    try:
        packed_unit, packed_page = dsv41_kv_unit_bytes(args), dsv41_cache_per_page(args, 0.25)
    finally:
        dsv41_packed_kv.packed_kv_enabled.cache_clear()
    rows_per_page = 128 // 1 + 128 // 2  # ratio-1 and ratio-2 sources
    assert bf16_page - packed_page == rows_per_page * (1024 - 288)
    assert packed_unit < bf16_unit


def test_source_selective_packing_accounts_and_allocates_per_slab(monkeypatch):
    from freetoken.kvcache.dsv41_cost_model import dsv41_pool_bytes, dsv41_pool_sizes
    from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache

    args = SimpleNamespace(
        n_layers=4,
        head_dim=512,
        index_head_dim=128,
        compress_ratios=(0, 1, 0, 2),
        kv_source_layers=(1, 3),
        window_size=128,
    )
    sizes = dsv41_pool_sizes(4, args, 0.25, n_win_pages=1)
    monkeypatch.setenv("FREETOKEN_DSV41_PACKED_KV", "0")
    monkeypatch.setenv("FREETOKEN_DSV41_PACKED_KV_SOURCES", "3")
    dsv41_packed_kv.packed_kv_enabled.cache_clear()
    try:
        pool = DSV41PagedKVCache(
            sizes, args, torch.device("cpu"), P=128, n_scratch=2
        )
        assert isinstance(pool.cmp_pool[1], torch.Tensor)
        assert isinstance(pool.cmp_pool[3], PackedRows)
        assert pool.total_bytes() == dsv41_pool_bytes(sizes, args, n_scratch=2)
        packed_rows = sizes.cmp_blocks[3] + 2
        assert pool.cmp_pool[3].numel_bytes() == packed_rows * 288
    finally:
        dsv41_packed_kv.packed_kv_enabled.cache_clear()


def test_decode_staging_flag_is_startup_cached(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DSV41_PACKED_DECODE_STAGE", "1")
    dsv41_packed_kv.packed_decode_stage_enabled.cache_clear()
    try:
        assert dsv41_packed_kv.packed_decode_stage_enabled()
        monkeypatch.setenv("FREETOKEN_DSV41_PACKED_DECODE_STAGE", "0")
        assert dsv41_packed_kv.packed_decode_stage_enabled()
    finally:
        dsv41_packed_kv.packed_decode_stage_enabled.cache_clear()


class _ShimTensor(torch.Tensor):
    """CPU stand-in for a kernel tensor: ``.to(dtype, bitcast=True)`` reinterprets bits."""

    def to(self, dtype, bitcast=False):
        out = torch.Tensor.view(self, dtype) if bitcast else torch.Tensor.to(self, dtype)
        return out.as_subclass(_ShimTensor)


class _TorchLanguage:
    """The few ``tl``/``gl`` names the dequant helpers use, on CPU tensors."""

    int16, int32, uint8 = torch.int16, torch.int32, torch.uint8
    float32, bfloat16, float8e4nv = torch.float32, torch.bfloat16, torch.float8_e4m3fn

    @staticmethod
    def where(cond, a, b):
        # Kernel constants take the other operand's type (int32 when both are Python ints).
        like = a if isinstance(a, torch.Tensor) else b
        dtype = like.dtype if isinstance(like, torch.Tensor) else (
            torch.float32 if isinstance(a, float) or isinstance(b, float) else torch.int32
        )
        out = torch.where(cond, torch.as_tensor(a, dtype=dtype), torch.as_tensor(b, dtype=dtype))
        return out.as_subclass(_ShimTensor)


def _on_cpu(jit_fn, language):
    """A kernel's own @triton.jit / @gluon.jit elementwise helper, run on CPU tensors."""
    import types

    fn = jit_fn.fn
    scope = {k: (_TorchLanguage if v is language else v) for k, v in fn.__globals__.items()}
    return types.FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__)


def _every_code_and_scale():
    """All 16 nibbles x every non-negative, non-NaN E4M3 scale byte (0x00..0x7E)."""
    nibble, scale_byte = torch.meshgrid(torch.arange(16), torch.arange(0x7F), indexing="ij")
    nibble = nibble.reshape(-1).to(torch.int32).as_subclass(_ShimTensor)
    scale_byte = scale_byte.reshape(-1).to(torch.int32).as_subclass(_ShimTensor)
    table = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
    sign = torch.where((nibble & 8) != 0, -1.0, 1.0)
    scale = scale_byte.to(torch.uint8).view(torch.float8_e4m3fn).float()
    expect = (table[nibble & 7] * sign * scale).to(torch.bfloat16)  # the BF16 pool's value
    return nibble, scale_byte, expect.view(torch.int16)


def test_triton_dequant_is_exact_for_every_code_and_scale():
    import triton.language as tl

    from freetoken.kernel.triton.dsv4 import sparse_attn

    nibble, scale_byte, expect = _every_code_and_scale()
    got = _on_cpu(sparse_attn._fp4_bf16, tl)(nibble, scale_byte)
    assert torch.equal(torch.Tensor.view(got, torch.int16), expect)


def test_gluon_dequant_is_exact_for_every_code_and_scale():
    from triton.experimental.gluon import language as gl

    from freetoken.kernel.triton.dsv41 import sparse_attn_gluon

    nibble, scale_byte, expect = _every_code_and_scale()
    scale = _on_cpu(sparse_attn_gluon._e4m3_f32, gl)(scale_byte)
    got = _on_cpu(sparse_attn_gluon._fp4_value, gl)(nibble, scale)
    assert torch.equal(torch.Tensor.view(got, torch.int16), expect)
    assert (expect[(nibble == 8) & (scale_byte == 0x38)] == torch.tensor(-0.0, dtype=torch.bfloat16).view(torch.int16)).all()


def test_unpack_reference_is_exact_for_every_code_and_scale():
    """The torch unpack (debug, tests and the unaligned-window fallback) over every code."""
    nibble, scale_byte, expect = _every_code_and_scale()
    n = nibble.numel()
    rows = torch.zeros(n, 512, dtype=torch.int32)
    rows[:, 0] = nibble  # channel 0: nibble 0 of word 0, scale group 0 (low byte of word 0)
    scales = torch.zeros(n, 16, dtype=torch.int32)
    scales[:, 0] = scale_byte
    words = rows.view(n, 4, 128)
    q = words[:, 0] | (words[:, 1] << 4) | (words[:, 2] << 8) | (words[:, 3] << 12)
    got = unpack_reference(dsv41_packed_kv._as_int16(q), dsv41_packed_kv._as_int16(scales))[:, 0]
    assert torch.equal(got.view(torch.int16), expect)


def test_unaligned_window_fallback_gathers_the_referenced_rows():
    """sparse_attn_paged's path for n_window % 32 != 0 (a first chunk shorter than the window):
    every compressed column must see the BF16 value of the row it referenced."""
    from freetoken.kernel.triton.dsv4.sparse_attn import _unpack_referenced_rows

    x = _rows(n=40)
    grid, scale, negative = _kernel_parts(x)
    pool = PackedRows(*pack_reference(grid, scale, negative))
    g = torch.Generator().manual_seed(3)
    window = torch.randint(0, 999, (2, 5, 50), generator=g)
    cmp = torch.randint(-1, 40, (2, 5, 70), generator=g)  # -1 masked, many duplicates
    rows, topk = _unpack_referenced_rows(pool, torch.cat([window, cmp], -1).int(), 50)
    assert topk.shape == (2, 5, 120) and topk.dtype == torch.int32
    assert torch.equal(topk[..., :50], window.int())
    remapped = topk[..., 50:]
    assert torch.equal(remapped < 0, cmp < 0)
    full = pool.unpack()
    live = cmp >= 0
    assert torch.equal(rows[remapped[live].long()].view(torch.int16), full[cmp[live]].view(torch.int16))
    assert len(rows) == len(torch.unique(cmp[live]))


def test_gluon_prefill_is_opt_in(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DSV41_PACKED_PREFILL_GLUON", raising=False)
    dsv41_packed_kv.packed_prefill_gluon_enabled.cache_clear()
    try:
        assert not dsv41_packed_kv.packed_prefill_gluon_enabled()
        monkeypatch.setenv("FREETOKEN_DSV41_PACKED_PREFILL_GLUON", "1")
        dsv41_packed_kv.packed_prefill_gluon_enabled.cache_clear()
        assert dsv41_packed_kv.packed_prefill_gluon_enabled()
    finally:
        dsv41_packed_kv.packed_prefill_gluon_enabled.cache_clear()


def test_serving_attention_does_not_import_the_gluon_kernel():
    import subprocess
    import sys

    code = (
        "import sys; import freetoken.kernel.triton.dsv4.sparse_attn, freetoken.kernel.triton.dsv41; "
        "sys.exit('freetoken.kernel.triton.dsv41.sparse_attn_gluon' in sys.modules)"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


def test_gluon_launcher_rejects_unsupported_inputs_before_launch():
    from freetoken.kernel.triton.dsv41.sparse_attn_gluon import sparse_attn_paged_packed_gluon

    q = torch.zeros(1, 2, 16, 512, dtype=torch.bfloat16)
    window = torch.zeros(4, 512, dtype=torch.bfloat16)
    idx = torch.zeros(1, 2, 160, dtype=torch.int32)
    with pytest.raises(ValueError, match="CUDA"):
        sparse_attn_paged_packed_gluon(q, window, PackedRows.zeros(4, 512, "cpu"), torch.zeros(16), idx, 128, 0.04)


# Experimental kernels: never launched on a serving GPU by an ordinary test run (and CUDA is
# not even queried) -- only in an approved admission-closed window, as test_dsv41_packed_gluon.
cuda = pytest.mark.skipif(
    os.environ.get("FREETOKEN_TEST_GPU_WINDOW") != "1" or not torch.cuda.is_available(),
    reason="experimental GPU kernel: requires FREETOKEN_TEST_GPU_WINDOW=1 in an admission-closed window",
)


@cuda
def test_triton_pack_matches_the_bf16_roundtrip():
    from freetoken.kernel.triton.dsv41 import rope_fp4_pack, rope_fp4_roundtrip

    torch.manual_seed(0)
    x = _rows(n=513).cuda()
    angles = torch.rand(513, 32, device="cuda") * 6.28
    freqs = torch.polar(torch.ones_like(angles), angles)
    expect = rope_fp4_roundtrip(x, freqs, 64, compressed_kv=True)
    packed = rope_fp4_pack(x, freqs, 64)
    assert torch.equal(packed.unpack().view(torch.int16), expect.view(torch.int16))


@cuda
@pytest.mark.parametrize("row_dtype", [torch.int32, torch.int64])
def test_triton_pack_into_matches_temporary_pack_and_preserves_other_rows(row_dtype):
    from freetoken.kernel.triton.dsv41 import rope_fp4_pack, rope_fp4_pack_into

    torch.manual_seed(1)
    x = _rows(n=37, seed=1).cuda()
    angles = torch.rand(37, 32, device="cuda") * 6.28
    freqs = torch.polar(torch.ones_like(angles), angles)
    rows = torch.randperm(97, device="cuda")[:37].to(row_dtype).contiguous()
    expect = rope_fp4_pack(x, freqs, 64)
    pool = PackedRows.zeros(97, 512, "cuda")
    rope_fp4_pack_into(x, freqs, 64, pool, rows)
    selected = rows.long()
    assert torch.equal(pool.q.index_select(0, selected), expect.q)
    assert torch.equal(pool.s.index_select(0, selected), expect.s)
    untouched = torch.ones(97, dtype=torch.bool, device="cuda")
    untouched[selected] = False
    assert torch.count_nonzero(pool.q[untouched]) == 0
    assert torch.count_nonzero(pool.s[untouched]) == 0


@cuda
def test_triton_pack_into_replays_with_new_rows_and_values_in_a_cuda_graph():
    from freetoken.kernel.triton.dsv41 import rope_fp4_pack, rope_fp4_pack_into

    n, capacity = 8, 32
    x = torch.empty(n, 512, dtype=torch.bfloat16, device="cuda")
    freqs = torch.empty(n, 32, dtype=torch.complex64, device="cuda")
    rows = torch.empty(n, dtype=torch.int64, device="cuda")
    pool = PackedRows.zeros(capacity, 512, "cuda")
    # Compile before capture.
    x.copy_(_rows(n=n, seed=20).cuda())
    angles = torch.rand(n, 32, device="cuda") * 6.28
    freqs.copy_(torch.polar(torch.ones_like(angles), angles))
    rows.copy_(torch.arange(n, device="cuda"))
    rope_fp4_pack_into(x, freqs, 64, pool, rows)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rope_fp4_pack_into(x, freqs, 64, pool, rows)
    for step in range(3):
        fresh = _rows(n=n, seed=30 + step).cuda()
        angles = torch.rand(n, 32, device="cuda") * 6.28
        fresh_freqs = torch.polar(torch.ones_like(angles), angles)
        fresh_rows = (torch.arange(n, device="cuda") + n * step).long()
        x.copy_(fresh)
        freqs.copy_(fresh_freqs)
        rows.copy_(fresh_rows)
        expect = rope_fp4_pack(fresh, fresh_freqs, 64)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool.q.index_select(0, fresh_rows), expect.q), step
        assert torch.equal(pool.s.index_select(0, fresh_rows), expect.s), step


def _attention_case(seed, m, batch, n_window=128, n_cmp=8192, topk_cmp=512, d=512, h=64):
    from freetoken.kernel.triton.dsv41 import rope_fp4_pack, rope_fp4_roundtrip

    g = torch.Generator(device="cuda").manual_seed(seed)
    window = (torch.randn(2048, d, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    latent = (torch.randn(n_cmp, d, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    latent[:8] = _rows(n=8, d=d, seed=seed).cuda()  # zeros, tiny, -0.0 and large rows
    angles = torch.rand(n_cmp, 32, device="cuda", generator=g) * 6.28
    freqs = torch.polar(torch.ones_like(angles), angles)
    bf16_pool = rope_fp4_roundtrip(latent, freqs, 64, compressed_kv=True)
    packed_pool = rope_fp4_pack(latent, freqs, 64)
    q = (torch.randn(batch, m, h, d, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
    win_idx = torch.randint(0, 2048, (batch, m, n_window), device="cuda", generator=g)
    cmp_idx = torch.randint(0, n_cmp, (batch, m, topk_cmp), device="cuda", generator=g)
    cmp_idx[..., :8] = torch.arange(8, device="cuda")  # the edge rows
    cmp_idx[..., 40:48] = cmp_idx[..., 8:16]  # duplicates
    idx = torch.cat([win_idx, cmp_idx], dim=-1).int()
    drop = torch.rand(idx.shape, device="cuda", generator=g) < 0.03
    idx = torch.where(drop, torch.full_like(idx, -1), idx)  # invalid columns in both halves
    sink = torch.randn(h, device="cuda", generator=g)
    return window, bf16_pool, packed_pool, q, idx, sink


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@cuda
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("m,batch", [(1, 1), (1, 2), (2, 1), (6, 1), (64, 1), (256, 1), (2048, 1), (64, 2)])
def test_packed_attention_is_bitwise_identical(seed, m, batch, monkeypatch):
    """Packed against BF16 storage: decode (split-K and forced single-program, with live-count
    boundaries) and prefill shapes, with invalid and duplicate columns and edge-value rows."""
    from freetoken.kernel.triton.dsv4 import sparse_attn

    window, bf16_pool, packed_pool, q, idx, sink = _attention_case(seed, m, batch)
    count_cases = [None]
    if m == 1:
        count_cases += [0, 1, 31, 32, 33, 300, 512]
    for counts in count_cases:
        cmp_counts = None if counts is None else torch.full((batch, m), counts, dtype=torch.int32, device="cuda")
        for force_single in (False, True) if m == 1 else (False,):
            if force_single:
                monkeypatch.setattr(sparse_attn, "split_count", lambda *a, **k: 0)
            out_bf16 = sparse_attn.sparse_attn_paged(q, window, bf16_pool, sink, idx, 128, 0.04, cmp_counts=cmp_counts)
            out_packed = sparse_attn.sparse_attn_paged(q, window, packed_pool, sink, idx, 128, 0.04, cmp_counts=cmp_counts)
            assert _bits_equal(out_bf16, out_packed), (seed, m, batch, counts, force_single)
            monkeypatch.undo()


@cuda
def test_packed_attention_replays_identically_in_a_cuda_graph():
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

    window, bf16_pool, packed_pool, q, idx, sink = _attention_case(7, 1, 1)
    counts = torch.full((1, 1), 300, dtype=torch.int32, device="cuda")
    expect = sparse_attn_paged(q, window, bf16_pool, sink, idx, 128, 0.04, cmp_counts=counts)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        sparse_attn_paged(q, window, packed_pool, sink, idx, 128, 0.04, cmp_counts=counts)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = sparse_attn_paged(q, window, packed_pool, sink, idx, 128, 0.04, cmp_counts=counts)
    for live in (300, 17, 512):
        counts.fill_(live)
        expect = sparse_attn_paged(q, window, bf16_pool, sink, idx, 128, 0.04, cmp_counts=counts)
        graph.replay()
        torch.cuda.synchronize()
        assert _bits_equal(out, expect), live


@cuda
@pytest.mark.parametrize("batch,live", [(1, 32), (1, 384), (1, 512), (2, 384)])
def test_decode_staging_dispatch_is_bitwise_identical(batch, live, monkeypatch):
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

    monkeypatch.setenv("FREETOKEN_DSV41_PACKED_DECODE_STAGE", "1")
    dsv41_packed_kv.packed_decode_stage_enabled.cache_clear()
    try:
        window, bf16_pool, packed_pool, q, idx, sink = _attention_case(9, 1, batch)
        counts = torch.full((batch, 1), live, dtype=torch.int32, device="cuda")
        expect = sparse_attn_paged(
            q, window, bf16_pool, sink, idx, 128, 0.04, cmp_counts=counts
        )
        actual = sparse_attn_paged(
            q, window, packed_pool, sink, idx, 128, 0.04, cmp_counts=counts
        )
        assert _bits_equal(actual, expect), (batch, live)
    finally:
        dsv41_packed_kv.packed_decode_stage_enabled.cache_clear()


@cuda
def test_decode_staging_replays_changing_counts_and_recycled_rows(monkeypatch):
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged
    from freetoken.kernel.triton.dsv41 import rope_fp4_pack_into, rope_fp4_roundtrip

    monkeypatch.setenv("FREETOKEN_DSV41_PACKED_DECODE_STAGE", "1")
    dsv41_packed_kv.packed_decode_stage_enabled.cache_clear()
    try:
        window, bf16_pool, packed_pool, q, idx, sink = _attention_case(10, 1, 1)
        counts = torch.full((1, 1), 512, dtype=torch.int32, device="cuda")
        # Compile before capture.
        sparse_attn_paged(q, window, packed_pool, sink, idx, 128, 0.04, cmp_counts=counts)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sparse_attn_paged(
                q, window, packed_pool, sink, idx, 128, 0.04, cmp_counts=counts
            )
        recycled = idx[0, 0, 128:].long()
        recycled = recycled[recycled >= 0].unique()[:64]
        for step, live in enumerate((32, 384, 512)):
            fresh = (torch.randn(len(recycled), 512, device="cuda") * (0.2 + step)).to(torch.bfloat16)
            angles = torch.rand(len(recycled), 32, device="cuda") * 6.28
            freqs = torch.polar(torch.ones_like(angles), angles)
            bf16_pool.index_copy_(
                0, recycled, rope_fp4_roundtrip(fresh, freqs, 64, compressed_kv=True)
            )
            rope_fp4_pack_into(fresh, freqs, 64, packed_pool, recycled)
            counts.fill_(live)
            expect = sparse_attn_paged(
                q, window, bf16_pool, sink, idx, 128, 0.04, cmp_counts=counts
            )
            graph.replay()
            torch.cuda.synchronize()
            assert _bits_equal(actual, expect), (step, live)
    finally:
        dsv41_packed_kv.packed_decode_stage_enabled.cache_clear()


@cuda
def test_triton_pack_bytes_match_the_reference_format():
    """Raw words, not just unpacked values: with an identity rotation the kernel's q/s must equal
    pack_reference of the reference rounding."""
    from freetoken.kernel.triton.dsv41 import rope_fp4_pack

    x = _rows(n=257).cuda()
    freqs = torch.polar(torch.ones(257, 32, device="cuda"), torch.zeros(257, 32, device="cuda"))
    packed = rope_fp4_pack(x, freqs, 64)
    grid, scale, negative = _kernel_parts(x.cpu())
    q, s = pack_reference(grid, scale, negative)
    assert torch.equal(packed.q.cpu(), q) and torch.equal(packed.s.cpu(), s)


@cuda
@pytest.mark.parametrize("pattern", ["all_window", "all_packed", "masked_packed_tile", "all_masked"])
def test_packed_attention_tile_transitions(pattern):
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

    window, bf16_pool, packed_pool, q, idx, sink = _attention_case(11, 3, 1)
    idx = idx.clone()
    if pattern == "all_window":
        idx[..., 128:] = -1
    elif pattern == "all_packed":
        idx[..., :128] = -1
    elif pattern == "masked_packed_tile":
        idx[..., 160:192] = -1  # one whole packed tile
    else:
        idx[:] = -1
    a = sparse_attn_paged(q, window, bf16_pool, sink, idx, 128, 0.04)
    b = sparse_attn_paged(q, window, packed_pool, sink, idx, 128, 0.04)
    assert _bits_equal(a, b), pattern


@cuda
def test_packed_attention_graph_replay_with_recycled_rows():
    """Rows rewritten between replays (as a recycled page would be) through the real writer path."""
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged
    from freetoken.kernel.triton.dsv41 import rope_fp4_pack_into, rope_fp4_roundtrip

    window, bf16_pool, packed_pool, q, idx, sink = _attention_case(13, 1, 1)
    counts = torch.full((1, 1), 512, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        sparse_attn_paged(q, window, packed_pool, sink, idx, 128, 0.04, cmp_counts=counts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = sparse_attn_paged(q, window, packed_pool, sink, idx, 128, 0.04, cmp_counts=counts)
    rows = idx[0, 0, 128:].long().clamp_min(0).unique()[:64]
    for step in range(3):
        fresh = (torch.randn(len(rows), 512, device="cuda") * (0.1 + step)).to(torch.bfloat16)
        freqs = torch.polar(torch.ones(len(rows), 32, device="cuda"), torch.rand(len(rows), 32, device="cuda") * 6.28)
        bf16_pool.index_copy_(0, rows, rope_fp4_roundtrip(fresh, freqs, 64, compressed_kv=True))
        rope_fp4_pack_into(fresh, freqs, 64, packed_pool, rows)
        expect = sparse_attn_paged(q, window, bf16_pool, sink, idx, 128, 0.04, cmp_counts=counts)
        graph.replay()
        torch.cuda.synchronize()
        assert _bits_equal(out, expect), step


@cuda
@pytest.mark.parametrize("gluon", [False, True])
@pytest.mark.parametrize("n", [6, 50, 100])
def test_packed_attention_first_chunk_shorter_than_the_window(n, gluon, monkeypatch):
    """A prompt's first chunk of n < 128 tokens attends to n window columns (model.py prefill):
    a width the packed kernels cannot tile, so the referenced rows are unpacked instead."""
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

    monkeypatch.setenv("FREETOKEN_DSV41_PACKED_PREFILL_GLUON", "1" if gluon else "0")
    dsv41_packed_kv.packed_prefill_gluon_enabled.cache_clear()
    try:
        window, bf16_pool, packed_pool, q, idx, sink = _attention_case(17, n, 1, n_window=n, topk_cmp=64)
        a = sparse_attn_paged(q, window, bf16_pool, sink, idx, n, 0.04)
        b = sparse_attn_paged(q, window, packed_pool, sink, idx, n, 0.04)
        assert _bits_equal(a, b), n
    finally:
        dsv41_packed_kv.packed_prefill_gluon_enabled.cache_clear()
