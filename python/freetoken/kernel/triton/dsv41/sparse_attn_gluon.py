"""Experimental packed-KV prefill attention with explicit BF16 shared staging.

Serving reaches this module only through ``sparse_attn_paged`` with a packed
compressed pool, ``m > 1`` and ``FREETOKEN_DSV41_PACKED_PREFILL_GLUON=1``
(startup-only, default off; ``kvcache/dsv41_packed_kv.py``).  The kernel is an
offline-codegen candidate until the GPU parity and performance gates have passed.
It covers the single-program (m > 1) path only; decode keeps the Triton kernels.

A four-warp producer expands the lossless int16 quarter-interleaved packed cache
into BF16 shared memory (window rows are copied as BF16).  Eight consumer warps
extend those exact BF16 values to FP32 and use the production [1, 8] MMA-v2 /
kWidth=2 / TF32 arithmetic, including FP32 probabilities and the original
online-softmax order, so the output bits are the BF16-pool kernel's.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mbarrier, mma_v2


@gluon.jit
def _e4m3_f32(scale_byte):
    """An E4M3 scale byte (0..255 in an int tensor) -> its exact FP32 value."""
    return scale_byte.to(gl.uint8).to(gl.float8e4nv, bitcast=True).to(gl.float32)


@gluon.jit
def _fp4_value(nibble, scale):
    """BF16 of one E2M1 nibble (bit 3 the sign) times its FP32 scale.

    The signed E2M1 value is built as FP32 bits: the same value as the packed Triton
    v1 dequant (``magnitude * sign``), in fewer instructions and registers, which keeps
    the producer spill-free. An E2M1 value times an E4M3 scale has at most 6
    significant bits, so the product and its BF16 rounding are exact; the sign bit is
    kept on zeros, so -0.0 survives as in the BF16 pool.
    """
    code = nibble & 7
    # (code << 22) + bits(0.5) is 2**((code >> 1) - 1) * (1 + (code & 1) / 2) for codes
    # 2..7; code 1 is 0.5 and code 0 is 0.
    bits = gl.where(code >= 2, (code << 22) + 0x3F000000, gl.where(code == 1, 0x3F000000, 0))
    value = (bits | ((nibble & 8) << 28)).to(gl.float32, bitcast=True)
    return (value * scale).to(gl.bfloat16)


@gluon.jit
def _chunk_scales(cs_ptr, rows2, valid2, groups, stride_csn, CHUNK: gl.constexpr):
    """FP32 scales of 64-channel chunk CHUNK: one load per thread, [BLOCK_T, 4, 1]."""
    group = CHUNK * 4 + groups
    words = gl.load(cs_ptr + rows2 * stride_csn + group // 2, mask=valid2, other=0).to(gl.int32) & 0xFFFF
    return gl.expand_dims(_e4m3_f32((words >> ((group % 2) * 8)) & 0xFF), 2)


@gluon.jit
def _packed_chunk(words, scales, QUARTER: gl.constexpr, BLOCK_T: gl.constexpr):
    """The BF16 [BLOCK_T, 64] of one chunk from its half's words [BLOCK_T, 4, 16]."""
    nibble = ((words.to(gl.int32) & 0xFFFF) >> (4 * QUARTER)) & 0xF
    return gl.reshape(_fp4_value(nibble, scales), [BLOCK_T, 64])


@gluon.jit
def _produce_kv(
    win_ptr, cq_ptr, cs_ptr, idx_base, n_active, n_window,
    stride_wn, stride_wd, stride_cqn, stride_csn, stride_it,
    kv_shared, ready, consumed,
    D: gl.constexpr, BLOCK_T: gl.constexpr, BUFFERS: gl.constexpr,
):
    NW: gl.constexpr = gl.num_warps()
    gl.static_assert(NW == 4 and BLOCK_T == 32 and D == 512)
    # A tile is produced as eight [BLOCK_T, 64] chunks, each viewed as
    # [BLOCK_T, 4 scale groups, 16 channels]: every thread owns one row's
    # 16-channel group, so the 128 threads partition a chunk exactly (no
    # replicated warps), load each scale word once, and move 32 contiguous
    # bytes per row segment.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 16], [8, 4, 1], [NW, 1, 1], [2, 1, 0])
    by_group: gl.constexpr = gl.SliceLayout(2, layout)
    rows = gl.arange(0, BLOCK_T, layout=gl.SliceLayout(1, by_group))
    groups = gl.expand_dims(gl.arange(0, 4, layout=gl.SliceLayout(0, by_group)), 0)
    channels = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    cols = gl.expand_dims(groups, 2) * 16 + gl.expand_dims(gl.expand_dims(channels, 0), 1)
    for tile in range(gl.cdiv(n_active, BLOCK_T)):
        slot = tile % BUFFERS
        phase = (tile // BUFFERS) & 1
        offsets = tile * BLOCK_T + rows
        idxs = gl.load(idx_base + offsets * stride_it, offsets < n_active, -1)
        rows2 = gl.expand_dims(idxs, 1)
        valid2 = rows2 >= 0
        rows3 = gl.expand_dims(rows2, 2)
        valid3 = gl.expand_dims(valid2, 2)
        shared_tile = kv_shared.index(slot)
        # Every global load of a tile is issued before waiting for its buffer,
        # so its latency overlaps the consumer's previous tile.
        # N_WINDOW is tile-aligned (asserted by the launcher).
        if tile * BLOCK_T < n_window:
            win_rows = win_ptr + rows3 * stride_wn
            v0 = gl.load(win_rows + (0 * 64 + cols) * stride_wd, mask=valid3, other=0.0)
            v1 = gl.load(win_rows + (1 * 64 + cols) * stride_wd, mask=valid3, other=0.0)
            v2 = gl.load(win_rows + (2 * 64 + cols) * stride_wd, mask=valid3, other=0.0)
            v3 = gl.load(win_rows + (3 * 64 + cols) * stride_wd, mask=valid3, other=0.0)
            v4 = gl.load(win_rows + (4 * 64 + cols) * stride_wd, mask=valid3, other=0.0)
            v5 = gl.load(win_rows + (5 * 64 + cols) * stride_wd, mask=valid3, other=0.0)
            v6 = gl.load(win_rows + (6 * 64 + cols) * stride_wd, mask=valid3, other=0.0)
            v7 = gl.load(win_rows + (7 * 64 + cols) * stride_wd, mask=valid3, other=0.0)
            if tile >= BUFFERS:
                mbarrier.wait(consumed.index(slot), phase ^ 1, deps=(kv_shared,))
            values = (v0, v1, v2, v3, v4, v5, v6, v7)
            for chunk in gl.static_range(D // 64):
                shared_tile.slice(chunk * 64, 64, dim=1).store(
                    gl.reshape(values[chunk], [BLOCK_T, 64])
                )
        else:
            # Chunk c holds channels 64c..64c+63: nibble c // 2 of word half c % 2
            # (kvcache/dsv41_packed_kv.py), scale groups 4c..4c+3.
            word_rows = cq_ptr + rows3 * stride_cqn
            w0 = gl.load(word_rows + cols, mask=valid3, other=0)
            w1 = gl.load(word_rows + (64 + cols), mask=valid3, other=0)
            s0 = _chunk_scales(cs_ptr, rows2, valid2, groups, stride_csn, 0)
            s1 = _chunk_scales(cs_ptr, rows2, valid2, groups, stride_csn, 1)
            s2 = _chunk_scales(cs_ptr, rows2, valid2, groups, stride_csn, 2)
            s3 = _chunk_scales(cs_ptr, rows2, valid2, groups, stride_csn, 3)
            s4 = _chunk_scales(cs_ptr, rows2, valid2, groups, stride_csn, 4)
            s5 = _chunk_scales(cs_ptr, rows2, valid2, groups, stride_csn, 5)
            s6 = _chunk_scales(cs_ptr, rows2, valid2, groups, stride_csn, 6)
            s7 = _chunk_scales(cs_ptr, rows2, valid2, groups, stride_csn, 7)
            if tile >= BUFFERS:
                mbarrier.wait(consumed.index(slot), phase ^ 1, deps=(kv_shared,))
            words = (w0, w1)
            scales = (s0, s1, s2, s3, s4, s5, s6, s7)
            for chunk in gl.static_range(D // 64):
                shared_tile.slice(chunk * 64, 64, dim=1).store(
                    _packed_chunk(words[chunk % 2], scales[chunk], chunk // 2, BLOCK_T)
                )
        mbarrier.arrive(ready.index(slot))


@gluon.jit
def _consume_attention(
    q_base, o_base, sink_ptr, idx_base, n_active, scale, H, head_base,
    stride_qh, stride_qd, stride_oh, stride_od, stride_it,
    kv_shared, query_shared, p_shared, ready, consumed,
    D: gl.constexpr, BLOCK_H: gl.constexpr, BLOCK_T: gl.constexpr,
    BUFFERS: gl.constexpr,
):
    NW: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, NW], instr_shape=[16, 8]
    )
    lhs_layout: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    q_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 2], [1, 0])
    q_heads = head_base + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, q_layout))
    q_dims = gl.arange(0, D, layout=gl.SliceLayout(0, q_layout))
    query = gl.load(
        q_base + q_heads[:, None] * stride_qh + q_dims[None, :] * stride_qd,
        mask=q_heads[:, None] < H, other=0.0,
    )
    query_shared.store(query)
    gl.thread_barrier()

    heads = head_base + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mma))
    dims = gl.arange(0, D, layout=gl.SliceLayout(0, mma))
    columns = gl.arange(0, BLOCK_T, layout=gl.SliceLayout(0, mma))
    m_i = gl.full([BLOCK_H], -float("inf"), gl.float32, gl.SliceLayout(1, mma))
    l_i = gl.zeros([BLOCK_H], gl.float32, gl.SliceLayout(1, mma))
    acc = gl.zeros([BLOCK_H, D], gl.float32, mma)
    for tile in range(gl.cdiv(n_active, BLOCK_T)):
        slot = tile % BUFFERS
        phase = (tile // BUFFERS) & 1
        offsets = tile * BLOCK_T + columns
        idxs = gl.load(idx_base + offsets * stride_it, offsets < n_active, -1)
        valid = idxs >= 0
        mbarrier.wait(ready.index(slot), phase, deps=(kv_shared,))
        shared_tile = kv_shared.index(slot)
        kt = shared_tile.permute([1, 0]).load(rhs_layout).to(gl.float32)
        q = query_shared.load(lhs_layout).to(gl.float32)
        scores = mma_v2(
            q, kt, gl.zeros([BLOCK_H, BLOCK_T], gl.float32, mma),
            input_precision="tf32",
        ) * scale
        scores = gl.where(valid[None, :], scores, -float("inf"))
        m_new = gl.maximum(m_i, gl.max(scores, axis=1))
        alpha = gl.where(m_new == -float("inf"), 1.0, gl.exp(m_i - m_new))
        p = gl.where(valid[None, :], gl.exp(scores - m_new[:, None]), 0.0)
        l_i = l_i * alpha + gl.sum(p, axis=1)
        p_shared.store(p)
        gl.thread_barrier()
        p_operand = p_shared.load(lhs_layout)
        kv = shared_tile.load(rhs_layout).to(gl.float32)
        # Production folds acc*alpha into the dot accumulator.  Preserve that
        # fusion, the K-order and FP32 p; BF16 p would change model arithmetic.
        acc = mma_v2(p_operand, kv, acc * alpha[:, None], input_precision="tf32")
        m_i = m_new
        # A partition barrier ensures every consumer warp finished reading
        # the tile before the producer may reuse this buffer.
        gl.thread_barrier()
        mbarrier.arrive(consumed.index(slot))

    sink = gl.load(sink_ptr + heads, mask=heads < H, other=0.0).to(gl.float32)
    l_i = l_i + gl.exp(sink - m_i)
    out = acc / l_i[:, None]
    gl.store(
        o_base + heads[:, None] * stride_oh + dims[None, :] * stride_od,
        out.to(o_base.dtype.element_ty), mask=heads[:, None] < H,
    )


@gluon.jit
def sparse_attn_paged_packed_gluon_kernel(
    q_ptr, win_ptr, cq_ptr, cs_ptr, o_ptr, sink_ptr, idx_ptr, cnt_ptr,
    scale, H, TOPK, N_WINDOW,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_wn, stride_wd, stride_cqn, stride_csn,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_ib, stride_im, stride_it, stride_nb, stride_nm,
    D: gl.constexpr = 512,
    BLOCK_H: gl.constexpr = 16,
    BLOCK_T: gl.constexpr = 32,
    HAS_COUNTS: gl.constexpr = True,
    BUFFERS: gl.constexpr = 2,
):
    gl.static_assert(D == 512 and BLOCK_H == 16 and BLOCK_T == 32)
    gl.static_assert(gl.num_warps() == 8)
    gl.static_assert(BUFFERS == 1 or BUFFERS == 2)
    pid_m = gl.program_id(0)
    pid_b = gl.program_id(1)
    head_base = gl.program_id(2) * BLOCK_H
    q_base = q_ptr + pid_b * stride_qb + pid_m * stride_qm
    o_base = o_ptr + pid_b * stride_ob + pid_m * stride_om
    idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im
    n_active = TOPK
    if HAS_COUNTS:
        n_active = N_WINDOW + gl.load(cnt_ptr + pid_b * stride_nb + pid_m * stride_nm)

    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_T, D], gl.bfloat16
    )
    kv_shared = gl.allocate_shared_memory(gl.bfloat16, [BUFFERS, BLOCK_T, D], kv_layout)
    query_shared = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_H, D],
        gl.NVMMASharedLayout.get_default_for([BLOCK_H, D], gl.bfloat16),
    )
    p_shared = gl.allocate_shared_memory(
        gl.float32, [BLOCK_H, BLOCK_T],
        gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=4, order=[1, 0]),
    )
    ready = gl.allocate_shared_memory(gl.int64, [BUFFERS, 1], mbarrier.MBarrierLayout())
    consumed = gl.allocate_shared_memory(gl.int64, [BUFFERS, 1], mbarrier.MBarrierLayout())
    for slot in gl.static_range(BUFFERS):
        mbarrier.init(ready.index(slot), count=1)
        mbarrier.init(consumed.index(slot), count=1)

    gl.warp_specialize(
        [
            (_consume_attention, (
                q_base, o_base, sink_ptr, idx_base, n_active, scale, H, head_base,
                stride_qh, stride_qd, stride_oh, stride_od, stride_it,
                kv_shared, query_shared, p_shared, ready, consumed,
                D, BLOCK_H, BLOCK_T, BUFFERS,
            )),
            (_produce_kv, (
                win_ptr, cq_ptr, cs_ptr, idx_base, n_active, N_WINDOW,
                stride_wn, stride_wd, stride_cqn, stride_csn, stride_it,
                kv_shared, ready, consumed, D, BLOCK_T, BUFFERS,
            )),
        ],
        [4],
        [160],
    )


# The Triton release whose code generation the qualification covers: bit-identity with the
# BF16 Triton kernel depends on both kernels' lowering, so a compiler change requalifies.
QUALIFIED_TRITON = "3.6.0"


def sparse_attn_paged_packed_gluon(
    q: torch.Tensor,            # [b, m, h, 512] BF16, m > 1
    window_pool: torch.Tensor,  # [n_win_slots, 512] BF16
    packed,                     # PackedRows: the layer's packed compressed pool
    attn_sink: torch.Tensor,    # [h]
    topk_idxs: torch.Tensor,    # [b, m, topk], GLOBAL slots, layout [window | compressed]
    n_window: int,
    softmax_scale: float,
    cmp_counts: torch.Tensor | None = None,
    *,
    buffers: int = 2,
) -> torch.Tensor:
    """``sparse_attn_paged`` (kernel/triton/dsv4/sparse_attn.py) for a packed pool and
    ``m > 1``: the same contract, and the same output bits once qualified."""
    if triton.__version__ != QUALIFIED_TRITON:
        raise RuntimeError(
            f"packed Gluon prefill is qualified on Triton {QUALIFIED_TRITON}, not {triton.__version__}"
        )
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    if not (q.is_cuda and window_pool.is_cuda and packed.q.is_cuda and packed.s.is_cuda):
        raise ValueError("packed Gluon prefill needs CUDA tensors")
    if m < 2 or d != 512 or packed.head_dim != d or window_pool.shape[1] != d:
        raise ValueError(f"packed Gluon prefill covers m > 1, D = 512: q {tuple(q.shape)}")
    if q.dtype != torch.bfloat16 or window_pool.dtype != torch.bfloat16:
        raise ValueError("packed Gluon prefill takes BF16 queries and window rows")
    if packed.q.stride(1) != 1 or packed.s.stride(1) != 1:
        raise ValueError("packed Gluon prefill needs row-contiguous PackedRows")
    if n_window % 32 or not 0 <= n_window <= topk:
        raise ValueError(f"n_window must be 32-aligned and within topk: {n_window}, {topk}")
    if buffers not in (1, 2):
        raise ValueError(f"buffers must be 1 or 2, got {buffers}")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise ValueError("packed Gluon prefill is built for sm_120 only")
    # The same normalisation as sparse_attn_paged.
    q = q.contiguous()
    window_pool = window_pool.contiguous()
    idx = topk_idxs.contiguous().to(torch.int32)
    sink = attn_sink.contiguous().to(torch.float32)
    o = torch.empty_like(q)
    if cmp_counts is not None:
        cnt = cmp_counts.contiguous().to(torch.int32).view(b, m)
        stride_nb, stride_nm = cnt.stride()
    else:
        cnt, stride_nb, stride_nm = idx, 0, 0
    sparse_attn_paged_packed_gluon_kernel[(m, b, triton.cdiv(h, 16))](
        q, window_pool, packed.q, packed.s, o, sink, idx, cnt,
        float(softmax_scale), h, topk, int(n_window),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        window_pool.stride(0), window_pool.stride(1), packed.q.stride(0), packed.s.stride(0),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2), stride_nb, stride_nm,
        D=d, BLOCK_H=16, BLOCK_T=32, HAS_COUNTS=cmp_counts is not None, BUFFERS=buffers,
        num_warps=8, num_stages=1,
    )
    return o


__all__ = ["QUALIFIED_TRITON", "sparse_attn_paged_packed_gluon", "sparse_attn_paged_packed_gluon_kernel"]
