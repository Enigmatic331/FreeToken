"""Sparse gathered-KV flash attention for DeepSeek-V4 MLA.

Each query attends to a per-query *gathered* set of KV positions plus a per-head
attention sink (a null key with logit ``attn_sink[h]`` and zero value, contributing
only to the softmax denominator). KV is a single shared latent head of width
``head_dim`` (MLA: K == V), broadcast across all query heads.

The gather is PAGED: the candidate KV lives in two GLOBAL pools -- the window-ring
pool ``window_pool[n_win_slots, d]`` and the compressed pool ``cmp_pool[n_cmp, d]``
(both batch-less, addressed by physical slot). Per (request, query) the top-k is a
list of GLOBAL slots laid out ``[window part | compressed part]``: entry ``topk[j]``
loads from ``window_pool`` when ``j < n_window`` else from ``cmp_pool`` (``-1`` is
masked). This mirrors sglang's FlashMLA backend, which takes global page indices
directly -- no per-forward ``index_select`` staging slab.

``topk_idxs``' width is a STATIC buffer width (a CUDA-graph capture bakes it), but the
number of columns a replay must actually visit is not: pass ``cmp_counts`` (a device
``[b, m]`` int32 tensor of per-query VALID compressed columns) and the kernel reads its
loop bound from memory instead of walking the whole buffer. This is the same contract as
sglang's ``flash_mla_with_kvcache(indices=..., topk_length=...)``. Without it the loop
covers the full width, which is correct but pays for every ``-1`` column.

Two implementations, picked from the launch shape -- there is no knob:
  * prefill (``m`` > 1) -- one program per (query, request, head block). The query axis alone
    already fills the device, and a split would only add a merge pass.
  * decode (``m`` == 1) -- flash-decoding: the candidate axis is split across programs and
    merged through log-sum-exp. Decode otherwise launches ``b * cdiv(H, BLOCK_H)`` programs
    (4 at bs=1), leaving nearly every SM idle. Falls back to the prefill kernel when the
    candidate list is too short to slice. NOT bit-identical (the reduction order differs).

Shapes: ``q[b, m, h, d]``, ``window_pool[n_win, d]``, ``cmp_pool[n_cmp, d]``,
``topk_idxs[b, m, topk]`` int32 (global slots, window-first), ``attn_sink[h]`` fp32
-> ``o[b, m, h, d]``. ``n_window`` (int) splits the top-k into its two pool halves.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_H = 16
# The gather has exactly ONE tl.load site (the pool base is selected per column), so it stages
# a single [BLOCK_T, D] KV tile -- 67968 B at BLOCK_T=32, num_stages=2, which fits the ~99KB
# consumer-Blackwell (sm_120, e.g. RTX 5090) budget. (BLOCK_T=64 would need ~103KB.)
BLOCK_T = 32
MAX_SPLITS = 32
MIN_TILES_PER_SPLIT = 4
# Packed kernels: launch pipelining depth, and the depth of their compressed loop (1: unpipelined;
# 2 needs 109,056 B against sm_120's 101,376 B, per the offline AOT gate).
PACKED_STAGES = 2
PACKED_LOOP_STAGES = tl.constexpr(1)


@triton.jit
def _e2m1_value(nibble):
    """E2M1 nibble -> FP32 (sign bit 3 also on zero, so -0.0 survives)."""
    code = (nibble & 7).to(tl.float32)
    magnitude = tl.where(code < 4.0, code * 0.5, tl.where(code < 6.0, code - 2.0, (code - 4.0) * 2.0))
    # A multiply keeps -0.0 (``-x`` may lower to ``0 - x``, which gives +0.0).
    return magnitude * tl.where((nibble & 8) != 0, -1.0, 1.0)


@triton.jit
def _fp4_bf16(nibble, sb):
    """BF16 of one E2M1 nibble times its E4M3 scale byte, built with integer arithmetic.
    The product is ``P * 2**K`` with an integer significand ``P`` of at most 6 bits, so it is
    exact in BF16 and so is this construction (sign kept on zeros, as in the BF16 pool)."""
    code = nibble & 7
    # E2M1 (code >= 1): (2 + f) / 2 * 2**e, f = code & 1 except code 1 (0.5), e = (code >> 1) - 1
    f = tl.where(code == 1, 0, code & 1)
    e = (code >> 1) - 1
    # E4M3 scale: normal (8 + m) / 8 * 2**(x - 7); subnormal (x == 0) m * 2**-9
    x = (sb >> 3) & 0xF
    m = sb & 7
    sig = tl.where(x > 0, (2 + f) * (8 + m), (2 + f) * m)
    k = tl.where(x > 0, e + x - 11, e - 10)
    msb = tl.where(sig >= 32, 5, tl.where(sig >= 16, 4, tl.where(sig >= 8, 3, tl.where(sig >= 4, 2, tl.where(sig >= 2, 1, 0)))))
    bits = ((msb + k + 127) << 7) | ((sig << (7 - msb)) & 0x7F)
    bits = tl.where((code == 0) | (sig == 0), 0, bits) | ((nibble >> 3) << 15)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _packed_kv_quarter(words, cs_ptr, rows, valid, stride_csn, QUARTER: tl.constexpr, D: tl.constexpr):
    """One [BLOCK_T, D/4] FP32 quarter (channels QUARTER*D/4 + w) of a packed FP4 KV tile
    (kvcache/dsv41_packed_kv.py), equal to the BF16 the unpacked pool holds. ``words`` is the
    tile's contiguous 16-bit nibble load: every operand stays a 1:1 view of 16-bit loads, so
    Triton gives the MMA operands the BF16 tile's layout (kWidth=2). From 8-bit loads it picks
    kWidth=4, a different accumulation order, and the results differ in the last bit."""
    offs_w = tl.arange(0, D // 4)
    nibble = (words >> (4 * QUARTER)) & 0xF
    group = QUARTER * (D // 64) + offs_w // 16
    scale_words = tl.load(cs_ptr + rows * stride_csn + (group // 2)[None, :], mask=valid[:, None], other=0).to(tl.int32) & 0xFFFF
    scale_bits = (scale_words >> ((group % 2) * 8)[None, :]) & 0xFF
    return _fp4_bf16(nibble, scale_bits).to(tl.float32)


@triton.jit
def _packed_kv_quarters(cq_ptr, cs_ptr, idxs, valid, stride_cqn, stride_csn, D: tl.constexpr):
    rows = idxs[:, None]
    words = tl.load(cq_ptr + rows * stride_cqn + tl.arange(0, D // 4)[None, :], mask=valid[:, None], other=0).to(tl.int32) & 0xFFFF
    return (
        _packed_kv_quarter(words, cs_ptr, rows, valid, stride_csn, 0, D),
        _packed_kv_quarter(words, cs_ptr, rows, valid, stride_csn, 1, D),
        _packed_kv_quarter(words, cs_ptr, rows, valid, stride_csn, 2, D),
        _packed_kv_quarter(words, cs_ptr, rows, valid, stride_csn, 3, D),
    )


@triton.jit
def _bf16_kv_quarters(row_ptr, offs_w, stride_d, valid, D: tl.constexpr):
    """The four [BLOCK_T, D/4] FP32 quarters of BF16 rows (window tiles in the packed kernels)."""
    return (
        tl.load(row_ptr + offs_w[None, :] * stride_d, mask=valid[:, None], other=0.0).to(tl.float32),
        tl.load(row_ptr + (offs_w + D // 4)[None, :] * stride_d, mask=valid[:, None], other=0.0).to(tl.float32),
        tl.load(row_ptr + (offs_w + D // 2)[None, :] * stride_d, mask=valid[:, None], other=0.0).to(tl.float32),
        tl.load(row_ptr + (offs_w + 3 * D // 4)[None, :] * stride_d, mask=valid[:, None], other=0.0).to(tl.float32),
    )


@triton.jit
def _online_softmax_quarters(q0, q1, q2, q3, k0, k1, k2, k3, valid, m_i, l_i, a0, a1, a2, a3, scale):
    """One online-softmax tile on 128-channel quarters. QK^T chains the quarters through the
    dot accumulator (the same K order as one full-width dot) and PV computes each output
    quarter: both give the full-width dots' bits."""
    scores = tl.dot(q3, tl.trans(k3), tl.dot(q2, tl.trans(k2), tl.dot(q1, tl.trans(k1), tl.dot(q0, tl.trans(k0))))) * scale
    scores = tl.where(valid[None, :], scores, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
    p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    a0 = a0 * alpha[:, None] + tl.dot(p.to(k0.dtype), k0)
    a1 = a1 * alpha[:, None] + tl.dot(p.to(k1.dtype), k1)
    a2 = a2 * alpha[:, None] + tl.dot(p.to(k2.dtype), k2)
    a3 = a3 * alpha[:, None] + tl.dot(p.to(k3.dtype), k3)
    return m_new, l_i, a0, a1, a2, a3


@triton.jit
def _sparse_attn_paged_kernel(
    q_ptr, win_ptr, cmp_ptr, o_ptr, sink_ptr, idx_ptr, cnt_ptr, cq_ptr, cs_ptr,
    scale,
    H, TOPK, N_WINDOW,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_wn, stride_wd,
    stride_cn, stride_cd,
    stride_cqn, stride_csn,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_ib, stride_im, stride_it,
    stride_nb, stride_nm,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    PACKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_d = tl.arange(0, D)

    q_ptrs = q_ptr + pid_b * stride_qb + pid_m * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_H, D]

    m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    # Columns this query must visit. The window half is always walked in full (N_WINDOW is a
    # small constant); only the compressed tail is bounded by the live count.
    n_active = TOPK
    if HAS_COUNTS:
        n_active = N_WINDOW + tl.load(cnt_ptr + pid_b * stride_nb + pid_m * stride_nm)

    idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im
    if PACKED:
        # Packed compressed pool (kvcache/dsv41_packed_kv.py), on 128-channel quarters (see
        # _online_softmax_quarters). Window tiles (BF16) and packed tiles get one loop each, in the
        # original tile order; N_WINDOW is BLOCK_T-aligned.
        offs_w = tl.arange(0, D // 4)
        q_base = q_ptr + pid_b * stride_qb + pid_m * stride_qm + offs_h[:, None] * stride_qh
        q0, q1, q2, q3 = _bf16_kv_quarters(q_base, offs_w, stride_qd, h_mask, D)
        a0 = tl.zeros((BLOCK_H, D // 4), dtype=tl.float32)
        a1 = tl.zeros((BLOCK_H, D // 4), dtype=tl.float32)
        a2 = tl.zeros((BLOCK_H, D // 4), dtype=tl.float32)
        a3 = tl.zeros((BLOCK_H, D // 4), dtype=tl.float32)
        n_tiles = tl.cdiv(n_active, BLOCK_T)
        n_win_tiles = tl.minimum(N_WINDOW // BLOCK_T, n_tiles)
        for t in range(0, n_win_tiles):
            offs_t = t * BLOCK_T + tl.arange(0, BLOCK_T)
            idxs = tl.load(idx_base + offs_t * stride_it, mask=offs_t < n_active, other=-1)
            valid = idxs >= 0
            k0, k1, k2, k3 = _bf16_kv_quarters(win_ptr + idxs[:, None] * stride_wn, offs_w, stride_wd, valid, D)
            m_i, l_i, a0, a1, a2, a3 = _online_softmax_quarters(q0, q1, q2, q3, k0, k1, k2, k3, valid, m_i, l_i, a0, a1, a2, a3, scale)
        for t in tl.range(n_win_tiles, n_tiles, num_stages=PACKED_LOOP_STAGES):
            offs_t = t * BLOCK_T + tl.arange(0, BLOCK_T)
            idxs = tl.load(idx_base + offs_t * stride_it, mask=offs_t < n_active, other=-1)
            valid = idxs >= 0
            k0, k1, k2, k3 = _packed_kv_quarters(cq_ptr, cs_ptr, idxs, valid, stride_cqn, stride_csn, D)
            m_i, l_i, a0, a1, a2, a3 = _online_softmax_quarters(q0, q1, q2, q3, k0, k1, k2, k3, valid, m_i, l_i, a0, a1, a2, a3, scale)
        sink = tl.load(sink_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        l_i = l_i + tl.exp(sink - m_i)
        o_base = o_ptr + pid_b * stride_ob + pid_m * stride_om + offs_h[:, None] * stride_oh
        tl.store(o_base + offs_w[None, :] * stride_od, (a0 / l_i[:, None]).to(o_ptr.dtype.element_ty), mask=h_mask[:, None])
        tl.store(o_base + (offs_w + D // 4)[None, :] * stride_od, (a1 / l_i[:, None]).to(o_ptr.dtype.element_ty), mask=h_mask[:, None])
        tl.store(o_base + (offs_w + D // 2)[None, :] * stride_od, (a2 / l_i[:, None]).to(o_ptr.dtype.element_ty), mask=h_mask[:, None])
        tl.store(o_base + (offs_w + 3 * D // 4)[None, :] * stride_od, (a3 / l_i[:, None]).to(o_ptr.dtype.element_ty), mask=h_mask[:, None])
    else:
        for t in range(0, tl.cdiv(n_active, BLOCK_T)):
            offs_t = t * BLOCK_T + tl.arange(0, BLOCK_T)
            t_mask = offs_t < n_active
            idxs = tl.load(idx_base + offs_t * stride_it, mask=t_mask, other=-1)
            valid = idxs >= 0
            # Window-first partition: top-k column j < N_WINDOW reads window_pool, else cmp_pool.
            # Both pools are contiguous [*, D] with identical strides (asserted in the wrapper), so we
            # select the per-column pool BASE pointer and issue a SINGLE gather load. Keeping exactly
            # one ``tl.load`` site (rather than a two-pool ``tl.where`` over two loaded tiles, or an
            # if/elif/else with several load sites) stops Triton's software pipeliner from staging
            # multiple KV tiles in shared memory. Result is bit-identical to the two-pool form -- each
            # column still reads from the same pool/slot.
            is_win = offs_t < N_WINDOW
            base = tl.where(is_win, win_ptr, cmp_ptr)  # [BLOCK_T] per-column pool base pointer
            kv_ptrs = base[:, None] + idxs[:, None] * stride_wn + offs_d[None, :] * stride_wd
            kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)  # [BLOCK_T, D]

            scores = tl.dot(q, tl.trans(kv)) * scale  # [BLOCK_H, BLOCK_T]
            scores = tl.where(valid[None, :], scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            # A tile with no valid column leaves m_new at -inf; -inf - -inf is NaN, so short-circuit
            # both terms. Identical results whenever m_new is finite (alpha -> exp(m_i - m_new),
            # masked-off p -> exp(-inf - finite) == 0.0), which is every non-degenerate query.
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
            m_i = m_new

        sink = tl.load(sink_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        l_i = l_i + tl.exp(sink - m_i)
        o = acc / l_i[:, None]

        o_ptrs = o_ptr + pid_b * stride_ob + pid_m * stride_om + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
        tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=h_mask[:, None])


@triton.jit
def _sparse_attn_paged_splitk_kernel(
    q_ptr, win_ptr, cmp_ptr, mid_o_ptr, mid_lse_ptr, idx_ptr, cnt_ptr, cq_ptr, cs_ptr,
    scale,
    H, TOPK, N_WINDOW,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_wn, stride_wd,
    stride_cn, stride_cd,
    stride_cqn, stride_csn,
    stride_mb, stride_mm, stride_mh, stride_ms, stride_md,
    stride_lb, stride_lm, stride_lh, stride_ls,
    stride_ib, stride_im, stride_it,
    stride_nb, stride_nm,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PACKED: tl.constexpr,
):
    """Stage 1: each program reduces one BLOCK_T-aligned slice of the candidate list and writes
    its normalized partial output + log-sum-exp. The split axis is folded into program_id(0)."""
    pid_ms = tl.program_id(0)
    pid_m = pid_ms // NUM_SPLITS
    split_id = pid_ms % NUM_SPLITS
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_d = tl.arange(0, D)

    n_active = TOPK
    if HAS_COUNTS:
        n_active = N_WINDOW + tl.load(cnt_ptr + pid_b * stride_nb + pid_m * stride_nm)

    # BLOCK_T-aligned slices so no program straddles a tile; trailing splits fall off the end
    # and exit, which is how a short candidate list stops paying for the full grid.
    per_split = tl.cdiv(tl.cdiv(n_active, NUM_SPLITS), BLOCK_T) * BLOCK_T
    split_start = per_split * split_id
    split_end = tl.minimum(split_start + per_split, n_active)

    m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    if PACKED:
        # As in the single-program kernel: 128-channel quarters, window tiles then packed tiles in
        # the original order. Slices are BLOCK_T-aligned and so is N_WINDOW.
        offs_w = tl.arange(0, D // 4)
        a0 = tl.zeros((BLOCK_H, D // 4), dtype=tl.float32)
        a1 = tl.zeros((BLOCK_H, D // 4), dtype=tl.float32)
        a2 = tl.zeros((BLOCK_H, D // 4), dtype=tl.float32)
        a3 = tl.zeros((BLOCK_H, D // 4), dtype=tl.float32)
        if split_end > split_start:
            q_base = q_ptr + pid_b * stride_qb + pid_m * stride_qm + offs_h[:, None] * stride_qh
            q0, q1, q2, q3 = _bf16_kv_quarters(q_base, offs_w, stride_qd, h_mask, D)
            idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im
            win_end = tl.minimum(split_end, N_WINDOW)
            for start in range(split_start, win_end, BLOCK_T):
                offs_t = start + tl.arange(0, BLOCK_T)
                idxs = tl.load(idx_base + offs_t * stride_it, mask=offs_t < split_end, other=-1)
                valid = idxs >= 0
                k0, k1, k2, k3 = _bf16_kv_quarters(win_ptr + idxs[:, None] * stride_wn, offs_w, stride_wd, valid, D)
                m_i, l_i, a0, a1, a2, a3 = _online_softmax_quarters(q0, q1, q2, q3, k0, k1, k2, k3, valid, m_i, l_i, a0, a1, a2, a3, scale)
            for start in tl.range(tl.maximum(split_start, N_WINDOW), split_end, BLOCK_T, num_stages=PACKED_LOOP_STAGES):
                offs_t = start + tl.arange(0, BLOCK_T)
                idxs = tl.load(idx_base + offs_t * stride_it, mask=offs_t < split_end, other=-1)
                valid = idxs >= 0
                k0, k1, k2, k3 = _packed_kv_quarters(cq_ptr, cs_ptr, idxs, valid, stride_cqn, stride_csn, D)
                m_i, l_i, a0, a1, a2, a3 = _online_softmax_quarters(q0, q1, q2, q3, k0, k1, k2, k3, valid, m_i, l_i, a0, a1, a2, a3, scale)
        lse = tl.where(l_i == 0.0, -float("inf"), m_i + tl.log(l_i))
        mid_base = (
            mid_o_ptr + pid_b * stride_mb + pid_m * stride_mm
            + offs_h[:, None] * stride_mh + split_id * stride_ms
        )
        tl.store(mid_base + offs_w[None, :] * stride_md, tl.where(l_i[:, None] == 0.0, 0.0, a0 / l_i[:, None]), mask=h_mask[:, None])
        tl.store(mid_base + (offs_w + D // 4)[None, :] * stride_md, tl.where(l_i[:, None] == 0.0, 0.0, a1 / l_i[:, None]), mask=h_mask[:, None])
        tl.store(mid_base + (offs_w + D // 2)[None, :] * stride_md, tl.where(l_i[:, None] == 0.0, 0.0, a2 / l_i[:, None]), mask=h_mask[:, None])
        tl.store(mid_base + (offs_w + 3 * D // 4)[None, :] * stride_md, tl.where(l_i[:, None] == 0.0, 0.0, a3 / l_i[:, None]), mask=h_mask[:, None])
        lse_base = (
            mid_lse_ptr + pid_b * stride_lb + pid_m * stride_lm
            + offs_h * stride_lh + split_id * stride_ls
        )
        tl.store(lse_base, lse, mask=h_mask)
    else:
        if split_end > split_start:
            q_ptrs = (
                q_ptr + pid_b * stride_qb + pid_m * stride_qm
                + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
            )
            q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)
            idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im

            for start in range(split_start, split_end, BLOCK_T):
                offs_t = start + tl.arange(0, BLOCK_T)
                t_mask = offs_t < split_end
                idxs = tl.load(idx_base + offs_t * stride_it, mask=t_mask, other=-1)
                valid = idxs >= 0
                is_win = offs_t < N_WINDOW
                base = tl.where(is_win, win_ptr, cmp_ptr)
                kv_ptrs = base[:, None] + idxs[:, None] * stride_wn + offs_d[None, :] * stride_wd
                kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

                scores = tl.dot(q, tl.trans(kv)) * scale
                scores = tl.where(valid[None, :], scores, -float("inf"))

                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
                p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
                m_i = m_new

        # Normalized partial + its lse; an empty (or all-masked) split reports lse = -inf so the
        # merge's exp(lse - m) weight is exactly 0.
        out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
        lse = tl.where(l_i == 0.0, -float("inf"), m_i + tl.log(l_i))

        mid_base = (
            mid_o_ptr + pid_b * stride_mb + pid_m * stride_mm
            + offs_h[:, None] * stride_mh + split_id * stride_ms + offs_d[None, :] * stride_md
        )
        tl.store(mid_base, out, mask=h_mask[:, None])
        lse_base = (
            mid_lse_ptr + pid_b * stride_lb + pid_m * stride_lm
            + offs_h * stride_lh + split_id * stride_ls
        )
        tl.store(lse_base, lse, mask=h_mask)


@triton.jit
def _sparse_attn_splitk_merge_kernel(
    mid_o_ptr, mid_lse_ptr, o_ptr, sink_ptr,
    stride_mb, stride_mm, stride_mh, stride_ms, stride_md,
    stride_lb, stride_lm, stride_lh, stride_ls,
    stride_ob, stride_om, stride_oh, stride_od,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Stage 2: log-sum-exp merge over the splits. The attention sink joins here (once, as a
    null key with logit ``attn_sink[h]`` and zero value) exactly as v1 applies it at the end.

    One program per (query, request, HEAD) so the accumulator is a [D] vector: a [BLOCK_H, D]
    tile would be 32 KB of registers per program and spill.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_d = tl.arange(0, D)

    m_i = tl.load(sink_ptr + pid_h).to(tl.float32)
    l_i = 1.0  # the sink's own exp(sink - m_i) == 1
    acc = tl.zeros((D,), dtype=tl.float32)

    mid_base = (
        mid_o_ptr + pid_b * stride_mb + pid_m * stride_mm + pid_h * stride_mh
        + offs_d * stride_md
    )
    lse_base = mid_lse_ptr + pid_b * stride_lb + pid_m * stride_lm + pid_h * stride_lh

    for split_id in tl.range(0, NUM_SPLITS, num_stages=2):
        partial = tl.load(mid_base + split_id * stride_ms)
        lse = tl.load(lse_base + split_id * stride_ls)
        m_new = tl.maximum(m_i, lse)
        alpha = tl.exp(m_i - m_new)
        beta = tl.where(lse == -float("inf"), 0.0, tl.exp(lse - m_new))
        acc = acc * alpha + partial * beta
        l_i = l_i * alpha + beta
        m_i = m_new

    o = acc / l_i
    o_ptrs = (
        o_ptr + pid_b * stride_ob + pid_m * stride_om + pid_h * stride_oh
        + offs_d * stride_od
    )
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty))


def split_count(b: int, m: int, h: int, topk: int, device) -> int:
    """How many ways to split the candidate axis; 0 means run the single-program kernel.

    Prefill (m > 1) never splits. Decode splits until either every SM has a program or the
    slices drop below MIN_TILES_PER_SPLIT tiles of real work -- every split, including the ones
    a small live count leaves empty, still stores a [D] partial the merge reads back, so
    over-splitting buys occupancy in stage 1 and pays for it twice in stage 2.
    """
    if m != 1:
        return 0
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    n_splits = min(
        MAX_SPLITS,
        triton.cdiv(topk, MIN_TILES_PER_SPLIT * BLOCK_T),
        max(1, sm_count // (b * triton.cdiv(h, BLOCK_H))),
    )
    return n_splits if n_splits > 1 else 0


def _unpack_referenced_rows(packed, topk_idxs: torch.Tensor, n_window: int):
    """BF16 copies of the packed rows the compressed columns of ``topk_idxs`` reference, and
    ``topk_idxs`` with those columns remapped onto the copies (window columns unchanged)."""
    if topk_idxs.is_cuda:
        assert not torch.cuda.is_current_stream_capturing(), "data-dependent unpack under graph capture"
    window_cols, cmp_cols = topk_idxs[..., :n_window], topk_idxs[..., n_window:]
    valid = cmp_cols >= 0
    rows, inverse = torch.unique(cmp_cols[valid].long(), return_inverse=True)
    remapped = torch.full_like(cmp_cols, -1)
    remapped[valid] = inverse.to(remapped.dtype)
    return packed.unpack(rows), torch.cat([window_cols, remapped], -1)


def sparse_attn_paged(
    q: torch.Tensor,            # [b, m, h, d]
    window_pool: torch.Tensor,  # [n_win_slots, d]  GLOBAL window-ring pool (this layer)
    cmp_pool: torch.Tensor,     # [n_cmp, d]        GLOBAL compressed pool (this layer)
    attn_sink: torch.Tensor,    # [h]
    topk_idxs: torch.Tensor,    # [b, m, topk] int32, GLOBAL slots, layout [window | compressed]
    n_window: int,              # # of window entries (the first n_window cols of topk)
    softmax_scale: float,
    cmp_counts: torch.Tensor | None = None,  # [b, m] int32, live compressed columns per query
) -> torch.Tensor:
    """Paged sparse MLA attention: gather KV from the two global pools inside the kernel.

    ``topk_idxs[..., j]`` is a GLOBAL physical slot: ``j < n_window`` -> ``window_pool``,
    else ``cmp_pool`` (``-1`` masked). No per-forward staging slab.

    ``cmp_counts`` bounds the compressed half per query, read from DEVICE memory so a captured
    graph's work tracks the live position instead of the buffer width. Omit it (prefill/extend,
    where the width is already the real length) to walk the full ``topk_idxs``.

    The kernel variant follows the launch shape (see the module docstring); nothing selects it.
    """
    from freetoken.kvcache.dsv41_packed_kv import (
        PackedRows,
        packed_decode_stage_enabled,
        packed_prefill_gluon_enabled,
    )

    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    packed = cmp_pool if isinstance(cmp_pool, PackedRows) else None
    if packed is not None and n_window % BLOCK_T:
        # A window width the packed kernels cannot tile: the first chunk of a prompt shorter
        # than the window (n_window = min(n, window)). Run the BF16 kernel on BF16 copies of
        # the few rows it references -- the same values in the same columns, so the same bits.
        cmp_pool, topk_idxs = _unpack_referenced_rows(packed, topk_idxs, n_window)
        packed = None
    if packed is not None and m > 1 and packed_prefill_gluon_enabled():
        from freetoken.kernel.triton.dsv41.sparse_attn_gluon import sparse_attn_paged_packed_gluon

        return sparse_attn_paged_packed_gluon(
            q, window_pool, packed, attn_sink, topk_idxs, n_window, softmax_scale, cmp_counts
        )
    assert 0 <= n_window <= topk, (n_window, topk)
    q = q.contiguous()
    window_pool = window_pool.contiguous()
    idx = topk_idxs.contiguous().to(torch.int32)
    sink = attn_sink.contiguous().to(torch.float32)
    o = torch.empty_like(q)

    has_counts = cmp_counts is not None
    if has_counts:
        cnt = cmp_counts.contiguous().to(torch.int32).view(b, m)
        stride_nb, stride_nm = cnt.stride()
    else:
        cnt, stride_nb, stride_nm = idx, 0, 0

    if packed is not None and m == 1 and packed_decode_stage_enabled():
        from freetoken.kernel.triton.dsv41.packed_decode_stage import stage_packed_decode

        cmp_pool, idx = stage_packed_decode(
            packed, idx, n_window, cnt if has_counts else None
        )
        packed = None
    elif packed is not None:
        # The packed kernels walk window and compressed tiles in separate loops.
        assert packed.head_dim == d and n_window % BLOCK_T == 0, (packed.head_dim, d, n_window)
        cmp_pool = window_pool  # placeholder for the unused BF16 compressed pointer
    cmp_pool = cmp_pool.contiguous()
    assert window_pool.shape[1] == d and cmp_pool.shape[1] == d, (window_pool.shape, cmp_pool.shape, d)
    # The BF16 kernel selects one pool base per column with a shared stride pair.
    assert window_pool.stride() == cmp_pool.stride(), (window_pool.stride(), cmp_pool.stride())
    cq, cs = (packed.q, packed.s) if packed is not None else (cmp_pool, cmp_pool)
    stride_cqn, stride_csn = (packed.q.stride(0), packed.s.stride(0)) if packed is not None else (0, 0)

    n_splits = split_count(b, m, h, topk, q.device)
    if n_splits:
        return _sparse_attn_paged_splitk(
            q, window_pool, cmp_pool, sink, idx, cnt, o,
            b, m, h, d, topk, n_window, softmax_scale, has_counts, stride_nb, stride_nm,
            n_splits, cq, cs, stride_cqn, stride_csn, packed is not None,
        )

    grid = (m, b, triton.cdiv(h, BLOCK_H))
    _sparse_attn_paged_kernel[grid](
        q, window_pool, cmp_pool, o, sink, idx, cnt, cq, cs,
        float(softmax_scale),
        h, topk, int(n_window),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        window_pool.stride(0), window_pool.stride(1),
        cmp_pool.stride(0), cmp_pool.stride(1),
        stride_cqn, stride_csn,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2),
        stride_nb, stride_nm,
        D=d,
        BLOCK_H=BLOCK_H,
        BLOCK_T=BLOCK_T,
        HAS_COUNTS=has_counts,
        PACKED=packed is not None,
        num_warps=8,
        num_stages=PACKED_STAGES if packed is not None else 2,
    )
    return o


def _sparse_attn_paged_splitk(
    q, window_pool, cmp_pool, sink, idx, cnt, o,
    b, m, h, d, topk, n_window, softmax_scale, has_counts, stride_nb, stride_nm, n_splits,
    cq, cs, stride_cqn, stride_csn, packed,
):
    head_blocks = triton.cdiv(h, BLOCK_H)
    mid_o = torch.empty((b, m, h, n_splits, d), dtype=torch.float32, device=q.device)
    mid_lse = torch.empty((b, m, h, n_splits), dtype=torch.float32, device=q.device)

    _sparse_attn_paged_splitk_kernel[(m * n_splits, b, head_blocks)](
        q, window_pool, cmp_pool, mid_o, mid_lse, idx, cnt, cq, cs,
        float(softmax_scale),
        h, topk, int(n_window),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        window_pool.stride(0), window_pool.stride(1),
        cmp_pool.stride(0), cmp_pool.stride(1),
        stride_cqn, stride_csn,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3), mid_o.stride(4),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2),
        stride_nb, stride_nm,
        D=d,
        BLOCK_H=BLOCK_H,
        BLOCK_T=BLOCK_T,
        HAS_COUNTS=has_counts,
        NUM_SPLITS=n_splits,
        PACKED=packed,
        num_warps=8,
        num_stages=PACKED_STAGES if packed else 2,
    )
    _sparse_attn_splitk_merge_kernel[(m, b, h)](
        mid_o, mid_lse, o, sink,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3), mid_o.stride(4),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        D=d,
        NUM_SPLITS=n_splits,
        num_warps=4,
    )
    return o


__all__ = ["sparse_attn_paged", "split_count"]
