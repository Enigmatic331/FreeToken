"""Symmetric AutoRound/GPTQ W4A16 group-128 routed-expert kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_TL = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


@triton.jit
def _decode_autoround_kernel(
    a_ptr, q_ptr, s_ptr, c_ptr, topk_weights_ptr, topk_ids_ptr,
    total_routes, N, K,
    stride_am, stride_ak,
    stride_qe, stride_qn, stride_qw,
    stride_se, stride_sn, stride_sg,
    stride_cm, stride_ck, stride_cn,
    stride_twm, stride_twk, stride_tidm, stride_tidk,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr, TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    route = tl.program_id(0)
    n_block = tl.program_id(1)
    token = route // TOP_K
    route_k = route - token * TOP_K
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    slot = tl.load(topk_ids_ptr + token * stride_tidm + route_k * stride_tidk).to(tl.int64)
    a_row = route if A_ROW_IS_ROUTE else token
    a_base = a_ptr + a_row * stride_am
    q_base = q_ptr + slot * stride_qe
    s_base = s_ptr + slot * stride_se

    k_words = K // 8
    offs_w = tl.arange(0, BLOCK_KW)
    partial = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
    for block in range(0, tl.cdiv(k_words, BLOCK_KW)):
        widx = block * BLOCK_KW + offs_w
        w_mask = widx < k_words
        word = tl.load(
            q_base + offs_n[None, :] * stride_qn + widx[:, None] * stride_qw,
            mask=w_mask[:, None] & n_mask[None, :], other=0,
        )
        group = widx // 16  # 16 int32 words * 8 nibbles = group size 128
        scale = tl.load(
            s_base + offs_n[None, :] * stride_sn + group[:, None] * stride_sg,
            mask=w_mask[:, None] & n_mask[None, :], other=0.0,
        ).to(tl.float32)
        kbase = 8 * widx
        acc_word = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
        for nibble in tl.static_range(8):
            code = ((word >> (4 * nibble)) & 0xF).to(tl.float32)
            act = tl.load(
                a_base + (kbase + nibble) * stride_ak,
                mask=w_mask, other=0.0,
            ).to(tl.float32)
            acc_word += act[:, None] * (code - 8.0)
        partial += acc_word * scale

    acc = tl.sum(partial, axis=0)
    if MUL_ROUTED_WEIGHT:
        acc *= tl.load(topk_weights_ptr + token * stride_twm + route_k * stride_twk)
    tl.store(
        c_ptr + token * stride_cm + route_k * stride_ck + offs_n * stride_cn,
        acc.to(compute_type), mask=(route < total_routes) & n_mask,
    )


def decode_gemm(a, q, s, c, topk_weights, topk_ids, *,
                mul_routed_weight: bool, a_row_is_route: bool) -> None:
    m, top_k = topk_ids.shape
    n, k = q.shape[1], q.shape[2] * 8
    block_n, block_kw = 16, 16
    _decode_autoround_kernel[(m * top_k, triton.cdiv(n, block_n))](
        a, q, s, c, topk_weights, topk_ids, m * top_k, n, k,
        a.stride(0), a.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_N=block_n, BLOCK_KW=block_kw, TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route, MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_TL.get(c.dtype, tl.bfloat16), num_warps=4,
    )


@triton.jit
def _prefill_autoround_kernel(
    a_ptr, q_ptr, s_ptr, c_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_qe, stride_qn, stride_qw,
    stride_se, stride_sn, stride_sg,
    stride_cm, stride_cn, stride_tw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr, compute_type: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    per_group = GROUP_M * num_pid_n
    group_id = pid // per_group
    first_m = group_id * GROUP_M
    group_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % per_group) % group_m)
    pid_n = (pid % per_group) // group_m
    if pid_m * BLOCK_M >= tl.load(num_tokens_post_padded_ptr):
        return

    offs_route = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    routes = tl.load(sorted_token_ids_ptr + offs_route).to(tl.int64)
    route_mask = routes < num_valid_tokens
    rows = routes // TOP_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    q_base = q_ptr + slot * stride_qe + offs_n[None, :] * stride_qn
    s_base = s_ptr + slot * stride_se + offs_n * stride_sn
    a_base = a_ptr + rows[:, None] * stride_am
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        kidx = kb * BLOCK_K + offs_k
        k_mask = kidx < K
        word_idx = kidx // 8
        shift = (kidx % 8) * 4
        word = tl.load(
            q_base + word_idx[:, None] * stride_qw,
            mask=k_mask[:, None] & n_mask[None, :], other=0,
        )
        code = ((word >> shift[:, None]) & 0xF).to(tl.float32)
        scale = tl.load(s_base + kb * stride_sg, mask=n_mask, other=0.0).to(tl.float32)
        weight = ((code - 8.0) * scale[None, :]).to(tl.bfloat16)
        act = tl.load(
            a_base + kidx[None, :] * stride_ak,
            mask=route_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(act, weight)

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            topk_weights_ptr + routes * stride_tw, mask=route_mask, other=0.0
        )
        acc *= routed_weight[:, None]
    tl.store(
        c_ptr + routes[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(compute_type),
        mask=route_mask[:, None] & n_mask[None, :],
    )


def prefill_gemm(a, q, s, c, topk_weights, sorted_ids, expert_ids,
                 ntpp, num_valid: int, *, kernel_top_k: int,
                 mul_routed_weight: bool) -> None:
    n, k = q.shape[1], q.shape[2] * 8
    block_m, block_n, block_k = 16, 64, 128
    em = sorted_ids.shape[0]
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, block_n),)
    _prefill_autoround_kernel[grid](
        a, q, s, c, topk_weights, sorted_ids, expert_ids, ntpp,
        n, k, em, num_valid,
        a.stride(0), a.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        c.stride(1), c.stride(2), topk_weights.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8,
        MUL_ROUTED_WEIGHT=mul_routed_weight, TOP_K=kernel_top_k,
        compute_type=_TL.get(c.dtype, tl.bfloat16), num_warps=4, num_stages=3,
    )


__all__ = ["decode_gemm", "prefill_gemm"]
