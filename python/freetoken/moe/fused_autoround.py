"""Fused routed-MoE wrappers for AutoRound/GPTQ symmetric INT4 group-128."""

from __future__ import annotations

import torch


def fused_experts_decode_autoround(
    hidden_states, gate_up_packed, gate_up_scale,
    down_packed, down_scale, topk_weights, topk_ids,
    num_experts=None, activation="silu", apply_router_weight_on_input=False,
):
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.kernel.triton.autoround_fused_moe import decode_gemm
    from freetoken.layers import silu_and_mul

    del num_experts
    assert activation == "silu"
    assert not apply_router_weight_on_input
    m, h = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    ic1 = torch.empty((m, top_k, two_i), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    decode_gemm(hidden_states, gate_up_packed, gate_up_scale,
                ic1, topk_weights, topk_ids,
                mul_routed_weight=False, a_row_is_route=False)
    ic2 = torch.empty((m * top_k, inter), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    silu_and_mul(ic1.view(-1, two_i), ic2)
    ic3 = torch.empty((m, top_k, h), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    decode_gemm(ic2, down_packed, down_scale, ic3,
                topk_weights, topk_ids,
                mul_routed_weight=True, a_row_is_route=True)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_autoround(
    hidden_states, gate_up_packed, gate_up_scale,
    down_packed, down_scale, topk_weights, topk_ids,
    num_experts, activation="silu", apply_router_weight_on_input=False,
):
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.kernel.triton.autoround_fused_moe import prefill_gemm
    from freetoken.layers import silu_and_mul
    from freetoken.moe.fused import moe_align_block_size

    assert activation == "silu"
    assert not apply_router_weight_on_input
    assert num_experts is not None
    m, h = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    block_m = 16
    sorted_ids, expert_ids, ntpp = moe_align_block_size(
        topk_ids, block_m, num_experts
    )
    routed_weights = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()
    ic1 = torch.empty((m, top_k, two_i), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    prefill_gemm(hidden_states, gate_up_packed, gate_up_scale,
                 ic1, routed_weights, sorted_ids, expert_ids, ntpp, num_valid,
                 kernel_top_k=top_k, mul_routed_weight=False)
    ic2 = torch.empty((m * top_k, inter), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    silu_and_mul(ic1.view(-1, two_i), ic2)
    ic3 = torch.empty((m, top_k, h), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    prefill_gemm(ic2, down_packed, down_scale, ic3,
                 routed_weights, sorted_ids, expert_ids, ntpp, num_valid,
                 kernel_top_k=1, mul_routed_weight=True)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = ["fused_experts_autoround", "fused_experts_decode_autoround"]
