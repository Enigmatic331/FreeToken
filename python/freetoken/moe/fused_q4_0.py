"""Grouped expert GEMM over native GGUF block-quant banks (borrowed ggml kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed Q4_0 block bytes and
dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We
use the MMVQ (vector) kernel for both prefill and decode: it consumes ``topk_ids``
directly (no ``moe_align_block_size`` needed) and on small batches it is the right
choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_IQ2_XS, GGML_IQ3_XXS, GGML_Q4_0

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    gate_up_type: int,
    down_type: int,
    intermediate_size: int | None = None,
) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.kernel.llama_iq_mmq import grouped_iq_mmq, supported

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    if intermediate_size is None:
        if gate_up_q.ndim < 3:
            raise ValueError("flat GGUF expert caches require intermediate_size")
        intermediate_size = gate_up_q.shape[1] // 2
    n2 = 2 * intermediate_size
    h = hidden_states.shape[1]
    top_k = topk_ids.shape[1]
    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    if supported(gate_up_type, num_tokens):
        gate_up = grouped_iq_mmq(
            gate_up_q, hidden_states, topk_ids, int(gate_up_type), n2
        )
    else:
        gate_up = ggml_moe_a8_vec(
            hidden_states, gate_up_q, topk_ids, top_k, int(gate_up_type), n2, num_tokens
        )
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    if supported(down_type, num_tokens):
        out = grouped_iq_mmq(
            down_q, inter, topk_ids.reshape(-1, 1), int(down_type), h
        )
    else:
        out = ggml_moe_a8_vec(
            inter, down_q, topk_ids, 1, int(down_type), h, num_tokens * top_k
        )
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def fused_experts_gguf_q4_0(*args, **kwargs) -> torch.Tensor:
    return fused_experts_gguf(
        *args, **kwargs, gate_up_type=GGML_Q4_0, down_type=GGML_Q4_0
    )


def fused_experts_gguf_q2_k_xl(
    *args,
    gate_up_type: int = GGML_IQ2_XS,
    down_type: int = GGML_IQ3_XXS,
    **kwargs,
) -> torch.Tensor:
    """Unsloth UD-Q2_K_XL, including its higher-bit exceptional layers."""
    return fused_experts_gguf(
        *args, **kwargs, gate_up_type=gate_up_type, down_type=down_type
    )


__all__ = [
    "fused_experts_gguf",
    "fused_experts_gguf_q4_0",
    "fused_experts_gguf_q2_k_xl",
]
