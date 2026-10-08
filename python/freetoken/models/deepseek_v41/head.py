"""Optional native-BF16 head storage with FP32 logits, without requantization."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def native_bf16_head_enabled() -> bool:
    value = os.environ.get("FREETOKEN_DSV41_NATIVE_BF16_HEAD", "0")
    if value not in ("0", "1"):
        raise ValueError("FREETOKEN_DSV41_NATIVE_BF16_HEAD must be 0 or 1")
    return value == "1"


def project_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Keep FP32 output and all leading dimensions; never round FP32 inputs down.

    BF16-input batched GEMM failed our real-checkpoint numerical gate, whereas
    the one-row primitive passed. Use that primitive for every row. This is
    intended for ordinary single-request generation, not MTP acceleration.
    No global precision flags are changed, including during graph capture.
    """
    if weight.dtype == torch.float32:
        return F.linear(hidden.float(), weight)
    if weight.dtype != torch.bfloat16 or hidden.dtype != torch.bfloat16:
        raise TypeError("native BF16 head requires BF16 weights and hidden states")
    if hidden.ndim < 1 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("invalid native BF16 head projection shape")
    if not hidden.is_cuda or weight.device != hidden.device:
        raise ValueError("native BF16 head requires inputs and weights on the same CUDA device")
    rows = hidden.reshape(-1, weight.shape[1])
    shape = (*hidden.shape[:-1], weight.shape[0])
    if rows.shape[0] == 0:
        return torch.empty(shape, dtype=torch.float32, device=hidden.device)
    if rows.shape[0] == 1:
        return torch.mm(rows, weight.t(), out_dtype=torch.float32).reshape(shape)
    return torch.cat([
        torch.mm(rows[i:i + 1], weight.t(), out_dtype=torch.float32)
        for i in range(rows.shape[0])
    ]).reshape(shape)
