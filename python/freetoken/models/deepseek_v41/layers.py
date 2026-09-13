"""Dense 32x32-FP8 and normalization primitives for DeepSeek-V4.1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.kernel.triton.dsv41 import block_fp8_linear_32
from freetoken.utils import div_even


class Linear(nn.Module):
    """BF16/FP32 or V4.1 block-FP8 linear.

    The released text checkpoint uses E4M3 weights with E8M0 scales for every
    dense projection except the explicitly BF16 compressor/indexer/router pieces.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        kind: str = "fp8",
        parallel: str | None = None,
    ) -> None:
        super().__init__()
        if parallel not in (None, "column", "row"):
            raise ValueError(f"unsupported V4.1 linear parallel mode: {parallel}")
        tp = get_tp_info() if parallel is not None else None
        self.parallel = parallel
        self.tp_size = tp.size if tp is not None else 1
        self.in_features = (
            div_even(int(in_features), self.tp_size)
            if parallel == "row"
            else int(in_features)
        )
        self.out_features = (
            div_even(int(out_features), self.tp_size)
            if parallel == "column"
            else int(out_features)
        )
        self._comm = DistributedCommunicator() if parallel == "row" else None
        self.kind = kind
        if kind == "fp8":
            self.weight = nn.Parameter(
                torch.empty(
                    self.out_features,
                    self.in_features,
                    dtype=torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(
                    (self.out_features + 31) // 32,
                    (self.in_features + 31) // 32,
                    dtype=torch.float8_e8m0fnu,
                ),
                requires_grad=False,
            )
        elif kind in ("bf16", "fp32"):
            dtype = torch.bfloat16 if kind == "bf16" else torch.float32
            self.weight = nn.Parameter(
                torch.empty(self.out_features, self.in_features, dtype=dtype),
                requires_grad=False,
            )
            self.register_parameter("scale", None)
        else:
            raise ValueError(f"unsupported V4.1 linear kind: {kind}")
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "fp8":
            if not x.is_cuda:
                raise RuntimeError("V4.1 block-FP8 linear requires CUDA")
            output = block_fp8_linear_32(x, self.weight, self.scale, self.bias)
        else:
            output = F.linear(x, self.weight.to(x.dtype), self.bias)
        if self.parallel == "row" and self.tp_size > 1:
            assert self._comm is not None
            output = self._comm.all_reduce(output)
        return output


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from freetoken.kernel.triton.dsv4.norm import rms_norm

            return rms_norm(x, self.weight, self.eps)
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight.to(x.dtype)


__all__ = ["Linear", "RMSNorm"]
