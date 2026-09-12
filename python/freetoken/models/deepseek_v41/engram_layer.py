"""DeepSeek-V4.1 Engram residual injection using a pluggable row store."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from .args import DeepseekV41Args
from .layers import Linear


class EngramTable(Protocol):
    def lookup(self, row_ids: torch.Tensor, **kwargs) -> torch.Tensor: ...

    def prefetch(self, row_ids: torch.Tensor) -> None: ...


class Engram(nn.Module):
    """Gate one table lookup into each of the four HC residual streams."""

    def __init__(self, args: DeepseekV41Args, layer_id: int) -> None:
        super().__init__()
        if layer_id not in args.engram_layer_ids:
            raise ValueError(f"layer {layer_id} is not an Engram layer")
        self.layer_id = int(layer_id)
        self.layer_hash_index = args.engram_layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.eps = args.norm_eps
        self.clamp_value = 1e-6
        input_dim = args.engram_hashes_per_token * args.engram_head_dim
        self.wkv = Linear(input_dim, args.dim * (args.hc_mult + 1))
        self.q_weight = nn.Parameter(
            torch.ones(args.hc_mult, args.dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.k_weight = nn.Parameter(
            torch.ones(args.hc_mult, args.dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.table: EngramTable | None = None

    def attach_table(self, table: EngramTable) -> None:
        self.table = table

    def prefetch(self, row_ids: torch.Tensor) -> None:
        if self.table is None:
            raise RuntimeError(f"Engram layer {self.layer_id} has no table backend")
        self.table.prefetch(row_ids)

    def forward(
        self,
        x: torch.Tensor,
        row_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.table is None:
            raise RuntimeError(f"Engram layer {self.layer_id} has no table backend")
        rows = self.table.lookup(row_ids)
        kv = self.wkv(rows.flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        hidden = x.float()
        weight = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(hidden.square().mean(-1) + self.eps)
        rstd *= torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hidden * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


__all__ = ["Engram", "EngramTable"]
