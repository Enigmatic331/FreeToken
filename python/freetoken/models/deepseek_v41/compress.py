"""CSA2 compressed-KV pooling math for DeepSeek-V4.1."""

from __future__ import annotations

import torch
from torch import nn

from .args import DeepseekV41Args
from .layers import Linear, RMSNorm


class Compressor(nn.Module):
    """Pool ratio-2 KV groups, or project directly for the ratio-1 tier."""

    def __init__(self, args: DeepseekV41Args, layer_id: int) -> None:
        super().__init__()
        self.compress_ratio = int(args.compress_ratios[layer_id])
        if self.compress_ratio not in (1, 2):
            raise ValueError(f"V4.1 compressor requires ratio 1 or 2, got {self.compress_ratio}")
        self.head_dim = args.head_dim
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        kind = "fp32" if self.compress_ratio > 1 else "bf16"
        self.wkv = Linear(args.dim, args.head_dim, kind=kind)
        if self.compress_ratio > 1:
            self.wgate = Linear(args.dim, args.head_dim, kind="fp32")
            shape = (args.max_batch_size, self.compress_ratio, args.head_dim)
            self.register_buffer("kv_state", torch.zeros(shape, dtype=torch.float32), persistent=False)
            self.register_buffer(
                "score_state",
                torch.full(shape, -torch.inf, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.wgate = None

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        batch, seqlen, _ = x.shape
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:
            return self.norm(self.wkv(x))
        assert self.wgate is not None
        values, scores = self.wkv(x.float()), self.wgate(x.float())
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                values, self.kv_state[:batch, :remainder] = values.split(
                    [cutoff, remainder], dim=1
                )
                scores, self.score_state[:batch, :remainder] = scores.split(
                    [cutoff, remainder], dim=1
                )
            values = values.unflatten(1, (-1, ratio))
            scores = scores.unflatten(1, (-1, ratio))
            values = (values * scores.softmax(2)).sum(2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:batch, slot] = values.squeeze(1)
            self.score_state[:batch, slot] = scores.squeeze(1)
            if should_compress:
                weights = self.score_state[:batch].softmax(1)
                values = (self.kv_state[:batch] * weights).sum(1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(values.to(dtype))

    def prefill_paged(
        self,
        x: torch.Tensor,
        start_pos: int,
        window_slots: torch.Tensor,
        *,
        layer_id: int,
        table_idx: int,
        backend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress one page-aligned single-request segment into source rows.

        Returns pre-RoPE latents and their physical compressed rows.  Ratio-2's
        pending odd token is persisted in the tail window page, so decode can
        resume without a request-indexed model buffer.
        """

        if x.shape[0] != 1:
            raise ValueError("V4.1 paged prefill currently requires one request")
        ratio = self.compress_ratio
        if start_pos % ratio:
            raise ValueError(
                f"paged prefill start {start_pos} is not ratio-{ratio} aligned"
            )
        if ratio == 1:
            latent = self.norm(self.wkv(x))
        else:
            assert self.wgate is not None
            values, scores = self.wkv(x.float()), self.wgate(x.float())
            complete = x.shape[1] // ratio
            cutoff = complete * ratio
            if complete:
                grouped_v = values[:, :cutoff].unflatten(1, (complete, ratio))
                grouped_s = scores[:, :cutoff].unflatten(1, (complete, ratio))
                latent = self.norm(
                    (grouped_v * grouped_s.softmax(2)).sum(2).to(x.dtype)
                )
            else:
                latent = x.new_empty((1, 0, self.head_dim))
            if x.shape[1] % ratio:
                block = torch.cat(
                    [
                        values.new_zeros(ratio, self.head_dim),
                        scores.new_full((ratio, self.head_dim), -torch.inf),
                    ],
                    dim=-1,
                )
                block[0, : self.head_dim] = values[0, -1]
                block[0, self.head_dim :] = scores[0, -1]
                backend.write_carry(
                    layer_id,
                    "attn",
                    int(window_slots[-1].item()),
                    ratio,
                    block,
                )
        starts = start_pos + torch.arange(
            0, latent.shape[1] * ratio, ratio, device=x.device
        )
        rows = backend.compress_rows_of(table_idx, starts, ratio)
        return latent, rows

    def decode_paged(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        window_slots: torch.Tensor,
        rows: torch.Tensor,
        *,
        layer_id: int,
        backend,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance source compression for a batched one-token decode step."""

        ratio = self.compress_ratio
        if ratio == 1:
            latent = self.norm(self.wkv(x))
            completed = torch.ones_like(positions, dtype=torch.bool)
        else:
            assert self.wgate is not None
            values = self.wkv(x.float()).squeeze(1)
            scores = self.wgate(x.float()).squeeze(1)
            blocks = backend.read_carry_blocks(
                layer_id, "attn", window_slots, ratio
            )
            slot = torch.remainder(positions, ratio)
            row_ids = torch.arange(x.shape[0], device=x.device)
            blocks[row_ids, slot, : self.head_dim] = values
            blocks[row_ids, slot, self.head_dim :] = scores
            backend.write_carry_blocks(
                layer_id, "attn", window_slots, ratio, blocks
            )
            pooled = (
                blocks[..., : self.head_dim]
                * blocks[..., self.head_dim :].softmax(1)
            ).sum(1)
            latent = self.norm(pooled.to(x.dtype)).unsqueeze(1)
            completed = slot == ratio - 1
        destinations = backend.decode_compress_rows(
            rows,
            positions,
            ratio,
            layer_id,
            "attn",
            completed,
        )
        return latent, destinations, completed


__all__ = ["Compressor"]
