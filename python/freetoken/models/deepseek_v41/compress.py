"""CSA2 compressed-KV pooling math for DeepSeek-V4.1."""

from __future__ import annotations

import torch
from torch import nn

from freetoken.core import get_global_ctx

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
        # Address-stable rejection journal for the captured DSpark verifier.
        # V4.1 ships at most ``dspark_block_size + 1`` target rows.  Ratio-1 has
        # no rolling state, but keeping a zero-size buffer makes the interface
        # uniform and costs no storage.
        journal_shape = (
            int(args.dspark_block_size) + 1,
            self.compress_ratio if self.compress_ratio > 1 else 0,
            2 * args.head_dim,
        )
        self.register_buffer(
            "spec_graph_journal",
            torch.empty(journal_shape, dtype=torch.float32),
            persistent=False,
        )

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
        try:
            active_batch = get_global_ctx().batch
        except AssertionError:
            # The compressor is also exercised as a standalone numerical unit.
            active_batch = None
        speculative = bool(getattr(active_batch, "speculative", False))
        if start_pos % ratio and not speculative:
            raise ValueError(
                f"paged prefill start {start_pos} is not ratio-{ratio} aligned"
            )
        if ratio == 1:
            latent = self.norm(self.wkv(x))
        elif speculative:
            # A DSpark verify resumes at the live decode frontier, which can be at
            # either parity, and then walks anchor + proposals in one short prefill.
            # Advance the request's page-local carry exactly as repeated decode
            # steps would.  Save the complete carry after every input row so target
            # rejection can restore the state selected by its accepted prefix.
            assert self.wgate is not None
            values, scores = self.wkv(x.float()), self.wgate(x.float())
            completed_values = []
            block_starts = []
            journal = active_batch.spec_carry_states
            if journal is None:
                raise RuntimeError("DSpark verify has no compressor carry journal")
            pieces = journal.setdefault((layer_id, "attn", ratio), [])
            last_window_page = None
            block = None
            for row in range(x.shape[1]):
                pos = start_pos + row
                window_slot = int(window_slots[row].item())
                window_page = window_slot // backend.window_size
                # A window page owns its own carry block. Read it when entering a
                # page; subsequent rows in that page advance the local clone.
                if window_page != last_window_page:
                    block = backend.read_carry(
                        layer_id, "attn", window_slot, ratio
                    ).clone()
                    last_window_page = window_page
                assert block is not None
                slot = pos % ratio
                block[slot, : self.head_dim] = values[0, row]
                block[slot, self.head_dim :] = scores[0, row]
                backend.write_carry(
                    layer_id, "attn", window_slot, ratio, block
                )
                pieces.append(block.unsqueeze(0).clone())
                if slot == ratio - 1:
                    pooled = (
                        block[:, : self.head_dim]
                        * block[:, self.head_dim :].softmax(0)
                    ).sum(0)
                    completed_values.append(pooled)
                    block_starts.append(pos + 1 - ratio)
            if completed_values:
                latent = self.norm(
                    torch.stack(completed_values, dim=0)
                    .to(x.dtype)
                    .unsqueeze(0)
                )
            else:
                latent = x.new_empty((1, 0, self.head_dim))
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
        if speculative and ratio > 1:
            starts = torch.tensor(block_starts, dtype=torch.long, device=x.device)
        else:
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

    def verify_paged(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        window_slots: torch.Tensor,
        *,
        layer_id: int,
        backend,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Graph-safe compressor pass over one fixed speculative span.

        ``x`` is ``[1, T, dim]``.  Ratio-2 advances its carry in query order,
        snapshots every post-row state into a persistent device journal, and
        routes incomplete groups to the pool's discard row.  The graph caller
        rejects spans crossing a window page, so one initial carry read and one
        final write exactly match eager row-by-row progression.
        """

        ratio = self.compress_ratio
        if ratio == 1:
            latent = self.norm(self.wkv(x))[0]
            completed = torch.ones_like(positions, dtype=torch.bool)
        else:
            assert self.wgate is not None and ratio == 2
            values = self.wkv(x.float())[0]
            scores = self.wgate(x.float())[0]
            block = backend.read_carry_blocks(
                layer_id, "attn", window_slots[:1], ratio
            )[0].clone()
            pooled_rows = []
            completed_rows = []
            for row in range(x.shape[1]):
                slot = torch.remainder(positions[row : row + 1], ratio)
                value_score = torch.cat(
                    [values[row : row + 1], scores[row : row + 1]], dim=-1
                )
                block.index_copy_(0, slot, value_score)
                self.spec_graph_journal[row].copy_(block)
                pooled_rows.append(
                    (
                        block[:, : self.head_dim]
                        * block[:, self.head_dim :].softmax(0)
                    ).sum(0)
                )
                completed_rows.append(slot[0] == ratio - 1)
            backend.write_carry_blocks(
                layer_id,
                "attn",
                window_slots[-1:],
                ratio,
                block.unsqueeze(0),
            )
            latent = self.norm(torch.stack(pooled_rows).to(x.dtype))
            completed = torch.stack(completed_rows)
        destinations = backend.verify_compress_rows(
            positions, ratio, layer_id, "attn", completed
        )
        return latent, destinations, completed

    def restore_graph_carry(
        self, selected_row: int, window_slot: int, *, layer_id: int, backend
    ) -> None:
        """Roll a captured ratio-2 verify back to its accepted target row."""

        if self.compress_ratio == 1:
            return
        backend.write_carry(
            layer_id,
            "attn",
            window_slot,
            self.compress_ratio,
            self.spec_graph_journal[selected_row],
        )


__all__ = ["Compressor"]
