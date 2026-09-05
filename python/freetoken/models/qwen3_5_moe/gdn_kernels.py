from __future__ import annotations

import os

import torch


def gdn_prefill_chunk_fla(
    q: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, total, num_v_heads, head_v_dim] bf16
    g: torch.Tensor,        # [1, total, num_v_heads] log-decay (<=0), fp32
    beta: torch.Tensor,     # [1, total, num_v_heads] fp32
    *,
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,       # [num_seqs] slot id per sequence
    cu_seqlens: torch.Tensor,    # [num_seqs+1] int64
    scale: float,
    return_h: bool = False,
    h_rows: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Chunked gated-delta-rule prefill via the vendored fla kernel. GQA is handled
    in-kernel (q/k at num_k_heads), q/k l2norm is done in-kernel, and the per-sequence
    recurrent state is read from and written back to ``state_source[indices]`` IN PLACE
    (no external l2norm, no Python stack of initial states, no copy_ writeback loop).
    Fresh sequences must have their ``state_source`` slot pre-zeroed by the caller.
    Returns ``o`` of shape ``[total, num_v_heads, head_v_dim]`` (bf16).

    When ``return_h=True`` also returns the per-chunk hidden-state buffer ``h``. If ``h_rows``
    is supplied, only those global chunk rows are returned, in the same order; otherwise the
    full ``[1, NT_total, num_v_heads, head_v_dim, head_k_dim]`` buffer is returned. Global row
    ``boh_i + c`` is the recurrent state after ``c*64`` tokens of packed sequence ``i`` (chunk
    granularity 64), where ``boh_i = prepare_chunk_offsets(cu_seqlens, 64)[i]``. The last two
    dims are ``[V, K]`` -- transposed vs ``state_source``'s ``[K, V]``. The row-select form keeps
    tiled hybrid-radix prefills memory-bounded instead of retaining and re-concatenating every
    tile's checkpoint slab."""
    if h_rows is not None and not return_h:
        raise ValueError("h_rows requires return_h=True")
    if h_rows is not None and tuple(sorted(h_rows)) != h_rows:
        raise ValueError("h_rows must be sorted")
    from freetoken.kernel.fla import chunk_gated_delta_rule

    def run(
        q_tile: torch.Tensor,
        k_tile: torch.Tensor,
        v_tile: torch.Tensor,
        g_tile: torch.Tensor,
        beta_tile: torch.Tensor,
        tile_cu_seqlens: torch.Tensor,
    ):
        return chunk_gated_delta_rule(
            q=q_tile, k=k_tile, v=v_tile, g=g_tile, beta=beta_tile, scale=scale,
            initial_state=state_source, initial_state_indices=indices.to(torch.int32),
            cu_seqlens=tile_cu_seqlens.to(torch.int64), head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

    tile_tokens = int(os.getenv("FREETOKEN_GDN_PREFILL_TILE_TOKENS", "0"))
    total = q.shape[1]
    # The server runs one request at a time. For that single packed sequence, the
    # FLA kernel's in-place final-state writeback is exactly the initial state for
    # the next 64-token-aligned tile. This bounds h=[B, NT, H, V, K] without
    # changing the recurrence. Keep ragged/multi-sequence batches on the existing
    # path until their per-sequence tiling metadata is implemented.
    tiled = tile_tokens > 0 and total > tile_tokens and cu_seqlens.numel() == 2
    if tiled:
        if tile_tokens % 64 != 0:
            raise ValueError("FREETOKEN_GDN_PREFILL_TILE_TOKENS must be a multiple of 64")
        # Write each tile into its final output slab immediately. Accumulating o_tile tensors
        # and torch.cat'ing them doubled the output peak; doing the same for h was worse (about
        # 492 MiB per 20k-token Qwen3.8 prefill) and caused rank-0 OOMs after tool continuations.
        o = torch.empty_like(v)
        states = [] if return_h else None
        h_row_offset = 0
        for start in range(0, total, tile_tokens):
            end = min(start + tile_tokens, total)
            tile_cu_seqlens = cu_seqlens.new_tensor((0, end - start))
            o_tile, _, h_tile = run(
                q[:, start:end], k[:, start:end], v[:, start:end],
                g[:, start:end], beta[:, start:end], tile_cu_seqlens,
            )
            o[:, start:end].copy_(o_tile)
            if states is not None:
                if h_rows is None:
                    states.append(h_tile)
                else:
                    tile_rows = [
                        row - h_row_offset
                        for row in h_rows
                        if h_row_offset <= row < h_row_offset + h_tile.shape[1]
                    ]
                    if tile_rows:
                        states.append(h_tile[:, tile_rows])
                h_row_offset += h_tile.shape[1]
        if states is None:
            h = None
        else:
            if h_rows is not None and sum(part.shape[1] for part in states) != len(h_rows):
                raise IndexError(f"h_rows {h_rows} are outside the {h_row_offset} chunk rows")
            if not states:
                raise ValueError("return_h=True requires at least one h row")
            h = states[0] if len(states) == 1 else torch.cat(states, dim=1)
    else:
        o, _, h = run(q, k, v, g, beta, cu_seqlens)
        if return_h and h_rows is not None:
            h = h[:, list(h_rows)]
    if return_h:
        assert h is not None
        return o[0], h
    return o[0]  # [total, num_v_heads, head_v_dim]


def gdn_decode_fla(
    q: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, B, num_v_heads, head_v_dim] bf16
    a: torch.Tensor,        # [B, num_v_heads] raw
    b: torch.Tensor,        # [B, num_v_heads] raw
    *,
    A_log: torch.Tensor,        # [num_v_heads]
    dt_bias: torch.Tensor,      # [num_v_heads]
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,      # [B] int32 slot id per request
    cu_seqlens: torch.Tensor,   # [B+1] query indptr (arange) from FLAMetadata
    scale: float,
) -> torch.Tensor:
    """Fused sigmoid-gating gated-delta-rule decode (vendored fla triton kernel): gating +
    in-kernel l2norm + recurrent update + state read/write-by-index in one kernel, with no
    external gating or gather/scatter/clone glue. Returns [B, num_v, V]."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,  # already fp32 (stored fp32)
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=state_source,
        initial_state_indices=indices,  # already int32 (built int32 in the scheduler)
        scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
    )
    # kernel returns o = [NK, *v.shape] then squeeze(NK) -> [1, B, num_v, V].
    # o[0] -> [B, num_v, V] (all B decode tokens; o[0,0] would drop B>1).
    return o[0]


__all__ = ["gdn_prefill_chunk_fla", "gdn_decode_fla"]
