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
) -> torch.Tensor:
    """Chunked gated-delta-rule prefill via the vendored fla kernel. GQA is handled
    in-kernel (q/k at num_k_heads), q/k l2norm is done in-kernel, and the per-sequence
    recurrent state is read from and written back to ``state_source[indices]`` IN PLACE
    (no external l2norm, no Python stack of initial states, no copy_ writeback loop).
    Fresh sequences must have their ``state_source`` slot pre-zeroed by the caller.
    Returns ``o`` of shape ``[total, num_v_heads, head_v_dim]`` (bf16).

    When ``return_h=True`` also returns the per-chunk hidden-state buffer ``h`` of shape
    ``[1, NT_total, num_v_heads, head_v_dim, head_k_dim]`` (bf16). ``h[0, boh_i + c]`` is the
    recurrent state after ``c*64`` tokens of packed sequence ``i`` (chunk granularity 64), where
    ``boh_i = prepare_chunk_offsets(cu_seqlens, 64)[i]``. Note the last two dims are ``[V, K]`` --
    transposed vs ``state_source``'s ``[K, V]``. Used by the hybrid-radix track-checkpoint path."""
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
        outputs = []
        states = [] if return_h else None
        for start in range(0, total, tile_tokens):
            end = min(start + tile_tokens, total)
            tile_cu_seqlens = cu_seqlens.new_tensor((0, end - start))
            o_tile, _, h_tile = run(
                q[:, start:end], k[:, start:end], v[:, start:end],
                g[:, start:end], beta[:, start:end], tile_cu_seqlens,
            )
            outputs.append(o_tile)
            if states is not None:
                states.append(h_tile)
        o = torch.cat(outputs, dim=1)
        h = torch.cat(states, dim=1) if states is not None else None
    else:
        o, _, h = run(q, k, v, g, beta, cu_seqlens)
    if return_h:
        assert h is not None
        return o[0], h  # h: [1, NT_total, num_v_heads, head_v_dim, head_k_dim]
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
