"""CSA2 source-sharing topology and exact per-request cache arithmetic."""

from __future__ import annotations

from dataclasses import dataclass

from .args import DeepseekV41Args


@dataclass(frozen=True)
class LayerAttentionPlan:
    layer_id: int
    compress_ratio: int
    kv_source: int | None
    index_source: int | None
    owns_kv: bool
    owns_index: bool
    is_candidate_source: bool
    uses_candidates: bool


@dataclass(frozen=True)
class AttentionCacheBytes:
    window: int
    compressed_kv: int
    index_keys: int
    compressor_state: int

    @property
    def total(self) -> int:
        return self.window + self.compressed_kv + self.index_keys + self.compressor_state


class AttentionLayout:
    """Resolve every layer to the latest source published earlier in the stack."""

    def __init__(self, args: DeepseekV41Args) -> None:
        kv_sources = set(args.kv_source_layers)
        index_sources = set(args.index_source_layers)
        if not kv_sources.issubset(index_sources):
            raise ValueError("every V4.1 KV source must also publish index selections")
        plans = []
        active_kv = active_index = None
        for layer_id in range(args.n_layers):
            ratio = int(args.compress_ratios[layer_id])
            if layer_id in kv_sources:
                if ratio not in (1, 2):
                    raise ValueError(f"KV source layer {layer_id} has invalid ratio {ratio}")
                active_kv = layer_id
            if ratio == 0:
                active_kv = active_index = None
            elif active_kv is None:
                raise ValueError(f"compressed layer {layer_id} precedes its KV source")
            elif int(args.compress_ratios[active_kv]) != ratio:
                raise ValueError(
                    f"layer {layer_id} ratio {ratio} reuses source {active_kv} with "
                    f"ratio {args.compress_ratios[active_kv]}"
                )
            if layer_id in index_sources:
                if ratio == 0:
                    raise ValueError(f"index source layer {layer_id} is not compressed")
                active_index = layer_id
            if ratio and active_index is None:
                raise ValueError(f"compressed layer {layer_id} precedes its index source")
            plans.append(
                LayerAttentionPlan(
                    layer_id=layer_id,
                    compress_ratio=ratio,
                    kv_source=active_kv,
                    index_source=active_index,
                    owns_kv=layer_id in kv_sources,
                    owns_index=layer_id in index_sources,
                    is_candidate_source=layer_id == args.candidate_source_layer,
                    uses_candidates=(
                        args.candidate_source_layer >= 0
                        and layer_id > args.candidate_source_layer
                    ),
                )
            )
        self.args = args
        self.layers = tuple(plans)

    def __getitem__(self, layer_id: int) -> LayerAttentionPlan:
        return self.layers[layer_id]

    def cache_bytes(self, max_seq_len: int) -> AttentionCacheBytes:
        """Physical source-only buffers for one fully admitted request."""

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        from freetoken.kvcache.dsv41_packed_kv import (
            packed_kv_enabled_for_source,
            packed_row_bytes,
        )

        item = 2  # window KV and index keys are BF16 dequantized storage
        window = self.args.n_layers * self.args.window_size * self.args.head_dim * item
        compressed = index = state = 0
        for layer_id in self.args.kv_source_layers:
            ratio = int(self.args.compress_ratios[layer_id])
            if max_seq_len % ratio:
                raise ValueError(f"max_seq_len {max_seq_len} is not divisible by ratio {ratio}")
            rows = max_seq_len // ratio
            cmp_row = (
                packed_row_bytes(self.args.head_dim)
                if packed_kv_enabled_for_source(layer_id)
                else self.args.head_dim * item
            )
            compressed += rows * cmp_row
            index += rows * self.args.index_head_dim * item
            if ratio > 1:
                # FP32 KV and softmax-score carries, each [ratio, head_dim].
                state += 2 * ratio * self.args.head_dim * 4
        return AttentionCacheBytes(window, compressed, index, state)


def decoder_source_layer(args: DeepseekV41Args) -> int:
    """First decoder layer of V4.1's causal encoder-decoder split.

    This is the last compressed-KV source.  Its compressed KV and index keys are per-row
    functions of its own input, which is the encoder output, and every later layer reads them.
    A later index source computes queries only.  So a row of a later layer depends on other
    rows only through that layer's own sliding window.  Raises ValueError when the checkpoint's
    layer plan does not have this shape.
    """

    if not args.kv_source_layers:
        raise ValueError("V4.1 encoder-decoder split needs a compressed-KV source")
    layout = AttentionLayout(args)
    source = max(int(layer) for layer in args.kv_source_layers)
    if not 0 < source < args.n_layers:
        raise ValueError(f"last KV source {source} leaves no encoder or no decoder layer")
    for plan in layout.layers[source:]:
        if not plan.compress_ratio or plan.kv_source != source:
            raise ValueError(
                f"layer {plan.layer_id} does not read the last KV source {source}"
            )
    late_engram = [int(layer) for layer in args.engram_layer_ids if int(layer) >= source]
    if late_engram:
        raise ValueError(f"Engram layers {late_engram} are not encoder layers (< {source})")
    return source


__all__ = [
    "AttentionCacheBytes",
    "AttentionLayout",
    "LayerAttentionPlan",
    "decoder_source_layer",
]
