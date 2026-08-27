"""Contiguous pipeline ownership for GLM-5.2.

The public ``--glm-pipeline-parallel`` switch reinterprets the existing TP worker
group as pipeline stages.  Dense/attention weights, KV, and routed-expert banks are
owned by contiguous layer ranges; routed experts are *not* expert-parallel inside a
stage.  Boundaries are chosen at a full IndexShare layer so a stage never depends on
selection state produced by the preceding stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from freetoken.distributed import try_get_tp_info


_ENABLED = False
_PREFILL_MICROBATCH_TOKENS = 0
_BOUNDARIES: tuple[int, ...] | None = None


def configure_glm_pipeline(
    enabled: bool,
    prefill_microbatch_tokens: int = 0,
    boundaries: tuple[int, ...] | None = None,
) -> None:
    global _ENABLED, _PREFILL_MICROBATCH_TOKENS, _BOUNDARIES
    _ENABLED = bool(enabled)
    _PREFILL_MICROBATCH_TOKENS = int(prefill_microbatch_tokens)
    _BOUNDARIES = boundaries


@dataclass(frozen=True)
class GlmPipelinePlan:
    enabled: bool
    rank: int
    world_size: int
    start_layer: int
    stop_layer: int
    num_layers: int
    prefill_microbatch_tokens: int = 0

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.start_layer, self.stop_layer))

    @property
    def is_first(self) -> bool:
        return not self.enabled or self.rank == 0

    @property
    def is_last(self) -> bool:
        return not self.enabled or self.rank == self.world_size - 1

    @property
    def final_rank(self) -> int:
        return self.world_size - 1 if self.enabled else 0

    def owns_layer(self, layer_id: int) -> bool:
        return self.start_layer <= layer_id < self.stop_layer

    def moe_index(self, layer_id: int, first_moe_layer: int) -> int:
        """Rank-local cache/bank index for a global MoE layer id."""
        first = max(self.start_layer, first_moe_layer)
        if not first <= layer_id < self.stop_layer:
            raise ValueError(
                f"layer {layer_id} is not a local MoE layer in "
                f"[{first}, {self.stop_layer})"
            )
        return layer_id - first


def _boundaries(num_layers: int, world_size: int, indexer_types: tuple[str, ...]) -> list[int]:
    if world_size < 1 or world_size > num_layers:
        raise ValueError(
            f"GLM pipeline size {world_size} must be in [1, {num_layers}]"
        )
    if world_size == 1:
        return [0, num_layers]

    full = [
        i for i, kind in enumerate(indexer_types[:num_layers])
        if i > 0 and kind == "full"
    ]
    if _BOUNDARIES is not None:
        if len(_BOUNDARIES) != world_size - 1:
            raise ValueError(
                "--glm-pipeline-boundaries needs one cut between each pair of stages: "
                f"expected {world_size - 1}, got {len(_BOUNDARIES)}"
            )
        cuts = [0, *_BOUNDARIES, num_layers]
        if cuts != sorted(set(cuts)):
            raise ValueError(
                "--glm-pipeline-boundaries must be strictly increasing internal layers"
            )
        unsafe = [cut for cut in _BOUNDARIES if cut not in full]
        if unsafe:
            raise ValueError(
                "GLM pipeline boundaries must begin full IndexShare layers; "
                f"unsafe cuts: {unsafe}"
            )
        return cuts
    cuts = [0]
    for stage in range(1, world_size):
        ideal = round(stage * num_layers / world_size)
        candidates = [i for i in full if cuts[-1] < i < num_layers]
        if not candidates:
            raise ValueError(
                "GLM pipeline cannot find an IndexShare-safe boundary after "
                f"layer {cuts[-1]}"
            )
        cut = min(candidates, key=lambda i: (abs(i - ideal), i))
        cuts.append(cut)
    cuts.append(num_layers)
    if len(set(cuts)) != len(cuts):
        raise ValueError(f"duplicate GLM pipeline boundaries: {cuts}")
    return cuts


def glm_pipeline_plan(num_layers: int, indexer_types: tuple[str, ...]) -> GlmPipelinePlan:
    info = try_get_tp_info()
    if info is None:
        # ModelConfig is first parsed in the parent process, before worker ranks are
        # initialized.  Engine._adjust_config resolves the real stage in each worker.
        rank, world_size = 0, 1
    else:
        rank, world_size = info.rank, info.size
    enabled = _ENABLED and world_size > 1
    if not enabled:
        return GlmPipelinePlan(False, rank, world_size, 0, num_layers, num_layers, 0)
    cuts = _boundaries(num_layers, world_size, indexer_types)
    return GlmPipelinePlan(
        True,
        rank,
        world_size,
        cuts[rank],
        cuts[rank + 1],
        num_layers,
        _PREFILL_MICROBATCH_TOKENS,
    )


__all__ = [
    "GlmPipelinePlan",
    "configure_glm_pipeline",
    "glm_pipeline_plan",
]
