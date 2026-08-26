"""Process-local execution roles for heterogeneous DeepSeek-V4 expert parallelism.

The default plan keeps the established replicated-backbone EP path.  Supplying a
backbone authority rank switches every other rank to expert-worker mode: the
authority executes attention / HC / shared experts, broadcasts each layer's
normalized MoE input, and all ranks compute only their owned routed experts.
"""

from __future__ import annotations

from dataclasses import dataclass

from freetoken.attention.base import BaseAttnBackend
from freetoken.distributed import get_tp_info


@dataclass(frozen=True, slots=True)
class DSV4ExecutionPlan:
    rank: int
    world_size: int
    backbone_rank: int | None = None
    expert_shards: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")
        if self.backbone_rank is not None and not 0 <= self.backbone_rank < self.world_size:
            raise ValueError(
                f"backbone_rank must be in [0, {self.world_size}), got {self.backbone_rank}"
            )
        if self.expert_shards is not None:
            shards = tuple(self.expert_shards)
            object.__setattr__(self, "expert_shards", shards)
            if len(shards) != self.world_size:
                raise ValueError(
                    "expert_shards must contain exactly one count per rank: "
                    f"expected {self.world_size}, got {len(shards)}"
                )

    @property
    def enabled(self) -> bool:
        return self.backbone_rank is not None and self.world_size > 1

    @property
    def is_backbone(self) -> bool:
        return not self.enabled or self.rank == self.backbone_rank

    @property
    def is_expert_worker(self) -> bool:
        return self.enabled and self.rank != self.backbone_rank


class ExpertWorkerAttentionBackend(BaseAttnBackend):
    """Scheduler-compatible attention shell for ranks that execute no attention."""

    def forward(self, q, k, v, layer_id, batch, attn_spec=None):
        raise RuntimeError("expert-worker ranks do not execute attention")

    def prepare_metadata(self, batch) -> None:
        # ExpertWorkerTransformer consumes only input ids and the prefill/decode phase.
        batch.attn_metadata = None

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        pass

    def prepare_for_capture(self, batch) -> None:
        pass

    def prepare_for_replay(self, batch) -> None:
        pass


_PLAN: DSV4ExecutionPlan | None = None


def configure_dsv4_execution(
    backbone_rank: int | None,
    expert_shards: tuple[int, ...] | None = None,
) -> DSV4ExecutionPlan:
    """Configure this engine process once TP rank information is available."""
    global _PLAN
    tp = get_tp_info()
    plan = DSV4ExecutionPlan(tp.rank, tp.size, backbone_rank, expert_shards)
    if _PLAN is not None and _PLAN != plan:
        raise RuntimeError(f"DSV4 execution plan already configured as {_PLAN}, got {plan}")
    _PLAN = plan
    return plan


def get_dsv4_execution_plan() -> DSV4ExecutionPlan:
    """Return the configured plan, defaulting to legacy replicated execution."""
    if _PLAN is not None:
        return _PLAN
    tp = get_tp_info()
    return DSV4ExecutionPlan(tp.rank, tp.size)


def _reset_dsv4_execution_for_tests() -> None:
    global _PLAN
    _PLAN = None


__all__ = [
    "DSV4ExecutionPlan",
    "ExpertWorkerAttentionBackend",
    "configure_dsv4_execution",
    "get_dsv4_execution_plan",
]
