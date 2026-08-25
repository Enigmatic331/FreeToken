"""Process-local execution roles for heterogeneous DeepSeek-V4 expert parallelism.

The default plan keeps the established replicated-backbone EP path.  Supplying a
backbone authority rank switches every other rank to expert-worker mode: the
authority executes attention / HC / shared experts, broadcasts each layer's
normalized MoE input, and all ranks compute only their owned routed experts.
"""

from __future__ import annotations

from dataclasses import dataclass

from freetoken.distributed import get_tp_info


@dataclass(frozen=True, slots=True)
class DSV4ExecutionPlan:
    rank: int
    world_size: int
    backbone_rank: int | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")
        if self.backbone_rank is not None and not 0 <= self.backbone_rank < self.world_size:
            raise ValueError(
                f"backbone_rank must be in [0, {self.world_size}), got {self.backbone_rank}"
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


_PLAN: DSV4ExecutionPlan | None = None


def configure_dsv4_execution(backbone_rank: int | None) -> DSV4ExecutionPlan:
    """Configure this engine process once TP rank information is available."""
    global _PLAN
    tp = get_tp_info()
    plan = DSV4ExecutionPlan(tp.rank, tp.size, backbone_rank)
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
    "configure_dsv4_execution",
    "get_dsv4_execution_plan",
]
