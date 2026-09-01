"""Execution roles for a TP1 Qwen backbone with rank-local routed experts."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

from freetoken.attention.base import BaseAttnBackend
from freetoken.distributed import get_tp_info, override_tp_info
from freetoken.moe.partition import ExpertPartition


@dataclass(frozen=True, slots=True)
class Qwen4ExpExecutionPlan:
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

    def partition(self, total_experts: int) -> ExpertPartition:
        return ExpertPartition(
            total_experts,
            world_size=self.world_size if self.enabled else 1,
            rank=self.rank if self.enabled else 0,
            shard_counts=self.expert_shards if self.enabled else None,
        )

    def model_tp_context(self):
        # All TP-aware Qwen layers must be constructed/loaded whole on the
        # authority. Expert workers construct no dense layers, but use the same
        # view so their rank-local MoE shells remain free of accidental TP.
        return override_tp_info(0, 1) if self.enabled else nullcontext()


class Qwen4ExpExpertWorkerAttentionBackend(BaseAttnBackend):
    """Scheduler-compatible shell for a rank that executes no attention."""

    def forward(self, q, k, v, layer_id, batch, attn_spec=None):
        raise RuntimeError("Qwen expert-worker ranks do not execute attention")

    def prepare_metadata(self, batch) -> None:
        batch.attn_metadata = None

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        pass

    def prepare_for_capture(self, batch) -> None:
        pass

    def prepare_for_replay(self, batch) -> None:
        pass


_PLAN: Qwen4ExpExecutionPlan | None = None


def configure_qwen4_exp_execution(
    backbone_rank: int | None,
    expert_shards: tuple[int, ...] | None = None,
) -> Qwen4ExpExecutionPlan:
    global _PLAN
    info = get_tp_info()
    plan = Qwen4ExpExecutionPlan(info.rank, info.size, backbone_rank, expert_shards)
    if _PLAN is not None and _PLAN != plan:
        raise RuntimeError(f"Qwen execution plan already configured as {_PLAN}, got {plan}")
    _PLAN = plan
    return plan


def get_qwen4_exp_execution_plan() -> Qwen4ExpExecutionPlan:
    if _PLAN is not None:
        return _PLAN
    info = get_tp_info()
    return Qwen4ExpExecutionPlan(info.rank, info.size)


def _reset_qwen4_exp_execution_for_tests() -> None:
    global _PLAN
    _PLAN = None


__all__ = [
    "Qwen4ExpExecutionPlan",
    "Qwen4ExpExpertWorkerAttentionBackend",
    "configure_qwen4_exp_execution",
    "get_qwen4_exp_execution_plan",
]
