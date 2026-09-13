"""V4.1 authority-EP and opt-in TP2+EP2 execution roles."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

from freetoken.attention.base import BaseAttnBackend
from freetoken.distributed import get_tp_info, override_tp_info
from freetoken.moe.partition import ExpertPartition


@dataclass(frozen=True, slots=True)
class DeepseekV41ExecutionPlan:
    rank: int
    world_size: int
    backbone_rank: int | None = None
    expert_shards: tuple[int, ...] | None = None
    tp2_ep2: bool = False

    def __post_init__(self) -> None:
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"invalid execution rank {self.rank}/{self.world_size}")
        if self.backbone_rank is not None and not 0 <= self.backbone_rank < self.world_size:
            raise ValueError(f"invalid backbone rank {self.backbone_rank}")
        if self.tp2_ep2 and (self.backbone_rank is None or self.world_size != 2):
            raise ValueError("V4.1 TP2+EP2 requires a backbone root and world size 2")
        if self.expert_shards is not None:
            shards = tuple(self.expert_shards)
            object.__setattr__(self, "expert_shards", shards)
            if len(shards) != self.world_size:
                raise ValueError("expert_shards must have one entry per rank")

    @property
    def enabled(self) -> bool:
        return self.backbone_rank is not None and self.world_size > 1

    @property
    def is_backbone(self) -> bool:
        return not self.enabled or self.tp2_ep2 or self.rank == self.backbone_rank

    @property
    def is_expert_worker(self) -> bool:
        return self.enabled and not self.tp2_ep2 and self.rank != self.backbone_rank

    @property
    def uses_authority_transport(self) -> bool:
        return self.enabled and not self.tp2_ep2

    @property
    def participates_in_engram(self) -> bool:
        # Every EP rank owns a row interval even though only one executes dense layers.
        return True

    def partition(self, total_experts: int) -> ExpertPartition:
        return ExpertPartition(
            total_experts,
            world_size=self.world_size if self.enabled else 1,
            rank=self.rank if self.enabled else 0,
            shard_counts=self.expert_shards if self.enabled else None,
        )

    def model_tp_context(self):
        # Dense weights are whole on the authority; workers build only local expert shells.
        return override_tp_info(0, 1) if self.uses_authority_transport else nullcontext()

    def expert_tp_context(self):
        # EP ranks own complete experts. Their intermediate dimensions are not
        # tensor-sharded even when the dense backbone is TP2.
        return override_tp_info(0, 1) if self.enabled else nullcontext()


class DeepseekV41ExpertWorkerAttentionBackend(BaseAttnBackend):
    """Scheduler shell for a V4.1 rank that executes no dense attention."""

    def forward(self, q, k, v, layer_id, batch, attn_spec=None):
        raise RuntimeError("V4.1 expert-worker ranks do not execute attention")

    def prepare_metadata(self, batch) -> None:
        batch.attn_metadata = None

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        pass

    def prepare_for_capture(self, batch) -> None:
        pass

    def prepare_for_replay(self, batch) -> None:
        pass


_PLAN: DeepseekV41ExecutionPlan | None = None


def configure_execution(
    backbone_rank: int | None,
    expert_shards: tuple[int, ...] | None = None,
    tp2_ep2: bool = False,
) -> DeepseekV41ExecutionPlan:
    global _PLAN
    info = get_tp_info()
    plan = DeepseekV41ExecutionPlan(
        info.rank, info.size, backbone_rank, expert_shards, tp2_ep2
    )
    if _PLAN is not None and _PLAN != plan:
        raise RuntimeError(f"V4.1 execution already configured as {_PLAN}, got {plan}")
    _PLAN = plan
    return plan


def get_execution_plan() -> DeepseekV41ExecutionPlan:
    if _PLAN is not None:
        return _PLAN
    try:
        info = get_tp_info()
    except RuntimeError:
        # Meta-device checkpoint validation and CPU tooling run before the
        # engine establishes distributed state; their faithful default is TP1.
        return DeepseekV41ExecutionPlan(0, 1)
    return DeepseekV41ExecutionPlan(info.rank, info.size)


def _reset_execution_for_tests() -> None:
    global _PLAN
    _PLAN = None


__all__ = [
    "DeepseekV41ExecutionPlan",
    "DeepseekV41ExpertWorkerAttentionBackend",
    "configure_execution",
    "get_execution_plan",
]
