"""V4.1 authority-EP and opt-in dense-parallel execution roles."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

from freetoken.attention.base import BaseAttnBackend
from freetoken.distributed import get_tp_info, override_tp_info
from freetoken.moe.partition import ExpertPartition, ExpertStorageRange


@dataclass(frozen=True, slots=True)
class DeepseekV41ExecutionPlan:
    rank: int
    world_size: int
    backbone_rank: int | None = None
    expert_shards: tuple[int, ...] | None = None
    prefill_expert_shards: tuple[int, ...] | None = None
    expert_storage_ranges: tuple[tuple[int, int], ...] | None = None
    engram_ranks: tuple[int, ...] | None = None
    tp2_ep2: bool = False
    attention_tp2_ep2: bool = False

    def __post_init__(self) -> None:
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"invalid execution rank {self.rank}/{self.world_size}")
        if self.backbone_rank is not None and not 0 <= self.backbone_rank < self.world_size:
            raise ValueError(f"invalid backbone rank {self.backbone_rank}")
        if self.tp2_ep2 and (self.backbone_rank is None or self.world_size != 2):
            raise ValueError("V4.1 TP2+EP2 requires a backbone root and world size 2")
        if self.attention_tp2_ep2 and (
            self.backbone_rank is None or self.world_size != 2
        ):
            raise ValueError(
                "V4.1 attention-TP2+EP2 requires a backbone root and world size 2"
            )
        if self.tp2_ep2 and self.attention_tp2_ep2:
            raise ValueError(
                "V4.1 full TP2+EP2 and attention-TP2+EP2 are mutually exclusive"
            )
        if self.expert_shards is not None:
            shards = tuple(self.expert_shards)
            object.__setattr__(self, "expert_shards", shards)
            if len(shards) != self.world_size:
                raise ValueError("expert_shards must have one entry per rank")
        if self.prefill_expert_shards is not None:
            shards = tuple(self.prefill_expert_shards)
            object.__setattr__(self, "prefill_expert_shards", shards)
            if len(shards) != self.world_size:
                raise ValueError("prefill_expert_shards must have one entry per rank")
            if any(count < 0 for count in shards):
                raise ValueError("prefill_expert_shards must be non-negative")
        if self.expert_storage_ranges is not None:
            ranges = tuple(tuple(item) for item in self.expert_storage_ranges)
            object.__setattr__(self, "expert_storage_ranges", ranges)
            if len(ranges) != self.world_size:
                raise ValueError("expert_storage_ranges must have one range per rank")
            if any(
                len(item) != 2 or any(value < 0 for value in item)
                for item in ranges
            ):
                raise ValueError(
                    "expert_storage_ranges must contain non-negative offset/count pairs"
                )
        if (self.prefill_expert_shards is None) != (
            self.expert_storage_ranges is None
        ):
            raise ValueError(
                "prefill_expert_shards and expert_storage_ranges must be set together"
            )
        if self.phase_aware and self.backbone_rank is None:
            raise ValueError("phase-aware V4.1 EP requires a backbone rank")
        if self.phase_aware and self.dense_parallel:
            raise ValueError("phase-aware V4.1 EP requires an authority backbone")
        if self.phase_aware and self.prefill_expert_shards[self.backbone_rank] == 0:
            raise ValueError("the backbone rank must participate in prefill")
        if self.phase_aware and len(self.prefill_active_ranks) < 2:
            raise ValueError(
                "phase-aware V4.1 requires at least two active prefill ranks"
            )
        if self.engram_ranks is not None:
            ranks = tuple(self.engram_ranks)
            object.__setattr__(self, "engram_ranks", ranks)
            if self.backbone_rank is None:
                raise ValueError("engram_ranks requires a backbone rank")
            if not ranks:
                raise ValueError("engram_ranks must not be empty")
            if len(set(ranks)) != len(ranks):
                raise ValueError("engram_ranks must not contain duplicates")
            if any(rank < 0 or rank >= self.world_size for rank in ranks):
                raise ValueError("engram_ranks contains a rank outside the EP world")
            if self.backbone_rank not in ranks:
                raise ValueError("engram_ranks must include the backbone rank")
            if self.dense_parallel and set(ranks) != set(range(self.world_size)):
                raise ValueError("dense-parallel V4.1 requires Engram on every rank")
            if self.phase_aware and not set(ranks).issubset(
                self.prefill_active_ranks
            ):
                raise ValueError(
                    "Engram ranks must participate in phase-aware prefill"
                )

    @property
    def enabled(self) -> bool:
        return self.backbone_rank is not None and self.world_size > 1

    @property
    def is_backbone(self) -> bool:
        return not self.enabled or self.dense_parallel or self.rank == self.backbone_rank

    @property
    def is_expert_worker(self) -> bool:
        return self.enabled and not self.dense_parallel and self.rank != self.backbone_rank

    @property
    def uses_authority_transport(self) -> bool:
        return self.enabled and not self.dense_parallel

    @property
    def dense_parallel(self) -> bool:
        return self.tp2_ep2 or self.attention_tp2_ep2

    @property
    def attention_parallel(self) -> bool:
        return self.dense_parallel

    @property
    def shared_expert_parallel(self) -> bool:
        return self.tp2_ep2

    @property
    def participates_in_engram(self) -> bool:
        return self.rank in self.resolved_engram_ranks

    @property
    def phase_aware(self) -> bool:
        return self.prefill_expert_shards is not None

    @property
    def participates_in_prefill(self) -> bool:
        return (
            not self.phase_aware
            or self.prefill_expert_shards[self.rank] > 0
        )

    @property
    def prefill_active_ranks(self) -> tuple[int, ...]:
        if not self.phase_aware:
            return tuple(range(self.world_size))
        return tuple(
            rank
            for rank, count in enumerate(self.prefill_expert_shards)
            if count > 0
        )

    @property
    def uses_prefill_subgroup(self) -> bool:
        """Whether prefill must exclude one or more decode-only ranks."""

        return self.phase_aware and len(self.prefill_active_ranks) < self.world_size

    @property
    def resolved_engram_ranks(self) -> tuple[int, ...]:
        return self.engram_ranks or tuple(range(self.world_size))

    @property
    def engram_rank(self) -> int:
        if not self.participates_in_engram:
            raise RuntimeError(f"rank {self.rank} does not participate in Engram")
        return self.resolved_engram_ranks.index(self.rank)

    @property
    def engram_world_size(self) -> int:
        return len(self.resolved_engram_ranks)

    @property
    def supports_packed_prefill(self) -> bool:
        return self.enabled and len(self.prefill_active_ranks) >= 2

    def partition(
        self, total_experts: int, *, prefill: bool = False
    ) -> ExpertPartition:
        return self.partition_for_rank(total_experts, self.rank, prefill=prefill)

    def partition_for_rank(
        self, total_experts: int, rank: int, *, prefill: bool = False
    ) -> ExpertPartition:
        shards = (
            self.prefill_expert_shards
            if prefill and self.phase_aware
            else self.expert_shards
        )
        return ExpertPartition(
            total_experts,
            world_size=self.world_size if self.enabled else 1,
            rank=rank if self.enabled else 0,
            shard_counts=shards if self.enabled else None,
        )

    def storage_partition(self, total_experts: int) -> ExpertStorageRange:
        if not self.enabled or self.expert_storage_ranges is None:
            owned = self.partition(total_experts)
            return ExpertStorageRange(
                total_experts, owned.global_offset, owned.local_count
            )
        offset, count = self.expert_storage_ranges[self.rank]
        storage = ExpertStorageRange(total_experts, offset, count)
        for phase, owned in (
            ("prefill", self.partition(total_experts, prefill=True)),
            ("decode", self.partition(total_experts)),
        ):
            # A phase-inactive rank owns no experts and therefore imposes no
            # storage requirement.  Its partition offset is merely the prefix
            # sum of earlier shards and need not fall inside this rank's
            # unrelated decode-storage interval.
            if owned.local_count == 0:
                continue
            if not (
                storage.global_offset <= owned.global_offset
                and owned.global_stop <= storage.global_stop
            ):
                raise ValueError(
                    f"rank {self.rank} {phase} ownership "
                    f"[{owned.global_offset}, {owned.global_stop}) is not contained in "
                    f"storage [{storage.global_offset}, {storage.global_stop})"
                )
        return storage

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
    prefill_expert_shards: tuple[int, ...] | None = None,
    expert_storage_ranges: tuple[tuple[int, int], ...] | None = None,
    engram_ranks: tuple[int, ...] | None = None,
    tp2_ep2: bool = False,
    attention_tp2_ep2: bool = False,
) -> DeepseekV41ExecutionPlan:
    global _PLAN
    info = get_tp_info()
    plan = DeepseekV41ExecutionPlan(
        rank=info.rank,
        world_size=info.size,
        backbone_rank=backbone_rank,
        expert_shards=expert_shards,
        prefill_expert_shards=prefill_expert_shards,
        expert_storage_ranges=expert_storage_ranges,
        engram_ranks=engram_ranks,
        tp2_ep2=tp2_ep2,
        attention_tp2_ep2=attention_tp2_ep2,
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
