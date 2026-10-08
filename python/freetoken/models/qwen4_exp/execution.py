"""Execution roles for a TP1 Qwen backbone with rank-local routed experts."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os

from freetoken.attention.base import BaseAttnBackend
from freetoken.distributed import get_tp_info, override_tp_info
from freetoken.moe.partition import ExpertPartition, ExpertStorageRange


def _parse_shards_env(name: str, world_size: int) -> tuple[int, ...] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integer counts, got {raw!r}") from exc
    if len(values) != world_size:
        raise ValueError(f"{name} needs {world_size} counts, got {len(values)}")
    if any(value < 0 for value in values):
        raise ValueError(f"{name} counts must be non-negative")
    return values


def _parse_storage_env(
    name: str, world_size: int
) -> tuple[tuple[int, int], ...] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    result = []
    try:
        for item in raw.split(","):
            offset, count = item.strip().split(":", 1)
            result.append((int(offset), int(count)))
    except ValueError as exc:
        raise ValueError(
            f"{name} must contain comma-separated OFFSET:COUNT ranges, got {raw!r}"
        ) from exc
    if len(result) != world_size:
        raise ValueError(f"{name} needs {world_size} ranges, got {len(result)}")
    if any(offset < 0 or count < 0 for offset, count in result):
        raise ValueError(f"{name} offsets/counts must be non-negative")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Qwen4ExpExecutionPlan:
    rank: int
    world_size: int
    backbone_rank: int | None = None
    expert_shards: tuple[int, ...] | None = None
    prefill_expert_shards: tuple[int, ...] | None = None
    expert_storage_ranges: tuple[tuple[int, int], ...] | None = None

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
        if self.prefill_expert_shards is not None:
            shards = tuple(self.prefill_expert_shards)
            object.__setattr__(self, "prefill_expert_shards", shards)
            if len(shards) != self.world_size:
                raise ValueError(
                    "prefill_expert_shards must contain exactly one count per rank: "
                    f"expected {self.world_size}, got {len(shards)}"
                )
            if any(not isinstance(count, int) or count < 0 for count in shards):
                raise ValueError("prefill_expert_shards must contain non-negative integers")
        if self.expert_storage_ranges is not None:
            ranges = tuple(tuple(item) for item in self.expert_storage_ranges)
            object.__setattr__(self, "expert_storage_ranges", ranges)
            if len(ranges) != self.world_size:
                raise ValueError(
                    "expert_storage_ranges must contain exactly one range per rank: "
                    f"expected {self.world_size}, got {len(ranges)}"
                )
            if any(
                len(item) != 2
                or any(not isinstance(value, int) or value < 0 for value in item)
                for item in ranges
            ):
                raise ValueError(
                    "expert_storage_ranges must contain non-negative (offset, count) pairs"
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

    def partition(self, total_experts: int, *, prefill: bool = False) -> ExpertPartition:
        shards = self.prefill_expert_shards if prefill else self.expert_shards
        return ExpertPartition(
            total_experts,
            world_size=self.world_size if self.enabled else 1,
            rank=self.rank if self.enabled else 0,
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
            ("decode", self.partition(total_experts, prefill=False)),
        ):
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

    def prefill_active_ranks(self, total_experts: int) -> tuple[int, ...]:
        if not self.enabled:
            return (0,)
        shards = self.prefill_expert_shards or self.expert_shards
        if shards is None:
            base, remainder = divmod(total_experts, self.world_size)
            shards = tuple(base + (rank < remainder) for rank in range(self.world_size))
        return tuple(rank for rank, count in enumerate(shards) if count > 0)

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
    prefill_shards = _parse_shards_env(
        "FREETOKEN_QWEN4_EXP_PREFILL_SHARDS", info.size
    )
    storage_ranges = _parse_storage_env(
        "FREETOKEN_QWEN4_EXP_STORAGE_RANGES", info.size
    )
    if (prefill_shards is None) != (storage_ranges is None):
        raise ValueError(
            "FREETOKEN_QWEN4_EXP_PREFILL_SHARDS and "
            "FREETOKEN_QWEN4_EXP_STORAGE_RANGES must be set together"
        )
    plan = Qwen4ExpExecutionPlan(
        info.rank,
        info.size,
        backbone_rank,
        expert_shards,
        prefill_shards,
        storage_ranges,
    )
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
