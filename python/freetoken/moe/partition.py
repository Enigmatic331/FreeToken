"""Contiguous expert ownership shared by loaders, routers, and caches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ExpertPartition:
    total_experts: int
    world_size: int = 1
    rank: int = 0
    shard_counts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.total_experts < 0:
            raise ValueError("total_experts must be non-negative")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")
        if self.shard_counts is not None:
            counts = tuple(self.shard_counts)
            object.__setattr__(self, "shard_counts", counts)
            if len(counts) != self.world_size:
                raise ValueError(
                    "shard_counts must contain exactly one count per rank: "
                    f"expected {self.world_size}, got {len(counts)}"
                )
            if any(not isinstance(count, int) or isinstance(count, bool) for count in counts):
                raise ValueError("shard_counts must contain integers")
            if any(count < 0 for count in counts):
                raise ValueError("shard_counts must be non-negative")
            if sum(counts) != self.total_experts:
                raise ValueError(
                    f"shard_counts must sum to total_experts={self.total_experts}, "
                    f"got {sum(counts)}"
                )

    @property
    def local_count(self) -> int:
        if self.shard_counts is not None:
            return self.shard_counts[self.rank]
        base, remainder = divmod(self.total_experts, self.world_size)
        return base + (self.rank < remainder)

    @property
    def global_offset(self) -> int:
        if self.shard_counts is not None:
            return sum(self.shard_counts[: self.rank])
        base, remainder = divmod(self.total_experts, self.world_size)
        return self.rank * base + min(self.rank, remainder)

    @property
    def global_stop(self) -> int:
        return self.global_offset + self.local_count

    @property
    def global_range(self) -> range:
        return range(self.global_offset, self.global_stop)

    def owns(self, global_expert: int) -> bool:
        return self.global_offset <= global_expert < self.global_stop

    def global_to_local(self, global_expert: int) -> int:
        if not self.owns(global_expert):
            raise ValueError(
                f"global expert {global_expert} is not owned by rank {self.rank}; "
                f"owned range is [{self.global_offset}, {self.global_stop})"
            )
        return global_expert - self.global_offset


def localize_expert_routes(
    weights: torch.Tensor,
    indices: torch.Tensor,
    partition: ExpertPartition,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = indices - partition.global_offset
    owned = (local >= 0) & (local < partition.local_count)
    local = torch.where(owned, local, local.new_full((), partition.local_count))
    weights = torch.where(owned, weights, weights.new_zeros(()))
    return weights, local


def cache_safe_route_ids(
    weights: torch.Tensor,
    local_ids: torch.Tensor,
) -> torch.Tensor:
    active = weights != 0
    first_pos = active.to(torch.int64).argmax(dim=-1, keepdim=True)
    fallback = local_ids.gather(-1, first_pos)
    fallback = torch.where(
        active.any(dim=-1, keepdim=True), fallback, fallback.new_zeros(())
    )
    return torch.where(active, local_ids, fallback)


__all__ = ["ExpertPartition", "cache_safe_route_ids", "localize_expert_routes"]
