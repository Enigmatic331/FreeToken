"""Expert ownership for expert-parallel model execution.

The partition is contiguous and balanced: lower ranks receive one extra expert
when ``total_experts`` is not evenly divisible by ``world_size``.  Keeping this
logic in one value object prevents loaders, caches, and routers from developing
slightly different ideas of expert ownership.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpertPartition:
    """A rank's contiguous share of a global routed-expert namespace."""

    total_experts: int
    world_size: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        if self.total_experts < 0:
            raise ValueError("total_experts must be non-negative")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {self.rank}"
            )

    @property
    def local_count(self) -> int:
        base, remainder = divmod(self.total_experts, self.world_size)
        return base + (self.rank < remainder)

    @property
    def global_offset(self) -> int:
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

    def local_to_global(self, local_expert: int) -> int:
        if not 0 <= local_expert < self.local_count:
            raise ValueError(
                f"local expert {local_expert} is outside [0, {self.local_count})"
            )
        return self.global_offset + local_expert

    def owner(self, global_expert: int) -> int:
        if not 0 <= global_expert < self.total_experts:
            raise ValueError(
                f"global expert {global_expert} is outside [0, {self.total_experts})"
            )
        base, remainder = divmod(self.total_experts, self.world_size)
        wide = base + 1
        wide_experts = wide * remainder
        if global_expert < wide_experts:
            return global_expert // wide
        # Reaching this branch implies base > 0: when base == 0 every valid
        # expert is in one of the leading one-expert partitions above.
        return remainder + (global_expert - wide_experts) // base


__all__ = ["ExpertPartition"]
