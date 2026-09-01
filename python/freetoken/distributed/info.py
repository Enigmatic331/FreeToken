from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


@contextmanager
def override_tp_info(rank: int, size: int):
    """Temporarily expose a TP view used only while constructing/loading a model."""
    global _TP_INFO
    previous = _TP_INFO
    if previous is None:
        raise RuntimeError("TP info has not been set")
    _TP_INFO = DistributedInfo(rank, size)
    try:
        yield _TP_INFO
    finally:
        _TP_INFO = previous


__all__ = [
    "DistributedInfo",
    "set_tp_info",
    "get_tp_info",
    "override_tp_info",
    "try_get_tp_info",
]
