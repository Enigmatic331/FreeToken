"""Opt-in NVTX ranges for DeepSeek-V4.1 performance investigations.

The ranges are deliberately disabled by default: their purpose is to make an
Nsight Systems capture explain *where* time and kernel launches go without
adding CUDA synchronizations (and therefore changing the workload being
measured). Set ``FREETOKEN_DSV41_PROFILE=1`` before starting the server.
"""

from __future__ import annotations

import functools
import os
from contextlib import nullcontext

import torch


_ENABLED = os.getenv("FREETOKEN_DSV41_PROFILE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _nvtx_range(name: str):
    return torch.cuda.nvtx.range(name)


def profile_range(name: str):
    """Return an NVTX range when profiling is enabled, otherwise a no-op."""

    return _nvtx_range(name) if _ENABLED else nullcontext()


def profile(name: str, *, layer_id_field: str | None = None):
    """Decorate a V4.1 stage with a stable, optionally layer-qualified range."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not _ENABLED:
                return fn(self, *args, **kwargs)
            display_name = name
            if layer_id_field is not None:
                display_name = name.format(getattr(self, layer_id_field))
            with _nvtx_range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator


__all__ = ["profile", "profile_range"]
