"""Opt-in, measurement-only digest of the MoE slot-cache admission state.

``FREETOKEN_MOE_CACHE_FINGERPRINT=1`` makes every rank log one line per scheduler idle epoch:

    MoE cache fingerprint rank=R step=S resident=N sha256=H

``H`` hashes the persistent admission state: ``slot_for_id``, ``id_of_slot``, ``usage``,
``step`` and ``expert_recency`` (dtype and shape included). Two instances that served the
same sequential requests from a cold start print equal fingerprints exactly when they made
the same admission and eviction decisions, so an A/B arm (for example with
``--moe-collect-stats`` off) can prove unchanged cache behaviour without the counters. No
request content, expert payload or routing history is read or logged.

Unset or ``0`` disables it and adds no work anywhere; any other value is refused at startup.
"""
from __future__ import annotations

import hashlib
import os

import torch

FINGERPRINT_ENV = "FREETOKEN_MOE_CACHE_FINGERPRINT"
FINGERPRINT_FIELDS = ("slot_for_id", "id_of_slot", "usage", "step", "expert_recency")


def fingerprint_enabled() -> bool:
    """Strict parse: unset/``0`` is off and ``1`` is on."""
    value = os.environ.get(FINGERPRINT_ENV)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{FINGERPRINT_ENV} must be '0' or '1', got {value!r}")


def cache_fingerprint(cache) -> dict:
    """Digest of ``cache``'s admission state after every queued writer has finished."""
    cache.finish_background_admission()
    if cache.id_of_slot.is_cuda:
        # Background-admission and prefill copy streams also write the slot map.
        torch.cuda.synchronize(cache.id_of_slot.device)
    digest = hashlib.sha256()
    host = {}
    for name in FINGERPRINT_FIELDS:
        tensor = getattr(cache, name).detach().to("cpu").contiguous()
        host[name] = tensor
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.numpy().tobytes())
    return dict(
        step=int(host["step"].item()),
        resident=int((host["id_of_slot"] >= 0).sum().item()),
        sha256=digest.hexdigest(),
    )


def format_fingerprint(rank: int, fingerprint: dict) -> str:
    return (
        f"MoE cache fingerprint rank={rank} step={fingerprint['step']} "
        f"resident={fingerprint['resident']} sha256={fingerprint['sha256']}"
    )


__all__ = [
    "FINGERPRINT_ENV",
    "FINGERPRINT_FIELDS",
    "cache_fingerprint",
    "fingerprint_enabled",
    "format_fingerprint",
]
