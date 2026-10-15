"""Real-weight authority-only qualification rig for Package A.

This module is deliberately test-only.  It creates the real authority backbone,
loads the checkpoint's dense and Engram weights, and allocates the real V4.1
attention/KV state.  Only the two things which cannot exist in a one-process
authority rig are substituted:

* inter-rank transport is an identity transport with CUDA-event markers; and
* routed-expert arithmetic is a deterministic local stub which still enters
  :meth:`RoutedExperts._maybe_all_reduce` in the production order.

The rig never replaces the router, shared expert, HC, attention, window,
compressor, indexer, Engram injection, projection, norm or head paths.  It is
therefore suitable for G2/G3 exactness and for G4 PRE-boundary timing, but is
not a serving implementation.
"""

from __future__ import annotations

import contextlib
import builtins
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import torch


AUTHORITY_PLAN = {
    "rank": 0,
    "world_size": 3,
    "backbone_rank": 0,
    "expert_shards": (32, 160, 192),
    "prefill_expert_shards": (128, 160, 96),
    "expert_storage_ranges": ((0, 128), (32, 256), (192, 192)),
    "engram_ranks": (0, 1),
}

APPROVED_EXTENSION_HASHES = {
    "_cpu_moe.cpython-312-x86_64-linux-gnu.so": (
        "770dcb4692f07b5b1b334b9baa2f67d02a4ae58f20a9be9893af2f6dccc00e6c"
    ),
    "_pinned_tensor.cpython-312-x86_64-linux-gnu.so": (
        "419716a3f615ac955e9d28904b795103a4d3f397bede3e25fe79630eec07e653"
    ),
}

LIVE_ENVIRONMENT = {
    "FREETOKEN_DSV41_AUTHORITY_FUSION": "1",
    "FREETOKEN_DSV41_FUSED_DECODE_DISPATCH": "1",
    "CUDA_VISIBLE_DEVICES": "GPU-d151df0f-4c65-0eb2-48f5-c10b135f63ff",
    "PYTHONDONTWRITEBYTECODE": "1",
}
AUTHORITY_UUID = LIVE_ENVIRONMENT["CUDA_VISIBLE_DEVICES"]
_QUALIFICATION_RIG: "RealWeightAuthorityRig | None" = None
_QUALIFICATION_INJECTION: dict[str, Any] | None = None
_QUALIFICATION_BUILTIN_KEY = "_freetoken_authority_qualification_injection_v1"
CUDA_NODE_COUNT = 10
CUDA_NODE_INVENTORY_SHA256 = (
    "601bb852124f7ad39e59f1e5ce236f813ae86bf72315a6b7066148cd274c3c50"
)

REPRESENTATIVE_LAYERS = (0, 1, 2, 5, 13, 14, 20, 24, 39)
LAYER_CLASSES = {
    0: "boundary",
    1: "engram-window",
    2: "kv-index-ratio2",
    5: "normal-consumer",
    13: "pre-engram-suppressed-producer",
    14: "engram-kv-index",
    20: "kv-candidate-source",
    24: "index-source",
    39: "tail-suppressed-producer",
}
LAYER_WEIGHTS = {0: 1, 1: 1, 2: 2, 5: 28, 13: 1, 14: 1, 20: 1, 24: 4, 39: 1}


def validate_live_environment() -> dict[str, str]:
    """Apply the same fail-closed live environment gate to CLI and pytest."""

    if os.getenv("FREETOKEN_AUTHORITY_RIG_LIVE") != "1":
        raise RuntimeError("FREETOKEN_AUTHORITY_RIG_LIVE=1 receipt is required")
    for name, expected in LIVE_ENVIRONMENT.items():
        if os.getenv(name) != expected:
            raise RuntimeError(f"{name} must be exactly {expected!r}")
    if os.getenv("FREETOKEN_DSV41_FUSED_ROUTER", "0") not in ("", "0"):
        raise RuntimeError("legacy fused router must remain off")
    if os.getenv("FREETOKEN_DSV41_HC_DIRECT_PRE", "0") == "1":
        raise RuntimeError("legacy all-phase HC direct-pre must remain off")
    if "TRITON_DEFAULT_FP_FUSION" in os.environ:
        raise RuntimeError("TRITON_DEFAULT_FP_FUSION must be absent")
    return {name: os.environ[name] for name in sorted(LIVE_ENVIRONMENT)}


def validate_live_candidate_runtime(
    *,
    candidate_root: Path,
    expected_commit: str,
    expected_tree: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Share clean-tree, private-cache and UUID gates across CLI and pytest."""

    root = candidate_root.resolve()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if (commit, tree) != (expected_commit, expected_tree):
        raise RuntimeError(f"candidate pin drift: {commit}/{tree}")
    if status:
        raise RuntimeError(f"candidate worktree is not clean:\n{status}")
    # nvidia-smi enumerates physical devices and does not honor
    # CUDA_VISIBLE_DEVICES.  Prove the process-visible CUDA namespace instead;
    # qualification must expose exactly the authority GPU as logical device 0.
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"authority GPU isolation drift: torch sees {torch.cuda.device_count()} devices"
        )
    raw_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if isinstance(raw_uuid, bytes):
        raw_uuid = raw_uuid.decode()
    if raw_uuid:
        normalized_uuid = str(raw_uuid)
        if not normalized_uuid.startswith("GPU-"):
            normalized_uuid = f"GPU-{normalized_uuid}"
        uuids = [normalized_uuid]
    else:
        uuids = []
    if uuids != [AUTHORITY_UUID]:
        raise RuntimeError(f"authority GPU isolation drift: {uuids}")
    resolved_run = run_dir.resolve()
    caches = {}
    for name in ("TRITON_CACHE_DIR", "FREETOKEN_KERNEL_CACHE_DIR"):
        value = os.getenv(name)
        if not value or not Path(value).resolve().is_relative_to(resolved_run):
            raise RuntimeError(f"{name} must be a run-local private cache")
        caches[name] = str(Path(value).resolve())
    return {
        "candidate": {"root": str(root), "commit": commit, "tree": tree, "clean": True},
        "authority_gpu_uuid": uuids[0],
        "private_caches": caches,
    }


def validate_cudagraph_debug_dump_capability(path: Path) -> dict[str, Any]:
    """Fail early unless this PyTorch/CUDA build can emit a retained graph DOT."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    source = torch.arange(16, dtype=torch.float32, device="cuda")
    target = torch.empty_like(source)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        target.copy_(source)
        target.add_(1)
    graph.instantiate()
    graph.debug_dump(str(path))
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"retained CUDA graph debug dump absent/empty: {path}")
    raw = path.read_bytes()
    if b"digraph" not in raw:
        raise RuntimeError(f"retained CUDA graph debug dump is not DOT: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def cudagraph_marker_sidecar(
    *, captured: "CapturedAuthorityDecode", rows: int, arm: str, dot_path: Path
) -> dict[str, Any]:
    """Build the fail-closed raw-DOT/event/destination correlation receipt."""

    if not dot_path.is_file() or not dot_path.stat().st_size:
        raise RuntimeError(f"CUDA graph debug dump absent/empty: {dot_path}")
    destinations = tuple(captured.wire_destinations)
    if [item["layer"] for item in destinations] != list(range(40)):
        raise RuntimeError("CUDA graph wire-hidden destination receipt is incomplete")
    hidden_destinations = tuple(
        {"layer": item["layer"], "base_ptr": item["base_ptr"], "nbytes": item["nbytes"]}
        for item in destinations
    )
    tail_destinations = tuple(
        {
            "layer": item["layer"],
            "base_ptr": item["tail_base_ptr"],
            "nbytes": item["tail_nbytes"],
        }
        for item in destinations
    )
    return {
        "schema": "authority-cudagraph-marker-events-v2",
        "rows": rows,
        "arm": arm,
        "dot_file": dot_path.name,
        "dot_sha256": hashlib.sha256(dot_path.read_bytes()).hexdigest(),
        "dot_receipt": {
            "producer": "torch.cuda.CUDAGraph.debug_dump",
            "grammar": "pytorch-cudagraph-debug-dot",
            "grammar_version": str(torch.__version__),
        },
        "markers": captured.marker_events,
        "copy_scope": {
            "begin": "copy_scope_begin(layer)",
            "end": "broadcast_start(layer)",
            "required_memcpy_kind": "Device-to-Device",
        },
        "copy_destinations": {
            "wire_hidden": hidden_destinations,
            "wire_tail": tail_destinations,
        },
        "validation_status": "RAW_UNEVALUATED",
        "validator_requirement": (
            "successor evaluator must parse real event-record nodes, prove graph "
            "reachability, and match D2D destination ranges fail-closed"
        ),
    }


@contextlib.contextmanager
def installed_qualification_rig(
    rig: "RealWeightAuthorityRig",
) -> Iterator[dict[str, Any]]:
    """Expose one already-loaded rig to in-process pytest qualification only."""

    global _QUALIFICATION_INJECTION, _QUALIFICATION_RIG
    if _QUALIFICATION_RIG is not None or hasattr(builtins, _QUALIFICATION_BUILTIN_KEY):
        raise RuntimeError("qualification rig already installed")
    receipt = {"rig": rig, "rig_id": id(rig), "consumed": False}
    _QUALIFICATION_RIG = rig
    _QUALIFICATION_INJECTION = receipt
    setattr(builtins, _QUALIFICATION_BUILTIN_KEY, receipt)
    try:
        yield receipt
    finally:
        if getattr(builtins, _QUALIFICATION_BUILTIN_KEY, None) is receipt:
            delattr(builtins, _QUALIFICATION_BUILTIN_KEY)
        _QUALIFICATION_RIG = None
        _QUALIFICATION_INJECTION = None


def current_qualification_rig() -> "RealWeightAuthorityRig | None":
    injection = getattr(builtins, _QUALIFICATION_BUILTIN_KEY, None)
    if injection is None:
        return None
    injection["consumed"] = True
    return injection["rig"]


@dataclass(frozen=True)
class FrozenProtocol:
    """G2/G3/G4 inputs fixed before any observation."""

    page_size: int = 128
    # Two independent six-page request ranges cover every preregistered
    # position through 767 without aliasing any full/window/cmp/index slot.
    full_pages: int = 12
    window_pages: int = 12
    table_rows: int = 2
    logical_width: int = 768
    scratch_rows: int = 3
    m1_positions: tuple[int, ...] = (255,)
    m2_positions: tuple[int, ...] = (255, 383)
    m1_table_indices: tuple[int, ...] = (0,)
    m2_table_indices: tuple[int, ...] = (0, 1)
    m1_stage_cap: int = 255
    m2_stage_cap: int = 383
    m2_arm_order: tuple[str, ...] = (
        "control",
        "full",
        "minus_R1",
        "minus_R2",
        "minus_R3",
        "minus_R4",
        "minus_R8",
        "minus_R10",
        "minus_R11",
        "minus_R12-prime",
        "minus_R13",
        "minus_P1",
    )
    representatives: tuple[int, ...] = REPRESENTATIVE_LAYERS
    rounds: int = 40
    replays_per_round: int = 20
    aa_null_pairs: int = 12
    inconclusive_reruns_max: int = 1

    def validate(self) -> None:
        if (
            self.page_size != 128
            or self.full_pages != 12
            or self.window_pages != 12
            or self.table_rows != 2
            or self.logical_width != 768
        ):
            raise ValueError("authority-rig cache geometry drift")
        if self.m1_positions != (255,) or self.m2_positions != (255, 383):
            raise ValueError("authority-rig position preregistration drift")
        if self.m1_stage_cap != 255 or self.m2_stage_cap != 383:
            raise ValueError("authority-rig stage-cap preregistration drift")
        if self.representatives != REPRESENTATIVE_LAYERS:
            raise ValueError("authority-rig representative-layer drift")
        if self.rounds != 40 or self.replays_per_round != 20:
            raise ValueError("authority-rig replay-count drift")
        if self.aa_null_pairs != 12 or self.inconclusive_reruns_max != 1:
            raise ValueError("authority-rig A/A or rerun policy drift")
        if len(self.m2_arm_order) != 12 or self.m2_arm_order[:2] != ("control", "full"):
            raise ValueError("authority-rig M2 arm-order drift")


FROZEN_PROTOCOL = FrozenProtocol()


@dataclass(frozen=True)
class WireLayout:
    rows: int
    hidden_offset: int
    weights_offset: int
    ids_offset: int
    total_bytes: int
    ids_alignment: int


def expected_wire_layout(rows: int, *, dim: int = 5120, topk: int = 6) -> WireLayout:
    """Return the byte-exact production wire contract for M=1/M=2."""

    if rows not in (1, 2):
        raise ValueError("Package-A wire rows must be 1 or 2")
    hidden_bytes = rows * dim * 2
    weights_bytes = rows * topk * 4
    ids_bytes = rows * topk * 4
    return WireLayout(
        rows=rows,
        hidden_offset=0,
        weights_offset=hidden_bytes,
        ids_offset=hidden_bytes + weights_bytes,
        total_bytes=hidden_bytes + weights_bytes + ids_bytes,
        ids_alignment=8 if rows == 1 else 16,
    )


def assert_real_wire_layout(hidden: torch.Tensor, topk: int = 6) -> dict[str, int]:
    """Construct the real candidate wire and reject any byte/alignment drift."""

    from freetoken.models.deepseek_v41.authority_fusion import _dispatch_wire

    wire, wire_hidden, weights, ids = _dispatch_wire(hidden, topk)
    expected = expected_wire_layout(hidden.shape[0], dim=hidden.shape[-1], topk=topk)
    base = wire.data_ptr()
    observed = {
        "hidden_offset": wire_hidden.data_ptr() - base,
        "weights_offset": weights.data_ptr() - base,
        "ids_offset": ids.data_ptr() - base,
        "total_bytes": wire.numel() * wire.element_size(),
        "ids_alignment": expected.ids_alignment,
    }
    for key in ("hidden_offset", "weights_offset", "ids_offset", "total_bytes"):
        if observed[key] != getattr(expected, key):
            raise AssertionError(f"wire {key} drift: {observed[key]} != {getattr(expected, key)}")
    if ids.data_ptr() % expected.ids_alignment:
        raise AssertionError("wire route-id alignment drift")
    return observed


@dataclass
class Boundary:
    token: int
    layer: int
    kind: str
    event: Any
    monotonic_ns: int


class BoundaryRecorder:
    """Mechanically define PRE boundaries from the production collective order.

    PRE(0) is token-start -> MoE broadcast(0).  PRE(l), l>0, is
    MoE all-reduce(l-1)-end -> MoE broadcast(l)-start.  The marker transport is
    installed only on ``MoEState._comm``; Engram collectives therefore cannot be
    mistaken for MoE boundaries.
    """

    def __init__(self, *, event_factory=None) -> None:
        self._event_factory = event_factory
        self._token = -1
        self._token_open = False
        self._records: list[Boundary] = []
        self._scope_layers: list[int] = []
        self._broadcast_layers: list[int] = []
        self._reduce_layers: list[int] = []
        self._wire_destinations: list[dict[str, int]] = []
        self._require_wire_destinations = False

    @property
    def records(self) -> tuple[Boundary, ...]:
        return tuple(self._records)

    def _event(self):
        if self._event_factory is not None:
            event = self._event_factory()
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA event requested without CUDA")
            # Make the capture emit literal event nodes for structural PRE
            # auditing; internal capture events are not an attribution oracle.
            event = torch.cuda.Event(enable_timing=True, external=True)
        event.record()
        return event

    def begin_token(self) -> None:
        if self._token_open:
            raise RuntimeError("PRE recorder token re-entered")
        self._token += 1
        self._token_open = True
        self._scope_layers = []
        self._broadcast_layers = []
        self._reduce_layers = []
        self._wire_destinations = []
        self._records.append(Boundary(self._token, 0, "token_start", self._event(), time.monotonic_ns()))

    def wire_destination(self, layer: int, wire_hidden: torch.Tensor) -> None:
        """Receipt the hidden and route-tail D2D targets of one real wire."""

        if not self._require_wire_destinations:
            raise RuntimeError("wire destination observed outside capture scope")
        if not self._token_open or layer != len(self._wire_destinations):
            raise RuntimeError(f"wire-destination order drift at layer {layer}")
        if self._scope_layers != list(range(layer + 1)):
            raise RuntimeError(f"wire destination lacks copy-scope start at layer {layer}")
        if (
            wire_hidden.dtype != torch.bfloat16
            or wire_hidden.ndim != 2
            or tuple(wire_hidden.shape) not in ((1, 5120), (2, 5120))
            or not wire_hidden.is_contiguous()
        ):
            raise RuntimeError(f"wire-hidden geometry drift at layer {layer}")
        layout = expected_wire_layout(int(wire_hidden.shape[0]))
        base_ptr = int(wire_hidden.data_ptr())
        storage = wire_hidden.untyped_storage()
        if int(storage.data_ptr()) != base_ptr or int(storage.nbytes()) < layout.total_bytes:
            raise RuntimeError(f"wire-hidden is not the leading view of the real wire at layer {layer}")
        self._wire_destinations.append(
            {
                "layer": layer,
                "base_ptr": base_ptr,
                "nbytes": layout.weights_offset,
                "wire_base_ptr": base_ptr,
                "wire_nbytes": layout.total_bytes,
                "tail_base_ptr": base_ptr + layout.weights_offset,
                "tail_nbytes": layout.total_bytes - layout.weights_offset,
            }
        )

    @contextlib.contextmanager
    def require_wire_destinations(self) -> Iterator[None]:
        if self._require_wire_destinations:
            raise RuntimeError("wire-destination capture re-entered")
        self._require_wire_destinations = True
        try:
            yield
        finally:
            self._require_wire_destinations = False

    def copy_scope_begin(self, layer: int) -> None:
        """Mark immediately after ffn_norm and before router/wire dispatch."""

        if not self._token_open or layer != len(self._scope_layers):
            raise RuntimeError(f"copy-scope order drift at layer {layer}")
        if self._broadcast_layers != list(range(layer)) or self._reduce_layers != list(
            range(layer)
        ):
            raise RuntimeError(f"copy-scope entered before prior layer completed: {layer}")
        self._scope_layers.append(layer)
        self._records.append(
            Boundary(self._token, layer, "copy_scope_begin", self._event(), time.monotonic_ns())
        )

    def broadcast_start(self, layer: int) -> None:
        if not self._token_open or layer != len(self._broadcast_layers):
            raise RuntimeError(f"MoE broadcast order drift at layer {layer}")
        if self._scope_layers != list(range(layer + 1)):
            raise RuntimeError(f"MoE broadcast lacks ffn_norm scope marker at layer {layer}")
        if layer and self._reduce_layers != list(range(layer)):
            raise RuntimeError(f"PRE({layer}) has no preceding all-reduce-end")
        self._broadcast_layers.append(layer)
        self._records.append(Boundary(self._token, layer, "broadcast_start", self._event(), time.monotonic_ns()))

    def all_reduce_end(self, layer: int) -> None:
        if not self._token_open or self._broadcast_layers != list(range(layer + 1)):
            raise RuntimeError(f"MoE all-reduce order drift at layer {layer}")
        if layer != len(self._reduce_layers):
            raise RuntimeError(f"duplicate/out-of-order all-reduce at layer {layer}")
        self._reduce_layers.append(layer)
        self._records.append(Boundary(self._token, layer, "all_reduce_end", self._event(), time.monotonic_ns()))

    def end_token(self) -> None:
        if (
            self._scope_layers != list(range(40))
            or self._broadcast_layers != list(range(40))
            or self._reduce_layers != list(range(40))
            or (
                self._require_wire_destinations
                and [item["layer"] for item in self._wire_destinations] != list(range(40))
            )
        ):
            raise RuntimeError("incomplete 40-layer authority collective/wire chain")
        self._records.append(Boundary(self._token, 39, "token_end", self._event(), time.monotonic_ns()))
        self._token_open = False

    def pre_event_pairs(self) -> dict[int, tuple[Any, Any]]:
        if self._token_open:
            raise RuntimeError("cannot segment an open token")
        latest = [r for r in self._records if r.token == self._token]
        by_key = {(r.kind, r.layer): r.event for r in latest}
        pairs = {0: (by_key[("token_start", 0)], by_key[("broadcast_start", 0)])}
        for layer in range(1, 40):
            pairs[layer] = (
                by_key[("all_reduce_end", layer - 1)],
                by_key[("broadcast_start", layer)],
            )
        return pairs

    def marker_receipt(self) -> tuple[dict[str, Any], ...]:
        """Expose semantic external-event identities for raw DOT resolution."""

        if self._token_open:
            raise RuntimeError("cannot receipt markers for an open token")
        latest = [record for record in self._records if record.token == self._token]
        receipt = []
        for record in latest:
            event_id = getattr(record.event, "event_id", None)
            cuda_event = getattr(record.event, "cuda_event", None)
            receipt.append(
                {
                    "kind": record.kind,
                    "layer": record.layer,
                    "communicator_scope": (
                        "MoEState._comm"
                        if record.kind in ("broadcast_start", "all_reduce_end")
                        else "authority-boundary"
                    ),
                    "event_id": int(event_id) if event_id is not None else None,
                    "cuda_event": int(cuda_event) if cuda_event is not None else None,
                }
            )
        return tuple(receipt)

    def wire_destination_receipt(self) -> tuple[dict[str, int], ...]:
        if self._token_open:
            raise RuntimeError("cannot receipt wire destinations for an open token")
        if [item["layer"] for item in self._wire_destinations] != list(range(40)):
            raise RuntimeError("wire destination receipt is incomplete")
        return tuple(dict(item) for item in self._wire_destinations)


class MarkerCommunicator:
    """Identity inter-rank stub which exposes only production MoE boundaries."""

    def __init__(self, recorder: BoundaryRecorder, layer: int) -> None:
        self.recorder = recorder
        self.layer = int(layer)

    def broadcast(self, tensor: torch.Tensor, root: int) -> torch.Tensor:
        if root != 0:
            raise RuntimeError("authority-rig broadcast root drift")
        self.recorder.broadcast_start(self.layer)
        return tensor

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.recorder.all_reduce_end(self.layer)
        return tensor

    def send(self, tensor: torch.Tensor, peer: int) -> torch.Tensor:
        raise RuntimeError("decode authority rig must not enter prefill send")

    def recv(self, tensor: torch.Tensor, peer: int) -> torch.Tensor:
        raise RuntimeError("decode authority rig must not enter prefill recv")


class ExpertComputeStub(torch.nn.Module):
    """Replace routed expert arithmetic, but retain production all-reduce code."""

    def __init__(self, *, layer: int, top_k: int, communicator: MarkerCommunicator) -> None:
        super().__init__()
        self.layer_id = int(layer)
        self.top_k = int(top_k)
        self.communicator = communicator
        self.packed_prefill_peer_ranks: tuple[int, ...] = ()
        self._authority_route_passthrough = False
        self._pending = None

    def prepare_packed_prefill_receive(self, *args, **kwargs) -> None:
        return None

    def _routes(self, hidden: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Deterministic, row/route-local arithmetic; routed expert compute is the
        # explicitly permitted stub.  The resulting 3-D payload follows the real
        # Package-A route-passthrough/all-reduce contract.
        return hidden[:, None, :].mul(weights[:, :, None].to(hidden.dtype))

    def _real_maybe_all_reduce(self, routes: torch.Tensor) -> torch.Tensor:
        from freetoken.models.deepseek_v41 import moe

        passthrough = self._authority_route_passthrough
        if passthrough:
            original = moe.DistributedCommunicator
            moe.DistributedCommunicator = lambda: self.communicator
            try:
                return moe.RoutedExperts._maybe_all_reduce(self, routes)
            finally:
                moe.DistributedCommunicator = original

        # Enter the actual production base reduction for the literal arm.  It
        # imports the communicator locally, so patch that factory rather than
        # copying the current reduction implementation into the rig.
        import freetoken.distributed as distributed
        from freetoken.layers.moe import ExpertParallelOffloadMoELayer

        original = distributed.DistributedCommunicator
        distributed.DistributedCommunicator = lambda: self.communicator
        try:
            return ExpertParallelOffloadMoELayer._maybe_all_reduce(self, routes)
        finally:
            distributed.DistributedCommunicator = original

    def routed_forward(self, hidden, weights, ids):
        return self._real_maybe_all_reduce(self._routes(hidden, weights))

    routed_decode_cache_safe = routed_forward

    def begin_routed_decode(self, weights, ids):
        self._pending = (weights, ids)
        return self._pending

    begin_routed_decode_cache_safe = begin_routed_decode

    def finish_routed_decode(self, hidden, weights, plan):
        if plan is not self._pending:
            raise RuntimeError("authority-rig refill plan identity drift")
        self._pending = None
        return self.routed_forward(hidden, weights, plan[1])


@dataclass
class TensorSnapshot:
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def capture(cls, named: Mapping[str, torch.Tensor]) -> "TensorSnapshot":
        return cls({name: tensor.clone() for name, tensor in named.items()})

    def restore(self, named: Mapping[str, torch.Tensor]) -> None:
        if set(named) != set(self.tensors):
            raise RuntimeError("authority-rig mutable-state inventory drift")
        for name, value in self.tensors.items():
            target = named[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise RuntimeError(f"authority-rig state shape/dtype drift: {name}")
            target.copy_(value)

    def digest(self) -> str:
        digest = hashlib.sha256()
        for name in sorted(self.tensors):
            value = self.tensors[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(repr(tuple(value.shape)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def assert_equal(self, other: "TensorSnapshot", *, label: str) -> None:
        """Byte-compare two snapshots without copying the complete cache to CPU."""

        if set(self.tensors) != set(other.tensors):
            raise AssertionError(f"{label} mutable-state inventory mismatch")
        for name in sorted(self.tensors):
            left, right = self.tensors[name], other.tensors[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise AssertionError(f"{label} mutable-state metadata mismatch: {name}")
            if not torch.equal(left.view(torch.uint8), right.view(torch.uint8)):
                raise AssertionError(f"{label} mutable-state bytes mismatch: {name}")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


@dataclass
class LayerStateSnapshot:
    """Post-layer oracle for G3 (HC outputs plus every mutable cache tensor)."""

    hidden: torch.Tensor
    incoming: torch.Tensor
    state: TensorSnapshot

    @classmethod
    def capture(cls, hidden: torch.Tensor, incoming: torch.Tensor, pool) -> "LayerStateSnapshot":
        return cls(
            hidden=hidden.detach().clone(),
            incoming=incoming.detach().clone(),
            state=TensorSnapshot.capture(mutable_attention_state(pool)),
        )

    def assert_equal(self, other: "LayerStateSnapshot", *, layer: int) -> None:
        label = f"layer {layer}"
        if not torch.equal(self.hidden.view(torch.uint8), other.hidden.view(torch.uint8)):
            raise AssertionError(f"{label} hidden bytes mismatch")
        if not torch.equal(self.incoming.view(torch.uint8), other.incoming.view(torch.uint8)):
            raise AssertionError(f"{label} incoming-pre bytes mismatch")
        self.state.assert_equal(other.state, label=label)

    def receipt(self) -> dict[str, Any]:
        return {
            "hidden_sha256": _tensor_sha256(self.hidden),
            "incoming_sha256": _tensor_sha256(self.incoming),
            "state_sha256": self.state.digest(),
            "state_inventory": tuple(sorted(self.state.tensors)),
        }

    def clone(self) -> "LayerStateSnapshot":
        return LayerStateSnapshot(
            self.hidden.clone(),
            self.incoming.clone(),
            TensorSnapshot({name: value.clone() for name, value in self.state.tensors.items()}),
        )


@dataclass
class CapturedAuthorityDecode:
    """One complete-token CUDA graph plus its static G3/PRE observables."""

    rig: "RealWeightAuthorityRig"
    graph: Any
    rows: int
    stage_cap: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    engram_rows: dict[int, torch.Tensor]
    output: torch.Tensor
    layers: dict[int, LayerStateSnapshot]
    boundaries: dict[int, tuple[Any, Any]]
    structure: tuple[tuple[str, int], ...]
    marker_events: tuple[dict[str, Any], ...]
    wire_destinations: tuple[dict[str, int], ...]
    batch: Any

    def replay(
        self, *, seed: int, positions: Sequence[int] | None = None
    ) -> tuple[
        torch.Tensor,
        dict[int, LayerStateSnapshot],
        TensorSnapshot,
        dict[int, float],
        dict[str, Any],
    ]:
        if positions is not None:
            if len(positions) != self.rows or any(int(p) > self.stage_cap for p in positions):
                raise ValueError("captured replay position/stage-cap drift")
            self.positions.copy_(
                torch.tensor(positions, dtype=self.positions.dtype, device=self.positions.device)
            )
        generator = torch.Generator(device=self.input_ids.device).manual_seed(seed)
        replacement = torch.randint(
            0,
            self.rig.transformer.args.vocab_size,
            self.input_ids.shape,
            generator=generator,
            device=self.input_ids.device,
        )
        self.input_ids.copy_(replacement)
        self.rig.fill_engram_rows(self.engram_rows, seed=seed)
        self.rig.reset()
        # This digest is the actual pre-replay state: reset first, receipt
        # second.  Capturing before reset accidentally receipts the previous
        # replay's dirty state and cannot prove replay independence.
        pre_reset_sha256 = TensorSnapshot.capture(
            mutable_attention_state(self.rig.pool)
        ).digest()
        measurement_start_monotonic_ns = time.monotonic_ns()
        self.graph.replay()
        torch.cuda.synchronize()
        measurement_end_monotonic_ns = time.monotonic_ns()
        if not bool(torch.isfinite(self.output).all().item()):
            raise AssertionError("captured authority output is non-finite")
        timings = {
            layer: float(start.elapsed_time(end))
            for layer, (start, end) in self.boundaries.items()
        }
        return (
            self.output.clone(),
            {layer: value.clone() for layer, value in self.layers.items()},
            TensorSnapshot.capture(mutable_attention_state(self.rig.pool)),
            timings,
            {
                "pre_reset_sha256": pre_reset_sha256,
                "measurement_start_monotonic_ns": measurement_start_monotonic_ns,
                "measurement_end_monotonic_ns": measurement_end_monotonic_ns,
                "pid": os.getpid(),
                "invocation_id": self.rig.invocation_id,
                "model_load_id": self.rig.model_load_id,
            },
        )


def mutable_attention_state(pool) -> dict[str, torch.Tensor]:
    """Inventory every mutable V4.1 attention/cache tensor used by decode."""

    result: dict[str, torch.Tensor] = {
        "full_to_window": pool.full_to_window,
    }
    if pool.full_loc_map is not None:
        result["full_loc_map"] = pool.full_loc_map
    for family in ("window_pool", "cmp_pool", "idx_pool"):
        for layer, value in enumerate(getattr(pool, family)):
            if value is not None:
                result[f"{family}.{layer}"] = value
    for family in ("state_ring", "indexer_state_ring"):
        for layer, value in enumerate(getattr(pool, family)):
            if value is not None:
                result[f"{family}.{layer}.buffer"] = value.buffer
    return result


def disjoint_full_loc_map(
    protocol: FrozenProtocol, *, device: torch.device
) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
    """Build two non-aliasing logical request rows and their page bindings."""

    protocol.validate()
    pages_per_row = protocol.logical_width // protocol.page_size
    if pages_per_row * protocol.table_rows != protocol.full_pages:
        raise RuntimeError("authority-rig full-page partition drift")
    mapping = torch.full(
        (protocol.table_rows, protocol.logical_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    bindings = []
    for row in range(protocol.table_rows):
        physical_row_base = row * protocol.logical_width
        mapping[row].copy_(
            torch.arange(
                physical_row_base,
                physical_row_base + protocol.logical_width,
                dtype=torch.int32,
                device=device,
            )
        )
        for page in range(pages_per_row):
            physical_page_base = physical_row_base + page * protocol.page_size
            bindings.append((physical_page_base, physical_page_base))
    if torch.isin(mapping[0], mapping[1]).any():
        raise RuntimeError("authority-rig request rows alias full-cache locations")
    return mapping, tuple(bindings)


def seed_attention_history(pool, *, seed: int = 0xA41) -> None:
    """Populate real cache payloads with deterministic non-degenerate history."""

    generator = torch.Generator(device=pool.device).manual_seed(seed)
    for name, tensor in mutable_attention_state(pool).items():
        if name in {"full_to_window", "full_loc_map"}:
            continue
        if tensor.is_floating_point():
            tensor.uniform_(-0.25, 0.25, generator=generator)


def reset_publication_state(transformer) -> None:
    """Clear per-forward publications and refuse pending Engram work."""

    from freetoken.models.deepseek_v41 import authority_fusion
    from freetoken.models.deepseek_v41 import model as model_module

    authority_fusion._FORWARD_MEMO.clear()
    model_module.shared_attention.reset()
    for layer in transformer.layers:
        table = getattr(getattr(layer, "engram", None), "table", None)
        coordinator = getattr(table, "coordinator", None)
        if coordinator is not None:
            for host_table in coordinator.tables.values():
                if getattr(host_table, "_pending", None) is not None:
                    raise RuntimeError("pending Engram prefetch at replay reset")


def process_identity_receipt(candidate_root: Path, extension_hashes: Mapping[str, str]) -> dict[str, Any]:
    """Fail closed on Python import and mapped native-extension identity."""

    import freetoken
    # Force both approved native dependencies into /proc/self/maps before the
    # receipt is evaluated; absence is not treated as evidence of correctness.
    import freetoken.kernel._cpu_moe  # noqa: F401
    import freetoken.kernel._pinned_tensor  # noqa: F401

    root = candidate_root.resolve()
    python_file = Path(freetoken.__file__).resolve()
    python_tree = (root / "python").resolve()
    if not python_file.is_relative_to(python_tree):
        raise RuntimeError(f"freetoken import escaped candidate: {python_file}")
    mapped: dict[str, dict[str, str]] = {}
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text(errors="replace").splitlines():
            path = line.split()[-1] if line.split() else ""
            if "/freetoken/" not in path or not path.endswith(".so"):
                continue
            real = Path(path).resolve()
            digest = hashlib.sha256(real.read_bytes()).hexdigest()
            mapped[str(real)] = {"sha256": digest}
    approved_links: dict[str, dict[str, str]] = {}
    unexpected_mapped = sorted(
        path for path in mapped if Path(path).name not in extension_hashes
    )
    if unexpected_mapped:
        raise RuntimeError(f"unapproved mapped freetoken extensions: {unexpected_mapped}")
    for basename, digest in extension_hashes.items():
        matches = [p for p, meta in mapped.items() if Path(p).name == basename and meta["sha256"] == digest]
        if len(matches) != 1:
            raise RuntimeError(f"mapped extension identity drift for {basename}: {matches}")
        links = list((python_tree / "freetoken").rglob(basename))
        if len(links) != 1:
            raise RuntimeError(f"candidate extension link inventory drift for {basename}: {links}")
        link = links[0]
        if not link.is_symlink() or link.resolve() != Path(matches[0]):
            raise RuntimeError(f"mapped extension escaped candidate symlink for {basename}")
        approved_links[basename] = {
            "link": str(link),
            "target": matches[0],
            "sha256": digest,
        }
    modules: dict[str, dict[str, str]] = {}
    for name, module in sorted(sys.modules.items()):
        if name != "freetoken" and not name.startswith("freetoken."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        displayed = Path(origin).absolute()
        real = displayed.resolve()
        if real.suffix == ".py" or real.name.endswith(".pyc"):
            source = real
            if real.name.endswith(".pyc") and "__pycache__" in real.parts:
                source = real.parent.parent / real.name.split(".", 1)[0]
                source = source.with_suffix(".py")
            if not source.is_relative_to(python_tree):
                raise RuntimeError(f"freetoken module escaped candidate: {name} -> {real}")
            digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else ""
            modules[name] = {"origin": str(displayed), "realpath": str(real), "sha256": digest}
        elif real.suffix == ".so":
            approval = approved_links.get(real.name)
            if approval is None or Path(approval["target"]) != real:
                raise RuntimeError(f"unapproved freetoken extension module: {name} -> {real}")
            modules[name] = {
                "origin": str(displayed),
                "realpath": str(real),
                "sha256": approval["sha256"],
            }
        else:
            raise RuntimeError(f"unclassified freetoken module origin: {name} -> {real}")
    return {
        "pid": os.getpid(),
        "executable": str(Path(sys.executable).resolve()),
        "candidate_root": str(root),
        "freetoken_file": str(python_file),
        "mapped_extensions": mapped,
        "candidate_extension_links": approved_links,
        "freetoken_modules": modules,
    }


def checkpoint_receipt(model_path: Path) -> dict[str, Any]:
    """Pin the config/index and exact shard inventory without hashing TBs twice."""

    model_path = model_path.resolve()
    required = (model_path / "config.json", model_path / "model.safetensors.index.json")
    if not all(path.is_file() for path in required):
        raise RuntimeError("authority-rig checkpoint config/index absent")
    index = json.loads(required[1].read_text())
    shards = sorted(set(index["weight_map"].values()))
    inventory = []
    for relative in shards:
        path = model_path / relative
        stat = path.stat()
        inventory.append({"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "model_path": str(model_path),
        "config_sha256": hashlib.sha256(required[0].read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(required[1].read_bytes()).hexdigest(),
        "shards": inventory,
    }


def _read_nodeids(path: Path) -> list[str]:
    nodes = [line.strip() for line in path.read_text().splitlines() if "::" in line]
    if not nodes or len(nodes) != len(set(nodes)):
        raise RuntimeError(f"invalid/duplicate CUDA node inventory: {path}")
    return nodes


def validate_junit_inventory(
    *,
    expected: Path,
    collected: Path,
    junit: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Require the exact pinned CUDA node set and zero non-pass outcomes.

    An xfail is encoded by pytest as ``<skipped type=\"pytest.xfail\">`` and is
    rejected with ordinary skips.  Deselecting a node changes the collected
    inventory and is rejected before the JUnit comparison.
    """

    digest = hashlib.sha256(expected.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"CUDA node inventory hash drift: {digest}")
    want, got = _read_nodeids(expected), _read_nodeids(collected)
    if got != want:
        raise RuntimeError("collected CUDA node IDs/order drift (including deselection)")
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    observed = []
    bad = []
    for case in cases:
        raw_file, name = case.attrib.get("file"), case.attrib.get("name")
        if not raw_file or not name:
            raise RuntimeError("JUnit lacks file/name; require junit_family=legacy")
        normalized = raw_file.replace("\\", "/")
        if "/tests/" in normalized:
            normalized = "tests/" + normalized.split("/tests/", 1)[1]
        observed.append(f"{normalized}::{name}")
        outcomes = [tag for tag in ("failure", "error", "skipped") if case.find(tag) is not None]
        if outcomes:
            bad.append({"nodeid": observed[-1], "outcomes": outcomes})
    if observed != want:
        raise RuntimeError("JUnit CUDA node IDs/order drift")
    if bad:
        raise RuntimeError(f"CUDA JUnit contains failure/error/skip/xfail: {bad}")
    return {
        "verdict": "PASS",
        "node_count": len(want),
        "node_inventory_sha256": digest,
        "zero_failure_error_skip_xfail_deselected": True,
    }


class RealWeightAuthorityRig:
    """Own the real authority model/state and expose exact G2/G3/G4 operations."""

    def __init__(self, transformer, pool, context, recorder, protocol=FROZEN_PROTOCOL):
        protocol.validate()
        self.transformer = transformer
        self.pool = pool
        self.context = context
        self.recorder = recorder
        self.protocol = protocol
        self.invocation_id: str | None = None
        self.model_load_id = str(uuid.uuid4())
        self._baseline = TensorSnapshot.capture(mutable_attention_state(pool))

    @classmethod
    def load(cls, *, model_path: Path, device: torch.device, protocol=FROZEN_PROTOCOL):
        """Load the genuine authority checkpoint and its mutable runtime state.

        This method is intentionally CUDA-only and is never called by G1/offline
        tests.  The later authorized gate supplies the authority UUID/ordinal.
        """

        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("real authority rig requires an authorized CUDA device")
        protocol.validate()
        from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend
        from freetoken.core import Context, set_global_ctx
        from freetoken.distributed import (
            override_tp_info,
            set_tp_info,
            try_get_tp_info,
        )
        from freetoken.engine.engine import _materialize_loaded_weight_state_dict
        from freetoken.kvcache.dsv41_cost_model import dsv41_pool_sizes
        from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
        from freetoken.models.deepseek_v41 import execution
        from freetoken.models.deepseek_v41.args import load_args
        from freetoken.models.deepseek_v41.engram import load_engram_host_table
        from freetoken.models.deepseek_v41.engram_runtime import EngramCoordinator
        from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM
        from freetoken.models.weight import load_weight
        from freetoken.utils import torch_dtype

        args = load_args(str(model_path), max_batch_size=2, max_seq_len=524288)
        plan = execution.DeepseekV41ExecutionPlan(**AUTHORITY_PLAN)
        tp_info = try_get_tp_info()
        if tp_info is None:
            set_tp_info(rank=0, size=1)
        elif (tp_info.rank, tp_info.size) != (0, 1):
            raise RuntimeError(
                f"authority-rig process has incompatible TP identity: {tp_info}"
            )
        previous_plan, execution._PLAN = execution._PLAN, plan
        try:
            config = SimpleNamespace(dsv41_args=args, is_multimodal=False)
            with plan.model_tp_context(), torch.device("meta"), torch_dtype(torch.bfloat16):
                adapter = DeepseekV41ForCausalLM(config)
            model_state = adapter.state_dict()
            with override_tp_info(0, 1):
                weights = load_weight(str(model_path), device, include_moe_experts=False)
                state = _materialize_loaded_weight_state_dict(model_state, weights, device=device)
            adapter.load_state_dict(state)
            transformer = adapter._model

            sizes = dsv41_pool_sizes(
                protocol.full_pages,
                args,
                1.0,
                P=protocol.page_size,
                n_win_pages=protocol.window_pages,
            )
            pool = DSV41PagedKVCache(
                sizes,
                args,
                device,
                P=protocol.page_size,
                n_scratch=protocol.scratch_rows,
            )
            pool.full_loc_map, bindings = disjoint_full_loc_map(protocol, device=device)
            for full_page_base, window_page_base in bindings:
                pool.bind_window_pages(full_page_base, window_page_base)
            context = Context(page_size=protocol.page_size)
            context.kv_cache = pool
            set_global_ctx(context)
            context.attn_backend = DSV41SparseAttnBackend(SimpleNamespace(dsv41_args=args))
            recorder = BoundaryRecorder()
            for layer, block in enumerate(transformer.layers):
                comm = MarkerCommunicator(recorder, layer)
                ffn = block.ffn
                ffn.execution = plan
                ffn.decode_partition = plan.partition(args.n_routed_experts)
                ffn.partition = ffn.decode_partition
                ffn.storage = plan.storage_partition(args.n_routed_experts)
                ffn._comm = comm
                ffn.experts = ExpertComputeStub(layer=layer, top_k=args.n_activated_experts, communicator=comm)

                def mark_scope(_module, _inputs, _output, *, _layer=layer):
                    recorder.copy_scope_begin(_layer)

                block.ffn_norm.register_forward_hook(mark_scope)

            # Use the production runtime attachment after installing the only
            # permitted stubs.  This warms direct-pre, Package A, candidate
            # bits and sparse-consumer variants exactly as serving does.
            transformer.attach_runtime(device)

            # The real table bytes and real Engram projection/gating remain in use.
            tables = {
                layer: load_engram_host_table(
                    str(model_path),
                    layer_id=layer,
                    num_embeddings=rows,
                    dim=args.engram_head_dim,
                    rank=0,
                    world_size=2,
                )
                for layer, rows in zip(args.engram_layer_ids, args.engram_num_embeddings)
            }
            class _EngramIdentityComm:
                def broadcast(self, tensor, root):
                    return tensor

                def all_reduce(self, tensor):
                    return tensor
            coordinator = EngramCoordinator(tables, execution=plan, communicator=_EngramIdentityComm())
            from freetoken.models.deepseek_v41.model import _CoordinatedEngramTable
            for layer in args.engram_layer_ids:
                transformer.attach_engram_table(layer, _CoordinatedEngramTable(coordinator, layer))

            seed_attention_history(pool)

            return cls(transformer, pool, context, recorder, protocol)
        except BaseException:
            execution._PLAN = previous_plan
            raise

    def reset(self) -> None:
        torch.cuda.synchronize()
        self._baseline.restore(mutable_attention_state(self.pool))
        reset_publication_state(self.transformer)
        torch.cuda.synchronize()

    def state_digest(self) -> str:
        return TensorSnapshot.capture(mutable_attention_state(self.pool)).digest()

    @contextlib.contextmanager
    def capture_representatives(
        self, *, fused: bool
    ) -> Iterator[dict[int, LayerStateSnapshot]]:
        """Capture the same post-layer boundary in literal and Package-A paths.

        The control wrapper records immediately after ``Block.decode_step``;
        the fused wrapper records immediately after ``authority_block_decode``.
        Both points are after that layer's MoE all-reduce and state publication,
        and before the next layer's MoE broadcast.  The wrappers are strictly
        test-local and are restored even when the forward raises.
        """

        captured: dict[int, LayerStateSnapshot] = {}
        representatives = frozenset(self.protocol.representatives)
        if fused:
            from freetoken.models.deepseek_v41 import authority_fusion

            original = authority_fusion.authority_block_decode

            @functools.wraps(original)
            def wrapped(block, *args, **kwargs):
                hidden, incoming, prepared = original(block, *args, **kwargs)
                layer = int(block.layer_id)
                if layer in representatives:
                    if layer in captured:
                        raise RuntimeError(f"duplicate fused G3 layer capture: {layer}")
                    captured[layer] = LayerStateSnapshot.capture(hidden, incoming, self.pool)
                return hidden, incoming, prepared

            authority_fusion.authority_block_decode = wrapped
            try:
                yield captured
            finally:
                authority_fusion.authority_block_decode = original
        else:
            originals: dict[int, tuple[bool, Any]] = {}
            for block in self.transformer.layers:
                layer = int(block.layer_id)
                if layer not in representatives:
                    continue
                had_instance = "decode_step" in block.__dict__
                original = block.decode_step
                originals[layer] = (had_instance, block.__dict__.get("decode_step"))

                @functools.wraps(original)
                def wrapped(*args, __layer=layer, __original=original, **kwargs):
                    hidden, incoming = __original(*args, **kwargs)
                    if __layer in captured:
                        raise RuntimeError(f"duplicate control G3 layer capture: {__layer}")
                    captured[__layer] = LayerStateSnapshot.capture(hidden, incoming, self.pool)
                    return hidden, incoming

                block.decode_step = wrapped
            try:
                yield captured
            finally:
                for block in self.transformer.layers:
                    layer = int(block.layer_id)
                    if layer not in originals:
                        continue
                    had_instance, value = originals[layer]
                    if had_instance:
                        block.decode_step = value
                    else:
                        delattr(block, "decode_step")

    def batch_inputs(
        self,
        rows: int,
        *,
        seed: int = 4101,
        positions_override: Sequence[int] | None = None,
        stage_cap_override: int | None = None,
    ):
        """Build the preregistered legal decode geometry and Engram row IDs."""

        if rows == 1:
            positions = self.protocol.m1_positions
            table_indices = self.protocol.m1_table_indices
            stage_cap = self.protocol.m1_stage_cap
        elif rows == 2:
            positions = self.protocol.m2_positions
            table_indices = self.protocol.m2_table_indices
            stage_cap = self.protocol.m2_stage_cap
        else:
            raise ValueError("authority rig supports only M=1/M=2")
        if positions_override is not None:
            if len(positions_override) != rows:
                raise ValueError("position override must have one position per row")
            positions = tuple(map(int, positions_override))
        if stage_cap_override is not None:
            stage_cap = int(stage_cap_override)
        if any(position < 0 or position > stage_cap for position in positions):
            raise ValueError("position/stage-cap geometry is not legal")
        if stage_cap >= self.protocol.full_pages * self.protocol.page_size:
            raise ValueError("stage cap exceeds authority-rig page geometry")
        device = self.pool.device
        positions_t = torch.tensor(positions, dtype=torch.int64, device=device)
        tables_t = torch.tensor(table_indices, dtype=torch.int64, device=device)
        from freetoken.attention.dsv4_sparse import DSV4AttnMetadata

        metadata = DSV4AttnMetadata(
            last_indices=torch.arange(rows, dtype=torch.int32, device=device),
            full_snap=self.pool.full_loc_map.index_select(0, tables_t).to(torch.int64),
            window_ar=torch.arange(self.protocol.page_size, device=device),
        )
        batch = SimpleNamespace(
            attn_metadata=metadata,
            is_decode=True,
            is_prefill=False,
            is_moe_prefill=False,
            speculative=False,
            padded_size=rows,
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        input_ids = torch.randint(
            0,
            self.transformer.args.vocab_size,
            (rows, 1),
            generator=generator,
            device=device,
        )
        engram_rows = {}
        for layer in self.transformer.args.engram_layer_ids:
            table = self.transformer.layers[layer].engram.table.coordinator.tables[layer]
            start, stop = table.plan.row_start, table.plan.row_end
            count = rows * self.transformer.args.engram_hashes_per_token
            span = stop - start - count + 1
            if span <= 0:
                raise RuntimeError("authority Engram shard is too small for rig rows")
            offset = int(seed) % span
            ids = torch.arange(
                start + offset,
                start + offset + count,
                device=device,
                dtype=torch.int64,
            )
            if int(ids[-1]) >= stop:
                raise RuntimeError("authority Engram shard is too small for rig rows")
            engram_rows[layer] = ids.view(rows, -1)
        return input_ids, positions_t, stage_cap, engram_rows, batch

    def fill_engram_rows(self, rows_by_layer: Mapping[int, torch.Tensor], *, seed: int) -> None:
        """Mutate captured Engram graph inputs within the real authority shards."""

        for layer, target in rows_by_layer.items():
            table = self.transformer.layers[layer].engram.table.coordinator.tables[layer]
            count = target.numel()
            start, stop = table.plan.row_start, table.plan.row_end
            span = stop - start - count + 1
            if span <= 0:
                raise RuntimeError("authority Engram shard is too small for captured rows")
            offset = int(seed) % span
            target.copy_(
                torch.arange(
                    start + offset,
                    start + offset + count,
                    device=target.device,
                    dtype=target.dtype,
                ).view_as(target)
            )

    def generated_positions(self, rows: int, state: int) -> tuple[int, ...]:
        """Generate legal, changing positions while retaining M1/M2 geometry."""

        if rows == 1:
            return (127 + int(state) % 129,)
        if rows == 2:
            return (127 + int(state) % 129, 383 + (int(state) * 7) % 385)
        raise ValueError("authority rig supports only M=1/M=2")

    def _run(
        self,
        *,
        fused: bool,
        rows: int,
        seed: int,
        policy=None,
        capture_layers=False,
        positions_override: Sequence[int] | None = None,
        stage_cap_override: int | None = None,
    ):
        from freetoken.models.deepseek_v41.authority_fusion import FULL_POLICY

        input_ids, positions, stage_cap, engram_rows, batch = self.batch_inputs(
            rows,
            seed=seed,
            positions_override=positions_override,
            stage_cap_override=stage_cap_override,
        )
        self.recorder.begin_token()
        with self.capture_representatives(fused=fused) if capture_layers else contextlib.nullcontext({}) as captures:
            with self.context.forward_batch(batch):
                if fused:
                    output = self._fused(
                        input_ids, positions, stage_cap, engram_rows, policy or FULL_POLICY
                    )
                else:
                    output = self.transformer.decode(
                        input_ids,
                        positions,
                        stage_cap,
                        engram_rows=engram_rows,
                    )
        self.recorder.end_token()
        if capture_layers and tuple(captures) != self.protocol.representatives:
            raise RuntimeError(f"incomplete G3 representative capture: {tuple(captures)}")
        return output, self.recorder.pre_event_pairs(), captures

    def _fused(self, input_ids, positions, stage_cap, engram_rows, policy):
        from freetoken.models.deepseek_v41.authority_fusion import decode_authority_fused

        return decode_authority_fused(
            self.transformer,
            input_ids,
            positions,
            stage_cap,
            engram_rows=engram_rows,
            policy=policy,
        )

    @contextlib.contextmanager
    def capture_wire_destinations(self, *, fused: bool) -> Iterator[None]:
        """Observe real per-layer wire buffers without changing their values/order."""

        from freetoken.models.deepseek_v41 import authority_fusion, moe

        if fused:
            original = authority_fusion._dispatch_wire

            @functools.wraps(original)
            def wrapped(hidden, topk):
                result = original(hidden, topk)
                layer = len(self.recorder._scope_layers) - 1
                self.recorder.wire_destination(layer, result[1])
                return result

            authority_fusion._dispatch_wire = wrapped
            try:
                with self.recorder.require_wire_destinations():
                    yield
            finally:
                authority_fusion._dispatch_wire = original
        else:
            original = moe._broadcast_decode_dispatch

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                # The literal control helper records broadcast_start inside
                # the wrapped call, so the just-completed layer is N-1.
                layer = len(self.recorder._broadcast_layers) - 1
                self.recorder.wire_destination(layer, result[0])
                return result

            moe._broadcast_decode_dispatch = wrapped
            try:
                with self.recorder.require_wire_destinations():
                    yield
            finally:
                moe._broadcast_decode_dispatch = original

    def capture_complete_token(
        self,
        *,
        fused: bool,
        rows: int,
        seed: int,
        policy=None,
        positions_override: Sequence[int] | None = None,
        stage_cap_override: int | None = None,
        pool=None,
        capture_layers: bool = True,
        keep_graph: bool = False,
    ) -> CapturedAuthorityDecode:
        """Capture exactly one full authority token including all PRE markers."""

        from freetoken.models.deepseek_v41.authority_fusion import FULL_POLICY

        input_ids, positions, stage_cap, engram_rows, batch = self.batch_inputs(
            rows,
            seed=seed,
            positions_override=positions_override,
            stage_cap_override=stage_cap_override,
        )
        # JIT and allocate every variant before capture.  This warm step is
        # never an observation and is reset completely before graph creation.
        self._run(
            fused=fused,
            rows=rows,
            seed=seed,
            policy=policy,
            positions_override=positions_override,
            stage_cap_override=stage_cap_override,
        )
        self.reset()
        graph = torch.cuda.CUDAGraph(keep_graph=keep_graph)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, pool=pool):
            with self.capture_wire_destinations(fused=fused):
                self.recorder.begin_token()
                capture_context = (
                    self.capture_representatives(fused=fused)
                    if capture_layers
                    else contextlib.nullcontext({})
                )
                with capture_context as captures:
                    with self.context.forward_batch(batch):
                        if fused:
                            output = self._fused(
                                input_ids,
                                positions,
                                stage_cap,
                                engram_rows,
                                policy or FULL_POLICY,
                            )
                        else:
                            output = self.transformer.decode(
                                input_ids,
                                positions,
                                stage_cap,
                                engram_rows=engram_rows,
                            )
                self.recorder.end_token()
        if keep_graph:
            # With keep_graph=True PyTorch intentionally defers instantiation.
            # Do it once here, before debug dumping and outside all G4 timing.
            graph.instantiate()
        marker_events = self.recorder.marker_receipt()
        event_ids = [item["event_id"] for item in marker_events]
        cuda_events = [item["cuda_event"] for item in marker_events]
        if (
            any(value is None for value in event_ids + cuda_events)
            or len(set(event_ids)) != len(event_ids)
            or len(set(cuda_events)) != len(cuda_events)
        ):
            raise RuntimeError("captured external CUDA marker identity is absent/ambiguous")
        structure = tuple(
            (record.kind, record.layer)
            for record in self.recorder.records
            if record.token == self.recorder._token and record.kind != "copy_scope_begin"
        )
        expected_structure = [("token_start", 0)]
        for layer in range(40):
            expected_structure.extend((("broadcast_start", layer), ("all_reduce_end", layer)))
        expected_structure.append(("token_end", 39))
        if structure != tuple(expected_structure):
            raise RuntimeError(f"captured PRE structure drift: {structure}")
        if capture_layers and tuple(captures) != self.protocol.representatives:
            raise RuntimeError(f"incomplete captured representatives: {tuple(captures)}")
        return CapturedAuthorityDecode(
            rig=self,
            graph=graph,
            rows=rows,
            stage_cap=stage_cap,
            input_ids=input_ids,
            positions=positions,
            engram_rows=engram_rows,
            output=output,
            layers=captures,
            boundaries=self.recorder.pre_event_pairs(),
            structure=structure,
            marker_events=marker_events,
            wire_destinations=self.recorder.wire_destination_receipt(),
            batch=batch,
        )

    def compare_captured_control_fused(
        self,
        rows: int,
        *,
        states: int = 256,
        seed_base: int = 9101,
    ) -> dict[str, Any]:
        """G3 whole-token graph equality under changing static-buffer inputs."""

        if states < 256:
            raise ValueError("captured G3 requires at least 256 generated states")
        capture_cap = 255 if rows == 1 else 767
        control = self.capture_complete_token(
            fused=False,
            rows=rows,
            seed=seed_base,
            stage_cap_override=capture_cap,
        )
        fused = self.capture_complete_token(
            fused=True,
            rows=rows,
            seed=seed_base,
            stage_cap_override=capture_cap,
            pool=control.graph.pool(),
        )
        receipts = []
        for index in range(states):
            seed = seed_base + index
            positions = self.generated_positions(rows, index)
            c_output, c_layers, c_state, _, c_meta = control.replay(
                seed=seed, positions=positions
            )
            f_output, f_layers, f_state, _, f_meta = fused.replay(
                seed=seed, positions=positions
            )
            if not torch.equal(c_output.view(torch.uint8), f_output.view(torch.uint8)):
                raise AssertionError(f"captured M{rows} output mismatch at state {index}")
            c_state.assert_equal(f_state, label=f"captured M{rows} state {index}")
            for layer in self.protocol.representatives:
                c_layers[layer].assert_equal(f_layers[layer], layer=layer)
            receipts.append(
                {
                    "state": index,
                    "control_output_sha256": _tensor_sha256(c_output),
                    "fused_output_sha256": _tensor_sha256(f_output),
                    "control_state_sha256": c_state.digest(),
                    "fused_state_sha256": f_state.digest(),
                    "control_pre_reset_sha256": c_meta["pre_reset_sha256"],
                    "fused_pre_reset_sha256": f_meta["pre_reset_sha256"],
                    "layers": {
                        str(layer): {
                            "control": c_layers[layer].receipt(),
                            "fused": f_layers[layer].receipt(),
                        }
                        for layer in self.protocol.representatives
                    },
                }
            )
        if len({item["control_output_sha256"] for item in receipts}) < 2:
            raise AssertionError(f"captured M{rows} generated states are degenerate")
        return {
            "rows": rows,
            "states": states,
            "shared_pool": True,
            "raw_states": receipts,
            "representatives": self.protocol.representatives,
        }

    def compare_shared_pool_graph_schedule(self, *, seed_base: int = 12101) -> dict[str, Any]:
        """Qualify bs2 -> bs1 -> bs2 graphs with different caps/inputs."""

        geometries = (
            (2, (255, 383), 383),
            (1, (511,), 511),
            (2, (639, 767), 767),
        )
        graph_pool = None
        pairs = []
        for index, (rows, positions, cap) in enumerate(geometries):
            control = self.capture_complete_token(
                fused=False,
                rows=rows,
                seed=seed_base + index,
                positions_override=positions,
                stage_cap_override=cap,
                pool=graph_pool,
            )
            if graph_pool is None:
                graph_pool = control.graph.pool()
            fused = self.capture_complete_token(
                fused=True,
                rows=rows,
                seed=seed_base + index,
                positions_override=positions,
                stage_cap_override=cap,
                pool=graph_pool,
            )
            pairs.append((control, fused))
        receipts = []
        for index, ((rows, positions, cap), (control, fused)) in enumerate(
            zip(geometries, pairs, strict=True)
        ):
            seed = seed_base + 100 + index
            c_output, c_layers, c_state, _, c_meta = control.replay(
                seed=seed, positions=positions
            )
            f_output, f_layers, f_state, _, f_meta = fused.replay(
                seed=seed, positions=positions
            )
            if not bool(torch.isfinite(c_output).all().item()) or not bool(
                torch.isfinite(f_output).all().item()
            ):
                raise AssertionError(f"shared-pool graph non-finite output at slot {index}")
            if not torch.equal(c_output.view(torch.uint8), f_output.view(torch.uint8)):
                raise AssertionError(f"shared-pool graph output mismatch at slot {index}")
            c_state.assert_equal(f_state, label=f"shared-pool graph slot {index}")
            for layer in self.protocol.representatives:
                c_layers[layer].assert_equal(f_layers[layer], layer=layer)
            receipts.append(
                {
                    "rows": rows,
                    "positions": positions,
                    "stage_cap": cap,
                    "control_output_sha256": _tensor_sha256(c_output),
                    "fused_output_sha256": _tensor_sha256(f_output),
                    "control_state_sha256": c_state.digest(),
                    "fused_state_sha256": f_state.digest(),
                    "control_pre_reset_sha256": c_meta["pre_reset_sha256"],
                    "fused_pre_reset_sha256": f_meta["pre_reset_sha256"],
                    "layers": {
                        str(layer): {
                            "control": c_layers[layer].receipt(),
                            "fused": f_layers[layer].receipt(),
                        }
                        for layer in self.protocol.representatives
                    },
                }
            )
        return {"order": (2, 1, 2), "shared_pool": True, "receipts": receipts}

    def compare_control_fused(
        self,
        rows: int,
        *,
        seed: int = 4101,
        positions_override: Sequence[int] | None = None,
        stage_cap_override: int | None = None,
    ) -> dict[str, Any]:
        """Byte-compare output and every mutable state after reset-equivalent runs."""

        self.reset()
        baseline_digest = self.state_digest()
        control, _, control_layers = self._run(
            fused=False,
            rows=rows,
            seed=seed,
            capture_layers=True,
            positions_override=positions_override,
            stage_cap_override=stage_cap_override,
        )
        control_state = TensorSnapshot.capture(mutable_attention_state(self.pool))
        control_bytes = control.detach().view(torch.uint8).clone()
        self.reset()
        if self.state_digest() != baseline_digest:
            raise AssertionError("authority-rig reset did not reproduce baseline state")
        fused, boundaries, fused_layers = self._run(
            fused=True,
            rows=rows,
            seed=seed,
            capture_layers=True,
            positions_override=positions_override,
            stage_cap_override=stage_cap_override,
        )
        fused_state = TensorSnapshot.capture(mutable_attention_state(self.pool))
        if not bool(torch.isfinite(control).all().item()) or not bool(
            torch.isfinite(fused).all().item()
        ):
            raise AssertionError("authority-rig control/fused output is non-finite")
        if not torch.equal(control_bytes, fused.detach().view(torch.uint8)):
            raise AssertionError("authority-rig control/fused output mismatch")
        if control_state.digest() != fused_state.digest():
            raise AssertionError("authority-rig control/fused mutable-state mismatch")
        for layer in self.protocol.representatives:
            control_layers[layer].assert_equal(fused_layers[layer], layer=layer)
        effective_positions = tuple(
            positions_override
            if positions_override is not None
            else (self.protocol.m1_positions if rows == 1 else self.protocol.m2_positions)
        )
        effective_stage_cap = (
            int(stage_cap_override)
            if stage_cap_override is not None
            else (self.protocol.m1_stage_cap if rows == 1 else self.protocol.m2_stage_cap)
        )
        return {
            "rows": rows,
            "positions": effective_positions,
            "stage_cap": effective_stage_cap,
            "control_output_sha256": hashlib.sha256(
                control_bytes.cpu().numpy().tobytes()
            ).hexdigest(),
            "fused_output_sha256": _tensor_sha256(fused),
            "control_state_sha256": control_state.digest(),
            "fused_state_sha256": fused_state.digest(),
            "layers": {
                str(layer): {
                    "control": control_layers[layer].receipt(),
                    "fused": fused_layers[layer].receipt(),
                }
                for layer in self.protocol.representatives
            },
            "pre_layers": tuple(boundaries),
        }

    def compare_generated_states(
        self, rows: int, *, states: int = 256, seed_base: int = 4101
    ) -> dict[str, Any]:
        """Exercise the frozen G3 minimum with exact whole-state comparisons."""

        if states < 256:
            raise ValueError("G3 requires at least 256 generated legal states")
        receipts = [
            self.compare_control_fused(
                rows,
                seed=seed_base + index,
                positions_override=self.generated_positions(rows, index),
                stage_cap_override=(255 if rows == 1 else 767),
            )
            for index in range(states)
        ]
        if len({item["control_output_sha256"] for item in receipts}) < 2:
            raise AssertionError(f"eager M{rows} generated states are degenerate")
        return {
            "rows": rows,
            "states": states,
            "representatives": REPRESENTATIVE_LAYERS,
            "raw_states": [
                {"state": index, **item} for index, item in enumerate(receipts)
            ],
        }

    def compare_page_crossing_sequence(
        self, *, seed_base: int = 7401
    ) -> dict[str, Any]:
        """Exercise the frozen C2 M2 boundary and exact M2 -> M1 recovery.

        Row A crosses 255 -> 256 while row B crosses 383 -> 384.  The sequence
        then cancels row B and advances row A alone before returning to M2.
        Every step is compared from an identical reset against the literal path;
        within each arm state is deliberately carried across steps.
        """

        sequence = (
            (2, (255, 383), 383),
            (2, (256, 384), 384),
            (1, (257,), 384),
            (2, (258, 385), 385),
        )

        def run_arm(fused: bool):
            outputs = []
            layers = []
            states = []
            for index, (rows, positions, cap) in enumerate(sequence):
                output, _boundaries, captures = self._run(
                    fused=fused,
                    rows=rows,
                    seed=seed_base + index,
                    capture_layers=True,
                    positions_override=positions,
                    stage_cap_override=cap,
                )
                outputs.append(output.detach().clone())
                layers.append(captures)
                states.append(TensorSnapshot.capture(mutable_attention_state(self.pool)))
            return outputs, layers, states

        self.reset()
        control = run_arm(False)
        self.reset()
        fused = run_arm(True)
        step_receipts = []
        for index, geometry in enumerate(sequence):
            if not bool(torch.isfinite(control[0][index]).all().item()) or not bool(
                torch.isfinite(fused[0][index]).all().item()
            ):
                raise AssertionError(f"page-crossing non-finite output at step {index}")
            if not torch.equal(control[0][index].view(torch.uint8), fused[0][index].view(torch.uint8)):
                raise AssertionError(f"page-crossing output mismatch at step {index}")
            control[2][index].assert_equal(fused[2][index], label=f"page-crossing step {index}")
            for layer in self.protocol.representatives:
                control[1][index][layer].assert_equal(fused[1][index][layer], layer=layer)
            step_receipts.append(
                {
                    "rows": geometry[0],
                    "positions": geometry[1],
                    "stage_cap": geometry[2],
                    "control_output_sha256": _tensor_sha256(control[0][index]),
                    "fused_output_sha256": _tensor_sha256(fused[0][index]),
                    "control_state_sha256": control[2][index].digest(),
                    "fused_state_sha256": fused[2][index].digest(),
                    "layers": {
                        str(layer): {
                            "control": control[1][index][layer].receipt(),
                            "fused": fused[1][index][layer].receipt(),
                        }
                        for layer in self.protocol.representatives
                    },
                }
            )
        return {
            "sequence": step_receipts,
            "page_crossing": ((255, 256), (383, 384)),
            "cancel_sequence": ("M2", "M2-cross", "M1", "M2-recovery"),
        }

    def compare_partner_mutation(self, *, seed: int = 4101) -> dict[str, Any]:
        """Prove M2 row A is unchanged when only the partner payload changes."""

        from freetoken.models.deepseek_v41.authority_fusion import FULL_POLICY

        inputs = list(self.batch_inputs(2, seed=seed))
        input_ids, positions, stage_cap, engram_rows, batch = inputs
        self.reset()
        self.recorder.begin_token()
        with self.context.forward_batch(batch):
            first = self._fused(input_ids, positions, stage_cap, engram_rows, FULL_POLICY)
        self.recorder.end_token()
        if not bool(torch.isfinite(first).all().item()):
            raise AssertionError("M2 partner-isolation baseline is non-finite")
        first_a = first[0].detach().view(torch.uint8).clone()
        self.reset()
        mutated = input_ids.clone()
        mutated[1, 0] = (mutated[1, 0] + 7919) % self.transformer.args.vocab_size
        mutated_rows = {layer: value.clone() for layer, value in engram_rows.items()}
        for value in mutated_rows.values():
            value[1].add_(31)
        self.recorder.begin_token()
        with self.context.forward_batch(batch):
            second = self._fused(mutated, positions, stage_cap, mutated_rows, FULL_POLICY)
        self.recorder.end_token()
        if not bool(torch.isfinite(second).all().item()):
            raise AssertionError("M2 partner-isolation mutation is non-finite")
        if not torch.equal(first_a, second[0].detach().view(torch.uint8)):
            raise AssertionError("M2 row A changed when only partner payload changed")
        if torch.equal(first[1].detach().view(torch.uint8), second[1].detach().view(torch.uint8)):
            raise AssertionError("partner mutation negative control was not observed")
        return {
            "row_a_before_sha256": hashlib.sha256(
                first_a.cpu().numpy().tobytes()
            ).hexdigest(),
            "row_a_after_sha256": _tensor_sha256(second[0]),
            "row_b_before_sha256": _tensor_sha256(first[1]),
            "row_b_after_sha256": _tensor_sha256(second[1]),
        }

    def measure_pre_arm(
        self,
        *,
        rows: int,
        arm: str,
        seed: int,
    ) -> dict[str, Any]:
        """Run one reset-separated replay with timing and exact-state receipts."""

        from freetoken.models.deepseek_v41.authority_fusion import FULL_POLICY, policy_without

        if arm == "control":
            fused, policy = False, None
        elif arm == "full":
            fused, policy = True, FULL_POLICY
        elif arm.startswith("minus_"):
            fused, policy = True, policy_without(arm.removeprefix("minus_"))
        else:
            raise ValueError(f"unknown G4 arm {arm}")
        self.reset()
        output, pairs, _ = self._run(fused=fused, rows=rows, seed=seed, policy=policy)
        torch.cuda.synchronize()
        timings = {layer: float(start.elapsed_time(end)) for layer, (start, end) in pairs.items()}
        if set(timings) != set(range(40)) or not torch.isfinite(output).all():
            raise RuntimeError("incomplete/non-finite G4 replay")
        return {
            "pre_ms": timings,
            "output_sha256": _tensor_sha256(output),
            "state_sha256": self.state_digest(),
        }

    def run_g4_matrix(
        self,
        rows: int,
        *,
        seed_base: int = 4101,
        graph_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Execute frozen rotation, including the A/A null self-qualification."""

        from freetoken.models.deepseek_v41.authority_fusion import FULL_POLICY, policy_without

        arms = self.protocol.m2_arm_order
        captured: dict[str, CapturedAuthorityDecode] = {}
        if graph_dir is not None:
            graph_dir.mkdir(parents=True, exist_ok=True)
        # Rotated arm order is deliberately different from capture order.
        # PyTorch shared-pool graphs are valid only when replayed in capture
        # order, so every arm owns an independent graph pool here.
        for arm in arms:
            if arm == "control":
                fused, policy = False, None
            elif arm == "full":
                fused, policy = True, FULL_POLICY
            else:
                fused, policy = True, policy_without(arm.removeprefix("minus_"))
            captured[arm] = self.capture_complete_token(
                fused=fused,
                rows=rows,
                seed=seed_base,
                policy=policy,
                capture_layers=False,
                keep_graph=graph_dir is not None,
            )
            if graph_dir is not None:
                dot_path = graph_dir / f"{arm}.dot"
                captured[arm].graph.debug_dump(str(dot_path))
                if not dot_path.is_file() or not dot_path.stat().st_size:
                    raise RuntimeError(f"CUDA graph debug dump absent/empty: {dot_path}")
                marker_path = graph_dir / f"{arm}.markers.json"
                marker_path.write_text(
                    json.dumps(
                        cudagraph_marker_sidecar(
                            captured=captured[arm], rows=rows, arm=arm, dot_path=dot_path
                        ),
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                )

        def replay(arm: str, seed: int) -> dict[str, Any]:
            output, _layers, state, timings, meta = captured[arm].replay(seed=seed)
            return {
                "pre_ms": timings,
                "logits_sha256": _tensor_sha256(output),
                "touched_state_sha256": state.digest(),
                **meta,
            }

        aa = []
        for arm in ("full", "control"):
            for pair in range(self.protocol.aa_null_pairs):
                a = replay(arm, seed_base + pair)
                b = replay(arm, seed_base + pair)
                if (a["logits_sha256"], a["touched_state_sha256"]) != (
                    b["logits_sha256"], b["touched_state_sha256"]
                ):
                    raise AssertionError(f"G4 A/A exactness drift for {arm} pair {pair}")
                for key in ("pid", "invocation_id", "model_load_id"):
                    if a[key] != b[key]:
                        raise RuntimeError(f"G4 A/A residency drift for {arm} pair {pair}: {key}")
                aa.append(
                    {
                        "arm": arm,
                        "pair": pair,
                        "round": pair,
                        "round_source": "preregistered-aa-pair-order",
                        "a_ms": [a["pre_ms"][layer] for layer in range(40)],
                        "b_ms": [b["pre_ms"][layer] for layer in range(40)],
                        "delta_ms": {
                            str(layer): b["pre_ms"][layer] - a["pre_ms"][layer]
                            for layer in range(40)
                        },
                        "a_logits_sha256": a["logits_sha256"],
                        "b_logits_sha256": b["logits_sha256"],
                        "a_touched_state_sha256": a["touched_state_sha256"],
                        "b_touched_state_sha256": b["touched_state_sha256"],
                        "a_pre_reset_sha256": a["pre_reset_sha256"],
                        "b_pre_reset_sha256": b["pre_reset_sha256"],
                        "a_measurement_start_monotonic_ns": a[
                            "measurement_start_monotonic_ns"
                        ],
                        "a_measurement_end_monotonic_ns": a[
                            "measurement_end_monotonic_ns"
                        ],
                        "b_measurement_start_monotonic_ns": b[
                            "measurement_start_monotonic_ns"
                        ],
                        "b_measurement_end_monotonic_ns": b[
                            "measurement_end_monotonic_ns"
                        ],
                        "pid": a["pid"],
                        "invocation_id": a["invocation_id"],
                        "model_load_id": a["model_load_id"],
                    }
                )
        measurements = []
        for round_id in range(self.protocol.rounds):
            rotated = arms[round_id % len(arms) :] + arms[: round_id % len(arms)]
            for position, arm in enumerate(rotated):
                replays = [
                    replay(
                        arm,
                        seed_base
                        + 1000
                        + round_id * self.protocol.replays_per_round
                        + replay_id,
                    )
                    for replay_id in range(self.protocol.replays_per_round)
                ]
                # Every arm must remain byte-exact with the literal control at
                # the same seed.  Since the rotation uses the same replay seed
                # for every arm in a round, evaluators can recompute this gate
                # without trusting a self-reported boolean.
                measurements.append(
                    {"round": round_id, "position": position, "arm": arm, "replays": replays}
                )
        exact_groups: dict[tuple[int, int], set[tuple[str, str]]] = {}
        for item in measurements:
            for replay_id, value in enumerate(item["replays"]):
                exact_groups.setdefault((item["round"], replay_id), set()).add(
                    (value["logits_sha256"], value["touched_state_sha256"])
                )
        drift = [key for key, values in exact_groups.items() if len(values) != 1]
        if drift:
            raise AssertionError(f"G4 arm exactness drift at round/replay: {drift[:8]}")
        return {
            "rows": rows,
            "protocol_sha256": protocol_receipt()["sha256"],
            "aa_null": aa,
            "measurements": measurements,
            "exact_groups": len(exact_groups),
            "inconclusive_reruns_max": self.protocol.inconclusive_reruns_max,
            "structure": captured["control"].structure,
            "control_marker_events": captured["control"].marker_events,
            "arm_graphs": {
                arm: {"dot": f"{arm}.dot", "markers": f"{arm}.markers.json"}
                for arm in arms
            }
            if graph_dir is not None
            else {},
        }

    def run_decode(self, *, input_ids, positions, stage_cap, engram_rows, policy, batch):
        from freetoken.models.deepseek_v41.authority_fusion import decode_authority_fused

        before = self.state_digest()
        self.recorder.begin_token()
        with self.context.forward_batch(batch):
            output = decode_authority_fused(
                self.transformer,
                input_ids,
                positions,
                stage_cap,
                engram_rows=engram_rows,
                policy=policy,
            )
        self.recorder.end_token()
        after = self.state_digest()
        return output, before, after, self.recorder.pre_event_pairs()


def protocol_receipt() -> dict[str, Any]:
    value = asdict(FROZEN_PROTOCOL)
    value["layer_classes"] = LAYER_CLASSES
    value["layer_weights"] = LAYER_WEIGHTS
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"protocol": value, "sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "APPROVED_EXTENSION_HASHES",
    "AUTHORITY_PLAN",
    "AUTHORITY_UUID",
    "BoundaryRecorder",
    "CUDA_NODE_COUNT",
    "CUDA_NODE_INVENTORY_SHA256",
    "ExpertComputeStub",
    "FROZEN_PROTOCOL",
    "LAYER_CLASSES",
    "LAYER_WEIGHTS",
    "LayerStateSnapshot",
    "MarkerCommunicator",
    "REPRESENTATIVE_LAYERS",
    "RealWeightAuthorityRig",
    "TensorSnapshot",
    "assert_real_wire_layout",
    "checkpoint_receipt",
    "cudagraph_marker_sidecar",
    "current_qualification_rig",
    "disjoint_full_loc_map",
    "validate_cudagraph_debug_dump_capability",
    "expected_wire_layout",
    "mutable_attention_state",
    "installed_qualification_rig",
    "process_identity_receipt",
    "seed_attention_history",
    "protocol_receipt",
    "reset_publication_state",
    "validate_junit_inventory",
    "validate_live_candidate_runtime",
    "validate_live_environment",
]
