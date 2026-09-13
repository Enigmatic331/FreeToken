"""Checkpoint planning and text-only weight streaming for DeepSeek-V4.1-Flash.

The official checkpoint mixes five independently placed payloads: the replicated
text backbone, routed experts, Engram tables, MTP/DSpark, and vision. Keeping the
classification explicit prevents the ordinary-generation baseline from pulling
the 189 GiB Engram tables or optional stacks through ``safe_open`` by accident.
"""

from __future__ import annotations

import json
import os
import re
import struct
from math import prod
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Literal

import safetensors
import torch

from freetoken.models.loader import drop_page_cache

from .args import DeepseekV41Args


Payload = Literal[
    "text_resident",
    "routed_experts",
    "engram_table",
    "mtp",
    "vision",
]


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    file_offset: int
    nbytes: int
    payload: Payload


@dataclass(frozen=True)
class CheckpointPlan:
    tensors: tuple[TensorInfo, ...]

    def bytes(self, payload: Payload) -> int:
        return sum(tensor.nbytes for tensor in self.tensors if tensor.payload == payload)

    def count(self, payload: Payload) -> int:
        return sum(tensor.payload == payload for tensor in self.tensors)

    @property
    def baseline_resident_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors if _is_baseline_resident(tensor))

    @property
    def baseline_device_bytes(self) -> int:
        """Resident state after the reference-fidelity load-time dtype conversions."""

        total = self.baseline_resident_bytes
        by_name = {tensor.name: tensor for tensor in self.tensors}
        for name, weight in by_name.items():
            if not name.endswith(".attn.wo_a.weight") or not _is_baseline_resident(weight):
                continue
            scale = by_name[name.removesuffix(".weight") + ".scale"]
            total -= weight.nbytes + scale.nbytes
            total += prod(weight.shape) * 2
        # The released head is BF16 on disk but FP32 in the reference model.
        total += by_name["head.weight"].nbytes
        # Ratio-2 pooling performs both projections in FP32. The presence of
        # wgate distinguishes those sources from the ratio-1 BF16 projection.
        for name, gate in by_name.items():
            if name.endswith(".attn.compressor.wgate.weight"):
                value = by_name[name.replace(".wgate.weight", ".wkv.weight")]
                total += gate.nbytes + value.nbytes
        return total


@dataclass(frozen=True)
class ExpertShardPlan:
    rank: int
    world_size: int
    global_offset: int
    local_count: int
    tensors: tuple[TensorInfo, ...]

    @property
    def source_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors)

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)


def _payload(name: str) -> Payload:
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith(("vision.", "aligner.", "image_")):
        return "vision"
    if ".engram.embed." in name:
        return "engram_table"
    if ".experts." in name:
        return "routed_experts"
    return "text_resident"


def _is_baseline_resident(tensor: TensorInfo) -> bool:
    # bias_vl is the only vision-only tensor nested inside the text layers.
    return tensor.payload == "text_resident" and not tensor.name.endswith(".ffn.gate.bias_vl")


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size)), 8 + size


def _weight_map(model_path: str) -> dict[str, str]:
    path = os.path.join(model_path, "model.safetensors.index.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["weight_map"]


def inspect_checkpoint(model_path: str) -> CheckpointPlan:
    """Read only safetensors headers and return the exact placement/byte plan."""

    weight_map = _weight_map(model_path)
    headers: dict[str, tuple[dict, int]] = {}
    tensors = []
    for name, shard in weight_map.items():
        if shard not in headers:
            headers[shard] = _read_header(os.path.join(model_path, shard))
        header, base = headers[shard]
        try:
            meta = header[name]
        except KeyError as exc:
            raise ValueError(f"index maps {name} to {shard}, but its header does not") from exc
        begin, end = meta["data_offsets"]
        tensors.append(
            TensorInfo(
                name=name,
                shard=shard,
                dtype=meta["dtype"],
                shape=tuple(meta["shape"]),
                file_offset=base + begin,
                nbytes=end - begin,
                payload=_payload(name),
            )
        )
    return CheckpointPlan(tuple(tensors))


def _linear_specs(
    result: dict[str, tuple[str, tuple[int, ...]]],
    prefix: str,
    out_features: int,
    in_features: int,
    *,
    dtype: str = "F8_E4M3",
) -> None:
    result[f"{prefix}.weight"] = (dtype, (out_features, in_features))
    if dtype == "F8_E4M3":
        result[f"{prefix}.scale"] = (
            "F8_E8M0",
            ((out_features + 31) // 32, (in_features + 31) // 32),
        )


def expected_resident_specs(args: DeepseekV41Args) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Raw checkpoint contract for the ordinary text-generation backbone."""

    result: dict[str, tuple[str, tuple[int, ...]]] = {
        "embed.weight": ("BF16", (args.vocab_size, args.dim)),
        "norm.weight": ("BF16", (args.dim,)),
        "head.weight": ("BF16", (args.vocab_size, args.dim)),
    }
    mix = (2 + args.hc_mult) * args.hc_mult
    hc_dim = args.hc_mult * args.dim
    for layer in range(args.n_layers):
        attn = f"layers.{layer}.attn"
        _linear_specs(result, f"{attn}.wq_a", args.q_lora_rank, args.dim)
        result[f"{attn}.q_norm.weight"] = ("BF16", (args.q_lora_rank,))
        _linear_specs(result, f"{attn}.wq_b", args.n_heads * args.head_dim, args.q_lora_rank)
        _linear_specs(result, f"{attn}.wkv", args.head_dim, args.dim)
        result[f"{attn}.kv_norm.weight"] = ("BF16", (args.head_dim,))
        _linear_specs(
            result,
            f"{attn}.wo_a",
            args.o_groups * args.o_lora_rank,
            args.n_heads * args.head_dim // args.o_groups,
        )
        _linear_specs(result, f"{attn}.wo_b", args.dim, args.o_groups * args.o_lora_rank)
        result[f"{attn}.attn_sink"] = ("F32", (args.n_heads,))

        if layer in args.kv_source_layers:
            compressor = f"{attn}.compressor"
            _linear_specs(result, f"{compressor}.wkv", args.head_dim, args.dim, dtype="BF16")
            if args.compress_ratios[layer] > 1:
                _linear_specs(
                    result, f"{compressor}.wgate", args.head_dim, args.dim, dtype="BF16"
                )
            result[f"{compressor}.norm.weight"] = ("BF16", (args.head_dim,))

        if layer in args.index_source_layers:
            indexer = f"{attn}.indexer"
            _linear_specs(
                result,
                f"{indexer}.wq_b",
                args.index_n_heads * args.index_head_dim,
                args.q_lora_rank,
            )
            _linear_specs(
                result,
                f"{indexer}.weights_proj",
                args.index_n_heads,
                args.dim,
                dtype="BF16",
            )
            if layer in args.kv_source_layers:
                _linear_specs(
                    result,
                    f"{indexer}.wk",
                    args.index_head_dim,
                    args.head_dim,
                    dtype="BF16",
                )
                result[f"{indexer}.k_norm.weight"] = ("BF16", (args.index_head_dim,))

        if layer in args.engram_layer_ids:
            engram = f"layers.{layer}.engram"
            _linear_specs(
                result,
                f"{engram}.wkv",
                args.dim * (args.hc_mult + 1),
                args.engram_hashes_per_token * args.engram_head_dim,
            )
            result[f"{engram}.q_weight"] = ("BF16", (args.hc_mult, args.dim))
            result[f"{engram}.k_weight"] = ("BF16", (args.hc_mult, args.dim))

        result[f"layers.{layer}.attn_norm.weight"] = ("BF16", (args.dim,))
        result[f"layers.{layer}.ffn_norm.weight"] = ("BF16", (args.dim,))
        gate = f"layers.{layer}.ffn.gate"
        result[f"{gate}.weight"] = ("BF16", (args.n_routed_experts, args.dim))
        result[f"{gate}.bias"] = ("F32", (args.n_routed_experts,))
        shared = f"layers.{layer}.ffn.shared_experts"
        _linear_specs(result, f"{shared}.w1", args.moe_inter_dim, args.dim)
        _linear_specs(result, f"{shared}.w2", args.dim, args.moe_inter_dim)
        _linear_specs(result, f"{shared}.w3", args.moe_inter_dim, args.dim)
        for sublayer in ("attn", "ffn"):
            result[f"layers.{layer}.hc_{sublayer}_fn"] = ("F32", (mix, hc_dim))
            result[f"layers.{layer}.hc_{sublayer}_base"] = ("F32", (mix,))
            result[f"layers.{layer}.hc_{sublayer}_scale"] = ("F32", (3,))
    return result


def validate_resident_checkpoint(checkpoint: CheckpointPlan, args: DeepseekV41Args) -> None:
    """Fail before allocation if the official text state differs from our model contract."""

    expected = expected_resident_specs(args)
    actual = {
        tensor.name: tensor for tensor in checkpoint.tensors if _is_baseline_resident(tensor)
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"resident checkpoint contract differs: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    errors = []
    for name, (dtype, shape) in expected.items():
        tensor = actual[name]
        if tensor.dtype != dtype or tensor.shape != shape:
            errors.append(
                f"{name}: expected {dtype} {shape}, got {tensor.dtype} {tensor.shape}"
            )
    if errors:
        raise ValueError("resident checkpoint metadata mismatch: " + "; ".join(errors[:8]))


_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)


def plan_expert_shard(
    checkpoint: CheckpointPlan,
    args: DeepseekV41Args,
    *,
    rank: int,
    world_size: int,
) -> ExpertShardPlan:
    """Select the exact contiguous main-model expert interval owned by one EP rank."""

    from freetoken.moe.partition import ExpertPartition

    partition = ExpertPartition(args.n_routed_experts, world_size=world_size, rank=rank)
    selected = []
    for tensor in checkpoint.tensors:
        match = _EXPERT_RE.match(tensor.name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        if layer < args.n_layers and partition.owns(expert):
            selected.append(tensor)
    expected = args.n_layers * partition.local_count * 6
    if len(selected) != expected:
        raise ValueError(f"expert shard has {len(selected)} tensors, expected {expected}")
    return ExpertShardPlan(
        rank=rank,
        world_size=world_size,
        global_offset=partition.global_offset,
        local_count=partition.local_count,
        tensors=tuple(selected),
    )


def _dequant_fp8_block32(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize one 32x32 E4M3/E8M0 matrix for the grouped ``wo_a`` einsum."""

    n, k = weight.shape
    values = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    values = values.repeat_interleave(32, 0).repeat_interleave(32, 1)[:n, :k]
    return (weight.float() * values).bfloat16()


def _tp2_ep2_resident_slice(name: str, value: torch.Tensor) -> torch.Tensor:
    """Slice the dense tensors sharded by the experimental TP2+EP2 model.

    Everything not listed here is deliberately replicated.  Attention assigns
    whole output groups to each rank; shared experts use conventional
    column/row parallel projections.  Routed experts remain whole-expert EP2.
    """

    from .execution import get_execution_plan

    execution = get_execution_plan()
    if not execution.tp2_ep2:
        return value
    rank, world = execution.rank, execution.world_size
    column = (
        ".attn.wq_b.weight",
        ".attn.wq_b.scale",
        ".ffn.shared_experts.w1.weight",
        ".ffn.shared_experts.w1.scale",
        ".ffn.shared_experts.w3.weight",
        ".ffn.shared_experts.w3.scale",
    )
    row = (
        ".attn.wo_b.weight",
        ".attn.wo_b.scale",
        ".ffn.shared_experts.w2.weight",
        ".ffn.shared_experts.w2.scale",
    )
    if name.endswith(column):
        return value.chunk(world, dim=0)[rank].contiguous()
    if name.endswith(row):
        return value.chunk(world, dim=1)[rank].contiguous()
    if name.endswith((".attn.wo_a", ".attn.attn_sink")):
        return value.chunk(world, dim=0)[rank].contiguous()
    return value


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream the text-only resident state without Engram tables or MTP.

    Routed experts have their own row-sharded host-bank loader and are never part
    of the GPU state dict. ``wo_a`` is expanded to BF16 under its parameter name,
    matching FreeToken's existing grouped-einsum DeepSeek path.
    """

    if include_moe_experts:
        raise ValueError("DeepSeek-V4.1 routed experts must use the offload/EP host banks")
    if not include_non_moe:
        return
    from .execution import get_execution_plan

    if get_execution_plan().is_expert_worker:
        return

    plan = inspect_checkpoint(model_path)
    by_shard: dict[str, list[TensorInfo]] = defaultdict(list)
    for tensor in plan.tensors:
        if _is_baseline_resident(tensor):
            by_shard[tensor.shard].append(tensor)

    for shard in sorted(by_shard):
        path = os.path.join(model_path, shard)
        with safetensors.safe_open(path, framework="pt", device=str(device)) as handle:
            for tensor in by_shard[shard]:
                name = tensor.name
                if name.endswith(".attn.wo_a.scale"):
                    continue
                value = handle.get_tensor(name)
                if name.endswith(".attn.wo_a.weight"):
                    prefix = name.removesuffix(".weight")
                    scale = handle.get_tensor(f"{prefix}.scale")
                    yield prefix, _tp2_ep2_resident_slice(
                        prefix, _dequant_fp8_block32(value, scale)
                    )
                elif name == "head.weight":
                    yield "head", value
                else:
                    yield name, _tp2_ep2_resident_slice(name, value)
        drop_page_cache(path)


def _expert_specs(args: DeepseekV41Args, local_experts: int):
    hidden, inter = args.dim, args.moe_inter_dim
    e8m0 = torch.float8_e8m0fnu
    return {
        "gate_up_packed": ((local_experts, 2 * inter, hidden // 2), torch.uint8),
        "gate_up_scale": ((local_experts, 2 * inter, hidden // 32), e8m0),
        "down_packed": ((local_experts, hidden, inter // 2), torch.uint8),
        "down_scale": ((local_experts, hidden, inter // 32), e8m0),
    }


def _place_expert(
    banks: dict[str, list[torch.Tensor]],
    name: str,
    tensor: torch.Tensor,
    *,
    inter_dim: int,
    global_offset: int,
) -> int:
    match = _EXPERT_RE.match(name)
    if match is None:
        raise ValueError(f"not a V4.1 routed-expert tensor: {name}")
    layer = int(match.group("layer"))
    expert = int(match.group("expert")) - global_offset
    projection, kind = match.group("proj"), match.group("kind")
    if kind == "weight":
        tensor = tensor.view(torch.uint8)
        if projection == "w1":
            banks["gate_up_packed"][layer][expert, :inter_dim] = tensor
        elif projection == "w3":
            banks["gate_up_packed"][layer][expert, inter_dim:] = tensor
        else:
            banks["down_packed"][layer][expert] = tensor
    elif projection == "w1":
        banks["gate_up_scale"][layer][expert, :inter_dim] = tensor
    elif projection == "w3":
        banks["gate_up_scale"][layer][expert, inter_dim:] = tensor
    else:
        banks["down_scale"][layer][expert] = tensor
    return layer


def load_dsfp4_expert_sources(
    model_path: str,
    args: DeepseekV41Args,
    *,
    rank: int,
    world_size: int,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Load only one rank's main-model FP4 experts into per-layer host banks.

    With ``layer_sink=None`` completed layers follow FreeToken's ambient
    pinned/locked/pageable residency plan. A sink is used by conversion and tests
    to consume completed layers without registering them with CUDA.
    """

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    shard = plan_expert_shard(inspect_checkpoint(model_path), args, rank=rank, world_size=world_size)
    host_banks = alloc_layer_banks(_expert_specs(args, shard.local_count), args.n_layers)
    banks = {name: [bank.tensor for bank in per_layer] for name, per_layer in host_banks.items()}
    by_file: dict[str, list[TensorInfo]] = defaultdict(list)
    for info in shard.tensors:
        by_file[info.shard].append(info)

    def load(sink) -> int:
        tracker = LayerCompletionTracker(shard.local_count * 6, host_banks, sink)
        placed = 0
        for filename in sorted(by_file):
            path = os.path.join(model_path, filename)
            with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
                for info in by_file[filename]:
                    layer = _place_expert(
                        banks,
                        info.name,
                        handle.get_tensor(info.name),
                        inter_dim=args.moe_inter_dim,
                        global_offset=shard.global_offset,
                    )
                    tracker.note(layer)
                    placed += 1
            drop_page_cache(path)
        return placed

    if layer_sink is None:
        with PinPipeline() as pipeline:
            placed = load(pipeline)
    else:
        placed = load(layer_sink)
    if placed != shard.tensor_count:
        raise RuntimeError(f"loaded {placed} expert tensors, planned {shard.tensor_count}")
    return banks


def setup_offload_expert_banks(
    model_path: str,
    model_config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    decode_target: str = "gpu",
    layer_sink=None,
):
    """Model-owned rank-local DS-FP4 bank provider for the EP2 topology."""

    if parallel:
        raise NotImplementedError(
            "V4.1 rank-sharded expert loading currently uses the serial safetensors path"
        )
    from dataclasses import replace

    from freetoken.moe.expert_banks import ExpertBanks

    from .execution import get_execution_plan

    args = model_config.dsv41_args
    execution = get_execution_plan()
    partition = execution.partition(args.n_routed_experts)
    if dummy:
        from freetoken.models.deepseek_v4.weight import dummy_dsfp4_expert_sources

        banks = dummy_dsfp4_expert_sources(
            replace(args, n_routed_experts=partition.local_count)
        )
    else:
        banks = load_dsfp4_expert_sources(
            model_path,
            args,
            rank=partition.rank,
            world_size=partition.world_size,
            layer_sink=layer_sink,
        )
    return ExpertBanks(
        "ds_fp4",
        {
            name: banks[name]
            for name in (
                "gate_up_packed",
                "gate_up_scale",
                "down_packed",
                "down_scale",
            )
        },
        streamed=layer_sink is not None and not dummy,
    )


__all__ = [
    "CheckpointPlan",
    "ExpertShardPlan",
    "TensorInfo",
    "expected_resident_specs",
    "inspect_checkpoint",
    "iter_weights",
    "load_dsfp4_expert_sources",
    "plan_expert_shard",
    "setup_offload_expert_banks",
    "validate_resident_checkpoint",
]
