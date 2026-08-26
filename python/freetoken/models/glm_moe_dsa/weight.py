"""Weight loading for GLM-5.2 (``glm_moe_dsa``).

Resident (non routed-expert) weights are bf16 in the checkpoint. What this loader
yields follows the quant modes RESOLVED IN ``parse_config`` (``ModelConfig.attn_quant``
/ ``dense_quant`` / ``lm_head_quant``, from the FREETOKEN_GLM_*_FP8 switches, default
on): in the default fp8 mode the big projections are requantized at load to W8A16
fp8-e4m3 with per-output-row scales (an extra ``*.weight_scale`` tensor per
projection); with the switches off everything streams through verbatim as bf16. The
router selection bias is remapped ``mlp.gate.e_score_correction_bias ->
mlp.e_score_correction_bias``; the DSA indexer tensors load bf16 on "full" indexer
layers (serving runs faithful DSA top-k sparse attention; see attention.py); only the
trailing MTP layer is skipped. Routed experts are either ModelOpt NVFP4 or native
mixed-Q2 GGUF bytes and go to the offload cache without bf16 materialization.

FTW caveat: an FTW checkpoint stores whatever iter_weights yielded at CONVERSION time,
and the model is built from the env at SERVE time -- the two must agree (a mismatch
fails loudly in load_state_dict on the ``*.weight_scale`` keys). The active modes are
logged at load so conversion logs record the choice.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)
from freetoken.models.loader import drop_page_cache
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config

# fp8-e4m3 dynamic range for the per-row W8A16 quantization of the big MLA projections.
_FP8_MAX = 448.0

_ROUTED_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale|weight_scale_2)$"
)


def _layer_to_bank(layer: int, config) -> int | None:
    if layer < config.first_k_dense_replace or layer >= config.num_layers:
        return None
    local_ids = getattr(config, "local_layer_ids", None)
    if local_ids is None:
        return layer - config.first_k_dense_replace
    local_moe = [i for i in local_ids if i >= config.first_k_dense_replace]
    try:
        return local_moe.index(layer)
    except ValueError:
        return None


_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_ROUTED_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.2 rank-local NVFP4 experts",
)


def _quant_fp8_per_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row fp8-e4m3 quantization: ``w ~= weight_fp8 * scale[:, None]``."""
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / _FP8_MAX).clamp(min=1e-12)
    q = (wf / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def dummy_moe_expert_sources(config, *, dtype: torch.dtype):
    """BF16 dummy banks with stage-local layers and unsharded expert widths."""
    from freetoken.kernel.pinned import copy_to_pinned_tensor

    layers = config.num_moe_layers
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    gate_up = [torch.randn(E, 2 * I, H, dtype=dtype) for _ in range(layers)]
    down = [torch.randn(E, H, I, dtype=dtype) for _ in range(layers)]
    if torch.cuda.is_available():
        gate_up = [copy_to_pinned_tensor(t) for t in gate_up]
        down = [copy_to_pinned_tensor(t) for t in down]
    return gate_up, down


def _gguf_q2_k_xl_types(layer: int) -> tuple[int, int]:
    """(gate/up, down) GGML types in Unsloth's GLM-5.2 UD-Q2_K_XL release."""
    from freetoken.models.gguf.dequant import (
        GGML_IQ2_XS,
        GGML_IQ3_XXS,
        GGML_IQ4_XS,
    )

    gate_up = GGML_IQ3_XXS if layer == 8 else GGML_IQ2_XS
    down = GGML_IQ4_XS if layer in (8, 75, 76, 77) else GGML_IQ3_XXS
    return gate_up, down


def load_gguf_q2_k_xl_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Load stage-local GLM UD-Q2_K_XL experts without dequantizing them.

    The release name is an importance-quant recipe, not one uniform type. Most
    routed gate/up tensors are IQ2_XS and down tensors IQ3_XXS; layer 8 promotes
    both sides and layers 75--77 promote down to IQ4_XS. Exact host sizes are
    retained; only the GPU slot cache uses a maximum stride.
    """
    from freetoken.models.gguf.dequant import row_bytes
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import HostBank, LayerCompletionTracker, PinPipeline

    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    args = getattr(config, "glm_dsa_args", None)
    global_E = getattr(args, "num_experts", E)
    if E == global_E:
        expert_start, expert_stop = 0, E
    else:
        from .config import ep_partition

        partition = ep_partition(global_E)
        assert partition.local_count == E
        expert_start, expert_stop = partition.global_offset, partition.global_stop
    configured_ids = getattr(config, "local_layer_ids", None)
    all_ids = range(config.num_layers) if configured_ids is None else configured_ids
    local_ids = tuple(i for i in all_ids if i >= config.first_k_dense_replace)
    layer_to_bank = {layer: slot for slot, layer in enumerate(local_ids)}
    types = [_gguf_q2_k_xl_types(layer) for layer in local_ids]
    hb = {"gate_up": [], "down": []}
    for gate_up_type, down_type in types:
        hb["gate_up"].append(
            HostBank((E, 2 * I, row_bytes(H, gate_up_type)), torch.uint8)
        )
        hb["down"].append(
            HostBank((E, H, row_bytes(I, down_type)), torch.uint8)
        )
    banks = {name: [bank.tensor for bank in layers] for name, layers in hb.items()}
    seen: dict[str, set[int]] = {"gate": set(), "up": set(), "down": set()}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None
        for tensor in iter_gguf_tensors(model_path):
            name = tensor.name
            if not name.startswith("blk.") or not name.endswith("_exps.weight"):
                continue
            layer = int(name.split(".")[1])
            slot = layer_to_bank.get(layer)
            if slot is None:
                continue
            gate_up_type, down_type = types[slot]
            if name.endswith("ffn_gate_exps.weight"):
                assert tensor.ggml_type == gate_up_type, (name, tensor.ggml_type)
                h_bytes = row_bytes(H, gate_up_type)
                banks["gate_up"][slot][:, :I].copy_(
                    tensor.packed().reshape(global_E, I, h_bytes)[expert_start:expert_stop]
                )
                role = "gate"
            elif name.endswith("ffn_up_exps.weight"):
                assert tensor.ggml_type == gate_up_type, (name, tensor.ggml_type)
                h_bytes = row_bytes(H, gate_up_type)
                banks["gate_up"][slot][:, I:].copy_(
                    tensor.packed().reshape(global_E, I, h_bytes)[expert_start:expert_stop]
                )
                role = "up"
            elif name.endswith("ffn_down_exps.weight"):
                assert tensor.ggml_type == down_type, (name, tensor.ggml_type)
                i_bytes = row_bytes(I, down_type)
                banks["down"][slot].copy_(
                    tensor.packed().reshape(global_E, H, i_bytes)[expert_start:expert_stop]
                )
                role = "down"
            else:
                continue
            drop_cache = getattr(tensor, "drop_cache", None)
            if drop_cache is not None:
                drop_cache()
            seen[role].add(layer)
            if tracker is not None:
                tracker.note(slot)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    want = set(local_ids)
    assert all(layers == want for layers in seen.values()), (
        "missing stage-local UD-Q2_K_XL experts: "
        + ", ".join(f"{role}={sorted(want - layers)}" for role, layers in seen.items())
    )
    return banks


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best effort
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.2 stores routed experts as NVFP4 and only supports the offload backend; "
        "experts are loaded into the offload cache via load_nvfp4_expert_sources()."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    from .execution import glm_pipeline_plan

    plan = glm_pipeline_plan(config.num_layers, config.glm_dsa_args.indexer_types)
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    dense = config.first_k_dense_replace
    attn_fp8 = config.attn_quant == "fp8_pertensor"
    mlp_fp8 = config.dense_quant == "fp8_pertensor"
    head_fp8 = config.lm_head_quant == "fp8_pertensor"
    if primary:
        from freetoken.utils import init_logger

        init_logger(__name__).info(
            f"GLM-5.2 resident quant: attn={config.attn_quant} dense={config.dense_quant} "
            f"lm_head={config.lm_head_quant} (FREETOKEN_GLM_ATTN_FP8/FREETOKEN_GLM_MLP_FP8; "
            "an FTW conversion records these choices implicitly -- serve with the same flags)"
        )
    try:
        for layer in tqdm(
            plan.layer_ids,
            desc="Loading GLM-5.2 dense weights",
            disable=not primary,
        ):
            a = f"model.layers.{layer}.self_attn"
            fp8_projs = (
                ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj") if attn_fp8 else ()
            )
            for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                w = reader.get(f"{a}.{proj}.weight")
                if proj in fp8_projs:
                    q, scale = _quant_fp8_per_row(w)
                    yield f"{a}.{proj}.weight", q
                    yield f"{a}.{proj}.weight_scale", scale
                else:
                    yield f"{a}.{proj}.weight", w
            for norm in ("q_a_layernorm", "kv_a_layernorm"):
                yield f"{a}.{norm}.weight", reader.get(f"{a}.{norm}.weight")
            # DSA lightning indexer ("full" layers only; "shared" layers reuse their
            # group leader's selection and ship no indexer tensors). Always bf16.
            idx_types = config.glm_dsa_args.indexer_types
            if idx_types and idx_types[layer] == "full":
                for proj in ("wq_b", "wk", "weights_proj"):
                    yield f"{a}.indexer.{proj}.weight", reader.get(f"{a}.indexer.{proj}.weight")
                yield f"{a}.indexer.k_norm.weight", reader.get(f"{a}.indexer.k_norm.weight")
                yield f"{a}.indexer.k_norm.bias", reader.get(f"{a}.indexer.k_norm.bias")
            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield (
                    f"model.layers.{layer}.{norm}.weight",
                    reader.get(f"model.layers.{layer}.{norm}.weight"),
                )

            m = f"model.layers.{layer}.mlp"

            def _mlp_weight(key: str):
                w = reader.get(f"{key}.weight")
                if mlp_fp8:
                    q, scale = _quant_fp8_per_row(w)
                    yield f"{key}.weight", q
                    yield f"{key}.weight_scale", scale
                else:
                    yield f"{key}.weight", w

            if layer < dense:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _mlp_weight(f"{m}.{proj}")
            else:
                yield f"{m}.gate.weight", reader.get(f"{m}.gate.weight")
                yield (
                    f"{m}.e_score_correction_bias",
                    reader.get(f"{m}.gate.e_score_correction_bias").to(torch.bfloat16),
                )
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _mlp_weight(f"{m}.shared_experts.{proj}")

        if plan.is_first:
            yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight")
        if plan.is_last:
            yield "model.norm.weight", reader.get("model.norm.weight")
            head = reader.get("lm_head.weight")
            if head_fp8 and not config.tie_word_embeddings:
                q, scale = _quant_fp8_per_row(head)
                yield "lm_head.weight", q
                yield "lm_head.weight_scale", scale
            else:
                yield "lm_head.weight", head
    finally:
        reader.close()


def _dequant_gguf_tensor(tensor, device: torch.device) -> torch.Tensor:
    """Materialize one non-expert GGUF tensor as bf16 on the serving device."""
    from freetoken.models.gguf.dequant import GGML_BF16, GGML_F16, GGML_F32, dequantize

    try:
        if tensor.ggml_type in (GGML_F32, GGML_F16, GGML_BF16):
            return dequantize(tensor.packed(), tensor.ggml_type, torch.bfloat16).reshape(
                tensor.shape
            ).to(device)
        if device.type != "cuda":
            raise NotImplementedError(
                "GLM mixed-quant GGUF dense weights currently dequantize through the CUDA "
                "ggml kernel; serve directly on CUDA (CPU FTW conversion is not yet supported)"
            )
        from freetoken.kernel.gguf import ggml_dequantize

        packed = tensor.packed().to(device)
        return ggml_dequantize(
            packed, tensor.ggml_type, tensor.rows, tensor.shape[-1], torch.bfloat16
        ).reshape(tensor.shape)
    finally:
        drop_cache = getattr(tensor, "drop_cache", None)
        if drop_cache is not None:
            drop_cache()


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Load resident GLM GGUF weights while routed experts remain packed/offloaded."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    assert not include_moe_experts, "GLM GGUF routed experts require --moe-backend offload"
    assert include_non_moe
    from .config import parse_gguf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    from .execution import glm_pipeline_plan

    plan = glm_pipeline_plan(config.num_layers, config.glm_dsa_args.indexer_types)
    attn_fp8 = config.attn_quant == "fp8_pertensor"
    mlp_fp8 = config.dense_quant == "fp8_pertensor"
    head_fp8 = config.lm_head_quant == "fp8_pertensor"
    kv_parts: dict[int, dict[str, torch.Tensor]] = {}

    def emit_projection(key: str, weight: torch.Tensor, use_fp8: bool):
        if use_fp8:
            quant, scale = _quant_fp8_per_row(weight)
            yield key, quant
            yield key.removesuffix(".weight") + ".weight_scale", scale
        else:
            yield key, weight

    direct_attention = {
        "attn_q_a.weight": "q_a_proj.weight",
        "attn_q_b.weight": "q_b_proj.weight",
        "attn_kv_a_mqa.weight": "kv_a_proj_with_mqa.weight",
        "attn_output.weight": "o_proj.weight",
    }
    attention_norms = {
        "attn_q_a_norm.weight": "q_a_layernorm.weight",
        "attn_kv_a_norm.weight": "kv_a_layernorm.weight",
    }
    indexer = {
        "indexer.attn_q_b.weight": "wq_b.weight",
        "indexer.attn_k.weight": "wk.weight",
        "indexer.proj.weight": "weights_proj.weight",
        "indexer.k_norm.weight": "k_norm.weight",
        "indexer.k_norm.bias": "k_norm.bias",
    }

    for tensor in iter_gguf_tensors(model_path):
        name = tensor.name
        if name == "token_embd.weight":
            if plan.is_first:
                yield "model.embed_tokens.weight", _dequant_gguf_tensor(tensor, device)
            continue
        if name == "output_norm.weight":
            if plan.is_last:
                yield "model.norm.weight", _dequant_gguf_tensor(tensor, device)
            continue
        if name == "output.weight":
            if plan.is_last:
                weight = _dequant_gguf_tensor(tensor, device)
                yield from emit_projection("lm_head.weight", weight, head_fp8)
            continue
        if not name.startswith("blk."):
            continue
        layer = int(name.split(".")[1])
        if not plan.owns_layer(layer):
            continue
        suffix = name.split(".", 2)[2]
        if "_exps.weight" in suffix or suffix.startswith("nextn."):
            continue
        base = f"model.layers.{layer}"
        attn = f"{base}.self_attn"

        if suffix == "attn_norm.weight":
            yield f"{base}.input_layernorm.weight", _dequant_gguf_tensor(tensor, device)
        elif suffix == "ffn_norm.weight":
            yield f"{base}.post_attention_layernorm.weight", _dequant_gguf_tensor(
                tensor, device
            )
        elif suffix in direct_attention:
            weight = _dequant_gguf_tensor(tensor, device)
            yield from emit_projection(
                f"{attn}.{direct_attention[suffix]}", weight, attn_fp8
            )
        elif suffix in attention_norms:
            yield f"{attn}.{attention_norms[suffix]}", _dequant_gguf_tensor(tensor, device)
        elif suffix in ("attn_k_b.weight", "attn_v_b.weight"):
            role = "k" if suffix.startswith("attn_k") else "v"
            kv_parts.setdefault(layer, {})[role] = _dequant_gguf_tensor(tensor, device)
            parts = kv_parts[layer]
            if len(parts) == 2:
                k = parts["k"].permute(0, 2, 1)
                v = parts["v"]
                yield f"{attn}.kv_b_proj.weight", torch.cat((k, v), dim=1).reshape(
                    -1, config.glm_dsa_args.kv_lora_rank
                ).contiguous()
                del kv_parts[layer]
        elif suffix in indexer:
            if config.glm_dsa_args.indexer_types[layer] == "full":
                yield f"{attn}.indexer.{indexer[suffix]}", _dequant_gguf_tensor(
                    tensor, device
                )
        elif layer < config.first_k_dense_replace and suffix.startswith("ffn_"):
            proj = suffix.removeprefix("ffn_").removesuffix(".weight")
            if proj in ("gate", "up", "down"):
                weight = _dequant_gguf_tensor(tensor, device)
                yield from emit_projection(
                    f"{base}.mlp.{proj}_proj.weight", weight, mlp_fp8
                )
        elif layer >= config.first_k_dense_replace:
            sparse = f"{base}.mlp"
            if suffix == "ffn_gate_inp.weight":
                yield f"{sparse}.gate.weight", _dequant_gguf_tensor(tensor, device)
            elif suffix == "exp_probs_b.bias":
                yield f"{sparse}.e_score_correction_bias", _dequant_gguf_tensor(
                    tensor, device
                )
            elif suffix.startswith("ffn_") and suffix.endswith("_shexp.weight"):
                proj = suffix.removeprefix("ffn_").removesuffix("_shexp.weight")
                weight = _dequant_gguf_tensor(tensor, device)
                yield from emit_projection(
                    f"{sparse}.shared_experts.{proj}_proj.weight", weight, mlp_fp8
                )

    assert not kv_parts, f"incomplete GGUF kv_b projection pairs: {sorted(kv_parts)}"


def load_nvfp4_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Load only this EP rank's contiguous GLM-5.2 expert shard."""
    from freetoken.moe.partition import ExpertPartition
    from .config import ep_partition
    from .execution import glm_pipeline_plan

    plan = glm_pipeline_plan(config.num_layers, config.glm_dsa_args.indexer_types)
    partition = (
        ExpertPartition(config.glm_dsa_args.num_experts)
        if plan.enabled
        else ep_partition(config.glm_dsa_args.num_experts)
    )

    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
        partition=partition,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str,
    config,
    *,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
):
    """Parallel-reader counterpart of :func:`load_nvfp4_expert_sources`."""
    from freetoken.moe.partition import ExpertPartition
    from .config import ep_partition
    from .execution import glm_pipeline_plan

    plan = glm_pipeline_plan(config.num_layers, config.glm_dsa_args.indexer_types)
    partition = (
        ExpertPartition(config.glm_dsa_args.num_experts)
        if plan.enabled
        else ep_partition(config.glm_dsa_args.num_experts)
    )

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
        partition=partition,
    )


__all__ = [
    "dummy_moe_expert_sources",
    "iter_gguf_weights",
    "iter_weights",
    "load_gguf_q2_k_xl_expert_sources",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
