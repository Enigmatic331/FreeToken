"""DeepSeek-V4.1-Flash runtime support.

The package is intentionally not entered in the model registry until its complete
text-model skeleton passes the checkpoint-load and oracle gates.  The Engram
primitives are usable independently while that port is in progress.
"""

from .engram import (
    EngramHasher,
    EngramHostTable,
    EngramLayout,
    EngramShardPlan,
    build_compressed_token_map,
    compute_engram_hash_ids,
    compute_hash_multipliers,
    load_engram_host_table,
)
from .engram_layer import Engram, EngramTable
from .engram_runtime import EngramCoordinator
from .execution import DeepseekV41ExecutionPlan
from .args import DeepseekV41Args, load_args
from .config import parse_config
from .attention_layout import AttentionCacheBytes, AttentionLayout, LayerAttentionPlan
from .hc import hc_mixes, hc_post, hc_pre, identity_pre_mix
from .layers import Linear, RMSNorm
from .moe import Gate
from .model import DeepseekV41ForCausalLM, Transformer
from .compress import Compressor
from .weight import (
    CheckpointPlan,
    ExpertShardPlan,
    TensorInfo,
    inspect_checkpoint,
    validate_resident_checkpoint,
    setup_offload_expert_banks,
)

__all__ = [
    "EngramHasher",
    "EngramHostTable",
    "EngramLayout",
    "EngramShardPlan",
    "Engram",
    "EngramTable",
    "EngramCoordinator",
    "build_compressed_token_map",
    "compute_engram_hash_ids",
    "compute_hash_multipliers",
    "load_engram_host_table",
    "DeepseekV41Args",
    "DeepseekV41ExecutionPlan",
    "DeepseekV41ForCausalLM",
    "CheckpointPlan",
    "AttentionCacheBytes",
    "AttentionLayout",
    "ExpertShardPlan",
    "load_args",
    "parse_config",
    "TensorInfo",
    "Transformer",
    "inspect_checkpoint",
    "validate_resident_checkpoint",
    "setup_offload_expert_banks",
    "Linear",
    "LayerAttentionPlan",
    "Compressor",
    "Gate",
    "RMSNorm",
    "hc_mixes",
    "hc_post",
    "hc_pre",
    "identity_pre_mix",
]
