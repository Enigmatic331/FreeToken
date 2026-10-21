"""DeepSeek-V4.1-specific Triton kernels."""

from .authority_fused import (
    hc_post_mixprep,
    hc_prepare_mix,
    hc_sinkhorn_rmsprep,
    router_scores_exact,
    router_wire_tail,
)
from .engram_gather import engram_gather_rows
from .fp8_linear import block_fp8_linear_32
from .route_partition import fused_localize_cache_safe_routes
from .rope_fp4 import rope_fp4_pack, rope_fp4_pack_into, rope_fp4_roundtrip
from .router import fused_sqrtsoftplus_topk

__all__ = [
    "block_fp8_linear_32",
    "engram_gather_rows",
    "fused_localize_cache_safe_routes",
    "fused_sqrtsoftplus_topk",
    "hc_post_mixprep",
    "hc_prepare_mix",
    "hc_sinkhorn_rmsprep",
    "rope_fp4_roundtrip",
    "router_scores_exact",
    "router_wire_tail",
    "rope_fp4_pack",
    "rope_fp4_pack_into",
]
