from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from freetoken.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # Optional precomputed multimodal soft-token embeddings. The offline API can submit
    # these directly; online image requests populate them on the scheduler backbone rank.
    mm_embeds: torch.Tensor | None = None
    # Online multimodal requests arrive as CPU processor output. Only rank 0 sees
    # pixel_values; non-primary EP ranks receive the small metadata-only copy.
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    rope_positions: torch.Tensor | None = None  # CPU [prompt_tokens, 3]
    mrope_position_delta: int = 0
    is_multimodal: bool = False
    # Full SHA-256 identities in image-token-span order. These are safe across
    # tokenizer workers; the scheduler converts them to local radix symbols.
    image_cache_keys: list[bytes] | None = None
    # Exact half-open placeholder ranges corresponding to image_cache_keys.
    image_token_spans: list[tuple[int, int]] | None = None
    # Original compressed bytes/URL strings for a scheduler feature-cache miss.
    image_inputs: list[str | bytes] | None = field(default=None, repr=False)


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)
