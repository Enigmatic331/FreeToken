"""Default-off exact dependency-cone prefill for DeepSeek-V4.1.

With ``FREETOKEN_DSV41_DEPENDENCY_CONE=1``, long prompts run the decoder half of the model only
over the rows whose outputs can still reach state the prompt leaves behind. Earlier chunks run
the encoder half alone. This is not an approximation: the final logits and every cache entry a
later forward can read are the plain prefill's values, by construction.

Why rows can be skipped. ``decoder_source_layer`` (models/deepseek_v41/attention_layout.py)
checks this against the checkpoint's layer plan.

* Layers [0, D) form a causal encoder. D is the last compressed-KV source. Every layer from D on
  reads D's compressed KV and index keys, and these are per-row functions of D's input, the
  encoder output. Engram lives in the encoder. mHC mixes, norms, routing and experts are per row.
* So, from layer D on, row t couples to other rows only through sliding windows. Layer j reads
  its own window KV at rows [t - w + 1, t]. Layer D's window KV is also an encoder-output
  function, and the encoder pass stores it for every row.

Let the decoder start at row r0, and let every layer above D clamp its window at r0 (rows below
r0 never ran those layers). By induction over layers:

* layer j's output at row t is exact (equal to the plain prefill's) once t >= r0 + (j - D)(w - 1);
* layer j's window KV is a function of its input, so it is exact once t >= r0 + (j - 1 - D)(w - 1).

With L layers, the final logits (row n - 1, layer L - 1) are exact if
r0 <= n - 1 - (L - 1 - D)(w - 1). Every layer's window KV is exact from row
r0 + (L - 2 - D)(w - 1) on. That covers the decode window, and it bounds what the prefix cache may
publish. For V4.1 (L = 40, D = 20, w = 128) the two depths are 2,413 and 2,286 rows.

Scheduling. Every rank decides each chunk from its own replicated scheduler state, and
``agree_cone_segment`` proves they agree:

* ``encoder`` chunk [c, e) with e <= R* (``ConeGeometry.latest_decoder_start``): layers [0, D),
  then only layer D's window KV and compressed-KV/index-key publication. Expert workers run
  layers [0, D). Such a chunk is never a prompt's final chunk.
* The first chunk ending past R* starts the decoder: r0 is its start, so r0 <= R*. That chunk
  and every later one run all L layers with the plain chunk shapes. Windows of layers above D
  are clamped at r0.
* If the first chunk already ends past R*, the prompt runs the plain prefill unchanged.

No executed kernel sees a different shape. An encoder chunk runs the plain path's first D layers
over the same rows, and decoder chunks are the plain chunks. Only the clamped edge rows read
different window contents, and no kept value depends on them.

Prefix cache. Below the frontier F = page_ceil(r0 + (L - 2 - D)(w - 1)), decoder window KV was
never written or is inexact. A commit therefore records those positions as SWA tombstones, so no
later match ends inside them (``ConeRequestState.swa_commit_frontier``). R* also keeps F at or
below the finish-time retention start (scheduler/cache.py). The next turn's resume point near the
prompt end therefore stays exactly as reusable as after a plain prefill.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Callable

import torch

from freetoken.utils import align_down

DEPENDENCY_CONE_ENV = "FREETOKEN_DSV41_DEPENDENCY_CONE"
DEPENDENCY_CONE_MIN_TOKENS_ENV = "FREETOKEN_DSV41_DEPENDENCY_CONE_MIN_TOKENS"

ENCODER = "encoder"
DECODER = "decoder"
_KIND_CODES = {ENCODER: 1, DECODER: 2}
_CODE_KINDS = {code: kind for kind, code in _KIND_CODES.items()}

_WIRE_MAGIC = 0xC0DE_0001
_WIRE_VERSION = 1
_WIRE_LEN = 10


def dependency_cone_enabled() -> bool:
    """Strict parse: unset or ``0`` is off, ``1`` is on, anything else is a startup error."""
    value = os.environ.get(DEPENDENCY_CONE_ENV)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{DEPENDENCY_CONE_ENV} must be '0' or '1', got {value!r}")


def dependency_cone_min_tokens() -> int:
    """Minimum uncached prompt tokens that may engage the cone.

    Zero preserves the original cone-only policy.  A positive value lets one service enable
    both exact optimisations: short extensions remain eligible for tail pairing, while long
    uncached prefills use the dependency cone.  The decision uses token counts only.
    """
    value = os.environ.get(DEPENDENCY_CONE_MIN_TOKENS_ENV)
    if value is None:
        return 0
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{DEPENDENCY_CONE_MIN_TOKENS_ENV} must be a non-negative integer, got {value!r}"
        ) from exc
    if parsed < 0 or str(parsed) != value:
        raise ValueError(
            f"{DEPENDENCY_CONE_MIN_TOKENS_ENV} must be a canonical non-negative integer, "
            f"got {value!r}"
        )
    return parsed


class DependencyConeMismatch(RuntimeError):
    """EP ranks disagree on, or received malformed, dependency-cone chunk geometry."""


def _require_int(name: str, value) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


def _page_ceil(value: int, page: int) -> int:
    return -(-value // page) * page


@dataclass(frozen=True)
class ConeGeometry:
    """Model and cache constants that the cone arithmetic depends on."""

    n_layers: int  # L
    source_layer: int  # D, the first decoder layer
    window: int  # w, window keys per query including itself
    page_size: int  # P, the SWA radix / window page
    retain_gap: int  # finish-time SWA retention gap (scheduler/cache.py _SWA_RETAIN_GAP)
    min_uncached_tokens: int = 0  # admission policy; geometry remains exact above this threshold

    def __post_init__(self) -> None:
        for item in fields(self):
            _require_int(item.name, getattr(self, item.name))
        if not 0 < self.source_layer < self.n_layers:
            raise ValueError(f"source layer {self.source_layer} outside (0, {self.n_layers})")
        if (self.window < 1 or self.page_size < 1 or self.retain_gap < 0
                or self.min_uncached_tokens < 0):
            raise ValueError(f"invalid window/page/gap in {self}")

    @classmethod
    def from_args(cls, args, *, page_size: int, retain_gap: int) -> "ConeGeometry":
        from freetoken.models.deepseek_v41.attention_layout import decoder_source_layer

        return cls(
            n_layers=int(args.n_layers),
            source_layer=decoder_source_layer(args),
            window=int(args.window_size),
            page_size=int(page_size),
            retain_gap=int(retain_gap),
        )

    @property
    def logits_depth(self) -> int:
        """Rows past r0 before the last layer's output is exact."""
        return (self.n_layers - 1 - self.source_layer) * (self.window - 1)

    @property
    def window_depth(self) -> int:
        """Rows past r0 before every layer's window KV is exact."""
        return max(0, self.n_layers - 2 - self.source_layer) * (self.window - 1)

    def retention_start(self, prompt_len: int) -> int:
        """First position whose window KV the cache keeps live once a request finishes.
        Mirrors ``CacheManager._cache_req_swa``'s finish-time ``keep_from``."""
        prompt = align_down(prompt_len, self.page_size)
        return align_down(max(prompt - self.window - self.retain_gap, 0), self.page_size)

    def latest_decoder_start(self, prompt_len: int) -> int:
        """R*: the latest decoder start (r0) for which everything the prompt leaves behind is
        exact. Three bounds must hold:

        * the final logits: r0 <= n - 1 - logits_depth;
        * a commit must publish a live window: page_ceil(r0 + window_depth) <= page_floor(n) - P;
        * finish-time retention is unchanged: page_ceil(r0 + window_depth) <= retention_start(n).

        The decode window (rows >= n - w + 1) is implied by the first bound.
        """
        _require_int("prompt_len", prompt_len)
        logits = prompt_len - 1 - self.logits_depth
        publish = align_down(prompt_len, self.page_size) - self.page_size - self.window_depth
        retained = self.retention_start(prompt_len) - self.window_depth
        return min(logits, publish, retained)

    def tombstone_frontier(self, decoder_start: int) -> int:
        """F: the first page boundary from which every layer's window KV is exact."""
        _require_int("decoder_start", decoder_start)
        return _page_ceil(decoder_start + self.window_depth, self.page_size)


@dataclass(frozen=True)
class ConeSegment:
    """One scheduled prefill chunk's cone role, identical on every EP rank.

    ``kind == ENCODER``: run layers [0, source_layer), then only the source layer's window KV and
    compressed-KV/index-key publication. ``kind == DECODER``: run every layer, with the windows
    of layers above ``source_layer`` clamped at ``decoder_start``.
    """

    uid: int
    start: int
    end: int
    kind: str
    decoder_start: int  # -1 for ENCODER
    source_layer: int
    n_layers: int

    def __post_init__(self) -> None:
        for item in fields(self):
            if item.name != "kind":
                _require_int(item.name, getattr(self, item.name))
        if self.kind not in _KIND_CODES:
            raise ValueError(f"unknown cone kind {self.kind!r}")
        if self.uid < 0 or not 0 <= self.start < self.end:
            raise ValueError(f"invalid cone chunk in {self}")
        if not 0 < self.source_layer < self.n_layers:
            raise ValueError(f"invalid cone layers in {self}")
        if self.kind == ENCODER and self.decoder_start != -1:
            raise ValueError(f"an encoder chunk has no decoder start: {self}")
        if self.kind == DECODER and not 0 <= self.decoder_start <= self.start:
            raise ValueError(f"decoder start after its chunk: {self}")

    @property
    def is_encoder(self) -> bool:
        return self.kind == ENCODER

    @property
    def executed_layers(self) -> int:
        """Layers whose MoE/Engram collectives this chunk enters on every rank."""
        return self.source_layer if self.is_encoder else self.n_layers

    def window_floor(self, layer_id: int) -> int | None:
        """Window clamp of ``layer_id`` in this chunk, or None (the plain window)."""
        if self.is_encoder or layer_id <= self.source_layer:
            return None
        return self.decoder_start

    def encode(self) -> torch.Tensor:
        return torch.tensor(
            [_WIRE_MAGIC, _WIRE_VERSION, 1, self.uid, self.start, self.end,
             _KIND_CODES[self.kind], self.decoder_start, self.source_layer, self.n_layers],
            dtype=torch.int64,
        )

    @staticmethod
    def encode_none() -> torch.Tensor:
        return torch.tensor([_WIRE_MAGIC, _WIRE_VERSION] + [0] * (_WIRE_LEN - 2), dtype=torch.int64)

    @classmethod
    def decode(cls, wire: torch.Tensor) -> "ConeSegment | None":
        if (not isinstance(wire, torch.Tensor) or wire.dtype != torch.int64
                or tuple(wire.shape) != (_WIRE_LEN,) or wire.device.type != "cpu"):
            raise DependencyConeMismatch(f"malformed dependency-cone wire: {wire!r}")
        magic, version, present, uid, start, end, code, *rest = (int(v) for v in wire.tolist())
        if magic != _WIRE_MAGIC or version != _WIRE_VERSION or present not in (0, 1):
            raise DependencyConeMismatch(f"unknown dependency-cone wire header: {wire.tolist()}")
        if not present:
            if any((uid, start, end, code, *rest)):
                raise DependencyConeMismatch(f"absent cone segment carries data: {wire.tolist()}")
            return None
        if code not in _CODE_KINDS:
            raise DependencyConeMismatch(f"unknown cone kind code in {wire.tolist()}")
        try:
            return cls(uid, start, end, _CODE_KINDS[code], *rest)
        except (TypeError, ValueError) as exc:
            raise DependencyConeMismatch(f"invalid cone segment {wire.tolist()}: {exc}") from exc


@dataclass
class ConeRequestState:
    """One prompt's cone. Every chunk Req of the prompt shares this object, and it is the same
    on every rank because each rank derives it from its replicated scheduler state.

    It exists only for a prompt whose first scheduled chunk is encoder-only, so it is always
    engaged: some rows in [admitted_len, prompt_len) skip the decoder.
    """

    geometry: ConeGeometry
    uid: int
    prompt_len: int
    admitted_len: int  # cached prefix at admission
    latest_decoder_start: int  # R*
    encoded_until: int = -1  # end of the last encoder-only chunk
    decoder_start: int | None = None  # r0, set when the first decoder chunk is scheduled

    def __post_init__(self) -> None:
        for name in ("uid", "prompt_len", "admitted_len", "latest_decoder_start"):
            _require_int(name, getattr(self, name))
        if self.encoded_until == -1:
            self.encoded_until = self.admitted_len
        if not 0 <= self.admitted_len < self.prompt_len:
            raise ValueError(f"invalid admitted prefix in {self}")

    def classify(self, start: int, end: int) -> ConeSegment:
        """The segment of the next scheduled chunk [start, end); advances the state."""
        _require_int("start", start)
        _require_int("end", end)
        if not self.admitted_len <= start < end <= self.prompt_len:
            raise ValueError(f"chunk [{start}, {end}) outside prompt state {self}")
        if self.decoder_start is None:
            if start != self.encoded_until:
                raise ValueError(f"chunk [{start}, {end}) does not continue {self}")
            if end <= self.latest_decoder_start:
                self.encoded_until = end
                return self._segment(start, end, ENCODER, -1)
            if start == self.admitted_len:
                raise ValueError(f"first chunk [{start}, {end}) of {self} is not encoder-only")
            self.decoder_start = start
        elif start < self.decoder_start:
            raise ValueError(f"chunk [{start}, {end}) precedes the decoder start of {self}")
        return self._segment(start, end, DECODER, self.decoder_start)

    def _segment(self, start: int, end: int, kind: str, decoder_start: int) -> ConeSegment:
        g = self.geometry
        return ConeSegment(self.uid, start, end, kind, decoder_start, g.source_layer, g.n_layers)

    def swa_commit_frontier(self, insert_len: int) -> int | None:
        """SWA tombstone frontier for a commit of [0, insert_len), or None if the commit must not
        publish any new region.

        Every inserted position below the frontier must become a tombstone, and the commit's
        last page always stays live (SWARadixCache.insert keeps a live leaf). So publishing is
        safe only if the frontier lies at least one page before ``insert_len``. Before any
        decoder chunk has run, nothing this request wrote past ``admitted_len`` is exact.
        """
        _require_int("insert_len", insert_len)
        if self.decoder_start is None:
            return None
        frontier = self.geometry.tombstone_frontier(self.decoder_start)
        if frontier > insert_len - self.geometry.page_size:
            return None
        return frontier


def plan_prefill_cone(batch, geometry: ConeGeometry) -> ConeSegment | None:
    """The cone segment of a prefill ``batch`` just built by the PrefillManager; None for the
    plain prefill.

    A prompt's state is created at its first chunk, from the complete prompt length that
    ``batch.prompt_admissions`` records, and only if that chunk is encoder-only. Later chunks
    reach the state through ``Req.dsv41_cone``, which PrefillAdder copies to every continuation.
    """
    from .prefill import ChunkedReq

    if not batch.is_prefill or getattr(batch, "speculative", False):
        return None
    if len(batch.reqs) != 1:
        if any(getattr(req, "dsv41_cone", None) is not None for req in batch.reqs):
            raise RuntimeError("a dependency-cone prompt shares its prefill batch")
        return None
    req = batch.reqs[0]
    state = getattr(req, "dsv41_cone", None)
    if state is None:
        if not isinstance(req, ChunkedReq):
            return None  # a final chunk can only be a plain chunk when no cone ran before it
        admissions = [item for item in batch.prompt_admissions if item[0] == req.uid]
        if not admissions:
            return None  # a continuation whose first chunk did not engage the cone
        _, prompt_len, cached_len = admissions[0]
        if cached_len != req.cached_len:
            raise RuntimeError(f"cone admission {admissions[0]} does not match {req!r}")
        if int(prompt_len) - int(cached_len) < geometry.min_uncached_tokens:
            return None
        latest = geometry.latest_decoder_start(int(prompt_len))
        if req.device_len > latest:
            return None
        state = ConeRequestState(
            geometry, int(req.uid), int(prompt_len), int(cached_len), int(latest)
        )
        req.dsv41_cone = state
    if state.uid != req.uid:
        raise RuntimeError(f"cone state {state} attached to request {req.uid}")
    segment = state.classify(int(req.cached_len), int(req.device_len))
    if segment.is_encoder and not isinstance(req, ChunkedReq):
        raise RuntimeError(f"encoder-only cone chunk {segment} is a prompt's final chunk")
    return segment


def validate_cone_segment(segment: ConeSegment, req, *, n_layers: int, source_layer: int) -> None:
    """Model-side check, on every rank, that ``segment`` describes ``req``'s chunk."""
    if (segment.uid, segment.start, segment.end) != (req.uid, req.cached_len, req.device_len):
        raise DependencyConeMismatch(
            f"cone segment {segment} does not describe chunk [{req.cached_len}, "
            f"{req.device_len}) of request {req.uid}"
        )
    if (segment.n_layers, segment.source_layer) != (n_layers, source_layer):
        raise DependencyConeMismatch(
            f"cone segment {segment} disagrees with the model's {n_layers} layers / "
            f"decoder source layer {source_layer}"
        )
    if segment.is_encoder and req.can_decode:
        raise DependencyConeMismatch(f"encoder-only cone chunk {segment} would be sampled")


def agree_cone_segment(
    segment: ConeSegment | None,
    *,
    rank: int,
    world_size: int,
    broadcast: Callable[[torch.Tensor], torch.Tensor],
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor],
) -> ConeSegment | None:
    """Return rank 0's segment once every rank has proven it computed the same one.

    Rank 0 broadcasts its encoded segment, or "none". Each rank compares it with its own, and the
    mismatch flags are summed. Every rank therefore reaches the same verdict: all proceed, or all
    raise ``DependencyConeMismatch`` before the forward issues its first collective.
    """
    local = segment.encode() if segment is not None else ConeSegment.encode_none()
    if world_size <= 1:
        return segment
    reference = broadcast(local.clone() if rank == 0 else torch.zeros_like(local))
    mismatch = torch.tensor([int(not torch.equal(reference, local))], dtype=torch.int64)
    mismatch = all_reduce_sum(mismatch)
    if int(mismatch.item()) != 0:
        raise DependencyConeMismatch(
            f"rank {rank}: {int(mismatch.item())} rank(s) disagree with rank 0's "
            f"dependency-cone segment (rank 0 {reference.tolist()}, this rank {local.tolist()})"
        )
    return ConeSegment.decode(reference)


def dependency_cone_geometry(config) -> ConeGeometry | None:
    """The scheduler's cone geometry; None (the default) when the flag is off."""
    if not dependency_cone_enabled():
        return None
    args = getattr(config.model_config, "dsv41_args", None)
    if args is None:
        raise ValueError(f"{DEPENDENCY_CONE_ENV}=1 is only supported for DeepSeek-V4.1")
    from .cache import _SWA_RETAIN_GAP
    minimum = dependency_cone_min_tokens()
    try:
        return replace(
            ConeGeometry.from_args(
                args, page_size=int(config.page_size), retain_gap=_SWA_RETAIN_GAP
            ),
            min_uncached_tokens=minimum,
        )
    except ValueError as exc:
        raise ValueError(f"{DEPENDENCY_CONE_ENV}=1 requires the V4.1 layer plan: {exc}") from exc


def validate_dependency_cone_cache_window(
    geometry: ConeGeometry | None, cache_window: int | None
) -> None:
    """Fail at startup if scheduler/cache geometry diverges from the proof geometry."""
    if geometry is not None and cache_window != geometry.window:
        raise ValueError(
            f"{DEPENDENCY_CONE_ENV}=1 requires scheduler/cache sliding window "
            f"{geometry.window}, got {cache_window!r}"
        )


def dependency_cone_problems(
    config, plan, *, requested_max_running_req: int, concurrency: int = 1
) -> list[str]:
    """Why the current execution cannot run the cone, as unmet requirements; [] when it can.

    The engine checks this before the weight load. The scope is the qualified geometry: one
    request at a time on authority EP, with MTP/DSpark and early refill off. ``concurrency`` 2
    is DeepSeek-V4.1's gated C2, whose prefill forwards stay one request each, so a cone chunk
    still runs alone while the other request's decode waits. Tail pairing may be enabled
    alongside the cone: the scheduler applies it only to batches for which the cone did not
    engage.
    """
    from freetoken.engine.engine import _feature_concurrency_problems
    from freetoken.models.deepseek_v41.moe import prefill_early_refill_enabled

    problems = []
    args = getattr(config.model_config, "dsv41_args", None)
    if args is None:
        return ["DeepSeek-V4.1"]
    try:
        ConeGeometry.from_args(args, page_size=int(config.page_size), retain_gap=0)
    except ValueError as exc:
        problems.append(f"the V4.1 encoder-decoder layer plan ({exc})")
    if plan is None or not plan.uses_authority_transport:
        problems.append("DeepSeek-V4.1 authority-EP execution (--dsv41-backbone-rank)")
    problems.extend(
        _feature_concurrency_problems(config, requested_max_running_req, concurrency)
    )
    if getattr(config, "speculative_dspark", False) or getattr(args, "dspark_enabled", False):
        problems.append("MTP/DSpark off")
    try:
        dependency_cone_min_tokens()
    except ValueError as exc:
        problems.append(str(exc))
    if prefill_early_refill_enabled():
        problems.append("FREETOKEN_DSV41_PREFILL_EARLY_REFILL off")
    return problems


__all__ = [
    "DECODER",
    "DEPENDENCY_CONE_ENV",
    "DEPENDENCY_CONE_MIN_TOKENS_ENV",
    "ENCODER",
    "ConeGeometry",
    "ConeRequestState",
    "ConeSegment",
    "DependencyConeMismatch",
    "agree_cone_segment",
    "dependency_cone_enabled",
    "dependency_cone_geometry",
    "dependency_cone_min_tokens",
    "dependency_cone_problems",
    "plan_prefill_cone",
    "validate_dependency_cone_cache_window",
    "validate_cone_segment",
]
