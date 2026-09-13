from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    rope_positions: torch.Tensor
    use_mrope: bool
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor
    engram_history: torch.Tensor | None
    engram_cu_seqlens: torch.Tensor | None
    engram_pad_id: int

    @classmethod
    def init(
        cls,
        bs: int,
        vocab_size: int,
        device: torch.device,
        *,
        use_mrope: bool = False,
        engram_history_width: int = 0,
        engram_pad_id: int = 0,
    ) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            rope_positions=torch.zeros((bs, 3), dtype=torch.int32, device=device),
            use_mrope=use_mrope,
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
            engram_history=(
                torch.full(
                    (bs, engram_history_width),
                    engram_pad_id,
                    dtype=torch.int64,
                    device=device,
                )
                if engram_history_width
                else None
            ),
            # Decode contributes exactly one token per request, hence arange.
            engram_cu_seqlens=(
                torch.arange(bs + 1, dtype=torch.int32, device=device)
                if engram_history_width
                else None
            ),
            engram_pad_id=int(engram_pad_id),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        if self.use_mrope:
            batch.rope_positions = self.rope_positions[_slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )
        if self.engram_history is not None:
            batch.engram_history = self.engram_history[_slice]
            batch.engram_cu_seqlens = self.engram_cu_seqlens[: bs + 1]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if self.use_mrope:
            if batch.rope_positions is None:
                self.rope_positions[_slice] = batch.positions[:, None]
            else:
                self.rope_positions[_slice] = batch.rope_positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx
        if self.engram_history is not None:
            history = self.engram_history[_slice]
            history.fill_(self.engram_pad_id)
            width = history.shape[1]
            for row, req in enumerate(batch.padded_reqs):
                prefix = req.input_ids[
                    max(0, req.cached_len - width) : req.cached_len
                ].long()
                take = min(width, prefix.numel())
                if take:
                    history[row, -take:].copy_(prefix[-take:], non_blocking=True)


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


def _dsv41_local_n_heads(model, args) -> int:
    """Return local attention heads, tolerating EP ranks with only MoE layers."""
    transformer = getattr(model, "_model", None)
    for block in getattr(transformer, "layers", ()):
        attention = getattr(block, "attn", None)
        if int(
            getattr(getattr(attention, "plan", None), "compress_ratio", 0)
        ) > 0:
            return int(attention.n_heads)
    return int(args.n_heads)


def _dsv41_graph_stage_ranges(
    args,
    *,
    batch_size: int,
    n_heads: int,
    max_seq_len: int,
    device: torch.device,
    split_counter: Callable[[int, int, int, int, torch.device], int] | None = None,
) -> tuple[tuple[int, int, int], ...]:
    """Group decode positions whose sparse-attention split topology is identical.

    A V4.1 graph needs a static compressed-history width large enough for every
    position it serves.  Capturing only at ``max_seq_len`` is numerically unsafe:
    ``split_count`` then bakes five split-K partitions into the graph even when an
    eager short-context launch uses two.  The resulting BF16 reduction-order delta
    compounds through the model.  Each returned ``(lo, hi, capture_cap)`` range
    therefore holds the split count constant for every positive compression ratio.
    The final range still captures at the admitted sequence ceiling so later KV is
    visible; device-side live counts mask its unused tail.
    """
    # A real decode follows at least one prefilled token, so position zero is
    # unreachable.  Leaving it eager avoids a one-use topology/capture variant.
    if max_seq_len <= 1:
        return ()
    ratios = tuple(sorted({int(r) for r in args.compress_ratios if int(r) > 0}))
    if not ratios:
        return ((1, max_seq_len - 1, max_seq_len - 1),)
    if split_counter is None:
        from freetoken.kernel.triton.dsv4.sparse_attn import split_count

        split_counter = split_count

    last_position = max_seq_len - 1
    # Once every ratio has reached index_topk, candidate width (and hence the
    # topology) cannot change again.  Avoid scanning a 32K/64K context ceiling.
    scan_end = min(last_position, max(ratios) * int(args.index_topk) - 1)

    def topology(position: int) -> tuple[int, ...]:
        return tuple(
            split_counter(
                batch_size,
                1,
                n_heads,
                int(args.window_size)
                + min(int(args.index_topk), (position + 1) // ratio),
                device,
            )
            for ratio in ratios
        )

    ranges: list[tuple[int, int, int]] = []
    start = 1
    current = topology(1)
    for position in range(2, scan_end + 1):
        candidate = topology(position)
        if candidate != current:
            ranges.append((start, position - 1, position - 1))
            start, current = position, candidate
    # The range's capture ceiling must cover every position it serves.  This is
    # especially important for the final, topology-stable long-context range.
    ranges.append((start, last_position, last_position))
    return tuple(ranges)


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.model = model
        self.stream = stream
        self.device = device
        self.graph_stage_ranges_by_bs: dict[
            int, tuple[tuple[int, int, int], ...]
        ] | None = None
        dsv41_args = getattr(getattr(model, "_config", None), "dsv41_args", None)
        if dsv41_args is not None and self.graph_bs_list:
            local_n_heads = _dsv41_local_n_heads(model, dsv41_args)
            self.graph_stage_ranges_by_bs = {
                bs: _dsv41_graph_stage_ranges(
                    dsv41_args,
                    batch_size=bs,
                    n_heads=local_n_heads,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                for bs in self.graph_bs_list
            }
        self._verify_replays = max(
            0, int(os.getenv("FREETOKEN_CUDA_GRAPH_VERIFY_STEPS", "0"))
        )
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[tuple[int, int | None], torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        if self.graph_stage_ranges_by_bs is None:
            capture_specs = [
                (bs, None) for bs in sorted(self.graph_bs_list, reverse=True)
            ]
        else:
            capture_specs = [
                (bs, cap)
                for bs in sorted(self.graph_bs_list, reverse=True)
                for _lo, _hi, cap in reversed(self.graph_stage_ranges_by_bs[bs])
            ]
            logger.info_rank0(
                "DeepSeek-V4.1 CUDA graph position ranges: %s",
                self.graph_stage_ranges_by_bs,
            )
        logger.info_rank0(
            "Start capturing CUDA graphs with (batch, stage-cap): %s", capture_specs
        )
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        qwen_args = getattr(getattr(model, "_config", None), "qwen4_args", None)
        use_mrope = bool(
            qwen_args
            and qwen_args.mrope_section
            and qwen_args.mrope_interleaved
        )
        dsv41_args = getattr(getattr(model, "_config", None), "dsv41_args", None)
        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs,
            vocab_size,
            self.device,
            use_mrope=use_mrope,
            engram_history_width=(
                int(dsv41_args.engram_max_ngram_size) - 1 if dsv41_args else 0
            ),
            engram_pad_id=(int(dsv41_args.engram_pad_id) if dsv41_args else 0),
        )
        self._reset_moe_offload_cache()

        pbar = tqdm(
            capture_specs,
            desc="Preparing for capturing CUDA graphs...",
            unit="graph",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs, stage_cap in pbar:
            free_memory = get_free_memory(self.device)
            cap_label = "default" if stage_cap is None else str(stage_cap)
            pbar.desc = (
                f"Capturing graphs: bs = {bs:<3} cap = {cap_label:<7} | "
                f"avail_mem = {mem_GB(free_memory)}"
            )
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            batch.dsv41_graph_stage_cap = stage_cap
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[(bs, stage_cap)] = graph

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if not batch.is_decode or batch.size > self.max_graph_bs:
            return False
        if self.graph_stage_ranges_by_bs is None:
            return True
        position = max(req.cached_len for req in batch.reqs)
        padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
        return any(
            lo <= position <= hi
            for lo, hi, _cap in self.graph_stage_ranges_by_bs[padded_size]
        )

    def _stage_cap_for_batch(self, batch: Batch) -> int | None:
        if self.graph_stage_ranges_by_bs is None:
            return None
        position = max(req.cached_len for req in batch.reqs)
        for lo, hi, cap in self.graph_stage_ranges_by_bs[batch.padded_size]:
            if lo <= position <= hi:
                return cap
        raise RuntimeError(
            f"decode position {position} is outside the captured V4.1 graph ranges"
        )

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        stage_cap = self._stage_cap_for_batch(batch)
        g = self.graph_map[(batch.padded_size, stage_cap)]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        if self._verify_replays:
            # Diagnostic-only shadow execution.  Re-running the same one-token forward is
            # state-idempotent for the transformer KV/carry stores and lets us compare graph
            # and eager logits from the same pre-step state without loading a second 400+ GiB
            # process.  Every EP rank enters the shadow forward, preserving collective order;
            # only rank 0 reports the real vocabulary logits.
            graph_logits = self.buffer.logits[: batch.size].clone()
            eager_logits = self.model.forward()
            if not get_tp_info().is_primary():
                self._verify_replays -= 1
            else:
                delta = (graph_logits - eager_logits[: batch.size]).abs()
                logger.info(
                    "CUDA graph shadow verification: remaining=%d max_abs=%g "
                    "graph_argmax=%s eager_argmax=%s exact=%s",
                    self._verify_replays,
                    float(delta.max().item()),
                    graph_logits.argmax(-1).tolist(),
                    eager_logits[: batch.size].argmax(-1).tolist(),
                    bool(torch.equal(graph_logits, eager_logits[: batch.size])),
                )
                self._verify_replays -= 1
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        gc.collect()
