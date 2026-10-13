"""Optional prefill-to-decode cache eviction priority for native DSv4.1 EP4.

Caller must fence cache-mutating side streams and run only at the phase boundary.
No slots/weights are freed, no mappings change, and no clock/stat is advanced.
"""
from contextlib import nullcontext

import torch
import triton
import triton.language as tl


@triton.jit
def _demote(owners, usage, C: tl.constexpr, E: tl.constexpr,
            START: tl.constexpr, STOP: tl.constexpr, BLOCK: tl.constexpr):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    owner = tl.load(owners + slot, slot < C, other=-1)
    expert = owner % E
    other_phase = (owner >= 0) & ((expert < START) | (expert >= STOP))
    tl.store(usage + slot, 0, (slot < C) & other_phase)


def demote(owners, usage, *, experts, start, count):
    if (owners.ndim != 1 or usage.shape != owners.shape or owners.numel() == 0
            or owners.dtype != torch.int32 or usage.dtype != torch.int64
            or owners.device != usage.device
            or not owners.is_contiguous() or not usage.is_contiguous()):
        raise ValueError("contiguous int32 owners and int64 usage on one device required")
    if (type(experts) is not int or experts <= 0 or type(start) is not int
            or type(count) is not int or min(start, count) < 0 or start + count > experts):
        raise ValueError("invalid storage-local decode range")
    if owners.is_cuda:
        _demote[(triton.cdiv(owners.numel(), 256),)](
            owners, usage, owners.numel(), experts, start, start + count, 256, num_warps=4)
    else:
        local = owners.remainder(experts)
        usage[(owners >= 0) & ((local < start) | (local >= start + count))] = 0


class PhaseCachePriority:
    def __init__(self, cache, *, start, count, max_decode_rows=1):
        if (cache.quant_format != "ds_fp4" or cache.decode_target not in ("gpu", "hybrid")
                or type(start) is not int or type(count) is not int
                or min(start, count) < 0 or start + count > cache.num_experts):
            raise ValueError("invalid native DS-FP4 decode ownership")
        if type(max_decode_rows) is not int or max_decode_rows not in (1, 2):
            raise ValueError("decode batches of one (C1) or two (C2) rows required")
        self.cache, self.start, self.count = cache, start, count
        # Decode ownership is per rank, not per request, so a C2 decode batch demotes alike.
        self.max_decode_rows = max_decode_rows
        self.pending = False

    @classmethod
    def from_engine(cls, engine):
        config, plan, cache = engine.config, engine._dsv41_plan, engine.moe_offload_cache
        rows = config.max_running_req
        # C1, or the gated C2: one-request prefills, decode graphs up to two rows.
        c2 = rows == 2 and getattr(config.model_config, "prefill_batch_size_limit", None) == 1
        if (plan is None or not plan.phase_aware or not (rows == 1 or c2)
                or config.speculative_dspark or config.cuda_graph_max_bs != rows
                or config.tp_info.size != 4 or cache is None or cache.num_layers != 40):
            raise ValueError("phase-cache priority currently requires target-only V4.1 EP4 at "
                             "one request, or C2 with one-request prefills")
        total = config.model_config.dsv41_args.n_routed_experts
        storage, decode = plan.storage_partition(total), plan.partition(total, prefill=False)
        if storage.local_count != cache.num_experts:
            raise ValueError("cache/storage expert geometry mismatch")
        obj = cls(cache, start=decode.global_offset - storage.global_offset,
                  count=decode.local_count, max_decode_rows=rows)
        obj.warm(engine.stream)
        return obj

    def warm(self, stream):
        """Compile/load outside collectives and capture, without changing the cache."""
        cache = self.cache
        cuda = cache.id_of_slot.is_cuda
        with torch.cuda.stream(stream) if cuda else nullcontext():
            owners = torch.full_like(cache.id_of_slot, -1)
            usage = torch.zeros_like(cache.usage)
            demote(owners, usage, experts=cache.num_experts, start=self.start, count=self.count)
            if cuda:
                # Startup or idle runtime rebuild only, never on a request/idle poll.
                stream.synchronize()

    def before(self, batch):
        """Call from Engine.forward_batch, outside the captured model forward."""
        rows = len(batch.reqs)
        if batch.speculative or not 1 <= rows <= (1 if batch.is_prefill else self.max_decode_rows):
            raise ValueError("phase-cache priority does not support speculative/concurrent batches")
        if batch.is_prefill:
            self.pending = True
            return
        if not self.pending:
            return  # No CUDA operation on subsequent decode steps.
        cache = self.cache
        cache.finish_background_admission()
        if cache.prefill_copy_stream is not None and cache.id_of_slot.is_cuda:
            torch.cuda.current_stream(cache.device).wait_stream(cache.prefill_copy_stream)
        demote(cache.id_of_slot, cache.usage, experts=cache.num_experts,
               start=self.start, count=self.count)
        trace = getattr(cache, "_admission_trace", None)
        if trace is not None:
            trace.note_demote(self.start, self.count)
        self.pending = False
