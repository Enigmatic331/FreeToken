"""Shared eager scratch for serial worker-only native Engram prefills.

Caller must qualify the same-stream NCCL subgroup and never retain the result.
Enabled by the coordinator only for the qualified worker/prefill geometry.
Decode and graph staging remain owned by the original per-table path.
"""
import torch


class WorkerPrefillScratch:
    def __init__(self):
        self.buffer = None
        self.stream = None

    def lookup_and_discard(self, table, row_ids, communicator):
        if (row_ids.ndim != 2 or row_ids.shape[1] != 24 or row_ids.shape[0] < 1
                or row_ids.dtype != torch.int32 or not row_ids.is_contiguous()
                or table.plan.dim != 256 or table.plan.world_size != 2):
            raise ValueError("expected native two-rank Engram geometry")
        if table._pending is not None:
            raise ValueError("shared scratch cannot consume a pending asynchronous lookup")
        if self.buffer is not None and self.buffer.device != row_ids.device:
            raise ValueError("one worker device per scratch")
        stream = None
        if row_ids.is_cuda:
            # This prototype has no decode/capture path. Production integration
            # must leave existing per-table graph staging entirely unchanged.
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("shared eager scratch must not enter CUDA capture")
            stream = torch.cuda.current_stream(row_ids.device)
            if self.stream is not None and self.stream != stream:
                stream.wait_stream(self.stream)
        rows = row_ids.numel()
        if self.buffer is None or self.buffer.shape[0] < rows:
            self.buffer = torch.empty((rows, 256), dtype=torch.bfloat16, device=row_ids.device)
        if stream is not None:
            # A later growth/rebuild may release this allocation after use on a
            # different stream from its allocator's creation stream.
            self.buffer.record_stream(stream)
        scratch = self.buffer[:rows]
        self.stream = stream
        # Same gather primitive and collective, same dtype, same row order.
        result = table._gather(row_ids, scratch)
        if result is not scratch:
            raise RuntimeError("gather did not return the provided scratch tensor")
        result = communicator.all_reduce(scratch)
        if result is not scratch:
            raise RuntimeError("qualified communicator must reduce in place")
        # Intentionally no returned tensor: workers do not consume these rows.
