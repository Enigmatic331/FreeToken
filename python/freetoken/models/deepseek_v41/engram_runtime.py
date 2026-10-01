"""Authority/worker synchronization for row-sharded Engram lookups."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from freetoken.distributed import DistributedCommunicator
from freetoken.utils import init_logger

from .engram import EngramHostTable
from .execution import DeepseekV41ExecutionPlan, get_execution_plan
from .profile import profile_range

logger = init_logger(__name__)
_TRACE_COLLECTIVES = os.getenv("FREETOKEN_DSV41_TRACE_COLLECTIVES", "0") == "1"


@dataclass
class TorchProcessGroupCommunicator:
    """Collective/P2P surface restricted to one NCCL process group."""

    group: dist.ProcessGroup
    name: str = "subgroup"
    _sequence: int = field(default=0, init=False, repr=False)

    def _trace(
        self,
        stage: str,
        operation: str,
        tensor: torch.Tensor,
        peer: int | None = None,
    ) -> None:
        if not _TRACE_COLLECTIVES:
            return
        stream = (
            torch.cuda.current_stream(tensor.device).cuda_stream
            if tensor.is_cuda
            else None
        )
        logger.info(
            "DSV41 collective %s name=%s seq=%d rank=%d group_rank=%d "
            "op=%s peer=%s shape=%s dtype=%s device=%s stream=%s",
            stage,
            self.name,
            self._sequence,
            dist.get_rank(),
            dist.get_rank(self.group),
            operation,
            peer,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            stream,
        )

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if dist.get_world_size(self.group) > 1:
            self._trace("enter", "all_reduce", tensor)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
            self._trace("exit", "all_reduce", tensor)
            self._sequence += 1
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        if dist.get_world_size(self.group) > 1:
            self._trace("enter", "broadcast", tensor, src)
            dist.broadcast(tensor, src=src, group=self.group)
            self._trace("exit", "broadcast", tensor, src)
            self._sequence += 1
        return tensor

    def send(self, tensor: torch.Tensor, dst: int) -> torch.Tensor:
        rank = dist.get_rank()
        if dist.get_world_size(self.group) != 2 or dst == rank:
            raise RuntimeError("subgroup send requires exactly one remote peer")
        # This communicator is used only by the two active prefill ranks. A
        # subgroup broadcast is therefore identical to point-to-point send but
        # avoids NCCL's lazy peer-link bootstrap, which can deadlock when the
        # process also owns a larger world communicator.
        self._trace("enter", "send_as_broadcast", tensor, dst)
        dist.broadcast(tensor, src=rank, group=self.group)
        self._trace("exit", "send_as_broadcast", tensor, dst)
        self._sequence += 1
        return tensor

    def recv(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        rank = dist.get_rank()
        if dist.get_world_size(self.group) != 2 or src == rank:
            raise RuntimeError("subgroup receive requires exactly one remote peer")
        self._trace("enter", "recv_as_broadcast", tensor, src)
        dist.broadcast(tensor, src=src, group=self.group)
        self._trace("exit", "recv_as_broadcast", tensor, src)
        self._sequence += 1
        return tensor


class EngramCoordinator:
    """Run the same broadcast/gather/all-reduce sequence on both EP ranks.

    The expert worker owns no dense blocks, but enters ``worker_lookup`` at layers
    1 and 14. The authority broadcasts global row IDs; both ranks gather their
    owned rows over UVA and all-reduce the 12 KiB/token partial result. Only the
    authority consumes the reconstructed rows.
    """

    def __init__(
        self,
        tables: Mapping[int, EngramHostTable],
        *,
        execution: DeepseekV41ExecutionPlan | None = None,
        communicator: DistributedCommunicator | None = None,
    ) -> None:
        self.tables = dict(tables)
        self.execution = execution or get_execution_plan()
        self.communicator = communicator or DistributedCommunicator()

    def _table(self, layer_id: int) -> EngramHostTable:
        try:
            return self.tables[layer_id]
        except KeyError as exc:
            raise KeyError(f"no Engram table attached for layer {layer_id}") from exc

    def authority_lookup(self, layer_id: int, row_ids: torch.Tensor) -> torch.Tensor:
        if not self.execution.is_backbone:
            raise RuntimeError("authority_lookup called on a V4.1 expert worker")
        if not self.execution.participates_in_engram:
            raise RuntimeError("the V4.1 backbone must participate in Engram")
        if self.execution.engram_world_size > 1:
            # The native NCCL bridge used by DistributedCommunicator does not
            # register an int64 datatype.  V4.1's largest Engram row id is only
            # ~384M, so int32 is an exact wire representation and is also what
            # the EP router already uses for its ids.
            with profile_range("DSV41/Engram/RowIdBroadcast"):
                row_ids = self.communicator.broadcast(
                    row_ids.to(torch.int32).contiguous(), self.execution.backbone_rank
                )
        return self._table(layer_id).lookup(
            row_ids,
            reduce=self.execution.engram_world_size > 1,
            communicator=self.communicator,
        )

    def worker_lookup(
        self,
        layer_id: int,
        *,
        num_tokens: int,
        hashes_per_token: int,
        device: torch.device,
    ) -> None:
        if not self.execution.is_expert_worker:
            raise RuntimeError("worker_lookup called on the V4.1 backbone authority")
        if not self.execution.participates_in_engram:
            raise RuntimeError("worker_lookup called on a non-Engram V4.1 rank")
        row_ids = torch.empty(
            (num_tokens, hashes_per_token), dtype=torch.int32, device=device
        )
        with profile_range("DSV41/Engram/RowIdBroadcast"):
            row_ids = self.communicator.broadcast(row_ids, self.execution.backbone_rank)
        self._table(layer_id).lookup(
            row_ids,
            reduce=True,
            communicator=self.communicator,
        )


__all__ = ["EngramCoordinator", "TorchProcessGroupCommunicator"]
