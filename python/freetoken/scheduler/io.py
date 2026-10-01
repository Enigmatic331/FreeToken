from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Final, List

import msgpack
import torch
from freetoken.message import (
    BaseBackendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    UserMsg,
)
from freetoken.utils import ZmqPullQueue, ZmqPushQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)
_MAX_RANK_MESSAGE_BYTES = 256 << 20
_RANK_IDLE_HEARTBEAT_MS = 1000
_TRACE_COLLECTIVES = os.getenv("FREETOKEN_DSV41_TRACE_COLLECTIVES", "0") == "1"


def _without_vision_pixels(msg: BaseBackendMsg) -> tuple[BaseBackendMsg, bool]:
    """Return the small EP-worker copy of a scheduler message.

    Only the backbone rank runs the visual tower. Broadcasting patch tensors to expert
    ranks wastes CPU copies and ZMQ bandwidth, while token ids and MRoPE coordinates must
    remain identical on every rank so scheduling stays lock-step.
    """
    if isinstance(msg, UserMsg) and (
        msg.pixel_values is not None or msg.image_inputs is not None
    ):
        return replace(
            msg, pixel_values=None, image_grid_thw=None, image_inputs=None
        ), True
    if isinstance(msg, BatchBackendMsg):
        changed = False
        data: list[BaseBackendMsg] = []
        for item in msg.data:
            stripped, item_changed = _without_vision_pixels(item)
            data.append(stripped)
            changed |= item_changed
        return (replace(msg, data=data), True) if changed else (msg, False)
    return msg, False


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        self._rank_wire_rank = tp_info.rank
        self._rank_wire_sequence = 0
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                recv = self._recv_msg_multi_rank0
            else:
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[BaseTokenizerMsg]) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        self._cpu_barrier("scheduler_sync")

    def _cpu_broadcast(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        sequence = getattr(self, "_rank_wire_sequence", 0)
        if _TRACE_COLLECTIVES:
            logger.info(
                "DSV41 CPU collective enter rank=%d seq=%d op=broadcast "
                "label=%s shape=%s dtype=%s",
                getattr(self, "_rank_wire_rank", 0),
                sequence,
                label,
                tuple(tensor.shape),
                tensor.dtype,
            )
        self.tp_cpu_group.broadcast(tensor, root=0).wait()
        if _TRACE_COLLECTIVES:
            logger.info(
                "DSV41 CPU collective exit rank=%d seq=%d op=broadcast label=%s",
                getattr(self, "_rank_wire_rank", 0),
                sequence,
                label,
            )
        self._rank_wire_sequence = sequence + 1
        return tensor

    def _cpu_barrier(self, label: str) -> None:
        sequence = getattr(self, "_rank_wire_sequence", 0)
        if _TRACE_COLLECTIVES:
            logger.info(
                "DSV41 CPU collective enter rank=%d seq=%d op=barrier label=%s",
                getattr(self, "_rank_wire_rank", 0),
                sequence,
                label,
            )
        self.tp_cpu_group.barrier().wait()
        if _TRACE_COLLECTIVES:
            logger.info(
                "DSV41 CPU collective exit rank=%d seq=%d op=barrier label=%s",
                getattr(self, "_rank_wire_rank", 0),
                sequence,
                label,
            )
        self._rank_wire_sequence = sequence + 1

    def _cpu_all_reduce_sum(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        sequence = getattr(self, "_rank_wire_sequence", 0)
        if _TRACE_COLLECTIVES:
            logger.info(
                "DSV41 CPU collective enter rank=%d seq=%d op=all_reduce_sum "
                "label=%s shape=%s dtype=%s",
                getattr(self, "_rank_wire_rank", 0),
                sequence,
                label,
                tuple(tensor.shape),
                tensor.dtype,
            )
        self.tp_cpu_group.allreduce([tensor]).wait()
        if _TRACE_COLLECTIVES:
            logger.info(
                "DSV41 CPU collective exit rank=%d seq=%d op=all_reduce_sum label=%s",
                getattr(self, "_rank_wire_rank", 0),
                sequence,
                label,
            )
        self._rank_wire_sequence = sequence + 1
        return tensor

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            if self._recv_from_tokenizer.wait(_RANK_IDLE_HEARTBEAT_MS):
                raw = self._recv_from_tokenizer.get_raw()
                msg = self._recv_from_tokenizer.decode(raw)
                self._broadcast_rank_message(self._worker_message_bytes(msg, raw))
                pending_msgs.append(msg)
            else:
                # Worker ranks are already waiting in the matching broadcast.
                # A bounded zero-length heartbeat keeps Gloo's collective
                # sequence progressing during long idle periods instead of
                # letting the workers hit the process-group timeout while rank
                # 0 blocks indefinitely on the tokenizer socket.
                self._broadcast_rank_message(None)

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self._cpu_broadcast(src_tensor, "pending_count")

        for raw in pending_raw_msgs:
            msg = self._recv_from_tokenizer.decode(raw)
            self._broadcast_rank_message(self._worker_message_bytes(msg, raw))
            pending_msgs.append(msg)
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._broadcast_rank_message(None)
            if raw:
                pending_msgs.append(self._decode_rank_message(raw))

        # ensure all ranks have the same number of raw messages
        dst_tensor = torch.tensor(-1)
        self._cpu_broadcast(dst_tensor, "pending_count")
        dst_length = int(dst_tensor.item())

        for _ in range(dst_length):
            pending_msgs.append(self._decode_rank_message(self._broadcast_rank_message(None)))
        return pending_msgs

    @staticmethod
    def _worker_message_bytes(msg: BaseBackendMsg, raw: bytes) -> bytes:
        worker_msg, changed = _without_vision_pixels(msg)
        if not changed:
            return raw
        return msgpack.packb(worker_msg.encoder(), use_bin_type=True)

    @staticmethod
    def _decode_rank_message(raw: bytes) -> BaseBackendMsg:
        return BaseBackendMsg.decoder(msgpack.unpackb(raw, raw=False))

    def _broadcast_rank_message(self, raw: bytes | None) -> bytes:
        """Reliably fan one scheduler message out over the CPU process group.

        PUB/SUB can drop a first request while a newly-started subscriber is
        still joining.  A missed request is fatal for EP because rank 0 then
        enters GPU P2P collectives that the worker never sees.  Gloo broadcast
        already orders the scheduler control plane, so carry the bounded wire
        payload on that same reliable channel.
        """

        length = torch.tensor(
            len(raw) if raw is not None else 0,
            dtype=torch.int64,
            device="cpu",
        )
        self._cpu_broadcast(length, "message_length")
        size = int(length.item())
        if not 0 <= size <= _MAX_RANK_MESSAGE_BYTES:
            raise ValueError(f"invalid scheduler rank message size {size}")
        if size == 0:
            return b""
        if raw is None:
            payload = torch.empty(size, dtype=torch.uint8, device="cpu")
        else:
            if len(raw) != size:
                raise RuntimeError("scheduler rank message length changed during broadcast")
            payload = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        self._cpu_broadcast(payload, "message_payload")
        return payload.numpy().tobytes()

    def _reply_tokenizer_rank0(self, reply: List[BaseTokenizerMsg]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[BaseTokenizerMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks
