from __future__ import annotations

import torch

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.compress import Compressor
from freetoken.models.deepseek_v41.engram_runtime import EngramCoordinator
from freetoken.models.deepseek_v41.execution import DeepseekV41ExecutionPlan


class FakeCommunicator:
    def __init__(self, broadcast_value=None) -> None:
        self.broadcast_value = broadcast_value
        self.broadcasts = []

    def broadcast(self, tensor, source):
        self.broadcasts.append((tuple(tensor.shape), source))
        return tensor if self.broadcast_value is None else self.broadcast_value


class FakeTable:
    def __init__(self) -> None:
        self.lookups = []

    def lookup(self, row_ids, *, reduce, communicator):
        self.lookups.append((row_ids.clone(), reduce, communicator))
        return torch.zeros(*row_ids.shape, 8)


class FakePagedCompressorBackend:
    def __init__(self, head_dim):
        self.block = torch.cat(
            [torch.zeros(2, head_dim), torch.full((2, head_dim), -torch.inf)], -1
        )

    def compress_rows_of(self, table_idx, starts, ratio):
        assert table_idx == 3
        return starts // ratio

    def write_carry(self, layer_id, tier, window_slot, ratio, block):
        assert (layer_id, tier, ratio) == (1, "attn", 2)
        self.block.copy_(block)

    def read_carry_blocks(self, layer_id, tier, window_slots, ratio):
        return self.block.unsqueeze(0).expand(window_slots.shape[0], -1, -1).clone()

    def write_carry_blocks(self, layer_id, tier, window_slots, ratio, blocks):
        self.block.copy_(blocks[0])

    def decode_compress_rows(
        self, rows, positions, ratio, layer_id, tier, completed
    ):
        return positions // ratio


def test_engram_authority_and_worker_enter_identical_collective_sequence():
    row_ids = torch.arange(24).view(2, 12)
    authority_comm, authority_table = FakeCommunicator(), FakeTable()
    authority = EngramCoordinator(
        {1: authority_table},
        execution=DeepseekV41ExecutionPlan(0, 2, backbone_rank=0),
        communicator=authority_comm,
    )
    result = authority.authority_lookup(1, row_ids)
    assert result.shape == (2, 12, 8)
    assert authority_comm.broadcasts == [((2, 12), 0)]
    assert authority_table.lookups[0][1]

    worker_comm, worker_table = FakeCommunicator(row_ids), FakeTable()
    worker = EngramCoordinator(
        {1: worker_table},
        execution=DeepseekV41ExecutionPlan(1, 2, backbone_rank=0),
        communicator=worker_comm,
    )
    worker.worker_lookup(
        1, num_tokens=2, hashes_per_token=12, device=torch.device("cpu")
    )
    assert worker_comm.broadcasts == [((2, 12), 0)]
    torch.testing.assert_close(worker_table.lookups[0][0], row_ids)
    assert worker_table.lookups[0][1]


def test_ratio2_prefill_matches_incremental_decode_groups():
    args = DeepseekV41Args(
        max_batch_size=1,
        dim=32,
        head_dim=32,
        rope_head_dim=16,
        n_layers=3,
        compress_ratios=(0, 2, 1),
        engram_layer_ids=(),
        engram_num_embeddings=(),
    )
    torch.manual_seed(41)
    prefill = Compressor(args, 1)
    prefill.wkv.weight.data.normal_(0, 0.1)
    prefill.wgate.weight.data.normal_(0, 0.1)
    decode = Compressor(args, 1)
    decode.load_state_dict(prefill.state_dict())
    hidden = torch.randn(1, 5, 32, dtype=torch.bfloat16)

    expected = prefill(hidden, 0)
    actual = []
    for position in range(hidden.shape[1]):
        value = decode(hidden[:, position : position + 1], position)
        if value is not None:
            actual.append(value)
    actual = torch.cat(actual, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_ratio1_compressor_returns_one_latent_per_token():
    args = DeepseekV41Args(
        dim=32,
        head_dim=32,
        rope_head_dim=16,
        n_layers=2,
        compress_ratios=(0, 1),
        engram_layer_ids=(),
        engram_num_embeddings=(),
    )
    compressor = Compressor(args, 1)
    hidden = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    assert compressor(hidden, 0).shape == (2, 7, 32)


def test_ratio2_paged_prefill_carry_resumes_bit_equivalent_decode_pair():
    args = DeepseekV41Args(
        max_batch_size=1,
        dim=32,
        head_dim=32,
        rope_head_dim=16,
        n_layers=2,
        compress_ratios=(0, 2),
        engram_layer_ids=(),
        engram_num_embeddings=(),
    )
    torch.manual_seed(42)
    reference = Compressor(args, 1)
    reference.wkv.weight.data.normal_(0, 0.1)
    reference.wgate.weight.data.normal_(0, 0.1)
    paged = Compressor(args, 1)
    paged.load_state_dict(reference.state_dict())
    hidden = torch.randn(1, 6, 32, dtype=torch.bfloat16)
    expected = reference(hidden, 0)

    backend = FakePagedCompressorBackend(args.head_dim)
    prefix, prefix_rows = paged.prefill_paged(
        hidden[:, :5],
        0,
        torch.arange(5),
        layer_id=1,
        table_idx=3,
        backend=backend,
    )
    tail, tail_rows, completed = paged.decode_paged(
        hidden[:, 5:6],
        torch.tensor([5]),
        torch.tensor([5]),
        torch.tensor([0]),
        layer_id=1,
        backend=backend,
    )
    assert prefix_rows.tolist() == [0, 1]
    assert tail_rows.tolist() == [2]
    assert completed.tolist() == [True]
    torch.testing.assert_close(
        torch.cat([prefix, tail], 1), expected, rtol=2e-2, atol=2e-2
    )
