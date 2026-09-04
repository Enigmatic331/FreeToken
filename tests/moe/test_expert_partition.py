from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import freetoken.distributed.info as distributed_info
import freetoken.kernel as freetoken_kernel
import freetoken.models.deepseek_v4.moe as dsv4_moe
from freetoken.distributed import DistributedInfo
from freetoken.models.deepseek_v4.moe import (
    DSV4OffloadMoELayer,
    localize_expert_routes,
)
from freetoken.moe.partition import ExpertPartition


def test_ep2_even_partition():
    partitions = [ExpertPartition(256, 2, rank) for rank in range(2)]

    assert [(p.global_offset, p.local_count) for p in partitions] == [
        (0, 128),
        (128, 128),
    ]


def test_ep3_balances_remainder_and_covers_global_experts():
    partitions = [ExpertPartition(256, 3, rank) for rank in range(3)]

    assert [(p.global_offset, p.local_count) for p in partitions] == [
        (0, 86),
        (86, 85),
        (171, 85),
    ]
    assert [expert for p in partitions for expert in p.global_range] == list(range(256))
    for global_expert in range(256):
        owner = partitions[0].owner(global_expert)
        partition = partitions[owner]
        assert partition.owns(global_expert)
        assert (
            partition.local_to_global(partition.global_to_local(global_expert))
            == global_expert
        )


def test_more_ranks_than_experts_gives_trailing_empty_partitions():
    partitions = [ExpertPartition(2, 4, rank) for rank in range(4)]

    assert [(p.global_offset, p.local_count) for p in partitions] == [
        (0, 1),
        (1, 1),
        (2, 0),
        (2, 0),
    ]
    assert partitions[0].owner(0) == 0
    assert partitions[0].owner(1) == 1


def test_explicit_heterogeneous_partition_covers_global_experts():
    counts = (104, 120, 32)
    partitions = [ExpertPartition(256, 3, rank, counts) for rank in range(3)]

    assert [(p.global_offset, p.local_count) for p in partitions] == [
        (0, 104),
        (104, 120),
        (224, 32),
    ]
    assert [expert for p in partitions for expert in p.global_range] == list(range(256))
    assert [partitions[0].owner(expert) for expert in (0, 103, 104, 223, 224, 255)] == [
        0, 0, 1, 1, 2, 2,
    ]


@pytest.mark.parametrize(
    "counts, match",
    [
        ((128, 128), "one count per rank"),
        ((104, 119, 32), "must sum"),
        ((104, 120, -1), "non-negative"),
    ],
)
def test_invalid_explicit_partition_counts(counts, match):
    with pytest.raises(ValueError, match=match):
        ExpertPartition(256, 3, 0, counts)


@pytest.mark.parametrize(
    "args",
    [(-1, 1, 0), (1, 0, 0), (1, 1, -1), (1, 1, 1)],
)
def test_invalid_partition_configuration(args):
    with pytest.raises(ValueError):
        ExpertPartition(*args)


def test_mapping_rejects_foreign_or_out_of_range_experts():
    partition = ExpertPartition(256, 3, 1)

    with pytest.raises(ValueError, match="not owned"):
        partition.global_to_local(85)
    with pytest.raises(ValueError, match="outside"):
        partition.local_to_global(85)
    with pytest.raises(ValueError, match="outside"):
        partition.owner(256)


def test_ep3_route_localization_preserves_weights_only_for_owned_experts():
    partition = ExpertPartition(256, 3, 1)  # owns global [86, 171)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    global_ids = torch.tensor([[4, 86, 142, 170, 171, 237]])

    local_weights, local_ids = localize_expert_routes(weights, global_ids, partition)

    torch.testing.assert_close(
        local_weights, torch.tensor([[0.0, 0.2, 0.3, 0.4, 0.0, 0.0]])
    )
    assert local_ids.tolist() == [[85, 0, 56, 84, 85, 85]]


def test_cache_safe_ids_deduplicate_skipped_routes_onto_a_live_expert():
    weights = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    local_ids = torch.tensor([[85, 56, 85], [85, 85, 85]])

    cache_ids = DSV4OffloadMoELayer._cache_safe_route_ids(weights, local_ids)

    assert cache_ids.tolist() == [[56, 56, 56], [0, 0, 0]]


def test_dsv4_packed_prefill_preserves_canonical_route_order(monkeypatch):
    class FakeCommunicator:
        packed = None

        def send(self, tensor, dst):
            assert dst == 0
            self.packed = tensor.clone()
            return tensor

        def recv(self, tensor, src):
            assert src == 1
            tensor.copy_(self.packed)
            return tensor

    fake = FakeCommunicator()

    def sum_routes(routes, output):
        output.copy_(routes.float().sum(dim=1).to(output.dtype))

    monkeypatch.setattr(freetoken_kernel, "moe_sum_reduce_triton", sum_routes)
    layer = object.__new__(DSV4OffloadMoELayer)
    layer.packed_prefill_root = 0
    layer._comm = fake

    full = torch.arange(48, dtype=torch.float32).view(2, 6, 4).to(torch.bfloat16)
    root_owned = torch.tensor(
        [[True, False, True, False, True, True], [False, True, False, True, True, False]]
    )
    root_routes = torch.where(root_owned.unsqueeze(-1), full, 0)
    worker_routes = torch.where(~root_owned.unsqueeze(-1), full, 0)

    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(1, 2))
    assert layer._maybe_combine_packed_prefill_routes(
        worker_routes, (~root_owned).float()
    ) is worker_routes
    torch.testing.assert_close(
        fake.packed,
        full.view(-1, 4).index_select(
            0, torch.nonzero((~root_owned).reshape(-1), as_tuple=False).flatten()
        ),
    )

    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 2))
    output = layer._maybe_combine_packed_prefill_routes(
        root_routes, root_owned.float()
    )
    torch.testing.assert_close(output, full.float().sum(dim=1).to(torch.bfloat16))


def test_dsv4_packed_prefill_posts_receive_before_combine(monkeypatch):
    class FakeStream:
        def __init__(self):
            self.waited_for = []

        def wait_stream(self, stream):
            self.waited_for.append(stream)

    class FakeCommunicator:
        def __init__(self, remote):
            self.remote = remote
            self.receives = 0

        def recv(self, tensor, src):
            assert src == 1
            self.receives += 1
            tensor.copy_(self.remote)
            return tensor

    full = torch.arange(48, dtype=torch.float32).view(2, 6, 4).to(torch.bfloat16)
    root_owned = torch.tensor(
        [[True, False, True, False, True, True], [False, True, False, True, True, False]]
    )
    remote_indices = torch.nonzero(~root_owned.reshape(-1), as_tuple=False).flatten()
    remote = full.view(-1, 4).index_select(0, remote_indices)
    root_routes = torch.where(root_owned.unsqueeze(-1), full, 0)

    layer = object.__new__(DSV4OffloadMoELayer)
    layer.packed_prefill_root = 0
    layer.hidden_size = 4
    layer._packed_prefill_pending = None
    layer._comm = FakeCommunicator(remote)

    current = FakeStream()
    receive = FakeStream()
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 2))
    monkeypatch.setattr(
        "freetoken.core.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=True)),
    )
    monkeypatch.setattr(dsv4_moe, "_EP_PACKED_PREFILL_OVERLAP", True)
    monkeypatch.setattr(dsv4_moe, "_ep_packed_recv_stream", lambda device: receive)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    def sum_routes(routes, output):
        output.copy_(routes.float().sum(dim=1).to(output.dtype))

    monkeypatch.setattr(freetoken_kernel, "moe_sum_reduce_triton", sum_routes)

    layer.prepare_packed_prefill_receive(root_owned.float(), torch.bfloat16)
    assert layer._comm.receives == 1
    assert receive.waited_for == [current]
    assert layer._packed_prefill_pending is not None

    output = layer._maybe_combine_packed_prefill_routes(
        root_routes, root_owned.float()
    )
    assert current.waited_for == [receive]
    assert layer._packed_prefill_pending is None
    torch.testing.assert_close(output, full.float().sum(dim=1).to(torch.bfloat16))


def test_offloaded_ep3_does_not_tensor_partition_expert_width(monkeypatch):
    import freetoken.layers.moe as moe_layers

    monkeypatch.setattr(moe_layers, "get_tp_info", lambda: SimpleNamespace(size=3))

    layer = moe_layers.OffloadMoELayer(
        layer_id=0,
        num_experts=86,
        top_k=6,
        hidden_size=4096,
        intermediate_size=2048,
    )

    assert layer.intermediate_size == 2048
    assert layer.tp_size == 3


def test_resident_tp3_still_requires_divisible_expert_width(monkeypatch):
    import freetoken.layers.moe as moe_layers

    monkeypatch.setattr(moe_layers, "get_tp_info", lambda: SimpleNamespace(size=3))

    with pytest.raises(AssertionError):
        moe_layers.MoELayer(
            num_experts=86,
            top_k=6,
            hidden_size=4096,
            intermediate_size=2048,
        )
