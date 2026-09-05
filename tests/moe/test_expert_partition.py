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


def test_dsv4_tiled_prefill_preserves_rows_and_single_layer_materialization(
    monkeypatch,
):
    class FakeCache:
        prefill_overlap = False
        collect_stats = False

        def __init__(self):
            self.materialized = []
            self.copies = 0
            self.profile = []

        def materialize_layer(self, layer_id):
            self.materialized.append(layer_id)

        def copy_missing(self):
            self.copies += 1

        def bank_views(self, num_experts):
            assert num_experts == 128
            return ("views",)

        def alphas_for_layer(self, layer_id):
            assert layer_id == 7
            return None

        def profile_begin(self, name, layer_id):
            self.profile.append(("begin", name, layer_id))

        def profile_end(self, name, layer_id):
            self.profile.append(("end", name, layer_id))

    layer = object.__new__(DSV4OffloadMoELayer)
    layer.layer_id = 7
    layer.num_experts = 128
    layer.packed_prefill_root = 0
    layer._packed_prefill_pending = None
    layer.offload_cache = FakeCache()

    receive_sizes = []

    def prepare(weights, route_dtype):
        assert route_dtype == torch.bfloat16
        receive_sizes.append(weights.shape[0])
        layer._packed_prefill_pending = object()

    def expert_gemm(cache, hidden, weights, ids, **kwargs):
        assert cache is layer.offload_cache
        assert kwargs["views"] == ("views",)
        assert kwargs["n"] == 128
        assert kwargs["is_prefill"] is True
        return hidden.unsqueeze(1).expand(-1, 6, -1).contiguous()

    def combine(routes, weights):
        assert routes.shape[:2] == weights.shape
        layer._packed_prefill_pending = None
        return routes.float().sum(dim=1).to(routes.dtype)

    layer.prepare_packed_prefill_receive = prepare
    layer._expert_gemm = expert_gemm
    layer._maybe_combine_packed_prefill_routes = combine
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 2))

    hidden = torch.arange(5 * 4, dtype=torch.float32).view(5, 4).to(torch.bfloat16)
    weights = torch.ones((5, 6), dtype=torch.float32)
    ids = torch.zeros((5, 6), dtype=torch.int32)
    output = layer._prefill_routed_tiled(hidden, weights, ids, tile_tokens=2)

    torch.testing.assert_close(output, hidden * 6)
    assert receive_sizes == [2, 2, 1]
    assert layer.offload_cache.materialized == [7]
    assert layer.offload_cache.copies == 1
    assert layer.offload_cache.profile == [
        ("begin", "prefill_copy", 7),
        ("end", "prefill_copy", 7),
        ("begin", "prefill_expert", 7),
        ("end", "prefill_expert", 7),
    ]


def test_dsv4_prefill_receive_bounds_the_first_route_tile(monkeypatch):
    class FakeStream:
        def wait_stream(self, stream):
            self.waited_for = stream

    class FakeCommunicator:
        def __init__(self):
            self.received_shapes = []

        def recv(self, tensor, src):
            assert src == 1
            self.received_shapes.append(tuple(tensor.shape))
            tensor.zero_()
            return tensor

    layer = object.__new__(DSV4OffloadMoELayer)
    layer.packed_prefill_root = 0
    layer.hidden_size = 4
    layer._packed_prefill_pending = None
    layer._comm = FakeCommunicator()
    root_owned = torch.tensor(
        [
            [True, False, True, False, True, True],
            [False, True, False, True, True, False],
            [False, False, False, True, True, True],
            [True, True, False, False, False, True],
            [True, False, True, False, True, False],
        ]
    )

    current = FakeStream()
    receive = FakeStream()
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 2))
    monkeypatch.setattr(
        "freetoken.core.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=True)),
    )
    monkeypatch.setattr(dsv4_moe, "_EP_PACKED_PREFILL_OVERLAP", True)
    monkeypatch.setattr(dsv4_moe, "_EP_PREFILL_ROUTE_TILE_TOKENS", 2)
    monkeypatch.setattr(dsv4_moe, "_ep_packed_recv_stream", lambda device: receive)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    layer.prepare_packed_prefill_receive(root_owned.float(), torch.bfloat16)

    first_tile_remote = int((~root_owned[:2]).sum().item())
    remote_indices, packed, pending_stream = layer._packed_prefill_pending
    assert layer._comm.received_shapes == [(first_tile_remote, 4)]
    assert remote_indices.max().item() < 2 * root_owned.shape[1]
    assert tuple(packed.shape) == (first_tile_remote, 4)
    assert pending_stream is receive


def test_dsv4_worker_async_send_preserves_owned_route_order(monkeypatch):
    class FakeStream:
        def __init__(self):
            self.waited_for = []

        def wait_stream(self, stream):
            self.waited_for.append(stream)

    class FakeCommunicator:
        def __init__(self):
            self.sent = []

        def send(self, tensor, dst):
            assert dst == 0
            self.sent.append(tensor.clone())
            return tensor

    layer = object.__new__(DSV4OffloadMoELayer)
    layer.packed_prefill_root = 0
    layer._comm = FakeCommunicator()
    routes = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4).to(
        torch.bfloat16
    )
    weights = torch.tensor(
        [[0.1, 0.0, 0.2, 0.0, 0.3, 0.4], [0.0, 0.5, 0.0, 0.6, 0.7, 0.0]]
    )

    current = FakeStream()
    send = FakeStream()
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(1, 2))
    monkeypatch.setattr(dsv4_moe, "_ep_packed_send_stream", lambda device: send)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    returned = layer._send_packed_prefill_routes_async(routes, weights)

    expected = routes.view(-1, routes.shape[-1])[weights.ne(0).reshape(-1)]
    assert returned is send
    assert send.waited_for == [current]
    assert len(layer._comm.sent) == 1
    torch.testing.assert_close(layer._comm.sent[0], expected)


def test_dsv4_tiled_worker_queues_every_send_then_joins_once(monkeypatch):
    class FakeCache:
        prefill_overlap = False
        collect_stats = False

        def materialize_layer(self, layer_id):
            self.materialized = layer_id

        def copy_missing(self):
            pass

        def bank_views(self, num_experts):
            return (num_experts,)

        def alphas_for_layer(self, layer_id):
            return None

        def profile_begin(self, name, layer_id):
            pass

        def profile_end(self, name, layer_id):
            pass

    class FakeStream:
        def __init__(self):
            self.waited_for = []

        def wait_stream(self, stream):
            self.waited_for.append(stream)

    layer = object.__new__(DSV4OffloadMoELayer)
    layer.layer_id = 9
    layer.num_experts = 128
    layer.packed_prefill_root = 0
    layer._packed_prefill_pending = None
    layer.offload_cache = FakeCache()
    send = FakeStream()
    current = FakeStream()
    send_sizes = []

    def expert_gemm(cache, hidden, weights, ids, **kwargs):
        return hidden.unsqueeze(1).expand(-1, 6, -1).contiguous()

    def send_async(routes, weights):
        assert routes.shape[:2] == weights.shape
        send_sizes.append(routes.shape[0])
        return send

    layer._expert_gemm = expert_gemm
    layer._send_packed_prefill_routes_async = send_async
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(1, 2))
    monkeypatch.setattr(dsv4_moe, "_EP_PACKED_ASYNC_SEND", True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current)

    hidden = torch.arange(5 * 4, dtype=torch.float32).view(5, 4).to(torch.bfloat16)
    weights = torch.ones((5, 6), dtype=torch.float32)
    ids = torch.zeros((5, 6), dtype=torch.int32)
    last = layer._prefill_routed_tiled(hidden, weights, ids, tile_tokens=2)

    assert send_sizes == [2, 2, 1]
    assert current.waited_for == [send]
    assert layer.offload_cache.materialized == 9
    assert tuple(last.shape) == (1, 6, 4)


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
