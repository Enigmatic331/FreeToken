from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv41.router import fused_sqrtsoftplus_topk
from freetoken.kernel.triton.dsv41.route_partition import (
    fused_localize_cache_safe_routes,
)
from freetoken.moe.partition import (
    ExpertPartition,
    cache_safe_route_ids,
    localize_expert_routes,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,experts", [(1, 385), (17, 385), (64, 384)])
def test_fused_sqrtsoftplus_router_matches_torch(rows: int, experts: int):
    generator = torch.Generator().manual_seed(41)
    logits = torch.randn(rows, experts, generator=generator, dtype=torch.float32)
    bias = torch.randn(experts, generator=generator, dtype=torch.float32) * 0.1
    scores = F.softplus(logits / 0.7).sqrt()
    expected_ids = (scores + bias).topk(8, dim=-1).indices
    expected_weights = scores.gather(-1, expected_ids)
    expected_weights /= expected_weights.sum(-1, keepdim=True) + 1e-20
    expected_weights *= 1.5

    weights, ids = fused_sqrtsoftplus_topk(
        logits.cuda(),
        bias.cuda(),
        topk=8,
        temperature=0.7,
        renormalize=True,
        route_scale=1.5,
    )

    torch.testing.assert_close(ids.cpu().long(), expected_ids)
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rank", [0, 1])
def test_fused_route_partition_matches_composed_reference_and_graph(rank: int):
    partition = ExpertPartition(192, world_size=2, rank=rank)
    weights = torch.tensor(
        [
            [0.7, 0.6, 0.0, 0.4, 0.3, 0.2, 0.1, 0.8, 0.9, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    ids = torch.tensor(
        [
            [0, 95, 96, 97, 191, 32, 160, 64, 128, 1],
            [191, 160, 128, 97, 96, 95, 64, 32, 1, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    local_weights, local_ids = localize_expert_routes(weights, ids, partition)
    expected_ids = cache_safe_route_ids(local_weights, local_ids)

    got_weights, got_ids = fused_localize_cache_safe_routes(
        weights,
        ids,
        global_offset=partition.global_offset,
        local_count=partition.local_count,
    )
    torch.testing.assert_close(got_weights, local_weights)
    torch.testing.assert_close(got_ids, expected_ids)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = fused_localize_cache_safe_routes(
            weights,
            ids,
            global_offset=partition.global_offset,
            local_count=partition.local_count,
        )
    weights.copy_(torch.flip(weights, dims=(1,)))
    ids.copy_(torch.flip(ids, dims=(1,)))
    local_weights, local_ids = localize_expert_routes(weights, ids, partition)
    expected_ids = cache_safe_route_ids(local_weights, local_ids)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_weights, local_weights)
    torch.testing.assert_close(graph_ids, expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_route_partition_supports_zero_owned_experts_and_graph():
    weights = torch.tensor(
        [[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.8]],
        dtype=torch.float32,
        device="cuda",
    )
    ids = torch.tensor(
        [[0, 95, 96, 97, 191, 32, 160, 64]],
        dtype=torch.int32,
        device="cuda",
    )

    got_weights, got_ids = fused_localize_cache_safe_routes(
        weights,
        ids,
        global_offset=0,
        local_count=0,
    )
    torch.testing.assert_close(got_weights, torch.zeros_like(weights))
    torch.testing.assert_close(got_ids, torch.zeros_like(ids))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = fused_localize_cache_safe_routes(
            weights,
            ids,
            global_offset=0,
            local_count=0,
        )
    weights.fill_(1.0)
    ids.copy_(torch.flip(ids, dims=(1,)))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_weights, torch.zeros_like(weights))
    torch.testing.assert_close(graph_ids, torch.zeros_like(ids))
