import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_align_can_exclude_ep_sentinel_routes():
    from freetoken.kernel.triton.moe_align import moe_align_block_size

    # Expert ids 0 and 1 are local; id 2 is EP's non-local sentinel.
    topk_ids = torch.tensor(
        [[0, 2, 1], [2, 2, 0]], device="cuda", dtype=torch.int32
    )
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids,
        block_size=4,
        num_experts=2,
        include_sentinel=False,
    )
    torch.cuda.synchronize()

    num_padded = int(num_tokens_post_pad.item())
    assert num_padded == 8
    assert expert_ids[: num_padded // 4].tolist() == [0, 1]

    sentinel = topk_ids.numel()
    valid_route_ids = [
        route_id
        for route_id in sorted_ids[:num_padded].tolist()
        if route_id != sentinel
    ]
    assert sorted(valid_route_ids) == [0, 2, 5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_sum_reduce_preserves_fp32_output():
    from freetoken.kernel import moe_sum_reduce_triton

    torch.manual_seed(19)
    routes = torch.randn(
        (3, 10, 257), device="cuda", dtype=torch.bfloat16
    ).contiguous()
    expected = torch.zeros((3, 257), device="cuda", dtype=torch.float32)
    for route in range(routes.shape[1]):
        expected += routes[:, route].float()

    fp32_output = torch.empty_like(expected)
    moe_sum_reduce_triton(routes, fp32_output)
    assert fp32_output.dtype == torch.float32
    assert torch.equal(fp32_output, expected)

    bf16_output = torch.empty_like(expected, dtype=torch.bfloat16)
    moe_sum_reduce_triton(routes, bf16_output)
    assert torch.equal(bf16_output, expected.to(torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("is_prefill", [False, True])
def test_fp8_moe_can_return_unrounded_fp32_partial(is_prefill):
    from freetoken.kernel.triton.fp8_block_linear import FP8
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_decode_fp8_blockscale,
        fused_experts_fp8_blockscale,
    )

    torch.manual_seed(23)
    experts, hidden_size, intermediate_size = 4, 256, 128
    gate_up = torch.randn(
        experts, 2 * intermediate_size, hidden_size, device="cuda"
    ).to(FP8)
    gate_up_scale = (
        torch.rand(
            experts,
            2 * intermediate_size // 128,
            hidden_size // 128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        + 0.5
    )
    down = torch.randn(
        experts, hidden_size, intermediate_size, device="cuda"
    ).to(FP8)
    down_scale = (
        torch.rand(
            experts,
            hidden_size // 128,
            intermediate_size // 128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        + 0.5
    )
    hidden = torch.randn(2, hidden_size, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 2], [1, 3]], device="cuda", dtype=torch.int32)
    topk_weights = torch.rand(2, 2, device="cuda", dtype=torch.float32)

    if is_prefill:
        run = lambda output_dtype=None, return_route_outputs=False: fused_experts_fp8_blockscale(  # noqa: E731
            hidden,
            gate_up,
            gate_up_scale,
            down,
            down_scale,
            topk_weights,
            topk_ids,
            experts,
            output_dtype=output_dtype,
            return_route_outputs=return_route_outputs,
        )
    else:
        run = lambda output_dtype=None, return_route_outputs=False: fused_experts_decode_fp8_blockscale(  # noqa: E731
            hidden,
            gate_up,
            gate_up_scale,
            down,
            down_scale,
            topk_weights,
            topk_ids,
            output_dtype=output_dtype,
            return_route_outputs=return_route_outputs,
        )

    bf16_output = run()
    fp32_output = run(torch.float32)
    assert bf16_output.dtype == torch.bfloat16
    assert fp32_output.dtype == torch.float32
    assert torch.equal(fp32_output.to(torch.bfloat16), bf16_output)

    route_outputs = run(return_route_outputs=True)
    assert route_outputs.shape == (2, 2, hidden_size)
    reconstructed = torch.empty_like(bf16_output)
    from freetoken.kernel import moe_sum_reduce_triton

    moe_sum_reduce_triton(route_outputs, reconstructed)
    assert torch.equal(reconstructed, bf16_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_prefill_compact_ep_routes_match_owned_outputs():
    from freetoken.kernel.triton.fp8_block_linear import FP8
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_fp8_blockscale,
    )

    torch.manual_seed(29)
    tokens, experts, hidden_size, intermediate_size, top_k = 17, 4, 256, 128, 4
    gate_up = torch.randn(
        experts, 2 * intermediate_size, hidden_size, device="cuda"
    ).to(FP8)
    gate_up_scale = (
        torch.rand(
            experts,
            2 * intermediate_size // 128,
            hidden_size // 128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        + 0.5
    )
    down = torch.randn(
        experts, hidden_size, intermediate_size, device="cuda"
    ).to(FP8)
    down_scale = (
        torch.rand(
            experts,
            hidden_size // 128,
            intermediate_size // 128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        + 0.5
    )
    hidden = torch.randn(
        tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    route_ids = torch.randint(
        0, experts + 1, (tokens, top_k), device="cuda", dtype=torch.int32
    )
    route_ids[0] = torch.tensor([0, experts, 1, experts], device="cuda")
    owned = route_ids < experts
    route_weights = torch.rand(
        tokens, top_k, device="cuda", dtype=torch.float32
    )
    route_weights.masked_fill_(~owned, 0.0)
    safe_ids = route_ids.masked_fill(~owned, 0)

    reference = fused_experts_fp8_blockscale(
        hidden,
        gate_up,
        gate_up_scale,
        down,
        down_scale,
        route_weights,
        safe_ids,
        experts,
        return_route_outputs=True,
    )
    compact = fused_experts_fp8_blockscale(
        hidden,
        gate_up,
        gate_up_scale,
        down,
        down_scale,
        route_weights,
        route_ids,
        experts,
        return_route_outputs=True,
        skip_inactive_routes=True,
        compact_inactive_routes=True,
    )
    torch.cuda.synchronize()

    assert torch.equal(
        compact.reshape(-1, hidden_size)[owned.reshape(-1)],
        reference.reshape(-1, hidden_size)[owned.reshape(-1)],
    )
