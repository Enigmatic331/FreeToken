from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _initialize(module):
    torch.manual_seed(4101)
    for name, value in module.named_parameters():
        if value.dtype == torch.float8_e8m0fnu:
            value.data.view(torch.uint8).fill_(127)
        elif value.dtype == torch.float8_e4m3fn:
            value.data.copy_(
                (torch.randn(value.shape, device=value.device) * 0.05).to(value.dtype)
            )
        elif name.endswith(".weight") and (
            "norm" in name or name.endswith(("q_weight", "k_weight"))
        ):
            value.data.fill_(1)
        elif name == "attn_sink":
            value.data.zero_()
        else:
            value.data.normal_(0, 0.03)


def test_source_layer_prefill_populates_only_source_pools_and_attends():
    from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend
    from freetoken.core import Context, get_global_ctx, set_global_ctx
    from freetoken.kvcache.dsv41_cost_model import dsv41_pool_sizes
    from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
    from freetoken.models.deepseek_v41.args import DeepseekV41Args
    from freetoken.models.deepseek_v41.attention_layout import AttentionLayout
    from freetoken.models.deepseek_v41.model import Attention, shared_attention

    args = DeepseekV41Args(
        max_batch_size=1,
        max_seq_len=32,
        vocab_size=64,
        dim=64,
        moe_inter_dim=32,
        n_layers=4,
        n_heads=2,
        n_routed_experts=4,
        n_activated_experts=2,
        q_lora_rank=64,
        head_dim=64,
        rope_head_dim=16,
        o_groups=1,
        o_lora_rank=32,
        window_size=4,
        compress_ratios=(0, 2, 2, 2),
        kv_source_layers=(1,),
        index_source_layers=(1, 2),
        index_n_heads=2,
        index_head_dim=32,
        index_topk=2,
        candidate_source_layer=1,
        candidate_topk_blocks=1,
        candidate_block_size=2,
        engram_layer_ids=(),
        engram_num_embeddings=(),
    )
    sizes = dsv41_pool_sizes(8, args, 1.0, P=4, n_win_pages=8)
    pool = DSV41PagedKVCache(
        sizes, args, torch.device("cuda"), P=4, n_scratch=2
    )
    pool.full_loc_map = torch.arange(32, device="cuda", dtype=torch.int32).view(1, -1)
    for base in range(0, 32, 4):
        pool.bind_window_pages(base, base)
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=4)
        set_global_ctx(ctx)
    ctx.kv_cache = pool
    backend = DSV41SparseAttnBackend(SimpleNamespace(dsv41_args=args))
    ctx.attn_backend = backend

    with torch.device("cuda"):
        attention = Attention(args, 1, AttentionLayout(args))
    _initialize(attention)
    attention.bind(torch.device("cuda"))
    # Two fp32 score columns cost eight bytes per query row.  A 16-byte cap
    # therefore forces this five-token prefill through 2, 2, 1 row chunks.
    attention.indexer.prefill_max_logits_bytes = 16
    indexer_chunk_rows = []
    indexer_prefill_logits = backend.indexer_prefill_logits

    def record_indexer_chunk(q, keys, weights):
        indexer_chunk_rows.append(q.shape[1])
        return indexer_prefill_logits(q, keys, weights)

    backend.indexer_prefill_logits = record_indexer_chunk
    shared_attention.reset()
    x = torch.randn(1, 5, 64, device="cuda", dtype=torch.bfloat16)
    out = attention.prefill_single(x, start_pos=0, table_idx=0)

    assert out.shape == x.shape and torch.isfinite(out).all()
    assert indexer_chunk_rows == [2, 2, 1]
    assert torch.count_nonzero(pool.cmp_pool[1][:2]) > 0
    assert torch.count_nonzero(pool.idx_pool[1][:2]) > 0
    assert all(pool.cmp_pool[layer] is None for layer in (0, 2, 3))
    assert shared_attention.topk_rows.shape == (1, 5, 2)
    assert shared_attention.candidates.mask.shape == (5, 2)

    with torch.device("cuda"):
        reindexer = Attention(args, 2, AttentionLayout(args))
    _initialize(reindexer)
    reindexer.bind(torch.device("cuda"))
    reindexer.indexer.prefill_max_logits_bytes = 16
    reindexed = reindexer.prefill_single(x, start_pos=0, table_idx=0)
    assert reindexed.shape == x.shape and torch.isfinite(reindexed).all()
    assert reindexer.plan.uses_candidates and reindexer.plan.owns_index
    assert not reindexer.plan.owns_kv
    assert indexer_chunk_rows == [2, 2, 1, 2, 2, 1]

    with torch.device("cuda"):
        consumer = Attention(args, 3, AttentionLayout(args))
    _initialize(consumer)
    consumer.bind(torch.device("cuda"))
    consumer_out = consumer.prefill_single(x, start_pos=0, table_idx=0)
    assert consumer_out.shape == x.shape and torch.isfinite(consumer_out).all()
    assert consumer.plan.kv_source == 1 and not consumer.plan.owns_kv
    assert consumer.plan.index_source == 2 and not consumer.plan.owns_index

    # The odd fifth prefix token is pending in the page ring. Position 5 closes
    # that pair, publishes compressed/index rows 2, and attends over them.
    from freetoken.attention.dsv4_sparse import DSV4AttnMetadata

    metadata = DSV4AttnMetadata(
        last_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        full_snap=pool.full_loc_map.to(torch.int64),
        window_ar=torch.arange(4, device="cuda"),
    )
    batch = SimpleNamespace(attn_metadata=metadata)
    positions = torch.tensor([5], device="cuda", dtype=torch.int64)
    rows = torch.tensor([0], device="cuda", dtype=torch.int64)
    window_ctx = (
        torch.tensor([5], device="cuda", dtype=torch.int64),
        torch.tensor([4], device="cuda", dtype=torch.int64),
        torch.tensor([[[2, 3, 4, 5]]], device="cuda", dtype=torch.int64),
    )
    x_decode = torch.randn(1, 1, 64, device="cuda", dtype=torch.bfloat16)
    shared_attention.reset()
    with ctx.forward_batch(batch):
        decoded = attention.decode_step(
            x_decode,
            positions,
            rows,
            cmp_stage_cap=5,
            window_ctx=window_ctx,
        )
    assert decoded.shape == x_decode.shape and torch.isfinite(decoded).all()
    assert torch.count_nonzero(pool.cmp_pool[1][2]) > 0
    assert torch.count_nonzero(pool.idx_pool[1][2]) > 0
