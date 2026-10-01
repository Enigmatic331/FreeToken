from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.scheduler import Scheduler


def _decode_req(*, output_len: int = 16) -> Req:
    return Req(
        input_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
        table_idx=0,
        cached_len=2,
        output_len=output_len,
        uid=7,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=None,
    )


def test_scheduler_expands_eligible_decode_to_anchor_plus_proposals():
    req = _decode_req()
    batch = Batch([req], phase="decode")
    release = object()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._speculative = SimpleNamespace(block_size=5, noise_token_id=128799)
    scheduler.engine = SimpleNamespace(should_speculate=lambda _req: True)
    scheduler.cache_manager = SimpleNamespace(release_speculative_tail=release)
    scheduler.token_pool = torch.empty(0, dtype=torch.int32)

    Scheduler._maybe_make_speculative(scheduler, batch)

    assert batch.phase == "prefill"
    assert batch.speculative and batch.spec_block == 5
    assert batch.is_prefill and not batch.is_moe_prefill
    assert req.input_ids.tolist() == [10, 11, 12, *([128799] * 5)]
    assert req.device_len == 8
    assert batch.release_tail is release


def test_scheduler_keeps_ordinary_decode_near_output_limit():
    req = _decode_req(output_len=5)
    batch = Batch([req], phase="decode")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._speculative = SimpleNamespace(block_size=5, noise_token_id=128799)
    scheduler.engine = SimpleNamespace(should_speculate=lambda _req: True)
    scheduler.token_pool = torch.empty(0, dtype=torch.int32)

    Scheduler._maybe_make_speculative(scheduler, batch)

    assert batch.phase == "decode"
    assert not batch.speculative
    assert req.input_ids.tolist() == [10, 11, 12]


def test_scheduler_uses_engine_selected_verification_length():
    req = _decode_req()
    batch = Batch([req], phase="decode")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._speculative = SimpleNamespace(block_size=5, noise_token_id=128799)
    scheduler.engine = SimpleNamespace(
        should_speculate=lambda _req: True,
        speculation_length=lambda _req, maximum: maximum - 3,
    )
    scheduler.cache_manager = SimpleNamespace(release_speculative_tail=object())
    scheduler.token_pool = torch.empty(0, dtype=torch.int32)

    Scheduler._maybe_make_speculative(scheduler, batch)

    assert batch.speculative and batch.spec_block == 2
    assert req.input_ids.tolist() == [10, 11, 12, 128799, 128799]


def test_ordinary_prefill_keeps_bulk_moe_movement():
    batch = Batch([_decode_req()], phase="prefill")

    assert batch.is_prefill and batch.is_moe_prefill


def test_release_speculative_tail_releases_uncomputed_bonus_page():
    manager = CacheManager.__new__(CacheManager)
    manager.page_size = 128
    manager.swa_paged = False
    manager.free_slots = torch.empty(0, dtype=torch.int32)
    manager.page_table = torch.arange(384, dtype=torch.int32).view(1, -1)
    req = SimpleNamespace(table_idx=0, device_len=260)

    # The accepted KV ends at 128. The sampled bonus logically occupies
    # position 128 but has not been forwarded, so it owns no KV yet. Both
    # speculative-only pages return; a later decode can reacquire page one.
    manager.release_speculative_tail(req, committed_len=128)

    assert manager.free_slots.tolist() == [128, 256]
