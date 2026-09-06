from __future__ import annotations

import torch

from freetoken.scheduler.multimodal_cache import (
    MultimodalCacheKeyRegistry,
    VisionFeatureCache,
)


def test_registry_reuses_same_image_and_isolates_different_images():
    registry = MultimodalCacheKeyRegistry()
    ids = torch.tensor([1, 99, 99, 2, 99, 99, 3], dtype=torch.int32)
    a, b = b"a" * 32, b"b" * 32

    first = registry.cache_ids(ids, 99, [a, b])
    repeated = registry.cache_ids(ids, 99, [a, b])
    swapped = registry.cache_ids(ids, 99, [b, a])

    assert torch.equal(first, repeated)
    assert first.tolist()[0::3] == ids.tolist()[0::3]
    assert first[1] < 0 and first[4] < 0 and first[1] != first[4]
    assert swapped[1] == first[4] and swapped[4] == first[1]


def test_registry_falls_back_to_no_reuse_at_its_identity_bound():
    registry = MultimodalCacheKeyRegistry(max_images=1)
    ids = torch.tensor([1, 99, 2], dtype=torch.int32)
    assert registry.cache_ids(ids, 99, [b"a" * 32]) is not None
    assert registry.cache_ids(ids, 99, [b"b" * 32]) is None


def test_vision_feature_cache_is_byte_bounded_lru():
    cache = VisionFeatureCache(max_bytes=32)
    a = torch.arange(4, dtype=torch.float32)
    b = torch.arange(6, dtype=torch.float32)
    assert cache.put((b"a" * 32,), a)
    assert cache.get((b"a" * 32,)) is not None
    assert cache.put((b"b" * 32,), b)
    assert cache.get((b"a" * 32,)) is None
    assert torch.equal(cache.get((b"b" * 32,)), b)
    assert cache.current_bytes == 24
