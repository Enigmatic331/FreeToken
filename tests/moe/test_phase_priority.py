import pytest
import torch

from freetoken.moe.phase_priority import demote


@pytest.mark.parametrize("experts,start,count", [(112, 0, 32), (208, 0, 144), (208, 32, 144),
                                               (96, 32, 64), (16, 0, 16), (16, 0, 0)])
@pytest.mark.parametrize("slots", [1, 255, 256, 257, 2350])
def test_only_nonowners_lose_recency(experts, start, count, slots):
    owners = torch.arange(slots, dtype=torch.int32).mul(17).remainder(40 * experts)
    owners[::7] = -1
    usage = torch.arange(slots, dtype=torch.int64).add(500)
    old_owners, original = owners.clone(), usage.clone()
    expected = original.clone()
    for slot, owner in enumerate(owners.tolist()):
        if owner >= 0 and not start <= owner % experts < start + count:
            expected[slot] = 0
    demote(owners, usage, experts=experts, start=start, count=count)
    assert torch.equal(owners, old_owners) and torch.equal(usage, expected)
    # Idempotent, not a per-token eviction-clock change.
    demote(owners, usage, experts=experts, start=start, count=count)
    assert torch.equal(usage, expected)


@pytest.mark.parametrize("bad", ["owners_dtype", "usage_dtype", "dimensions", "noncontiguous", "range", "bool"])
def test_reject_unsupported_buffers_and_ranges(bad):
    owners = torch.zeros(8, dtype=torch.int32)
    usage = torch.ones(8, dtype=torch.int64)
    kwargs = dict(experts=16, start=0, count=8)
    if bad == "owners_dtype":
        owners = owners.long()
    elif bad == "usage_dtype":
        usage = usage.int()
    elif bad == "dimensions":
        owners, usage = owners.view(1, 8), usage.view(1, 8)
    elif bad == "noncontiguous":
        owners, usage = owners[::2], usage[::2]
    elif bad == "range":
        kwargs["count"] = 17
    else:
        kwargs["start"] = True
    with pytest.raises(ValueError):
        demote(owners, usage, **kwargs)
