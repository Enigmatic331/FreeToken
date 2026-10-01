from __future__ import annotations

import os

import pytest

from freetoken.gpu_select import rank_local_cuda_plan
from freetoken.server.launch import _normalize_vision_device_spec


TEXT = ("GPU-text0", "GPU-text1", "GPU-text2")
VISION = "GPU-vision"
DSPARK = "GPU-dspark"


def test_authority_sees_text_then_vision_with_local_remap():
    visible, vision_ordinal = rank_local_cuda_plan(
        rank=0,
        text_uuids=TEXT,
        vision_uuid=VISION,
        vision_owner_rank=0,
    )
    assert visible == ("GPU-text0", "GPU-vision")
    assert vision_ordinal == 1


@pytest.mark.parametrize("rank", [1, 2])
def test_worker_sees_only_its_text_gpu(rank):
    visible, vision_ordinal = rank_local_cuda_plan(
        rank=rank,
        text_uuids=TEXT,
        vision_uuid=VISION,
        vision_owner_rank=0,
    )
    assert visible == (TEXT[rank],)
    assert vision_ordinal is None


def test_shared_text_and_vision_gpu_is_not_duplicated():
    visible, vision_ordinal = rank_local_cuda_plan(
        rank=0,
        text_uuids=TEXT,
        vision_uuid=TEXT[0],
        vision_owner_rank=0,
    )
    assert visible == (TEXT[0],)
    assert vision_ordinal == 0


def test_authority_alone_sees_the_dspark_auxiliary_gpu():
    visible, _ = rank_local_cuda_plan(
        rank=0,
        text_uuids=TEXT,
        vision_uuid=VISION,
        vision_owner_rank=0,
        auxiliary_uuids=(DSPARK,),
    )
    assert visible == (TEXT[0], VISION, DSPARK)

    for rank in (1, 2):
        worker_visible, _ = rank_local_cuda_plan(
            rank=rank,
            text_uuids=TEXT,
            vision_uuid=VISION,
            vision_owner_rank=0,
            auxiliary_uuids=(DSPARK,),
        )
        assert worker_visible == (TEXT[rank],)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, (TEXT[0], TEXT[1], TEXT[2], VISION)),
        (1, (TEXT[1], TEXT[0], TEXT[2])),
        (2, (TEXT[2], TEXT[0], TEXT[1])),
    ],
)
def test_text_peer_visibility_preserves_assigned_first_and_vision_owner_only(
    rank, expected
):
    visible, vision_ordinal = rank_local_cuda_plan(
        rank=rank,
        text_uuids=TEXT,
        vision_uuid=VISION,
        vision_owner_rank=0,
        text_peer_visibility=True,
    )
    assert visible == expected
    assert vision_ordinal == (3 if rank == 0 else None)


def test_plan_is_side_effect_free_for_parent_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-mask")
    rank_local_cuda_plan(
        rank=0,
        text_uuids=TEXT,
        vision_uuid=VISION,
        vision_owner_rank=0,
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-mask"


def test_plan_rejects_unresolved_or_out_of_range_devices():
    with pytest.raises(ValueError, match="resolved GPU UUIDs"):
        rank_local_cuda_plan(
            rank=0,
            text_uuids=("0", "1"),
            vision_uuid=None,
            vision_owner_rank=0,
        )
    with pytest.raises(ValueError, match="outside world size"):
        rank_local_cuda_plan(
            rank=0,
            text_uuids=TEXT,
            vision_uuid=VISION,
            vision_owner_rank=3,
        )


@pytest.mark.parametrize(
    ("spec", "normalized"),
    [("3", "3"), ("cuda:3", "3"), ("GPU-abcd", "GPU-abcd")],
)
def test_vision_device_normalization(spec, normalized):
    assert _normalize_vision_device_spec(spec) == normalized
