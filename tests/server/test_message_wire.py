"""Encoder/decoder round-trips for the ZMQ control messages (no GPU).

Every message that crosses api -> tokenizer -> scheduler -> tokenizer -> api must survive the
wire with its fields intact; these pin the ones carrying state a later consumer reads back
(rebuild control, prompt admission, per-reply token deltas and KV usage).
"""

from __future__ import annotations

import torch

from freetoken.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
    UserReply,
    UserMsg,
)
from freetoken.core import SamplingParams
from freetoken.scheduler.io import _without_vision_pixels


def test_cache_rebuild_msg_roundtrip():
    msg = CacheRebuildMsg(request_id="abc", moe_cache_size=8, num_pages=1024, mode="if_idle")
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("abc", 8, 1024, "if_idle")


def test_cache_rebuild_backend_msg_roundtrip():
    msg = CacheRebuildBackendMsg(request_id="r1", moe_cache_size=None, num_pages=256, mode="drain")
    out = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(out, CacheRebuildBackendMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("r1", None, 256, "drain")


def test_cache_rebuild_result_msg_roundtrip():
    msg = CacheRebuildResultMsg(request_id="r2", status="ok", moe_cache_size=16, num_pages=512)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildResultMsg)
    assert (out.request_id, out.status, out.moe_cache_size, out.num_pages, out.error) == (
        "r2", "ok", 16, 512, None,
    )


def test_cache_rebuild_reply_roundtrip():
    msg = CacheRebuildReply(request_id="r3", status="failed", error="boom")
    out = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))
    assert isinstance(out, CacheRebuildReply)
    assert (out.request_id, out.status, out.error) == ("r3", "failed", "boom")


def test_prompt_admitted_msg_roundtrip():
    msg = PromptAdmittedMsg(uid=42, prompt_tokens=1234, cached_tokens=500)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, PromptAdmittedMsg)
    assert (out.uid, out.prompt_tokens, out.cached_tokens) == (42, 1234, 500)


def test_user_reply_token_deltas_round_trip():
    msg = UserReply(
        uid=7,
        incremental_output="hello",
        finished=False,
        prompt_tokens_delta=11,
        completion_tokens_delta=3,
        cached_tokens=4,
        kv_used_pages=40,
        kv_total_pages=512,
        gpu_mem_bytes=64 * (1 << 30),
    )

    decoded = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))

    assert isinstance(decoded, UserReply)
    assert decoded.uid == 7
    assert decoded.incremental_output == "hello"
    assert decoded.finished is False
    assert decoded.prompt_tokens_delta == 11
    assert decoded.completion_tokens_delta == 3
    assert decoded.cached_tokens == 4
    assert decoded.kv_used_pages == 40
    assert decoded.kv_total_pages == 512
    assert decoded.gpu_mem_bytes == 64 * (1 << 30)


def test_detokenize_msg_carries_kv_usage_round_trip():
    msg = DetokenizeMsg(
        uid=3, next_token=42, finished=True,
        kv_used_pages=10, kv_total_pages=256, gpu_mem_bytes=1 << 30,
        mamba_used_slots=7, mamba_total_slots=64,
        swa_used_tokens=8448, swa_total_tokens=76800,
    )
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(decoded, DetokenizeMsg)
    assert (decoded.kv_used_pages, decoded.kv_total_pages, decoded.gpu_mem_bytes) == (10, 256, 1 << 30)
    assert (decoded.mamba_used_slots, decoded.mamba_total_slots) == (7, 64)
    assert (decoded.swa_used_tokens, decoded.swa_total_tokens) == (8448, 76800)


def test_client_dicts_with_the_wire_tag_key_survive_intact():
    """Tool JSON Schemas and chat_template_kwargs are free-form client data. A field literally
    named ``__type__`` (a common discriminator) must not be read back as a serialized class --
    that used to kill the tokenizer worker on an unknown/incompatible name."""
    hostile = [
        {"__type__": "AbortMsg"},                                    # a real class name
        {"__type__": "NoSuchClassAnywhere"},                         # an unknown one
        {"type": "object", "properties": {"__type__": {"type": "string"}}},
        {"__raw_dict__": {"a": 1}},                                  # collides with the escape key
        {"deep": {"__type__": "AbortMsg", "l": [{"__type__": "x"}]}},
    ]
    for payload in hostile:
        msg = TokenizeMsg(
            uid=1, text="hi", sampling_params=SamplingParams(),
            chat_template_kwargs=payload,
            tools=[{"type": "function", "function": {"name": "f", "parameters": payload}}],
        )
        out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
        assert isinstance(out, TokenizeMsg)
        assert out.chat_template_kwargs == payload
        assert out.tools[0]["function"]["parameters"] == payload


def test_multimodal_tensors_round_trip_with_shape_and_bfloat16():
    pixels = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    rope = torch.arange(18, dtype=torch.int32).reshape(6, 3)
    soft_tokens = torch.arange(20, dtype=torch.float32).to(torch.bfloat16).reshape(4, 5)
    msg = UserMsg(
        uid=9,
        input_ids=torch.arange(6, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=2),
        mm_embeds=soft_tokens,
        pixel_values=pixels,
        image_grid_thw=torch.tensor([[1, 2, 3]], dtype=torch.int32),
        rope_positions=rope,
        mrope_position_delta=-3,
        is_multimodal=True,
    )

    out = BaseBackendMsg.decoder(msg.encoder())

    assert isinstance(out, UserMsg)
    assert out.is_multimodal and out.mrope_position_delta == -3
    for actual, expected in (
        (out.input_ids, msg.input_ids),
        (out.mm_embeds, soft_tokens),
        (out.pixel_values, pixels),
        (out.image_grid_thw, msg.image_grid_thw),
        (out.rope_positions, rope),
    ):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)


def test_ep_worker_copy_drops_pixels_but_keeps_mrope_metadata():
    msg = UserMsg(
        uid=10,
        input_ids=torch.arange(4, dtype=torch.int32),
        sampling_params=SamplingParams(),
        pixel_values=torch.ones(2, 3),
        image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.int32),
        rope_positions=torch.arange(12, dtype=torch.int32).reshape(4, 3),
        mrope_position_delta=-1,
        is_multimodal=True,
    )

    worker, changed = _without_vision_pixels(msg)

    assert changed and isinstance(worker, UserMsg)
    assert worker.pixel_values is None and worker.image_grid_thw is None
    assert torch.equal(worker.input_ids, msg.input_ids)
    assert torch.equal(worker.rope_positions, msg.rope_positions)
    assert worker.is_multimodal and worker.mrope_position_delta == -1
