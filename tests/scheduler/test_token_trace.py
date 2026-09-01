import json
import struct

from freetoken.scheduler.token_trace import TokenTraceRecorder, token_id_sha256


def test_token_id_sha256_uses_uint32_big_endian():
    import hashlib

    token_ids = [0, 1, 42, 0xFFFFFFFF]
    expected = hashlib.sha256(b"".join(struct.pack(">I", value) for value in token_ids))
    assert token_id_sha256(token_ids) == expected.hexdigest()


def test_recorder_writes_atomic_rank_local_trace(tmp_path):
    recorder = TokenTraceRecorder(tmp_path, rank=1, run_id="oracle repeat/1")
    recorder.record(
        uid=7,
        token_id=11,
        prompt_tokens=123,
        finished=False,
        finish_reason=None,
    )
    recorder.record(
        uid=7,
        token_id=12,
        prompt_tokens=123,
        finished=True,
        finish_reason="length",
    )

    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    assert not list(tmp_path.glob("*.tmp"))
    payload = json.loads(paths[0].read_text())
    assert payload["run_id"] == "oracle_repeat_1"
    assert payload["rank"] == 1
    assert payload["uid"] == 7
    assert payload["prompt_tokens"] == 123
    assert payload["token_ids"] == [11, 12]
    assert payload["token_id_sha256"] == token_id_sha256([11, 12])


def test_recorder_discard_drops_partial_trace(tmp_path):
    recorder = TokenTraceRecorder(tmp_path, rank=0, run_id="abort")
    recorder.record(
        uid=5,
        token_id=99,
        prompt_tokens=12,
        finished=False,
        finish_reason=None,
    )
    recorder.discard(5)
    assert recorder._tokens == {}
    assert not list(tmp_path.iterdir())
