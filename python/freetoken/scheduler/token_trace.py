from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from pathlib import Path


def token_id_sha256(token_ids: list[int]) -> str:
    """Hash token ids using a portable, unambiguous unsigned big-endian encoding."""
    digest = hashlib.sha256()
    for token_id in token_ids:
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise ValueError(f"token id is outside uint32: {token_id}")
        digest.update(struct.pack(">I", token_id))
    return digest.hexdigest()


class TokenTraceRecorder:
    """Opt-in correctness trace that never records prompts or decoded text.

    Set ``FREETOKEN_TOKEN_TRACE_DIR`` to enable it. Each scheduler rank writes a
    separate atomic JSON file when a request finishes. The recorder is kept off
    for performance runs so filesystem activity cannot affect measurements.
    """

    def __init__(self, output_dir: Path, rank: int, run_id: str) -> None:
        self.output_dir = output_dir
        self.rank = rank
        self.run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id) or "trace"
        self.pid = os.getpid()
        self._tokens: dict[int, list[int]] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, rank: int) -> TokenTraceRecorder | None:
        raw_dir = os.environ.get("FREETOKEN_TOKEN_TRACE_DIR")
        if not raw_dir:
            return None
        run_id = os.environ.get("FREETOKEN_TOKEN_TRACE_RUN_ID", "trace")
        return cls(Path(raw_dir), rank, run_id)

    def record(
        self,
        *,
        uid: int,
        token_id: int,
        prompt_tokens: int,
        finished: bool,
        finish_reason: str | None,
    ) -> None:
        tokens = self._tokens.setdefault(uid, [])
        tokens.append(token_id)
        if not finished:
            return

        payload = {
            "schema": "freetoken-token-trace-v1",
            "run_id": self.run_id,
            "rank": self.rank,
            "pid": self.pid,
            "uid": uid,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokens),
            "finish_reason": finish_reason,
            "hash_encoding": "uint32-big-endian",
            "token_id_sha256": token_id_sha256(tokens),
            "token_ids": tokens,
        }
        name = f"{self.run_id}-rank{self.rank}-pid{self.pid}-uid{uid}.json"
        destination = self.output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        del self._tokens[uid]

    def discard(self, uid: int) -> None:
        self._tokens.pop(uid, None)

