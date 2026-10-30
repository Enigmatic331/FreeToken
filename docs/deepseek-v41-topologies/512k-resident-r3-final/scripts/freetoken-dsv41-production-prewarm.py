#!/usr/bin/env python3
"""Idempotent per-instance sampler prewarm and readiness certificate."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:19291"
MODEL = "DeepSeek-V4.1-Flash-1017"
ROOT = "/home/enigmatic331/models/DeepSeek-V4.1-Flash-1017"
ORBIT_PROMPT = "<｜begin▁of▁sentence｜><｜User｜>Reply with exactly ORBIT.<｜Assistant｜></think>"
VISION_FIXTURE = Path("/usr/local/share/freetoken-dsv41/ocr-7429.png")


def get(path: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
        return json.load(response)


def post(path: str, body: dict, timeout: float = 900) -> dict:
    request = urllib.request.Request(
        BASE + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(deadline_s: float) -> tuple[dict, dict]:
    deadline = time.monotonic() + deadline_s
    last: object = None
    while time.monotonic() < deadline:
        try:
            health = get("/health")
            models = get("/v1/models")
            rows = [row for row in models.get("data", []) if row.get("id") == MODEL]
            if (
                health.get("status") == "ok"
                and health.get("model") == MODEL
                and len(rows) == 1
                and rows[0].get("root") == ROOT
                and int(rows[0].get("max_model_len", 0)) == 524288
            ):
                return health, models
            last = {"health": health, "models": models}
        except Exception as exc:  # loading or not listening yet
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise RuntimeError(f"model did not become exactly ready: {last}")


def wait_idle(deadline_s: float = 90, *, allow_uninitialized: bool = False) -> dict:
    deadline = time.monotonic() + deadline_s
    last: dict = {}
    while time.monotonic() < deadline:
        last = get("/v1/stats")
        kv_value = last.get("kv")
        swa_value = last.get("swa")
        kv = kv_value or {}
        swa = swa_value or {}
        requests = last.get("requests", {})
        if (
            allow_uninitialized
            and kv_value is None
            and swa_value is None
            and int(requests.get("active", -1)) == 0
            and int(requests.get("completed", -1)) == 0
        ):
            return last
        if (
            int(requests.get("active", -1)) == 0
            and int(kv.get("used_pages", -1)) == 0
            and int(swa.get("used_pages", -1)) == 0
        ):
            return last
        time.sleep(0.25)
    raise RuntimeError(f"cache cleanup did not reach zero: {last}")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=1800)
    args = parser.parse_args()

    started = time.time()
    health, models = wait_ready(args.wait_seconds)
    instance = health["instance_id"]
    prior = None
    try:
        prior = json.loads(args.receipt.read_text())
    except (OSError, ValueError, TypeError):
        pass
    if prior and prior.get("verdict") == "PASS" and prior.get("instance_id") == instance:
        wait_idle()
        print(json.dumps({"verdict": "ALREADY_WARM", "instance_id": instance}))
        return 0

    # A freshly ready engine reports KV/SWA as null until its first request.
    # Accept that one bootstrap-only state when no request has ever run; every
    # post-request gate below still requires explicit numeric zero accounting.
    before = wait_idle(allow_uninitialized=True)
    sampled = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply briefly with OK."}],
            "max_tokens": 8,
            "temperature": 0.8,
            "top_p": 0.95,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=900,
    )
    sampled_choice = sampled.get("choices", [{}])[0]
    sampled_text = (sampled_choice.get("message") or {}).get("content") or ""
    if not sampled_text.strip() or not sampled.get("usage"):
        raise RuntimeError(f"stochastic sampler warmup failed: {sampled}")

    orbit = post(
        "/v1/completions",
        {"model": MODEL, "prompt": ORBIT_PROMPT, "max_tokens": 8,
         "temperature": 0, "top_p": 1, "stream": False},
    )
    orbit_choice = orbit.get("choices", [{}])[0]
    if orbit_choice.get("text") != "ORBIT" or orbit_choice.get("finish_reason") != "stop":
        raise RuntimeError(f"exact ORBIT failed after sampler prewarm: {orbit}")

    omitted = post(
        "/v1/completions",
        {"model": MODEL, "prompt": ORBIT_PROMPT, "temperature": 0,
         "top_p": 1, "stream": False},
    )
    omitted_choice = omitted.get("choices", [{}])[0]
    if omitted_choice.get("text") != "ORBIT" or omitted_choice.get("finish_reason") != "stop":
        raise RuntimeError(f"omitted-output-token request failed: {omitted}")

    fixture = base64.b64encode(VISION_FIXTURE.read_bytes()).decode()
    vision = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + fixture,
                    }},
                    {"type": "text", "text":
                     "Read the large red number in this image. Reply with only the four digits."},
                ],
            }],
            "max_tokens": 16,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    vision_choice = vision.get("choices", [{}])[0]
    vision_message = vision_choice.get("message") or {}
    if (vision_message.get("content") != "7429"
            or vision_choice.get("finish_reason") != "stop"):
        raise RuntimeError(f"native vision OCR prewarm failed: {vision}")

    after = wait_idle()
    if get("/health").get("instance_id") != instance:
        raise RuntimeError("service instance changed during prewarm")
    result = {
        "schema": "dsv41-production-prewarm-v2",
        "verdict": "PASS",
        "captured_unix": time.time(),
        "elapsed_s": time.time() - started,
        "instance_id": instance,
        "health": health,
        "models": models,
        "before_stats": before,
        "sampled": {
            "text": sampled_text,
            "finish_reason": sampled_choice.get("finish_reason"),
            "usage": sampled.get("usage"),
        },
        "orbit": {
            "text": orbit_choice.get("text"),
            "finish_reason": orbit_choice.get("finish_reason"),
            "usage": orbit.get("usage"),
        },
        "omitted_max_tokens": {
            "text": omitted_choice.get("text"),
            "finish_reason": omitted_choice.get("finish_reason"),
            "usage": omitted.get("usage"),
        },
        "native_vision": {
            "text": vision_message.get("content"),
            "finish_reason": vision_choice.get("finish_reason"),
            "usage": vision.get("usage"),
            "fixture": str(VISION_FIXTURE),
        },
        "after_stats": after,
    }
    atomic_json(args.receipt, result)
    print(json.dumps({"verdict": "PASS", "instance_id": instance,
                      "elapsed_s": result["elapsed_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
