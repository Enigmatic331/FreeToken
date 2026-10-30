#!/usr/bin/env python3
"""Two-strike local-only watchdog for the final DSv4.1 service."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

UNIT = "freetoken-dsv41-production.service"
PREWARM = "freetoken-dsv41-production-prewarm.service"
INGRESS_SOCKET = "freetoken-dsv41-production-ingress.socket"
INGRESS_SERVICE = "freetoken-dsv41-production-ingress.service"
MODEL = "DeepSeek-V4.1-Flash-1017"
ROOT = "/home/enigmatic331/models/DeepSeek-V4.1-Flash-1017"
STATE = Path("/var/lib/freetoken-dsv41/watchdog.json")
RECEIPT = Path("/var/lib/freetoken-dsv41/prewarm.json")
LOCK = Path("/run/lock/freetoken-dsv41-production-watchdog.lock")
STARTUP_GRACE_S = 30 * 60


def systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/usr/bin/systemctl", *args], text=True,
                          capture_output=True, timeout=60, check=False)


def prop(name: str) -> str:
    return systemctl("show", UNIT, f"--property={name}", "--value").stdout.strip()


def fetch(path: str) -> dict:
    with urllib.request.urlopen("http://127.0.0.1:19291" + path, timeout=8) as response:
        return json.load(response)


def healthy() -> tuple[bool, str, str | None]:
    try:
        health = fetch("/health")
        models = fetch("/v1/models")
        rows = [row for row in models.get("data", []) if row.get("id") == MODEL]
        good = (health.get("status") == "ok" and health.get("model") == MODEL
                and len(rows) == 1 and rows[0].get("root") == ROOT
                and int(rows[0].get("max_model_len", 0)) == 524288)
        return good, f"health={health.get('status')} model={health.get('model')}", health.get("instance_id")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def active_requests() -> int | None:
    try:
        return int(fetch("/v1/stats")["requests"]["active"])
    except Exception:
        return None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}


def write_state(value: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, STATE)


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0

        state = read_json(STATE)
        is_active = prop("ActiveState") in {"active", "activating", "reloading"}
        if not is_active:
            systemctl("reset-failed", UNIT)
            result = systemctl("start", "--no-block", UNIT)
            write_state({"failures": 0, "action": "start", "at": time.time(),
                         "rc": result.returncode})
            return result.returncode

        ok, detail, instance = healthy()
        if ok:
            receipt = read_json(RECEIPT)
            if receipt.get("verdict") == "PASS" and receipt.get("instance_id") == instance:
                systemctl("start", "--no-block", INGRESS_SOCKET)
            else:
                systemctl("stop", INGRESS_SERVICE, INGRESS_SOCKET)
                systemctl("restart", "--no-block", PREWARM)
            write_state({"failures": 0, "detail": detail, "instance_id": instance,
                         "at": time.time()})
            return 0

        try:
            uptime = float(prop("ActiveEnterTimestampMonotonic") or 0) / 1_000_000
            boot = float(Path("/proc/uptime").read_text().split()[0])
            age = max(0.0, boot - uptime)
        except Exception:
            age = 0.0
        if age < STARTUP_GRACE_S:
            write_state({"failures": 0, "detail": detail, "startup_grace": True,
                         "age_s": age, "at": time.time()})
            return 0
        active = active_requests()
        if active not in (None, 0):
            write_state({"failures": 0, "detail": detail, "deferred_active": active,
                         "at": time.time()})
            return 0
        failures = int(state.get("failures", 0)) + 1
        if failures < 2:
            write_state({"failures": failures, "detail": detail, "at": time.time()})
            return 0
        systemctl("stop", INGRESS_SERVICE, INGRESS_SOCKET)
        result = systemctl("restart", "--no-block", UNIT)
        write_state({"failures": 0, "action": "restart", "detail": detail,
                     "at": time.time(), "rc": result.returncode})
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
