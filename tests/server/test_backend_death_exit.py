from __future__ import annotations

import pytest

from freetoken.server import api_server


class _HardExit(RuntimeError):
    def __init__(self, code: int) -> None:
        self.code = code


def test_backend_death_hard_exit_is_nonzero(monkeypatch):
    def fake_exit(code: int) -> None:
        raise _HardExit(code)

    api_server._SHUTTING_DOWN.clear()
    monkeypatch.setattr(api_server.os, "_exit", fake_exit)

    with pytest.raises(_HardExit) as caught:
        api_server._hard_exit_after_backend_death()

    assert caught.value.code == 1


def test_backend_death_exit_is_suppressed_during_external_shutdown(monkeypatch):
    def unexpected_exit(_code: int) -> None:
        raise AssertionError("external shutdown must not be converted into a failure")

    api_server._SHUTTING_DOWN.set()
    monkeypatch.setattr(api_server.os, "_exit", unexpected_exit)
    try:
        api_server._hard_exit_after_backend_death()
    finally:
        api_server._SHUTTING_DOWN.clear()
