from __future__ import annotations

import pytest

from freetoken.server import launch


class _HardExit(RuntimeError):
    def __init__(self, code: int) -> None:
        self.code = code


def test_failed_scheduler_uses_immediate_nonzero_exit(monkeypatch, capsys):
    def fake_exit(code: int) -> None:
        raise _HardExit(code)

    monkeypatch.setattr(launch.os, "_exit", fake_exit)

    with pytest.raises(_HardExit) as caught:
        try:
            raise MemoryError("synthetic worker failure")
        except MemoryError as exc:
            launch._hard_exit_failed_scheduler(exc)

    assert caught.value.code == 1
    assert "terminating without interpreter cleanup" in capsys.readouterr().out
