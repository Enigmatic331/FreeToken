"""G3 P1 stream-lifetime and dependency qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="G3 needs CUDA")
SOURCE = Path(__file__).resolve().parents[3] / "python/freetoken/models/deepseek_v41/authority_fusion.py"


def test_p1_uses_one_module_stream_and_finally_join():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "authority_attention_decode"
    )
    text = ast.unparse(function)
    assert text.count("torch.cuda.stream(_KV_STREAM)") == 1
    assert "_KV_STREAM.wait_stream(current)" in text
    assert "current.wait_stream(_KV_STREAM)" in text
    assert "finally:" in text
    assert "all_reduce" not in text and ".broadcast(" not in text


def test_missing_join_negative_control_is_observable():
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    source = torch.zeros(1, device="cuda")
    with torch.cuda.stream(side):
        torch.cuda._sleep(200_000_000)
        source.fill_(7)
    # The planted missing-join path must be capable of exposing stale state.
    stale = source.clone()
    main.wait_stream(side)
    exact = source.clone()
    torch.cuda.synchronize()
    assert exact.item() == 7
    assert stale.item() == 0  # planted missing join must be observed, never waved through.
