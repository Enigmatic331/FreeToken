from __future__ import annotations

import pytest

from freetoken.server.args import parse_args


def _parse(*extra: str):
    args, _ = parse_args(
        ["--model", "/models/anonymous", "--dtype", "bfloat16", *extra]
    )
    return args


def test_swa_full_tokens_ratio_is_exposed_on_the_serve_cli():
    assert _parse("--swa-full-tokens-ratio", "0.28125").swa_full_tokens_ratio == 0.28125


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "not-a-number"])
def test_swa_full_tokens_ratio_rejects_values_outside_open_unit_interval(value: str):
    with pytest.raises(SystemExit):
        _parse("--swa-full-tokens-ratio", value)
