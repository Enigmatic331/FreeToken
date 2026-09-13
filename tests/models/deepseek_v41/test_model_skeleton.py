from __future__ import annotations

import os

import pytest
import torch

from freetoken.models.deepseek_v41.args import load_args
from freetoken.models.deepseek_v41.model import Transformer
from freetoken.models.deepseek_v41.weight import (
    expected_resident_specs,
    inspect_checkpoint,
)


MODEL_PATH = os.environ.get("FREETOKEN_DSV41_MODEL_PATH", "")
DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="official checkpoint absent")
def test_meta_model_matches_every_resident_checkpoint_parameter():
    args = load_args(MODEL_PATH)
    with torch.device("meta"):
        model = Transformer(args)
    actual = dict(model.named_parameters())
    raw = expected_resident_specs(args)
    expected = {}
    for name, (dtype, shape) in raw.items():
        if name.endswith(".attn.wo_a.scale"):
            continue
        if name.endswith(".attn.wo_a.weight"):
            name = name.removesuffix(".weight")
            dtype = "BF16"
        elif name == "head.weight":
            name, dtype = "head", "F32"
        elif name.endswith((".attn.compressor.wgate.weight",)):
            dtype = "F32"
        elif name.endswith(".attn.compressor.wkv.weight"):
            layer = int(name.split(".")[1])
            if args.compress_ratios[layer] > 1:
                dtype = "F32"
        expected[name] = (DTYPES[dtype], shape)

    assert len(actual) == len(expected) == 1_174
    assert set(actual) == set(expected)
    for name, parameter in actual.items():
        dtype, shape = expected[name]
        assert parameter.dtype == dtype, name
        assert tuple(parameter.shape) == shape, name
    device_bytes = sum(p.numel() * p.element_size() for p in actual.values())
    assert device_bytes == inspect_checkpoint(MODEL_PATH).baseline_device_bytes
