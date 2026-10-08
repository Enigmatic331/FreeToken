import os

import pytest
import torch
import torch.nn.functional as F

from freetoken.models.deepseek_v41.args import load_args
from freetoken.models.deepseek_v41.head import native_bf16_head_enabled, project_logits
from freetoken.models.deepseek_v41.model import Transformer


def test_flag_default_and_validation(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DSV41_NATIVE_BF16_HEAD", raising=False)
    assert not native_bf16_head_enabled()
    monkeypatch.setenv("FREETOKEN_DSV41_NATIVE_BF16_HEAD", "1")
    assert native_bf16_head_enabled()
    monkeypatch.setenv("FREETOKEN_DSV41_NATIVE_BF16_HEAD", "0")
    assert not native_bf16_head_enabled()
    monkeypatch.setenv("FREETOKEN_DSV41_NATIVE_BF16_HEAD", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        native_bf16_head_enabled()


@pytest.mark.parametrize("shape", [(4,), (1, 4), (5, 4), (2, 3, 4), (0, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_default_projection_is_unchanged(shape, dtype):
    torch.manual_seed(71006)
    hidden = torch.randn(shape).to(dtype)
    weight = torch.randn(7, 4)
    assert torch.equal(project_logits(hidden, weight), F.linear(hidden.float(), weight))


def test_no_silent_input_rounding_or_cpu_fallback():
    weight = torch.empty(7, 4, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="BF16 weights and hidden"):
        project_logits(torch.zeros(1, 4), weight)
    with pytest.raises(ValueError, match="same CUDA device"):
        project_logits(torch.zeros(1, 4, dtype=torch.bfloat16), weight)
    with pytest.raises(ValueError, match="shape"):
        project_logits(torch.zeros(1, 5, dtype=torch.bfloat16), weight)


@pytest.mark.parametrize("shape", [(4,), (1, 4), (2, 3, 4), (0, 4)])
def test_rowwise_shape_contract_without_cuda(monkeypatch, shape):
    # Mock only the device/kernel boundary; the installed CUDA helper is also
    # checked against the full checkpoint in the separate numerical lab.
    original_mm = torch.mm
    calls = []

    def fp32_output_mm(a, b, *, out_dtype):
        assert a.shape[0] == 1 and a.dtype == b.dtype == torch.bfloat16
        assert out_dtype == torch.float32
        calls.append(a.shape)
        return original_mm(a.float(), b.float())

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch, "mm", fp32_output_mm)
    hidden = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(torch.bfloat16)
    weight = torch.arange(28).reshape(7, 4).to(torch.bfloat16)
    result = project_logits(hidden, weight)
    expected = F.linear(hidden.float(), weight.float())
    assert result.shape == (*shape[:-1], 7) and result.dtype == torch.float32
    assert torch.equal(result, expected)
    assert len(calls) == hidden.numel() // 4


@pytest.mark.skipif(not os.path.exists(os.getenv("FREETOKEN_DSV41_MODEL_PATH", "")), reason="checkpoint absent")
def test_meta_storage_saving_and_mtp_guard(monkeypatch):
    args = load_args(os.environ["FREETOKEN_DSV41_MODEL_PATH"])
    monkeypatch.setenv("FREETOKEN_DSV41_NATIVE_BF16_HEAD", "0")
    with torch.device("meta"):
        baseline = Transformer(args)
    monkeypatch.setenv("FREETOKEN_DSV41_NATIVE_BF16_HEAD", "1")
    with torch.device("meta"):
        candidate = Transformer(args)
    old, new = dict(baseline.named_parameters()), dict(candidate.named_parameters())
    assert old.keys() == new.keys()
    assert old["head"].dtype == torch.float32 and new["head"].dtype == torch.bfloat16
    assert all(old[k].dtype == new[k].dtype and old[k].shape == new[k].shape for k in old if k != "head")
    size = lambda params: sum(p.numel() * p.element_size() for p in params.values())
    assert size(old) - size(new) == 1_323_827_200
    args.dspark_enabled = True
    with torch.device("meta"), pytest.raises(ValueError, match="MTP/DSpark"):
        Transformer(args)
