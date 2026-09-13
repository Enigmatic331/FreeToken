from __future__ import annotations

import torch

from freetoken.engine.engine import _make_dummy_weight_state_dict


def test_dummy_e8m0_scale_uses_finite_unit_code() -> None:
    model_state = {
        "weight_scale_inv": torch.empty(7, dtype=torch.float8_e8m0fnu),
    }

    dummy = _make_dummy_weight_state_dict(model_state, device=torch.device("cpu"))

    scale = dummy["weight_scale_inv"]
    assert scale.dtype == torch.float8_e8m0fnu
    assert torch.equal(scale.view(torch.uint8), torch.full((7,), 127, dtype=torch.uint8))
