"""Qwen3-VL visual encoder adapter used by Qwen3.8.

The checkpoint's 333 ``model.visual.*`` tensors are byte-for-byte compatible with
Transformers' Qwen3VLVisionModel.  Keeping that well-tested tower behind a small
BaseOP adapter avoids duplicating its patch ordering, interpolation and packed
bidirectional-attention rules while letting FreeToken place it independently.
"""

from __future__ import annotations

from typing import Any

import torch

from freetoken.layers import BaseOP


class QwenVLVisualEncoder(BaseOP):
    def __init__(self, vision_config: Any) -> None:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        if hasattr(vision_config, "to_dict"):
            raw = vision_config.to_dict()
        elif isinstance(vision_config, dict):
            raw = dict(vision_config)
        elif isinstance(getattr(vision_config, "_data", None), dict):
            raw = dict(vision_config._data)
        else:
            raw = vars(vision_config)
        raw.pop("model_type", None)
        self._module = Qwen3VLVisionModel(Qwen3VLVisionConfig(**raw))
        self._target_device: torch.device | None = None

    def set_target_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError(f"Qwen vision device must be CUDA, got {device}")
        self._target_device = device

    @property
    def device(self) -> torch.device:
        if self._target_device is not None:
            return self._target_device
        return next(self._module.parameters()).device

    def state_dict(self, *, prefix: str = "", result=None):
        result = {} if result is None else result
        for name, parameter in self._module.named_parameters():
            result[f"{prefix}.{name}" if prefix else name] = parameter
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        target = self._target_device
        casted = {}
        for name, parameter in self._module.named_parameters():
            key = f"{prefix}.{name}" if prefix else name
            if key not in state_dict:
                raise RuntimeError(f"Missing Qwen vision weight: {key}")
            value = state_dict.pop(key)
            casted[name] = value.to(
                device=target if target is not None else value.device,
                dtype=parameter.dtype,
            )
        if state_dict and not _internal:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
        self._module.load_state_dict(casted, assign=True, strict=True)
        # HF keeps the vision rotary inverse frequencies as a non-persistent buffer, so
        # they are absent from both checkpoint and state_dict. Construction happens under
        # FreeToken's meta-device context; materialize this sole buffer explicitly.
        rotary = self._module.rotary_pos_emb
        rotary.inv_freq = torch.nn.Buffer(
            1.0
            / (
                rotary.theta
                ** (
                    torch.arange(
                        0,
                        rotary.dim,
                        2,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    / rotary.dim
                )
            ),
            persistent=False,
        )
        self._module.eval()

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        device = self.device
        output = self._module(
            pixel_values.to(device=device, dtype=next(self._module.parameters()).dtype),
            grid_thw=image_grid_thw.to(device=device, dtype=torch.int64),
            return_dict=True,
        )
        return output.pooler_output


__all__ = ["QwenVLVisualEncoder"]
