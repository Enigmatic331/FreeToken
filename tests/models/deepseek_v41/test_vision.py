from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from freetoken.models.deepseek_v41.image_processor import image_token_types
from freetoken.models.deepseek_v41.moe import Gate
from freetoken.models.deepseek_v41.vision import Aligner, ViT, image_span_embeddings


def _vision_args():
    return SimpleNamespace(
        dim=8,
        vision_n_layers=1,
        vision_dim=8,
        vision_n_heads=2,
        vision_inter_dim=16,
        vision_patch_size=2,
        vision_downsample_ratio=2,
        vision_rope_theta=10_000.0,
    )


def test_vision_tower_and_aligner_build_complete_native_span_on_cpu():
    args = _vision_args()

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = ViT(args)
            self.aligner = Aligner(args)
            self.image_start = nn.Parameter(torch.randn(args.dim, dtype=torch.bfloat16))
            self.image_newline = nn.Parameter(torch.randn(args.dim, dtype=torch.bfloat16))
            self.image_end = nn.Parameter(torch.randn(args.dim, dtype=torch.bfloat16))

        def encode_image(self, patches, n_h, n_w):
            return self.aligner(self.vision(patches, n_h, n_w), n_h, n_w)

    model = Transformer().eval()
    patches = torch.randn(4, 3, 2, 2, dtype=torch.bfloat16)
    span = image_span_embeddings(model, patches, 2, 2, image_token_types(1, 1))
    assert span.shape == (4, args.dim)
    torch.testing.assert_close(span[0], model.image_start)
    torch.testing.assert_close(span[2], model.image_newline)
    torch.testing.assert_close(span[3], model.image_end)


def test_router_uses_visual_bias_only_for_image_rows():
    args = SimpleNamespace(
        n_activated_experts=1,
        score_func="sqrtsoftplus",
        gate_temp=1.0,
        norm_topk_prob=True,
        route_scale=1.0,
        n_routed_experts=3,
        dim=3,
    )
    gate = Gate(args, enable_vision=True)
    gate.weight.data.zero_()
    gate.bias.data.copy_(torch.tensor([3.0, 2.0, 1.0]))
    gate.bias_vl.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
    hidden = torch.zeros((2, 3), dtype=torch.bfloat16)

    _, indices = gate(hidden, torch.tensor([False, True]))
    assert indices.tolist() == [[0], [2]]
