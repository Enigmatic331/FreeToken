"""MTP draft-head sidecar for DeepSeek-V4-Flash speculative decoding.

Wraps the checkpoint's own reference implementation (``<ckpt>/inference/model.py``,
which FreeToken already requires to be present) of the ``mtp.0.*`` block — a full
extra DSV4 layer whose attention is a 128-token sliding-window ring (state = one
small buffer; rollback = trivial) — loaded on ``FREETOKEN_MTP_DEVICE`` (default
``cuda:0``; on this rig ``cuda:2`` keeps drafting off the serving GPUs entirely).

API (bs=1):
  sidecar.prefill(hc, tokens)  -- ingest prompt: hc [1,S,hc,d] final-layer states
                                  from the main model, tokens [S] the NEXT-token
                                  ids aligned to those states (i.e. tokens[i] is
                                  the token AFTER the one that produced hc[i]).
  sidecar.draft(hc1, tok)      -- one step: returns the greedy draft token id.
  sidecar.rewind(n)            -- back the ring position up n steps (rejection).

The reference block is used verbatim except for three loading/runtime fixes
proven in scripts/mtp_acceptance_probe.py: fp4 params direct-assigned (torch has
no fp4 copy_), wo_a explicitly fp8-block-dequantized, and a plain-torch
sparse_attn (the tilelang kernel wants 141KB smem — H100-class only).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import torch

__all__ = ["MTPSidecar"]


def _torch_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, s, h, d = q.shape
    topk_idxs = topk_idxs.to(q.device)
    attn_sink = attn_sink.to(q.device)
    idx = topk_idxs.long().clamp_min(0)
    kvg = kv[torch.arange(b, device=kv.device)[:, None, None], idx]
    scores = torch.einsum("bshd,bstd->bsht", q, kvg).float() * softmax_scale
    scores = scores.masked_fill((topk_idxs < 0)[:, :, None, :], float("-inf"))
    m = scores.amax(-1, keepdim=True)
    e = torch.exp(scores - m)
    sumexp = e.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - m.squeeze(-1))
    o = torch.einsum("bsht,bstd->bshd", e.to(q.dtype), kvg).float() / sumexp[..., None]
    return o.to(q.dtype)


class MTPSidecar:
    def __init__(self, model_path: str, device: str | None = None):
        self.device = torch.device(device or os.environ.get("FREETOKEN_MTP_DEVICE", "cuda:0"))
        inf = os.path.join(model_path, "inference")
        spec = importlib.util.spec_from_file_location("_dsv4_reference", os.path.join(inf, "model.py"))
        ref = importlib.util.module_from_spec(spec)
        sys.path.insert(0, inf)  # reference imports its sibling kernel.py
        try:
            spec.loader.exec_module(ref)
        finally:
            sys.path.remove(inf)
        self._ref = ref
        ref.sparse_attn = _torch_sparse_attn
        # per-position logits + bf16 head matmul (reference keeps the head fp32:
        # a 2GB read per draft; bf16 halves it -- argmax is insensitive)
        ref.ParallelHead.get_logits = lambda head, x: torch.nn.functional.linear(
            x.to(head.weight.dtype), head.weight)

        with open(os.path.join(inf, "config.json")) as f:
            cfg = json.load(f)
        args = ref.ModelArgs(**cfg)
        args.max_batch_size = 1
        args.max_seq_len = 8192  # ring is window-sized; this only caps positional buffers
        ref.world_size, ref.rank = 1, 0
        ref.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        ref.scale_fmt = "ue8m0" if getattr(args, "scale_dtype", None) == "fp8" else getattr(args, "scale_fmt", None)
        ref.scale_dtype = torch.float8_e8m0fnu if getattr(args, "scale_dtype", None) == "fp8" else torch.float32

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(self.device):
            self.blk = ref.MTPBlock(args.n_layers, args)
            self.blk.embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
            self.blk.head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        torch.set_default_dtype(prev_dtype)
        self._load(model_path)
        if os.environ.get("FREETOKEN_MTP_BF16", "1") == "1":
            self._dequant_all_bf16()
        self.pos = 0

    def _dequant_all_bf16(self) -> None:
        """Convert every quantized Linear to a plain bf16 weight so forwards run as
        single F.linear calls (the reference linear() dispatches on weight dtype).
        Cuts draft latency ~5x by removing dozens of tiny act-quant/gemm launches;
        costs ~13.7 GB on the sidecar device. fp4: e2m1 nibbles (low nibble =
        element 2i) x e8m0 per-32 block scale. fp8: e4m3 x e8m0 per-128x128 block."""
        import torch.nn as nn

        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                           device=self.device)

        def deq_fp4(wp, sp):
            b = wp.data.view(torch.uint8)
            out, half = b.shape
            v = torch.empty(out, half * 2, device=self.device)
            v[:, 0::2] = lut[(b & 0xF).long()]
            v[:, 1::2] = lut[(b >> 4).long()]
            s = torch.exp2(sp.data.view(torch.uint8).to(torch.float32) - 127.0)
            return (v * s.repeat_interleave(32, dim=1)[:, : half * 2]).to(torch.bfloat16)

        def deq_fp8(wp, sp):
            s = torch.exp2(sp.data.view(torch.uint8).to(torch.float32) - 127.0)
            s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: wp.shape[0], : wp.shape[1]]
            return (wp.data.to(torch.float32) * s).to(torch.bfloat16)

        for mod in self.blk.modules():
            w = getattr(mod, "weight", None)
            if not isinstance(w, nn.Parameter) or getattr(mod, "scale", None) is None:
                continue
            if w.dtype == torch.float4_e2m1fn_x2:
                new = deq_fp4(w, mod.scale)
            elif w.dtype == torch.float8_e4m3fn:
                new = deq_fp8(w, mod.scale)
            else:
                continue
            mod.weight = nn.Parameter(new, requires_grad=False)
            mod.scale = None
        torch.cuda.empty_cache()

    def _load(self, model_path: str) -> None:
        import safetensors

        idx = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]
        handles: dict = {}

        def get(name):
            shard = idx[name]
            if shard not in handles:
                handles[shard] = safetensors.safe_open(
                    os.path.join(model_path, shard), framework="pt", device="cpu").__enter__()
            return handles[shard].get_tensor(name)

        sd = {n[len("mtp.0."):]: get(n) for n in idx if n.startswith("mtp.0.")}
        w, s = sd["attn.wo_a.weight"], sd.pop("attn.wo_a.scale")
        codes = s.view(torch.uint8).to(torch.float32)
        sf = torch.exp2(codes - 127.0).repeat_interleave(128, 0).repeat_interleave(128, 1)
        sd["attn.wo_a.weight"] = (w.to(torch.float32) * sf[: w.shape[0], : w.shape[1]]).to(torch.bfloat16)
        fp4 = set()
        for name, p in self.blk.named_parameters():
            if p.dtype == torch.float4_e2m1fn_x2 and name in sd:
                fp4.add(name)
                p.requires_grad_(False)
                p.data = sd[name].view(torch.float4_e2m1fn_x2).to(self.device)
        rest = {k: v for k, v in sd.items() if k not in fp4}
        missing, unexpected = self.blk.load_state_dict(rest, strict=False)
        missing = [m for m in missing if not m.startswith(("embed.", "head.")) and m not in fp4]
        assert not missing and not unexpected, (missing, unexpected)
        self.blk.embed.load_state_dict({"weight": get("embed.weight")})
        self.blk.head.load_state_dict({"weight": get("head.weight")})
        import torch.nn as _nn
        self.blk.head.weight = _nn.Parameter(
            self.blk.head.weight.data.to(torch.bfloat16), requires_grad=False)
        for h in handles.values():
            h.__exit__(None, None, None)

    class _fwd_guard:
        # reference forwards allocate under the global default dtype AND device
        def __init__(self, device):
            self._dev_ctx = torch.device(device)
        def __enter__(self):
            self._prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            self._dev_ctx.__enter__()
        def __exit__(self, *a):
            self._dev_ctx.__exit__(*a)
            torch.set_default_dtype(self._prev)

    @torch.inference_mode()
    def prefill(self, hc: torch.Tensor, tokens: torch.Tensor) -> int:
        """Ingest S aligned (state, next-token) pairs from pos 0; returns the greedy
        draft for the position after the last pair. (Reference forwards allocate
        under the global default dtype — guard to bf16 for the duration.)"""
        hc = hc.to(self.device, torch.bfloat16, non_blocking=True)
        tokens = tokens.to(self.device).view(1, -1)
        with self._fwd_guard(self.device):
            logits = self.blk(hc, 0, tokens)
        self.pos = hc.shape[1]
        return int(logits.view(hc.shape[1], -1)[-1].argmax().item())

    @torch.inference_mode()
    def draft(self, hc1: torch.Tensor, token: int) -> int:
        """One decode-position step: hc1 [1,1,hc,d] (or [hc,d]) main-model state,
        ``token`` the token sampled after it. Returns the draft for the next slot."""
        hc1 = hc1.to(self.device, torch.bfloat16, non_blocking=True).view(1, 1, *hc1.shape[-2:])
        t = torch.tensor([[token]], device=self.device)
        with self._fwd_guard(self.device):
            logits = self.blk(hc1, self.pos, t)
        self.pos += 1
        return int(logits.view(-1).argmax().item())

    def rewind(self, n: int = 1) -> None:
        """Rejection rollback: the ring slot(s) will be overwritten on re-entry."""
        self.pos -= n
