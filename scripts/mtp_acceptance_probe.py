#!/usr/bin/env python3
"""Offline MTP acceptance probe.

Replays (token, final-hc-state) dumps from a FREETOKEN_MTP_PROBE_DIR serve
through NVIDIA's reference MTPBlock (ground-truth math) and measures how often
the draft head's greedy prediction matches the token the model actually
produced (temperature-0 serving => sampled == argmax).

Draft convention: MTP(h_state[p], token[p+1]) predicts token[p+2].
"""
import glob, json, os, sys

import torch

CKPT = "/home/enigmatic331/models/DeepSeek-V4-Flash-DSFP4"
PROBE = "/home/enigmatic331/models/mtp_probe"
sys.path.insert(0, os.path.join(CKPT, "inference"))

import model as ref  # noqa: E402  (reference implementation)


def sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Plain-torch mirror of kernel.sparse_attn (its tilelang kernel needs 141KB
    smem — H100-class; consumer cards cap at ~99KB). Same math: gather kv rows at
    topk_idxs (-1 masked), scaled dot, softmax with an extra exp(sink) mass term
    that contributes no value. bf16 matmuls (fp32 accum) to match the kernel."""
    b, s, h, d = q.shape
    idx = topk_idxs.long().clamp_min(0)
    kvg = kv[torch.arange(b, device=kv.device)[:, None, None], idx]  # [b,s,t,d]
    scores = torch.einsum("bshd,bstd->bsht", q, kvg).float() * softmax_scale
    scores = scores.masked_fill((topk_idxs < 0)[:, :, None, :], float("-inf"))
    m = scores.amax(-1, keepdim=True)
    e = torch.exp(scores - m)
    sumexp = e.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - m.squeeze(-1))
    o = torch.einsum("bsht,bstd->bshd", e.to(q.dtype), kvg).float() / sumexp[..., None]
    return o.to(q.dtype)


ref.sparse_attn = sparse_attn_torch
# per-position logits for whole-stream drafting (reference keeps only x[:, -1])
ref.ParallelHead.get_logits = lambda self, x: torch.nn.functional.linear(x.float(), self.weight)

# ---- reference globals normally set by Transformer.__init__ ----
with open(os.path.join(CKPT, "inference", "config.json")) as f:
    cfg = json.load(f)
args = ref.ModelArgs(**cfg)
args.max_batch_size = 1
args.max_seq_len = 4096
ref.world_size, ref.rank = 1, 0
ref.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
ref.scale_fmt = "ue8m0" if getattr(args, "scale_dtype", None) == "fp8" else getattr(args, "scale_fmt", None)
ref.scale_dtype = torch.float8_e8m0fnu if getattr(args, "scale_dtype", None) == "fp8" else torch.float32
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")

# ---- build MTP block + shared embed/head, load weights ----
print("building reference MTPBlock ...", flush=True)
blk = ref.MTPBlock(args.n_layers, args)
blk.embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
blk.head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)

idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
import safetensors  # noqa: E402

handles = {}
def get(name):
    shard = idx[name]
    if shard not in handles:
        handles[shard] = safetensors.safe_open(os.path.join(CKPT, shard), framework="pt", device="cpu").__enter__()
    return handles[shard].get_tensor(name)

sd = {n[len("mtp.0."):]: get(n) for n in idx if n.startswith("mtp.0.")}
# wo_a: fp8+scale in the checkpoint but a bf16 param in the reference block —
# dequantize explicitly (128-block e8m0 exponent codes, value = 2^(code-127)),
# matching FreeToken's loader; a silent load_state_dict cast would be garbage.
_w, _s = sd["attn.wo_a.weight"], sd.pop("attn.wo_a.scale")
_codes = _s.view(torch.uint8).to(torch.float32)
_sf = torch.exp2(_codes - 127.0).repeat_interleave(128, 0).repeat_interleave(128, 1)
sd["attn.wo_a.weight"] = (_w.to(torch.float32) * _sf[: _w.shape[0], : _w.shape[1]]).to(torch.bfloat16)
# torch has no copy_ kernel for Float4_e2m1fn_x2: pull fp4 params out of the
# state-dict load and assign their storage directly instead (checkpoint stores
# the packed nibbles as int8 — view to the fp4x2 dtype the reference expects).
fp4_names = set()
for name, p in blk.named_parameters():
    if p.dtype == torch.float4_e2m1fn_x2 and name in sd:
        fp4_names.add(name)
        p.requires_grad_(False)
        p.data = sd[name].view(torch.float4_e2m1fn_x2).to("cuda")
rest = {k: v for k, v in sd.items() if k not in fp4_names}
missing, unexpected = blk.load_state_dict(rest, strict=False)
missing = [m for m in missing if not m.startswith(("embed.", "head.")) and m not in fp4_names]
print(f"fp4 direct-assigned: {len(fp4_names)} | missing: {missing[:6]} | unexpected: {unexpected[:6]}", flush=True)
blk.embed.load_state_dict({"weight": get("embed.weight")})
blk.head.load_state_dict({"weight": get("head.weight")})

# ---- reassemble request streams from probe dumps ----
entries = []
for f in sorted(glob.glob(os.path.join(PROBE, "probe_*.pt"))):
    entries.extend(torch.load(f, map_location="cpu", weights_only=False))
print(f"{len(entries)} probe entries", flush=True)

requests = []  # each: list of (pos, token, h[hc,d])
cur = {}
for kind, input_ids, h, positions in entries:
    if kind == "prefill":
        toks, hs, poss = input_ids[0], h[0], positions
        if (poss == 0).any() and cur:
            requests.append(cur); cur = {}
        for i in range(toks.numel()):
            cur[int(poss[i])] = (int(toks[i]), hs[i])
    else:  # decode: [B,1]; probe serve runs bs=1
        cur[int(positions[0])] = (int(input_ids[0, 0]), h[0, 0])
if cur:
    requests.append(cur)
print(f"{len(requests)} request streams", flush=True)

# ---- replay ----
total = accept = 0
per_req = []
for req in requests:
    poss = sorted(req)
    if len(poss) < 8 or poss != list(range(poss[0], poss[0] + len(poss))):
        print("  skipping stream (gaps or too short):", len(poss)); continue
    toks = torch.tensor([req[p][0] for p in poss])
    hs = torch.stack([req[p][1] for p in poss])  # [N, hc, d]
    N = hs.shape[0]
    # inputs: x = h[0..N-3], tok = toks[1..N-2] -> draft target toks[2..N-1]
    x = hs[: N - 2].unsqueeze(0).cuda().to(torch.bfloat16)   # [1, S, hc, d]
    t_in = toks[1 : N - 1].unsqueeze(0).cuda()               # [1, S]
    target = toks[2:].cuda()                                  # [S]
    S = x.shape[1]
    # single whole-stream call at start_pos=0: the reference decode branch only
    # accepts s=1 writes, but its prefill branch handles arbitrary s in one shot.
    with torch.inference_mode():
        logits = blk(x, 0, t_in)
    drafts = logits.view(S, -1).argmax(-1)
    ok = (drafts == target).sum().item()
    per_req.append((ok, S))
    total += S; accept += ok
    print(f"  stream len {S}: acceptance {ok}/{S} = {ok/S:.1%}", flush=True)

if total:
    print(f"\nOVERALL DRAFT ACCEPTANCE: {accept}/{total} = {accept/total:.1%}")
    p = accept / total
    print(f"projected decode speedup (MTP-1, verify≈decode cost): ~{(1+p):.2f}x upper bound")
