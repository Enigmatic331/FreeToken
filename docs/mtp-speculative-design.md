# MTP speculative decoding for DeepSeek-V4-Flash — design notes

Status: groundwork + acceptance probe DONE — measured 80.4% greedy draft acceptance
(essay 78% / technical 82% / code 90% / fiction 70%, 1,608 tokens, T=0) via
FREETOKEN_MTP_PROBE_DIR dumps replayed through the reference MTPBlock
(scripts/mtp_acceptance_probe.py). Projected ~1.5-1.7x decode. BUILD IS GO.
Measured motivation: decode is latency-bound (~50ms/token walks 43 layers
sequentially; DRAM/PCIe mostly idle between bursts). MTP verify converts
decode into 2-token micro-prefill: same weight traffic serves 2 tokens.
Expected 1.5–1.9× tg at DeepSeek-reported ~80–90% acceptance.

## What the checkpoint provides

`mtp.0.*`: a full extra DSV4 block (own MLA attention w/ indexer-compatible
config, own 256 routed ds_fp4 experts + shared expert + gate w/ bias) plus:
- `e_proj`/`h_proj` (+`enorm`/`hnorm`): merge sampled-token embedding with the
  final-layer hyper-connection hidden state `x:[b,s,hc,d]`
- own `hc_head_fn/base/scale` + `norm`: its own head mixing; shares main
  `embed` and `head` weights.
- NOTE: nvidia's NVFP4 cast never touched `mtp.0` experts — they are ds_fp4
  (e8m0/blk32) in both the original and our converted checkpoint.

Loaders (landed in `models/deepseek_v4/weight.py`):
- `iter_mtp_weights(path, device)` — 39 dense tensors, checkpoint dtype.
- `load_dsfp4_mtp_expert_bank(path, args)` — pinned single-layer bank,
  EP-sharded via `ep_shard` like the main layers.

## Draft step (per decoded token)

Inputs: final-layer hc state `x_t` (BEFORE the head's hc collapse) and the
just-sampled token `s`. Compute:
  `h = h_proj(hnorm-collapsed x_t) + e_proj(enorm(embed(s)))` → one full block
  forward (attention over MTP-lane KV at position t+1) → MTP head → draft d.
Then verify `[s, d]` in one 2-token extend through the main 43 layers.

## Integration work items (in dependency order)

1. **MTP KV lane**: 44th attention state. Options: (a) extend the DSV4 pool
   to 44 layers (touches `_dsv4_pool_sizes`, page tables, indexer state);
   (b) dedicated side pool for the MTP block only. (b) is less invasive.
   The MTP block needs KV appended only for ACCEPTED tokens; on rejection its
   tentative entry must roll back too.
2. **Expert compute for MTP layer**: the resident bank (3.3 GB pinned / rank
   under EP) can reuse the offload cache machinery as a pinned 257th layer,
   or a dedicated always-resident GPU copy (1.65 GB/rank EP-sharded VRAM) to
   keep the draft step off the PCIe critical path. Prefer GPU-resident:
   drafting must be FAST or acceptance gains drown in draft latency.
3. **2-token verify step**: reuse the chunked-prefill (`extend`) path for
   [s, d]; needs decode-manager plumbing for a 2-token step + CUDA graph
   capture at the new shape (or run verify uncaptured first, measure).
4. **Rollback on rejection**: tentative KV/window/indexer state for position
   t+2 must be discarded. Relevant prior art: upstream
   `feat/decode-token-checkpoint` (ea5348b) snapshots decode state into the
   prefix cache at tool-call anchors — right primitive, too heavy per-step.
   Needed: a lightweight tail rollback (decrement seq_len, free window-pool
   slot, restore indexer/compressor tail state). This is the hardest item;
   audit `kvcache/hybrid_swa_pool.py` + dsv4 compressor state first.
5. **Sampling acceptance**: greedy-match first (temperature 0 benchmarking),
   then standard speculative rejection sampling for temperature > 0.
6. **Scheduler loop**: decode step becomes draft→verify→accept{1,2}; overlap
   scheduling must handle variable tokens-per-step (radix/streaming already
   tolerate multi-token appends on the prefill path).

## Suggested first milestone (next session)

Acceptance-rate probe BEFORE any KV/scheduler surgery: hook the final-layer
hc state + sampled token in a live single-GPU serve, run the MTP block
forward "stateless" (recompute its attention over the last W tokens' hc
states kept in a ring buffer — approximation acceptable for measurement),
and log draft-vs-actual agreement over a few thousand tokens. If acceptance
lands < ~60% the full build isn't worth it; > 80% strongly is.
