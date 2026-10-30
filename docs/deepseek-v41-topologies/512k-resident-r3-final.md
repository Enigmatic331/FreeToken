# DeepSeek V4.1 Flash 512K resident-r3 topology

This is the frozen production topology for the local four-GPU EP deployment.
It complements the generic [DeepSeek V4.1 documentation](../deepseek-v41.md);
the values below are deployment-qualified settings, not portable defaults.

## Hardware and placement

Logical rank order:

1. RTX 4090 48 GiB: text backbone, attention/KV authority, native vision,
   decode experts 0-79, prefill experts 0-159.
2. RTX 4080 SUPER 32 GiB: decode experts 80-219, prefill experts 160-303.
3. CMP 170HX x4 64 GiB: decode-only experts 220-303.
4. CMP 170HX x16 64 GiB: decode and prefill experts 304-383.

The exact geometry is:

```text
decode shards:   80,140,84,80
prefill shards:  160,144,0,80
storage ranges:  0:160,80:224,220:84,304:80
MoE cache slots: 960,1472,3360,3360
Engram ranks:    0,1
context tokens:  524288
vision device:   0
```

The deployment uses native BF16 head output, SM89 FP8 decode tuning, the
qualified SM80 DeepSeek-FP4 grouped-prefill policy `m64n32w4s2`, 8K prefill
chunks, BF16 KV, and one concurrent request. Packed KV, MTP, P2P, and IB
collectives are disabled. Both CMP expert banks are fully resident; a static
CMP miss/refill is a rejection condition.

Native image understanding is enabled with:

```text
FREETOKEN_LOAD_VISION=1
--vision-device 0
```

The vision tower shares the RTX 4090 with the text backbone. This avoids a
cross-device image-embedding copy and uses the fastest available vision
compute while retaining a measured 2,924 MiB minimum free at 506K context.

## Qualification summary

- Exact native OCR on the sealed 1200x700 fixture returned `7429`, stopped
  naturally, and used 533 prompt tokens. The answer was absent from the prompt.
- Exact 506K retrieval passed at 519.4854 seconds TTFT / 974.04 prefill
  token/s, with 3954 of 4096 KV pages used.
- Minimum free VRAM at 506K was 2,924 / 3,171 / 3,994 / 3,186 MiB on the
  4090 / 4080 SUPER / CMP x4 / CMP x16 respectively.
- Both CMP ranks recorded zero expert misses. Cleanup returned main KV and SWA
  to zero without a service restart.
- Maximum GPU and DIMM temperatures were 72C and 57C.
- The retained text-only qualification measured 1,203.87 prefill token/s at
  cold 40K and 21.3776 token/s median decode, with the full topology passing
  soak, cancellation, and boot-readiness gates.

## Production identity

The machine release uses `freetoken-dsv41-production.service`, loopback
`127.0.0.1:19291`, and gated application ingress `172.17.0.1:1919`. Startup
prewarm verifies stochastic text, exact text, omitted-output-token behavior,
exact native OCR, and zero KV/SWA residue before opening ingress. The local
release bundle and operator runbook are retained under:

```text
/home/enigmatic331/models/dsv41-flash-results/2026-10-29-dual-cmp-topology/
  release-512k-residentr3-final/
```

The exact production systemd units, prewarm/watchdog helpers, and sealed OCR
fixture are also versioned in
[`512k-resident-r3-final/`](512k-resident-r3-final/README.md). They retain the
qualified host paths and GPU UUIDs intentionally and must be reviewed before
installation on another machine.

Do not copy this topology to a different driver, PCIe layout, GPU order, or
checkpoint without rerunning exact output, near-capacity, cleanup, and thermal
gates.
