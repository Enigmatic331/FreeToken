# DeepSeek-V4.1-Flash experimental serving

FreeToken serves DeepSeek-V4.1 directly from the official safetensors checkpoint.
It implements the main 40-layer model, CSA2 paged attention, DS-FP4 routed
experts, the two row-sharded Engram tables, the native vision tower, and the
checkpoint's DSpark/MTP speculative drafter.

In heterogeneous EP, one rank owns the complete TP1 text backbone while every
rank can own an independently sized routed-expert shard and selected ranks own
rows of each Engram table. Engram's three-token history has an address-stable
graph input. Position-bucketed heterogeneous-EP CUDA graphs have passed exact
real-checkpoint replay and long-generation gates on qualified RTX 5090 systems.
They remain an explicit opt-in because P2P, driver, allocator, and topology
changes require requalification: set `FREETOKEN_DSV41_CUDA_GRAPH=1` and use
`--cuda-graph-max-bs 1` only after passing the correctness gates on the target
machine.

## Native vision and DSpark/MTP

Pass `--vision-device` to load the checkpoint's native image encoder. The
auxiliary device may sit outside the text EP group. Image requests deliberately
use ordinary target decoding; the checkpoint's DSpark context path is text-only.

DSpark is opt-in with `--speculative-dspark --dspark-device DEVICE` and currently
requires `--max-running-requests 1`. The auxiliary device holds the MTP stack so
it does not evict target experts. Set
`FREETOKEN_DSV41_DSPARK_DRAFT_CUDA_GRAPH=1` to graph the resident draft backbone,
greedy Markov chain, and context catch-up. Greedy drafting does not allocate a
vocabulary-sized probability matrix; sampled decoding retains exact rejection
sampling.

Acceptance depends strongly on the request. A fixed verification length can be
selected with `--dspark-verification-length`. For serving, the cumulative
circuit breaker can permanently return an unprofitable request to ordinary
decode:

```bash
--dspark-fallback-acceptance 0.52 \
--dspark-fallback-min-drafted 16 \
--dspark-fallback-cumulative
```

The threshold is topology-specific and should be derived from an A/B test. The
qualified heterogeneous EP3 launcher in
`scripts/run_dsv41_dspark_4090_candidate.sh` uses a four-token verification
prefix and keeps multimodal serving enabled on the same auxiliary card.

Authority EP can also overlap its decode-time expert-cache refill with the
independent shared-expert projection by setting
`FREETOKEN_DSV41_DECODE_REFILL_OVERLAP=1`. The routed-expert GEMM joins the refill
stream before consuming cache slots, and the existing collective remains after
the GEMM. The split is disabled for prefill, CPU/hybrid decode, expert workers,
and dense-parallel modes. It is CUDA-graph safe but remains opt-in pending
qualification on each target topology.

Decode route partitioning can be collapsed from the ordinary tensor-operation
chain into one exact Triton kernel with `FREETOKEN_DSV41_FUSED_ROUTE_PREP=1`.
The kernel localizes global expert ids, zeros peer-owned weights, and substitutes
cache-safe ids for inactive routes in one launch. Prefill retains its existing
sentinel-aware path. This switch is independent of cache hit rate and remains
opt-in until the target topology passes exact graph-replay and throughput gates.

Authority EP can also combine the decode hidden state, route weights, and route
ids into one bit-preserving dispatch payload with
`FREETOKEN_DSV41_FUSED_DECODE_DISPATCH=1`. This replaces three small per-layer
broadcasts with one without changing dtypes or route accumulation order. It is
decode-only; phase-aware and ordinary prefill retain their existing transport.
The fused dispatch is CUDA-graph safe on the qualified PyTorch NCCL path, but it
remains opt-in because small-collective latency is topology dependent.

The fused sqrt-softplus router is likewise retained behind
`FREETOKEN_DSV41_FUSED_ROUTER=1`. Its standalone CUDA numerical fixture passes,
but the first full EP2 prefill does not complete, so ordinary serving keeps the
proven PyTorch selection/normalization path.

Two dense-parallel research modes are available but are not recommended as the
default serving topology:

- `--dsv41-tp2-ep2` shards both attention and shared-expert projections. It
  executes the complete dense backbone on both ranks.
- `--dsv41-attention-tp2-ep2` shards attention only. The backbone root retains
  the router and shared expert, overlaps shared-expert compute with the peer's
  routed experts, and broadcasts the completed MoE result to the peer.

The flags are mutually exclusive and require TP size 2 plus
`--dsv41-backbone-rank`. Both preserve whole-expert EP2 and row-sharded Engram.
They change floating-point reduction order, so greedy output need not be bitwise
identical to the authority-EP topology. On a dual RTX 5090 PCIe system,
attention-only TP recovered part of full TP's prefill regression but remained
slower than authority EP; keep it as a profiling/experimentation switch.

```bash
export FREETOKEN_DSV41_CUDA_GRAPH=1
export FREETOKEN_DSV41_DECODE_REFILL_OVERLAP=1
export FREETOKEN_DSV41_FUSED_ROUTE_PREP=1
export FREETOKEN_DSV41_FUSED_DECODE_DISPATCH=1

ft serve \
  --model /path/to/DeepSeek-V4.1-Flash \
  --gpu <first-5090>,<second-5090> \
  --tp-size 2 \
  --dsv41-backbone-rank 0 \
  --max-running-requests 1 \
  --cuda-graph-max-bs 1 \
  --moe-backend offload \
  --moe-cache-auto \
  --attention-backend dsv4_sparse
```

Use the original checkpoint directory, including `inference/config.json` and
the tokenizer files. `ft checkpoint` deliberately rejects V4.1 for now because
FTW cannot yet encode the rank-sharded Engram payload.

## Qualified 64K reference profile

One dual-RTX-5090 deployment has passed exact-output, uncapped-sampling, unique
60,000-token, and 64,000-token-plus-generation gates with the following explicit
geometry:

```bash
export FREETOKEN_DSV41_CUDA_GRAPH=1
export FREETOKEN_DSV41_DECODE_REFILL_OVERLAP=1
export FREETOKEN_DSV41_FUSED_ROUTE_PREP=1
export FREETOKEN_DSV41_FUSED_DECODE_DISPATCH=1

ft serve \
  --model /path/to/DeepSeek-V4.1-Flash \
  --gpu <first-5090>,<second-5090> \
  --tp-size 2 \
  --dsv41-backbone-rank 0 \
  --max-running-requests 1 \
  --max-seq-len-override 65536 \
  --max-prefill-length 4096 \
  --num-pages 512 \
  --swa-full-tokens-ratio 0.28125 \
  --cache-type radix \
  --moe-backend offload \
  --moe-cache-sizes 704,1450 \
  --moe-prefill-hit-d2d \
  --expert-load serial \
  --attention-backend dsv4_sparse \
  --cuda-graph-max-bs 1 \
  --sampling-defaults none \
  --default-temperature 1.0 \
  --default-top-p 0.95 \
  --reasoning-parser deepseekv32 \
  --default-reasoning-effort 25
```

This is a reproducible hardware-specific reference, not a portable default. The
64K long-prefill peak left less than 100 MiB driver-visible free on the backbone
rank; use auto-sizing or requalify cache/KV geometry on other cards. The numeric
reasoning effort is also only a soft checkpoint prompt signal. It does not bound
reasoning tokens, and an omitted output limit intentionally allows generation to
continue until EOS or the remaining context boundary.

## Host-memory pinning

With the official checkpoint and an even EP2 split, each process owns about
94.42 GiB of Engram rows and 134.47 GiB of expert banks. Pinning both completely
therefore needs about 228.89 GiB of memlock per process (457.78 GiB across the
two ranks), plus modest headroom.

The loader honors the process's soft `RLIMIT_MEMLOCK`. Under a 124.44 GiB limit,
it reserves the first 94.42 GiB for Engram, pins the middle eight expert layers
(about 26.89 GiB), and keeps the other 32 expert layers pageable for CPU decode.
Pageable here still means resident host allocations, not disk offload.

For the all-pinned configuration, raise the shell or service's soft memlock to
at least 240 GiB per worker (256 GiB is a practical setting, or use unlimited)
before launch. `FREETOKEN_PIN_BUDGET_GB` can lower FreeToken's allocation policy,
but it cannot raise the operating-system limit. Confirm both values in the exact
launch environment:

```bash
ulimit -Sl
ulimit -Hl
awk '/Max locked memory/ {print}' /proc/self/limits
```

Do not treat successful allocation as a stability result. A full load makes
hundreds of GiB unreclaimable and should be followed by DIMM-temperature,
ECC/EDAC, GPU Xid and short deterministic generation checks before performance
measurements.

## Indexer prefill workspace

V4.1 prefill uses a fused Lightning Indexer kernel that reduces the 32 heads
before writing scores, avoiding the otherwise enormous `[tokens, heads,
compressed_tokens]` intermediate. The remaining fp32 `[tokens,
compressed_tokens]` logits are processed in query-row chunks and reduced to
top-k rows immediately. One chunk is capped at 512 MiB by default; set
`FREETOKEN_DSV41_INDEXER_MAX_LOGITS_MB` to a smaller positive integer when the
GPU needs tighter transient-memory headroom. This is an operator-internal
workspace limit, not a scheduler prefill-chunk setting, so it does not replay
the expert bank.

## Profiling

Set `FREETOKEN_DSV41_PROFILE=1` before starting the server to emit nested NVTX
ranges for each layer, prefill/decode attention, Engram hash/UVA
gather/all-reduce, EP broadcasts, router, shared expert, and routed expert.
The ranges do not synchronize CUDA; leave the flag unset for ordinary serving.
Because the model executes in spawned rank processes, launch the server under
Nsight first and use an interactive `nsys start`/`nsys stop` window after the
server is ready. A capture-range trigger attached only to the frontend process
will not see the rank-local ranges.
