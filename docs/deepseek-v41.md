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

## Selective short prefill

`--moe-sparse-prefill-max-tokens N` opts DS-FP4 EP into loading only the experts
routed by an ordinary prefill of at most `N` new tokens. This includes a short
uncached suffix of a much longer radix-cached conversation. The normal prefill
GEMV/grouped-GEMM crossover, 32-element activation quantization, phase ownership,
and exact route reduction remain unchanged; only weight movement uses LRU slots.
Large slot-cache dimensions use the generic Triton route sorter. Startup warms
the new local modules before prefill collectives can be posted. Synthetic model
warmup uses the reserved, already mapped KV page rather than unallocated slots.

The generic default is zero (disabled). The heterogeneous launcher exposes
`DSPARK_SPARSE_PREFILL_MAX_TOKENS`; qualify a threshold on the target topology
using exact cold and cached-prefix fixtures before enabling it. Above the
threshold the existing overlapped whole-layer path is retained. This does not
change the scheduler's maximum prefill chunk, KV capacity, or MTP verification
path. It is not intended as a replacement for large-chunk streaming prefill.

## Other execution optimizations

`FREETOKEN_DSV41_FP8_DECODE_TUNING=1` opts SM120 single-row decode into
V4.1-specific FP8 launch layouts and loop unrolling. It retains the checkpoint's
native E4M3/E8M0 bytes, 32x32 scales, activation quantizer, split-K boundaries and
ordered FP32 accumulation. Other GPU architectures, unknown shapes, emulated FP8
and multi-row prefill use the existing implementation. Set the flag before
startup warmup and graph capture; restart to change it.

An isolated traversal of all 290 native dense projections (5.49 GB of weights)
matched 130,252,800 FP32-partial/BF16-output values over 16 inputs per projection
and reduced summed kernel time from 9.41 to 5.33 ms on an RTX 5090. This is not
an end-to-end serving speedup: shared-expert overlap can hide part of that
saving. The switch remains off by default and requires exact serving outputs,
full-context prefill, native vision and idle/wake qualification on each deployment.
On one qualified two-RTX-5090 plus 48 GiB RTX-4090 CPU-assisted EP3 deployment,
three matched-output rounds improve mean relative decode speed by 8.80 percent
over its target-only CPU control. The 524,288-token context, 8,192-token chunks,
expert caches and native quantization are unchanged; a fresh 520K prompt and the
full serving/vision/cancellation/idle gates pass. This hardware-specific result
does not imply the same benefit on other GPUs or expert placements.

`FREETOKEN_DSV41_SKIP_ZERO_ROUTES=1` separately opts SM120/SM89 single-row,
32-value-activation-group DS-FP4 decode into skipping zero-weight local routes.
EP routes owned by peers and hybrid routes assigned to the CPU already carry
zero GPU weights. Both gate/up and down write explicit zero outputs for those
routes, including when route ownership changes during CUDA graph replay. Active
routes retain the original FP4 lookup, scales, activation quantization and
accumulation order; no extra expert bank or tensor workspace is allocated.
Multi-row prefill and the default-off path retain their existing kernels.
Set the switch before startup and qualify exact serving outputs, full-context
capacity and the deployed CPU/GPU topology before enabling it in production.
On the same heterogeneous deployment, this second switch passes exact outputs,
520K and operational gates but adds only 2.53 percent mean relative decode speed
over FP8 tuning. It remains disabled there because that trial's predeclared
promotion threshold was three percent; isolated expert-kernel gains must not be
presented as whole-model speedups.

V4.1's native DS-FP4 banks can also feed the experimental `--moe-backend hybrid`
CPU-overflow path. The CPU executor must be rebuilt with support for V4.1's
32-value FP8 activation groups; an older extension is rejected rather than
silently applying V4's 128-value groups. `--moe-hybrid-ranks` can exclude the
backbone authority, and distributed CPU pools receive disjoint physical-core
affinities. `--moe-cpu-threads` is a per-participating-rank worker count.

Linux distributed serving can optionally set `FREETOKEN_RANK_MAIN_CPUS` to a
CPU list such as `56-82,84-110`. This limits only each rank's calling main thread;
it does not repin existing expert workers, coordinators or I/O helpers. The list
is validated against the inherited allowed mask and existing CPU-pool allocation
before model initialization, then applied after scheduler initialization and the
all-rank startup barrier, before readiness. Leaving it unset preserves the
original behavior. Exact logical-CPU overlap with workers/coordinators is rejected,
but physical-core sharing through SMT siblings is allowed: this is not a core
reservation, CPU quota, NUMA policy or governor change. Select CPUs for the actual
host topology rather than copying this example blindly. Helpers created later can
inherit the main mask, so qualify cold startup, sustained decode before and after
long prefill, native vision and idle/wake before production use. No idle polling
or recurring affinity updates are introduced.

Alternatively, `FREETOKEN_RANK_MAIN_EXCLUDE_COORDINATORS=1` derives a narrower
main-thread mask by removing only the CPU pools' coordinator logical CPUs from
the inherited mask. Worker CPUs and coordinator SMT siblings remain available.
It uses the same pre-initialization pool partition and late application point;
it does not move workers, reserve physical cores, or change existing helpers.
This switch requires distributed CPU pools, defaults to `0`, and is mutually
exclusive with `FREETOKEN_RANK_MAIN_CPUS`. It is an opt-in qualification candidate,
not a claim that all historical throughput drift came from coordinator overlap.

The explicit SMT-heavy mask remains experimental: a local four-rank cold-start trial preserved exact
outputs and sustained decode speed, but failed its 17.5K prefill performance
gate. It was rolled back before full-capacity qualification. Do not treat the
example mask as a qualified production policy or a proven fix for performance
drift.

The narrower coordinator-only mode subsequently passed exact decode/artifact/
vision, matched cold 17.5K and 520K prefill, post-capacity sustained decode and
service operations. It was nevertheless **not promoted**: a corrected PCIe
BadTLP on an auxiliary GPU link failed the clean-hardware gate. A prior event on
that link predates the affinity patch. Original placement was restored; this
mode remains opt-in and must not be advertised as a fully qualified production
policy or as a substantial throughput improvement.

`--moe-hybrid-max-bs 1` limits CPU assistance to ordinary one-token decode while
larger speculative verification batches retain GPU offload and refill overlap.
Prefill and KV allocation are unchanged. DS-FP4 merges individual weighted BF16
routes before the original route reduction rather than summing independently
rounded CPU/GPU partials. This experimental backend needs numerical, semantic,
artifact-execution, and throughput qualification before promotion.
The generic and heterogeneous launcher defaults remain GPU offload.

The V4.1 CPU path additionally requires the GPU-order DS-FP4 extension: its
AVX2/FMA eight-output-row kernel preserves the decode GEMV accumulation order
at the native 5120/2304 geometry. An older extension is rejected explicitly.
`FREETOKEN_CPU_MOE_MAPPED_HANDOFF=1` opts into conditional mapped-pinned
single-token submission, avoiding coordinator dispatch when there is no CPU
overflow. Completion uses CUDA stream memory operations, not an SM polling
kernel; larger batches retain the DMA handoff. Raising the hybrid batch ceiling
to include speculative verification warms every enabled target row geometry
locally before the startup collective barrier. These numerical and startup
guards do not establish a throughput gain; retain the full matched-output A/B.

Two additional switches are opt-in and default off:

- `FREETOKEN_DSV41_SKIP_EMPTY_CACHE=1` guards one-token cache admission on all
  V4.1 text ranks. When a rank owns no nonzero route, it schedules no expert
  copy and does not refresh a dummy expert's LRU state. It also masks zero-weight
  GPU reads, because an empty cache's placeholder slot has no valid payload.
  Output zeros and collective participation are preserved; prefill is unchanged.
  Startup explicitly loads the guarded admission kernel locally before any EP
  collective, including when CUDA module loading is lazy.
- `FREETOKEN_DSV41_CPU_FUSE_PREP=1` requires a rebuilt GPU-order CPU extension.
  For one-token, 32-value activation groups, each gate/up tile also performs its
  FP8 round-trip and deinterleave. This removes the separate per-route preparation
  pass and one worker barrier, with the same BF16 rounding and accumulation.
  Larger batches keep the original preparation schedule.
- `FREETOKEN_DSV41_CPU_COMPACT_ACTIVE_ROUTES=1` compacts only the first-stage
  one-token CPU queue to routes actually assigned to the CPU. Original route
  positions, scratch/output addresses, the down-pass schedule, BF16 rounding,
  and route reduction order are unchanged. Larger batches retain the original
  queue. The switch requires a rebuilt GPU-order CPU extension.

These switches change different parts of execution and should be qualified
independently before combination. No throughput gain is implied by enabling them.

On the four-GPU EP4 production geometry, active-route queue compaction preserved
all exact outputs but regressed the four-workload decode mean by 0.52%. It is
therefore retained only as a default-off experimental path, not a recommended
production setting.
On the October 3 two-5090/48-GiB-4090 single-miss admission trial, one warmup and
three measured rounds gave +0.42% mean relative decode speed for the empty-rank
guard, +1.93% for CPU preparation fusion, and +2.93% together. All four workloads
preserved their exact 255-token outputs. The combined path used 7.28% less
serving-process CPU time across the matched gates, including prefill and serving
overhead; this is not isolated expert-compute time or an energy measurement.
The combination narrowly missed that trial's predeclared 3% cutoff, so its fresh
520K and full operational gates were not run in that trial and it was not promoted. Fixed-order
separate-process runs show a useful cumulative candidate, not proof of a specific
CPU/GPU interaction or an MTP speedup.

A subsequent user-approved qualification retained that historical rejection and
predeclared acceptance of a positive mean gain with no workload worse than -1%,
subject to all correctness, capacity and operational gates. One warmup and three
new measured rounds against the pinned single-miss control reproduced +2.93%:
coding/science/Malay/planning reached 28.102/29.523/32.203/30.317 token/s, with
all actual 255-token outputs exact. Fresh uncached 520K prefill passed at 426.673
seconds TTFT with unchanged 8K chunks; all ten operational gates passed, including
native vision, cancellation/recovery and settled-idle/exact-wake. The combination
was then selected for that local production deployment without restarting its
qualified live process. Repository defaults remain off. This is an incremental
gain over single-miss admission, not a new matched A/B against the older FP8-only
profile, a bulk-prefill gain, or a decode-rate claim at 520K context. Minimum
observed authority driver-free memory was only 13 MiB; future changes still need
the full capacity gate.

`FREETOKEN_DSV41_BACKGROUND_ADMISSION=1` additionally opts V4.1 expert workers
into CPU computation of a single cache miss while a CUDA side stream fills its
GPU cache row. It requires native DS-FP4 hybrid decode and the exact GPU-order
CPU executor. GPU hits execute concurrently; multiple misses retain the existing
fetch-plus-CPU-overflow policy. The authority, prefill and multi-row verification
retain their existing paths. Cache admission and eviction are unchanged, so the
policy hides transfer latency rather than reducing transferred weight bytes.

The copy stream is fenced before subsequent cache operations and before a worker
forward returns, including errors. Pinned-source replacement waits before freeing
the old inputs. Zero-weight GPU routes explicitly skip reads of dummy slots that
may be copy destinations; this is enforced locally even when the separate global
zero-route switch is off. Settled idle submits no background polling work.
Set the flag before local startup warmup and graph capture; restart to change it.

On the same heterogeneous deployment, one warmup and three measured rounds match
all actual 255-token outputs and improve mean relative decode speed by 7.92 percent
over the FP8-tuned CPU control. Coding/science/Malay/planning reach
27.545/28.971/31.266/29.204 token/s. Fresh uncached 520K prefill and the full
serving/vision/cancellation/idle/wake gates pass with unchanged 8K chunks, caches
and native quantization. This does not establish 30 token/s across workloads,
the same decode rate at 520K context, or benefits from buying more CPU cores.
The switch remains off by default and requires deployment-specific qualification.

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

## Routed prefill projection workspace

Grouped DS-FP4 prefill reserves one contiguous workspace sized for the larger of
gate/up and down, then reuses it after SwiGLU produces independent activations.
This preserves both projection kernels and their accumulation order. In Flash,
`H=5120` is wider than `2I=4608`: a 4K tile with six routes needs a 240 MiB down
output, while the smaller gate/up output is 216 MiB. Releasing the smaller tensor
cannot guarantee that a fragmented allocator can satisfy the larger allocation.
Preallocation avoids that resize without reducing scheduler chunks, route tiles,
expert caches or KV capacity. Exact GPU regressions compare against separate
buffers at 128/4096 rows on SM89 and SM120, including both width orders.

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

`FREETOKEN_DSV41_COMPACT_PREFILL_CANDIDATES=1` is an experimental, default-off
prefill-only workspace option. It retains hierarchical candidate selections as
one boolean per candidate block, expanding only the bounded logits row tile for
each consumer. Selection math, decode/verification publications and scheduler
chunks are unchanged. For block size 8 at 8,192 query rows by 520,000 keys, the
published tensor shrinks from 3.967 to 0.496 GiB. This is tensor-size arithmetic,
not a measured total peak-memory saving or a serving-performance claim. Keep it
off until exact full-model, long-context and operational qualification passes.

## Native BF16 output head

`FREETOKEN_DSV41_NATIVE_BF16_HEAD=1` is a default-off storage/kernel option for
ordinary target-only generation. It retains the released BF16 head values
instead of expanding them to FP32, saving 1,262.5 MiB of parameter storage for
Flash. It uses CUDA `torch.mm(..., out_dtype=torch.float32)`, preserving FP32
logits without changing global matmul precision settings. Qualification used
PyTorch 2.11 on an SM120 authority; other stacks require separate testing.

The one-row primitive passed real-checkpoint numerical and graph-replay gates;
the batched primitive did not. Multiple rows therefore execute independently
and may be slower. The option explicitly rejects MTP/DSpark and rejects FP32
hidden states rather than silently rounding them. Keep it disabled for MTP.
The default path remains the original FP32 linear projection. Correctness,
capacity, vision and operational qualification remain mandatory before enabling
it on another placement; exact tested answers do not prove universal bitwise
equivalence of logits or arbitrary future generations.

## Phase-local cache replacement priority

`FREETOKEN_DSV41_PHASE_CACHE_PRIORITY=1` is a default-off option for the
qualified native DS-FP4, phase-aware EP4, single-request, target-only path.
After prefill and before the first decode, it lowers the replacement priority
of occupied entries outside the rank's decode ownership. Storage-local ranges
are derived from the actual execution plan; the option does not change cache
budgets, weights, ownership maps, recency or miss counters.

Background admission and prefill copy streams are joined before the update.
The small operation is warmed before graph capture and runs once per phase
transition outside capture, not on every replay or during idle. MTP and other
unsupported geometries are rejected. On one two-5090/4090/4080 SUPER EP4 setup,
short exact-output fixtures improved by about 2.7% against the saved control;
longer operational observations were essentially flat. This is not a universal
long-answer gain. Requalify full context, output correctness, cancellation,
vision and idle/wake behavior before enabling it elsewhere.

## Worker-only Engram prefill scratch

`FREETOKEN_DSV41_WORKER_ENGRAM_SCRATCH=1` is an experimental, default-off
memory-lifetime option. Only a participating EP4 expert worker with two native
Engram tables and a two-rank PyTorch NCCL subgroup shares the eager-prefill
gather/all-reduce buffer. The authority still retains independent table
outputs. Decode and CUDA-graph staging use the original path.

The worker discards the reconstructed rows, so sequential prefill lookups can
reuse one high-watermark BF16 allocation. Cross-stream reuse is fenced and
allocator stream use recorded; pending asynchronous lookups and graph capture
are rejected on the shared path. No gather math, collective order, table
placement or host weights change. At 8,192 tokens, the two native eager
buffers occupy 192 MiB; one occupies 96 MiB. A two-GPU synthetic native-format
UVA/NCCL lab verifies that live-tensor difference and exact outputs, but this
does not establish a full-model peak saving or qualify larger chunks. Keep the
option off until the native-checkpoint capacity and service gates pass on the
intended placement. It performs no idle work and is not a decode-speed claim.

## Profiling

Set `FREETOKEN_DSV41_PROFILE=1` before starting the server to emit nested NVTX
ranges for each layer, prefill/decode attention, Engram hash/UVA
gather/all-reduce, EP broadcasts, router, shared expert, and routed expert.
The ranges do not synchronize CUDA; leave the flag unset for ordinary serving.
Because the model executes in spawned rank processes, launch the server under
Nsight first and use an interactive `nsys start`/`nsys stop` window after the
server is ready. A capture-range trigger attached only to the frontend process
will not see the rank-local ranges.
