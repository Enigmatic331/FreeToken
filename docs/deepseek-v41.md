# DeepSeek-V4.1-Flash experimental baseline

FreeToken's first DeepSeek-V4.1 gate serves ordinary text generation directly
from the official safetensors checkpoint. It implements the main 40-layer model,
CSA2 paged attention, DS-FP4 routed experts and the two row-sharded Engram tables.
Vision and DSpark/MTP speculative decoding are intentionally outside this gate.

The current topology is exactly two ranks: rank 0 owns the complete TP1 text
backbone, while both ranks own 192 routed experts per layer and half of every
Engram table. Decode is eager and single-stream while Engram's three-token
history is still prepared from live request state.

```bash
ft serve \
  --model /path/to/DeepSeek-V4.1-Flash \
  --gpu <first-5090>,<second-5090> \
  --tp-size 2 \
  --dsv41-backbone-rank 0 \
  --moe-backend offload \
  --moe-cache-auto \
  --attention-backend dsv4_sparse
```

Use the original checkpoint directory, including `inference/config.json` and
the tokenizer files. `ft checkpoint` deliberately rejects V4.1 for now because
FTW cannot yet encode the rank-sharded Engram payload.

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
