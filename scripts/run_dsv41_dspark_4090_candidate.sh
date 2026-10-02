#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${FREETOKEN_PYTHON:?set FREETOKEN_PYTHON to the virtual-environment Python executable}"
: "${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4.1 checkpoint directory}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the qualified device UUIDs}"

export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export CUDA_MODULE_LOADING=EAGER
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FREETOKEN_LOAD_VISION=1
export FREETOKEN_KERNEL_CACHE_DIR="${FREETOKEN_KERNEL_CACHE_DIR:-/home/enigmatic331/models/dsv41-flash-results/kernel-cache-sm89-sm120}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-/tmp/freetoken-dsv41-dspark-4090-tvm-cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=4
export LOG_PID=1
export FREETOKEN_DSV41_INDEXER_MAX_LOGITS_MB=256
export FREETOKEN_DSV41_CUDA_GRAPH=1
export FREETOKEN_DSV41_DSPARK_CUDA_GRAPH_LENGTHS="${DSPARK_VERIFY_LENGTH:-4}"
export FREETOKEN_DSV41_DSPARK_DRAFT_CUDA_GRAPH=1
export FREETOKEN_DSV41_DECODE_REFILL_OVERLAP=1
export FREETOKEN_DSV41_FUSED_ROUTE_PREP=1
export FREETOKEN_DSV41_FUSED_DECODE_DISPATCH=1
export FREETOKEN_EP_PREFILL_ROUTE_TILE_TOKENS=4096

exec "$FREETOKEN_PYTHON" -m freetoken.cli serve \
  --model "$MODEL_PATH" \
  --served-model-name "${DSPARK_SERVED_MODEL_NAME:-DeepSeek (Experimental)}" \
  --host "${DSPARK_HOST:-172.17.0.1}" \
  --port "${DSPARK_PORT:-1919}" \
  --gpu 0,1,2 \
  --tp-size 3 \
  --disable-pynccl \
  --rank-local-cuda-visibility \
  --rank-local-cuda-peer-visibility \
  --dsv41-backbone-rank 0 \
  --dsv41-expert-shards 32,160,192 \
  --dsv41-prefill-expert-shards 128,160,96 \
  --dsv41-expert-storage-ranges 0:128,32:256,192:192 \
  --dsv41-engram-ranks 0,1 \
  --vision-device 3 \
  --speculative-dspark \
  --dspark-device 3 \
  --dspark-verification-length "${DSPARK_VERIFY_LENGTH:-4}" \
  --dspark-fallback-acceptance "${DSPARK_FALLBACK_ACCEPTANCE:-0.52}" \
  --dspark-fallback-min-drafted "${DSPARK_FALLBACK_MIN_DRAFTED:-16}" \
  --dspark-fallback-steps "${DSPARK_FALLBACK_STEPS:-64}" \
  --dspark-fallback-cumulative \
  --max-running-requests 1 \
  --max-seq-len-override 524288 \
  --max-prefill-length 8192 \
  --num-pages 4096 \
  --swa-full-tokens-ratio 0.28125 \
  --memory-ratio 0.90 \
  --cache-type radix \
  --moe-backend offload \
  --moe-cache-sizes 256,1472,2350 \
  --moe-prefill-hit-d2d \
  --expert-load serial \
  --attention-backend dsv4_sparse \
  --cuda-graph-max-bs 1 \
  --sampling-defaults none \
  --default-temperature 1.0 \
  --default-top-p 0.95 \
  --reasoning-parser deepseekv32 \
  --default-reasoning-effort 25 \
  --decode-log-interval 8
