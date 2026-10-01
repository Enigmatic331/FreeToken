#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${FREETOKEN_PYTHON:?set FREETOKEN_PYTHON to the virtual-environment Python executable}"
: "${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4.1 checkpoint directory}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the four qualified device IDs}"

mode=${DSPARK_EXPERIMENT_MODE:-fixed}
extra_args=()
case "$mode" in
  fixed)
    verify_length=${DSPARK_VERIFY_LENGTH:-4}
    extra_args+=(--dspark-verification-length "$verify_length")
    ;;
  adaptive)
    if [[ -z ${DSPARK_COSTS_MS:-} ]]; then
      echo "DSPARK_COSTS_MS is required for adaptive mode" >&2
      exit 2
    fi
    extra_args+=(--dspark-adaptive-verification --dspark-adaptive-costs-ms "$DSPARK_COSTS_MS")
    ;;
  profile)
    if [[ -z ${DSPARK_PROFILE_SCHEDULE:-} ]]; then
      echo "DSPARK_PROFILE_SCHEDULE is required for profile mode" >&2
      exit 2
    fi
    extra_args+=(--dspark-verification-schedule "$DSPARK_PROFILE_SCHEDULE")
    ;;
  *)
    echo "DSPARK_EXPERIMENT_MODE must be fixed, profile, or adaptive" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${PYTHONPATH:-$repo_dir/python}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Speculative verification introduces target batch shapes that may load a Triton
# module for the first time after NCCL kernels are already in flight.  CUDA lazy
# loading can then context-synchronize behind a collective waiting for the other
# ranks, producing the documented lazy-module deadlock.  Load modules eagerly at
# startup for this candidate so no rank enters cuModuleLoadData mid-collective.
export CUDA_MODULE_LOADING=EAGER
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FREETOKEN_LOAD_VISION=1
export FREETOKEN_KERNEL_CACHE_DIR="${FREETOKEN_KERNEL_CACHE_DIR:-/tmp/freetoken-dsv41-kernel-cache}"
export TVM_FFI_CACHE_DIR=/tmp/freetoken-dsv41-ep3-dspark-adaptive-tvm-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=4
export LOG_PID=1
export FREETOKEN_DSV41_INDEXER_MAX_LOGITS_MB=512
export FREETOKEN_DSV41_CUDA_GRAPH=1
export FREETOKEN_DSV41_DSPARK_CUDA_GRAPH_LENGTHS="${DSPARK_CUDA_GRAPH_LENGTHS:-4}"
export FREETOKEN_DSV41_DECODE_REFILL_OVERLAP=1
export FREETOKEN_DSV41_FUSED_ROUTE_PREP=1
export FREETOKEN_DSV41_FUSED_DECODE_DISPATCH=1
cuda_graph_max_bs=${DSPARK_CUDA_GRAPH_MAX_BS:-0}

exec "$FREETOKEN_PYTHON" -m freetoken.cli serve \
  --model "$MODEL_PATH" \
  --served-model-name "${DSPARK_SERVED_MODEL_NAME:-DeepSeek (Experimental)}" \
  --host "${DSPARK_HOST:-127.0.0.1}" \
  --port "${DSPARK_PORT:-1919}" \
  --gpu "${DSPARK_TEXT_GPUS:-0,1,2}" \
  --tp-size 3 \
  --disable-pynccl \
  --rank-local-cuda-visibility \
  --dsv41-backbone-rank 0 \
  --dsv41-expert-shards 136,136,112 \
  --dsv41-prefill-expert-shards 192,192,0 \
  --dsv41-expert-storage-ranges 0:192,136:248,272:112 \
  --dsv41-engram-ranks 0,1 \
  --vision-device "${DSPARK_AUX_DEVICE:-3}" \
  --speculative-dspark \
  --dspark-device "${DSPARK_AUX_DEVICE:-3}" \
  --dspark-fallback-acceptance 0 \
  --dspark-fallback-min-drafted 16 \
  --dspark-fallback-steps 64 \
  --max-running-requests 1 \
  --max-seq-len-override 262144 \
  --max-prefill-length 8192 \
  --num-pages 2048 \
  --swa-full-tokens-ratio 0.28125 \
  --memory-ratio 0.90 \
  --cache-type radix \
  --moe-backend offload \
  --moe-cache-sizes 512,1472,1650 \
  --moe-prefill-hit-d2d \
  --expert-load serial \
  --attention-backend dsv4_sparse \
  --cuda-graph-max-bs "$cuda_graph_max_bs" \
  --sampling-defaults none \
  --default-temperature 1.0 \
  --default-top-p 0.95 \
  --reasoning-parser deepseekv32 \
  --default-reasoning-effort 25 \
  --decode-log-interval 8 \
  "${extra_args[@]}"
