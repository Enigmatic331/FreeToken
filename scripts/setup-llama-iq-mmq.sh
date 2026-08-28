#!/usr/bin/env bash
# Prepare the exact llama.cpp build validated by FreeToken's optional IQ MMQ bridge.
set -euo pipefail

PINNED_REVISION="6fdd0ac8907fd973a42b876357823ad2124cd8ed"
UPSTREAM="https://github.com/ggml-org/llama.cpp.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
FREETOKEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
CHECKOUT="${1:-${FREETOKEN_LLAMA_CPP_DIR:-$FREETOKEN_ROOT/.deps/llama.cpp}}"
CUDA_ARCHITECTURES="${FREETOKEN_LLAMA_CUDA_ARCHITECTURES:-120}"
BUILD_JOBS="${FREETOKEN_LLAMA_BUILD_JOBS:-$(nproc)}"

if [[ -e "$CHECKOUT" && ! -d "$CHECKOUT/.git" ]]; then
  printf 'error: %s exists but is not a llama.cpp git checkout\n' "$CHECKOUT" >&2
  exit 1
fi

if [[ ! -d "$CHECKOUT/.git" ]]; then
  git clone --filter=blob:none "$UPSTREAM" "$CHECKOUT"
fi

if [[ -n "$(git -C "$CHECKOUT" status --porcelain --untracked-files=no)" ]]; then
  printf 'error: refusing to change a llama.cpp checkout with tracked modifications: %s\n' "$CHECKOUT" >&2
  exit 1
fi

if ! git -C "$CHECKOUT" cat-file -e "$PINNED_REVISION^{commit}" 2>/dev/null; then
  git -C "$CHECKOUT" fetch origin "$PINNED_REVISION"
fi
git -C "$CHECKOUT" checkout --detach "$PINNED_REVISION"

cmake -S "$CHECKOUT" -B "$CHECKOUT/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_NCCL=ON
cmake --build "$CHECKOUT/build" --parallel "$BUILD_JOBS"

printf '\nPinned llama.cpp IQ MMQ build is ready. Enable it with:\n'
printf 'export FREETOKEN_LLAMA_CPP_DIR=%q\n' "$CHECKOUT"
