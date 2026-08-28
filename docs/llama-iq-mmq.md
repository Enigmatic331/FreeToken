# Optional llama.cpp IQ MMQ bridge

FreeToken can use modern llama.cpp grouped IQ matrix-matrix kernels for supported
GGUF prefill projections. Model loading, expert caching, pipeline parallelism,
attention, KV cache, and serving remain inside FreeToken. Single-token decode
continues to use FreeToken's MMVQ path.

This bridge is experimental and opt-in. It is validated against exactly:

```text
llama.cpp 6fdd0ac8907fd973a42b876357823ad2124cd8ed
```

Prepare a checkout and CUDA build with:

```bash
scripts/setup-llama-iq-mmq.sh /path/to/llama.cpp
export FREETOKEN_LLAMA_CPP_DIR=/path/to/llama.cpp
```

The helper defaults to CUDA architecture 120 for RTX 50-series systems. Override
`FREETOKEN_LLAMA_CUDA_ARCHITECTURES` when building for another GPU architecture.
It refuses to replace tracked local llama.cpp changes.

At runtime, FreeToken verifies the checkout's Git revision before compiling its
small adapter. A different revision is rejected because llama.cpp's internal CUDA
interfaces are not a stable ABI. Unset `FREETOKEN_LLAMA_CPP_DIR` to retain the
normal vendored MMVQ behavior and avoid the external dependency entirely.
