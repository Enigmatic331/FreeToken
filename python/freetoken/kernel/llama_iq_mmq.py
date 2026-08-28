"""Opt-in bridge to llama.cpp's grouped IQ matrix kernels.

Set ``FREETOKEN_LLAMA_CPP_DIR`` to a built llama.cpp checkout.  This is kept
optional while the path is experimental: normal FreeToken installs retain the
portable vendored MMVQ implementation and acquire no new runtime dependency.
"""

from __future__ import annotations

import functools
import hashlib
import os
import pathlib
import shutil

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "llama_iq_mmq" / "llama_iq_mmq.cu"
_SUPPORTED_TYPES = {18, 19}  # GGML_IQ3_XXS, GGML_IQ1_S
_MAX_SORT_TOKENS = 4096


def configured() -> bool:
    return bool(os.environ.get("FREETOKEN_LLAMA_CPP_DIR"))


def supported(quant_type: int, tokens: int) -> bool:
    """Whether the experimental prefill path should handle this operation."""
    return configured() and int(quant_type) in _SUPPORTED_TYPES and tokens > 6


def _checkout() -> pathlib.Path:
    value = os.environ.get("FREETOKEN_LLAMA_CPP_DIR")
    if not value:
        raise RuntimeError("FREETOKEN_LLAMA_CPP_DIR is not set")
    root = pathlib.Path(value).expanduser().resolve()
    required = (
        root / "ggml" / "include" / "ggml.h",
        root / "ggml" / "src" / "ggml-cuda" / "mmq.cuh",
        root / "build" / "bin" / "libggml-cuda.so",
        root / "build" / "bin" / "libggml.so",
        root / "build" / "bin" / "libggml-base.so",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "FREETOKEN_LLAMA_CPP_DIR must point to a built llama.cpp checkout; "
            f"missing: {', '.join(missing)}"
        )
    return root


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    root = _checkout()
    libdir = root / "build" / "bin"
    # Include the library metadata in the extension name so rebuilding/updating
    # llama.cpp cannot silently reuse an adapter compiled for an older ABI.
    stamp_source = "|".join(
        f"{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
        for path in (_CSRC, libdir / "libggml-cuda.so")
    )
    stamp = hashlib.sha256(stamp_source.encode()).hexdigest()[:12]
    extra_cuda_cflags = [
        "-O3",
        "--expt-relaxed-constexpr",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    # Match the compiler selection used by the regular GGUF extension.  Newer
    # distro GCC releases can be ahead of the CUDA/libtorch support window.
    from freetoken.kernel.gguf import _c_compiler_for, _host_compiler

    host_cxx = _host_compiler()
    if host_cxx is not None:
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)
    return load(
        name=f"freetoken_llama_iq_mmq_{stamp}",
        sources=[str(_CSRC)],
        extra_include_paths=[
            str(root / "ggml" / "include"),
            str(root / "ggml" / "src"),
            str(root / "ggml" / "src" / "ggml-cuda"),
        ],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=[
            f"-L{libdir}",
            "-lggml-cuda",
            "-lggml",
            "-lggml-base",
            f"-Wl,-rpath,{libdir}",
        ],
        verbose=True,
    )


def grouped_iq_mmq(
    weight: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type: int,
    rows: int,
) -> torch.Tensor:
    """Run grouped IQ MMQ and restore FreeToken's activation dtype."""
    if int(quant_type) not in _SUPPORTED_TYPES:
        raise ValueError(f"unsupported llama IQ MMQ type {quant_type}")
    output_dtype = x.dtype
    x_f32 = x.float().contiguous()
    ids_i32 = topk_ids.to(dtype=torch.int32).contiguous()
    module = _module()
    routes_per_token = ids_i32.shape[1]
    out = torch.empty(
        (x_f32.shape[0] * routes_per_token, rows),
        dtype=torch.float32,
        device=x.device,
    )
    for start in range(0, x_f32.shape[0], _MAX_SORT_TOKENS):
        end = min(start + _MAX_SORT_TOKENS, x_f32.shape[0])
        route_start = start * routes_per_token
        route_end = end * routes_per_token
        module.grouped_iq_mmq_out(
            weight,
            x_f32[start:end],
            ids_i32[start:end],
            int(quant_type),
            rows,
            out[route_start:route_end],
        )
    return out.to(output_dtype)


__all__ = ["configured", "grouped_iq_mmq", "supported"]
