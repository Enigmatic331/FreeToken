// Optional adapter for llama.cpp's grouped IQ MMQ kernels.
//
// This deliberately builds against a user-supplied llama.cpp checkout rather than
// copying its fast-moving CUDA implementation into FreeToken.  The Python wrapper
// only enables it when FREETOKEN_LLAMA_CPP_DIR is set.
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/extension.h>

#include "common.cuh"
#include "ggml-backend-impl.h"
#include "mmid.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

static ggml_backend_cuda_context& backend_context(const int device) {
  static std::vector<ggml_backend_t> backends(16, nullptr);
  TORCH_CHECK(device >= 0 && device < static_cast<int>(backends.size()), "unsupported CUDA device index");
  if (backends[device] == nullptr) {
    backends[device] = ggml_backend_cuda_init(device);
    TORCH_CHECK(backends[device] != nullptr, "llama.cpp CUDA backend initialization failed");
  }
  return *static_cast<ggml_backend_cuda_context*>(backends[device]->context);
}

template <ggml_type type>
static torch::Tensor grouped_iq_mmq_impl(
    torch::Tensor weight,
    torch::Tensor x,
    torch::Tensor topk_ids,
    const int64_t rows,
    torch::Tensor out) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda() && topk_ids.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be uint8");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32, "topk_ids must be int32");
  TORCH_CHECK(weight.device() == x.device() && x.device() == topk_ids.device(), "inputs must share a device");
  TORCH_CHECK(out.is_cuda() && out.device() == x.device(), "out must share the CUDA device");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32 && out.is_contiguous(), "out must be contiguous float32");
  TORCH_CHECK((weight.dim() == 2 || weight.dim() == 3) && x.dim() == 2 && topk_ids.dim() == 2,
              "invalid input ranks");
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous() && topk_ids.is_contiguous(), "inputs must be contiguous");
  if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(1) == rows, "weight row count does not match rows");
  }
  TORCH_CHECK(topk_ids.size(0) == x.size(0), "topk_ids token count does not match x");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const int device = x.get_device();
  ggml_backend_cuda_context& ctx = backend_context(device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();

  const int64_t experts = weight.size(0);
  const int64_t cols = x.size(1);
  const int64_t tokens = x.size(0);
  const int64_t top_k = topk_ids.size(1);
  const int64_t routes = tokens * top_k;
  TORCH_CHECK(top_k > 0 && routes > 0, "empty routes are not supported");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == routes && out.size(1) == rows,
              "out shape does not match routed output");
  const int64_t cols_padded = GGML_PAD(cols, MATRIX_ROW_PADDING);
  const int64_t q_row_bytes = cols_padded * sizeof(block_q8_1) / QK8_1;

  // Current upstream llama.cpp's short final MMQ tile can read one tile beyond
  // the logical route arrays.  Zero tail padding keeps that speculative read in
  // bounds; remove this once the upstream tail fix is available in our baseline.
  constexpr int64_t tail = 128;
  const int64_t qbytes = routes * q_row_bytes + tail * sizeof(block_q8_1_mmq);
  auto byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
  auto q8 = torch::zeros({qbytes}, byte_options);
  auto ids_src1 = torch::zeros({routes + tail}, int_options);
  auto ids_dst = torch::zeros({routes + tail}, int_options);
  auto expert_bounds = torch::empty({experts + 1}, int_options);

  // Router top-k is unique by construction.  write_inverse=true lets llama.cpp
  // quantize each activation once, then scatter it into expert-contiguous rows.
  ggml_cuda_launch_mm_ids_helper(
      topk_ids.data_ptr<int32_t>(), ids_src1.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),
      expert_bounds.data_ptr<int32_t>(), experts, tokens, top_k,
      /*nchannels_y=*/1, /*si1=*/top_k, /*sis1=*/1, /*write_inverse=*/true, stream);
  quantize_scatter_mmq_q8_1_cuda(
      x.data_ptr<float>(), ids_src1.data_ptr<int32_t>(), q8.data_ptr(), type,
      cols, /*stride_token=*/cols, cols_padded, tokens, routes, top_k, stream);

  // Variable-bit GGUF banks reserve every expert at the maximum byte stride used
  // by any layer.  Rows inside that reservation are still tightly packed at the
  // canonical size for this layer's quant type.
  const int64_t packed_row_bytes = ggml_row_size(type, cols);
  TORCH_CHECK(weight.stride(0) * weight.element_size() >= rows * packed_row_bytes,
              "expert cache stride is smaller than the packed layer");
  TORCH_CHECK(packed_row_bytes % ggml_type_size(type) == 0, "invalid packed weight row stride");
  const int64_t weight_row_stride = packed_row_bytes / ggml_type_size(type);
  TORCH_CHECK(weight.stride(0) * weight.element_size() % ggml_type_size(type) == 0,
              "expert cache stride is not aligned to the quant block size");
  const int64_t weight_expert_stride = weight.stride(0) * weight.element_size() / ggml_type_size(type);
  const int64_t q_row_stride_ints = q_row_bytes / sizeof(int);
  const mmq_args args = {
      reinterpret_cast<const char*>(weight.data_ptr()), type,
      reinterpret_cast<const int*>(q8.data_ptr()), ids_dst.data_ptr<int32_t>(),
      expert_bounds.data_ptr<int32_t>(), out.data_ptr<float>(), nullptr,
      cols, rows, routes, weight_row_stride, routes, rows,
      experts, experts, weight_expert_stride, q_row_stride_ints, rows * top_k,
      1, 1, weight_expert_stride * experts, q_row_stride_ints * tokens, rows * routes,
      tokens,
  };
  mul_mat_q_case<type>(ctx, args, stream);
  return out;
}

static torch::Tensor grouped_iq_mmq(
    torch::Tensor weight,
    torch::Tensor x,
    torch::Tensor topk_ids,
    const int64_t type,
    const int64_t rows) {
  auto out = torch::empty({x.size(0) * topk_ids.size(1), rows}, x.options());
  switch (type) {
    case GGML_TYPE_IQ1_S:
      return grouped_iq_mmq_impl<GGML_TYPE_IQ1_S>(weight, x, topk_ids, rows, out);
    case GGML_TYPE_IQ3_XXS:
      return grouped_iq_mmq_impl<GGML_TYPE_IQ3_XXS>(weight, x, topk_ids, rows, out);
    default:
      TORCH_CHECK(false, "unsupported llama IQ MMQ type ", type);
  }
}

static torch::Tensor grouped_iq_mmq_out(
    torch::Tensor weight,
    torch::Tensor x,
    torch::Tensor topk_ids,
    const int64_t type,
    const int64_t rows,
    torch::Tensor out) {
  switch (type) {
    case GGML_TYPE_IQ1_S:
      return grouped_iq_mmq_impl<GGML_TYPE_IQ1_S>(weight, x, topk_ids, rows, out);
    case GGML_TYPE_IQ3_XXS:
      return grouped_iq_mmq_impl<GGML_TYPE_IQ3_XXS>(weight, x, topk_ids, rows, out);
    default:
      TORCH_CHECK(false, "unsupported llama IQ MMQ type ", type);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_iq_mmq", &grouped_iq_mmq, "llama.cpp grouped IQ MMQ");
  m.def("grouped_iq_mmq_out", &grouped_iq_mmq_out, "llama.cpp grouped IQ MMQ into output");
}
