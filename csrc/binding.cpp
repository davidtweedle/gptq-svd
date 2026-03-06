#include <torch/extension.h>
#include "gptq_kernel.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static void validate_inputs(
        torch::Tensor W,
        torch::Tensor H_T,
        torch::Tensor Scales,
        torch::Tensor Zeros,
        torch::Tensor Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax
        ) {
    CHECK_INPUT(W);
    CHECK_INPUT(H_T);
    CHECK_INPUT(Scales);
    CHECK_INPUT(Zeros);
    CHECK_INPUT(Err);
    TORCH_CHECK(col_offset >= 0, "col_offset must be >= 0");
    TORCH_CHECK(block_cols > 0, "block_cols must be > 0");
    TORCH_CHECK(col_offset + block_cols <= total_cols, "col_offset + block_cols exceeds total_cols");

    TORCH_CHECK(W.dim() == 2, "W must be 2D");
    TORCH_CHECK(H_T.dim() == 2, "H_T must be 2D");
    TORCH_CHECK(Scales.dim() == 2, "Scales must be 2D");
    TORCH_CHECK(Zeros.dim() == 2, "Zeros must be 2D");
    TORCH_CHECK(Err.dim() == 2, "Err must be 2D");

    TORCH_CHECK(W.size(1) == total_cols, "W must have total_cols cols");
    TORCH_CHECK(H_T.size(0) == block_cols && H_T.size(1) == block_cols, "H_T must be block_cols x block_cols");
    TORCH_CHECK(Scales.size(0) == W.size(0), "Scales rows must match W rows");
    TORCH_CHECK(Zeros.size(0) == W.size(0), "Zeros rows must match W rows");
    TORCH_CHECK(Scales.size(1) == block_cols, "Scales must have block_cols cols");
    TORCH_CHECK(Zeros.size(1) == block_cols, "Zeros must have block_cols cols");
    TORCH_CHECK(Err.sizes() == W.sizes(), "Err must match W shape");
}

void gptq_fused_lazy_py(
        torch::Tensor W,
        torch::Tensor H_T,
        torch::Tensor Scales,
        torch::Tensor Zeros,
        torch::Tensor Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax,
        bool accum_fp64
        ) {
    validate_inputs(W, H_T, Scales, Zeros, Err, total_cols, col_offset, block_cols);
    gptq_fused_lazy_cuda(
            W, H_T, Scales, Zeros, Err, total_cols, col_offset, block_cols, qmin, qmax, accum_fp64
            );
}

void gptq_fused_immediate_py(
        torch::Tensor W,
        torch::Tensor H_T,
        torch::Tensor Scales,
        torch::Tensor Zeros,
        torch::Tensor Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax,
        bool accum_fp64
        ) {
    validate_inputs(W, H_T, Scales, Zeros, Err, total_cols, col_offset, block_cols);
    gptq_fused_immediate_cuda(
            W, H_T, Scales, Zeros, Err, total_cols, col_offset, block_cols, qmin, qmax, accum_fp64
            );
}

void gptq_fused_lazy_reduce_py(
        torch::Tensor W,
        torch::Tensor H_T,
        torch::Tensor Scales,
        torch::Tensor Zeros,
        torch::Tensor Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax,
        bool accum_fp64
        ) {
    validate_inputs(W, H_T, Scales, Zeros, Err, total_cols, col_offset, block_cols);
    gptq_fused_lazy_reduce_cuda(
            W, H_T, Scales, Zeros, Err, total_cols, col_offset, block_cols, qmin, qmax, accum_fp64
            );
}


void gptq_fused_py(
        torch::Tensor W,
        torch::Tensor H_T,
        torch::Tensor Scales,
        torch::Tensor Zeros,
        torch::Tensor Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax
        ) {
    validate_inputs(W, H_T, Scales, Zeros, Err, total_cols, col_offset, block_cols);

    gptq_fused_lazy_cuda(
            W, H_T, Scales, Zeros, Err, total_cols, col_offset, block_cols, qmin, qmax, false
            );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gptq_fused", &gptq_fused_py, "GPTQ Fused Kernel (CUDA)");
    m.def("gptq_fused_lazy", &gptq_fused_lazy_py, "GPTQ Fused Lazy Kernel (CUDA)");
    m.def("gptq_fused_immediate", &gptq_fused_immediate_py, "GPTQ Fused Immediate Kernel (CUDA)");
    m.def("gptq_fused_lazy_reduce", &gptq_fused_lazy_reduce_py, "GPTQ Fused Lazy-Reduce Kernel (CUDA)");
}
