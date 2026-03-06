#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include "gptq_kernel.h"

#define WARP 32
#define CHUNK 256

__device__ __forceinline__ float quantize_val(
        float w,
        float scale,
        float zero,
        float qmin,
        float qmax
        ) {
    float q = rintf(w / scale + zero);
    q = fmaxf(q, qmin);
    q = fminf(q, qmax);
    return (q - zero) * scale;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}


__global__ void gptq_fused_kernel(
        float* __restrict__ W,          // M by block_size
        const float* __restrict__ H_T,    // block_size by block_size
        const float* __restrict__ Scales,
        const float* __restrict__ Zeros,
        float* __restrict__ Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin, float qmax
        ) {

    int row = blockIdx.x;
    int lane = threadIdx.x;

    extern __shared__ float sh_mem[];
    float* sh_H = sh_mem;

    float err[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        err[i] = 0.0f;

    for (int j = 0; j < block_cols; ++j) {
        float corr = 0.0f;

        for (int i0 = 0; i0 < j; i0 += CHUNK) {

            int i_max = min(i0 + CHUNK, j);
            int chunk_size = i_max - i0;

            for (int idx = lane; idx < chunk_size; idx += WARP) {
                sh_H[idx] = H_T[j * block_cols + (i0 + idx)];
            }

            __syncwarp();
            for (int i = i0; i < i_max; ++i) {
                int owner_lane = i % WARP;
                int idx_in_lane = i / WARP;

                float e_i = __shfl_sync(0xFFFFFFFF, err[idx_in_lane], owner_lane);
                float Hij = sh_H[i - i0];

                corr = fmaf(e_i, Hij, corr);
            }

        }
        if ((j % WARP) == lane) {
            int j_global = col_offset + j;
            float w = W[(long long)row * total_cols + j_global];

            w -= corr;

            float s = Scales[(long long)row * block_cols + j];
            float z = Zeros[(long long)row * block_cols + j];

            float q = quantize_val(w, s, z, qmin, qmax);
            err[j / WARP] = w - q;

            W[(long long)row * total_cols + j_global] = q;
            Err[(long long)row * total_cols + j_global] = err[j / WARP];
        }
        __syncwarp();
    }
}

template <typename AccT>
__global__ void gptq_fused_lazy_reduce_kernel(
        float* __restrict__ W,
        const float* __restrict__ H_T,
        const float* __restrict__ Scales,
        const float* __restrict__ Zeros,
        float* __restrict__ Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax
        ) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    extern __shared__ float sh_err_hist[];

    for (int j = 0; j < block_cols; ++j) {
        AccT partial_corr = static_cast<AccT>(0.0);
        for (int i = lane; i < j; i += WARP) {
            partial_corr = fmaf(
                    static_cast<AccT>(sh_err_hist[i]),
                    static_cast<AccT>(H_T[j * block_cols + i]),
                    partial_corr
                    );
        }

        AccT corr = warp_reduce_sum(partial_corr);
        corr = __shfl_sync(0xFFFFFFFF, corr, 0);

        if ((j % WARP) == lane) {
            int j_global = col_offset + j;
            float w = W[(long long)row * total_cols + j_global];
            w -= static_cast<float>(corr);

            float s = Scales[(long long)row * block_cols + j];
            float z = Zeros[(long long)row * block_cols + j];
            float q = quantize_val(w, s, z, qmin, qmax);
            float err = w - q;

            sh_err_hist[j] = err;
            W[(long long)row * total_cols + j_global] = q;
            Err[(long long)row * total_cols + j_global] = err;
        }
        __syncwarp();
    }
}

template <typename AccT>
__global__ void gptq_fused_immediate_kernel(
        float* __restrict__ W,
        const float* __restrict__ H_T,
        const float* __restrict__ Scales,
        const float* __restrict__ Zeros,
        float* __restrict__ Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax
        ) {
    int row = blockIdx.x;
    int lane = threadIdx.x;

    for (int j = 0; j < block_cols; ++j) {
        int owner_lane = j % WARP;
        AccT err_j = static_cast<AccT>(0.0);
        int j_global = col_offset + j;

        if (lane == owner_lane) {
            float w = W[(long long)row * total_cols + j_global];
            float s = Scales[(long long)row * block_cols + j];
            float z = Zeros[(long long)row * block_cols + j];
            float q = quantize_val(w, s, z, qmin, qmax);

            err_j = static_cast<AccT>(w - q);
            W[(long long)row * total_cols + j_global] = q;
            Err[(long long)row * total_cols + j_global] = static_cast<float>(err_j);
        }

        err_j = __shfl_sync(0xFFFFFFFF, err_j, owner_lane);

        for (int k = j + 1 + lane; k < block_cols; k += WARP) {
            int k_global = col_offset + k;
            float w_old = W[(long long)row * total_cols + k_global];
            AccT delta = err_j * static_cast<AccT>(H_T[k * block_cols + j]);
            W[(long long)row * total_cols + k_global] = static_cast<float>(
                    static_cast<AccT>(w_old) - delta
                    );
        }
        __syncwarp();
    }
}

static void gptq_fused_core_cuda(
        torch::Tensor W,
        const torch::Tensor& H_T,
        const torch::Tensor& Scales,
        const torch::Tensor& Zeros,
        torch::Tensor Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax
        ) {
    const int rows = W.size(0);

    const int threads = WARP;
    const int blocks = rows;

    size_t shared_mem_bytes = CHUNK * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    gptq_fused_kernel<<<blocks, threads, shared_mem_bytes, stream>>>(
            W.data_ptr<float>(),
            H_T.data_ptr<float>(),
            Scales.data_ptr<float>(),
            Zeros.data_ptr<float>(),
            Err.data_ptr<float>(),
            total_cols,
            col_offset,
            block_cols, 
            qmin,
            qmax
            );

    AT_CUDA_CHECK(cudaGetLastError());
}

void gptq_fused_lazy_cuda(
        torch::Tensor W,
        const torch::Tensor& H_T,
        const torch::Tensor& Scales,
        const torch::Tensor& Zeros,
        torch::Tensor Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax,
        bool accum_fp64
        ) {
    (void)accum_fp64;
    gptq_fused_core_cuda(
            W, H_T, Scales, Zeros, Err, total_cols, col_offset, block_cols, qmin, qmax
            );
}

void gptq_fused_immediate_cuda(
        torch::Tensor W,
        const torch::Tensor& H_T,
        const torch::Tensor& Scales,
        const torch::Tensor& Zeros,
        torch::Tensor Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax,
        bool accum_fp64
        ) {
    const int rows = W.size(0);
    const int threads = WARP;
    const int blocks = rows;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (accum_fp64) {
        gptq_fused_immediate_kernel<double><<<blocks, threads, 0, stream>>>(
                W.data_ptr<float>(),
                H_T.data_ptr<float>(),
                Scales.data_ptr<float>(),
                Zeros.data_ptr<float>(),
                Err.data_ptr<float>(),
                total_cols,
                col_offset,
                block_cols,
                qmin,
                qmax
                );
    } else {
        gptq_fused_immediate_kernel<float><<<blocks, threads, 0, stream>>>(
                W.data_ptr<float>(),
                H_T.data_ptr<float>(),
                Scales.data_ptr<float>(),
                Zeros.data_ptr<float>(),
                Err.data_ptr<float>(),
                total_cols,
                col_offset,
                block_cols,
                qmin,
                qmax
                );
    }
    AT_CUDA_CHECK(cudaGetLastError());
}

void gptq_fused_lazy_reduce_cuda(
        torch::Tensor W,
        const torch::Tensor& H_T,
        const torch::Tensor& Scales,
        const torch::Tensor& Zeros,
        torch::Tensor Err,
        int total_cols,
        int col_offset,
        int block_cols,
        float qmin,
        float qmax,
        bool accum_fp64
        ) {
    const int rows = W.size(0);
    const int threads = WARP;
    const int blocks = rows;
    size_t shared_mem_bytes = static_cast<size_t>(block_cols) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (accum_fp64) {
        gptq_fused_lazy_reduce_kernel<double><<<blocks, threads, shared_mem_bytes, stream>>>(
                W.data_ptr<float>(),
                H_T.data_ptr<float>(),
                Scales.data_ptr<float>(),
                Zeros.data_ptr<float>(),
                Err.data_ptr<float>(),
                total_cols,
                col_offset,
                block_cols,
                qmin,
                qmax
                );
    } else {
        gptq_fused_lazy_reduce_kernel<float><<<blocks, threads, shared_mem_bytes, stream>>>(
                W.data_ptr<float>(),
                H_T.data_ptr<float>(),
                Scales.data_ptr<float>(),
                Zeros.data_ptr<float>(),
                Err.data_ptr<float>(),
                total_cols,
                col_offset,
                block_cols,
                qmin,
                qmax
                );
    }
    AT_CUDA_CHECK(cudaGetLastError());
}
