#include <torch/extension.h>
#include <cuda_runtime.h>
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

            float s = Scales[j];
            float z = Zeros[j];

            float q = quantize_val(w, s, z, qmin, qmax);
            err[j / WARP] = w - q;

            W[(long long)row * total_cols + j_global] = q;
            Err[(long long)row * total_cols + j_global] = err[j / WARP];
        }
        __syncwarp();
    }
}

void gptq_fused_cuda(
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


