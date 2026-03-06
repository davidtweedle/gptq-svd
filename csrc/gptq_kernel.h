#pragma once
#include <torch/extension.h>

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
        );

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
        );

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
        );
