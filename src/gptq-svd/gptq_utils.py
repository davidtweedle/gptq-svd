"""
GPTQ/Block LDLQ utilities
-------------------------
Implements low-rank aware quantization using svd/qr with pivoting.
Integrates JAX for QR with pivoting and Triton for blockwise processing.
Includes reference implementation of regular gptq.
"""
import os
import math
import gc
from typing import Tuple, Optional, Generator, Union
import logging

# Environment configuration (Must be before JAX import)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import torch
from torch import nn
import jax
from jax.dlpack import from_dlpack
import triton
from triton import language as tl

# JAX Configuration
# Enable x64 for precision in QR decomposition
jax.config.update("jax_enable_x64", True)
# Use magma for dgeqp3 kernel (pivoted QR)
jax.config.update("jax_use_magma", 'on')
jax.config.update("jax_platforms", "cuda,cpu")



def process_sketch(
        sketch: torch.Tensor,
        threshold: float = 1e-2,
        threshold_method: str = "mean_trimmed"
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    device = sketch.device
    n_features = sketch.shape[1]
    logging.info(f"   [Memory defrag] Moving sketch {sketch.shape} to CPU")
    sketch_cpu = sketch.cpu()
    del sketch
    torch.cuda.empty_cache()
    logging.info(f"   [Memory alloc] Allocating Float64 buffer on gpu")
    sketch_double = sketch_cpu.to(device=device, dtype=torch.float64)
    del sketch_cpu
    factor, _ = torch.geqrf(sketch_double)
    R_reduced = torch.triu(factor[:n_features, :])
    del factor, sketch_double
    torch.cuda.empty_cache()

    _, S, Vh = torch.linalg.svd(R_reduced, full_matrices=False)

    if threshold_method == "energy":
        energy = S ** 2
        target = (1.0 - threshold) * torch.sum(energy)
        current_rank = int((torch.cumsum(energy, dim=0) <= target).sum().item())
        if current_rank < len(S):
            current_rank += 1
    elif threshold_method == "mean_trimmed":
        ref_k = min(33, len(S))
        ref_val = torch.mean(S[1:ref_k]) if len(S) > 1 else S[0]
        current_rank = int((S > threshold * ref_val).sum().item())

    current_rank = max(1, min(current_rank, len(S)))
    S = S[:current_rank]
    Vh = Vh[:current_rank, :]

    H_sqrt = S.unsqueeze(1) * Vh

    H_sqrt_jax = from_dlpack(H_sqrt)
    _, _, perm_jax = jax.scipy.linalg.qr(H_sqrt_jax, pivoting=True, mode='economic')
    perm = torch.from_dlpack(perm_jax).long()

    S_inv = 1.0 / S
    H_inv_partial = S_inv.unsqueeze(1) * Vh

    H_inv_permuted = H_inv_partial[:, perm]

    _, R_prime = torch.linalg.qr(H_inv_permuted, mode='reduced')
    diag_sign = torch.sign(torch.diagonal(R_prime))
    R = R_prime * diag_sign.unsqueeze(1)

    return R, perm


def process_hessian_alt(
        H: torch.Tensor,
        threshold: float = 0.0005,
        threshold_method: str = "mean_trimmed"
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    H_double = H.to(dtype=torch.float64)
    L, V = torch.linalg.eigh(H_double)
    S = torch.sqrt(L.clamp(min=1e-12)).flip(0)
    Vh = V.T.flip(0)
    del H_double, L, V
    if threshold_method == "energy":
        energy = S ** 2
        target = (1.0 - threshold) * torch.sum(energy)
        current_rank = int((torch.cumsum(energy, dim=0) <= target).sum())
        if current_rank < len(S):
            current_rank += 1
    elif threshold_method == "mean_trimmed":
        ref_k = min(33, len(S))
        ref_val = torch.mean(S[1:ref_k]) if len(S) > 1 else S[0]
        current_rank = int((S > threshold * ref_val).sum().item())
    else:
        current_rank = int(len(S))
    S = S[:current_rank]
    Vh = Vh[:current_rank, :]
    S_inv = 1.0 / S
    H_sqrt = S.unsqueeze(1) * Vh
    H_sqrt_jax = from_dlpack(H_sqrt)
    _, _, perm_jax = jax.scipy.linalg.qr(H_sqrt_jax, pivoting=True, mode='economic')
    perm = torch.from_dlpack(perm_jax).long()
    del H_sqrt_jax
    H_inv_partial = S_inv.unsqueeze(1) * Vh
    H_inv_permuted = H_inv_partial[:, perm]
    _, R_prime = torch.linalg.qr(H_inv_permuted, mode='reduced')
    diag_sign = torch.sign(torch.diagonal(R_prime))
    R = (R_prime * diag_sign.unsqueeze(1))

    return R, perm


def process_hessian(
        H: torch.Tensor,
        actorder: bool = False,
        damp_percent: float = 0.01
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    H_double = H.to(dtype=torch.float64)
    n_features = H.shape[0]
    device = H.device
    if actorder:
        perm = torch.argsort(torch.diag(H_double), descending=True)
        H_double = H_double[perm][:, perm]
    else:
        perm = torch.arange(n_features, device=device)
    diag = torch.diagonal(H_double)
    mean_diag = torch.mean(diag)
    if mean_diag == 0:
        mean_diag = 1.0

    H_inv_factor = None
    for damp in [damp_percent, 0.1, 1.0, 10.0]:
        try:
            H_damped = H_double.clone()
            H_damped.diagonal().add_(damp * mean_diag)
            L = torch.linalg.cholesky(H_damped)
            H_inv = torch.cholesky_inverse(L)
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)
            if damp > damp_percent:
                logging.info(f"  Ref-GPTQ required high damping: {damp}")
            break
        except RuntimeError:
            continue

    if H_inv_chol is None:
        logging.warning(" Hessian is singular. Using Identity fallback.")
        H_inv_chol = torch.eye(n_features, device=device, dtype=torch.float64)
    return H_inv_chol, perm

# ==============================================================================
# CLASSES
# ==============================================================================

class Sketcher:
    """
    Accumulates sketch of inputs Y = R @ X during the forward pass.
    Used to approximate input activation covariance without storing full X.
    """
    def __init__(self, layer: nn.Module, rank: int, device: Union[str, torch.device] = 'cuda'):
        self.layer = layer
        self.rank = rank
        self.device = device
        self.in_features = layer.in_features
        self.Y = torch.zeros((rank, self.in_features), device=device, dtype=torch.float32).contiguous()
        self.n_samples = 0

    def hook_fn(self, module: nn.Module, input_args: Tuple[torch.Tensor], output: torch.Tensor):
        x = input_args[0]
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        self.n_samples += batch_count
        x_float = x.to(dtype=torch.float32)

        # Gaussian matrix for sketching
        # R is (rank, batch_size)
        R_batch = torch.randn((self.rank, batch_count), device=self.device, dtype=torch.float32)

        # Accumulate sketch
        # self.Y += R_batch @ x_float
        self.Y.addmm_(R_batch, x_float, beta=1.0, alpha=1.0)

        del R_batch, x_float

    def get_scaled_sketch(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        if self.n_samples == 0:
            return None, None, 0

        # Normalize by sqrt(N * rank) to stabilize numerical scale
        scale_factor = 1.0 / math.sqrt(self.n_samples * self.rank)
        self.Y.mul_(scale_factor)
        return self.Y

class HessianAccumulator:
    def __init__(self, in_features, device, dtype=torch.float64):
        self.H = torch.zeros((in_features, in_features), device=device, dtype=dtype)
        self.n_samples = 0

    def add_batch(self, x):
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(self.H.dtype)
        self.H += x.T @ x
        self.H.addmm_(x.T, x)
        self.n_samples += x.shape[0]

    def get_hessian(self):
        if self.n_samples == 0:
            return self.H
        return self.H / self.n_samples

class Quantizer:
    """
    Simple pseudo quantizer.
    Calculates scales based on max of absolute values of weights.
    """
    def __init__(self, per_channel: bool = True, w_bits: int = 4):
        self.per_channel = per_channel
        self.w_bits = w_bits
        self.scale = None
        self.max_val = 2 ** (w_bits - 1) - 1
        self.min_val = 1 - 2 ** (w_bits - 1)

    def init_scale(self, weights: torch.Tensor):
        if self.per_channel:
            # scale per output channel (dim=-1)
            max_abs, _ = torch.max(torch.abs(weights), dim=-1, keepdim=True)
        else:
            max_abs = torch.max(torch.abs(weights))

        self.scale = max_abs / self.max_val

    def quantize(self, weights: torch.Tensor) -> torch.Tensor:
        # assumes q is of shape (m, 1) or (m, n)
        q = torch.round(torch.clamp(weights / self.scale, min=self.min_val, max=self.max_val))
        return q * self.scale


# ==============================================================================
# TRITON KERNELS
# ==============================================================================

@triton.jit
def gptq_block_kernel(
        W_ptr,
        Q_ptr,
        E_ptr,
        R_ptr,
        Scales_ptr,
        Zeros_ptr,
        stride_w_row, stride_w_col,
        stride_q_row, stride_q_col,
        stride_e_row, stride_e_col,
        stride_r_row, stride_r_col,
        stride_s,
        n_rows,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        MIN_VAL: tl.constexpr,
        MAX_VAL: tl.constexpr
        ):
    """
    Triton kernel for block-wise GPTQ/LDLQ block quantization with error propogation

    Operations:
    1. Quantize a column 'k'
    2. Compute error 'E'.
    3. Propogate 'E' to future columns within block
    4. Update W[:, j] -= E * (R[k, j] / R[k, k])
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    offsets_rows = row_start + tl.arange(0, BLOCK_ROWS)
    mask_rows = offsets_rows < n_rows

    # Load scales
    scale_ptrs = Scales_ptr + offsets_rows * stride_s
    scales = tl.load(scale_ptrs, mask=mask_rows, other=1.0)

    # Pointers for weight block
    offsets_cols = tl.arange(0, BLOCK_SIZE)
    w_ptrs = W_ptr + (offsets_rows[:, None] * stride_w_row) + (offsets_cols[None, :] * stride_w_col)
    w_data = tl.load(w_ptrs, mask=mask_rows[:, None], other=0.0)

    # Iterative quantization within block
    for k in range(BLOCK_SIZE):
        mask_k = (offsets_cols == k)[None, :]

        # select current column
        w_col = tl.sum(w_data * mask_k, axis=1)

        # quantize
        w_scaled = w_col / scales
        w_clamped = tl.clamp(w_scaled, float(MIN_VAL), float(MAX_VAL))
        q_int = tl.floor(w_clamped + 0.5)
        q_val = q_int * scales

        # calculate error
        error = w_col - q_val

        # Write output and error
        e_out_ptrs = E_ptr + (offsets_rows * stride_e_row) + (k * stride_e_col)
        tl.store(e_out_ptrs, error, mask=mask_rows)

        q_out_ptrs = Q_ptr + (offsets_rows * stride_q_row) + (k * stride_q_col)
        tl.store(q_out_ptrs, q_val, mask=mask_rows)

        # Fetch R info where R^T R ~ (X^TX)^{-1}
        r_ptrs = R_ptr + (k * stride_r_row) + (offsets_cols * stride_r_col)
        r_row = tl.load(r_ptrs)

        # diagonal element for scaling
        diag_mask = (offsets_cols == k)
        diag = tl.sum(r_row * diag_mask, axis=0)
        inv_diag = 1.0 / diag

        correction_vec = r_row * inv_diag

        # error propogation
        err_broad = error[:, None]
        corr_broad = correction_vec[None, :]
        delta = err_broad * corr_broad

        # apply update to future columns
        update_mask = offsets_cols > k
        w_data = tl.where(update_mask[None, :], w_data - delta, w_data)


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def triton_process_block(
        w_block: torch.Tensor,
        R_block: torch.Tensor,
        quantizer: 'Quantizer'
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper to launch Triton quantization kernel on a block of weights.
    Handles padding to nearest power of 2 for efficiency.
    """
    out_features, n_cols = w_block.shape
    target_block_size = next_power_of_2(n_cols)
    if target_block_size < 16:
        target_block_size = 16

    # Padding
    pad_cols = target_block_size - n_cols
    if pad_cols > 0:
        w_input = torch.nn.functional.pad(w_block, (0, pad_cols), mode='constant', value=0.0)
        R_input = torch.nn.functional.pad(R_block, (0, pad_cols, 0, pad_cols), mode='constant', value=1.0)
    else:
        w_input = w_block
        R_input = R_block

    q_output = torch.empty_like(w_input)
    e_output = torch.empty_like(w_input)

    BLOCK_ROWS = 64
    grid = lambda meta: (triton.cdiv(out_features, BLOCK_ROWS),)

    gptq_block_kernel[grid](
            w_input, q_output, e_output, R_input,
            quantizer.scale.squeeze(),
            None,  # Zeros_ptr placeholder
            w_input.stride(0), w_input.stride(1),
            q_output.stride(0), q_output.stride(1),
            e_output.stride(0), e_output.stride(1),
            R_input.stride(0), R_input.stride(1),
            quantizer.scale.stride(0),
            out_features,
            BLOCK_SIZE=target_block_size,
            BLOCK_ROWS=BLOCK_ROWS,
            MIN_VAL=quantizer.min_val,
            MAX_VAL=quantizer.max_val
            )

    # Unpad
    if pad_cols > 0:
        q_final = q_output[:, :n_cols]
        e_final = e_output[:, :n_cols]
    else:
        q_final = q_output
        e_final = e_output

    return q_final, e_final


# ==============================================================================
# MAIN ALGORITHM: SVD + QR + TRITON
# ==============================================================================

def gptq_svd_qr_fwrd(
        weight_mat: torch.Tensor,
        R: torch.Tensor,
        quantizer: Quantizer,
        perm: torch.Tensor,
        block_size: int = 1024
        ) -> Tuple[torch.Tensor, int]:
    """
    SVD/QR with pivoting based GPTQ using pre-computed sketch

    1. Decomposes Y = USV^T
    2. Selects rank based on threshold
    3. Uses JAX QR with pivoting to compute most important input channels
    4. Computes inverse Cholesky factor R from SVD components
    5. Runs triton kernel for block-wise quantization from R
    """

    out_features, in_features = weight_mat.shape
    device = weight_mat.device
    dtype = weight_mat.dtype

    R = R.to(dtype)

    current_rank = R.shape[0]

    logging.info(f"   Rank percent used: {float(current_rank) / R.shape[1]}")

    # Block wise quantization
    W = weight_mat[:, perm]
    quantizer.init_scale(W)
    Q_W = torch.zeros_like(W)

    for i in range(0, current_rank, block_size):
        j = min(i + block_size, current_rank)
        w_block = W[:, i:j]
        R_block_diag = R[i:j, i:j]

        # Run Triton kernel
        w_block_quantized, E_block = triton_process_block(
                w_block,
                R_block_diag,
                quantizer
                )
        Q_W[:, i:j] = w_block_quantized

        # Propogate error to remaining blocks
        if j < in_features:
            R_cross = R[i:j, j:]
            R_diags = torch.diagonal(R_block_diag)
            Scale_Mat = R_cross / R_diags.unsqueeze(1)
            Global_Delta = E_block @ Scale_Mat
            W[:, j:] -= Global_Delta

    # Quantize remaining weights directly
    if current_rank < in_features:
        w_rem = W[:, current_rank:]
        Q_W[:, current_rank:] = quantizer.quantize(w_rem)

    # restore original column order
    inv_perm = torch.argsort(perm)
    final_W = Q_W[:, inv_perm]

    return final_W, current_rank


# ==============================================================================
# REFERENCE IMPLEMENTATION
# ==============================================================================

def gptq_ref_fwrd(
        H_inv_chol: torch.Tensor,
        weight_mat: torch.Tensor,
        out_weight: torch.Tensor,
        quantizer: Quantizer,
        blocksize: int,
        perm: torch.Tensor,
        ):
    """
    Standard GPTQ/Block LDLQ algorithm (for benchmarking).
    Accumulates H = X^TX and performs Cholesky based quantization.
    """
    m, n = weight_mat.shape
    device = weight_mat.device
    dtype = weight_mat.dtype

    H_inv_chol = H_inv_chol.to(dtype)

    W = weight_mat[:, perm]

    quantizer.init_scale(W)
    Q_final = torch.zeros_like(W)
    Losses = torch.zeros_like(W)

    # Standard loop
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = H_inv_chol[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = quantizer.quantize(w.unsqueeze(1)).flatten()
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q_final[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2
        if i2 < n:
            W[:, i2:] -= Err1.matmul(H_inv_chol[i1:i2, i2:])

    inv_perm = torch.argsort(perm)
    out_weight[:] = Q_final[:, inv_perm]

    del W, Q_final, Losses
    torch.cuda.empty_cache()
