"""
GPTQ/Block LDLQ utilities
-------------------------
Implements low-rank aware quantization using svd/qr with pivoting.
Integrates JAX for QR with pivoting and Triton for blockwise processing.
Includes reference implementation of regular gptq.
"""
import os
import math
from typing import Tuple, Optional, Union
import logging

# Environment configuration (Must be before JAX import)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import torch
from torch import nn
import jax
from jax.dlpack import from_dlpack

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
    _, R_x_jax, perm_jax = jax.scipy.linalg.qr(H_sqrt_jax, pivoting=True, mode='economic')
    perm = torch.from_dlpack(perm_jax).long()
    R_x = torch.from_dlpack(R_x_jax)
    del H_sqrt_jax
    H_inv_partial = S_inv.unsqueeze(1) * Vh
    H_inv_permuted = H_inv_partial[:, perm]
    _, R_prime = torch.linalg.qr(H_inv_permuted, mode='reduced')
    diag_sign = torch.sign(torch.diagonal(R_prime))
    diag_sign_x = torch.sign(torch.diagonal(R_x))
    R_x = R_x * diag_sign_x.unsqueeze(1)
    R = R_prime * diag_sign.unsqueeze(1)

    return R, R_x, perm


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
    for damp_exp in range(5):
        try:
            damp = 10 ** damp_exp * damp_percent
            H_damped = H_double.clone()
            H_damped.diagonal().add_(damp * mean_diag)
            L = torch.linalg.cholesky(H_damped)
            H_inv = torch.cholesky_inverse(L)
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)
            if damp_exp > 0:
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

        # Normalize by sqrt(N * rank / 2) to stabilize numerical scale
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
    def __init__(self, w_bits: int = 4, group_size: int = 128, sym: bool = False):
        self.w_bits = w_bits
        self.group_size = group_size
        self.sym = sym
        if self.sym:
            half_range = 2 ** (w_bits - 1) - 1
            self.max_q = half_range
            self.min_q = -half_range
        else:
            self.max_q = 2 ** w_bits - 1
            self.min_q = 0
        self.scale = None
        self.zero = None

    def find_params(self, weights: torch.Tensor):
        m, n = weights.shape
        g_size = self.group_size if self.group_size > 0 else n

        assert n % g_size == 0

        w = weights.reshape(m, -1, g_size)

        if self.sym:
            x_abs_max = torch.amax(torch.abs(w), dim=2, keepdim=True)
            x_abs_max = x_abs_max.clamp(min=1e-5)
            self.scale = x_abs_max / self.max_q
            self.zero = torch.zeros_like(self.scale)
        else:
            mn = torch.amin(w, dim=2, keepdim=True)
            mx = torch.amax(w, dim=2, keepdim=True)
            self.scale = (mx - mn).clamp(min=1e-5) / self.max_q
            self.zero = torch.round(-mn / self.scale).clamp(0, self.max_q)

    def get_expanded_params(self, m, n):
        s_expanded = torch.repeat_interleave(self.scale, self.group_size if self.group_size > 0 else n, dim=1)
        z_expanded = torch.repeat_interleave(self.zero, self.group_size if self.group_size > 0 else n, dim=1)

        return s_expanded[:, :n].squeeze(-1), z_expanded[:, :n].squeeze(-1)


def log_quantization_error(
        W_orig: torch.Tensor,
        W_quant: torch.Tensor,
        R_x: torch.Tensor,
        perm: torch.Tensor
        ):
    if R_x is None or perm is None:
        return
    w_dtype = W_orig.dtype
    r_device = W_orig.device
    R_mat = R_x.to(device=r_device, dtype=torch.float32)
    W_o = W_orig[:, perm]
    W_q = W_quant[:, perm]
    y_orig_norm = torch.linalg.norm(torch.matmul(W_o, R_mat.T))
    y_diff_norm = torch.linalg.norm(torch.matmul(W_o - W_q, R_mat.T))
    relative_error = (y_diff_norm / y_orig_norm).item()
    logging.info(f"   [Metric] Relative prediction error: {relative_error:.6f}")

# ==============================================================================
# GPTQ Updates
# ==============================================================================

def gptq_fwrd(
        weight_mat: torch.Tensor,
        H_inv_sqrt: torch.Tensor,
        quantizer: Quantizer,
        perm: torch.Tensor,
        block_size: int = 128,
        R_x: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, int]:
    """
    Unified GPTQ implementation

    - Supports rank-reduced H_inv_sqrt
    - Supports error logging
    """

    out_features, in_features = weight_mat.shape
    device = weight_mat.device
    orig_dtype = weight_mat.dtype
    weight_mat = weight_mat.to(device=device, dtype=torch.float32)

    H_inv_sqrt = H_inv_sqrt.to(device=device)

    current_rank = H_inv_sqrt.shape[0]

    if current_rank < in_features:
        logging.info(f"   Rank percent used: {float(current_rank) / in_features:.2%}")

    # Block wise quantization
    quantizer.find_params(weight_mat)
    S_full, Z_full = quantizer.get_expanded_params(out_features, in_features)
    W = weight_mat[:, perm].clone()
    S = S_full[:, perm].to(device=device, dtype=torch.float32).clone()
    Z = Z_full[:, perm].to(device=device, dtype=torch.float32).clone()

    Q_final = torch.zeros_like(W)

    for i1 in range(0, current_rank, block_size):
        i2 = min(i1 + block_size, current_rank)
        count = i2 - i1

        W1 = W[:, i1:i2]
        S1 = S[:, i1:i2]
        Z1 = Z[:, i1:i2]
        Hinv1 = H_inv_sqrt[i1:i2, i1:i2]

        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            s = S1[:, i]
            z = Z1[:, i]

            q = torch.round(w / s + z).clamp(quantizer.min_q, quantizer.max_q)
            q_dequant = (q - z) * s
            Q1[:, i] = q_dequant

            err1 = (w - q_dequant) / d
            delta = err1.to(Hinv1.dtype).unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            W1[:, i:] -= delta.to(dtype=torch.float32)
            Err1[:, i] = err1.to(dtype=torch.float32)
        Q_final[:, i1:i2] = Q1

        if i2 < in_features:
            block_update = Err1.to(H_inv_sqrt.dtype).matmul(H_inv_sqrt[i1:i2, i2:])
            W[:, i2:] -= block_update.to(dtype=torch.float32)

    if current_rank < in_features:
        W_tail = W[:, current_rank:]
        S_tail = S[:, current_rank:]
        Z_tail = Z[:, current_rank:]

        q_tail = torch.round(W_tail / S_tail + Z_tail).clamp(quantizer.min_q, quantizer.max_q)
        Q_final[:, current_rank:] = (q_tail - Z_tail) * S_tail

    # restore original column order
    inv_perm = torch.argsort(perm)
    final_W = Q_final[:, inv_perm]

    if R_x is not None:
        log_quantization_error(weight_mat, final_W, R_x, perm)

    return final_W, current_rank
