import torch
import jax
import math
from jax.dlpack import from_dlpack

class Quantizer:

    def __init__(self, per_channel=True, w_bits=4):
        self.per_channel = per_channel
        self.w_bits = w_bits
        self.scale = None
        self.max_val = 2 ** (w_bits - 1) - 1
        self.min_val = 1 - 2 ** (w_bits - 1)
        # max_val = 2 ** (w_bits - 1) - 1
        # min_val = 1 - 2 ** (w_bits - 1)

    def init_scale(self, weights):
        if self.per_channel:
            max_abs, _ = torch.max(torch.abs(weights), dim=-1, keepdim=True)
        else:
            max_abs = torch.max(torch.abs(weights))
        self.scale = max_abs / self.max_val

    def quantize(self, weights):
        # assumes q is of shape (m, 1) or (m, n)
        q = torch.round(torch.clamp(weights / self.scale, min=self.min_val, max=self.max_val))
        return q * self.scale


def streaming_sketch(
        make_stream,
        n: int,
        d: int,
        oversample: int = 8,
        k_iter: int = 2,
        device=None,
        dtype=None
        ):
    if device is None or dtype is None:
        first_batch = next(make_stream())
        device = first_batch.device
        dtype = first_batch.dtype

    d_eff = d + oversample
    R = torch.randn(n, d_eff, dtype=dtype, device=device)

    def apply_X(B):
        Y_parts = []
        for chunk in make_stream():
            Y_parts.append(chunk @ B)
        return torch.cat(Y_parts, dim=0)

    def apply_XT(Y):
        Z = torch.zeros(n, Y.shape[1], device=device, dtype=dtype)
        start = 0
        for chunk in make_stream():
            b = chunk.shape[0]
            Y_chunk = Y[start : start + b]
            Z += chunk.T @ Y_chunk
            start += b
        return Z

    Y = torch.linalg.qr(apply_X(R), mode='reduced').Q
    for _ in range(k_iter):
        Z = apply_XT(Y)
        Y = torch.linalg.qr(apply_X(Z), mode='reduced').Q
    B = torch.zeros(Y.shape[1], n, device=device, dtype=dtype)
    start = 0
    for chunk in make_stream():
        b = chunk.shape[0]
        Q_chunk = Y[start : start + b]
        B += Q_chunk.T @ chunk
        start += b
    return B, Y


def gptq_fwrd(
        oversample,
        k_iter,
        make_stream,
        weight_mat,
        out_weight,
        quantizer
        ):
    m, n = weight_mat.shape
    device = weight_mat.device
    dtype = weight_mat.dtype
    d = int(max(math.sqrt(n), 1))
    # first compute sketch of input_stream
    B, _ = streaming_sketch(
            make_stream,
            n,
            d,
            oversample,
            k_iter,
            device=device,
            dtype=dtype
            )
    B = B.to(torch.float32)
    U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U_tilde, S, Vh = U_tilde[:, :d], S[:d], Vh[:d]
    SVh = torch.diag(S) @ Vh
    SVh_jax = from_dlpack(SVh)
    _, _, P_jax = jax.scipy.linalg.qr(SVh_jax, pivoting=True, mode='economic')
    P = torch.from_dlpack(P_jax)
    quantizer.init_scale(weight_mat)
    mask = torch.ones(n, dtype=bool, device=device)
    for i in range(d):
        j = P[i]
        mask[j] = False
        SVh_mask = SVh[:, mask]
        Up, Sp, Vph = torch.linalg.svd(SVh_mask, full_matrices=False)
        q_j = quantizer.quantize(weight_mat[:, j: j + 1])
        u_j = U_tilde.T @ B[:, j]
        c = Up.T @ u_j
        c_scaled = c / Sp
        delta_mask = (weight_mat[:, j: j + 1] - q_j) * (Vph.T @ c_scaled)
        full_delta = torch.zeros_like(weight_mat)
        full_delta[:, mask] = delta_mask
        weight_mat += full_delta.to(dtype)
        out_weight[:, j: j + 1] = q_j

    out_weight[:, mask] = quantizer.quantize(weight_mat[:, mask])

    # notes -- B = Q.T @ X -- d by n
    # U ~ Q @ U_tilde
    # U^T X_1 = U_tilde.T Q.T X_1
    # U^T X_1 = U_tilde.T B[0]
    # (v1 - q) V Sigma^{-1} U_tilde.T B[col]
    # first compute X = U S V^T
    # Then compute S[:mask] V[:mask]^T
    # then compute X[P[i]]
    # SVh[:, mask] = Up Sp Vp.T
    # X_R = U Svh[:, mask]
    # U = Q @ U_tilde
    # X_R = Q U_tilde Up Sp Vp.T
    # delta = (v1 - q) Vp Sp^{-1} Up.T U_tilde.T B[col]

if __name__ == '__main__':
    torch.manual_seed(0)

    num_samples = 1024 * 256
    n = 1024
    m = 256
    device = torch.device("cuda")
    dtype = torch.float32
    X = torch.randn(num_samples, n, device=device, dtype=dtype)
    weight_mat = torch.randn(m, n, device=device, dtype=dtype)
    out_weight = torch.zeros_like(weight_mat)

    batch_size = 1024

    def make_stream():
        for i in range(0, num_samples, batch_size):
            yield X[i : i + batch_size]

    Y_full = X @ weight_mat.T

    q = Quantizer(per_channel=True, w_bits=3)

    gptq_fwrd(
            oversample=8,
            k_iter=2,
            make_stream=make_stream,
            weight_mat=weight_mat,
            out_weight=out_weight,
            quantizer=q
            )
    Y_quant = X @ out_weight.T

    diff = Y_full - Y_quant
    rel_err = torch.norm(diff) / torch.norm(Y_full)
    max_err = diff.abs().max()
    print(f"Relative output error ||XW - XW_q|| / ||XW|| = {rel_err.item():.4e}")
    print(f"Max absolute entrywise error on outputs      = {max_err.item():.4e}")

    # Also check weight error
    w_diff = torch.norm(weight_mat - out_weight) / torch.norm(weight_mat)
    print(f"Relative weight error ||W - W_q|| / ||W||    = {w_diff.item():.4e}")
    # Baseline: plain quantization with no GPTQ corrections
    q_baseline = Quantizer(per_channel=True, w_bits=4)
    q_baseline.init_scale(weight_mat_original := weight_mat.clone())
    W_plain_q = q_baseline.quantize(weight_mat_original)
    Y_plain_q = X @ W_plain_q.T

    diff_plain = Y_full - Y_plain_q
    rel_err_plain = torch.norm(diff_plain) / torch.norm(Y_full)
    max_err_plain = diff_plain.abs().max()
    w_rel_plain = torch.norm(weight_mat_original - W_plain_q) / torch.norm(weight_mat_original)

    print("=== Plain quantization baseline ===")
    print(f"Rel output error (plain) = {rel_err_plain.item():.4e}")
    print(f"Max output error (plain)  = {max_err_plain.item():.4e}")
    print(f"Rel weight error (plain)  = {w_rel_plain.item():.4e}")
