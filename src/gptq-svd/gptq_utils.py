import torch
import jax
import itertools
from jax.dlpack import from_dlpack
import gc

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

def gptq_ref_fwrd(
        make_stream,
        weight_mat,
        out_weight,
        quantizer,
        blocksize,
        ):
    m, n = weight_mat.shape
    device = weight_mat.device
    dtype = weight_mat.dtype
    H = torch.zeros(n, n, dtype=dtype, device=device)
    n_samples = 0
    for X in make_stream():
        chunk_samples = X.shape[0]
        H = H * (n_samples) / (n_samples + chunk_samples) + (1/ (n_samples + chunk_samples)) * X.T @ X
        n_samples += chunk_samples
    percdamp = 0.01
    diag = H.diagonal()
    mean = torch.mean(diag)

    diag.add_(percdamp * mean)
    H2 = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(H2), upper=True)
    del H2
    W = weight_mat.clone()
    quantizer.init_scale(W)
    Losses = torch.zeros_like(W)
    for i1 in range(0, n, blocksize): 
        i2 = min(i1 + blocksize, n)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = quantizer.quantize(w.unsqueeze(1)).flatten()
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        out_weight[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    avg_loss = torch.sum(Losses).item() / n_samples
    print(f"Losses sum item: {torch.sum(Losses).item()}")
    print(f"Average loss: {avg_loss}")


def gptq_svd_fwrd_test(
        sketch_dim,
        make_stream,
        weight_mat,
        out_weight,
        quantizer,
        eps,
        ):
    m, n = weight_mat.shape
    device = weight_mat.device
    dtype = weight_mat.dtype
    B = torch.zeros(sketch_dim, n, device=device, dtype=dtype)
    for chunk in make_stream():
        bsz = chunk.shape[0]
        R = torch.randn(sketch_dim, bsz, device=device, dtype=dtype)
        B += R @ chunk
    # first compute sketch of input_stream
    U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
    threshold = eps * S[0]
    d = torch.count_nonzero(S >= threshold)
    U_tilde, S, Vh = U_tilde[:, :d], S[:d], Vh[:d]

    SVh = torch.diag(S) @ Vh
    SVh_jax = from_dlpack(SVh)
    _, _, P_jax = jax.scipy.linalg.qr(SVh_jax, pivoting=True, mode='economic')
    P = torch.from_dlpack(P_jax)
    W = weight_mat.clone()
    quantizer.init_scale(W)
    mask = torch.ones(n, dtype=bool, device=device)
    for i in range(d):
        j = P[i]
        mask[j] = False
        SVh_mask = SVh[:, mask]
        Up, Sp, Vph = torch.linalg.svd(SVh_mask, full_matrices=False)
        q_j = quantizer.quantize(W[:, j: j + 1])
        u_j = U_tilde.T @ B[:, j]
        c = Up.T @ u_j
        c_scaled = c / Sp
        delta_mask = (W[:, j: j + 1] - q_j) * (Vph.T @ c_scaled)
        full_delta = torch.zeros_like(W)
        full_delta[:, mask] = delta_mask
        W += full_delta.to(dtype)
        out_weight[:, j: j + 1] = q_j

    out_weight[:, mask] = quantizer.quantize(W[:, mask])


def gptq_svd_fwrd(
        sketch_dim,
        oversample,
        k_iter,
        make_stream,
        weight_mat,
        out_weight,
        quantizer,
        eps,
        ):
    m, n = weight_mat.shape
    device = weight_mat.device
    dtype = weight_mat.dtype
    # first compute sketch of input_stream
    B, _ = streaming_sketch(
            make_stream,
            n,
            sketch_dim,
            oversample,
            k_iter,
            device=device,
            dtype=dtype
            )
    # B = B.to(torch.float32)
    U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
    threshold = S[0] * eps
    d = torch.count_nonzero(S >= threshold)
    U_tilde, S, Vh = U_tilde[:, :d], S[:d], Vh[:d]

    SVh = torch.diag(S) @ Vh
    SVh_jax = from_dlpack(SVh)
    _, _, P_jax = jax.scipy.linalg.qr(SVh_jax, pivoting=True, mode='economic')
    P = torch.from_dlpack(P_jax)
    W = weight_mat.clone()
    quantizer.init_scale(W)
    mask = torch.ones(n, dtype=bool, device=device)
    for i in range(d):
        j = P[i]
        mask[j] = False
        SVh_mask = SVh[:, mask]
        Up, Sp, Vph = torch.linalg.svd(SVh_mask, full_matrices=False)
        q_j = quantizer.quantize(W[:, j: j + 1])
        u_j = U_tilde.T @ B[:, j]
        c = Up.T @ u_j
        c_scaled = c / Sp
        delta_mask = (W[:, j: j + 1] - q_j) * (Vph.T @ c_scaled)
        full_delta = torch.zeros_like(W)
        full_delta[:, mask] = delta_mask
        W += full_delta.to(dtype)
        out_weight[:, j: j + 1] = q_j

    out_weight[:, mask] = quantizer.quantize(W[:, mask])

def make_ar1_cholesky(n, rho=0.9, device="cuda", dtype=torch.float32):
    idx = torch.arange(n, device=device)
    dist = (idx[None, :] - idx[:, None]).abs()
    Sigma = rho ** dist
    Sigma = Sigma + 1e-6 * torch.eye(n, device=device, dtype=dtype)
    L = torch.linalg.cholesky(Sigma)
    return L


def make_X(
        num_samples: int,
        n: int,
        mode: str = "gaussian",
        rho: float = 0.9,
        nu: float = 3.0,
        device="cuda",
        dtype=torch.float32
        ):
    device = torch.device(device)
    if mode == "gaussian":
        X = torch.randn(num_samples, n, device=device, dtype=dtype)
        return X

    if mode == "gaussian_corr":
        L = make_ar1_cholesky(n, rho=rho, device=device, dtype=dtype)
        Z = torch.randn(num_samples, n, device=device, dtype=dtype)
        X = Z @ L.T
        return X

    if mode == "student_t":
        t_dist = torch.distributions.StudentT(df=nu)
        X = t_dist.sample((num_samples, n)).to(device=device, dtype=dtype)
        if nu > 2:
            X = X / torch.sqrt(torch.tensor(nu / (nu - 2), device=device, dtype=dtype))
        return X
    
    if mode == "student_t_corr":
        L = make_ar1_cholesky(n, rho=rho, device=device, dtype=dtype)
        Z = torch.randn(num_samples, n, device=device, dtype=dtype)
        Y = Z @ L.T
        chi2 = torch.distributions.Chi2(df=nu).sample((num_samples,)).to(device=device, dtype=dtype)
        scale = torch.sqrt(nu / chi2).view(-1, 1)
        X = Y * scale
        return X

    if mode == "lognormal_corr":
        L = make_ar1_cholesky(n, rho=rho, device=device, dtype=dtype)
        Z = torch.randn(num_samples, n, device=device, dtype=dtype)
        G = Z @ L.T
        X = torch.exp(G)
        return X

    raise ValueError(f"Unknown mode: {mode}")


def run_tuning_grid(
        n_samples=2048,
        n=1024,
        m=1024,
        device="cuda"
        ):
    print(f"--- Starting hyperparam tuning (N={n}, M={m}) ---")
    dtype = torch.float32
    X = make_X(n_samples, n, mode="gaussian_corr", rho=0.9, nu=3.0, device=device, dtype=dtype)
    W = torch.randn(m, n, device=device, dtype=dtype)

    Y_true = X @ W.T
    norm_Y_true = torch.norm(Y_true)
    param_grid_main = {
            'sketch_ratio': [0.1, 0.5, 1.0, 2.0],
            'k_iter': [0, 1, 2],
            'eps': [1e-2, 1e-4, 1e-8],
            'oversample': [16],
            'test_sketch': [False]
            }
    param_grid_test = {
            'sketch_ratio': [1.0, 2.0, 4.0, 8.0],
            'eps': [1e-2, 1e-4, 1e-8],
            'test_sketch': [True]
            }

    def expand_grid(grid):
        keys, values = zip(*grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]
    configs = expand_grid(param_grid_main) + expand_grid(param_grid_test)

    results = []
    print(f"Total configurations: {len(configs)}")

    for i, cfg in enumerate(configs):
        torch.cuda.empty_cache()
        gc.collect()

        sketch_dim = int(n * cfg['sketch_ratio'])

        out_weight = torch.zeros_like(W)
        quantizer = Quantizer(per_channel=True, w_bits=4)
        batch_size = 512
        def make_stream():
            for i in range(0, n_samples, batch_size):
                yield X[i : i + batch_size]

        try:
            if cfg['test_sketch']:
                gptq_svd_fwrd_test(
                        sketch_dim=sketch_dim,
                        make_stream=make_stream,
                        weight_mat=W,
                        out_weight=out_weight,
                        quantizer=quantizer,
                        eps=cfg['eps']
                        )
            else:
                gptq_svd_fwrd(
                        sketch_dim=sketch_dim,
                        oversample=cfg['oversample'],
                        k_iter=cfg['k_iter'],
                        make_stream=make_stream,
                        weight_mat=W,
                        out_weight=out_weight,
                        quantizer=quantizer,
                        eps=cfg['eps']
                        )
            Y_quant = X @ out_weight.T
            diff = Y_true - Y_quant
            rel_err = torch.norm(diff) / norm_Y_true
            res = cfg.copy()
            res['rel_err'] = rel_err.item()
            res['sketch_dim'] = sketch_dim
            results.append(res)

            print(f"Config: {cfg} | Rel Err: {rel_err.item():.5e}")
        except Exception as e:
            print(f"Config {cfg} failed: {e}")

    print(f"\n--- Tuning complete ---")
    results.sort(key=lambda x: x['rel_err'])
    header = f"{'Method':<6} | {'Ratio':<6} | {'k_iter':<6} | {'eps':<8} | {'Rel Error':<12}"
    print("-" * len(header))
    print(header)
    for r in results[:5]:
        method = "TEST" if r['test_sketch'] else "MAIN"
        k_val = r.get('k_iter', '-')
        print(f"{method:<6} | {r['sketch_ratio']:<6} | {k_val:<6} | {r['eps']:<8} | {r['rel_err']:.5e}")

    return results






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

def experiment():
    torch.manual_seed(0)

    num_samples = 1024 * 32
    n = 1024
    m = 1024
    device = torch.device("cuda")
    dtype = torch.float32
    # X = torch.randn(num_samples, n, device=device, dtype=dtype)
    modes = ["gaussian", "gaussian_corr", "student_t", "student_t_corr", "lognormal_corr"]
    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        X = make_X(num_samples, n, mode=mode, rho=0.9, nu=3.0, device=device, dtype=dtype)
        W0 = torch.randn(m, n, device=device, dtype=dtype)
        weight_mat = W0.clone()
        out_weight = torch.zeros_like(weight_mat)

        batch_size = 1024

        def make_stream():
            for i in range(0, num_samples, batch_size):
                yield X[i : i + batch_size]

        Y_full = X @ W0.T

        q = Quantizer(per_channel=True, w_bits=2)

        gptq_svd_fwrd(
                sketch_dim=4*n,
                oversample=0,
                k_iter=0,
                make_stream=make_stream,
                weight_mat=weight_mat,
                out_weight=out_weight,
                quantizer=q,
                eps=1e-1
                )
        Y_quant_svd = X @ out_weight.T

        diff_svd = Y_full - Y_quant_svd
        rel_err_svd = torch.norm(diff_svd) / torch.norm(Y_full)
        max_err_svd = diff_svd.abs().max()
        print("SVD Errors:\n")
        print(f"Relative output error ||XW - XW_q|| / ||XW|| = {rel_err_svd.item():.4e}")
        print(f"Max absolute entrywise error on outputs      = {max_err_svd.item():.4e}")
        w_diff_svd = torch.norm(W0 - out_weight) / torch.norm(W0)
        print(f"Relative weight error ||W - W_q|| / ||W||    = {w_diff_svd.item():.4e}")
        weight_mat_ref = W0.clone()
        q_ref = Quantizer(per_channel=True, w_bits=2)
        out_weight_ref = torch.zeros_like(weight_mat_ref)

        print("Reference GPTQ errors")
        gptq_ref_fwrd(
                make_stream=make_stream,
                weight_mat=weight_mat_ref,
                out_weight=out_weight_ref,
                quantizer=q_ref,
                blocksize=128,
                )
        Y_quant_ref = X @ out_weight_ref.T

        diff_ref = Y_full - Y_quant_ref
        rel_err_ref = torch.norm(diff_ref) / torch.norm(Y_full)
        max_err_ref = diff_ref.abs().max()
        print(f"Relative output error ||XW - XW_q|| / ||XW|| = {rel_err_ref.item():.4e}")
        print(f"Max absolute entrywise error on outputs      = {max_err_ref.item():.4e}")


        # Also check weight error
        w_diff_ref = torch.norm(W0 - out_weight_ref) / torch.norm(W0)
        print(f"Relative weight error ||W - W_q|| / ||W||    = {w_diff_ref.item():.4e}")
        # Baseline: plain quantization with no GPTQ corrections
        q_baseline = Quantizer(per_channel=True, w_bits=2)
        q_baseline.init_scale(weight_mat_original := W0.clone())
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
        del X, W0, weight_mat, out_weight, Y_full, q
        del Y_quant_svd, diff_svd, rel_err_svd, max_err_svd
        del weight_mat_ref, q_ref, out_weight_ref, Y_quant_ref
        del diff_ref, rel_err_ref, max_err_ref, w_diff_ref
        del q_baseline, W_plain_q, Y_plain_q, diff_plain, rel_err_plain
        del max_err_plain, w_rel_plain
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.manual_seed(42)
    best_results = run_tuning_grid(n_samples=2048, n=1024, m=1024)
