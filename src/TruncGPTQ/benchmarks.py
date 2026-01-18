"""
GPTQ/Block LDLQ SVD Benchmarks
------------------------------
Synthetic data generation and benchmarking scripts to compare SVD-based GPTQ
against reference GPTQ and baseline round to nearest.
"""

import torch
import itertools
import gc
from typing import Generator, Dict, List, Any
from gptq_utils import Quantizer, gptq_svd_qr_fwrd, gptq_ref_fwrd

# ==============================================================================
# DATA GENERATION
# ==============================================================================

def make_ar1_cholesky(n: int, rho: float = 0.9, device: str = "cuda", dtype=torch.float32) -> torch.Tensor:
    """
    Generates Cholesky factor L for an AR(1) covariance matrix.
    """
    idx = torch.arange(n, device=device)
    dist = (idx[None, :] - idx[:, None]).abs()
    Sigma = rho ** dist
    # add jitter for numerical stability
    Sigma = Sigma + 1e-6 * torch.eye(n, device=device, dtype=dtype)
    L = torch.linalg.cholesky(Sigma)
    return L


def make_X(
        num_samples: int,
        n: int,
        mode: str = "gaussian",
        rho: float = 0.9,
        nu: float = 3.0,
        device: str = "cuda",
        dtype=torch.float32
        ) -> torch.Tensor:
    """
    Generates synthetic input activation matrices X.
    Supports Gaussian, Student-t, and log-normal distributions with AR(1) correlations.
    """
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
        n_samples: int = 2048,
        n: int = 1024,
        m: int = 1024,
        device: str = "cuda"
        ) -> List[Dict[str, Any]]:
    """
    Runs a grid search over hyperparameters (epsilon, sketch ratio)
    to find optimal settings for SVD-GPTQ algorithm.
    """
    print(f"--- Starting hyperparam tuning (N={n}, M={m}) ---")
    dtype = torch.float32

    X = make_X(n_samples, n, mode="gaussian_corr", rho=0.9, nu=3.0, device=device, dtype=dtype)
    W = torch.randn(m, n, device=device, dtype=dtype)

    Y_true = X @ W.T
    norm_Y_true = torch.norm(Y_true)
    param_grid_test = {
            'sketch_ratio': [1.0, 2.0],
            'eps': [1e-1, 1e-2, 1e-3, 1e-4]
            }

    def expand_grid(grid):
        keys, values = zip(*grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    configs = expand_grid(param_grid_test)

    results = []
    print(f"Total configurations: {len(configs)}")

    for i, cfg in enumerate(configs):
        torch.cuda.empty_cache()
        gc.collect()

        sketch_dim = int(n * cfg['sketch_ratio'])

        scale = 1.0 / torch.sqrt(torch.tensor(sketch_dim, device=device))
        Omega = torch.randn(sketch_dim, n_samples, device=device, dtype=dtype) * scale
        input_sketch = Omega @ X

        out_weight = torch.zeros_like(W)
        quantizer = Quantizer(per_channel=True, w_bits=4)
        try:
            out_weight, _ = gptq_svd_qr_fwrd(
                    weight_mat=W,
                    input_sketch=input_sketch,
                    quantizer=quantizer,
                    threshold=cfg['eps']
                    )
            Y_quant = X @ out_weight

            diff = Y_true - Y_quant
            rel_err = torch.norm(diff) / norm_Y_true
            res = cfg.copy()
            res['rel_err'] = rel_err.item()
            res['sketch_dim'] = sketch_dim
            results.append(res)

            print(f"Config: {cfg} | Rel Err: {rel_err.item():.5e}")
        except Exception as e:
            print(f"Config {cfg} failed: {e}")
            import traceback
            traceback.print_Exc()

    print("\n--- Tuning complete ---")
    results.sort(key=lambda x: x['rel_err'])
    header = f"{'Ratio':<8} | {'eps':<8} | {'Rel Error':<12}"
    print("-" * len(header))
    print(header)
    for r in results[:5]:
        print(f"{r['sketch_ratio']:<8} | {r['eps']:<8} | {r['rel_err']:.5e}")

    return results


def experiment():
    """
    Comparative experiment, svd-gptq vs reference gptq vs baseline
    Runs on multiple synthetic data distributions.
    """
    torch.manual_seed(0)

    num_samples = 1024 * 32
    n = 1024
    m = 1024
    device = torch.device("cuda")
    dtype = torch.float32

    modes = ["gaussian", "gaussian_corr", "student_t", "student_t_corr", "lognormal_corr"]
    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        X = make_X(num_samples, n, mode=mode, rho=0.9, nu=3.0, device=device, dtype=dtype)
        W0 = torch.randn(m, n, device=device, dtype=dtype)

        Y_full = X @ W0.T

        weight_mat = W0.clone()
        q = Quantizer(per_channel=True, w_bits=4)

        sketch_dim = 4 * n
        scale = 1.0 / torch.sqrt(torch.tensor(sketch_dim, device=device))
        Omega = torch.randn(sketch_dim, num_samples, device=device, dtype=dtype) * scale
        input_sketch = Omega @ X

        out_weight, _ = gptq_svd_qr_fwrd(
                weight_mat=weight_mat,
                input_sketch=input_sketch,
                quantizer=q,
                threshold=1e-1
                )

        Y_quant_svd = X @ out_weight.T

        diff_svd = Y_full - Y_quant_svd
        rel_err_svd = torch.norm(diff_svd) / torch.norm(Y_full)
        max_err_svd = diff_svd.abs().max()
        w_diff_svd = torch.norm(W0 - out_weight) / torch.norm(W0)
        print("SVD-GPTQ Errors:")
        print(f"  Relative output err: {rel_err_svd.item():.4e}")
        print(f"  Max output err:      {max_err_svd.item():.4e}")
        print(f"  Relative weight err: {w_diff_svd.item():.4e}")

        weight_mat_ref = W0.clone()
        q_ref = Quantizer(per_channel=True, w_bits=4)
        out_weight_ref = torch.zeros_like(weight_mat_ref)

        def make_stream():
            batch_size = 1024
            for i in range(0, num_samples, batch_size):
                yield X[i : i + batch_size]

        print("\nReference GPTQ Errors:")
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
        w_diff_ref = torch.norm(W0 - out_weight_ref) / torch.norm(W0)

        print(f"  Relative Output Err: {rel_err_ref.item():.4e}")
        print(f"  Max Output Err:      {max_err_ref.item():.4e}")
        print(f"  Relative Weight Err: {w_diff_ref.item():.4e}")

        # Baseline: plain quantization with no GPTQ corrections
        q_baseline = Quantizer(per_channel=True, w_bits=4)
        q_baseline.init_scale(weight_mat_original := W0.clone())
        W_plain_q = q_baseline.quantize(weight_mat_original)
        Y_plain_q = X @ W_plain_q.T

        diff_plain = Y_full - Y_plain_q
        rel_err_plain = torch.norm(diff_plain) / torch.norm(Y_full)
        max_err_plain = diff_plain.abs().max()
        w_rel_plain = torch.norm(weight_mat_original - W_plain_q) / torch.norm(weight_mat_original)

        print("\nBaseline (RTN) Errors:")
        print(f"  Relative Output Err: {rel_err_plain.item():.4e}")
        print(f"  Max Output Err:      {max_err_plain.item():.4e}")
        print(f"  Relative Weight Err: {w_rel_plain.item():.4e}")

        del X, W0, weight_mat, out_weight, Y_full, q, input_sketch
        del Y_quant_svd, weight_mat_ref, q_ref, out_weight_ref
        del Y_quant_ref, diff_ref, q_baseline, W_plain_q, Y_plain_q, diff_plain
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.manual_seed(42)
    # Run grid search
    best_results = run_tuning_grid(n_samples=2048, n=1024, m=1024)
