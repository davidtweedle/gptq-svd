# TruncGPTQ: Quantization for LLMs using truncated spectral decomposition

[License](LICENSE)

**TruncGPTQ** is a numerically stable quantization framework for Large Language Models. It replaces the Cholesky-based solver in GPTQ with a truncated spectral approach that robustly handles rank-deficiency. By preserving the true signal structure of the Hessian, $H = X^TX$, TruncGPTQ avoids the need for the damping ($+\lambda I$) required by standard GPTQ.

TruncGPTQ is a drop-in replacement for the "Hessian Inverse" step of GPTQ, requiring no new inference kernels. On Qwen3-8B, we see up to a 75% reduction in perplexity degradation compared to the FP16 baseline (symmetric quantization to 4-bit).

## Key Contributions

1. Stability
* Replaces Cholesky factorization with truncated spectral decomposition
* Explicitly handles rank-deficient Hessians
* Eliminates the need for damping $H + \lambda I$

2. Signal-preserving
* GPTQ damping injects a small amount of noise to all signals
* TruncGPTQ truncates only numerically meaningless signal
* Preserves dominant correlation structure in $H = X^TX$

3. Rank-revealing ordering
* Replaces activation ordering by norm (ActOrder) by Pivoted QR (a rank-revealing factorization)
* Columns are ordered by conditional variance, prioritizing features that contribute the most unique information to the Hessian structure while deferring redundant (correlated) features.

## Benchmark Results (WikiText-2 test set Perplexity)
*Evaluation performed on **Qwen3-8B**. Lower is better.*

| Method | Bit-Width | Mode | PPL | $\Delta$ vs FP16 |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 Baseline** | 16-bit | - | **8.57** | - |
| GPTQ | 4-bit | Asym | 8.75 | +0.18 |
| **TruncGPTQ** | 4-bit | Asym | **8.64** | **+0.07** |
| GPTQ | 4-bit | Sym | 8.92 | +0.35 |
| **TruncGPTQ** | 4-bit | Sym | **8.66** | **+0.09** |
| GPTQ | 3-bit | Asym | 9.59 | +1.02 |
| **TruncGPTQ** | 3-bit | Asym | **9.20** | **+0.63** |
| GPTQ | 3-bit | Sym | 10.25 | +1.68 |
| **TruncGPTQ** | 3-bit | Sym | **9.85** | **+1.28** |
| GPTQ | 2-bit | Asym | 24.98  | +16.41 |
| **TruncGPTQ** | 2-bit | Asym | **21.65** | **+13.08** |


### Comparison of Standard GPTQ vs TruncGPTQ

| Feature | Standard GPTQ | **TruncGPTQ** (Ours) |
| :--- | :--- | :--- |
| **Matrix Decomposition** | Cholesky ($LL^T$) | Truncated spectral decomposition |
| **Stability Strategy** | Damping ($H + \lambda I$) | Hard Truncation |
| **Numerics** | Adds noise to all features | Preserves signal, ignores noise |
| **Column Selection** | Actorder (norm-based) | Pivoted QR (Rank-based) |

## Why TruncGPTQ?
Recall that GPTQ (and originally OBS and OBQ) aim to quantize the weights while minimizing the error in the output activations.
This is an optimization problem involving the inverse of the Hessian, $H^{-1}$, where $H = X^TX$.

Indeed, the insight of GPTQ is that if we can write $H^{-1} = U^TU$ where $U$ is upper-triangular, we can quantize each column and propagate the error to the unquantized columns in a computationally tractable manner (compared to OBS/OBQ).


### The Standard Approach: GPTQ
1. **Damping:** Since $H$ is often rank-deficient or ill-conditioned, GPTQ forces it to be positive definite by adding a damping factor $\lambda$:

$$ H' = X^TX + \lambda I $$

Problem: This adds noise to every feature, diluting the correlation structure.

2. **ActOrder:** We may reorder the activations to better represent the important activations. In GPTQ, this is done by sorting according to the norms of the activations (the diagonal of $H$).
3. **Cholesky Factorization:** Compute $H' = R^TR$ where $R$ is upper-triangular.
4. **Inversion:** Compute $H^{-1} = R^{-1}R^{-T}$.
5. **Re-Factorization:** Compute the Cholesky factorization of the inverse to find the error propagation matrix $U$:

$$ H^{-1} = U^TU. $$

**Note:** If $H$ is indefinite, Cholesky fails. If $\lambda$ is too large, the data is corrupted.

### The Spectral Approach: TruncGPTQ
The aim of TruncGPTQ is to identify the true rank of $H$ and discard the noise.

1. **Hessian Factorization:** We decompose the raw Hessian (no damping) 

$$ H = V\Lambda V^T $$

2. **Truncation:** Set $\lambda_i = 0$ for all eigenvalues below a threshold. This creates a cleaned diagonal $\tilde{\Lambda}$.
3. **Square Root Representation:** We construct a matrix $S$ that represents the square root of the cleaned Hessian:

$$ S = \tilde{\Lambda}^{1/2} V^T. $$

*Note that $S^TS \cong H$.*

4. **Rank-revealing ordering:** We apply pivoted QR to $S$: 

$$SP^T = QR, $$

where $P$ is a permutation matrix, $Q$ is orthogonal, and $R$ is upper triangular.

This greedily selects columns with the largest projection onto the orthogonal complement of the previously selected columns. This orders features by their conditional variance - prioritizing those that add the most new information and pushing highly correlated features to the end.

5. **Direct Solution:** The error propagation matrix $U$ is derived directly from the pseudoinverse of $R$.

## Installation & Environment Setup

This project requires JAX and PyTorch. JAX is used for the pivoted QR routines (via Magma) which is not currently exposed in PyTorch (or CuSOLVER).

```bash
git clone https://github.com/davidtweedle/TruncGPTQ.git
cd TruncGPTQ
chmod +x setup_env.sh
./setup_env.sh
conda activate trunc-gptq
```

**Note on `setup_env.sh`:** This script handles Magma integration and environment variables to prevent JAX/Torch VRAM conflicts.

## Run benchmark
```bash
python TruncGPTQ/src/TruncGPTQ/run_benchmark.py
```

## Limitations and hardware support
* Tested Hardware: Currently this implementation is validated specifically for Nvidia A100 (40GB). Performance and compatibility on other architectures are not yet guaranteed.
* Model Support: Development has been focused on **Qwen3-8B** architecture.
* Statistical Significance: Results reflect single-run experiments. Median results over multiple seeds are pending.
* Computational Cost: The quantization process uses Eigendecomposition (torch.eigh), once, one QR with pivoting, and one regular QR, as opposed to two Cholesky factorizations for GPTQ. This makes it about 1.5X slower than regular GPTQ. This has zero impact on inference speed.

## Roadmap: Future developments

[ ] Support for saving quantized weights (compatible with AutoGPTQ/vLLM)
[ ] Support Mixture of Experts models
[ ] Support Llama 3, Mistral, Mixtral
[ ] Layer-wise adaptive truncation (auto-tune epsilon)
[ ] Integration with other libraries (e.g., QUIP\#, etc.)
[ ] Apply Streaming QR or Randomized Linear Algebra to accumulation of Hessian
[ ] Ablation study

## Citation
```bibtex
@software{Tweedle-TruncGPTQ-2026,
  author = {David Tweedle},
  title = {TruncGPTQ: Quantization for LLMs via Truncated Spectral Decomposition},
  url = {https://github.com/davidtweedle/TruncGPTQ},
  year = {2026}
}
```
