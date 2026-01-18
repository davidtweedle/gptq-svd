# TruncGPTQ: Quantization for LLMs using truncated LDL decomposition

[License](LICENSE)

**TruncGPTQ** is a numerically stable quantization framework for Large Language Models. It replaces the Cholesky-based solver in GPTQ with a truncated spectral approach that explicitly handles rank-deficiency. By explicitly handling rank-deficiency in the Hessian $H = X^TX$ where $X$ are the activations, TruncGPTQ preserves signal integrity where standard GPTQ injects noise.

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
* Columns are ordered according to their statistical contribution to the Hessian.

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
| **Quantization time** | Fast | 1.5x Slower |

## Why TruncGPTQ?
Recall that GPTQ (and originally OBS and OBQ) aim to quantize the weights in a way which minimizes the difference $(\hat{W} - W) X)$, for the sample inputs $X$. The approach of GPTQ is to compute $H = X^TX$ and then write
$$ H^{-1} = \Rho^T \Rho $$
where $\Rho$ is upper-triangular and has positive diagonal. The matrix $\Rho$ can be used to quantize each column of $W$ in order propogate the error of the quantization to the remaining (unquantized) columns.

Here is how GPTQ works.

1a. $H = X^TX + \lambda I$
1b. We may reorder the activations to better represent the important activations. In GPTQ, this is done by sorting according to the norms of the activations (the diagonal of $H$).
2. Write $H = R^TR$ where $R$ is upper-triangular and has positive diagonal elements. GPTQ uses Cholesky factorization (which is the justification for adding $\lambda I$ in step 1 - if you don't Cholesky may fail to find a factorization)
3. Compute $H^{-1}$ from $R$ using triangular solve
4. Compute $H^{-1} = \Rho^T\Rho$ again using Cholesky factorization, where $\Rho$ is upper-triangular and has positive elements on the diagonal.
5. $\Rho$ serves as the input to the quantization step, in which each column is quantized, and the errors are propogated to the unquantized columns.

Here is how TruncGPTQ works.
1. **Hessian calculation:** $H = X^TX$ (no damping)
2. **Spectral Decomposition:** $H = Q\Lambda Q^T$ (spectral decomposition), $\Lambda$ is diagonal, $Q$ is orthogonal
3. **Truncation:** Set $\lambda_i = 0$ for all eigenvalues below a threshold.
4. **Square Root Representation:** Set $S = Lambda^{1/2}Q^T$. Note that $S^TS = H$.
5. **Rank-revealing ordering:** Perform pivoted QR on $S$: $SP = Q'R$ where $P$ is a permutation matrix.
6. **Inverse Factorization:** Compute the pseudoinverse of the truncated Hessian to derive the error-propogation matrix $\Rho$.

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
[ ] Streaming QR (perhaps the most numerically stable option)
[ ] Larger models
[ ] Support for MOE
[ ] Variety of models (Llama, Mixtral, etc.)
[ ] Tuning of truncation for individual layers
[ ] QUIP\# integration
[ ] Randomized Linear Algebra
[ ] Proper ablation study

## Citation
```bibtex
@software{Tweedle-TruncGPTQ-2026,
  author = {David Tweedle},
  title = {TruncGPTQ: Quantization for LLMs via Truncated Spectral Decomposition},
  url = {https://github.com/davidtweedle/TruncGPTQ},
  year = {2026}
}
```
