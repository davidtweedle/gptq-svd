# Low rank quantization for LLMs using GPTQ/Block LDLQ

**A hybrid quantization pipeline that combines randomized SVD, pivoted QR with GPTQ/Block LDLQ

- GPTQ is an applied least squares problem
- computationally more constrained because least squares repeatedly needs to be applied on the same data
- GPTQ/Block LDLQ solved this (starting from OBQ/OBS) by using the Cholesky factorization, so Cholesky can be computed once then repeatedly applied.
- But from the perspective of numerical linear algebra, Cholesky can be unstable
- Why? Cholesky says that if \(H\) is positive definite, then we can factor \(H = R^TR\) where \(R\) is upper-triangular with positive diagonal.
- In practice, \(H = X^TX\) where \(X\) is \(d\) by \(n\) and \(d\gg n\).
- Recall GPTQ: \(H = X^TX\), then compute \(H = R^TR\), where \(R\) is upper-triangular (from Cholesky Factorization), then compute \(H^{-1} = R^{-1}R^{-T}\), and find the Cholesky factorization \(H^{-1} = \rho^T \rho\).
- The problem is that slight numerical inaccuracies from the accumulation of \(H\) may cause \(H\) to not be positive definite.
- To resolve this, implementations of GPTQ calculate \(H = X^TX + \lambda I\) where \(\lambda\) is small. If \(H\) remains indefinite, \(\lambda\) is slowly increased until \(H\) is positive definite.
- This has the effect of diluting the information of \(H\).
- I propose the following alternative: attempt to write \(X = U\Sigma V^T\) and use this to find \(R\) such that \(H^{-1} = R^TR\)
- If we were able to compute this factorization of \(X\), what could we do with it?
- We could compute the GPTQ update as follows:
- write \(\Sigma^{-1} V^T = QR\) and then (if necessary) multiply by appropriate signs so that \(R\) is upper-triangular, has positive diagonals and satisfies \(H^{-1} = R^TR\).
- We could also truncate the singular values of \(\Sigma\) for numerical stability.
- In that case, the rank of \(R\) will be less than the number of columns we have to quantize.
- So let's choose the most important columns to quantize.
- We do it by calculating \(\Sigma V^T P = QR\) where \(P\) is a permutation matrix (pivoted QR).
- Can also do it according to the norms of the columns.
- Computing \(X = U\Sigma V^T\) is infeasible, so we sketch the inputs \(Y = S(X)\) and then calculate \(Y = U\Sigma V^T\).
