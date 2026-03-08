import math
from typing import Callable, Optional, Tuple

import torch


def _act_order_perm(H: torch.Tensor, actorder: bool) -> torch.Tensor:
    n = H.shape[0]
    if actorder:
        return torch.argsort(torch.diag(H), descending=True)
    return torch.arange(n, device=H.device)


def _damped_inverse(H: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, float]:
    diag_mean = torch.mean(torch.diag(H))
    if diag_mean <= 0:
        diag_mean = torch.tensor(1.0, device=H.device, dtype=H.dtype)
    damp = float(alpha) * float(diag_mean)
    H_damped = H.clone()
    H_damped.diagonal().add_(damp)
    L = torch.linalg.cholesky(H_damped)
    iH = torch.cholesky_inverse(L)
    return iH, damp


def qronos_single_layer_update_ref(
    weight: torch.Tensor,
    weight_orig: torch.Tensor,
    H: torch.Tensor,
    G: torch.Tensor,
    quant_fn: Callable[[torch.Tensor], torch.Tensor],
    actorder: bool = True,
    alpha: float = 1e-6,
    beta: float = 1e4,
    blocksize: int = 128,
    return_debug: bool = False,
) -> Tuple[torch.Tensor, Optional[dict]]:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape={tuple(weight.shape)}")
    if weight.shape != weight_orig.shape:
        raise ValueError("weight and weight_orig shape mismatch")
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    if G.shape != H.shape:
        raise ValueError("G and H must have same shape")
    if weight.shape[1] != H.shape[0]:
        raise ValueError("weight in_features must match H/G shape")

    dev = weight.device
    orig_dtype = weight.dtype

    W_work = weight.to(dtype=torch.float32, device=dev).clone()
    W_orig = weight_orig.to(dtype=torch.float32, device=dev)
    Hp = H.to(dtype=torch.float32, device=dev).clone()
    Gp = G.to(dtype=torch.float32, device=dev).clone()

    perm = _act_order_perm(Hp, actorder)
    inv_perm = torch.argsort(perm)

    Wp = W_work[:, perm]
    W_orig_p = W_orig[:, perm]
    Hp = Hp[perm][:, perm]
    Gp = Gp[perm][:, perm]

    dead = torch.diag(Hp) == 0
    if dead.any():
        Wp[:, dead] = 0.0

    Dhi = torch.where(torch.diag(Hp) != 0, 1.0 / torch.diag(Hp), torch.zeros_like(torch.diag(Hp)))
    Uh = torch.triu(Hp, diagonal=1)
    iH, damp = _damped_inverse(Hp, alpha=alpha)

    Gw0 = W_orig_p.matmul(Gp[:, 0] * Dhi[0])
    Uv0 = Wp.matmul(Uh[0, :] * Dhi[0])
    Wp[:, 0] = Gw0 - Uv0

    if iH.shape[0] > 1:
        c = iH[0, 0]
        b = iH[1:, [0]]
        iH_tail = iH[1:, 1:] - (b.matmul(b.T) / c)
    else:
        iH_tail = iH.new_zeros((0, 0))

    if iH_tail.numel() > 0:
        I_damp = torch.eye(Hp.shape[0], device=dev, dtype=torch.float32) * damp
        Gh = Gp + I_damp
        q_hist = quant_fn(Wp)[:, :1].to(dtype=torch.float32, device=dev)
        Gw_tail = W_orig_p.matmul(Gh[:, 1:].matmul(iH_tail))
        Hq_tail = q_hist.matmul(Hp[:1, 1:].matmul(iH_tail))
        Wp[:, 1:] = Gw_tail - Hq_tail

        L = torch.linalg.cholesky(iH_tail * beta, upper=True) / math.sqrt(beta)
        n = Wp.shape[1]
        for i1 in range(1, n, blocksize):
            i2 = min(i1 + blocksize, n)
            count = i2 - i1
            h_inv_block = L[i1 - 1:i2 - 1, i1 - 1:i2 - 1]
            err_block = Wp.new_zeros((Wp.shape[0], count))
            for li in range(count):
                col = i1 + li
                q_col = quant_fn(Wp)[:, col].to(dtype=torch.float32, device=dev)
                w_col = Wp[:, col]
                d = h_inv_block[li, li]
                err = (w_col - q_col) / d
                err_block[:, li] = err
                Wp[:, col:i2] -= err.unsqueeze(1).matmul(h_inv_block[li, li:].unsqueeze(0))
            if i2 < n:
                Wp[:, i2:] -= err_block.matmul(L[i1 - 1:i2 - 1, i2 - 1:])

    Wq = quant_fn(Wp).to(dtype=torch.float32, device=dev)
    W_final = Wq[:, inv_perm].to(dtype=orig_dtype)

    dbg = None
    if return_debug:
        dbg = {
            "perm": perm.detach().cpu(),
            "dead_count": int(dead.sum().item()),
            "damp": float(damp),
        }
    return W_final, dbg
