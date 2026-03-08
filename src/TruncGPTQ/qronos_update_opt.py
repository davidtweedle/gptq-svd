from typing import Callable, Optional, Tuple

import torch

from qronos_update_ref import qronos_single_layer_update_ref


def qronos_single_layer_update_opt(
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
    """
    Optimization entry-point (currently passthrough to faithful reference).
    """
    return qronos_single_layer_update_ref(
            weight=weight,
            weight_orig=weight_orig,
            H=H,
            G=G,
            quant_fn=quant_fn,
            actorder=actorder,
            alpha=alpha,
            beta=beta,
            blocksize=blocksize,
            return_debug=return_debug,
            )
