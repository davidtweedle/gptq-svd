import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def _flatten_inputs(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])
    elif x.dim() == 2:
        pass
    else:
        raise ValueError(f"Unsupported input rank for Qronos collector: shape={tuple(x.shape)}")
    return x


@dataclass
class QronosStats:
    H: torch.Tensor
    G: torch.Tensor
    nsamples: int


class QronosPairCollector:
    """
    Collects matched (quant-input, float-input) batches for one linear layer.
    H := Xq^T Xq
    G := Xf^T Xq
    """

    def __init__(self, layer: nn.Linear, dtype: torch.dtype = torch.float32):
        self.layer = layer
        self.dtype = dtype
        self.in_features = layer.in_features
        dev = layer.weight.device
        self.H = torch.zeros((self.in_features, self.in_features), device=dev, dtype=dtype)
        self.G = torch.zeros((self.in_features, self.in_features), device=dev, dtype=dtype)
        self.nsamples = 0
        self._pending_quant_input: Optional[torch.Tensor] = None

    def add_quant_input(self, x_quant: torch.Tensor) -> None:
        xq = _flatten_inputs(x_quant.detach()).to(device=self.H.device, dtype=self.dtype)
        self.H.addmm_(xq.T, xq)
        self.nsamples += xq.shape[0]
        self._pending_quant_input = xq

    def add_float_input(self, x_float: torch.Tensor) -> None:
        if self._pending_quant_input is None:
            raise RuntimeError("add_float_input called before matching add_quant_input")
        xf = _flatten_inputs(x_float.detach()).to(device=self.H.device, dtype=self.dtype)
        if xf.shape != self._pending_quant_input.shape:
            raise RuntimeError(
                "Mismatched quant/float input shapes: "
                f"{tuple(self._pending_quant_input.shape)} vs {tuple(xf.shape)}"
            )
        self.G.addmm_(xf.T, self._pending_quant_input)
        self._pending_quant_input = None

    def finalize(self, normalize_running_avg: bool = True) -> QronosStats:
        if self._pending_quant_input is not None:
            raise RuntimeError("Collector finalized with unmatched quant input pending")
        if normalize_running_avg and self.nsamples > 0:
            inv_n = 1.0 / float(self.nsamples)
            return QronosStats(H=self.H * inv_n, G=self.G * inv_n, nsamples=self.nsamples)
        return QronosStats(H=self.H, G=self.G, nsamples=self.nsamples)


class PairedInputHook:
    """
    Forward hook helper that routes captured inputs to a QronosPairCollector.
    mode='quant'  -> updates H and stores pending Xq.
    mode='float'  -> consumes pending Xq and updates G.
    """

    def __init__(self, collector: QronosPairCollector, mode: str):
        if mode not in {"quant", "float"}:
            raise ValueError(f"Unknown mode={mode}, expected one of ['quant', 'float']")
        self.collector = collector
        self.mode = mode

    def __call__(self, module: nn.Module, inp, out):
        x = inp[0]
        if self.mode == "quant":
            self.collector.add_quant_input(x)
        else:
            self.collector.add_float_input(x)
