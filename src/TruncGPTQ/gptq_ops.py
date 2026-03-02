import torch

try:
    from . import _C
except ImportError as e:
    _C = None
    print(f"Failed to import TruncGPTQ._C: {e}")
    raise

def fused_gptq_step(W, H, Scales, Zeros, Err, col_offset, block_cols=1024, qmin=-7.0, qmax=7.0):
    if _C is None:
        raise RuntimeError("CUDA extension not compiled.")

    total_cols = W.size(1)
    block_cols = min(block_cols, total_cols - col_offset)

    H_block = H[col_offset: col_offset + block_cols, col_offset: col_offset + block_cols].T.contiguous()
    S_block = Scales[:, col_offset: col_offset + block_cols].contiguous()
    Z_block = Zeros[:, col_offset: col_offset + block_cols].contiguous()
    W = W.contiguous()
    Err = Err.contiguous()

    _C.gptq_fused(W, H_block, S_block, Z_block, Err, total_cols, col_offset, block_cols, qmin, qmax)

    return W
