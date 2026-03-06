import argparse
import torch

try:
    from . import gptq_ops as custom_kernels
    from .gptq_utils import Quantizer, triton_process_block
except ImportError:
    import gptq_ops as custom_kernels
    from gptq_utils import Quantizer, triton_process_block


def parse_args():
    parser = argparse.ArgumentParser(
            description="Debug parity between Triton block kernel and CUDA immediate kernel."
            )
    parser.add_argument("--artifact_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--w_bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group_size", type=int, default=128, choices=[-1, 128])
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--block_start", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
            "--kernel_impl",
            type=str,
            default="cuda_immediate",
            choices=["cuda_immediate", "cuda_lazy_reduce", "cuda_lazy"],
            help="CUDA kernel variant to compare against Triton.",
            )
    return parser.parse_args()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor):
    return float((a - b).abs().max().item())


def summarize_diff(name: str, a: torch.Tensor, b: torch.Tensor, tol: float):
    diff = (a - b).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    mismatch = int((diff > tol).sum().item())
    print(f"{name}: max={max_diff:.6e}, mean={mean_diff:.6e}, mismatches(>{tol})={mismatch}")


def prepare_tensors(payload, device, rows, quantizer):
    W = payload["weight"].to(device=device, dtype=torch.float32)
    H_inv_sqrt = payload["H_inv_sqrt"].to(device=device, dtype=torch.float32)
    perm = payload["perm"].to(device=device, dtype=torch.long)

    if rows > 0:
        W = W[:rows]

    quantizer.find_params(W)
    m, n = W.shape
    S_full, Z_full = quantizer.get_expanded_params(m, n)
    Wp = W[:, perm].contiguous()
    S = S_full[:, perm].to(device=device, dtype=torch.float32).contiguous()
    Z = Z_full[:, perm].to(device=device, dtype=torch.float32).contiguous()
    return Wp, H_inv_sqrt, S, Z


def run_triton_block(Wp, H_inv_sqrt, S, Z, i1, count, quantizer):
    i2 = i1 + count
    W1 = Wp[:, i1:i2].contiguous()
    S1 = S[:, i1:i2].contiguous()
    Z1 = Z[:, i1:i2].contiguous()
    H1 = H_inv_sqrt[i1:i2, i1:i2].contiguous()
    Q, E = triton_process_block(W1, S1, Z1, H1, quantizer)
    return Q, E


def run_cuda_block(Wp, H_inv_sqrt, S, Z, i1, count, quantizer, kernel_impl):
    Wc = Wp.clone()
    Err = torch.zeros_like(Wc)
    if kernel_impl == "cuda_immediate":
        custom_kernels.fused_gptq_step_immediate(
                Wc,
                H_inv_sqrt,
                S,
                Z,
                Err,
                i1,
                count,
                quantizer.min_q,
                quantizer.max_q,
                False,
                )
    elif kernel_impl == "cuda_lazy_reduce":
        custom_kernels.fused_gptq_step_lazy_reduce(
                Wc,
                H_inv_sqrt,
                S,
                Z,
                Err,
                i1,
                count,
                quantizer.min_q,
                quantizer.max_q,
                False,
                )
    else:
        custom_kernels.fused_gptq_step_lazy(
                Wc,
                H_inv_sqrt,
                S,
                Z,
                Err,
                i1,
                count,
                quantizer.min_q,
                quantizer.max_q,
                False,
                )
    i2 = i1 + count
    Q = Wc[:, i1:i2].contiguous()
    E = Err[:, i1:i2].contiguous()
    return Q, E


def main():
    args = parse_args()
    payload = torch.load(args.artifact_file, map_location="cpu")

    quantizer = Quantizer(
            w_bits=args.w_bits,
            group_size=args.group_size,
            sym=args.sym,
            beta=args.beta,
            )
    Wp, H_inv_sqrt, S, Z = prepare_tensors(
            payload, args.device, args.rows, quantizer
            )

    total_cols = Wp.shape[1]
    i1 = args.block_start
    count = min(args.block_size, total_cols - i1, H_inv_sqrt.shape[0] - i1)
    if count <= 0:
        raise ValueError("Invalid block selection; no columns to process.")

    print(f"Layer={payload.get('layer_index')} {payload.get('submodule_name')}")
    print(f"Testing block: start={i1}, count={count}, rows={Wp.shape[0]}")

    Qt, Et = run_triton_block(Wp, H_inv_sqrt, S, Z, i1, count, quantizer)
    Qc, Ec = run_cuda_block(Wp, H_inv_sqrt, S, Z, i1, count, quantizer, args.kernel_impl)

    summarize_diff(f"Q(triton vs {args.kernel_impl})", Qt, Qc, args.tol)
    summarize_diff(f"E(triton vs {args.kernel_impl})", Et, Ec, args.tol)

    first_bad = None
    for k in range(1, count + 1):
        Qt_k, Et_k = run_triton_block(Wp, H_inv_sqrt, S, Z, i1, k, quantizer)
        Qc_k, Ec_k = run_cuda_block(Wp, H_inv_sqrt, S, Z, i1, k, quantizer, args.kernel_impl)
        if max_abs_diff(Qt_k, Qc_k) > args.tol or max_abs_diff(Et_k, Ec_k) > args.tol:
            first_bad = k
            break

    if first_bad is None:
        print("Prefix parity check: no divergence detected within tolerance.")
    else:
        print(f"Prefix parity check: first divergence at prefix length k={first_bad}")


if __name__ == "__main__":
    main()
