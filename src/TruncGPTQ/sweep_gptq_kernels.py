import argparse
import csv
import json
import os
import time

import torch

from gptq_utils import Quantizer, gptq_fwrd


def parse_args():
    parser = argparse.ArgumentParser(
            description="Sweep GPTQ kernel implementations on dumped layer artifacts."
            )
    parser.add_argument(
            "--artifacts_dir",
            type=str,
            required=True,
            help="Directory containing manifest.json and artifact .pt files from dump_layer_artifacts.py",
            )
    parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="Where to write sweep outputs (defaults to artifacts_dir/sweep_results)",
            )
    parser.add_argument(
            "--kernel_impls",
            type=str,
            default="triton,cuda_lazy,cuda_lazy_reduce,cuda_immediate",
            help="Comma-separated kernel implementations for gptq_fwrd.",
            )
    parser.add_argument(
            "--accum_dtypes",
            type=str,
            default="fp32,fp64",
            help="Comma-separated accumulation dtypes to test.",
            )
    parser.add_argument("--w_bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group_size", type=int, default=128, choices=[-1, 128])
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_csv_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def rel_pred_error(W_orig, W_quant, R_x, perm):
    if R_x is None:
        return None
    W_o = W_orig[:, perm].to(torch.float64)
    W_q = W_quant[:, perm].to(torch.float64)
    R = R_x.to(torch.float64)
    y_orig_norm = torch.linalg.norm(W_o @ R.T)
    if y_orig_norm == 0:
        return None
    y_diff_norm = torch.linalg.norm((W_o - W_q) @ R.T)
    return float((y_diff_norm / y_orig_norm).item())


def main():
    args = parse_args()
    kernel_impls = parse_csv_list(args.kernel_impls)
    accum_dtypes = parse_csv_list(args.accum_dtypes)
    output_dir = args.output_dir or os.path.join(args.artifacts_dir, "sweep_results")
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(args.artifacts_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        raise ValueError(f"No artifacts found in {manifest_path}")

    results = []
    for item in artifacts:
        payload = torch.load(item["artifact_file"], map_location="cpu")
        layer_idx = int(payload["layer_index"])
        name = payload["submodule_name"]

        W_base = payload["weight"].to(torch.float32, device=args.device)
        H_inv_sqrt = payload["H_inv_sqrt"].to(torch.float32, device=args.device)
        perm = payload["perm"].to(torch.long, device=args.device)
        R_x = payload.get("R_x")
        if R_x is not None:
            R_x = R_x.to(torch.float32, device=args.device)

        for kernel_impl in kernel_impls:
            for accum_dtype in accum_dtypes:
                if kernel_impl in {"triton", "python"} and accum_dtype == "fp64":
                    continue

                times = []
                rel_werrs = []
                rel_perrs = []
                used_rank = None

                for _ in range(args.repeats):
                    quantizer = Quantizer(
                            w_bits=args.w_bits,
                            group_size=args.group_size,
                            sym=args.sym,
                            beta=args.beta,
                            )
                    W = W_base.clone()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    W_q, used_rank = gptq_fwrd(
                            weight_mat=W,
                            H_inv_sqrt=H_inv_sqrt,
                            quantizer=quantizer,
                            perm=perm,
                            block_size=args.block_size,
                            kernel_impl=kernel_impl,
                            accum_dtype=accum_dtype,
                            R_x=None,
                            )
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0

                    rel_werr = torch.linalg.norm(W_base - W_q) / torch.linalg.norm(W_base)
                    p_err = rel_pred_error(W_base, W_q, R_x, perm) if R_x is not None else None

                    times.append(dt)
                    rel_werrs.append(float(rel_werr.item()))
                    if p_err is not None:
                        rel_perrs.append(p_err)

                    del quantizer, W, W_q

                row = {
                        "layer_index": layer_idx,
                        "submodule_name": name,
                        "kernel_impl": kernel_impl,
                        "accum_dtype": accum_dtype,
                        "rank": int(used_rank) if used_rank is not None else None,
                        "time_sec_mean": float(sum(times) / len(times)),
                        "time_sec_min": float(min(times)),
                        "rel_weight_error_mean": float(sum(rel_werrs) / len(rel_werrs)),
                        "rel_prediction_error_mean": (
                            float(sum(rel_perrs) / len(rel_perrs)) if rel_perrs else None
                        ),
                }
                results.append(row)
                print(
                        f"[layer={layer_idx} {name}] kernel={kernel_impl:<16} "
                        f"accum={accum_dtype:<4} time={row['time_sec_mean']:.4f}s "
                        f"rel_werr={row['rel_weight_error_mean']:.6e} "
                        f"rel_perr={row['rel_prediction_error_mean']}"
                        )

        del W_base, H_inv_sqrt, perm
        if R_x is not None:
            del R_x

    json_path = os.path.join(output_dir, "kernel_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(output_dir, "kernel_sweep_results.csv")
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    summary_path = os.path.join(output_dir, "run_config.json")
    with open(summary_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved run config: {summary_path}")


if __name__ == "__main__":
    main()
