import argparse
import json
import os
import gc

import jax
import torch
from jax.dlpack import from_dlpack

from TruncGPTQ import data_utils, model_utils
from TruncGPTQ.gptq_utils import (
    HessianAccumulator,
    process_hessian,
    process_hessian_alt,
)
from TruncGPTQ.model_utils import prepare_batch_kwargs
from TruncGPTQ.utils import setup_logging


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    jax.clear_caches()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Hessian diagnostics for selected layers."
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"]
    )
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="mean_trimmed",
        choices=["mean_trimmed", "energy"],
    )
    parser.add_argument("--actorder", action="store_true")
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument(
        "--layer_indices",
        type=str,
        default="2,10,34",
        help="Comma-separated 0-based transformer layer indices.",
    )
    parser.add_argument(
        "--submodule_name",
        type=str,
        default="mlp.down_proj",
        help="Submodule to analyze inside each selected transformer layer.",
    )
    parser.add_argument("--save_path", type=str, default="./layer_stats")
    return parser.parse_args()


def get_submodule(root, name):
    curr = root
    for part in name.split("."):
        curr = getattr(curr, part)
    return curr


def get_gptq_raw_diag(H: torch.Tensor, damp_percent: float):
    H_double = H.to(dtype=torch.float64)
    mean_diag = torch.mean(torch.diagonal(H_double))
    if mean_diag == 0:
        mean_diag = H_double.new_tensor(1.0)

    for damp_exp in range(5):
        try:
            damp = (10 ** damp_exp) * damp_percent
            H_damped = H_double.clone()
            H_damped.diagonal().add_(damp * mean_diag)
            L = torch.linalg.cholesky(H_damped)
            H_inv = torch.cholesky_inverse(L)
            H_inv_factor = torch.linalg.cholesky(H_inv, upper=True)
            return torch.diagonal(H_inv_factor).clone().cpu()
        except RuntimeError:
            continue

    eye = torch.eye(H.shape[0], dtype=torch.float64, device=H.device)
    return torch.diagonal(eye).clone().cpu()


def get_trunc_raw_diags(
    H: torch.Tensor,
    threshold: float,
    threshold_method: str,
):
    H_double = H.to(dtype=torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(H_double)
    singular_vals = torch.sqrt(eigvals.clamp(min=1e-12)).flip(0)
    Vh = eigvecs.T.flip(0)

    if threshold_method == "energy":
        energy = singular_vals ** 2
        target = (1.0 - threshold) * torch.sum(energy)
        current_rank = int((torch.cumsum(energy, dim=0) <= target).sum().item())
        if current_rank < len(singular_vals):
            current_rank += 1
    elif threshold_method == "mean_trimmed":
        ref_k = min(33, len(singular_vals))
        ref_val = (
            torch.mean(singular_vals[1:ref_k]) if len(singular_vals) > 1 else singular_vals[0]
        )
        current_rank = int((singular_vals > threshold * ref_val).sum().item())
    else:
        current_rank = int(len(singular_vals))

    current_rank = max(1, min(current_rank, len(singular_vals)))
    singular_vals = singular_vals[:current_rank]
    Vh = Vh[:current_rank, :]

    H_sqrt = singular_vals.unsqueeze(1) * Vh
    H_sqrt_jax = from_dlpack(H_sqrt)
    _, R_x_jax, perm_jax = jax.scipy.linalg.qr(
        H_sqrt_jax, pivoting=True, mode="economic"
    )
    perm = torch.from_dlpack(perm_jax).long()
    R_x = torch.from_dlpack(R_x_jax)

    S_inv = 1.0 / singular_vals
    H_inv_partial = S_inv.unsqueeze(1) * Vh
    H_inv_permuted = H_inv_partial[:, perm]
    _, R_prime = torch.linalg.qr(H_inv_permuted, mode="reduced")

    return (
        torch.diagonal(R_prime).clone().cpu(),
        torch.diagonal(R_x).clone().cpu(),
        current_rank,
    )


def main():
    args = parse_args()
    setup_logging(args.save_path)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    target_layers = {int(x.strip()) for x in args.layer_indices.split(",") if x.strip()}
    os.makedirs(args.save_path, exist_ok=True)

    model, tokenizer = model_utils.get_model(args.model_id, "cpu")
    model.config.use_cache = False
    if not hasattr(model, "seqlen"):
        model.seqlen = args.seq_len

    input_ids_list = data_utils.get_loaders(
        args.dataset, tokenizer, args.n_samples, args.seq_len, args.seed
    )
    inps, layer_kwargs = model_utils.capture_initial_inputs(
        model, input_ids_list, device=args.device, batch_size=args.batch_size
    )
    outs = torch.zeros_like(inps)
    layers = model_utils.get_layers(model)

    manifest = []

    for layer_idx, layer in enumerate(layers):
        layer = layer.to(args.device)
        hook_handle = None
        accumulator = None
        target_name = f"layer_{layer_idx}.{args.submodule_name}"

        if layer_idx in target_layers:
            submodule = get_submodule(layer, args.submodule_name)
            in_features = submodule.weight.shape[1]
            accumulator = HessianAccumulator(in_features, device=args.device)

            def h_hook(module, inp, out):
                accumulator.add_batch(inp[0].detach())

            hook_handle = submodule.register_forward_hook(h_hook)

        for start in range(0, args.n_samples, args.batch_size):
            batch_inp = inps[start : start + args.batch_size].to(args.device)
            curr_batch_size = batch_inp.shape[0]
            batch_kwargs = {
                k: prepare_batch_kwargs(v, args.device)
                for k, v in layer_kwargs.items()
            }
            batch_kwargs["use_cache"] = False

            out_batch = layer(batch_inp, **batch_kwargs)
            if isinstance(out_batch, tuple):
                out_batch = out_batch[0]
            outs[start : start + curr_batch_size] = out_batch.cpu()

            del batch_inp, batch_kwargs, out_batch
            cleanup()

        if hook_handle is not None:
            hook_handle.remove()

            H = accumulator.get_hessian()
            H_cpu = H.to(dtype=torch.float64, device="cpu")
            eigvals = torch.linalg.eigvalsh(H_cpu)

            gptq_raw_diag = get_gptq_raw_diag(H, args.damp_percent)
            gptq_factor, gptq_perm = process_hessian(
                H=H,
                actorder=args.actorder,
                damp_percent=args.damp_percent,
            )
            trunc_raw_diag, trunc_pivot_diag, trunc_rank = get_trunc_raw_diags(
                H=H,
                threshold=args.eps,
                threshold_method=args.threshold_method,
            )
            trunc_factor, trunc_rx, trunc_perm = process_hessian_alt(
                H=H,
                threshold=args.eps,
                threshold_method=args.threshold_method,
            )

            out_file = os.path.join(
                args.save_path, f"layer_{layer_idx}_{args.submodule_name.replace('.', '_')}.pt"
            )
            torch.save(
                {
                    "layer_index": layer_idx,
                    "submodule_name": args.submodule_name,
                    "H_diag": torch.diagonal(H_cpu).clone(),
                    "H_eigvals": eigvals.clone(),
                    "gptq_raw_diag": gptq_raw_diag,
                    "gptq_diag_normalized": torch.diagonal(
                        gptq_factor.to(device="cpu")
                    ).clone(),
                    "gptq_perm": gptq_perm.to(device="cpu"),
                    "trunc_raw_diag": trunc_raw_diag,
                    "trunc_diag_normalized": torch.diagonal(
                        trunc_factor.to(device="cpu")
                    ).clone(),
                    "trunc_pivot_diag": trunc_pivot_diag,
                    "trunc_rx_diag_normalized": torch.diagonal(
                        trunc_rx.to(device="cpu")
                    ).clone(),
                    "trunc_perm": trunc_perm.to(device="cpu"),
                    "trunc_rank": int(trunc_rank),
                    "full_dim": int(H_cpu.shape[0]),
                },
                out_file,
            )

            manifest.append(
                {
                    "layer_index": layer_idx,
                    "submodule_name": args.submodule_name,
                    "output_file": out_file,
                    "trunc_rank": int(trunc_rank),
                    "full_dim": int(H_cpu.shape[0]),
                }
            )

            del H, H_cpu, eigvals, gptq_factor, gptq_perm, trunc_factor, trunc_rx, trunc_perm
            cleanup()

        inps, outs = outs, inps
        layer = layer.to("cpu")
        cleanup()

    manifest_file = os.path.join(args.save_path, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
