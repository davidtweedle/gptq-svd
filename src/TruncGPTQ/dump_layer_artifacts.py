import argparse
import gc
import json
import os
import time

import jax
import torch

import data_utils
import model_utils
from gptq_utils import HessianAccumulator, process_hessian, process_hessian_alt
from model_utils import prepare_batch_kwargs
from utils import setup_logging


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    jax.clear_caches()


def parse_args():
    parser = argparse.ArgumentParser(
            description="Dump per-layer GPTQ artifacts for kernel-only sweeps."
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
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument(
            "--threshold_method",
            type=str,
            default="mean_trimmed",
            choices=["mean_trimmed", "energy"],
            )
    parser.add_argument("--actorder", action="store_true")
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument(
            "--hessian_mode",
            type=str,
            default="eigh",
            choices=["eigh", "gptq"],
            help="How to compute the H_inv_sqrt artifact.",
            )
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
    parser.add_argument("--save_path", type=str, default="./layer_artifacts")
    return parser.parse_args()


def get_submodule(root, name):
    curr = root
    for part in name.split("."):
        curr = getattr(curr, part)
    return curr


def parse_layer_indices(layer_indices_str):
    return sorted({int(x.strip()) for x in layer_indices_str.split(",") if x.strip()})


def main():
    args = parse_args()
    setup_logging(args.save_path)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    os.makedirs(args.save_path, exist_ok=True)

    target_layers = set(parse_layer_indices(args.layer_indices))
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
    t0 = time.time()

    for layer_idx, layer in enumerate(layers):
        layer = layer.to(args.device)
        hook_handle = None
        accumulator = None

        if layer_idx in target_layers:
            submodule = get_submodule(layer, args.submodule_name)
            in_features = submodule.weight.shape[1]
            accumulator = HessianAccumulator(in_features, device=args.device)

            def h_hook(module, inp, out):
                accumulator.add_batch(inp[0].detach())

            hook_handle = submodule.register_forward_hook(h_hook)

        for start in range(0, args.n_samples, args.batch_size):
            batch_inp = inps[start: start + args.batch_size].to(args.device)
            curr_batch_size = batch_inp.shape[0]
            batch_kwargs = {
                    k: prepare_batch_kwargs(v, args.device)
                    for k, v in layer_kwargs.items()
                    }
            batch_kwargs["use_cache"] = False

            out_batch = layer(batch_inp, **batch_kwargs)
            if isinstance(out_batch, tuple):
                out_batch = out_batch[0]
            outs[start: start + curr_batch_size] = out_batch.cpu()
            del batch_inp, batch_kwargs, out_batch
            cleanup()

        if hook_handle is not None:
            hook_handle.remove()
            submodule = get_submodule(layer, args.submodule_name)
            H = accumulator.get_hessian()

            if args.hessian_mode == "eigh":
                R, R_x, perm = process_hessian_alt(
                        H=H,
                        threshold=args.eps,
                        threshold_method=args.threshold_method
                        )
            else:
                R, perm = process_hessian(
                        H=H,
                        actorder=args.actorder,
                        damp_percent=args.damp_percent
                        )
                R_x = None

            payload = {
                    "model_id": args.model_id,
                    "dataset": args.dataset,
                    "layer_index": layer_idx,
                    "submodule_name": args.submodule_name,
                    "hessian_mode": args.hessian_mode,
                    "eps": args.eps,
                    "threshold_method": args.threshold_method,
                    "damp_percent": args.damp_percent,
                    "actorder": bool(args.actorder),
                    "weight": submodule.weight.detach().to(torch.float32).cpu(),
                    "H_inv_sqrt": R.detach().to(torch.float32).cpu(),
                    "perm": perm.detach().to(torch.long).cpu(),
            }
            if R_x is not None:
                payload["R_x"] = R_x.detach().to(torch.float32).cpu()

            out_file = os.path.join(
                    args.save_path,
                    f"layer_{layer_idx}_{args.submodule_name.replace('.', '_')}.pt",
                    )
            torch.save(payload, out_file)
            manifest.append(
                    {
                            "layer_index": layer_idx,
                            "submodule_name": args.submodule_name,
                            "artifact_file": out_file,
                            "shape": list(payload["weight"].shape),
                            "rank": int(payload["H_inv_sqrt"].shape[0]),
                    }
                    )
            del H, R, perm, payload
            if R_x is not None:
                del R_x
            cleanup()

        inps, outs = outs, inps
        layer = layer.to("cpu")
        cleanup()

    manifest_path = os.path.join(args.save_path, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
                {
                        "config": vars(args),
                        "total_layers": len(layers),
                        "targets": sorted(target_layers),
                        "elapsed_sec": time.time() - t0,
                        "artifacts": manifest,
                },
                f,
                indent=2,
                )


if __name__ == "__main__":
    main()
