import os
import time
import gc
import json
import logging
import torch
import jax
from tqdm import tqdm

from utils import get_args, setup_logging
import data_utils
import model_utils
import eval_utils
from gptq_utils import gptq_svd_qr_fwrd, Quantizer, gptq_ref_fwrd, Sketcher


def cleanup():
    """
    Forces aggressive garbage collection to prevent VRAM fragmentation.
    """
    gc.collect()
    torch.cuda.empty_cache()
    jax.clear_caches()
    torch.cuda.synchronize()


def get_submodule(root, name):
    """
    Helper to traverse 'layer.self_attn.q_proj' strings.
    """
    parts = name.split('.')
    curr = root
    for p in parts:
        curr = getattr(curr, p)
    return curr


def main():
    args = get_args()
    setup_logging(args.save_path)

    logging.info(f"Starting quantization for {args.model_id}...")
    logging.info(f"Config: Mode={args.mode.upper()} | Eps={args.eps} | SketchRatio={args.sketch_ratio}")

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    cleanup()

    experiment_log = {
            "config": vars(args),
            "layer_stats": [],
            "metrics": {}
            }

    model, tokenizer = model_utils.get_model(args.model_id, args.device)
    model.config.use_cache = False
    if not hasattr(model, "seqlen"):
        model.seqlen = args.seq_len

    if args.mode == "baseline":
        logging.info("Running Baseline PPL check (FP16)...")
        ppl_baseline = eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)
        logging.info(f"Baseline PPL: {ppl_baseline:.2f}")
        return

    input_ids_list = data_utils.get_loaders(args.dataset, tokenizer, args.n_samples, args.seq_len)

    inps, outs, layer_kwargs = model_utils.capture_initial_inputs(
            model, input_ids_list, device=args.device
            )
    layers = model_utils.get_layers(model)

    layer_inputs = {}
    sketch_cache = {}

    logging.info(f"\n--- Starting {args.mode.upper()} Pipeline ---")
    start_global = time.time()

    for i, layer in enumerate(layers):
        logging.info(f"Processing Layer {i + 1}/{len(layers)}...")
        layer_start_time = time.time()
        layer = layer.to(args.device)

        groups = model_utils.get_sequenced_groups(layer)

        for group_idx, group_names in enumerate(groups):
            logging.info(f"  -> Group {group_idx+1}: {group_names}")

            handles = []
            for name in group_names:
                submodule = get_submodule(layer, name)
                if args.mode == "svd":
                    out_features, in_features = submodule.weight.shape
                    rank = int(in_features * args.sketch_ratio)
                    sketch_cache[name] = Sketcher(submodule, rank, device=args.device)
                    handles.append(submodule.register_forward_hook(sketch_cache[name].hook_fn))
                elif args.mode == "gptq":
                    def get_hook(n):
                        def hook(module, inp, output):
                            x = inp[0].detach()
                            if x.dim() == 3:
                                x = x.reshape(-1, x.shape[-1])
                            if n not in layer_inputs:
                                layer_inputs[n] = []
                            layer_inputs[n].append(x.cpu())
                        return hook
                    handles.append(submodule.register_forward_hook(get_hook(name)))
            for j in range(args.n_samples):
                inp_batch = inps[j].unsqueeze(0).to(args.device)
                batch_kwargs = {}
                if layer_kwargs:
                    for k, v in layer_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            batch_kwargs[k] = v.detach().clone().to(args.device)
                        elif isinstance(v, (tuple, list)):
                            moved_list = [x.detach().clone().to(args.device) if isinstance(x, torch.Tensor) else x for x in v ]
                            batch_kwargs[k] = tuple(moved_list) if isinstance(v, tuple) else moved_list
                        else:
                            batch_kwargs[k] = v
                batch_kwargs["use_cache"] = False
                batch_kwargs["attention_mask"] = None
                out = layer(inp_batch, **batch_kwargs)[0]
                del inp_batch, batch_kwargs, out
                cleanup()
            for h in handles:
                h.remove()
            cleanup()
            layer_ranks = []
            for name in group_names:
                submodule = get_submodule(layer, name)
                W = submodule.weight.data.float()
                m, n = W.shape

                quantizer = Quantizer(per_channel=True, w_bits=args.w_bits)
                module_stat = {"name": f"layer_{i}.{name}", "n_cols": n}
                if args.mode == "svd":
                    logging.info(f"Quantizing {name} (SVD)")
                    solve_start = time.time()
                    Y_sketch = sketch_cache[name].get_scaled_sketch()
                    final_W, used_rank = gptq_svd_qr_fwrd(
                            weight_mat=W,
                            input_sketch=Y_sketch,
                            quantizer=quantizer,
                            threshold=args.eps,
                            threshold_method=args.threshold_method,
                            permute_order=None,
                            block_size=256
                            )
                    submodule.weight.copy_(final_W)
                    module_stat["solve_time"] = time.time() - solve_start
                    module_stat["rank_kept"] = used_rank
                    module_stat["rank_fraction"] = float(used_rank) / n
                    del Y_sketch, final_W, sketch_cache[name]
                elif args.mode == "gptq":
                    logging.info(f"Quantizing {name} (Ref-GPTQ)...")
                    if name not in layer_inputs:
                        continue
                    X_list = layer_inputs[name]
                    out_weight = torch.zeros_like(W)
                    def make_stream_adapter():
                        for x_chunk in X_list:
                            yield x_chunk.to(args.device, torch.float32)
                    gptq_ref_fwrd(
                            make_stream=make_stream_adapter,
                            weight_mat=W,
                            out_weight=out_weight,
                            quantizer=quantizer,
                            blocksize=128
                            )
                    submodule.weight.copy_(out_weight)
                    del out_weight, X_list, layer_inputs[name]
                del W, quantizer
                experiment_log["layer_stats"].append(module_stat)
                cleanup()
            layer_inputs.clear()
            sketch_cache.clear()
            cleanup()
        for j in range(args.n_samples):
            inp_batch = inps[j].unsqueeze(0).to(args.device)
            batch_kwargs = {}
            if layer_kwargs:
                for k, v in layer_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        batch_kwargs[k] = v.to(args.device)
                    elif isinstance(v, (tuple, list)):
                        moved_list = [x.to(args.device) if isinstance(x, torch.Tensor) else x for x in v]
                        batch_kwargs[k] = tuple(moved_list) if isinstance(v, tuple) else moved_list
                    else:
                        batch_kwargs[k] = v
            batch_kwargs['use_cache'] = False
            batch_kwargs['attention_mask'] = None
            outs[j] = layer(inp_batch, **batch_kwargs)[0].squeeze(0).to("cpu")
            del inp_batch, batch_kwargs
        inps, outs = outs, inps
        cleanup()
        logging.info(f"Layer {i} Done. Time: {time.time() - layer_start_time:.2f}s")
    del inps, outs, layer_inputs
    if 'layer_kwargs' in locals():
        del layer_kwargs
    cleanup()
    total_duration = time.time() - start_global
    experiment_log["metrics"]["total_time"] = total_duration

    logging.info(f"Quantization finished in {total_duration:.2f}s")
    if not args.no_save:
        logging.info(f"Saving model to {args.save_path}...")
        try:
            model.cpu()
            model.save_pretrained(args.save_path, safe_serialization=False)
            tokenizer.save_pretrained(args.save_path)
            logging.info("Save successful.")
        except Exception as e:
            logging.error(f"Standard save failed: {e}")
            torch.save(model.state_dict(), os.path.join(args.save_path, "pytorch_model.bin"))
            model.config.save_pretrained(args.save_path)

    else:
        logging.info("Skipping model weight save (--no_save was set).")
    model.to(args.device)

    ppl_q = eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)
    logging.info(f"{args.mode.upper()} PPL: {ppl_q:.2f}")
    experiment_log["metrics"]["quantized_ppl"] = ppl_q
    log_file = os.path.join(args.save_path, "results.json")
    with open(log_file, "w") as f:
        json.dump(experiment_log, f, indent=4)
    logging.info(f"Results saved to {log_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"CRASH: {e}")
        with open("crash_log.json", "w") as f:
            json.dump({"error": str(e)}, f)
        raise e
