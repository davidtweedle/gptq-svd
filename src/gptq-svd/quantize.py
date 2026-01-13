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
from gptq_utils import gptq_svd_qr_fwrd, Quantizer, gptq_ref_fwrd, Sketcher, process_sketch, process_hessian, process_hessian_alt, HessianAccumulator
from model_utils import prepare_batch_kwargs


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

    model, tokenizer = model_utils.get_model(args.model_id, "cpu")
    model.config.use_cache = False
    if not hasattr(model, "seqlen"):
        model.seqlen = args.seq_len

    if args.mode == "baseline":
        logging.info("Running Baseline PPL check (FP16)...")
        ppl_baseline = eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)
        logging.info(f"Baseline PPL: {ppl_baseline:.2f}")
        return

    input_ids_list = data_utils.get_loaders(args.dataset, tokenizer, args.n_samples, args.seq_len)

    inps, layer_kwargs = model_utils.capture_initial_inputs(
            model, input_ids_list, device=args.device, batch_size=args.batch_size
            )
    outs = torch.zeros_like(inps)
    layers = model_utils.get_layers(model)
    rotary_emb = model.model.rotary_emb

    logging.info(f"\n--- Starting {args.mode.upper()} Pipeline ---")
    start_global = time.time()

    for i, layer in enumerate(layers):
        logging.info(f"Processing Layer {i + 1}/{len(layers)}...")
        layer_start_time = time.time()
        layer = layer.to(args.device)

        groups = model_utils.get_sequenced_groups(layer)

        for group_idx, group_names in enumerate(groups):
            logging.info(f"  -> Group {group_idx+1}: {group_names} Inputs")

            handles = []
            name = group_names[0]
            submodule = get_submodule(layer, name)
            out_features, in_features = submodule.weight.shape
            capture_start = time.time()
            if args.mode == "svd":
                rank = int(in_features * args.sketch_ratio)
                accumulator = Sketcher(submodule, rank, device=args.device)
                handles.append(submodule.register_forward_hook(accumulator.hook_fn))
            elif args.mode == "gptq" or args.mode == "eigh":
                accumulator = HessianAccumulator(in_features, device=args.device)
                def h_hook(module, inp, out):
                    accumulator.add_batch(inp[0].detach())
                handles.append(submodule.register_forward_hook(h_hook))
            elif args.mode == "test":
                accumulator_hessian = HessianAccumulator(in_features, device=args.device)
                rank = int(in_features * args.sketch_ratio)
                accumulator_sketch = Sketcher(submodule, rank, device=args.device)
                def h_hook(module, inp, out):
                    accumulator_hessian.add_batch(inp[0].detach())
                handles.append(submodule.register_forward_hook(accumulator_sketch.hook_fn))
                handles.append(submodule.register_forward_hook(h_hook))
            for j in range(0, args.n_samples, args.batch_size):
                batch_inp = inps[j: j + args.batch_size]
                curr_batch_size = batch_inp.shape[0]
                seq_len = batch_inp.shape[1]
                batch_kwargs = {k: prepare_batch_kwargs(v, args.device) for k, v in layer_kwargs.items()}
                batch_kwargs["use_cache"] = False
                batch_kwargs["attention_mask"] = None
                position_ids = torch.arange(seq_len, dtype=torch.long, device=args.device)
                batch_kwargs["position_ids"] = position_ids
                cos, sin = rotary_emb(batch_inp, position_ids)
                batch_kwargs["position_embeddings"] = (cos.to(args.device), sin.to(args.device))
                for k in ["cache_position", "past_key_values"]:
                    if k in batch_kwargs:
                        del batch_kwargs[k]
                out = layer(batch_inp, **batch_kwargs)[0]
                del batch_inp, batch_kwargs, out
                cleanup()
            for h in handles:
                h.remove()
            cleanup()
            layer_ranks = []
            if args.mode == 'svd':
                Y_sketch = accumulator.get_scaled_sketch()
                del accumulator
                logging.info(f"   Processing Sketch (Shape: {Y_sketch.shape})")
                process_start = time.time()
                R, perm = process_sketch(
                        sketch=Y_sketch,
                        threshold=args.eps,
                        threshold_method=args.threshold_method
                        )
                logging.info(f"   Sketch processed in {time.time() - process_start}")
                del Y_sketch
                cleanup()
                shared_stats = {"R": R, "perm": perm}
            elif args.mode == 'gptq':
                H_matrix = accumulator.get_hessian()
                del accumulator
                R, perm = process_hessian(
                        H=H_matrix,
                        actorder=args.actorder,
                        damp_percent=args.damp_percent
                        )
                shared_stats = {"R": R, "perm": perm}
            elif args.mode == 'eigh':
                H_matrix = accumulator.get_hessian()
                del accumulator
                R, perm = process_hessian_alt(
                        H=H_matrix,
                        threshold=args.eps,
                        threshold_method=args.threshold_method
                        )
                shared_stats = {"R": R, "perm": perm}
            elif args.mode == 'test':
                H_matrix = accumulator_hessian.get_hessian()
                Y_sketch = accumulator_sketch.get_scaled_sketch()
                H_eigvals = torch.linalg.eigvalsh(H_matrix)
                H_max_val = H_eigvals[-1]
                del H_eigvals, H_matrix, accumulator_hessian
                Y_max_sv = torch.linalg.svdvals(Y_sketch)[0]
                H_max_sqrt = torch.sqrt(H_max_val)
                ratio = H_max_sqrt / Y_max_sv
                shared_stats = {}
                logging.info(f"Spectral check for {name}:")
                logging.info(f"   sqrt(max_eig(H)): {H_max_sqrt.item():.4f}")
                logging.info(f"   max_sv(Y):        {Y_max_sv.item():.4f}")
                logging.info(f"   ratio:            {ratio.item():.4f}")

            logging.info(f"Time for processing inputs of {name}: {time.time() - capture_start}s")
            for name in group_names:
                submodule = get_submodule(layer, name)
                W = submodule.weight.data.float()
                m, n = W.shape

                quantizer = Quantizer(w_bits=args.w_bits, group_size=args.group_size)
                module_stat = {"name": f"layer_{i}.{name}", "n_cols": n}
                solve_start = time.time()
                if args.mode == "svd" or args.mode == "eigh":
                    logging.info(f"Quantizing {name} (SVD)")
                    final_W, used_rank = gptq_svd_qr_fwrd(
                            weight_mat=W,
                            R=shared_stats["R"],
                            quantizer=quantizer,
                            perm=shared_stats["perm"],
                            block_size=1024
                            )
                    module_stat["rank_kept"] = used_rank
                elif args.mode == "gptq":
                    logging.info(f"Quantizing {name} (Ref-GPTQ)...")
                    final_W = torch.zeros_like(W)
                    gptq_ref_fwrd(
                            H_inv_chol=shared_stats["R"],
                            weight_mat=W,
                            out_weight=final_W,
                            quantizer=quantizer,
                            blocksize=128,
                            perm=shared_stats["perm"]
                            )
                elif args.mode == "test":
                    final_W = W
                    logging.info(f"Test mode - skipping quantization.")
                submodule.weight.copy_(final_W)
                module_stat["solve_time"] = time.time() - solve_start
                experiment_log["layer_stats"].append(module_stat)
                cleanup()
            del shared_stats
            cleanup()
        for j in range(0, args.n_samples, args.batch_size):
            inp_batch = inps[j: j + args.batch_size]
            curr_batch_size = inp_batch.shape[0]
            seq_len = inp_batch.shape[1]
            batch_kwargs = {k: prepare_batch_kwargs(v, args.device) for k, v in layer_kwargs.items()}
            batch_kwargs["use_cache"] = False
            batch_kwargs["attention_mask"] = None
            position_ids = torch.arange(seq_len, dtype=torch.long, device=args.device).unsqueeze(0)
            batch_kwargs["position_ids"] = position_ids
            cos, sin = rotary_emb(batch_inp, position_ids)
            batch_kwargs["position_embeddings"] = (cos.to(args.device), sin.to(args.device))
            for k in ["cache_position", "past_key_values"]:
                if k in batch_kwargs:
                    del batch_kwargs[k]
            out_batch = layer(inp_batch, **batch_kwargs)[0]
            for i in range(curr_batch_size):
                outs[j + i] = out_batch[i]
            del inp_batch, batch_kwargs, out_batch
            cleanup()
        inps, outs = outs, inps
        layer = layer.to("cpu")
        cleanup()
        logging.info(f"Layer {i + 1} Done. Time: {time.time() - layer_start_time:.2f}s")
    del inps, outs
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
