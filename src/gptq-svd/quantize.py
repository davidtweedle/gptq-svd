import torch
import time
import gc
import json
import gptq_utils
import utils
import data_utils
import model_utils
import eval_utils
from gptq_utils import gptq_svd_qr_fwrd, Quantizer, gptq_ref_fwrd, Sketcher
import jax



def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    jax.clear_caches()
    torch.cuda.synchronize()


def log_mem(msg):
    print(f"[{msg}] GPU Alloc: {torch.cuda.memory_allocated()/1e9:.2f} | Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")


def main():
    print(f"Starting quantization")
    args = utils.get_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    cleanup()
    experiment_log = {
            "config": vars(args),
            "layer_stats": [],
            "metrics": {}
            }
    print(f"Starting experiment: eps={args.eps}, sketch={args.sketch_ratio}")
    print(f"Mode: {args.mode.upper()}")

    model, tokenizer = model_utils.get_model(args.model_id, args.device)
    model.config.use_cache = False
    if not hasattr(model, "seqlen"):
        model.seqlen = args.seq_len
    if args.mode == "baseline":
        ppl_baseline = eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)
        print(f"Baseline PPL: {ppl_baseline:.2f}")
        experiment_log["metrics"]["baseline_ppl"] = ppl_baseline
        with open(f"{args.save_path}/log.json", "w") as f:
            json.dump(experiment_log, f, indent=4)
        return

    input_ids_list = data_utils.get_loaders(args.dataset, tokenizer, args.n_samples, args.seq_len)

    inps, outs, layer_kwargs = model_utils.capture_initial_inputs(
            model, input_ids_list, device="cpu"
            )
    layers = model_utils.get_layers(model)

    layer_inputs = {}

    sketch_cache = {}

    def get_submodule(root, name):
        parts = name.split('.')
        curr = root
        for p in parts:
            curr = getattr(curr, p)
        return curr

    def capture_hook(name, submodule):
#        out_features, in_features = submodule.weight.shape
        def hook(module, input, output):
            x = input[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])

#            if args.mode == "svd":
#                if name not in sketch_cache:
#                    raw_rank = int(in_features * args.sketch_ratio)
#                    rank = min(
#                            raw_rank,
#                            out_features
#                            )
#                    sketch_cache[name] = torch.zeros(
#                            (rank, in_features),
#                            device=args.device,
#                            dtype=torch.float32
#                            )
#                Y = sketch_cache[name]
#                rank = Y.shape[0]
#                current_batch = x.shape[0]
#                scale = 1.0 / (rank ** 0.5)
#                omega = torch.randn((rank, current_batch), device=args.device, dtype=torch.float32) * scale
#                torch.addmm(input=Y, mat1=omega, mat2=x.to(args.device, torch.float32), beta=1.0, alpha=1.0, out=Y)
            if args.mode == "gptq":
                x_cpu = x.cpu()
                if name not in layer_inputs:
                    layer_inputs[name] = []
                layer_inputs[name].append(x_cpu)
        return hook

    print(f"\n--- Starting {args.mode.upper()} Quantization ---")
    start_global = time.time()

    for i, layer in enumerate(layers):
        print(f"Processing Layer {i + 1}/{len(layers)}...")
        layer_start_time = time.time()
        layer = layer.to(args.device)

        groups = model_utils.get_sequenced_groups(layer)

        for group_idx, group_names in enumerate(groups):
            print(f"  -> Group {group_idx+1}: {group_names}")

            handles = []
            for name in group_names:
                submodule = get_submodule(layer, name)
                out_features, in_features = submodule.weight.shape
                rank = int(in_features * args.sketch_ratio)
                sketch_cache[name] = Sketcher(submodule, rank)
                handles.append(submodule.register_forward_hook(sketch_cache[name].hook_fn))
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
                print(f"Quantizing {name}")
                submodule = get_submodule(layer, name)
                sketcher = sketch_cache[name]
                W = submodule.weight.data.float()
                m, n = W.shape

                quantizer = Quantizer(per_channel=True, w_bits=args.w_bits)
                module_stat = {"name": f"layer_{i}.{name}", "n_cols": n}
                if args.mode == "svd":
                    #if name not in sketch_cache:
                    #    print(f"Warning: No sketch for {name}")
                    #    continue
                    #Y_sketch = sketch_cache[name]
                    solve_start = time.time()
                    Y_sketch = sketch_cache[name].get_scaled_sketch()
                    final_W, used_rank = gptq_svd_qr_fwrd(
                            weight_mat=W,
                            input_sketch=Y_sketch,
                            quantizer=quantizer,
                            threshold=args.eps,
                            permute_order=None,
                            block_size=256
                            )
                    module_stat["solve_time"] = time.time() - solve_start
                    module_stat["rank_kept"] = used_rank
                    module_stat["rank_fraction"] = float(used_rank) / n
                    submodule.weight.copy_(final_W)
                    del Y_sketch, final_W, sketch_cache[name]
                elif args.mode == "gptq":
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
            cleanup()
        inps, outs = outs, inps
        cleanup()
        print(f"Layer {i} Done. Time: {time.time() - layer_start_time:.2f}s")
    del inps, outs, layer_inputs
    if 'layer_kwargs' in locals():
        del layer_kwargs
    cleanup()
    total_duration = time.time() - start_global
    experiment_log["metrics"]["total_time"] = total_duration

    print(f"Quantization finished in {total_duration:.2f}s")
    if not args.no_save:
        print(f"Saving model to {args.save_path}...")
        try:
            model.cpu()
            model.save_pretrained(args.save_path, safe_serialization=False)
            tokenizer.save_pretrained(args.save_path)
            print("Save successful.")
        except Exception as e:
            print(f"Standard save failed: {e}")
            print("Fallback: Dumping state_dict...")
            torch.save(model.state_dict(), os.path.join(args.save_path, "pytorch_model.bin"))
            model.config.save_pretrained(args.save_path)

    else:
        print("Skipping model weight save (--no_save was set).")
    model.to(args.device)

    ppl_q = eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)
    print(f"{args.mode.upper()} PPL: {ppl_q:.2f}")
    experiment_log["metrics"]["quantized_ppl"] = ppl_q
    log_file = os.path.join(args.save_path, "results.json")
    with open(log_file, "w") as f:
        json.dump(experiment_log, f, indent=4)
    print(f"Results saved to {log_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRASH: {e}")
        with open("crash_log.json", "w") as f:
            json.dump({"error": str(e)}, f)
        raise e
