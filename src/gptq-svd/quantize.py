import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
import gc
import json
import jax
import utils
import data_utils
import model_utils
import eval_utils
from gptq_utils import gptq_svd_qr_fwrd, Quantizer, gptq_ref_fwrd


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
    ppl_baseline = eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)
    print(f"Baseline PPL: {ppl_baseline:.2f}")
    experiment_log["metrics"]["baseline_ppl"] = ppl_baseline
    if args.mode == "baseline":
        with open(f"{args.save_path}/log.json", "w") as f:
            json.dump(experiment_log, f, indent=4)
        return

    input_ids_list = data_utils.get_loaders(args.dataset, tokenizer, args.n_samples, args.seq_len)

    inps, outs, layer_kwargs = model_utils.capture_initial_inputs(
            model, input_ids_list, device="cpu"
            )
    layers = model_utils.get_layers(model)

    layer_inputs = {}

    def add_batch(name):
        def hook(module, input, output):
            inp = input[0].detach()
            if len(inp.shape) == 3:
                inp = inp.squeeze(0)
            inp_cpu = inp.detach().clone().cpu()
            if name not in layer_inputs:
                layer_inputs[name] = []
            layer_inputs[name].append(inp_cpu)
        return hook

    def get_submodule(root, name):
        parts = name.split('.')
        curr = root
        for p in parts:
            curr = getattr(curr, p)
        return curr
    print(f"\n--- Starting {args.mode.upper()} Quantization ---")
    start_global = time.time()

    for i, layer in enumerate(layers):
        print(f"Processing Layer {i}/{len(layers)}...")
        layer_start_time = time.time()
        layer = layer.to(args.device)

        groups = model_utils.get_sequenced_groups(layer)

        for group_idx, group_names in enumerate(groups):
            print(f"  -> Group {group_idx+1}: {group_names}")

            handles = []
            for name in group_names:
                submodule = get_submodule(layer, name)
                handles.append(submodule.register_forward_hook(add_batch(name)))
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
                if name not in layer_inputs or len(layer_inputs[name]) == 0:
                    print(f"Warning: No inputs captured for {name}, skipping quantization")
                    # should round to nearest
                    continue
                X_list = layer_inputs[name]
                submodule = get_submodule(layer, name)
                W = submodule.weight.data.float()
                m, n = W.shape

                def make_stream_adapter():
                    for x_chunk in X_list:
                        yield x_chunk.to(args.device, torch.float32)
                quantizer = Quantizer(per_channel=True, w_bits=args.w_bits)
                module_stat = {"name": f"layer_{i}.{name}", "n_cols": n}
                if args.mode == "svd":
                    sketch_start = time.time()
                    sketch_dim = int(n * args.sketch_ratio)
                    Y_sketch = torch.zeros((sketch_dim, n), device=args.device, dtype=torch.float32)
                    for x_chunk in make_stream_adapter():
                        current_batch = x_chunk.shape[0]
                        omega = torch.randn((sketch_dim, current_batch), device=args.device, dtype=torch.float32) / (sketch_dim ** 0.5)
                        Y_sketch += omega @ x_chunk
                        del omega, x_chunk
                    module_stat["sketch_time"] = time.time() - sketch_start
                    solve_start = time.time()
                    final_W, used_rank = gptq_svd_qr_fwrd(
                            weight_mat=W,
                            input_sketch=Y_sketch,
                            quantizer=quantizer,
                            threshold=args.eps,
                            permute_order=None
                            )
                    module_stat["solve_time"] = time.time() - solve_start
                    module_stat["rank_kept"] = used_rank
                    module_stat["rank_fraction"] = float(used_rank) / n
                    submodule.weight.copy_(final_W)
                    del Y_sketch, final_W
                elif args.mode == "gptq":
                    out_weight = torch.zeros_like(W)
                    gptq_ref_fwrd(
                            make_stream=make_stream_adapter,
                            weight_mat=W,
                            out_weight=out_weight,
                            quantizer=quantizer,
                            blocksize=128
                            )
                    submodule.weight.copy_(out_weight)
                    del out_weight
                del W, quantizer
                del X_list, layer_inputs[name]
                experiment_log["layer_stats"].append(module_stat)
                cleanup()
            layer_inputs.clear()
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
    print(f"Saving model to {args.save_path}...")
    try:
        model.cpu()
        model.save_pretrained(args.save_path, safe_serialization=False)
        tokenizer.save_pretrained(args.save_path)
        print("Save successful.")
    except Exception as e:
        print(f"Standard save failed: {e}")
        print(f"Fallback: Dumping state_dict...")
        torch.save.model.state_dict(), os.path.join(args.save_path, "pytorch_model.bin"))
        model.config.save_pretrained(args.save_path)

    model.to(args.device)

    ppl_q = eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)
    print(f"{args.mode.upper()} PPL: {ppl_q:.2f}")
    experiment_log["metrics"]["quantized_ppl"] = ppl_q
    log_file = os.path.join(args.save_path, "results.json")
    with open(log_file, "w") as f:
        json.dump(experiment_log, f, indent=4)
    print(f"Results saved to {log_file}")


if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(max_entries=100000)
    try:
        main()
    except Exception as e:
        print("Dumping memory snapshot")
        torch.cuda.memory._dump_snapshot("oom_snapshot.pickle")
        raise e
