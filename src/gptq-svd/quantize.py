import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
import gc
import jax
import utils
import data_utils
import model_utils
import eval_utils
from gptq_utils import gptq_svd_fwrd, Quantizer, gptq_ref_fwrd


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    jax.clear_caches()
    torch.cuda.synchronize()


def main():
    print(f"Starting quantization")
    args = utils.get_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    cleanup()
    print(f"Mode: {args.mode.upper()}")

    model, tokenizer = model_utils.get_model(args.model_id, args.device)
    ppl = eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)
    print(f"Baseline PPL: {ppl:.2f}")
    if args.mode == "baseline":
        return

    input_ids_list = data_utils.get_loaders(args.dataset, tokenizer, args.n_samples, args.seq_len)

    inps, outs, layer_kwargs = model_utils.capture_initial_inputs(
            model, input_ids_list, device=args.device
            )
    layers = model_utils.get_layers(model)

    layer_inputs = {}

    def add_batch(name):
        def hook(module, input, output):
            inp = input[0].detach()
            if len(inp.shape) == 3:
                inp = inp.squeeze(0)
            if name not in layer_inputs:
                layer_inputs[name] = []
            layer_inputs[name].append(inp)
        return hook

    def get_submodule(root, name):
        parts = name.split('.')
        curr = root
        for p in parts:
            curr = getattr(curr, p)
        return curr
    print(f"\n--- Starting {args.mode.upper()} Quantization ---")
    start_time = time.time()

    for i, layer in enumerate(layers):
        print(f"Processing Layer {i}/{len(layers)}...")

        groups = model_utils.get_sequenced_groups(layer)

        for group_idx, group_names in enumerate(groups):
            print(f"  -> Group {group_idx+1}: {group_names}")

            handles = []
            for name in group_names:
                submodule = get_submodule(layer, name)
                handles.append(submodule.register_forward_hook(add_batch(name)))
            for j in range(args.n_samples):
                inp_batch = inps[j].unsqueeze(0)
                batch_kwargs = {}
                if layer_kwargs:
                    for k, v in layer_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            batch_kwargs[k] = v.to(args.device)
                        elif isinstance(v, (tuple, list)):
                            moved_list = [x.to(args.device) if isinstance(x, torch.Tensor) else x for x in v ]
                            batch_kwargs[k] = tuple(moved_list) if isinstance(v, tuple) else moved_list
                        else:
                            batch_kwargs[k] = v
                batch_kwargs["use_cache"] = False
                layer(inp_batch, **batch_kwargs)
                del inp_batch, batch_kwargs
            for h in handles:
                h.remove()
            for name in group_names:
                print(f"Quantizing {name}")
                if name not in layer_inputs or len(layer_inputs[name]) == 0:
                    print(f"Warning: No inputs captured for {name}, skipping quantization")
                    # should round to nearest
                    continue
                X_list = layer_inputs[name]
                submodule = get_submodule(layer, name)
                W = submodule.weight.data.float().to(args.device)
                m, n = W.shape

                def make_stream_adapter():
                    for x_chunk in X_list:
                        yield x_chunk.to(torch.float32)
                out_weight = torch.zeros_like(W)
                quantizer = Quantizer(per_channel=True, w_bits=args.w_bits)
                if args.mode == "svd":
                    sketch_dim = int(n * args.sketch_ratio)
                    gptq_svd_fwrd(
                            sketch_dim=sketch_dim,
                            oversample=16,
                            k_iter=args.k_iter,
                            make_stream=make_stream_adapter,
                            weight_mat=W,
                            out_weight=out_weight,
                            quantizer=quantizer,
                            eps=args.eps,
                            update_block_size=256
                            )
                elif args.mode == "gptq":
                    gptq_ref_fwrd(
                            make_stream=make_stream_adapter,
                            weight_mat=W,
                            out_weight=out_weight,
                            quantizer=quantizer,
                            blocksize=128
                            )
                submodule.weight.copy_(out_weight)
                del out_weight, W, quantizer
                del X_list, layer_inputs[name]
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
            outs[j] = layer(inp_batch, **batch_kwargs)[0].squeeze(0)
        inps, outs = outs, inps
        cleanup()

    print(f"Quantization finished in {time.time() - start_time:.2f}s")
    print(f"Saving model to {args.save_path}...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    ppl = eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)
    print(f"{args.mode.upper()} PPL: {ppl:.2f}")


if __name__ == "__main__":
    main()
