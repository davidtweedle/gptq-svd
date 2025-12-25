import torch
import gc
import utils
import data_utils
import model_utils
import eval_utils
from gptq_utils import gptq_svd_fwrd, Quantizer


def main():
    print(f"Starting quantization")
    args = utils.get_args()
    torch.manual_seed(args.seed)
    model, tokenizer = model_utils.get_model(args.model_id, args.device)
    input_ids_list = data_utils.get_loaders(args.dataset, tokenizer, args.n_samples, args.seq_len)

    inps, outs, layer_kwargs = model_utils.capture_initial_inputs(
            model, input_ids_list, args.device
            )
    layers = model_utils.get_layers(model)

    layer_inputs = {}
    def add_batch(name):
        def hook(module, input, output):
            inp = input[0].detach()
            if len(inp.shape) == 3:
                inp = inp.squeeze(0)
            inp_cpu = inp.to("cpu", non_blocking=True).clone()
            if name not in layer_inputs:
                layer_inputs[name] = []
            layer_inputs[name].append(inp_cpu)
        return hook

    for i, layer in enumerate(layers):
        print(f"Processing Layer {i}/{len(layers)}...")
        layer = layer.to(args.device)

        subset = model_utils.find_linear_layers(layer)
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.n_samples):
            inp_batch = inps[j].to(args.device).unsqueeze(0)
            batch_kwargs = {}
            if layer_kwargs:
                for k, v in layer_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        batch_kwargs[k] = v.to(args.device)
                    else:
                        batch_kwargs[k] = v
            with torch.no_grad():
                layer(inp_batch, **batch_kwargs)
        for h in handles:
            h.remove()

        for name, submodule in subset.items():
            if name not in layer_inputs or len(layer_inputs[name]) == 0:
                print(f"Warning: No inputs captured for {name}, skipping quantization")
                # should round to nearest
                continue
            X_list = layer_inputs[name]
            W = submodule.weight.data.float()
            m, n = W.shape
            sketch_dim = int(n * args.sketch_ratio)

            def make_stream_adapter():
                for x_chunk in X_list:
                    yield x_chunk.to(torch.float32)
            
            out_weight = torch.zeros_like(W)
            quantizer = Quantizer(per_channel=True, w_bits=args.w_bits)

            gptq_svd_fwrd(
                    sketch_dim=sketch_dim,
                    oversample=16,
                    k_iter=args.k_iter,
                    make_stream=make_stream_adapter,
                    weight_mat=W,
                    out_weight=out_weight,
                    quantizer=quantizer,
                    eps=args.eps
                    )

            submodule.weight.copy_(out_weight)

            del X_list, layer_inputs[name]
            gc.collect()
        for j in range(args.n_samples):
            inp_batch = inps[j].to(args.device).unsqueeze(0)
            batch_kwargs = {}
            if layer_kwargs:
                for k, v in layer_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        batch_kwargs[k] = v.to(args.device)
                    else:
                        batch_kwargs[k] = v
            with torch.no_grad():
                outs[j] = layer(inp_batch, **batch_kwargs)[0].to("cpu")
        inps, outs = outs, inps
        layer = layer.to("cpu")
        torch.cuda.empty_cache()

    print(f"Saving model to {args.save_path}...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)

if __name__ == "__main__":
    main()
