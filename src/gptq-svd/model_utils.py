import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_id, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
            )
    if not hasattr(model, 'seqlen'):
        model.seqlen = 2048
    return model, tokenizer

def get_layers(model):
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model.model, "decoder"):
            return model.model.decoder.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Could not find layers in model architecture")

def find_linear_layers(module):
    res = {}
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            res[name] = m
    return res

def capture_initial_inputs(model, input_ids_list, device="cuda"):
    layers = get_layers(model)
    n_samples = len(input_ids_list)
    seq_len = input_ids_list[0].shape[1]
    hidden_dim = model.config.hidden_size

    inps = torch.zeros((n_samples, seq_len, hidden_dim), dtype=torch.float16, device='cpu')
    cache = {'i': 0, 'layer_kwargs': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to('cpu')
            cache['i'] += 1
            if cache['layer_kwargs'] is None:
                cache['layer_kwargs'] = {k: v for k, v in kwargs.items()}
            raise ValueError("Stop forward")

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    for batch in input_ids_list:
        batch = batch.to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    return inps, outs, cache['layer_kwargs']
