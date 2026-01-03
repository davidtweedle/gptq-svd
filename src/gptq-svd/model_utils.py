import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_id, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map=device,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            low_cpu_mem_usage=True
            )
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

def get_sequenced_groups(layer):
    groups = []
    layer_modules = {name: m for name, m in layer.named_modules()}
    g1 = [n for n in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'] if n in layer_modules]
    if g1:
        groups.append(g1)
    g2 = [n for n in ['self_attn.o_proj'] if n in layer_modules]
    if g2:
        groups.append(g2)

    g3 = [n for n in ['mlp.gate_proj', 'mlp.up_proj'] if n in layer_modules]
    if g3:
        groups.append(g3)

    g4 = [n for n in ['mlp.down_proj'] if n in layer_modules]
    if g4:
        groups.append(g4)

    return groups

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

    inps = torch.zeros((n_samples, seq_len, hidden_dim), dtype=torch.float16, device=device)
    cache = {'i': 0, 'layer_kwargs': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['layer_kwargs'] is None:
                params = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        params[k] = v.to(device)
                    elif isinstance(v, (tuple, list)):
                        moved = []
                        for x in v:
                            if isinstance(x, torch.Tensor):
                                moved.append(x.to(device))
                            else:
                                moved.append(x)
                        params[k] = tuple(moved) if isinstance(v, tuple) else moved
                    else:
                        params[k] = v
                cache['layer_kwargs'] = params
            raise ValueError("Stop forward")

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    model_device = next(model.parameters()).device
    for i, batch in enumerate(input_ids_list):
        batch = batch.to(model_device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    return inps, outs, cache['layer_kwargs']
