"""
Model Utilities
---------------
Helper functions to load LLMs, identify their layers, and capture
calibration inputs for quantization.
"""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from typing import Tuple, List, Dict, Any
import logging


def prepare_batch_kwargs(v, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    elif isinstance(v, (list, tuple)):
        return type(v)(prepare_batch_kwargs(x, device) for x in v)
    return v


def get_model(model_id: str, device: str = "cuda") -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Loads a Causal LM and its tokenizer in FP16.
    Automatically detects if Flash Attention is supported.
    """
    logging.info(f"Loading {model_id}...")

    # check for flash attention 2
    attn_implementation = "eager"
    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                import flash_attn
                attn_implementation = "flash_attention_2"
                logging.info("[MODEL] Using Flash attention 2")
    except ImportError:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            low_cpu_mem_usage=True
            )
    model.eval()
    return model, tokenizer


def get_layers(model: nn.Module) -> nn.ModuleList:
    """
    Retrieves the list of Transformer layers from the model.
    Supports standard architectures (Llama, Qwen, so far).
    """
    if hasattr(model, "model"):

        if hasattr(model.model, "layers"):
            return model.model.layers

        if hasattr(model.model, "decoder"):
            return model.model.decoder.layers

    if hasattr(model, "layers"):
        return model.layers

    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            return model.transformer.h

    raise ValueError("Could not find layers in model architecture")


def get_sequenced_groups(layer: nn.Module) -> List[List[str]]:
    """
    Defines the processing order for sub-layers during quantization.
    Groups are processed sequentially to respect dependencies.

    Returns:
        List of lists, where each inner list contains module names to process in parallel.
    """
    groups = []
    layer_modules = {name: m for name, m in layer.named_modules()}

    # Group 1
    g1 = [n for n in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'] if n in layer_modules]
    if g1:
        groups.append(g1)

    # Group 2
    g2 = [n for n in ['self_attn.o_proj'] if n in layer_modules]
    if g2:
        groups.append(g2)

    # Group 3
    g3 = [n for n in ['mlp.gate_proj', 'mlp.up_proj'] if n in layer_modules]
    if g3:
        groups.append(g3)

    # Group 4
    g4 = [n for n in ['mlp.down_proj'] if n in layer_modules]
    if g4:
        groups.append(g4)

    return groups


def find_linear_layers(module: nn.Module) -> Dict[str, nn.Linear]:
    """
    Recursively finds all Linear layers in a module.
    """
    res = {}
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            res[name] = m
    return res


def capture_initial_inputs(
        model: nn.Module,
        input_ids_list: List[torch.Tensor],
        device: str = "cuda"
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Captures the inputs to the FIRST layer of the model.
    This allows running layer-wise quantization sequentially.
    """
    logging.info("Capturing calibration inputs...")
    layers = get_layers(model)

    n_samples = len(input_ids_list)
    seq_len = input_ids_list[0].shape[1]
    hidden_dim = model.config.hidden_size
    dtype = next(model.parameters()).dtype

    inps = torch.zeros((n_samples, seq_len, hidden_dim), dtype=dtype, device=device)

    cache = {'i': 0, 'layer_kwargs': None}

    class Catcher(nn.Module):
        """
        Temporary wrapper to intercept the forward pass.
        Raises ValueError to stop execution after capturing.
        """
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']: cache['i'] + 1] = inp
            cache['i'] += 1

            if cache['layer_kwargs'] is None:
                cache['layer_kwargs'] = kwargs
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

    print(f"[MODEL] Captured {n_samples} input sequences.")
    return inps, cache['layer_kwargs']
