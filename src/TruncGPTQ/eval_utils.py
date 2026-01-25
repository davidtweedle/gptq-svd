"""
Evaluation utilities
--------------------
Standard perplexity calculation using a strided sliding window.
Matches the logic used by Hugging Face's leaderboard.
"""

import torch
from torch import nn
from datasets import load_dataset
from tqdm import tqdm
import logging
import gc
from typing import Optional, List
import model_utils

logger = logging.getLogger(__name__)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def evaluate_perplexity(
        model: nn.Module,
        tokenizer,
        dataset: str = "wikitext2",
        device: str = "cuda",
        batch_size: int = 32,
        stride: int = 512
        ) -> float:
    """
    Evaluates model perplexity on wikitext2 test set.
    """
    logger.info(f"  [eval] Loading {dataset}...")

    if dataset == "wikitext2":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(testdata["text"])
    else:
        return -1

    encodings = tokenizer(text, return_tensors="pt")
    if hasattr(model, "seqlen"):
        max_length = model.seqlen
    elif hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    else:
        max_length = 2048

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    dataset_size = encodings.input_ids.size(1)

    logger.info(f"  [eval] Dataset tokens: {dataset_size} | Window: {max_length} | Stride: {stride}")

    requests = []
    prev_end_loc = 0
    for begin_loc in range(0, dataset_size, stride):
        end_loc = min(begin_loc + max_length, dataset_size)

        target_len = end_loc - prev_end_loc

        requests.append({
            "begin": begin_loc,
            "end": end_loc,
            "target_len": target_len
            })
        prev_end_loc = end_loc
        if end_loc == dataset_size:
            break

    model.eval()

    input_ids_list = []
    target_ids_list = []
    for req in requests:
        inp = encodings.input_ids[:, req["begin"]: req["end"]]
        tar = inp.clone()
        tar[:, :-req["target_len"]] = -100
        pad_len = max_length - inp.shape[1]
        if pad_len > 0:
            inp = torch.cat([inp, torch.full((1, pad_len), pad_token_id)], dim=1)
            tar = torch.cat([tar, torch.full((1, pad_len), -100)], dim=1)
        input_ids_list.append(inp)
        target_ids_list.append(tar)
    logger.info("  [eval] Capturing embeddings...")

    hidden_states, layer_kwargs = model_utils.capture_initial_inputs(model, input_ids_list, device=device, batch_size=batch_size)
    hidden_states = hidden_states.cpu()
    layers = model_utils.get_layers(model)

    for i, layer in enumerate(layers):
        layer = layer.to(device)
        seq_len = hidden_states.shape[1]
        hidden_dim = hidden_states.shape[2]
        dtype = hidden_states.dtype

        for j in range(0, hidden_states.shape[0], batch_size):

            end_idx = min(j + batch_size, hidden_states.shape[0])
            real_batch_size = end_idx - j

            if real_batch_size < batch_size:
                input_tensor = torch.zeros((batch_size, seq_len, hidden_dim), dtype=dtype, device=device)
                input_tensor[:real_batch_size] = hidden_states[j: end_idx].to(device)
            else:
                input_tensor = hidden_states[j: end_idx].to(device)
            batch_kwargs = {
                    k: model_utils.prepare_batch_kwargs(v, device)
                    for k, v in layer_kwargs.items()
                    }
            batch_kwargs["use_cache"] = False
            out = layer(input_tensor, **batch_kwargs)
            if isinstance(out, tuple):
                out = out[0]

            if real_batch_size < batch_size:
                out = out[:real_batch_size]

            hidden_states[j: end_idx] = out.cpu()
            del input_tensor, batch_kwargs, out
        layer = layer.cpu()
        cleanup()
        if (i + 1) % 10 == 0:
            logger.info(f"  [eval] Layer {i + 1}/{len(layers)} processed")

    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        final_norm = model.transformer.ln_f
    else:
        final_norm = nn.Identity()

    lm_head = model.lm_head
    final_norm = final_norm.to(device)
    lm_head = lm_head.to(device)

    total_nll = 0.0
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss()

    pbar = tqdm(range(0, hidden_states.shape[0], batch_size), desc="  [eval] Final Head", leave=False)
    for j in pbar:
        end_idx = min(j + batch_size, hidden_states.shape[0])
        real_batch_size = end_idx - j

        batch_states = hidden_states[j: end_idx].to(device)
        batch_targets = torch.stack(target_ids_list[j: end_idx]).to(device)

        batch_states = final_norm(batch_states)
        logits = lm_head(batch_states)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_targets[..., 1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        num_active_tokens = (shift_labels != -100).sum().item()
        if num_active_tokens > 0:
            total_nll += loss.float().item() * num_active_tokens
            total_tokens += num_active_tokens

        del batch_states, batch_targets, logits

        if total_tokens > 0:
            pbar.set_postfix({"ppl": f"{torch.exp(torch.tensor(total_nll / total_tokens)):.2f}"})

    final_norm = final_norm.cpu()
    lm_head = lm_head.cpu()
    cleanup()
    return torch.exp(torch.tensor(total_nll / total_tokens)).item()
