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
from typing import Optional

logger = logging.getLogger(__name__)

def evaluate_perplexity(
        model: nn.Module,
        tokenizer,
        dataset: str = "wikitext2",
        device: str = "cuda",
        batch_size=4,
        stride=512
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
    total_nll = 0.0
    total_tokens = 0

    pbar = tqdm(range(0, len(requests), batch_size), desc="  [eval] Progress", leave=False)
    for i in pbar:
        batch_requests = requests[i: i + batch_size]

        input_ids_list = []
        target_ids_list = []

        attention_mask_list = []
        for req in batch_requests:
            inp = encodings.input_ids[:, req["begin"]:req["end"]]
            tar = inp.clone()
            tar[:, :-req["target_len"]] = -100
            pad_len = max_length - inp.shape[1]
            mask = torch.ones_like(inp)
            if pad_len > 0:
                inp = torch.cat([inp, torch.full((1, pad_len), pad_token_id)], dim=1)
                tar = torch.cat([tar, torch.full((1, pad_len), -100)], dim=1)
                mask = torch.cat([mask, torch.zeros((1, pad_len), dtype=torch.long)], dim=1)
            input_ids_list.append(inp)
            target_ids_list.append(tar)
            attention_mask_list.append(mask)
        input_ids = torch.cat(input_ids_list, dim=0).to(device)
        target_ids = torch.cat(target_ids_list, dim=0).to(device)
        attention_mask = torch.cat(attention_mask_list, dim=0).to(device)

        outputs = model(input_ids, labels=target_ids, attention_mask=attention_mask)
        num_active_tokens = (target_ids != -100).sum().item()
        if num_active_tokens > 0:
            total_nll += outputs.loss.float().item() * num_active_tokens
            total_tokens += num_active_tokens

        if total_tokens > 0:
            current_ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
            pbar.set_postfix({"ppl": f"{current_ppl:.2f}"})
    if total_tokens > 0:
        final_ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
    else:
        final_ppl = float('inf')
    return final_ppl
