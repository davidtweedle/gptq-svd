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


def evaluate_perplexity(
        model: nn.Module,
        tokenizer,
        dataset: str = "wikitext2",
        device: str = "cuda"
        ) -> float:
    """
    Evaluates model perplexity on wikitext2 test set.
    """
    logging.info(f"---Evaluating perplexity on {dataset} ---")

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

    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    model.eval()

    pbar = tqdm(range(0, seq_len, stride), desc="Evaluating PPL")
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / prev_end_loc)

    print(f"\nPerplexity: {ppl.item():.4f}")
    return ppl.item()
