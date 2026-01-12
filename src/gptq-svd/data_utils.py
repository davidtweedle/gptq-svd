"""
Data loading utilities
----------------------
Handles loading and tokenization of calibration datasets (WikiText2, C4)
for quantization.
"""

import random
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import List
import logging


def get_loaders(
        name: str,
        tokenizer: PreTrainedTokenizer,
        n_samples: int = 128,
        seq_len: int = 2048,
        batch_size: int = 1,
        seed: int = 42
        ) -> List[torch.Tensor]:
    """
    Factory function to get data loaders for supported datasets.
    """
    if name == "wikitext2":
        return get_wikitext2(tokenizer, n_samples, seq_len, batch_size, seed)
    elif name == "c4":
        return get_c4(tokenizer, n_samples, seq_len, batch_size, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_wikitext2(
        tokenizer: PreTrainedTokenizer,
        n_samples: int,
        seq_len: int,
        batch_size: int,
        seed: int = 42
        ) -> List[torch.Tensor]:
    """
    Loads wikitext2 and selects random chunks for calibration.
    """
    logging.info(f"Loading wikitext2... (total samples: {n_samples}, batch size: {batch_size})")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # merge into one massive string
    text = "\n\n".join(data["text"])

    # tokenize entire dataset
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    input_ids_list = []
    full_len = encodings.input_ids.shape[1]
    logging.info(f"[DATA] Full dataset tokens: {full_len}")
    random.seed(seed)
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, full_len - seq_len - 1)

        chunk = encodings.input_ids[:, i : i + seq_len].clone()
        samples.append(chunk)

    for i in range(0, len(samples), batch_size):
        group = samples[i: i + batch_size]
        if len(group) < batch_size:
            break
        batch = torch.cat(group, dim=0)
        input_ids_list.append(batch)
    logging.info(f"[DATA] Collected {len(input_ids_list)} random batches of batch size: {batch_size} and length {seq_len}.")
    return input_ids_list


def get_c4(
        tokenizer: PreTrainedTokenizer,
        n_samples: int,
        seq_len: int,
        batch_size: int,
        seed: int = 42
        ) -> List[torch.Tensor]:
    """
    Streams C4 dataset and filters for documents logn enough to fill the context window.
    """
    logging.info("Streaming C4 (en) (total samples: {n_samples}, batch size: {batch_size})...")
    # load streaming
    data = load_dataset("allenai/c4", "en", split="train", streaming=True)
    data = data.shuffle(seed=42, buffer_size=10000)

    input_ids_list = []
    current_batch_samples = []
    count = 0
    for batch in data:
        if count >= n_samples:
            break
        tokens = tokenizer(
                batch["text"],
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
                add_special_tokens=False
                ).input_ids
        if tokens.shape[1] >= seq_len:
            current_batch_samples.append(tokens[:, :seq_len])
            count += 1
            if len(current_batch_samples) == batch_size:
                input_ids_list.append(torch.cat(current_batch_samples, dim=0))
                current_batch_samples = []

    logging.info(f"[DATA] Collected {len(input_ids_list)} C4 batches")
    return input_ids_list
