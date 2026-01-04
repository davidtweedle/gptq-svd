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
from typing import List, Optional


def get_loaders(
        name: str,
        tokenizer: PreTrainedTokenizer,
        n_samples: int = 128,
        seq_len: int = 2048,
        seed: int = 42
        ) -> List[torch.Tensor]:
    """
    Factory function to get data loaders for supported datasets.
    """
    if name == "wikitext2":
        return get_wikitext2(tokenizer, n_samples, seq_len, seed)
    elif name == "c4":
        return get_c4(tokenizer, n_samples, seq_len, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_wikitext2(
        tokenizer: PreTrainedTokenizer,
        n_samples: int,
        seq_len: int,
        seed: int = 42
        ) -> List[torch.Tensor]:
    """
    Loads wikitext2 and selects random chunks for calibration.
    """
    print("[DATA] Loading wikitext2...")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # merge into one massive string
    text = "\n\n".join(data["text"])

    # tokenize entire dataset
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    input_ids_list = []
    full_len = encodings.input_ids.shape[1]
    print(f"[DATA] Full dataset tokens: {full_len}")
    random.seed(seed)
    for _ in range(n_samples):
        i = random.randint(0, full_len - seq_len - 1)

        chunk = encodings.input_ids[:, i : i + seq_len].clone()
        input_ids_list.append(chunk)
    print(f"[DATA] Collected {len(input_ids_list)} random samples of length {seq_len}")
    return input_ids_list



def get_c4(
        tokenizer: PreTrainedTokenizer,
        n_samples: int,
        seq_len: int,
        seed: int = 42
        ) -> List[torch.Tensor]:
    """
    Streams C4 dataset and filters for documents logn enough to fill the context window.
    """
    print("[DATA] Streaming C4 (en)...")
    # load streaming
    data = load_dataset("allenai/c4", "en", split="train", streaming=True)
    data = data.shuffle(seed=42, buffer_size=10000)

    input_ids_list = []
    for batch in data:
        if len(input_ids_list) >= n_samples:
            break
        tokens = tokenizer(
                batch["text"],
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
                add_special_tokens=False
                ).input_ids
        if tokens.shape[1] >= seq_len:
            input_ids_list.append(tokens[:, :seq_len])

    print(f"[DATA] Collected {len(input_ids_list)} C4 samples")
    return input_ids_list
