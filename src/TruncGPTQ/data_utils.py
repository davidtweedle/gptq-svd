"""
Data loading utilities
----------------------
Handles loading and tokenization of calibration datasets (WikiText2, C4)
for quantization.
"""

import random
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Literal
import logging

logger = logging.getLogger(__name__)


def get_loaders(
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        n_samples: int = 128,
        seq_len: int = 2048,
        seed: int = 42
        ) -> List[torch.Tensor]:
    """
    Factory function to get data loaders for supported datasets.
    Args:
        name: Name of the dataset to load. Supported: "wikitext2", "c4".
        tokenizer: The tokenizer to use for processing text.
        n_samples: Number of samples to collect for calibration.
        seq_len: The sequence length (context window) for each sample.
        seed: Random seed for reproducibility.

    Returns:
        A list of tensors, where each tensor has shape (1, seq_len).

    Raises:
        ValueError: If an unsupported dataset name is provided.
    """
    if name.lower() == "wikitext2":
        return get_wikitext2(tokenizer, n_samples, seq_len, seed)
    elif name.lower() == "c4":
        return get_c4(tokenizer, n_samples, seq_len, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: 'wikitext2', 'c4'")


def get_wikitext2(
        tokenizer: PreTrainedTokenizerBase,
        n_samples: int,
        seq_len: int,
        seed: int = 42
        ) -> List[torch.Tensor]:
    """
    Loads wikitext2 and selects random chunks for calibration.
    """
    logging.info(f"Loading wikitext2... (total samples: {n_samples})")
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    # merge into one massive string
    text = "\n\n".join(data["text"])

    # tokenize entire dataset
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    input_ids_list = []
    full_len = encodings.input_ids.shape[1]
    logging.info(f"[DATA] Full dataset tokens: {full_len}")
    random.seed(seed)
    for _ in range(n_samples):
        i = random.randint(0, full_len - seq_len - 1)

        chunk = encodings.input_ids[:, i : i + seq_len].clone()
        input_ids_list.append(chunk)

    logging.info(f"[DATA] Collected {len(input_ids_list)} random batches of length {seq_len}.")
    return input_ids_list


def get_c4(
        tokenizer: PreTrainedTokenizerBase,
        n_samples: int,
        seq_len: int,
        seed: int = 42
        ) -> List[torch.Tensor]:
    """
    Streams C4 dataset and filters for documents logn enough to fill the context window.
    """
    logging.info("Streaming C4 (en) (total samples: {n_samples})...")
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
            input_ids_list.append(tokens[:, :seq_len])
            count += 1

    logging.info(f"[DATA] Collected {len(input_ids_list)} C4 batches")
    return input_ids_list
