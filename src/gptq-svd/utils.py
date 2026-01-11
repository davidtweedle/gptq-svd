"""
General utilities
-----------------
Argument parsing and logging configuration.
"""

import argparse
import logging
import sys
import os

def setup_logging(save_path: str = None, log_level: str = "INFO"):
    """
    Configures the root logger to print to console and optionally to a file.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        log_file = os.path.join(save_path, "quantization.log")
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers
            )


def get_args():
    parser = argparse.ArgumentParser(
            description="GPTQ-SVD: Low-rank aware quantization for LLMs",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )


    # --- Model Configuration ---
    model_group = parser.add_argument_group("Model Configuration")

    model_group.add_argument(
            "--model_id", type=str, default="Qwen/Qwen3-8B",
            help="HuggingFace model identifier or local path"
            )
    model_group.add_argument(
            "--device", type=str, default="cuda",
            help="Compute device (cuda/cpu)"
            )
    model_group.add_argument(
            "--seed", type=int, default=42,
            help="Random seed for reproducibility"
            )

    # --- Data Configuration ---
    data_group = parser.add_argument_group("Data Configuration")

    data_group.add_argument(
            "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"],
            help="Calibration dataset to use"
            )
    data_group.add_argument(
            "--n_samples", type=int, default=128,
            help="Number of calibration samples to capture"
            )
    data_group.add_argument(
            "--seq_len", type=int, default=2048,
            help="Sequence length for calibration"
            )
    data_group.add_argument(
            "--batch_size", type=int, default=8,
            help="Batch size for processing (default: 8)"
            )


    # --- Quantization Configuration ---
    quant_group = parser.add_argument_group("Quantization Parameters")
    quant_group.add_argument(
            "--w_bits", type=int, default=4, choices=[2, 3, 4, 8],
            help="Target bit-width for quantized weights"
            )
    quant_group.add_argument(
            "--group_size", type=int, default=-1, choices=[-1, 128],
            help="Group size for block scaling"
            )
    quant_group.add_argument(
            "--eps", type=float, default=1e-2,
            help="Threshold strength. For 'mean_trimmed', it is relative to the ref value. For 'energy', it is the allowed error variance."
            )
    quant_group.add_argument(
            "--sketch_ratio", type=float, default=4.0,
            help="Ratio of sketch size to input dimension (d = ratio * n)"
            )
    quant_group.add_argument(
            "--mode", type=str, choices=["svd", "gptq", "eigh", "test", "baseline"], default="svd",
            help="Quantization Strategy: 'svd' (Ours), 'gptq' (Reference), or 'baseline' (RTN)"
            )
    quant_group.add_argument(
            "--threshold_method", type=str, default="mean_trimmed", choices=["mean_trimmed", "energy"],
            help="Strategy for rank selection. 'mean_trimmed' uses mean(S[1:32]). 'energy' preserves (1-eps) variance."
            )
    quant_group.add_argument(
            "--actorder", action="store_true",
            help="Enable actorder for reference GPTQ."
            )
    quant_group.add_argument(
            "--damp_percent", type=float, default=0.01,
            help="Damping fraction for reference gptq (default: 0.01)"
            )

    # --- Output Configuration ---
    out_group = parser.add_argument_group("Output Configuration")
    out_group.add_argument(
            "--save_path", type=str, default="./output",
            help="Directory to save quantized model and logs"
            )
    out_group.add_argument(
            "--no_save", action="store_true",
            help="If set, skips saving the final model weights"
            )

    args = parser.parse_args()
    return args
