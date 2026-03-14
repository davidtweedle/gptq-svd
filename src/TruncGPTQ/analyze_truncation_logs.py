import argparse
import json
import re
from pathlib import Path

import pandas as pd


REL_ERR_RE = re.compile(r"Relative prediction error:\s+([0-9eE+\-\.]+)")
LAYER_RE = re.compile(r"layer_(\d+)\.([A-Za-z0-9_\.]+)\s+\|\s+Rank:\s+([^\|]+)\s+\|\s+Time:\s+([0-9eE+\-\.]+)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape quantization logs for per-layer relative output error and truncation sensitivity."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Top-level tuning_results_* directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many best configs to show per layer",
    )
    return parser.parse_args()


def parse_rank(rank_text: str):
    rank_text = rank_text.strip()
    if rank_text == "N/A":
        return None
    try:
        return float(rank_text)
    except ValueError:
        return None


def load_config(result_file: Path):
    if not result_file.exists():
        return {}
    with open(result_file, "r") as f:
        payload = json.load(f)
    cfg = payload.get("config", {}).copy()
    metrics = payload.get("metrics", {})
    cfg["quantized_ppl"] = metrics.get("quantized_ppl")
    return cfg


def parse_log(log_file: Path, config: dict):
    rows = []
    current_rel_err = None
    with open(log_file, "r") as f:
        for line in f:
            err_match = REL_ERR_RE.search(line)
            if err_match:
                current_rel_err = float(err_match.group(1))
                continue

            layer_match = LAYER_RE.search(line)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                submodule = layer_match.group(2)
                rank = parse_rank(layer_match.group(3))
                layer_time_s = float(layer_match.group(4))
                rows.append(
                    {
                        "experiment": log_file.parent.name,
                        "layer_index": layer_idx,
                        "submodule": submodule,
                        "layer_name": f"layer_{layer_idx}.{submodule}",
                        "rank": rank,
                        "layer_time_s": layer_time_s,
                        "relative_output_error": current_rel_err,
                        "threshold_method": config.get("threshold_method"),
                        "eps": config.get("eps"),
                        "mode": config.get("mode"),
                        "kernel_impl": config.get("kernel_impl"),
                        "large_update_impl": config.get("large_update_impl"),
                        "block_size": config.get("block_size"),
                        "w_bits": config.get("w_bits"),
                        "sym": config.get("sym"),
                        "beta": config.get("beta"),
                        "seed": config.get("seed"),
                        "quantized_ppl": config.get("quantized_ppl"),
                    }
                )
                current_rel_err = None
    return rows


def load_rows(results_dir: Path):
    rows = []
    for exp_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        log_file = exp_dir / "quantization.log"
        if not log_file.exists():
            continue
        config = load_config(exp_dir / "results.json")
        rows.extend(parse_log(log_file, config))
    return pd.DataFrame(rows)


def print_method_summary(df: pd.DataFrame):
    cols = ["threshold_method", "eps"]
    summary = (
        df.groupby(cols)
        .agg(
            runs=("experiment", "nunique"),
            mean_rel_err=("relative_output_error", "mean"),
            std_rel_err=("relative_output_error", "std"),
            mean_rank=("rank", "mean"),
            mean_ppl=("quantized_ppl", "mean"),
        )
        .reset_index()
        .sort_values(["threshold_method", "mean_rel_err", "eps"])
    )
    print("\n=== METHOD / EPS SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def print_layer_sensitivity(df: pd.DataFrame):
    layer_summary = (
        df.groupby("layer_name")
        .agg(
            min_rel_err=("relative_output_error", "min"),
            max_rel_err=("relative_output_error", "max"),
            rel_err_range=("relative_output_error", lambda s: s.max() - s.min()),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
        .reset_index()
        .sort_values("rel_err_range", ascending=False)
    )
    print("\n=== MOST TRUNCATION-SENSITIVE LAYERS ===")
    print(layer_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def print_best_per_layer(df: pd.DataFrame, top_k: int):
    print("\n=== BEST CONFIGS PER LAYER BY RELATIVE OUTPUT ERROR ===")
    for layer_name, sub in df.groupby("layer_name"):
        best = sub.sort_values(
            ["relative_output_error", "rank", "eps"],
            ascending=[True, False, True],
        ).head(top_k)
        cols = [
            "threshold_method",
            "eps",
            "rank",
            "relative_output_error",
            "layer_time_s",
            "quantized_ppl",
            "experiment",
        ]
        print(f"\n[{layer_name}]")
        print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def print_percent_layers(df: pd.DataFrame):
    percent_df = df[df["threshold_method"] == "percent"].copy()
    if percent_df.empty:
        return
    summary = (
        percent_df.groupby(["layer_name", "eps"])
        .agg(
            mean_rel_err=("relative_output_error", "mean"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
        .sort_values(["layer_name", "mean_rel_err", "eps"])
    )
    print("\n=== FIXED-PERCENT LAYER PREFERENCES ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main():
    args = parse_args()
    df = load_rows(Path(args.results_dir))
    if df.empty:
        raise ValueError(f"No parsable log records found under {args.results_dir}")

    print(f"Loaded {len(df)} per-layer records from {args.results_dir}")
    print_method_summary(df)
    print_layer_sensitivity(df)
    print_best_per_layer(df, args.top_k)
    print_percent_layers(df)


if __name__ == "__main__":
    main()
