import argparse
import json
import re
from pathlib import Path

import pandas as pd


REL_ERR_RE = re.compile(r"Relative prediction error:\s*([0-9eE+\-\.]+)")
LAYER_RE = re.compile(
    r"INFO:\s+([A-Za-z0-9_\.]+)\s+\|\s+Rank:\s+([^\|]+)\s+\|\s+Time:\s+([0-9eE+\-\.]+)s"
)


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
    parser.add_argument(
        "--oracle_baseline_method",
        type=str,
        default="energy",
        help="Threshold method used as the per-layer baseline for the strongest-tolerable oracle",
    )
    parser.add_argument(
        "--oracle_baseline_eps",
        type=float,
        default=1e-6,
        help="Epsilon/value used for the baseline truncation config in the strongest-tolerable oracle",
    )
    parser.add_argument(
        "--oracle_tolerance_factor",
        type=float,
        default=1.1,
        help="Allowed multiplicative increase over baseline per-layer relative output error",
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


def parse_log(log_file: Path):
    parsed = []
    current_rel_err = None
    with open(log_file, "r") as f:
        for line in f:
            err_match = REL_ERR_RE.search(line)
            if err_match:
                current_rel_err = float(err_match.group(1))
                continue

            layer_match = LAYER_RE.search(line)
            if layer_match:
                submodule = layer_match.group(1)
                rank = parse_rank(layer_match.group(2))
                layer_time_s = float(layer_match.group(3))
                parsed.append((submodule, rank, layer_time_s, current_rel_err))
                current_rel_err = None
    return parsed


def load_rows(results_dir: Path):
    rows = []
    for exp_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        log_file = exp_dir / "quantization.log"
        if not log_file.exists():
            continue
        result_file = exp_dir / "results.json"
        config = load_config(result_file)
        if not result_file.exists():
            continue
        with open(result_file, "r") as f:
            payload = json.load(f)
        layer_stats = payload.get("layer_stats", [])
        parsed = parse_log(log_file)

        count = min(len(layer_stats), len(parsed))
        for js, lg in zip(layer_stats[:count], parsed[:count]):
            layer_name = js.get("name")
            layer_index = None
            submodule = lg[0]
            if isinstance(layer_name, str) and layer_name.startswith("layer_") and "." in layer_name:
                prefix, submodule_json = layer_name.split(".", 1)
                try:
                    layer_index = int(prefix.replace("layer_", ""))
                except ValueError:
                    layer_index = None
                if submodule_json:
                    submodule = submodule_json

            rows.append(
                {
                    "experiment": exp_dir.name,
                    "layer_index": layer_index,
                    "submodule": submodule,
                    "layer_name": layer_name,
                    "rank": js.get("rank", lg[1]),
                    "layer_time_s": js.get("time", lg[2]),
                    "relative_output_error": lg[3],
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


def print_best_energy_per_layer(df: pd.DataFrame):
    energy_df = df[df["threshold_method"] == "energy"].copy()
    if energy_df.empty:
        return
    rows = []
    for layer_name, sub in energy_df.groupby("layer_name"):
        best = sub.sort_values(
            ["relative_output_error", "rank", "eps"],
            ascending=[True, False, True],
        ).iloc[0]
        rows.append(
            {
                "layer_name": layer_name,
                "best_energy_eps": best["eps"],
                "rank": best["rank"],
                "relative_output_error": best["relative_output_error"],
                "quantized_ppl": best["quantized_ppl"],
                "experiment": best["experiment"],
            }
        )
    out = pd.DataFrame(rows).sort_values(["best_energy_eps", "relative_output_error"])
    print("\n=== BEST ENERGY EPS PER LAYER ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def print_best_percent_per_layer(df: pd.DataFrame):
    percent_df = df[df["threshold_method"] == "percent"].copy()
    if percent_df.empty:
        return
    rows = []
    for layer_name, sub in percent_df.groupby("layer_name"):
        best = sub.sort_values(
            ["relative_output_error", "rank", "eps"],
            ascending=[True, False, True],
        ).iloc[0]
        rows.append(
            {
                "layer_name": layer_name,
                "best_percent": best["eps"],
                "rank": best["rank"],
                "relative_output_error": best["relative_output_error"],
                "quantized_ppl": best["quantized_ppl"],
                "experiment": best["experiment"],
            }
        )
    out = pd.DataFrame(rows).sort_values(["best_percent", "relative_output_error"])
    print("\n=== BEST FIXED-PERCENT VALUE PER LAYER ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def print_strongest_tolerable_per_layer(
    df: pd.DataFrame,
    baseline_method: str,
    baseline_eps: float,
    tolerance_factor: float,
):
    baseline_df = df[
        (df["threshold_method"] == baseline_method) & (df["eps"] == baseline_eps)
    ].copy()
    if baseline_df.empty:
        print("\n=== STRONGEST-TOLERABLE PER LAYER ===")
        print(
            f"No baseline rows found for threshold_method={baseline_method!r}, eps={baseline_eps}."
        )
        return

    baseline = (
        baseline_df.groupby("layer_name")
        .agg(
            baseline_rel_err=("relative_output_error", "mean"),
            baseline_rank=("rank", "mean"),
            baseline_ppl=("quantized_ppl", "mean"),
        )
        .reset_index()
    )
    merged = df.merge(baseline, on="layer_name", how="inner")
    merged["oracle_tau"] = tolerance_factor * merged["baseline_rel_err"]
    feasible = merged[merged["relative_output_error"] <= merged["oracle_tau"]].copy()

    print("\n=== STRONGEST-TOLERABLE PER LAYER ===")
    print(
        "Baseline:",
        f"threshold_method={baseline_method}, eps={baseline_eps}, tolerance_factor={tolerance_factor}",
    )

    if feasible.empty:
        print("No feasible rows met the oracle tolerance.")
        return

    rows = []
    for layer_name, sub in feasible.groupby("layer_name"):
        chosen = sub.sort_values(
            [
                "rank",
                "relative_output_error",
                "eps",
            ],
            ascending=[True, True, True],
        ).iloc[0]
        rows.append(
            {
                "layer_name": layer_name,
                "oracle_tau": chosen["oracle_tau"],
                "baseline_rel_err": chosen["baseline_rel_err"],
                "baseline_rank": chosen["baseline_rank"],
                "chosen_method": chosen["threshold_method"],
                "chosen_eps": chosen["eps"],
                "chosen_rank": chosen["rank"],
                "rank_drop_vs_baseline": chosen["baseline_rank"] - chosen["rank"],
                "chosen_rel_err": chosen["relative_output_error"],
                "quantized_ppl": chosen["quantized_ppl"],
                "experiment": chosen["experiment"],
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["rank_drop_vs_baseline", "chosen_rel_err"], ascending=[False, True]
    )
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    policy = (
        out.groupby(["chosen_method", "chosen_eps"])
        .size()
        .reset_index(name="layers")
        .sort_values(["layers", "chosen_method", "chosen_eps"], ascending=[False, True, True])
    )
    print("\n=== STRONGEST-TOLERABLE POLICY MIX ===")
    print(policy.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main():
    args = parse_args()
    df = load_rows(Path(args.results_dir))
    if df.empty:
        raise ValueError(f"No parsable log records found under {args.results_dir}")

    print(f"Loaded {len(df)} per-layer records from {args.results_dir}")
    print_method_summary(df)
    print_layer_sensitivity(df)
    print_best_per_layer(df, args.top_k)
    print_best_energy_per_layer(df)
    print_best_percent_per_layer(df)
    print_strongest_tolerable_per_layer(
        df,
        baseline_method=args.oracle_baseline_method,
        baseline_eps=args.oracle_baseline_eps,
        tolerance_factor=args.oracle_tolerance_factor,
    )
    print_percent_layers(df)


if __name__ == "__main__":
    main()
