import argparse
import re
from pathlib import Path
from io import StringIO

import pandas as pd


ORACLE_START = "=== STRONGEST-TOLERABLE PER LAYER ==="
ORACLE_END = "=== STRONGEST-TOLERABLE POLICY MIX ==="


def parse_args():
    parser = argparse.ArgumentParser(
        description="Join spectral_stats.csv with strongest-tolerable oracle labels."
    )
    parser.add_argument(
        "--spectral_csv",
        type=str,
        required=True,
        help="Path to spectral_stats.csv",
    )
    parser.add_argument(
        "--oracle_summary",
        type=str,
        required=True,
        help="Path to truncation_log_summary_*.txt containing STRONGEST-TOLERABLE PER LAYER",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults next to spectral_csv as spectral_oracle_join.csv",
    )
    return parser.parse_args()


def parse_oracle_table(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == ORACLE_START)
        end = next(i for i, line in enumerate(lines) if line.strip() == ORACLE_END)
    except StopIteration as exc:
        raise ValueError(f"Could not find oracle section markers in {path}") from exc

    section = [
        line.rstrip("\n")
        for line in lines[start + 1 : end]
        if line.strip() and not line.startswith("Baseline:")
    ]
    if len(section) < 2:
        raise ValueError(f"Oracle section in {path} is empty")

    table_text = "\n".join(section)
    df = pd.read_fwf(StringIO(table_text))
    if df.empty:
        raise ValueError(f"Failed to parse oracle rows from {path}")

    numeric_cols = [
        "oracle_tau",
        "baseline_rel_err",
        "baseline_rank",
        "chosen_eps",
        "chosen_rank",
        "rank_drop_vs_baseline",
        "chosen_rel_err",
        "quantized_ppl",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_layer_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["layer_type"] = out["layer_name"].str.split(".").str[-1]
    return out


def print_group_summary(df: pd.DataFrame, group_cols):
    features = [
        "tail32_energy_frac",
        "tail64_energy_frac",
        "spectral_entropy_norm",
        "effective_rank",
        "top1_over_top2",
        "top1_over_mean32",
        "rank_99_energy",
        "trunc_rank_frac",
    ]
    summary = (
        df.groupby(group_cols)
        .agg(
            layers=("layer_name", "count"),
            **{feat: (feat, "mean") for feat in features},
        )
        .reset_index()
        .sort_values("layers", ascending=False)
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main():
    args = parse_args()
    spectral_csv = Path(args.spectral_csv)
    oracle_summary = Path(args.oracle_summary)
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else spectral_csv.with_name("spectral_oracle_join.csv")
    )

    spectral = pd.read_csv(spectral_csv)
    oracle = parse_oracle_table(oracle_summary)

    spectral = add_layer_type(spectral)
    oracle = add_layer_type(oracle)

    # One row per layer/config in spectral_csv; oracle labels are per-layer only.
    # Join on layer_name and keep only oracle-labeled layers.
    joined = spectral.merge(
        oracle[
            [
                "layer_name",
                "layer_type",
                "chosen_method",
                "chosen_eps",
                "chosen_rank",
                "rank_drop_vs_baseline",
                "chosen_rel_err",
                "baseline_rel_err",
                "oracle_tau",
            ]
        ],
        on=["layer_name", "layer_type"],
        how="inner",
    )
    joined.to_csv(out_csv, index=False)

    print(f"Wrote joined table to {out_csv}")
    print(f"Rows: {len(joined)}")

    print("\n=== ORACLE LABEL COUNTS ===")
    label_counts = (
        oracle.groupby(["chosen_method", "chosen_eps"])
        .size()
        .reset_index(name="layers")
        .sort_values("layers", ascending=False)
    )
    print(label_counts.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== FEATURE MEANS BY ORACLE LABEL ===")
    print_group_summary(oracle.merge(spectral.drop_duplicates("layer_name"), on=["layer_name", "layer_type"], how="left"), ["chosen_method", "chosen_eps"])

    print("\n=== FEATURE MEANS BY ORACLE LABEL AND LAYER TYPE ===")
    print_group_summary(
        oracle.merge(spectral.drop_duplicates("layer_name"), on=["layer_name", "layer_type"], how="left"),
        ["chosen_method", "chosen_eps", "layer_type"],
    )


if __name__ == "__main__":
    main()
