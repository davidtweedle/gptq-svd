import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract compact spectral features from collect_layer_stats .pt files."
    )
    parser.add_argument(
        "--stats_dir",
        type=str,
        required=True,
        help="Directory produced by collect_layer_stats.py",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults to <stats_dir>/spectral_features.csv",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def compute_features(payload: dict) -> dict:
    eigvals = payload["H_eigvals"].to(torch.float64)
    eigvals = torch.sort(eigvals, descending=True).values
    eigvals = eigvals.clamp(min=0)

    total = float(eigvals.sum().item())
    n = int(eigvals.numel())
    positive = eigvals[eigvals > 0]
    pos_n = int(positive.numel())

    probs = positive / positive.sum() if pos_n > 0 and positive.sum() > 0 else positive
    spectral_entropy = (
        float((-(probs * torch.log(probs))).sum().item()) if pos_n > 0 else 0.0
    )
    spectral_entropy_norm = safe_div(spectral_entropy, float(torch.log(torch.tensor(max(pos_n, 2), dtype=torch.float64)).item()))

    energy_cumsum = torch.cumsum(eigvals, dim=0)

    def rank_for_energy(frac: float) -> int:
        if total <= 0:
            return 0
        target = frac * total
        return int(torch.searchsorted(energy_cumsum, torch.tensor(target, dtype=eigvals.dtype)).item()) + 1

    top1 = float(eigvals[0].item()) if n > 0 else 0.0
    top2 = float(eigvals[1].item()) if n > 1 else 0.0
    top4_mean = float(eigvals[: min(4, n)].mean().item()) if n > 0 else 0.0
    top8_mean = float(eigvals[: min(8, n)].mean().item()) if n > 0 else 0.0
    top32_mean = float(eigvals[: min(32, n)].mean().item()) if n > 0 else 0.0
    top64_mean = float(eigvals[: min(64, n)].mean().item()) if n > 0 else 0.0

    tail_from_32 = float(eigvals[32:].sum().item()) if n > 32 else 0.0
    tail_from_64 = float(eigvals[64:].sum().item()) if n > 64 else 0.0

    return {
        "layer_index": int(payload["layer_index"]),
        "submodule_name": payload["submodule_name"],
        "layer_name": f"layer_{int(payload['layer_index'])}.{payload['submodule_name']}",
        "full_dim": int(payload["full_dim"]),
        "trunc_rank": int(payload["trunc_rank"]),
        "positive_rank": pos_n,
        "rank_90_energy": rank_for_energy(0.90),
        "rank_95_energy": rank_for_energy(0.95),
        "rank_99_energy": rank_for_energy(0.99),
        "rank_999_energy": rank_for_energy(0.999),
        "top1_eig": top1,
        "top2_eig": top2,
        "top1_over_top2": safe_div(top1, top2),
        "top1_over_mean32": safe_div(top1, top32_mean),
        "top4_mean": top4_mean,
        "top8_mean": top8_mean,
        "top32_mean": top32_mean,
        "top64_mean": top64_mean,
        "tail32_energy_frac": safe_div(tail_from_32, total),
        "tail64_energy_frac": safe_div(tail_from_64, total),
        "spectral_entropy": spectral_entropy,
        "spectral_entropy_norm": spectral_entropy_norm,
        "effective_rank": float(torch.exp(torch.tensor(spectral_entropy)).item()) if pos_n > 0 else 0.0,
        "trunc_rank_frac": safe_div(int(payload["trunc_rank"]), int(payload["full_dim"])),
    }


def main():
    args = parse_args()
    stats_dir = Path(args.stats_dir)
    out_csv = Path(args.out_csv) if args.out_csv else stats_dir / "spectral_features.csv"

    rows = []
    for pt_file in sorted(stats_dir.glob("layer_*_*.pt")):
        payload = torch.load(pt_file, map_location="cpu")
        rows.append(compute_features(payload))

    if not rows:
        raise ValueError(f"No layer stats .pt files found under {stats_dir}")

    df = pd.DataFrame(rows).sort_values(["layer_index", "submodule_name"])
    df.to_csv(out_csv, index=False)

    manifest_path = stats_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        print(f"Loaded manifest entries: {len(manifest)}")
    print(f"Wrote {len(df)} rows to {out_csv}")
    print(df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))


if __name__ == "__main__":
    main()
