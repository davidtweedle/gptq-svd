import argparse
import json
import os

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot saved layer diagnostics from collect_layer_stats.py"
    )
    parser.add_argument(
        "--stats_dir",
        type=str,
        required=True,
        help="Directory containing manifest.json and saved .pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write plots (defaults to stats_dir/plots)",
    )
    return parser.parse_args()


def sort_desc(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sort(tensor.detach().cpu().to(torch.float64), descending=True).values


def load_records(stats_dir: str):
    manifest_path = os.path.join(stats_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    records = []
    for item in sorted(manifest, key=lambda x: x["layer_index"]):
        payload = torch.load(item["output_file"], map_location="cpu")
        records.append(payload)
    return records


def layer_label(record):
    return f"Layer {record['layer_index']} {record['submodule_name']}"


def plot_eigvals(records, output_dir):
    fig, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4), squeeze=False)
    axes = axes[0]

    for ax, record in zip(axes, records):
        eigvals = sort_desc(record["H_eigvals"]).clamp_min(1e-20)
        rank = int(record["trunc_rank"])

        ax.plot(eigvals.numpy(), linewidth=2)
        if 0 < rank < len(eigvals):
            ax.axvline(rank - 1, color="red", linestyle="--", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_title(layer_label(record))
        ax.set_xlabel("Sorted Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hessian Eigenvalue Spectra")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h_eigvals.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diag_vs_gptq(records, output_dir):
    fig, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4), squeeze=False)
    axes = axes[0]

    for ax, record in zip(axes, records):
        h_diag = sort_desc(record["H_diag"]).clamp_min(1e-20)
        gptq_diag = sort_desc(record["gptq_raw_diag"]).clamp_min(1e-20)

        ax.plot(h_diag.numpy(), label="diag(H)", linewidth=2)
        ax.plot(gptq_diag.numpy(), label="diag(GPTQ factor)", linewidth=2)
        ax.set_yscale("log")
        ax.set_title(layer_label(record))
        ax.set_xlabel("Sorted Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("diag(H) vs GPTQ Factor Diagonal")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "diag_h_vs_gptq.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_update_diag_by_order(records, output_dir):
    fig, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4), squeeze=False)
    axes = axes[0]

    for ax, record in zip(axes, records):
        h_diag = record["H_diag"].detach().cpu().to(torch.float64)
        gptq_diag = record["gptq_raw_diag"].detach().cpu().to(torch.float64).abs()
        trunc_diag = record["trunc_raw_diag"].detach().cpu().to(torch.float64).abs()

        # Standard GPTQ is commonly discussed in the ordering induced by diag(H).
        gptq_order = torch.argsort(h_diag, descending=True)
        gptq_ordered = gptq_diag[gptq_order]

        # TruncGPTQ's update diagonal is already produced in pivot order.
        ax.plot(
            torch.arange(len(gptq_ordered)).numpy(),
            gptq_ordered.numpy(),
            label="GPTQ diag(H) order",
            linewidth=2,
        )
        ax.plot(
            torch.arange(len(trunc_diag)).numpy(),
            trunc_diag.numpy(),
            label="TruncGPTQ pivot order",
            linewidth=2,
        )
        ax.set_yscale("log")
        ax.set_title(layer_label(record))
        ax.set_xlabel("Update Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Update Diagonal by Induced Ordering")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "update_diag_by_order.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.stats_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    records = load_records(args.stats_dir)
    if not records:
        raise ValueError(f"No records found in {args.stats_dir}")

    plot_eigvals(records, output_dir)
    plot_update_diag_by_order(records, output_dir)


if __name__ == "__main__":
    main()
