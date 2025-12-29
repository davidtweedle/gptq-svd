import subprocess
import os
import json
import itertools
import pandas as pd
from datetime import datetime
PYTHON_INTERPRETER = "python"
SCRIPT_PATH = "quantize.py"
MODEL_ID = "Qwen/Qwen3-8B"
DATASET = "wikitext2"
BASE_SAVE_DIR = "experiments_dec28"
param_grid = {
        "eps": [1e-2, 1e-4, 1e-6],
        "sketch_ratio": [0.5, 1.0, 8.0]
        }

def run_command(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def main():
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results_summary = []
    print(f"Found {len(experiments)} experiments to run.")
    for i, params in enumerate(experiments):
        print(f"\n\n=== RUNNING EXPERIMENT {i + 1}/{len(experiments)} ===")
        print(f"Params: {params}")

        run_name = f"svd_eps{params['eps']}_ratio{params['sketch_ratio']}"
        save_path = os.path.join(BASE_SAVE_DIR, run_name)
        os.makedirs(save_path, exist_ok=True)
        cmd = (
                f"{PYTHON_INTERPRETER} {SCRIPT_PATH} "
                f"--model_id {MODEL_ID} "
                f"--dataset {DATASET} "
                f"--mode svd "
                f"--eps {params['eps']} "
                f"--sketch_ratio {params['sketch_ratio']} "
                f"--save_path {save_path} "
                f"--device cuda:0 "
                )

        try:
            run_command(cmd)
        except Exception as e:
            print(f"Experiment failed: {e}")

        result_file = os.path.join(save_path, "results.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                data = json.load(f)

            metrics = data.get("metrics", {})

            layer_stats = data.get("layer_stats", [])
            avg_rank_pct = 0
            if layer_stats:
                avg_rank_pct = sum(x['rank_fraction'] for x in layer_stats) / len(layer_stats)
            summary = {
                    'eps': params['eps'],
                    'sketch_ratio': params['sketch_ratio'],
                    'baseline_ppl': metrics.get("baseline_ppl", -1),
                    'quantized_ppl': metrics.get("quantized_ppl", -1),
                    "avg_rank_kept_%": round(avg_rank_pct * 100, 2),
                    "total_time_s": round(metrics.get("total_time", 0), 2)
                    }
            results_summary.append(summary)
            print(f"Result: PPL {summary['quantized_ppl']} (Avg Rank {summary['avg_rank_kept_%']}%)")
        else:
            print("Error: No results.json found.")
    df = pd.DataFrame(results_summary)
    summary_path = os.path.join(BASE_SAVE_DIR, "final_summary.csv")
    df.to_csv(summary_path, index=False)

    print(f"\n\n=== EXPERIMENT SUMMARY ===")
    print(df.to_markdown(index=False))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

