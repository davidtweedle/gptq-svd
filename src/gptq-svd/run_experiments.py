import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- Configuration ---
PYTHON_INTERPRETER = "python"
SCRIPT_PATH = "quantize.py"
MODEL_ID = "Qwen/Qwen3-8B"
DATASET = "wikitext2"
DEVICE = "cuda:0"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_SAVE_DIR = Path(f"experiments_{TIMESTAMP}")
experiments = []
for r in [1.0, 2.0]:
    for m in ["energy", "mean_trimmed"]:
        experiments.append({"eps": 0.1, "sketch_ratio": r, "threshold_method": m})

for e in [0.01, 0.0001]:
    for r in [1.0, 2.0]:
        experiments.append({"eps": e, "sketch_ratio": r, "threshold_method": "energy"})

for e in [0.01, 0.0001]:
    for m in ["energy", "mean_trimmed"]:
        experiments.append({"eps": e, "sketch_ratio": 4.0, "threshold_method": m})


def run_command(cmd: str) -> int:
    print(f"CMD: {cmd}")
    process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
            )
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    if rc != 0:
        raise RuntimeError(f"Command failed with return code {rc}")
    return rc


def main():
    results_summary = []
    print("--- Starting Batch Experiment Runner ---")
    print(f"Model: {MODEL_ID}")
    print(f"Output Directory: {BASE_SAVE_DIR}")
    print(f"Total Experiments: {len(experiments)}\n")

    BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for i, params in enumerate(experiments):
        print(f"\n\n=== RUNNING EXPERIMENT {i + 1}/{len(experiments)} ===")
        print(f"Params: {params}")

        run_name = f"eps{params['eps']}_ratio{params['sketch_ratio']}_{params['threshold_method']}"
        save_path = BASE_SAVE_DIR / run_name
        save_path.mkdir(exist_ok=True)
        cmd = (
                f"{PYTHON_INTERPRETER} {SCRIPT_PATH} "
                f"--model_id {MODEL_ID} "
                f"--dataset {DATASET} "
                f"--mode svd "
                f"--eps {params['eps']} "
                f"--sketch_ratio {params['sketch_ratio']} "
                f"--threshold_method {params['threshold_method']} "
                f"--save_path {save_path} "
                f"--device  {DEVICE} "
                f"--no_save "
                )

        try:
            run_command(cmd)
            status = "Success"
        except Exception as e:
            print(f"Experiment failed: {e}")
            status = "Failed"

        result_file = save_path / "results.json"
        summary = {
                'eps': params['eps'],
                'sketch_ratio': params['sketch_ratio'],
                'threshold_method': params['threshold_method'],
                'status': status,
                'quantized_ppl': None,
                'avg_rank_kept_pct': None,
                'total_time_s': None
                }
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)

                metrics = data.get("metrics", {})
                layer_stats = data.get("layer_stats", [])

                summary.update({
                    'quantized_ppl': metrics.get("quantized_ppl"),
                    'total_time_s': round(metrics.get("total_time", 0), 2)
                    })
                print(f"--> Result: PPL {summary['quantized_ppl']}")
            except json.JSONDecodeError:
                print("Error: Could not decode results.json")

        results_summary.append(summary)
        df = pd.DataFrame(results_summary)
        summary_path = BASE_SAVE_DIR / "summary_partial.csv"
        df.to_csv(summary_path, index=False)

    print("\n\n=== FINAL EXPERIMENT SUMMARY ===")
    df = pd.DataFrame(results_summary)
    df = df.sort_values(by=["threshold_method", "sketch_ratio", "eps"])
    final_csv_path = BASE_SAVE_DIR / "final_summary.csv"
    df.to_csv(final_csv_path, index=False)
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

