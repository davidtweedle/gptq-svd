import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# --- Configuration ---
PYTHON_INTERPRETER = "python"
SCRIPT_PATH = "quantize.py"
MODEL_ID = "Qwen/Qwen3-8B"
DATASET = "wikitext2"
DEVICE = "cuda:0"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_SAVE_DIR = Path(f"benchmark_results_{TIMESTAMP}")
experiments = []

experiments.append({
    "name": "FP16_Baseline",
    "mode": "baseline",
    "w_bits": 16,
    "group": 0,
    "sym": False,
    "algo": "FP16",
    "eps": 0.0
    })

for bits in [4, 3, 2]:
    for sym in [False, True]:
        for group in [-1, 128]:
            sym_label = "Sym" if sym else "Asym"
            experiments.append({
                "name": f"GPTQ_W{bits}_{sym_label}",
                "mode": "gptq",
                "w_bits": bits,
                "group": group,
                "sym": sym,
                "algo": "GPTQ",
                "adaptive_eps": False,
                "eps": 0.0
                })

for bits in [4, 3, 2]:
    if bits == 3:
        base_eps = 0.0001
    else:
        base_eps = 0.00001
    for sym in [False, True]:
        for group in [-1, 128]:
            sym_label = "Sym" if sym else "Asym"
            experiments.append({
                "name": f"SVD_W{bits}_{sym_label}_adaptive",
                "mode": "eigh",
                "w_bits": bits,
                "group": group,
                "sym": sym,
                "algo": "SVD-quant",
                "adaptive_eps": True,
                "eps": base_eps
                })

def run_command(cmd_list):
    cmd_str = " ".join(cmd_list)
    print(f"\n[EXEC] {cmd_str}")
    with subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
            ) as process:
        for line in process.stdout:
            print(line, end='')
    if process.returncode != 0:
        print(f"!!! FAILED with code {process.returncode} !!!")
        return False
    return True


def main():
    print("--- Starting Batch Experiment Runner ---")
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET}")
    print(f"Output Directory: {BASE_SAVE_DIR}")
    print(f"Total Experiments: {len(experiments)}\n")

    BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, exp in enumerate(experiments):
        print(f"\n\n=== RUNNING EXPERIMENT {i + 1}/{len(experiments)}: {exp['name']} ===")
        save_path = BASE_SAVE_DIR / exp['name']
        save_path.mkdir(exist_ok=True)
        cmd = [
                PYTHON_INTERPRETER, SCRIPT_PATH,
                f"--model_id {MODEL_ID}",
                f"--dataset {DATASET}",
                f"--save_path {save_path}",
                f"--device {DEVICE}",
                "--threshold_method energy",
                "--sketch_ratio 1.0",
                "--no_save"
                ]
        if exp["mode"] == "baseline":
            cmd.append("--mode baseline")
        else:
            cmd.append(f"--mode {exp['mode']}")
            cmd.append(f"--w_bits {exp['w_bits']}")
            cmd.append(f"--group_size {exp['group']}")

            if exp["mode"] == "eigh":
                cmd.append(f"--eps {exp['eps']}")
                if exp["adaptive_eps"]:
                    cmd.append("--adaptive_eps")
            if exp["sym"]:
                cmd.append("--sym")
        start_t = datetime.now()
        success = run_command(cmd)
        duration = (datetime.now() - start_t).total_seconds()

        row = exp.copy()
        row["status"] = "Success" if success else "Failed"
        row["time_s"] = round(duration, 1)
        row["ppl"] = "N/A"

        result_file = save_path / "results.json"
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)
                    metrics = data.get("metrics", {})
                    row["ppl"] = metrics.get("quantized_ppl", "N/A")
                    print(f"--> Captured PPL: {row['ppl']}")
            except Exception as e:
                row["status"] = f"JSON Error: {e}"
        results.append(row)

        pd.DataFrame(results).to_csv(BASE_SAVE_DIR / "results_partial.csv", index=False)
    print("\n\n=== BENCHMARK COMPLETED ===")
    df = pd.DataFrame(results)

    display_cols = ["algo", "w_bits", "sym", "ppl", "time_s", "status"]
    if set(display_cols).issubset(df.columns):
        final_df = df[display_cols]
    else:
        final_df = df

    print(final_df.to_string(index=False))

    final_path = BASE_SAVE_DIR / "final_benchmark.csv"
    final_df.to_csv(final_path, index=False)
    print(f"\nSaved to: {final_path}")


if __name__ == "__main__":
    main()
