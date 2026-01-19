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
BASE_SAVE_DIR = Path(f"benchmark_results_{TIMESTAMP}")
experiments = []

#experiments.append({
#    "name": "FP16_Baseline",
#    "mode": "baseline",
#    "w_bits": 16,
#    "group": 0,
#    "sym": False,
#    "algo": "FP16",
#    "eps": 0.0
#    })
#
#for bits in [4, 3, 2]:
#    experiments.append({
#        "name": f"GPTQ_W{bits}_Asym",
#        "mode": "gptq",
#        "w_bits": bits,
#        "group": 128,
#        "sym": False,
#        "algo": "GPTQ",
#        "eps": 0.0,
#        "batch_size": 32
#        })
#
#for bits in [4, 3]:
#    experiments.append({
#        "name": f"GPTQ_W{bits}_Sym",
#        "mode": "gptq",
#        "w_bits": bits,
#        "sym": True,
#        "algo": "GPTQ",
#        "group": 128,
#        "batch_size": 32
#        })

eps_list = [1e-6, 1e-4, 1e-5]
for bits, eps in zip([4, 3, 2], eps_list):
    experiments.append({
        "name": f"Trunc_W{bits}_Asym",
        "mode": "eigh",
        "w_bits": bits,
        "sym": False,
        "algo": "TruncGPTQ",
        "group": 128,
        "adaptive_eps": False,
        "eps": eps,
        "batch_size": 32
        })

eps_list = [1e-4, 1e-4]
for bits, eps in zip([4, 3], eps_list):
    experiments.append({
        "name": f"Trunc_W{bits}_Sym",
        "mode": "eigh",
        "w_bits": bits,
        "sym": True,
        "algo": "TruncGPTQ",
        "group": 128,
        "adaptive_eps": False,
        "eps": eps,
        "batch_size": 32
        })


def run_command(cmd_list):
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    with subprocess.Popen(
            cmd_list,
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
    print("--- Starting Bechmark Run ---")
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
                "--model_id", MODEL_ID,
                "--dataset", DATASET,
                "--save_path", str(save_path),
                "--device", DEVICE,
                "--threshold_method", "energy",
                "--sketch_ratio", "1.0",
                "--no_save"
                ]
        if exp["mode"] == "baseline":
            cmd.extend(["--mode", "baseline"])
        else:
            cmd.extend(["--mode", exp["mode"]])
            cmd.extend(["--w_bits", str(exp['w_bits'])])
            cmd.extend(["--group_size", str(exp['group'])])
            cmd.extend(["--batch_size", str(exp['batch_size'])])

            if exp["mode"] == "eigh":
                cmd.extend(["--eps", str(exp['eps'])])
                if exp.get("adaptive_eps", False):
                    cmd.append("--adaptive_eps")
            if exp.get("sym", False):
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

                    q_ppl = metrics.get("quantized_ppl")
                    b_ppl = metrics.get("baseline_ppl")
                    final_ppl = q_ppl if q_ppl is not None else b_ppl
                    if isinstance(final_ppl, (int, float)):
                        row["ppl"] = round(final_ppl, 4)
                    print(f"--> Captured PPL: {row['ppl']}")
            except Exception as e:
                row["status"] = f"JSON Error: {e}"
        results.append(row)

        pd.DataFrame(results).to_csv(BASE_SAVE_DIR / "results_partial.csv", index=False)
    print("\n\n=== BENCHMARK COMPLETED ===")
    df = pd.DataFrame(results)

    display_cols = ["algo", "w_bits", "group", "sym", "eps", "ppl", "time_s", "status"]
    existing_display_cols = [c for c in display_cols if c in df.columns]
    final_df = df[existing_display_cols]
    print(final_df.to_string(index=False, justify='center'))

    final_path = BASE_SAVE_DIR / "final_benchmark.csv"
    final_df.to_csv(final_path, index=False)
    print(f"\nSaved to: {final_path}")


if __name__ == "__main__":
    main()
