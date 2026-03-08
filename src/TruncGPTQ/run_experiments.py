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
EVAL_MODE = "regular"
EVAL_BATCH_SIZE = 4
SEEDS = [42, 43]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_SAVE_DIR = Path(f"tuning_results_{TIMESTAMP}")
experiments = []

# Six selected settings (fp32), plus fp64 variants for CUDA settings only.
SELECTED_SETTINGS = [
    {"normalize_hinv_diag": True, "kernel_impl": "cuda_immediate", "large_update_impl": "matmul"},
    {"normalize_hinv_diag": True, "kernel_impl": "triton", "large_update_impl": "addmm"},
    {"normalize_hinv_diag": False, "kernel_impl": "cuda_lazy_reduce", "large_update_impl": "addmm"},
    {"normalize_hinv_diag": True, "kernel_impl": "cuda_lazy_reduce", "large_update_impl": "matmul"},
    {"normalize_hinv_diag": False, "kernel_impl": "cuda_immediate", "large_update_impl": "matmul"},
    {"normalize_hinv_diag": True, "kernel_impl": "cuda_immediate", "large_update_impl": "addmm"},
]

for cfg in SELECTED_SETTINGS:
    accum_dtypes = ["fp32", "fp64"] if cfg["kernel_impl"].startswith("cuda_") else ["fp32"]
    for accum_dtype in accum_dtypes:
        for seed in SEEDS:
            experiments.append({
                "name": (
                    "Trunc_W4_Asym_1e-05_"
                    f"{'norm' if cfg['normalize_hinv_diag'] else 'no_norm'}_"
                    f"{cfg['kernel_impl']}_{accum_dtype}_{cfg['large_update_impl']}_seed{seed}"
                ),
                "mode": "eigh",
                "w_bits": 4,
                "group": 128,
                "sym": False,
                "algo": "TruncGPTQ",
                "adaptive_eps": False,
                "eps": 1e-5,
                "batch_size": 32,
                "beta": 1.0,
                "rotate_weights": False,
                "kernel_impl": cfg["kernel_impl"],
                "accum_dtype": accum_dtype,
                "large_update_impl": cfg["large_update_impl"],
                "normalize_hinv_diag": cfg["normalize_hinv_diag"],
                "seed": seed,
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
                "--model_id", MODEL_ID,
                "--dataset", DATASET,
                "--save_path", str(save_path),
                "--device", DEVICE,
                "--seed", str(exp.get("seed", 42)),
                "--eval_mode", EVAL_MODE,
                "--batch_size", str(exp['batch_size']),
                "--threshold_method", "energy",
                "--sketch_ratio", "1.0",
                "--no_save",
                "--beta", str(exp['beta']),
                "--kernel_impl", exp.get("kernel_impl", "triton"),
                "--accum_dtype", exp.get("accum_dtype", "fp32"),
                "--large_update_impl", exp.get("large_update_impl", "matmul"),
                ]
        if EVAL_BATCH_SIZE is not None:
            cmd.extend(["--eval_batch_size", str(EVAL_BATCH_SIZE)])
        if exp["mode"] == "baseline":
            cmd.extend(["--mode", "baseline"])
        else:
            cmd.extend(["--mode", exp['mode']])
            cmd.extend(["--w_bits", str(exp['w_bits'])])
            cmd.extend(["--group_size", str(exp['group'])])

            if exp["mode"] == "eigh":
                cmd.extend(["--eps", str(exp['eps'])])
                if exp.get("adaptive_eps", False):
                    cmd.append("--adaptive_eps")
            if exp.get("sym", False):
                cmd.append("--sym")
            if exp.get("rotate_weights", False):
                cmd.append("--rotate_weights")
            if not exp.get("normalize_hinv_diag", True):
                cmd.append("--no_normalize_hinv_diag")
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
                    ppl_val = metrics.get("quantized_ppl") or metrics.get("baseline_ppl")
                    row["ppl"] = round(ppl_val, 4) if ppl_val else "N/A"
                    print(f"--> Captured PPL: {row['ppl']}")
            except:
                pass
        results.append(row)

        pd.DataFrame(results).to_csv(BASE_SAVE_DIR / "results_partial.csv", index=False)
    print("\n\n=== EXPERIMENTS COMPLETED ===")
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(" TUNING SUMMARY")
    print("="*50)


    display_cols = ["w_bits", "sym", "eps", "ppl", "time_s", "status"]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))

    final_path = BASE_SAVE_DIR / "final_results.csv"
    df.to_csv(final_path, index=False)
    print(f"\nSaved to: {final_path}")


if __name__ == "__main__":
    main()
