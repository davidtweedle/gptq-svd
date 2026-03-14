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
SEEDS = [40, 41, 42, 43, 44]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_SAVE_DIR = Path(f"tuning_results_{TIMESTAMP}")
experiments = []

# Best tuned GPTQ baseline so far for symmetric 4-bit:
# damp_percent=0.1, kernel_impl=cuda_immediate, block_size=512, large_update_impl=addmm
#
# Section A: Phase-1 truncation ablation (single-seed screening).
# Fix the best known execution setting and compare the four truncation families.
for threshold_method, eps_values in [
    ("abs", [1e-6, 1e-5, 1e-4]),
    ("percent", [1, 2, 5, 10]),
    ("mean_trimmed", [1e-3, 1e-2, 1e-1]),
    ("energy", [1e-6, 1e-5, 1e-4, 1e-3]),
]:
    for eps in eps_values:
        experiments.append({
            "name": (
                f"Trunc_W4_Sym_{threshold_method}_{eps}_"
                f"cuda_immediate_fp32_addmm_block512_seed40"
            ),
            "section": "trunc_threshold_phase1",
            "mode": "eigh",
            "w_bits": 4,
            "group": 128,
            "sym": True,
            "algo": "TruncGPTQ",
            "adaptive_eps": False,
            "eps": eps,
            "threshold_method": threshold_method,
            "batch_size": 32,
            "beta": 1.0,
            "rotate_weights": False,
            "kernel_impl": "cuda_immediate",
            "accum_dtype": "fp32",
            "large_update_impl": "addmm",
            "normalize_hinv_diag": True,
            "block_size": 512,
            "seed": 40,
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
                "--threshold_method", exp.get("threshold_method", "energy"),
                "--sketch_ratio", "1.0",
                "--no_save",
                "--beta", str(exp['beta']),
                "--kernel_impl", exp.get("kernel_impl", "triton"),
                "--accum_dtype", exp.get("accum_dtype", "fp32"),
                "--large_update_impl", exp.get("large_update_impl", "matmul"),
                "--block_size", str(exp.get("block_size", 1024)),
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
            if exp["mode"] == "gptq":
                cmd.extend(["--damp_percent", str(exp.get("damp_percent", 0.01))])
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
