import subprocess
import json
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- Configuration ---
PYTHON_INTERPRETER = "python"
SCRIPT_PATH = "quantize.py"
MODEL_ID = "Qwen/Qwen2.5-7B"  # Updated to the model we are using
DATASET = "wikitext2"
DEVICE = "cuda:0"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_SAVE_DIR = Path(f"experiments_ref_{TIMESTAMP}")

# Define the two critical runs
experiments = [
    {"mode": "baseline", "name": "FP16_Baseline"},
    {"mode": "gptq",     "name": "GPTQ_Reference_4bit"}
]

def run_command(cmd: str) -> str:
    """Runs command and returns the full stdout output."""
    print(f"CMD: {cmd}")
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    full_output = []
    while True:
        line = process.stdout.readline()
        if line == '' and process.poll() is not None:
            break
        if line:
            print(line.strip())
            full_output.append(line)

    rc = process.poll()
    if rc != 0:
        raise RuntimeError(f"Command failed with return code {rc}")
    return "".join(full_output)

def main():
    results_summary = []
    print("--- Starting Reference Benchmark Runner ---")
    print(f"Model: {MODEL_ID}")
    print(f"Output Directory: {BASE_SAVE_DIR}")

    BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for i, params in enumerate(experiments):
        print(f"\n\n=== RUNNING {params['name']} ({i + 1}/{len(experiments)}) ===")

        save_path = BASE_SAVE_DIR / params["name"]
        save_path.mkdir(exist_ok=True)

        # Construct command
        # Note: We pass dummy values for eps/ratio because argparse likely requires them
        # (or defaults exist), but 'mode' determines execution path.
        cmd = (
            f"{PYTHON_INTERPRETER} {SCRIPT_PATH} "
            f"--model_id {MODEL_ID} "
            f"--dataset {DATASET} "
            f"--mode {params['mode']} "
            f"--save_path {save_path} "
            f"--device {DEVICE} "
            f"--no_save "
        )

        status = "Failed"
        ppl = None
        time_s = None

        try:
            # Run and capture output
            output_log = run_command(cmd)
            status = "Success"

            # --- Result Parsing ---

            if params['mode'] == 'baseline':
                # Baseline mode exits early in your quantize.py, so no results.json.
                match = re.search(r"Baseline PPL: (\d+\.\d+)", output_log)
                if match:
                    ppl = float(match.group(1))

            elif params['mode'] == 'gptq':
                # GPTQ mode writes results.json
                result_file = save_path / "results.json"
                if result_file.exists():
                    with open(result_file, "r") as f:
                        data = json.load(f)
                        metrics = data.get("metrics", {})
                        ppl = metrics.get("quantized_ppl")
                        time_s = metrics.get("total_time")

        except Exception as e:
            print(f"Experiment failed: {e}")
            status = "Failed"

        # Record Summary
        summary = {
            'experiment': params['name'],
            'mode': params['mode'],
            'status': status,
            'ppl': ppl,
            'time_s': round(time_s, 2) if time_s else None
        }
        results_summary.append(summary)
        print(f"--> Result: {params['name']} | PPL: {ppl}")

    # --- Final Report ---
    print("\n\n=== FINAL BENCHMARK SUMMARY ===")
    df = pd.DataFrame(results_summary)
    final_csv_path = BASE_SAVE_DIR / "benchmark_summary.csv"
    df.to_csv(final_csv_path, index=False)

    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df)

    print(f"\nSaved to {final_csv_path}")

if __name__ == "__main__":
    main()
