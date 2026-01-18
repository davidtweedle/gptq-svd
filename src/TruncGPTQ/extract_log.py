import re
import pandas as pd
import os

# Configuration
# Set this to the top-level directory containing your experiment subfolders
ROOT_DIR = "./experiments"
LOG_FILENAME = "quantization.log"

def parse_single_log(filepath):
    """
    Parses a single log file and returns a DataFrame of its contents.
    """
    data = []
    # State variables
    current_run_id = "Unknown_Run"
    current_error = None
    # Regex Patterns
    run_pattern = re.compile(r"INFO:\s+Params:\s+(.+)")
    error_pattern = re.compile(r"Relative prediction error:\s+([\d\.]+)")
    module_pattern = re.compile(r"INFO:\s+([\w\.]+)\s+\|\s+Rank:")

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                # 1. Detect Run Configuration
                run_match = run_pattern.search(line)
                if run_match:
                    current_run_id = run_match.group(1)
                    continue

                # 2. Detect Error Metric
                err_match = error_pattern.search(line)
                if err_match:
                    current_error = float(err_match.group(1))
                    continue

                # 3. Detect Module Name
                mod_match = module_pattern.search(line)
                if mod_match and current_error is not None:
                    full_module_name = mod_match.group(1)
                    if '.' in full_module_name:
                        layer_type = full_module_name.split('.')[-1]
                    else:
                        layer_type = full_module_name

                    # Add to list, including the source directory
                    data.append({
                        'Source_Dir': os.path.dirname(filepath),
                        'Run_Config': current_run_id,
                        'Layer_Type': layer_type,
                        'Relative_Error': current_error
                    })
                    current_error = None
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

    return pd.DataFrame(data) if data else None


def main():
    print(f"Scanning for '{LOG_FILENAME}' in: {ROOT_DIR} ...")
    all_dfs = []
    # 1. Recursive Walk
    for root, dirs, files in os.walk(ROOT_DIR):
        if LOG_FILENAME in files:
            full_path = os.path.join(root, LOG_FILENAME)
            # print(f"  Found: {full_path}")
            df = parse_single_log(full_path)
            if df is not None and not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        print("No log files found or all were empty.")
        return

    # 2. Combine Data
    master_df = pd.concat(all_dfs, ignore_index=True)
    # 3. Aggregate Stats
    # Group by Directory AND Run Config (just in case configs are identical across folders)
    stats = master_df.groupby(['Source_Dir', 'Run_Config', 'Layer_Type'])['Relative_Error'].agg(
        Mean_Error='mean',
        Max_Error='max',
        Count='count'
    ).reset_index()
    # Sort: High errors first
    stats = stats.sort_values(by=['Source_Dir', 'Run_Config', 'Mean_Error'], ascending=[True, True, False])

    # 4. Print Summary
    print("\n" + "="*100)
    print("  MULTI-FILE QUANTIZATION SUMMARY")
    print("="*100)
    # Iterate through unique experiments (Dir + Run Config)
    unique_exps = stats[['Source_Dir', 'Run_Config']].drop_duplicates()
    for _, exp in unique_exps.iterrows():
        src = exp['Source_Dir']
        cfg = exp['Run_Config']
        print(f"\nDIR:  {src}")
        print(f"CONF: {cfg}")
        print("-" * 80)
        print(f"{'Layer Type':<15} | {'Mean Error':<12} | {'Max Error':<12} | {'Count':<5}")
        print("-" * 80)
        # Filter for this specific experiment
        exp_stats = stats[(stats['Source_Dir'] == src) & (stats['Run_Config'] == cfg)]
        for _, row in exp_stats.iterrows():
            print(f"{row['Layer_Type']:<15} | {row['Mean_Error']:.6f}     | {row['Max_Error']:.6f}     | {row['Count']:<5}")

    out_csv = "multi_run_summary.csv"
    stats.to_csv(out_csv, index=False)
    print(f"\n[+] Full aggregated results saved to {out_csv}")


if __name__ == "__main__":
    main()
