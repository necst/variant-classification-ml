#!/bin/bash

# Array of YAML files to process
YAML_FILES=(
    "multirun/xgboost_dbnfs_2c_bal_full_218142_all_features/optimization_results.yaml"
    "multirun/xgboost_dbnfs_2c_bal_100k_100000_all_features/optimization_results.yaml"
    "multirun/xgboost_dbnfs_2c_bal_10k_10000_all_features/optimization_results.yaml"
    "multirun/xgboost_dbnfs_2c_bal_full_218142_drop_raw_metadata_keep_clinical_context/optimization_results.yaml"
    "multirun/xgboost_dbnfs_2c_bal_full_218142_drop_raw_metadata_and_labs/optimization_results.yaml"
)

# Output log file
LOG_FILE="experiments/benchmark_results.log"
echo "--- Benchmark Started at $(date) ---" | tee "$LOG_FILE"

# Loop through each file
for YAML in "${YAML_FILES[@]}"; do
    echo "----------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "RUNNING: $YAML" | tee -a "$LOG_FILE"
    echo "----------------------------------------------------------" | tee -a "$LOG_FILE"

    # We use /usr/bin/time with a custom format:
    # %e = Elapsed time (seconds)
    # %M = Maximum Resident Set Size (Peak RAM in KB)
    # %P = CPU percentage
    /usr/bin/time -f "\nBENCHMARK DATA:\nTime: %e sec\nMax RAM: %M KB\nCPU: %P" \
        python train_from_yaml.py "$YAML" 2>&1 | tee -a "$LOG_FILE"

    echo -e "\nFinished: $YAML\n" | tee -a "$LOG_FILE"
done

echo "--- All benchmarks complete. Results saved to $LOG_FILE ---"

  