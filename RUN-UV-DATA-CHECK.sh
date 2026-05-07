#!/bin/bash
#SBATCH --job-name=airopi_uv_data_check
#SBATCH --partition=2-def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.out

set -e

echo "[INFO]: Job started at $(date)"
echo "[INFO]: Running on node: $(hostname)"

COMMON_SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}}"
COMMON_SCRIPT="${COMMON_SCRIPT_DIR}/RUN-UV-COMMON.sh"
if [ ! -f "${COMMON_SCRIPT}" ]; then
  echo "[ERROR]: Common script not found: ${COMMON_SCRIPT}"
  exit 1
fi
# shellcheck source=/dev/null
. "${COMMON_SCRIPT}"

openpi_load_cuda_module
openpi_resolve_script_dir "$0"
openpi_load_env_file
openpi_set_default_path_vars
openpi_set_default_train_vars
openpi_export_standard_vars
openpi_require_vars WORKING_DIR DATA_DIR HF_LEROBOT_HOME CACHE_ROOT WANDB_API_KEY
openpi_setup_cache_env
openpi_log_standard_paths
openpi_cd_working_dir
openpi_uv_sync

# uv run python scripts/scan_action_outliers.py pi05_hsr_optimal_microwave --action-key action.state_diff --dim 6 --limit 5000 --top-k 20
# uv run python scripts/scan_action_outliers.py pi05_hsr_optimal_microwave --action-key action.state_diff --dim 6 --top-k 50
# uv run python scripts/scan_action_outliers.py pi05_hsr_optimal_microwave --compare-normalize --dim 11 --top-k 20 --limit 5000
# uv run python scripts/scan_action_outliers.py pi05_hsr_optimal_microwave --compare-normalize --dim 11 --top-k 20 --limit 5000
# uv run python scripts/scan_action_outliers.py pi05_hsr_optimal_microwave --action-key action.state_diff --dim 6 --fast-column --stop-threshold 1e6
# uv run python scripts/scan_action_outliers.py pi05_hsr_optimal_microwave --fast-column --print-keys
# uv run python scripts/scan_action_outliers.py pi05_hsr_optimal_microwave --compare-normalize --dim 11 --stop-threshold 1e6
# uv run python scripts/scan_action_outliers.py pi05_hsr_optimal_microwave --parquet-fast --action-key action.state_diff --dim 6 --stop-threshold 1e6

uv run python scripts/scan_action_outliers.py pi05_hsr_optimal_microwave \
  --scan-invalid --threshold 1e4 \
  --output-csv outputs/invalid_actions.csv \
  --exclude-episodes-csv outputs/exclude_episodes.csv

echo "[INFO]: Job finished at $(date)"
