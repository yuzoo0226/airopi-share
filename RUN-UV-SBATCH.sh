#!/bin/bash
#SBATCH --job-name=airo_openpi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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

PARENT_DATA_DIR=$(basename "${DATA_DIR}")
echo "[INFO]: Running compute_norm_stats.py..."
echo "[INFO]: DATASET_NAME=${DATASET_NAME}"
JAX_PLATFORMS=cpu uv run python scripts/aggregate_stats_simple.py \
  --episodes-stats "${DATA_DIR}/${DATASET_NAME}/meta/episodes_stats.jsonl" \
  --output-file "assets/${CONFIG_NAME}/${PARENT_DATA_DIR}/${DATASET_NAME}/norm_stats.json" \
  --chunk-dir "${DATA_DIR}/${DATASET_NAME}/data/" \
  --action-column "action.relative" \
  --action-mode "state_diff_arm_head_relative_gripper_base"

echo "[INFO]: Running train.py..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "${CONFIG_NAME}" --exp-name="${EXP_NAME}" --checkpoint-base-dir "${CHECKPOINT_DIR}" --overwrite
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "${CONFIG_NAME}" --exp-name="${EXP_NAME}" --checkpoint-base-dir "${CHECKPOINT_DIR}" --resume

echo "[INFO]: Job finished at $(date)"
