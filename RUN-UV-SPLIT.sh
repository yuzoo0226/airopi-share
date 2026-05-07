#!/bin/bash
#PBS -q R1482708
#PBS -v RTYPE=rt_HC
#PBS -P gch51606
#PBS -l select=1
#PBS -l walltime=60:00:00
#PBS -j oe

set -e

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

: "${SPLIT_DATASET:=1}"
: "${SPLIT_TRAIN_FRACTION:=0.9}"
: "${SPLIT_SEED:=42}"
: "${SPLIT_OVERWRITE:=0}"
: "${SPLIT_SKIP_EXISTING:=1}"
: "${SPLIT_STRATIFY_BY_TASK:=1}"
: "${SPLIT_STRATIFY_TASK_MODE:=first}"
: "${TRAIN_DATASET_NAME:=${DATASET_NAME}_train}"
: "${VAL_DATASET_NAME:=${DATASET_NAME}_val}"

openpi_export_standard_vars
openpi_require_vars WORKING_DIR DATA_DIR HF_LEROBOT_HOME CACHE_ROOT WANDB_API_KEY
openpi_setup_cache_env
openpi_log_standard_paths
openpi_cd_working_dir
openpi_uv_sync

if [ "${SPLIT_DATASET}" = "1" ]; then
  echo "[INFO]: Splitting dataset into train/val..."
  SPLIT_ARGS="--dataset-root ${DATA_DIR}/${DATASET_NAME} --train-output ${DATA_DIR}/${TRAIN_DATASET_NAME} --val-output ${DATA_DIR}/${VAL_DATASET_NAME} --train-fraction ${SPLIT_TRAIN_FRACTION} --seed ${SPLIT_SEED}"
  if [ "${SPLIT_STRATIFY_BY_TASK}" = "1" ]; then
    SPLIT_ARGS="${SPLIT_ARGS} --stratify-by-task --stratify-task-mode ${SPLIT_STRATIFY_TASK_MODE} --check-task-ratio"
  fi
  if [ "${SPLIT_OVERWRITE}" = "1" ]; then
    SPLIT_ARGS="${SPLIT_ARGS} --overwrite"
  fi
  if [ "${SPLIT_SKIP_EXISTING}" = "1" ]; then
    SPLIT_ARGS="${SPLIT_ARGS} --skip-existing"
  fi
  uv run python scripts/split_lerobot_dataset.py ${SPLIT_ARGS}
  DATASET_NAME="${TRAIN_DATASET_NAME}"
  echo "[INFO]: Using train split for stats/training: DATASET_NAME=${DATASET_NAME}"
fi

echo "[INFO]: Done!"
