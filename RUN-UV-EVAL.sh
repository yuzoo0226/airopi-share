#!/bin/sh
#PBS -q rt_HG
#PBS -P gch51606
#PBS -l select=1
#PBS -l walltime=01:00:00
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
: "${TRAIN_DATASET_NAME:=${DATASET_NAME}_train}"
: "${VAL_DATASET_NAME:=${DATASET_NAME}_val}"

openpi_export_standard_vars
openpi_require_vars WORKING_DIR DATA_DIR HF_LEROBOT_HOME CACHE_ROOT WANDB_API_KEY
openpi_setup_cache_env
openpi_log_standard_paths
openpi_cd_working_dir
openpi_uv_sync

: "${EVAL_CONFIG_NAME:=pi0_task689_level12_state_diff_train}"
: "${EVAL_EXP_NAME:=pi0_hsr_ph2_task689_level12_state_diff_arm_head_relative_gripper_base_train_gpu8}"
: "${EVAL_VAL_REPO_ID:=lerobot_datasets/task689_level12_val}"
: "${EVAL_ASSET_ID:=lerobot_datasets/task689_level12_train}"
: "${EVAL_NUM_BATCHES:=50}"
: "${EVAL_NUM_SAMPLE_STEPS:=10}"
: "${EVAL_CHECKPOINT_DIR:=${CHECKPOINT_DIR}/${EVAL_CONFIG_NAME}/${EVAL_EXP_NAME}}"

if [ ! -d "${EVAL_CHECKPOINT_DIR}" ]; then
  echo "[ERROR]: Checkpoint directory not found: ${EVAL_CHECKPOINT_DIR}"
  exit 1
fi

: "${EVAL_MODE:=all}"

steps=$(find "${EVAL_CHECKPOINT_DIR}" -maxdepth 2 -type d -name params -print \
  | awk -F/ '{print $(NF-1)}' \
  | grep -E '^[0-9]+$' \
  | sort -n -u)

if [ -z "${steps}" ]; then
  echo "[ERROR]: No checkpoint steps found under ${EVAL_CHECKPOINT_DIR}"
  exit 1
fi

num_steps=$(printf "%s\n" "${steps}" | wc -l | tr -d ' ')
case "${EVAL_MODE}" in
  all)
    selected_steps="${steps}"
    ;;
  quartile)
    selected_steps=$(printf "%s\n" "${steps}" | awk -v n="${num_steps}" '
    BEGIN {
      split("0.25 0.5 0.75 1.0", p);
      for (i = 1; i <= 4; i++) {
        val = n * p[i];
        idx = int(val);
        if (val > idx) idx += 1;
        if (idx < 1) idx = 1;
        if (idx > n) idx = n;
        want[idx] = 1;
      }
    }
    {
      if (want[NR]) print $0
    }
    ')
    ;;
  *)
    echo "[ERROR]: Unknown EVAL_MODE=${EVAL_MODE} (use quartile or all)"
    exit 1
    ;;
esac

if [ -z "${selected_steps}" ]; then
  echo "[ERROR]: Failed to select checkpoints from ${num_steps} steps"
  exit 1
fi

echo "[INFO]: Evaluating ${EVAL_MODE} checkpoints (${num_steps} total steps): ${selected_steps}"
for step in ${selected_steps}; do
  echo "[INFO]: Evaluating checkpoint step=${step}"
  uv run python scripts/eval_val_loss.py "${EVAL_CONFIG_NAME}" \
    --exp-name "${EVAL_EXP_NAME}" \
    --val-repo-id "${EVAL_VAL_REPO_ID}" \
    --asset-id "${EVAL_ASSET_ID}" \
    --num-batches "${EVAL_NUM_BATCHES}" \
    --num-sample-steps "${EVAL_NUM_SAMPLE_STEPS}" \
    --checkpoint-dir "${EVAL_CHECKPOINT_DIR}" \
    --step "${step}"
done

echo "[INFO]: Done!"
