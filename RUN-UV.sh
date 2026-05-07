#!/bin/sh
#PBS -q rt_HG
#PBS -P gch51606
#PBS -l select=1
#PBS -l walltime=60:00:00
#PBS -j oe

set -e

echo "[INFO]: Started at $(date)"

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

# ---- YAML config support (ported from RUN-UV-SBATCH-OPENPI-MULTINODE.sh) ----
: "${OPENPI_CONFIG_YAML:=}"
: "${CONFIG_NAME:=}"
: "${OPENPI_AGGREGATE_STATS_ENABLE:=0}"
: "${OPENPI_RESUME:=1}"
: "${OPENPI_OVERWRITE:=0}"

ENV_DATA_DIR="${DATA_DIR:-}"
ENV_DATASET_NAME="${DATASET_NAME:-}"
YAML_DATASET_DATA_DIR=""
YAML_DATASET_REPO_ID=""
YAML_GPU_NUM_GPUS=""
YAML_DATASET_ASSETS_DIR=""
YAML_DATASET_ASSET_ID=""
YAML_BASE_MODEL_URL=""
YAML_DATASET_HF_HOME=""

: "${OPENPI_UTILS_SCRIPT_DIR:=${SCRIPT_DIR}/scripts/openpi_utils}"
OPENPI_YAML_METADATA_SCRIPT="${OPENPI_UTILS_SCRIPT_DIR}/resolve_training_yaml_metadata.py"

# Ensure uv cache dirs exist for YAML metadata parsing.
if [ -z "${CACHE_ROOT:-}" ]; then
  CACHE_ROOT="${SCRIPT_DIR}/tmp_storage"
fi
mkdir -p "${CACHE_ROOT}/uv-cache" "${CACHE_ROOT}/uv-data" "${CACHE_ROOT}/uv-python" "${CACHE_ROOT}/cache"
export UV_CACHE_DIR="${CACHE_ROOT}/uv-cache"
export UV_DATA_DIR="${CACHE_ROOT}/uv-data"
export UV_PYTHON_DIR="${CACHE_ROOT}/uv-python"
export XDG_DATA_HOME="${CACHE_ROOT}/uv-data"
export XDG_CACHE_HOME="${CACHE_ROOT}/cache"

if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  if [ ! -f "${OPENPI_CONFIG_YAML}" ] && [ -f "${SCRIPT_DIR}/${OPENPI_CONFIG_YAML}" ]; then
    OPENPI_CONFIG_YAML="${SCRIPT_DIR}/${OPENPI_CONFIG_YAML}"
  fi
  if [ ! -f "${OPENPI_CONFIG_YAML}" ]; then
    echo "[ERROR]: OPENPI_CONFIG_YAML not found: ${OPENPI_CONFIG_YAML}"
    exit 1
  fi
  if [ ! -f "${OPENPI_YAML_METADATA_SCRIPT}" ]; then
    echo "[ERROR]: YAML metadata helper not found: ${OPENPI_YAML_METADATA_SCRIPT}"
    exit 1
  fi

  yaml_meta_lines=$(pixi run uv run python "${OPENPI_YAML_METADATA_SCRIPT}" "${OPENPI_CONFIG_YAML}")
  while IFS='=' read -r key value; do
    case "${key}" in
      DATASET_DATA_DIR) YAML_DATASET_DATA_DIR="${value}" ;;
      DATASET_REPO_ID) YAML_DATASET_REPO_ID="${value}" ;;
      DATASET_ASSETS_DIR) YAML_DATASET_ASSETS_DIR="${value}" ;;
      DATASET_ASSET_ID) YAML_DATASET_ASSET_ID="${value}" ;;
      GPU_NUM_GPUS) YAML_GPU_NUM_GPUS="${value}" ;;
      BASE_MODEL_URL) YAML_BASE_MODEL_URL="${value}" ;;
      DATASET_HF_HOME) YAML_DATASET_HF_HOME="${value}" ;;
    esac
  done <<EOF
${yaml_meta_lines}
EOF
fi

# Reconcile DATA_DIR / DATASET_NAME between .env and YAML.
if [ -n "${YAML_DATASET_DATA_DIR}" ]; then
  YAML_DATASET_NAME_FROM_YAML="$(basename "${YAML_DATASET_DATA_DIR}")"
  YAML_DATA_DIR_FROM_YAML="$(dirname "${YAML_DATASET_DATA_DIR}")"

  if [ -n "${ENV_DATA_DIR}" ] && [ "${ENV_DATA_DIR}" != "${YAML_DATA_DIR_FROM_YAML}" ]; then
    echo "[ERROR]: DATA_DIR mismatch between .env and YAML."
    echo "[ERROR]: .env DATA_DIR=${ENV_DATA_DIR}"
    echo "[ERROR]: YAML data_dir parent=${YAML_DATA_DIR_FROM_YAML} (from ${YAML_DATASET_DATA_DIR})"
    exit 1
  fi
  if [ -n "${ENV_DATASET_NAME}" ] && [ "${ENV_DATASET_NAME}" != "${YAML_DATASET_NAME_FROM_YAML}" ]; then
    echo "[ERROR]: DATASET_NAME mismatch between .env and YAML."
    echo "[ERROR]: .env DATASET_NAME=${ENV_DATASET_NAME}"
    echo "[ERROR]: YAML inferred DATASET_NAME=${YAML_DATASET_NAME_FROM_YAML}"
    exit 1
  fi

  DATA_DIR="${YAML_DATA_DIR_FROM_YAML}"
  DATASET_NAME="${YAML_DATASET_NAME_FROM_YAML}"
elif [ -n "${YAML_DATASET_REPO_ID}" ]; then
  YAML_DATASET_NAME_FROM_REPO="$(basename "${YAML_DATASET_REPO_ID}")"
  if [ -n "${ENV_DATASET_NAME}" ] && [ "${ENV_DATASET_NAME}" != "${YAML_DATASET_NAME_FROM_REPO}" ]; then
    echo "[ERROR]: DATASET_NAME mismatch between .env and YAML repo_id."
    echo "[ERROR]: .env DATASET_NAME=${ENV_DATASET_NAME}"
    echo "[ERROR]: YAML repo_id basename=${YAML_DATASET_NAME_FROM_REPO}"
    exit 1
  fi
  if [ -z "${DATASET_NAME:-}" ]; then
    DATASET_NAME="${YAML_DATASET_NAME_FROM_REPO}"
  fi
fi

openpi_set_default_path_vars
if [ -n "${YAML_DATASET_HF_HOME}" ]; then
  yaml_meta_info="${YAML_DATASET_HF_HOME}/${YAML_DATASET_REPO_ID}/meta/info.json"
  if [ -f "${yaml_meta_info}" ]; then
    if [ -n "${HF_LEROBOT_HOME:-}" ] && [ "${HF_LEROBOT_HOME}" != "${YAML_DATASET_HF_HOME}" ]; then
      echo "[WARN]: HF_LEROBOT_HOME (${HF_LEROBOT_HOME}) differs from YAML-derived root (${YAML_DATASET_HF_HOME})."
      echo "[WARN]: Overriding HF_LEROBOT_HOME with YAML-derived root because dataset metadata exists there."
    fi
    HF_LEROBOT_HOME="${YAML_DATASET_HF_HOME}"
  else
    echo "[WARN]: YAML-derived dataset metadata not found at ${yaml_meta_info}."
    echo "[WARN]: Keeping existing HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-<unset>}."
  fi
fi

# Resolve train config name.
if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  if [ -n "${CONFIG_NAME}" ]; then
    TRAIN_CONFIG_NAME="${CONFIG_NAME}"
  else
    TRAIN_CONFIG_NAME="$(basename "${OPENPI_CONFIG_YAML%.*}")"
  fi
else
  openpi_set_default_train_vars
  TRAIN_CONFIG_NAME="${CONFIG_NAME}"
fi
if [ -z "${DATASET_NAME:-}" ]; then
  DATASET_NAME="${TRAIN_CONFIG_NAME}"
fi

openpi_export_standard_vars
openpi_require_vars WORKING_DIR DATA_DIR HF_LEROBOT_HOME CACHE_ROOT
openpi_warn_if_unset WANDB_API_KEY
openpi_setup_cache_env
openpi_log_standard_paths

echo "[INFO]: Config ownership:"
echo "[INFO]:   YAML (experiment semantics): ${OPENPI_CONFIG_YAML:-<none>}"
echo "[INFO]:   .env (launcher/runtime): OPENPI_RESUME=${OPENPI_RESUME}, OPENPI_OVERWRITE=${OPENPI_OVERWRITE}"
echo "[INFO]: Resolved dataset root: ${DATA_DIR}/${DATASET_NAME}"

openpi_cd_working_dir
openpi_uv_sync

# ---- Aggregate stats ----
PARENT_DATA_DIR=$(basename "${DATA_DIR}")
EPISODES_STATS_FILE="${DATA_DIR}/${DATASET_NAME}/meta/episodes_stats.jsonl"
DATA_CHUNK_DIR="${DATA_DIR}/${DATASET_NAME}/data/"

# Determine norm_stats output path: prefer YAML assets_dir/asset_id, fall back to config name.
if [ -n "${YAML_DATASET_ASSETS_DIR}" ]; then
  _norm_asset_id="${YAML_DATASET_ASSET_ID:-${YAML_DATASET_REPO_ID:-${PARENT_DATA_DIR}/${DATASET_NAME}}}"
  NORM_STATS_OUTPUT_FILE="${YAML_DATASET_ASSETS_DIR%/}/${_norm_asset_id}/norm_stats.json"
else
  NORM_STATS_OUTPUT_FILE="assets/${TRAIN_CONFIG_NAME}/${PARENT_DATA_DIR}/${DATASET_NAME}/norm_stats.json"
fi

echo "[INFO]: DATASET_NAME=${DATASET_NAME}"

if [ "${OPENPI_AGGREGATE_STATS_ENABLE}" = "1" ]; then
  echo "[INFO]: Running aggregate_stats_simple.py..."
  JAX_PLATFORMS=cpu pixi run uv run python scripts/aggregate_stats_simple.py \
    --episodes-stats "${EPISODES_STATS_FILE}" \
    --output-file "${NORM_STATS_OUTPUT_FILE}" \
    --chunk-dir "${DATA_CHUNK_DIR}" \
    --action-column "action.relative" \
    --action-mode "relative"
else
  echo "[INFO]: aggregate stats is disabled"
fi

# ---- Train ----
if [ "${OPENPI_RESUME}" = "1" ] && [ "${OPENPI_OVERWRITE}" = "1" ]; then
  echo "[ERROR]: OPENPI_RESUME and OPENPI_OVERWRITE cannot both be 1."
  exit 1
fi

train_args=""
if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  train_args="--config-yaml ${OPENPI_CONFIG_YAML}"
else
  train_args="${CONFIG_NAME}"
fi
train_args="${train_args} --exp-name=${EXP_NAME}"

if [ "${OPENPI_RESUME}" = "1" ]; then
  train_args="${train_args} --resume"
elif [ "${OPENPI_OVERWRITE}" = "1" ]; then
  train_args="${train_args} --overwrite"
fi

echo "[INFO]: Running train.py..."
# shellcheck disable=SC2086
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 pixi run uv run python scripts/train.py ${train_args}

echo "[INFO]: Done at $(date)"
