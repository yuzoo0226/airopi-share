#!/bin/bash
#PBS -q R1601791
#PBS -v RTYPE=rt_HF
#PBS -P gch51606
#PBS -l select=1:ncpus=192:mpiprocs=8
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -m n
#PBS -koed
#PBS -o outputs/

set -eu -o pipefail

echo "Job ID: ${PBS_JOBID:-local}"
echo "Node Allocated: $(cat "$PBS_NODEFILE" | sort -u | tr '\n' ' ')"
echo "Date: $(date)"
echo "----------------------------------------"

source /etc/profile.d/modules.sh
module load hpcx/2.20

SCRIPT_DIR=$PBS_O_WORKDIR
cd "${SCRIPT_DIR}"

# Load .env (training config), then .env.local (personal overrides).
ENV_FILE="${HSR_OPENPI_ENV_FILE:-${SCRIPT_DIR}/.env}"
if [ -f "${ENV_FILE}" ]; then
  set -a; . "${ENV_FILE}"; set +a
  echo "[INFO] Loaded ${ENV_FILE}"
fi
if [ -f "${SCRIPT_DIR}/.env.local" ]; then
  set -a; . "${SCRIPT_DIR}/.env.local"; set +a
  echo "[INFO] Loaded ${SCRIPT_DIR}/.env.local (personal overrides)"
fi

: "${DATA_DIR:=/groups/grp00000/lerobot_datasets}"
: "${HF_LEROBOT_HOME:=/groups/grp00000}"
: "${CACHE_ROOT:=${SCRIPT_DIR}/tmp_storage}"
: "${CHECKPOINT_DIR:=/groups/grp00000/${USER}/AiroPi/checkpoints}"
export DATA_DIR HF_LEROBOT_HOME CACHE_ROOT CHECKPOINT_DIR

# Cache directories
mkdir -p "${CACHE_ROOT}/uv-cache" "${CACHE_ROOT}/uv-data" "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/cache" "${CACHE_ROOT}/openpi_cache"
export UV_CACHE_DIR="${CACHE_ROOT}/uv-cache"
export UV_DATA_DIR="${CACHE_ROOT}/uv-data"
export TMPDIR="${CACHE_ROOT}/tmp"
export XDG_CACHE_HOME="${CACHE_ROOT}/cache"
export OPENPI_DATA_HOME="${CACHE_ROOT}/openpi_cache"

# ---- Training config ----
OPENPI_CONFIG_YAML="${OPENPI_CONFIG_YAML:-}"
EXP_NAME="${EXP_NAME:-pi05_hsr_multinode}"
OPENPI_RESUME="${OPENPI_RESUME:-1}"
OPENPI_OVERWRITE="${OPENPI_OVERWRITE:-0}"
YAML_DATASET_DATA_DIR=""
YAML_DATASET_REPO_ID=""
YAML_DATASET_ASSETS_DIR=""
YAML_DATASET_ASSET_ID=""
YAML_BASE_MODEL_URL=""
YAML_GPU_NUM_GPUS=""
YAML_DATASET_HF_HOME=""
OPENPI_YAML_METADATA_SCRIPT="${SCRIPT_DIR}/scripts/openpi_utils/resolve_training_yaml_metadata.py"

# Dataset root (used by train.py for local dataset loading)
if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  if [ ! -f "${OPENPI_CONFIG_YAML}" ] && [ -f "${SCRIPT_DIR}/${OPENPI_CONFIG_YAML}" ]; then
    OPENPI_CONFIG_YAML="${SCRIPT_DIR}/${OPENPI_CONFIG_YAML}"
  fi
  if [ ! -f "${OPENPI_CONFIG_YAML}" ]; then
    echo "[ERROR] OPENPI_CONFIG_YAML not found: ${OPENPI_CONFIG_YAML}"
    exit 1
  fi
  if [ ! -f "${OPENPI_YAML_METADATA_SCRIPT}" ]; then
    echo "[ERROR] YAML metadata helper not found: ${OPENPI_YAML_METADATA_SCRIPT}"
    exit 1
  fi

  yaml_meta_lines=$(python3 "${OPENPI_YAML_METADATA_SCRIPT}" "${OPENPI_CONFIG_YAML}")

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

  if [ -n "${YAML_DATASET_DATA_DIR}" ]; then
    export OPENPI_DATASET_ROOT="${YAML_DATASET_DATA_DIR}"
  fi
fi

# ---- Distributed settings ----
NODEFILE=$PBS_NODEFILE
NODE_COUNT=$(sort -u "$NODEFILE" | wc -l)
NUM_NODES=$NODE_COUNT
NUM_GPU_PER_NODE=8
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

MASTER_ADDR=$(sort -u "${PBS_NODEFILE}" | head -1)
PBS_JOBID_NUM=$(echo "${PBS_JOBID}" | grep -o '^[0-9]*')
MASTER_PORT=$((10000 + PBS_JOBID_NUM % 50000))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${PBS_JOBID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

echo "[INFO] MASTER_ADDR=${MASTER_ADDR}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] NUM_NODES=${NUM_NODES}"
echo "[INFO] CONFIG=${OPENPI_CONFIG_YAML:-tyro}"

# ---- Pre-download base model weights (single process, before mpirun) ----
# Avoids race condition where 16 processes try to download simultaneously.
if [ -n "${YAML_BASE_MODEL_URL}" ]; then
  echo "[INFO] Pre-downloading base model: ${YAML_BASE_MODEL_URL}"
  pixi run uv run python -c "
from openpi.shared.download import maybe_download
maybe_download('${YAML_BASE_MODEL_URL}')
print('[INFO] Base model download complete')
"
fi

# ---- Pre-compute norm stats (single process, before mpirun) ----
if [ -n "${OPENPI_DATASET_ROOT:-}" ]; then
  dataset_parent_dir=$(basename "$(dirname "${OPENPI_DATASET_ROOT}")")
  dataset_name=$(basename "${OPENPI_DATASET_ROOT}")
  episodes_stats_file="${OPENPI_DATASET_ROOT}/meta/episodes_stats.jsonl"
  data_chunk_dir="${OPENPI_DATASET_ROOT}/data/"

  if [ -n "${YAML_DATASET_ASSETS_DIR}" ]; then
    norm_stats_asset_id="${YAML_DATASET_ASSET_ID:-${YAML_DATASET_REPO_ID:-${dataset_parent_dir}/${dataset_name}}}"
    norm_stats_output_file="${YAML_DATASET_ASSETS_DIR%/}/${norm_stats_asset_id}/norm_stats.json"
  else
    if [ -n "${OPENPI_CONFIG_YAML}" ]; then
      train_config_name=$(basename "${OPENPI_CONFIG_YAML%.*}")
    else
      train_config_name="${EXP_NAME}"
    fi
    norm_stats_output_file="assets/${train_config_name}/${dataset_parent_dir}/${dataset_name}/norm_stats.json"
  fi

  if [ -f "${episodes_stats_file}" ]; then
    mkdir -p "$(dirname "${norm_stats_output_file}")"
    echo "[INFO] Resolved dataset root: ${OPENPI_DATASET_ROOT}"
    echo "[INFO] Norm stats output: ${norm_stats_output_file}"
    echo "[INFO] Running aggregate stats: scripts.aggregate_stats_fast"
    JAX_PLATFORMS=cpu pixi run uv run python -m scripts.aggregate_stats_fast \
      --episodes-stats "${episodes_stats_file}" \
      --output-file "${norm_stats_output_file}" \
      --chunk-dir "${data_chunk_dir}" \
      --action-column "action.relative" \
      --action-mode "relative"
    echo "[INFO] Norm stats updated: ${norm_stats_output_file}"
  else
    echo "[WARN] Skip aggregate stats (episodes stats not found): ${episodes_stats_file}"
  fi
else
  echo "[WARN] Skip aggregate stats (OPENPI_DATASET_ROOT is unset)"
fi

# ---- Build train command ----
train_args=()
if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  train_args+=(--config-yaml "${OPENPI_CONFIG_YAML}")
fi
train_args+=(--exp-name "${EXP_NAME}")
if [ "${OPENPI_RESUME}" = "1" ]; then
  train_args+=(--resume)
elif [ "${OPENPI_OVERWRITE}" = "1" ]; then
  train_args+=(--overwrite)
fi

# ---- Launch ----
# OMPI_COMM_WORLD_RANK etc. are auto-set by mpirun.
# _parse_env() in distributed.py maps them to RANK/WORLD_SIZE/etc.
# Only MASTER_ADDR and MASTER_PORT need explicit passing.
mpirun \
  -np $NUM_GPUS \
  -hostfile $HOSTFILE_NAME \
  --map-by ppr:$NUM_GPU_PER_NODE:node \
  -x MASTER_ADDR="${MASTER_ADDR}" \
  -x MASTER_PORT="${MASTER_PORT}" \
  -x CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  -x DATA_DIR -x HF_LEROBOT_HOME -x CACHE_ROOT -x CHECKPOINT_DIR \
  -x UV_CACHE_DIR -x UV_DATA_DIR -x TMPDIR -x XDG_CACHE_HOME \
  -x OPENPI_DATA_HOME \
  -x OPENPI_DATASET_ROOT \
  -x WANDB_API_KEY \
  -bind-to none \
  pixi run uv run python scripts/train.py "${train_args[@]}"

echo "[INFO] Finished at $(date)"
