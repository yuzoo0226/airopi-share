#!/bin/sh
#PBS -q rt_HG
#PBS -P gch51606
#PBS -l select=1
#PBS -l walltime=30:00:00
#PBS -j oe

module load singularitypro
echo "[INFO]: Modules loaded"

# ---- Config loading (per-user) ----
# Load environment overrides from .env.local or .env if present.
# You can set AIROPI_ENV_FILE to point to a custom env file.
# Resolve working directory correctly under PBS (qsub) or local shell
if [ -n "${PBS_O_WORKDIR:-}" ]; then
  SCRIPT_DIR="$PBS_O_WORKDIR"
else
  SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
fi
cd "$SCRIPT_DIR" || exit 1
echo "[INFO]: Using SCRIPT_DIR=$SCRIPT_DIR"
ENV_FILE="${AIROPI_ENV_FILE:-${SCRIPT_DIR}/.env.local}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  . "$ENV_FILE"
  set +a
  echo "[INFO]: Loaded env from $ENV_FILE"
elif [ -f "${SCRIPT_DIR}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  . "${SCRIPT_DIR}/.env"
  set +a
  echo "[INFO]: Loaded env from .env"
else
  echo "[WARN]: No .env.local or .env found. Using script defaults."
fi

# ↓----- Defaults (used only if not set via env file) -----↓
# Change respectively to your .sif file and data directory
: "${WORKING_DIR:=${SCRIPT_DIR}}"
: "${SIF_PATH:=${SCRIPT_DIR}/airopi.sif}"
: "${DATA_DIR:=/groups/grp00000/lerobot_datasets}"
: "${CACHE_ROOT:=/groups/grp00000/${USER}/airopi/tmp_storage}"
: "${HF_LEROBOT_HOME:=/groups/grp00000}"
: "${CHECKPOINT_DIR:=/groups/grp00000/${USER}/airopi/checkpoints}"
: "${CONFIG_NAME:=pi0_hsr_test}"
: "${EXP_NAME:=test}"
export WORKING_DIR SIF_PATH DATA_DIR CACHE_ROOT HF_LEROBOT_HOME CHECKPOINT_DIR CONFIG_NAME EXP_NAME WANDB_API_KEY
# ↑----- Defaults -----↑

# Basic validation
for _v in WORKING_DIR SIF_PATH DATA_DIR HF_LEROBOT_HOME CACHE_ROOT WANDB_API_KEY; do
  eval "_val=\${$_v}"
  if [ -z "$_val" ]; then
    echo "[ERROR]: Required variable '$_v' is not set. Set it in .env.local/.env or here."
    exit 1
  fi
done

# Create tmp folder
mkdir -p "${CACHE_ROOT}"/{uv-cache,uv-data,uv-python,openpi_cache,huggingface_home,tmp,cache}


singularity exec --nv \
  --bind "${HF_LEROBOT_HOME}:${HF_LEROBOT_HOME}" \
  --bind "${DATA_DIR}:${DATA_DIR}" \
  --bind "${CACHE_ROOT}:${CACHE_ROOT}" \
  --bind "${CHECKPOINT_DIR}:${CHECKPOINT_DIR}" \
  --bind /groups/gch51606/dataset:/groups/gch51606/dataset \
  --bind /etc/ssl:/etc/ssl \
  --bind /etc/pki:/etc/pki \
  "$SIF_PATH" \
  bash -c '
  export PATH="/usr/local/:$PATH"

  echo "[INFO]: Variables set:"
  echo "[INFO]: HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "[INFO]: CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  echo "[INFO]: DATA_DIR=${DATA_DIR}"
  echo "[INFO]: SIF_PATH=${SIF_PATH}"
  echo "[INFO]: CACHE_ROOT=${CACHE_ROOT}"
  echo "[DEBUG]: WandB key starts with ${WANDB_API_KEY:0:5}"

  echo "[INFO]: Setting variables..."
  export UV_CACHE_DIR=${CACHE_ROOT}/uv-cache
  export UV_DATA_DIR=${CACHE_ROOT}/uv-data
  export UV_PYTHON_DIR=${CACHE_ROOT}/uv-python
  export XDG_DATA_HOME=${CACHE_ROOT}/uv-data
  export OPENPI_DATA_HOME=${CACHE_ROOT}/openpi_cache
  export HF_HOME=${CACHE_ROOT}/huggingface_home
  export TMPDIR=${CACHE_ROOT}/tmp
  export XDG_CACHE_HOME=${CACHE_ROOT}/cache

  cd ${WORKING_DIR}

  echo "[INFO]: Starting uv sync..."
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  
  echo "[INFO]: Running fine-tuning..."
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=${EXP_NAME} --checkpoint-base-dir ${CHECKPOINT_DIR} --overwrite
'
