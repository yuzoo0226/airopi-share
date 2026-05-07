#!/bin/bash
# PaliGemmaの重みを事前にダウンロードするスクリプト
# 計算ノードではなくログインノードで実行してください

set -e

echo "[INFO]: PaliGemma weights download script started at $(date)"

# ---- Config loading (per-user) ----
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1
echo "[INFO]: Using SCRIPT_DIR=$SCRIPT_DIR"

ENV_FILE="${HSR_OPENPI_ENV_FILE:-${SCRIPT_DIR}/.env.local}"
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
: "${WORKING_DIR:=${SCRIPT_DIR}}"
: "${CACHE_ROOT:=/path/to/${USER}/AiroPi/tmp_storage}"
export WORKING_DIR CACHE_ROOT
# ↑----- Defaults -----↑

# Create tmp folder
mkdir -p "${CACHE_ROOT}"/{uv-cache,uv-data,uv-python,openpi_cache,huggingface_home,tmp,cache}

# Set UV and other environment variables
export UV_CACHE_DIR=${CACHE_ROOT}/uv-cache
export UV_DATA_DIR=${CACHE_ROOT}/uv-data
export UV_PYTHON_DIR=${CACHE_ROOT}/uv-python
export XDG_DATA_HOME=${CACHE_ROOT}/uv-data
export OPENPI_DATA_HOME=${CACHE_ROOT}/openpi_cache
export HF_HOME=${CACHE_ROOT}/huggingface_home
export TMPDIR=${CACHE_ROOT}/tmp
export XDG_CACHE_HOME=${CACHE_ROOT}/cache

echo "[INFO]: CACHE_ROOT=${CACHE_ROOT}"
echo "[INFO]: OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"

cd "${WORKING_DIR}" || exit 1

echo "[INFO]: Starting uv sync..."
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "[INFO]: Downloading PaliGemma weights..."
uv run python -c "
import openpi.shared.download as download

print('[INFO]: Downloading PaliGemma model weights (pt_224.npz)...')
path1 = download.maybe_download(
    'gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz', 
    gs={'token': 'anon'}
)
print(f'[INFO]: Downloaded to: {path1}')

print('[INFO]: Downloading PaliGemma tokenizer...')
path2 = download.maybe_download(
    'gs://big_vision/paligemma_tokenizer.model', 
    gs={'token': 'anon'}
)
print(f'[INFO]: Downloaded to: {path2}')

print('[INFO]: All PaliGemma weights downloaded successfully!')
"

echo "[INFO]: Download completed at $(date)"
echo "[INFO]: Weights are cached in ${OPENPI_DATA_HOME}"
