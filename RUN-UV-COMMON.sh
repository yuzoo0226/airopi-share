#!/bin/sh

# Shared helpers for RUN-UV*.sh launch scripts.
# This file is sourced by launcher scripts and should not be executed directly.

openpi_load_cuda_module() {
  if command -v module >/dev/null 2>&1; then
    module load cuda
    echo "[INFO]: Modules loaded"
  fi
}

openpi_resolve_script_dir() {
  _openpi_script_path="$1"
  if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
  elif [ -n "${PBS_O_WORKDIR:-}" ]; then
    SCRIPT_DIR="$PBS_O_WORKDIR"
  else
    SCRIPT_DIR=$(cd "$(dirname "$_openpi_script_path")" && pwd)
  fi
  export SCRIPT_DIR

  cd "$SCRIPT_DIR" || return 1
  mkdir -p "${SCRIPT_DIR}/outputs"
  echo "[INFO]: Using SCRIPT_DIR=$SCRIPT_DIR"
}

openpi_load_env_file() {
  # Load .env (training config), then .env.local (personal overrides).
  # .env.local values take precedence over .env.
  # HSR_OPENPI_ENV_FILE overrides the base .env path.
  _openpi_base_env="${HSR_OPENPI_ENV_FILE:-${SCRIPT_DIR}/.env}"
  _openpi_local_env="${SCRIPT_DIR}/.env.local"
  _openpi_loaded=0

  if [ -f "$_openpi_base_env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$_openpi_base_env"
    set +a
    echo "[INFO]: Loaded env from $_openpi_base_env"
    _openpi_loaded=1
  fi

  if [ -f "$_openpi_local_env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$_openpi_local_env"
    set +a
    echo "[INFO]: Loaded env from $_openpi_local_env (personal overrides)"
    _openpi_loaded=1
  fi

  if [ "$_openpi_loaded" -eq 0 ]; then
    echo "[WARN]: No .env or .env.local found. Using script defaults."
  fi
}

openpi_set_default_path_vars() {
  : "${WORKING_DIR:=${SCRIPT_DIR}}"
  : "${DATA_DIR:=/groups/grp00000/lerobot_datasets}"
  : "${CACHE_ROOT:=/groups/grp00000/${USER}/AiroPi/tmp_storage}"
  : "${HF_LEROBOT_HOME:=/groups/grp00000}"
  : "${CHECKPOINT_DIR:=/groups/grp00000/${USER}/AiroPi/checkpoints}"
}

openpi_set_default_train_vars() {
  : "${CONFIG_NAME:=pi0_hsr_test}"
  : "${EXP_NAME:=test}"
  : "${DATASET_NAME:=${CONFIG_NAME}}"
}

openpi_export_standard_vars() {
  export WORKING_DIR DATA_DIR CACHE_ROOT HF_LEROBOT_HOME CHECKPOINT_DIR
  export CONFIG_NAME EXP_NAME DATASET_NAME WANDB_API_KEY
}

openpi_require_vars() {
  while [ "$#" -gt 0 ]; do
    _openpi_name="$1"
    _openpi_val=$(eval "printf '%s' \"\${${_openpi_name}:-}\"")
    if [ -z "$_openpi_val" ]; then
      echo "[ERROR]: Required variable '${_openpi_name}' is not set. Set it in .env (or HSR_OPENPI_ENV_FILE target) or here."
      return 1
    fi
    shift
  done
}

openpi_warn_if_unset() {
  _openpi_name="$1"
  _openpi_val=$(eval "printf '%s' \"\${${_openpi_name}:-}\"")
  if [ -z "$_openpi_val" ]; then
    echo "[WARN]: ${_openpi_name} is not set."
  fi
}

openpi_setup_cache_env() {
  mkdir -p \
    "${CACHE_ROOT}/uv-cache" \
    "${CACHE_ROOT}/uv-data" \
    "${CACHE_ROOT}/uv-python" \
    "${CACHE_ROOT}/openpi_cache" \
    "${CACHE_ROOT}/huggingface_home" \
    "${CACHE_ROOT}/tmp" \
    "${CACHE_ROOT}/cache"

  export UV_CACHE_DIR="${CACHE_ROOT}/uv-cache"
  export UV_DATA_DIR="${CACHE_ROOT}/uv-data"
  export UV_PYTHON_DIR="${CACHE_ROOT}/uv-python"
  export XDG_DATA_HOME="${CACHE_ROOT}/uv-data"
  export OPENPI_DATA_HOME="${CACHE_ROOT}/openpi_cache"
  export HF_HOME="${CACHE_ROOT}/huggingface_home"
  export TMPDIR="${CACHE_ROOT}/tmp"
  export XDG_CACHE_HOME="${CACHE_ROOT}/cache"
}

openpi_log_standard_paths() {
  echo "[INFO]: HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "[INFO]: CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  echo "[INFO]: DATA_DIR=${DATA_DIR}"
  echo "[INFO]: CACHE_ROOT=${CACHE_ROOT}"
}

openpi_cd_working_dir() {
  cd "${WORKING_DIR}" || return 1
}

openpi_uv_sync() {
  echo "[INFO]: Starting uv sync..."
  GIT_LFS_SKIP_SMUDGE=1 pixi run uv sync
}
