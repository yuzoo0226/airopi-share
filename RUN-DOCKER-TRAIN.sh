#!/bin/bash

set -euo pipefail

COMMON_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMON_SCRIPT="${COMMON_SCRIPT_DIR}/RUN-UV-COMMON.sh"
if [ ! -f "${COMMON_SCRIPT}" ]; then
  echo "[ERROR]: Common script not found: ${COMMON_SCRIPT}"
  exit 1
fi
# shellcheck source=/dev/null
. "${COMMON_SCRIPT}"

openpi_resolve_script_dir "$0"
openpi_load_env_file

if [ ! -f "/.dockerenv" ] && [ ! -d "/home/openpi" ]; then
  echo "[ERROR]: RUN-DOCKER-TRAIN.sh is intended to run inside the Docker container."
  echo "[ERROR]: Repo mount /home/openpi was not found."
  exit 1
fi

: "${OPENPI_PROJECT_PYTHON:=3.11}"
: "${OPENPI_DOCKER_WORKING_DIR:=/home/openpi}"
: "${OPENPI_DOCKER_CACHE_ROOT:=/home/cache}"
: "${OPENPI_DOCKER_CHECKPOINT_DIR:=/home/openpi/checkpoints}"
: "${OPENPI_DOCKER_HF_LEROBOT_HOME:=/home/datasets}"
: "${OPENPI_DOCKER_DATA_DIR:=/home/datasets/lerobot_datasets}"
: "${OPENPI_AGGREGATE_STATS_ENABLE:=1}"
: "${OPENPI_RESUME:=0}"
: "${OPENPI_OVERWRITE:=1}"
: "${EXP_NAME:=docker_train}"
: "${CONFIG_NAME:=}"
: "${OPENPI_CONFIG_YAML:=}"

# .env is still the source of truth for run identity and credentials,
# but Docker paths must resolve to container mounts.
WORKING_DIR="${OPENPI_DOCKER_WORKING_DIR}"
CACHE_ROOT="${OPENPI_DOCKER_CACHE_ROOT}"
CHECKPOINT_DIR="${OPENPI_DOCKER_CHECKPOINT_DIR}"
HF_LEROBOT_HOME="${OPENPI_DOCKER_HF_LEROBOT_HOME}"
DATA_DIR="${DATA_DIR:-${OPENPI_DOCKER_DATA_DIR}}"
DATASET_NAME="${DATASET_NAME:-}"

if [ -z "${OPENPI_CONFIG_YAML}" ] && [ -z "${CONFIG_NAME}" ]; then
  echo "[ERROR]: Set OPENPI_CONFIG_YAML or CONFIG_NAME in .env."
  exit 1
fi
if [ "${OPENPI_RESUME}" = "1" ] && [ "${OPENPI_OVERWRITE}" = "1" ]; then
  echo "[ERROR]: OPENPI_RESUME and OPENPI_OVERWRITE cannot both be 1."
  exit 1
fi

export WORKING_DIR CACHE_ROOT CHECKPOINT_DIR HF_LEROBOT_HOME
export DATA_DIR DATASET_NAME WANDB_API_KEY CONFIG_NAME EXP_NAME OPENPI_CONFIG_YAML

openpi_warn_if_unset WANDB_API_KEY
openpi_setup_cache_env
openpi_log_standard_paths

VENV_DIR="${UV_PROJECT_ENVIRONMENT:-${WORKING_DIR}/.venv}"

run_python_overlay() {
  if [ -z "${OPENPI_PYTHON_OVERLAY_HOOK:-}" ]; then
    return 0
  fi
  if [ ! -x "${OPENPI_PYTHON_OVERLAY_HOOK}" ]; then
    echo "[ERROR]: OPENPI_PYTHON_OVERLAY_HOOK is set but not executable: ${OPENPI_PYTHON_OVERLAY_HOOK}"
    exit 1
  fi

  echo "[INFO]: Applying python overlay via ${OPENPI_PYTHON_OVERLAY_HOOK}"
  "${OPENPI_PYTHON_OVERLAY_HOOK}" "${WORKING_DIR}"
}

apply_python_overlay_env() {
  local env_file="${VENV_DIR}/.gb200_overlay_env.sh"
  if [ -f "${env_file}" ]; then
    echo "[INFO]: Sourcing python overlay env from ${env_file}"
    # shellcheck disable=SC1090
    . "${env_file}"
  fi
}

require_venv_python() {
  VENV_PYTHON="${VENV_DIR}/bin/python"
  if [ ! -x "${VENV_PYTHON}" ]; then
    echo "[ERROR]: Python executable not found in venv: ${VENV_PYTHON}"
    exit 1
  fi
}

mkdir -p "${CHECKPOINT_DIR}"
openpi_cd_working_dir

sync_args=(--python "${OPENPI_PROJECT_PYTHON}")
arch="$(uname -m)"
if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
  # The RLDS dependency group pulls tensorflow-cpu, which is not available in this repo's lock for Linux aarch64.
  sync_args+=(--no-group rlds)
fi

echo "[INFO]: Starting uv sync..."
echo "[INFO]: uv sync args: ${sync_args[*]}"
GIT_LFS_SKIP_SMUDGE=1 uv sync "${sync_args[@]}"
run_python_overlay
apply_python_overlay_env
require_venv_python

YAML_DATASET_DATA_DIR=""
YAML_DATASET_REPO_ID=""
YAML_DATASET_ASSETS_DIR=""
YAML_DATASET_ASSET_ID=""

if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  if [ ! -f "${OPENPI_CONFIG_YAML}" ] && [ -f "${SCRIPT_DIR}/${OPENPI_CONFIG_YAML}" ]; then
    OPENPI_CONFIG_YAML="${SCRIPT_DIR}/${OPENPI_CONFIG_YAML}"
  fi
  if [ ! -f "${OPENPI_CONFIG_YAML}" ]; then
    echo "[ERROR]: OPENPI_CONFIG_YAML not found: ${OPENPI_CONFIG_YAML}"
    exit 1
  fi

  yaml_meta_lines=$("${VENV_PYTHON}" - "${OPENPI_CONFIG_YAML}" <<'PY'
import pathlib
import sys
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_with_inheritance(yaml_path: pathlib.Path, seen: set[pathlib.Path] | None = None) -> dict:
    yaml_path = yaml_path.resolve()
    if seen is None:
        seen = set()
    if yaml_path in seen:
        raise ValueError(f"Cyclic _base_ reference detected: {yaml_path}")
    seen.add(yaml_path)

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")

    base_ref = data.pop("_base_", None)
    if not base_ref:
        return data

    base_path = pathlib.Path(base_ref)
    if not base_path.is_absolute():
        base_path = (yaml_path.parent / base_path).resolve()
    base_data = load_yaml_with_inheritance(base_path, seen)
    return deep_merge(base_data, data)


yaml_path = pathlib.Path(sys.argv[1])
data = load_yaml_with_inheritance(yaml_path)
dataset = data.get("dataset") or {}

print(f"DATASET_DATA_DIR={dataset.get('data_dir') or ''}")
print(f"DATASET_REPO_ID={dataset.get('repo_id') or ''}")
print(f"DATASET_ASSETS_DIR={dataset.get('assets_dir') or ''}")
print(f"DATASET_ASSET_ID={dataset.get('asset_id') or ''}")
PY
)

  while IFS='=' read -r key value; do
    case "${key}" in
      DATASET_DATA_DIR) YAML_DATASET_DATA_DIR="${value}" ;;
      DATASET_REPO_ID) YAML_DATASET_REPO_ID="${value}" ;;
      DATASET_ASSETS_DIR) YAML_DATASET_ASSETS_DIR="${value}" ;;
      DATASET_ASSET_ID) YAML_DATASET_ASSET_ID="${value}" ;;
    esac
  done <<EOF
${yaml_meta_lines}
EOF

  if [ -n "${YAML_DATASET_DATA_DIR}" ]; then
    DATASET_NAME="$(basename "${YAML_DATASET_DATA_DIR}")"
    DATA_DIR="$(dirname "${YAML_DATASET_DATA_DIR}")"
  elif [ -z "${DATASET_NAME}" ] && [ -n "${YAML_DATASET_REPO_ID}" ]; then
    DATASET_NAME="$(basename "${YAML_DATASET_REPO_ID}")"
  fi

  if [ -n "${CONFIG_NAME}" ]; then
    TRAIN_CONFIG_NAME="${CONFIG_NAME}"
  else
    TRAIN_CONFIG_NAME="$(basename "${OPENPI_CONFIG_YAML%.*}")"
  fi
else
  TRAIN_CONFIG_NAME="${CONFIG_NAME}"
  if [ -z "${DATASET_NAME}" ]; then
    DATASET_NAME="${TRAIN_CONFIG_NAME}"
  fi
fi

if [ -z "${DATASET_NAME}" ]; then
  echo "[ERROR]: DATASET_NAME could not be resolved. Set DATASET_NAME or dataset.data_dir/repo_id."
  exit 1
fi

export DATA_DIR DATASET_NAME CHECKPOINT_DIR OPENPI_CONFIG_YAML

PARENT_DATA_DIR="$(basename "${DATA_DIR}")"
EPISODES_STATS_FILE="${DATA_DIR}/${DATASET_NAME}/meta/episodes_stats.jsonl"
if [ -n "${YAML_DATASET_ASSETS_DIR}" ]; then
  NORM_STATS_ASSET_ID="${YAML_DATASET_ASSET_ID:-${YAML_DATASET_REPO_ID:-${PARENT_DATA_DIR}/${DATASET_NAME}}}"
  NORM_STATS_OUTPUT_FILE="${YAML_DATASET_ASSETS_DIR%/}/${NORM_STATS_ASSET_ID}/norm_stats.json"
else
  NORM_STATS_OUTPUT_FILE="assets/${TRAIN_CONFIG_NAME}/${PARENT_DATA_DIR}/${DATASET_NAME}/norm_stats.json"
fi
DATA_CHUNK_DIR="${DATA_DIR}/${DATASET_NAME}/data/"

echo "[INFO]: Config ownership:"
echo "[INFO]:   YAML (experiment semantics): ${OPENPI_CONFIG_YAML:-<none>}"
echo "[INFO]:   .env (runtime): OPENPI_RESUME=${OPENPI_RESUME}, OPENPI_OVERWRITE=${OPENPI_OVERWRITE}, OPENPI_AGGREGATE_STATS_ENABLE=${OPENPI_AGGREGATE_STATS_ENABLE}"
echo "[INFO]: Resolved dataset root: ${DATA_DIR}/${DATASET_NAME}"
echo "[INFO]: Norm stats output: ${NORM_STATS_OUTPUT_FILE}"

if [ "${OPENPI_AGGREGATE_STATS_ENABLE}" = "1" ]; then
  if [ -f "${EPISODES_STATS_FILE}" ]; then
    mkdir -p "$(dirname "${NORM_STATS_OUTPUT_FILE}")"
    echo "[INFO]: Running aggregate_stats_simple.py..."
    JAX_PLATFORMS=cpu "${VENV_PYTHON}" scripts/aggregate_stats_simple.py \
      --episodes-stats "${EPISODES_STATS_FILE}" \
      --output-file "${NORM_STATS_OUTPUT_FILE}" \
      --chunk-dir "${DATA_CHUNK_DIR}" \
      --action-column "action.relative" \
      --action-mode "relative"
  else
    echo "[WARN]: aggregate stats skipped; episodes stats file not found: ${EPISODES_STATS_FILE}"
  fi
else
  echo "[INFO]: aggregate stats is disabled"
fi

train_args=()
if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  train_args+=(--config-yaml "${OPENPI_CONFIG_YAML}")
else
  train_args+=("${CONFIG_NAME}")
fi
train_args+=(--exp-name "${EXP_NAME}")
train_args+=(--checkpoint-base-dir "${CHECKPOINT_DIR}")
if [ "${OPENPI_RESUME}" = "1" ]; then
  train_args+=(--resume)
elif [ "${OPENPI_OVERWRITE}" = "1" ]; then
  train_args+=(--overwrite)
fi
if [ "$#" -gt 0 ]; then
  train_args+=("$@")
fi

echo "[INFO]: Running train.py..."
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  "${VENV_PYTHON}" scripts/train.py "${train_args[@]}"

echo "[INFO]: Done!"
