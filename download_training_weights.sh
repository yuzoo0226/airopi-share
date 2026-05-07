#!/bin/bash
# Training用の重み/トークナイザを事前にダウンロードするスクリプト
# 計算ノードではなくログインノードで実行してください

set -e

echo "[INFO]: Training weights download script started at $(date)"

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

CONFIG_NAME="${1:-${CONFIG_NAME:-}}"
if [ -z "${CONFIG_NAME}" ]; then
  echo "[ERROR]: CONFIG_NAME is not set. Usage: $0 <config_name>"
  exit 1
fi

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
echo "[INFO]: CONFIG_NAME=${CONFIG_NAME}"

cd "${WORKING_DIR}" || exit 1

echo "[INFO]: Starting uv sync..."
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "[INFO]: Downloading training weights/assets for ${CONFIG_NAME}..."
uv run python - "$CONFIG_NAME" <<'PY'
import dataclasses
import sys
import urllib.parse
from typing import Iterable

import openpi.models.model as model
import openpi.shared.download as download
import openpi.training.config as config


def iter_urls(obj, *, seen: set[int]) -> Iterable[str]:
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, str):
        if obj.startswith(("gs://", "s3://")):
            yield obj
        return

    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            yield from iter_urls(getattr(obj, field.name), seen=seen)
        return

    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_urls(value, seen=seen)
        return

    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from iter_urls(value, seen=seen)
        return


config_name = sys.argv[1] if len(sys.argv) > 1 else None
if not config_name:
    raise SystemExit("CONFIG_NAME is not set.")

train_config = config.get_config(config_name)
urls = sorted(set(iter_urls(train_config, seen=set())))

if train_config.model.model_type in {model.ModelType.PI0, model.ModelType.PI05}:
    urls.append("gs://big_vision/paligemma_tokenizer.model")

if not urls:
    print("[WARN]: No remote URLs found in config.")
else:
    for url in urls:
        print(f"[INFO]: Downloading {url}")
        parsed = urllib.parse.urlparse(url)
        kwargs = {"gs": {"token": "anon"}} if parsed.scheme == "gs" else {}
        path = download.maybe_download(url, **kwargs)
        print(f"[INFO]: Cached at {path}")

print("[INFO]: Done.")
PY

echo "[INFO]: Download completed at $(date)"
echo "[INFO]: Weights are cached in ${OPENPI_DATA_HOME}"
