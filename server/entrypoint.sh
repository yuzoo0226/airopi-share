#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"

HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"
CONFIG_NAME="${POLICY_CONFIG_NAME:-}"
CONFIG_YAML="${POLICY_CONFIG_YAML:-}"

if [[ -z "${CONFIG_NAME}" && -z "${CONFIG_YAML}" ]]; then
  echo "[ERROR] Either POLICY_CONFIG_NAME or POLICY_CONFIG_YAML must be set." >&2
  exit 1
fi

ARGS=(
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

if [[ -n "${CONFIG_NAME}" ]]; then
  ARGS+=("--config-name" "${CONFIG_NAME}")
fi

if [[ -n "${CONFIG_YAML}" ]]; then
  ARGS+=("--config-yaml" "${CONFIG_YAML}")
fi

if [[ -n "${POLICY_DEFAULT_PROMPT:-}" ]]; then
  ARGS+=("--default-prompt" "${POLICY_DEFAULT_PROMPT}")
fi

if [[ -n "${POLICY_RECORD_DIR:-}" ]]; then
  ARGS+=("--record-dir" "${POLICY_RECORD_DIR}")
fi

if [[ -n "${POLICY_PYTORCH_DEVICE:-}" ]]; then
  ARGS+=("--pytorch-device" "${POLICY_PYTORCH_DEVICE}")
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
