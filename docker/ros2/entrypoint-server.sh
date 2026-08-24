#!/usr/bin/env bash
# Entry point of the openpi policy server container.
#
#   serve   (default) start the websocket policy server
#   bash    drop into a shell
set -euo pipefail

if [[ "${1:-serve}" != "serve" ]]; then
    exec "$@"
fi

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"

HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"
CONFIG_NAME="${POLICY_CONFIG_NAME:-}"
CONFIG_YAML="${POLICY_CONFIG_YAML:-}"

ARGS=(--checkpoint-dir "${POLICY_CHECKPOINT_DIR}" --host "${HOST}" --port "${PORT}")

# A checkpoint produced by scripts/train.py embeds the merged experiment YAML at
# <step>/experiment_config/experiment_config.yaml; serve_hsr_policy_ws.py finds
# it on its own, so neither variable has to be set for released checkpoints.
if [[ -n "${CONFIG_NAME}" ]]; then
    ARGS+=(--config-name "${CONFIG_NAME}")
fi
if [[ -n "${CONFIG_YAML}" ]]; then
    ARGS+=(--config-yaml "${CONFIG_YAML}")
fi
if [[ -n "${POLICY_DEFAULT_PROMPT:-}" ]]; then
    ARGS+=(--default-prompt "${POLICY_DEFAULT_PROMPT}")
fi
if [[ -n "${POLICY_RECORD_DIR:-}" ]]; then
    ARGS+=(--record-dir "${POLICY_RECORD_DIR}")
fi
if [[ -n "${POLICY_PYTORCH_DEVICE:-}" ]]; then
    ARGS+=(--pytorch-device "${POLICY_PYTORCH_DEVICE}")
fi

echo "[INFO] python: $(python --version)"
echo "[INFO] serve_hsr_policy_ws.py ${ARGS[*]}"
exec python /home/openpi/server/serve_hsr_policy_ws.py "${ARGS[@]}"
