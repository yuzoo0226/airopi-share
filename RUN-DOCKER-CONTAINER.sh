#!/bin/bash

set -euo pipefail

COMMON_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMON_SCRIPT="${COMMON_SCRIPT_DIR}/RUN-DOCKER-COMMON.sh"
if [ ! -f "${COMMON_SCRIPT}" ]; then
  echo "[ERROR] Common Docker helper not found: ${COMMON_SCRIPT}"
  exit 1
fi
# shellcheck source=/dev/null
. "${COMMON_SCRIPT}"

openpi_require_docker_project
openpi_set_docker_mode "${1:-train}"
openpi_resolve_deploy_network
openpi_prepare_docker_paths
openpi_log_docker_context "$0"

docker compose -p "${PROJECT}" -f ./docker/docker-compose.yml up -d

if command -v xhost >/dev/null 2>&1; then
  xhost + >/dev/null 2>&1 || true
fi

case "${2:-}" in
  "")
    exec_cmd=(docker exec -i -t)
    if [ "${MODE}" = "deploy" ]; then
      exec_cmd+=(
        -e "ROS_IP=${ROS_IP}"
        -e "HSR_IP=${HSR_IP}"
        -e "ROS_MASTER_URI=${ROS_MASTER_URI}"
      )
    fi
    exec_cmd+=(
      "${CONTAINER}"
      bash
      -lc
      "source ~/.bashrc; cd /home/openpi; exec bash"
    )
    "${exec_cmd[@]}"
    ;;
  *)
    echo "Failed to enter the Docker container '${CONTAINER}': '${2}' is not a valid argument value."
    exit 1
    ;;
esac
