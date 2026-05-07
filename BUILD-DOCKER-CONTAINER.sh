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
openpi_prepare_docker_paths
openpi_require_deploy_build_credentials
openpi_log_docker_context "$0"

EXISTING_CONTAINER_ID="$(docker ps -aq -f name="${CONTAINER}")"
if [ -n "${EXISTING_CONTAINER_ID}" ]; then
  echo "The container name ${CONTAINER} is already in use" 1>&2
  echo "${EXISTING_CONTAINER_ID}"
  exit 1
fi

echo "starting build"
docker compose -p "${PROJECT}" -f ./docker/docker-compose.yml build
