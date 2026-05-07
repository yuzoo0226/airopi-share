#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_INIT_SCRIPT="/home/docker_scripts/initialize-ros-env.sh"
BASHRC_PATH="/root/.bashrc"
CACHE_ROOT="${OPENPI_CACHE_ROOT:-/home/cache}"
OPENPI_DOCKER_MODE="${OPENPI_DOCKER_MODE:-train}"
OPENPI_ENABLE_ROS_INIT="${OPENPI_ENABLE_ROS_INIT:-auto}"
OPENPI_ENABLE_UV_WARMUP="${OPENPI_ENABLE_UV_WARMUP:-1}"

ensure_bashrc_line() {
  local line="$1"
  touch "${BASHRC_PATH}"
  if ! grep -Fqx "${line}" "${BASHRC_PATH}"; then
    printf '%s\n' "${line}" >> "${BASHRC_PATH}"
  fi
}

prepare_cache_dirs() {
  mkdir -p     "${UV_CACHE_DIR:-${CACHE_ROOT}/uv-cache}"     "${UV_DATA_DIR:-${CACHE_ROOT}/uv-data}"     "${UV_PYTHON_DIR:-${CACHE_ROOT}/uv-python}"     "${OPENPI_DATA_HOME:-${CACHE_ROOT}/openpi_cache}"     "${HF_HOME:-${CACHE_ROOT}/huggingface_home}"     "${TMPDIR:-${CACHE_ROOT}/tmp}"     "${XDG_CACHE_HOME:-${CACHE_ROOT}/cache}"     "${CACHE_ROOT}"
}

resolve_ros_init_enabled() {
  case "${OPENPI_ENABLE_ROS_INIT}" in
    1|true|TRUE|yes|YES)
      echo 1
      ;;
    0|false|FALSE|no|NO)
      echo 0
      ;;
    auto)
      if [ "${OPENPI_DOCKER_MODE}" = "deploy" ]; then
        echo 1
      else
        echo 0
      fi
      ;;
    *)
      echo "[INIT] Warning: unknown OPENPI_ENABLE_ROS_INIT=${OPENPI_ENABLE_ROS_INIT}; defaulting to disabled."
      echo 0
      ;;
  esac
}

prepare_shell_env() {
  local venv_dir="${UV_PROJECT_ENVIRONMENT:-/home/openpi/.venv}"
  ensure_bashrc_line "if [ -d ${venv_dir}/bin ]; then export PATH=\"${venv_dir}/bin:\$PATH\"; fi"
  ensure_bashrc_line "if [ -f ${venv_dir}/.gb200_overlay_env.sh ]; then source ${venv_dir}/.gb200_overlay_env.sh; fi"
  if [ "$1" = "1" ]; then
    ensure_bashrc_line 'if [ -f /home/docker_scripts/initialize-ros-env.sh ]; then source /home/docker_scripts/initialize-ros-env.sh; fi'
  fi
}

start_system_services() {
  if [ "$1" != "1" ]; then
    return
  fi

  if command -v service >/dev/null 2>&1 && [ -x /etc/init.d/dbus ]; then
    echo "[INIT] Starting dbus service (if not already running)..."
    service dbus start || true
  fi

  if command -v service >/dev/null 2>&1 && [ -x /etc/init.d/avahi-daemon ]; then
    echo "[INIT] Starting avahi-daemon (if not already running)..."
    service avahi-daemon start || true
  fi
}

apply_roslogging_fix() {
  if [ "$1" != "1" ]; then
    return
  fi

  if [ -d "/opt/ros/noetic" ]; then
    echo "[INIT] Applying rosgraph/roslogging.py fix with mode 0644..."
    bash "${SCRIPT_DIR}/fix_roslogging.sh" || echo "[INIT] Warning: fix_roslogging.sh failed; continuing."
  fi
}

warm_uv_environment() {
  if [ "${OPENPI_ENABLE_UV_WARMUP}" = "0" ] || [ ! -d "/home/openpi" ]; then
    return
  fi

  cd /home/openpi || return
  local sync_marker="${CACHE_ROOT}/.uv_synced_${OPENPI_DOCKER_MODE}"
  if [ -f "uv.lock" ] && [ ! -f "${sync_marker}" ]; then
    local sync_args=(--python "${OPENPI_PROJECT_PYTHON:-3.11}")
    local arch
    arch="$(uname -m)"
    if [ "${arch}" = "aarch64" ] || [ "${arch}" = "arm64" ]; then
      sync_args+=(--no-group rlds)
    fi
    echo "[INIT] Performing initial 'uv sync' to warm caches..."
    echo "[INIT] uv sync args: ${sync_args[*]}"
    GIT_LFS_SKIP_SMUDGE=1 uv sync "${sync_args[@]}" && touch "${sync_marker}" || true
  fi
}

apply_python_overlay() {
  if [ -z "${OPENPI_PYTHON_OVERLAY_HOOK:-}" ]; then
    return
  fi

  if [ -x "${OPENPI_PYTHON_OVERLAY_HOOK}" ]; then
    echo "[INIT] Applying python overlay: ${OPENPI_PYTHON_OVERLAY_HOOK}"
    "${OPENPI_PYTHON_OVERLAY_HOOK}" /home/openpi || echo "[INIT] Warning: python overlay failed; continuing."
  else
    echo "[INIT] Warning: OPENPI_PYTHON_OVERLAY_HOOK is set but not executable: ${OPENPI_PYTHON_OVERLAY_HOOK}"
  fi
}

run_ros_init() {
  if [ "$1" != "1" ]; then
    return
  fi

  if [ -f "${ROS_INIT_SCRIPT}" ]; then
    echo "[INIT] Initializing ROS environment from ${ROS_INIT_SCRIPT}"
    # shellcheck source=/dev/null
    source "${ROS_INIT_SCRIPT}" || echo "[INIT] Warning: ROS initialization failed; continuing."
  else
    echo "[INIT] Warning: ROS init script not found at ${ROS_INIT_SCRIPT}. Skipping ROS initialization."
  fi
}

ros_init_enabled="$(resolve_ros_init_enabled)"
prepare_cache_dirs
prepare_shell_env "${ros_init_enabled}"
start_system_services "${ros_init_enabled}"
apply_roslogging_fix "${ros_init_enabled}"
warm_uv_environment
apply_python_overlay
run_ros_init "${ros_init_enabled}"

tail -f /dev/null
