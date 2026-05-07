#!/bin/bash

# This script is often sourced from interactive shells, so it must not leak
# shell options like `errexit`/`nounset` into the caller.

# Initialize ROS and catkin workspace; source into current shell if called via 'source'.
# Safe to run multiple times.

ROS_SETUP="/opt/ros/noetic/setup.bash"
WS_ROOT="/root/catkin_ws"
WS_SRC="$WS_ROOT/src"
WS_DEVEL_DIR="$WS_ROOT/devel"
WS_SETUP="$WS_DEVEL_DIR/setup.bash"

source_compat() {
  local target="$1"
  local restore_opts
  local status
  restore_opts="$(set +o)"
  set +eu
  # shellcheck disable=SC1090
  source "$target"
  status=$?
  eval "$restore_opts"
  return "$status"
}

if [ ! -d "/opt/ros/noetic" ]; then
  echo "[ROS-INIT] /opt/ros/noetic not found. Skipping ROS init."
  return 0 2>/dev/null || exit 0
fi

# Ensure ROS_DISTRO is defined to satisfy profile scripts
export ROS_DISTRO=${ROS_DISTRO:-noetic}
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
export ROS_IP=${ROS_IP:-}
export ROS_HOSTNAME=${ROS_HOSTNAME:-}

# If devel already exists, just source and exit
if [ -d "$WS_DEVEL_DIR" ] && [ -f "$WS_SETUP" ]; then
  source_compat "$WS_SETUP" || true
  rospack profile || true
  echo "[ROS-INIT] Using existing workspace: sourced $WS_SETUP"
  return 0 2>/dev/null || exit 0
fi

# Otherwise, build once and then source
source_compat "$ROS_SETUP" || true

mkdir -p "$WS_SRC"
pushd "$WS_ROOT" >/dev/null || true

# Best-effort dependency resolution
rosdep update || true
rosdep install --from-paths src --ignore-src -r -y || true

if command -v catkin >/dev/null 2>&1; then
  catkin build || true
elif command -v catkin_make >/dev/null 2>&1; then
  catkin_make || true
else
  echo "[ROS-INIT] Warning: catkin tools not installed. Skipping build."
fi

if [ -f "$WS_SETUP" ]; then
  source_compat "$WS_SETUP" || true
  rospack profile || true
  echo "[ROS-INIT] Sourced $WS_SETUP"
else
  echo "[ROS-INIT] Warning: $WS_SETUP not found after build."
fi

popd >/dev/null || true
