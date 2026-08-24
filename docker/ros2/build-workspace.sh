#!/usr/bin/env bash
# Resolve dependencies and build the HSR ROS 2 workspace inside the running
# hsr-sim container.
#
#   ./build-workspace.sh                 # build everything needed for the sim
#   ./build-workspace.sh --all           # build every package in the workspace
#   ./build-workspace.sh --packages-select hsr_openpi
set -euo pipefail

CONTAINER="${HSR_SIM_CONTAINER:-hsr-ros2-sim}"
WS=/home/hsr/hsr_ros2_ws

# Packages needed for "Gazebo + the openpi client".
DEFAULT_TARGETS=(--packages-up-to hsrb_gazebo_launch hsrb_gripper_fake_interface hsr_openpi)

if [[ "${1:-}" == "--all" ]]; then
    TARGETS=()
    shift
elif [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    TARGETS=("${DEFAULT_TARGETS[@]}")
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "[ERROR] container '${CONTAINER}' is not running. Start it with:" >&2
    echo "          HOST_UID=\$(id -u) HOST_GID=\$(id -g) docker compose up -d hsr-sim" >&2
    exit 1
fi

echo "[INFO] resolving rosdep keys ..."
docker exec "${CONTAINER}" bash -lc '
    set -e
    rosdep update --rosdistro humble >/dev/null 2>&1 || true
    sudo apt-get update -qq
    # hsrb_bringup only exists for the real robot; hsrb_robot_launch is COLCON_IGNOREd.
    rosdep install --from-paths '"${WS}"'/src --ignore-src -y --rosdistro humble --skip-keys "hsrb_bringup"
'

echo "[INFO] colcon build ${TARGETS[*]:-(all packages)}"
# The GB10 box shares its 128 GB between CPU and GPU, so keep the build from
# fanning out too far; raise COLCON_WORKERS / MAKE_JOBS on a roomier machine.
docker exec "${CONTAINER}" bash -lc '
    set -e
    cd '"${WS}"'
    source /opt/ros/humble/setup.bash
    [ -f install/setup.bash ] && source install/setup.bash
    export MAKEFLAGS="-j'"${MAKE_JOBS:-3}"'"
    colcon build --symlink-install \
        --parallel-workers '"${COLCON_WORKERS:-3}"' \
        --cmake-args -DCMAKE_BUILD_TYPE=Release '"${TARGETS[*]:-}"'
'
echo "[INFO] done."
