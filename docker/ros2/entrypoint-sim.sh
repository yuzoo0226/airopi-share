#!/usr/bin/env bash
# Entry point for the HSR ROS 2 simulator container.
set -e

source /opt/ros/humble/setup.bash
WS=/home/hsr/hsr_ros2_ws
if [ -f "${WS}/install/setup.bash" ]; then
    source "${WS}/install/setup.bash"
fi

# Ignition needs a writable home for its fuel cache / logs.
export IGN_GAZEBO_RESOURCE_PATH="${IGN_GAZEBO_RESOURCE_PATH:-}"
export HOME=/home/hsr

exec "$@"
