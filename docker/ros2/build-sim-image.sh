#!/usr/bin/env bash
# Build the HSR ROS 2 simulator docker image.
#
#   ./build-sim-image.sh [workspace_src_dir]
#
# The package.xml files of the workspace are copied into ./manifests so that
# rosdep can resolve every dependency at image build time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_SRC="${1:-${HERE}/../../../hsr_ros2_ws/src}"
WS_SRC="$(cd "${WS_SRC}" && pwd)"

echo "[INFO] collecting package.xml from ${WS_SRC}"
rm -rf "${HERE}/manifests"
mkdir -p "${HERE}/manifests"
(
  cd "${WS_SRC}"
  find . -name package.xml -not -path '*/.git/*' -print0 \
    | xargs -0 -I{} cp --parents {} "${HERE}/manifests/"
)
echo "[INFO] $(find "${HERE}/manifests" -name package.xml | wc -l) manifests collected"

docker build \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -f "${HERE}/Dockerfile.sim" \
  -t hsr-ros2-sim:humble \
  "${HERE}"

echo "[INFO] built image hsr-ros2-sim:humble"
