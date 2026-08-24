#!/usr/bin/env bash
# =============================================================================
#  Create the HSR ROS 2 (Humble) simulator workspace next to this repository.
#
#      <parent>/airopi-share      <- this repository
#      <parent>/hsr_ros2_ws/src   <- created here
#
#  Sources come from the public https://github.com/hsr-project organisation
#  (Toyota Motor Corporation's open source HSR stack, `humble` branches) plus
#  two packages that the ROS buildfarm does not publish for arm64 and therefore
#  have to be built from source: ros_gz (ros_gz_sim) and gz_ros2_control.
#
#  Usage: scripts/ros2/setup_hsr_ros2_ws.sh [workspace_dir]
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
WS="${1:-$(cd "${REPO_ROOT}/.." && pwd)/hsr_ros2_ws}"
SRC="${WS}/src"

# Repository list from hsr-project's official simulation setup guide
# (https://github.com/hsr-project/hsr_ros2_doc/blob/humble/docs/setup_sim_jp.md)
HSR_REPOS=(
  hsrb_controllers hsrb_common hsrb_drivers hsrb_launch hsrb_manipulation
  hsrb_rosnav hsrb_simulator hsr_common hsrb_teleop
  tmc_gazebo tmc_teleop tmc_common tmc_common_msgs tmc_drivers tmc_database
  tmc_manipulation tmc_manipulation_base tmc_manipulation_planner
  tmc_point_cloud tmc_realtime_control tmc_voice tmc_navigation
)

# Packages that only make sense on the real robot / need proprietary SDKs.
# COLCON_IGNORE is used instead of deleting them so the sources stay available.
IGNORE_PKGS=(
  hsrb_launch/hsrb_robot_launch     # real robot bring-up (needs hsrb_bringup)
  tmc_drivers/tmc_pgr_camera        # needs the proprietary PointGrey SDK
  _extern/ros_gz/ros_gz_point_cloud # ROS 1 package shipped inside ros_gz
)

echo "[INFO] workspace: ${WS}"
mkdir -p "${SRC}/_extern"

clone() {  # clone <url> <branch> <dest>
  local url="$1" branch="$2" dest="$3"
  if [ -d "${dest}/.git" ]; then
    echo "[SKIP] ${dest} already exists"
    return 0
  fi
  echo "[GIT ] ${url} (${branch})"
  git clone -q --depth 1 -b "${branch}" "${url}" "${dest}"
}

for r in "${HSR_REPOS[@]}"; do
  clone "https://github.com/hsr-project/${r}.git" humble "${SRC}/${r}"
done

# ros-humble-ros-gz-sim and ros-humble-gz-ros2-control have no arm64 debian
# packages, so build them from source in the same workspace.
clone https://github.com/gazebosim/ros_gz.git humble "${SRC}/_extern/ros_gz"
clone https://github.com/ros-controls/gz_ros2_control.git humble "${SRC}/_extern/gz_ros2_control"

for p in "${IGNORE_PKGS[@]}"; do
  if [ -d "${SRC}/${p}" ]; then
    touch "${SRC}/${p}/COLCON_IGNORE"
    echo "[SKIP] ${p} (COLCON_IGNORE)"
  fi
done

# Expose the ROS 2 openpi packages of this repository to the workspace through a
# *relative* symlink: it resolves both on the host and inside the container,
# where the two trees are mounted as siblings (/home/hsr/{hsr_ros2_ws,airopi-share}).
REL="$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" \
        "${REPO_ROOT}/deploy/hsr_openpi_ros2" "${SRC}")"
ln -sfn "${REL}" "${SRC}/hsr_openpi_ros2"
echo "[LINK] ${SRC}/hsr_openpi_ros2 -> ${REL}"

echo
echo "[INFO] $(find "${SRC}" -name package.xml -not -path '*/.git/*' | wc -l) packages available."
echo "[INFO] next:"
echo "         cd ${REPO_ROOT}/docker/ros2 && ./build-sim-image.sh"
echo "         HOST_UID=\$(id -u) HOST_GID=\$(id -g) docker compose up -d hsr-sim"
echo "         ./build-workspace.sh"
