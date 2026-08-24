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
# COLCON_IGNOREd packages are skipped: they are not built, and some of them are
# ROS 1 packages (e.g. ros_gz/ros_gz_point_cloud) whose keys rosdep cannot
# resolve under humble.
python3 - "${WS_SRC}" "${HERE}/manifests" <<'PYEOF'
import pathlib
import shutil
import sys

src = pathlib.Path(sys.argv[1]).resolve()
dst = pathlib.Path(sys.argv[2]).resolve()
ignored = {p.parent for p in src.rglob("COLCON_IGNORE")}


def is_ignored(path: pathlib.Path) -> bool:
    return any(parent in ignored for parent in path.parents)


count = 0
for manifest in src.rglob("package.xml"):
    if ".git" in manifest.parts or is_ignored(manifest):
        continue
    target = dst / manifest.parent.relative_to(src) / "package.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, target)
    count += 1
print(f"[INFO] {count} manifests collected")
PYEOF

docker build \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -f "${HERE}/Dockerfile.sim" \
  -t hsr-ros2-sim:humble \
  "${HERE}"

echo "[INFO] built image hsr-ros2-sim:humble"
