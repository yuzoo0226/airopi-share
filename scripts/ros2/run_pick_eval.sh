#!/usr/bin/env bash
# =============================================================================
#  Evaluate one checkpoint on the Gazebo pick task.
#
#      scripts/ros2/run_pick_eval.sh <step_dir> <tag> [episodes]
#
#  e.g. scripts/ros2/run_pick_eval.sh \
#           /home/openpi/checkpoints/_train/example_hsr_pick_gazebo/pick_v1/1151 \
#           ep02 20
#
#  Training and the simulator cannot share this GPU: Gazebo fails to bring its
#  controllers up while a training job is running. Stop training first
#  (and resume it afterwards with scripts/train.py --resume).
#
#  Steps:
#    1. (re)start the simulator in the pick world
#    2. point the policy server at <step_dir>
#    3. run eval_pick.launch.py, which resets the scene, spawns a random object
#       and scores each episode while hsr_openpi_node drives
#    4. render an mp4 of the first few episodes
# =============================================================================
set -euo pipefail

STEP_DIR="${1:?usage: run_pick_eval.sh <step_dir> <tag> [episodes]}"
TAG="${2:?usage: run_pick_eval.sh <step_dir> <tag> [episodes]}"
EPISODES="${3:-20}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
COMPOSE_DIR="${REPO_ROOT}/docker/ros2"

SIM_CONTAINER="${HSR_SIM_CONTAINER:-hsr-ros2-sim}"
TRAIN_CONTAINER="${TRAIN_CONTAINER:-airopi_ros2_deep_1}"
WORLD="${PICK_WORLD:-pick_table}"
BAG_DIR="${BAG_DIR:-/home/hsr/hsr_ros2_ws/_bags}"
VIDEO_DIR="${VIDEO_DIR:-/home/datasets/_videos}"
SEED="${EVAL_SEED:-1000}"

sim() { docker exec "${SIM_CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash; source /home/hsr/hsr_ros2_ws/install/setup.bash; $1"; }

echo "[1/4] simulator"
"${COMPOSE_DIR}/stop-sim.sh" >/dev/null 2>&1 || true
sleep 2
docker exec -d "${SIM_CONTAINER}" bash -lc "
    source /opt/ros/humble/setup.bash
    source /home/hsr/hsr_ros2_ws/install/setup.bash
    exec ros2 launch hsr_openpi hsr_sim.launch.py world:=${WORLD} > /tmp/gz.log 2>&1"
for _ in $(seq 1 60); do
    n=$(docker exec "${SIM_CONTAINER}" bash -lc 'grep -ac "Configured and activated" /tmp/gz.log 2>/dev/null || echo 0')
    [ "${n}" -ge 6 ] && break
    sleep 5
done
[ "${n:-0}" -ge 6 ] || { echo "[ERROR] the simulator did not come up (see /tmp/gz.log)"; exit 1; }
echo "      controllers active"

echo "[2/4] policy server -> ${STEP_DIR}"
( cd "${COMPOSE_DIR}" && \
  POLICY_CHECKPOINT_DIR="${STEP_DIR}" \
  POLICY_DEFAULT_PROMPT="pick up the object" \
  HOST_UID="$(id -u)" HOST_GID="$(id -g)" \
  docker compose up -d --force-recreate openpi-server >/dev/null )
for _ in $(seq 1 120); do
    docker logs "${OPENPI_SERVER_CONTAINER:-airopi-policy-server}" 2>&1 | grep -q "server listening" && break
    sleep 5
done
docker logs "${OPENPI_SERVER_CONTAINER:-airopi-policy-server}" 2>&1 | grep -q "server listening" \
    || { echo "[ERROR] the policy server did not start"; exit 1; }
echo "      serving"

echo "[3/4] evaluation: ${EPISODES} episodes"
sim "rm -rf ${BAG_DIR}/eval_${TAG} ${BAG_DIR}/eval_${TAG}.json"
docker exec -d "${SIM_CONTAINER}" bash -lc "
    source /opt/ros/humble/setup.bash
    source /home/hsr/hsr_ros2_ws/install/setup.bash
    exec ros2 launch hsr_openpi eval_pick.launch.py \
        num_episodes:=${EPISODES} seed:=${SEED} \
        bag_path:=${BAG_DIR}/eval_${TAG} \
        results_path:=${BAG_DIR}/eval_${TAG}.json > /tmp/eval_${TAG}.log 2>&1"

while ! docker exec "${SIM_CONTAINER}" bash -lc "test -f ${BAG_DIR}/eval_${TAG}/metadata.yaml" 2>/dev/null; do
    sleep 15
done
sim "grep -aE 'success=' /tmp/eval_${TAG}.log | tail -3" || true
docker exec "${SIM_CONTAINER}" bash -lc "cat ${BAG_DIR}/eval_${TAG}.json" \
    | python3 "${HERE}/print_eval_result.py" "${TAG}" || true

echo "[4/4] video"
# The simulator writes bags to ${BAG_DIR}; docker/ros2/docker-compose.train.yml
# mounts that same host directory at /home/bags inside the training container.
TRAIN_BAG_DIR="${TRAIN_BAG_DIR:-/home/bags}"
docker exec "${TRAIN_CONTAINER}" bash -lc "
    cd /home/openpi && /home/cache/venv/bin/python deploy/hsr_openpi_ros2/tools/bag2video.py \
        --bag ${TRAIN_BAG_DIR}/eval_${TAG} \
        --out ${VIDEO_DIR}/02_inference_${TAG}.mp4 \
        --max-episodes 4 --fps 15 --title 'policy inference (${TAG})'" || \
    echo "[WARN] video rendering failed (is the bag directory visible to ${TRAIN_CONTAINER}?)"

echo "[done] results: ${BAG_DIR}/eval_${TAG}.json"
