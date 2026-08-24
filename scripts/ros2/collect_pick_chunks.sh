#!/usr/bin/env bash
# =============================================================================
#  Collect scripted pick demonstrations in chunks.
#
#      scripts/ros2/collect_pick_chunks.sh <name> <chunks> <episodes_per_chunk> [first_seed]
#
#  e.g. scripts/ros2/collect_pick_chunks.sh pick_b 3 100 100
#       -> _bags/pick_b_00 .. pick_b_02, 100 episodes each, seeds 100..102
#
#  Why chunks: Ignition aborts every so often during a long run
#  (dart::collision::OdeCollisionDetector -> dDebug -> abort, exit 134). A crash
#  costs one chunk instead of the whole session, and the simulator is restarted
#  fresh for the next one. Bags whose recorder did not get a clean SIGINT are
#  repaired here as well, since `rosbags` refuses a file with no end magic.
# =============================================================================
set -uo pipefail

NAME="${1:?usage: collect_pick_chunks.sh <name> <chunks> <episodes_per_chunk> [first_seed]}"
CHUNKS="${2:?}"
EPISODES="${3:?}"
FIRST_SEED="${4:-100}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
COMPOSE_DIR="${REPO_ROOT}/docker/ros2"

SIM_CONTAINER="${HSR_SIM_CONTAINER:-hsr-ros2-sim}"
WORLD="${PICK_WORLD:-pick_table}"
BAG_DIR="${BAG_DIR:-/home/hsr/hsr_ros2_ws/_bags}"
APPROACH_TIMEOUT="${APPROACH_TIMEOUT:-12.0}"

sim_exec() { docker exec "${SIM_CONTAINER}" bash -lc "$1"; }

start_sim() {
    "${COMPOSE_DIR}/stop-sim.sh" >/dev/null 2>&1 || true
    sleep 3
    # Truncate the log *before* starting: otherwise the readiness grep below can
    # match the previous run's lines before the redirect truncates the file, and
    # report the simulator as up when it has not started yet.
    sim_exec ": > /tmp/gz.log"
    docker exec -d "${SIM_CONTAINER}" bash -lc "
        source /opt/ros/humble/setup.bash
        source /home/hsr/hsr_ros2_ws/install/setup.bash
        exec ros2 launch hsr_openpi hsr_sim.launch.py world:=${WORLD} > /tmp/gz.log 2>&1"
    for _ in $(seq 1 72); do
        sleep 5
        local n
        n=$(sim_exec 'grep -ac "Configured and activated" /tmp/gz.log 2>/dev/null; true' | head -1)
        if [ "${n:-0}" -ge 6 ] && sim_alive; then
            sleep 3
            return 0
        fi
    done
    return 1
}

sim_alive() {
    local n
    n=$(sim_exec 'ps -eo cmd | grep -ac "ign gazeb[o]"; true' | head -1)
    [ "${n:-0}" -ge 1 ]
}

for i in $(seq 0 $((CHUNKS - 1))); do
    TAG=$(printf "%s_%02d" "${NAME}" "${i}")
    SEED=$((FIRST_SEED + i))
    echo "=============================================================="
    echo "[chunk ${i}] ${TAG}  seed=${SEED}  episodes=${EPISODES}"

    if ! start_sim; then
        echo "[ERROR] the simulator did not come up for ${TAG}; skipping"
        continue
    fi

    sim_exec "rm -rf ${BAG_DIR}/${TAG} ${BAG_DIR}/${TAG}_results.json"
    docker exec -d "${SIM_CONTAINER}" bash -lc "
        source /opt/ros/humble/setup.bash
        source /home/hsr/hsr_ros2_ws/install/setup.bash
        exec ros2 launch hsr_openpi collect_data.launch.py driver:=pick \
            bag_path:=${BAG_DIR}/${TAG} num_episodes:=${EPISODES} seed:=${SEED} \
            approach_timeout:=${APPROACH_TIMEOUT} \
            results_path:=${BAG_DIR}/${TAG}_results.json > /tmp/collect_${TAG}.log 2>&1"

    # Finished either when the recorder wrote metadata.yaml, or when the
    # simulator died under it.
    while true; do
        sleep 20
        if sim_exec "test -f ${BAG_DIR}/${TAG}/metadata.yaml"; then
            echo "[chunk ${i}] recorder closed the bag cleanly"
            break
        fi
        if ! sim_alive; then
            echo "[chunk ${i}] the simulator died; stopping this chunk"
            "${COMPOSE_DIR}/stop-sim.sh" >/dev/null 2>&1 || true
            sleep 3
            # rosbags cannot read a bag whose recorder never got SIGINT.
            sim_exec "source /opt/ros/humble/setup.bash
                      source /home/hsr/hsr_ros2_ws/install/setup.bash
                      ros2 bag reindex ${BAG_DIR}/${TAG} -s mcap >/dev/null 2>&1
                      printf 'output_bags:\\n- uri: ${BAG_DIR}/${TAG}_fixed\\n  storage_id: mcap\\n  all: true\\n' > /tmp/conv_${TAG}.yaml
                      ros2 bag convert -i ${BAG_DIR}/${TAG} -o /tmp/conv_${TAG}.yaml >/dev/null 2>&1
                      if [ -f ${BAG_DIR}/${TAG}_fixed/metadata.yaml ]; then
                          rm -rf ${BAG_DIR}/${TAG} && mv ${BAG_DIR}/${TAG}_fixed ${BAG_DIR}/${TAG}
                          echo '[chunk] bag repaired'
                      fi" || true
            break
        fi
    done

    sim_exec "grep -aE 'running [0-9]+/' /tmp/collect_${TAG}.log | tail -1" || true
done

"${COMPOSE_DIR}/stop-sim.sh" >/dev/null 2>&1 || true
echo "=============================================================="
echo "[done] bags:"
sim_exec "ls -d ${BAG_DIR}/${NAME}_* 2>/dev/null" || true
