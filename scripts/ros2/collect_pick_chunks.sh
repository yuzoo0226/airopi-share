#!/usr/bin/env bash
# =============================================================================
#  Collect scripted pick demonstrations in chunks.
#
#      scripts/ros2/collect_pick_chunks.sh <name> <chunks> <episodes_per_chunk> [first_seed]
#
#  e.g. scripts/ros2/collect_pick_chunks.sh pick_b 8 50 100
#       -> _bags/pick_b_00 .. pick_b_07, 50 episodes each, seeds 100..107
#
#  Why chunks: a simulator session degrades as it runs. Measured over five
#  100-episode chunks, the scripted pick succeeds on essentially every episode
#  at first and then falls off a cliff:
#
#      chunk        1st qtr  2nd qtr  3rd qtr  4th qtr
#      pick_d_00     100%     100%     95.7%    69.6%
#      pick_d_03     100%     100%     100%     66.7%
#      pick_e_00     100%     100%     88.0%    64.0%
#
#  The failures are all the same shape: the base stops 14 to 25 cm from the object
#  with the arm and gripper doing exactly the right thing. What causes it is not
#  known. Three things have been ruled out -- machine load (pick_d_00 and
#  pick_d_03 ran with nothing else on the machine and decay like pick_e_00, which
#  ran beside a training job), objects accumulating on the gripper from a failed
#  detach (the world holds six models throughout), and odometry drift (median
#  error against ground truth is 0.0000). Four measurements still do not add up:
#  cmd_vel keeps publishing at a steady 6.2 Hz, commanded speed falls from 0.20
#  to 0.14 m/s as though the servo saw a *smaller* error, odometry is accurate,
#  and yet dxy at the grasp is 0.14 to 0.25 m.
#
#  Do not assume this predicts the abort. pick_e_00 decayed to 59.3% and finished
#  all 100 episodes; pick_e_01 decayed only to 89.3% and aborted at episode 97.
#
#  What does reproduce is where it starts: episode 69 in pick_d_00 and pick_e_00,
#  episode 76 in pick_e_01. So keep chunks short -- 50 episodes stops before the
#  onset, at the cost of one extra simulator restart, about two minutes, per
#  chunk. Expect that to protect the yield, not to make the decay or the aborts
#  go away.
#
#  A crash also costs one chunk instead of the whole session, and bags whose
#  recorder did not get a clean SIGINT are repaired here, since `rosbags`
#  refuses a file with no end magic.
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
