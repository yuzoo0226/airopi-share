#!/usr/bin/env bash
# Stop everything the simulator / inference stack started inside the container.
#
#   ./stop-sim.sh            # stop the simulator and the openpi node
#   ./stop-sim.sh --node     # stop only the openpi inference node
#
# The bracket trick ("laun[c]h") keeps the patterns from matching the shell that
# runs pkill itself.
set -uo pipefail

CONTAINER="${HSR_SIM_CONTAINER:-hsr-ros2-sim}"
MODE="${1:-all}"

if [[ "${MODE}" == "--node" ]]; then
    # only the openpi inference node
    PATTERNS=('hsr_openpi_nod[e]' 'hsr_openpi\.launc[h]')
elif [[ "${MODE}" == "--collect" ]]; then
    # data collection / evaluation only - leave the simulator and, crucially,
    # its own ros_gz parameter_bridge running. That bridge carries /clock, so
    # killing every "parameter_bridge" process freezes simulation time for every
    # node with use_sim_time and everything silently stops.
    PATTERNS=(
        'pick_tas[k]'
        'random_motio[n]'
        'rosbag2_recorde[r]'
        'ros2 ba[g] record'
        'collect_data\.launc[h]'
        'eval_pick\.launc[h]'
        'republis[h]'
        'dynamic_pos[e]'
    )
else
    # Everything. Collection/evaluation nodes MUST be in this list: a leftover
    # pick_task keeps publishing cmd_vel and spawning objects, and a cmd_vel
    # arriving while omni_base_controller is still being activated segfaults the
    # freshly started simulator (hsrb_base_controllers::OmniBaseController, exit
    # code 139).
    PATTERNS=(
        'hsr_openpi_nod[e]'
        'pick_tas[k]'
        'random_motio[n]'
        'bag_recorde[r]'
        'rosbag2_recorde[r]'
        'ros2 ba[g] record'
        'republis[h]'
        'ros2 laun[c]h'
        'ig[n] gazebo'
        'robot_state_publishe[r]'
        'parameter_bridg[e]'
        'joint_state_publishe[r]'
        'odometry_switche[r]'
        'component_containe[r]'
        'topic_tool[s]'
        'controller_manager/spawne[r]'
    )
fi

for p in "${PATTERNS[@]}"; do
    docker exec "${CONTAINER}" pkill -9 -f "${p}" >/dev/null 2>&1 || true
done
sleep 2
echo "[INFO] remaining:"
docker exec "${CONTAINER}" bash -c "ps -eo pid,comm | grep -aE 'ign|hsr_openpi|ros2' | grep -v grep || echo '  (none)'"
