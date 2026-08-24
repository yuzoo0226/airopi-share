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
NODE_ONLY=0
[[ "${1:-}" == "--node" ]] && NODE_ONLY=1

if [[ "${NODE_ONLY}" == "1" ]]; then
    PATTERNS=('hsr_openpi_nod[e]' 'hsr_openpi\.launc[h]')
else
    PATTERNS=(
        'hsr_openpi_nod[e]'
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
