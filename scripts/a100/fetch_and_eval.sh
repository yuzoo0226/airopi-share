#!/usr/bin/env bash
# =============================================================================
#  Bring one checkpoint back from the A100 cluster and score it in Gazebo.
#
#      scripts/a100/fetch_and_eval.sh <step> [episodes]
#      scripts/a100/fetch_and_eval.sh 10000 20
#
#  Training runs on the cluster; the simulator only exists on this machine, so
#  every evaluation needs the checkpoint copied back. ~9.3 GB per checkpoint.
#
#  The copy lands in the directory the policy server mounts, which is *not* the
#  one training writes to -- see configs/experiments/example_hsr_pick_gazebo.yaml
#  for why those differ and what happens when they are confused.
# =============================================================================
set -uo pipefail

STEP="${1:?usage: fetch_and_eval.sh <step> [episodes]}"
EPISODES="${2:-20}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

SSH_TARGET="${A100_SSH:-yano21@150.69.197.6}"
EXP="${EXP_NAME:-pick_a100}"
REMOTE="\${HOME}/usr/airopi/checkpoints/_train/example_hsr_pick_gazebo/${EXP}/${STEP}"
LOCAL_ROOT="${CHECKPOINT_ROOT:-/home/tamhome/usr/airoa_ws/airopi_ws/checkpoints/_train/example_hsr_pick_gazebo/${EXP}}"
# The path as the policy server container sees it.
SERVER_PATH="/home/openpi/checkpoints/_train/example_hsr_pick_gazebo/${EXP}/${STEP}"

echo "[1/3] checking the checkpoint exists on the cluster"
if ! ssh -o BatchMode=yes "${SSH_TARGET}" "test -d ${REMOTE} && test -d ${REMOTE}/params" 2>/dev/null; then
    echo "[ERROR] ${EXP}/${STEP} is not on the cluster yet."
    ssh -o BatchMode=yes "${SSH_TARGET}" \
        "ls \${HOME}/usr/airopi/checkpoints/_train/example_hsr_pick_gazebo/${EXP}/ 2>/dev/null | sort -n | tr '\\n' ' '" 2>/dev/null
    echo
    exit 1
fi

echo "[2/3] copying it back (~9 GB)"
mkdir -p "${LOCAL_ROOT}"
# Trailing slash on neither side: rsync then creates ${LOCAL_ROOT}/${STEP}.
rsync -a --info=progress2 --partial \
    "${SSH_TARGET}:${REMOTE}" "${LOCAL_ROOT}/" || { echo "[ERROR] copy failed"; exit 1; }

# A checkpoint without its assets loads, and then normalises with whatever the
# server happens to have -- silently wrong rather than an error.
[ -d "${LOCAL_ROOT}/${STEP}/assets" ] || { echo "[ERROR] no assets/ in the copied checkpoint"; exit 1; }
du -sh "${LOCAL_ROOT}/${STEP}"

echo "[3/3] evaluating ${EPISODES} episodes"
exec "${REPO_ROOT}/scripts/ros2/run_pick_eval.sh" "${SERVER_PATH}" "${EXP}_${STEP}" "${EPISODES}"
