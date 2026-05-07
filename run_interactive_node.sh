#!/bin/bash
# Interactive node allocation via Slurm (salloc)
#
# Usage:
#   ./run_interactive_node.sh              # defaults: 1 GPU, 1h, partition 2-def
#   ./run_interactive_node.sh -g 4 -t 2:00:00
#   ./run_interactive_node.sh -p 2-def -g 8 -t 4:00:00 -c 56
#   ./run_interactive_node.sh --cpu-only   # connector node, no GPU

set -eu

# --- defaults ---
PARTITION="part-group_8cccb4"
NGPUS=8
CPUS=224
TIME="192:00:00"
MEM="1900G"
CPU_ONLY=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -p, --partition PART   Slurm partition       (default: ${PARTITION})
  -g, --gpus N           Number of GPUs         (default: ${NGPUS})
  -c, --cpus N           CPUs per task          (default: ${CPUS})
  -t, --time HH:MM:SS    Wall time             (default: ${TIME})
  -m, --mem SIZE         Memory (e.g. 128G)     (default: ${MEM})
      --cpu-only         No GPU, use connector nodes
  -h, --help             Show this help
EOF
  exit 0
}

# --- parse args ---
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--partition) PARTITION="$2"; shift 2 ;;
    -g|--gpus)      NGPUS="$2";    shift 2 ;;
    -c|--cpus)      CPUS="$2";     shift 2 ;;
    -t|--time)      TIME="$2";     shift 2 ;;
    -m|--mem)       MEM="$2";      shift 2 ;;
    --cpu-only)     CPU_ONLY=1;    shift   ;;
    -h|--help)      usage                   ;;
    *) echo "Unknown option: $1"; usage     ;;
  esac
done

# --- build srun command ---
SRUN_ARGS=(
  --partition="${PARTITION}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${TIME}"
  --job-name="interactive"
  --pty
)

if [ "${CPU_ONLY}" -eq 0 ]; then
  SRUN_ARGS+=(--gpus-per-node="${NGPUS}")
fi

echo "[INFO] Requesting interactive node:"
echo "  Partition : ${PARTITION}"
if [ "${CPU_ONLY}" -eq 0 ]; then
  echo "  GPUs      : ${NGPUS}"
fi
echo "  CPUs      : ${CPUS}"
echo "  Memory    : ${MEM}"
echo "  Time      : ${TIME}"
echo ""

exec srun "${SRUN_ARGS[@]}" bash
