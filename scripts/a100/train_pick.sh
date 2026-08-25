#!/bin/bash
# =============================================================================
#  Fine-tune pi0.5 on the Gazebo pick demonstrations, on the A100 cluster.
#
#      sbatch -p part_80gb scripts/a100/train_pick.sh
#
#  Submit it, do not run it. `bash train_pick.sh` on the login node would take
#  the login node's GPUs outside the scheduler.
#
#  This machine is x86_64 with sm_80, so none of the pinning the GB10 needed
#  applies: openpi's own jax[cuda12]==0.5.3 works as shipped. The one deviation
#  from uv.lock is av, pinned at 14.4.0 which PyPI ships only as a source
#  distribution requiring ffmpeg 7 headers that are not installed and cannot be
#  without root. av 16.1.0 has a wheel and satisfies lerobot's `av>=14.2.0`.
#  Nothing here decodes video -- the dataset stores PNG frames in parquet -- so
#  av is imported and never used.
# =============================================================================
#SBATCH --job-name=pi05_pick
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append

set -uo pipefail

ROOT="${HOME}/usr/airopi"
REPO="${ROOT}/airopi-share"
VENV="${ROOT}/venv"
EXP="${EXP_NAME:-pick_a100}"
CONFIG="${CONFIG_YAML:-configs/experiments/example_hsr_pick_gazebo.yaml}"

echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "exp=${EXP} config=${CONFIG}"

cd "${REPO}" || exit 1

export OPENPI_DATA_HOME="${ROOT}/.cache/openpi"
export HF_HOME="${ROOT}/.cache/huggingface"
mkdir -p "${OPENPI_DATA_HOME}" "${HF_HOME}"

# Without this, lerobot resolves the dataset by asking huggingface.co for its
# revisions and dies with "Repository Not Found" -- the local directory is never
# consulted. FastLeRobotDatasetMetadata falls back to HF_LEROBOT_HOME/<repo_id>,
# so this is what makes the copy on disk the one that gets used. The GB10
# containers set it in the image, which is why it was easy to miss here.
export HF_LEROBOT_HOME="${ROOT}/datasets"

# The YAML holds the paths the GB10 containers mount (/home/datasets,
# /home/checkpoints). Rather than keep a second copy of the config in step with
# the first, derive one at submit time with just those two lines rewritten --
# train.py has no CLI override for either.
DERIVED="${ROOT}/.config_${EXP}.yaml"
sed -e "s|^  data_dir: /home/datasets/|  data_dir: ${ROOT}/datasets/|" \
    -e "s|^  params_path: /home/checkpoints/|  params_path: ${ROOT}/checkpoints/|" \
    "${CONFIG}" > "${DERIVED}"
echo "--- paths in effect ---"
grep -nE "data_dir|params_path|assets_dir" "${DERIVED}"
for f in $(grep -oE "${ROOT}[^ ]*" "${DERIVED}"); do
    [ -e "${f}" ] || { echo "MISSING: ${f}"; exit 1; }
done

exec "${VENV}/bin/python" scripts/train.py \
    --config-yaml "${DERIVED}" \
    --exp-name "${EXP}" \
    --checkpoint-base-dir "${ROOT}/checkpoints/_train" \
    --overwrite
