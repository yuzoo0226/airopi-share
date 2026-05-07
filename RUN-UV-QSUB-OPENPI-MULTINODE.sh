#!/bin/bash

set -e

echo "[INFO]: Started at $(date)"

COMMON_SCRIPT_DIR="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}"
COMMON_SCRIPT="${COMMON_SCRIPT_DIR}/RUN-UV-COMMON.sh"
if [ ! -f "${COMMON_SCRIPT}" ]; then
  echo "[ERROR]: Common script not found: ${COMMON_SCRIPT}"
  exit 1
fi
# shellcheck source=/dev/null
. "${COMMON_SCRIPT}"

openpi_load_cuda_module
openpi_resolve_script_dir "$0"
openpi_load_env_file

# ---- openpi_utils launcher config (PBS qsub + MPI) ----
: "${OPENPI_UTILS_SCRIPT_DIR:=${SCRIPT_DIR}/scripts/openpi_utils}"
: "${OPENPI_SCRIPT_RUN:=${OPENPI_UTILS_SCRIPT_DIR}/script_run.sh}"
: "${OPENPI_DISTRIBUTE_SCRIPT:=${OPENPI_UTILS_SCRIPT_DIR}/script_distribute_pbs_mpi_tasks.sh}"
: "${OPENPI_NUM_NODES:=4}"
: "${OPENPI_MPI_TASKS_PER_NODE:=1}"
: "${OPENPI_MPI_MODULE:=hpcx/2.20}"
: "${OPENPI_MPI_LAUNCHER:=mpirun}"
: "${OPENPI_MPI_MAP_BY:=ppr:${OPENPI_MPI_TASKS_PER_NODE}:node}"
: "${OPENPI_MPI_SHARED_TMPDIR:=}"
: "${OPENPI_CPUS_PER_TASK:=240}"
: "${OPENPI_GRES:=gpu:8}"
: "${OPENPI_QSUB_ARGS:=-V -q rt_HF -P gch51606 -l walltime=336:00:00}"
: "${OPENPI_QSUB_SELECT:=select=${OPENPI_NUM_NODES}:mpiprocs=${OPENPI_MPI_TASKS_PER_NODE}}"
: "${OPENPI_EXTRA_LAUNCH_COMMAND:=uv run}"
: "${CONFIG_NAME:=}"
: "${OPENPI_CONFIG_YAML:=}"
: "${EXP_NAME:=pi05_hsr_task689101112_level12_stationary_train_2node8gpu}"
: "${DATASET_NAME:=}"
: "${OPENPI_RESUME:=1}"
: "${OPENPI_OVERWRITE:=0}"
: "${OPENPI_AGGREGATE_STATS_ENABLE:=0}"
: "${OPENPI_PREFLIGHT_ENABLE:=0}"
: "${OPENPI_PREFLIGHT_SCRIPT:=${OPENPI_UTILS_SCRIPT_DIR}/script_check_parquet_path_across_nodes.sh}"
: "${OPENPI_PREFLIGHT_SAMPLES_PER_NODE:=20}"
: "${OPENPI_PREFLIGHT_RETRIES:=3}"
: "${OPENPI_PREFLIGHT_BACKOFF_SEC:=0.5}"
: "${OPENPI_PREFLIGHT_NO_PYARROW:=0}"
: "${OPENPI_PREFLIGHT_CHECK_ALL_EPISODES:=0}"

ENV_DATA_DIR="${DATA_DIR:-}"
ENV_DATASET_NAME="${DATASET_NAME:-}"
YAML_DATASET_DATA_DIR=""
YAML_DATASET_REPO_ID=""
YAML_DATASET_ASSETS_DIR=""
YAML_DATASET_ASSET_ID=""
YAML_GPU_NUM_GPUS=""
YAML_BASE_MODEL_URL=""
YAML_DATASET_HF_HOME=""
OPENPI_YAML_METADATA_SCRIPT="${OPENPI_UTILS_SCRIPT_DIR}/resolve_training_yaml_metadata.py"

# Ensure `uv run` used for YAML metadata parsing has writable cache dirs,
# even before full runtime env setup.
if [ -z "${CACHE_ROOT:-}" ]; then
  CACHE_ROOT="${SCRIPT_DIR}/tmp_storage"
fi
mkdir -p "${CACHE_ROOT}/uv-cache" "${CACHE_ROOT}/uv-data" "${CACHE_ROOT}/uv-python" "${CACHE_ROOT}/cache"
export UV_CACHE_DIR="${CACHE_ROOT}/uv-cache"
export UV_DATA_DIR="${CACHE_ROOT}/uv-data"
export UV_PYTHON_DIR="${CACHE_ROOT}/uv-python"
export XDG_DATA_HOME="${CACHE_ROOT}/uv-data"
export XDG_CACHE_HOME="${CACHE_ROOT}/cache"

if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  if [ ! -f "${OPENPI_CONFIG_YAML}" ] && [ -f "${SCRIPT_DIR}/${OPENPI_CONFIG_YAML}" ]; then
    OPENPI_CONFIG_YAML="${SCRIPT_DIR}/${OPENPI_CONFIG_YAML}"
  fi
  if [ ! -f "${OPENPI_CONFIG_YAML}" ]; then
    echo "[ERROR]: OPENPI_CONFIG_YAML not found: ${OPENPI_CONFIG_YAML}"
    exit 1
  fi
  if [ ! -f "${OPENPI_YAML_METADATA_SCRIPT}" ]; then
    echo "[ERROR]: YAML metadata helper not found: ${OPENPI_YAML_METADATA_SCRIPT}"
    exit 1
  fi

  yaml_meta_lines=$(uv run python "${OPENPI_YAML_METADATA_SCRIPT}" "${OPENPI_CONFIG_YAML}")
  while IFS='=' read -r key value; do
    case "${key}" in
      DATASET_DATA_DIR) YAML_DATASET_DATA_DIR="${value}" ;;
      DATASET_REPO_ID) YAML_DATASET_REPO_ID="${value}" ;;
      DATASET_ASSETS_DIR) YAML_DATASET_ASSETS_DIR="${value}" ;;
      DATASET_ASSET_ID) YAML_DATASET_ASSET_ID="${value}" ;;
      GPU_NUM_GPUS) YAML_GPU_NUM_GPUS="${value}" ;;
      BASE_MODEL_URL) YAML_BASE_MODEL_URL="${value}" ;;
      DATASET_HF_HOME) YAML_DATASET_HF_HOME="${value}" ;;
    esac
  done <<EOF_YAML_META
${yaml_meta_lines}
EOF_YAML_META
fi

if [ -n "${YAML_DATASET_DATA_DIR}" ]; then
  YAML_DATASET_NAME_FROM_YAML="$(basename "${YAML_DATASET_DATA_DIR}")"
  YAML_DATA_DIR_FROM_YAML="$(dirname "${YAML_DATASET_DATA_DIR}")"

  if [ -n "${ENV_DATA_DIR}" ] && [ "${ENV_DATA_DIR}" != "${YAML_DATA_DIR_FROM_YAML}" ]; then
    echo "[ERROR]: DATA_DIR mismatch between .env and YAML."
    echo "[ERROR]: .env DATA_DIR=${ENV_DATA_DIR}"
    echo "[ERROR]: YAML data_dir parent=${YAML_DATA_DIR_FROM_YAML} (from ${YAML_DATASET_DATA_DIR})"
    echo "[ERROR]: Use YAML as source of truth and unset DATA_DIR in .env, or make them match."
    exit 1
  fi
  if [ -n "${ENV_DATASET_NAME}" ] && [ "${ENV_DATASET_NAME}" != "${YAML_DATASET_NAME_FROM_YAML}" ]; then
    echo "[ERROR]: DATASET_NAME mismatch between .env and YAML."
    echo "[ERROR]: .env DATASET_NAME=${ENV_DATASET_NAME}"
    echo "[ERROR]: YAML inferred DATASET_NAME=${YAML_DATASET_NAME_FROM_YAML}"
    echo "[ERROR]: Use YAML as source of truth and unset DATASET_NAME in .env, or make them match."
    exit 1
  fi

  DATA_DIR="${YAML_DATA_DIR_FROM_YAML}"
  DATASET_NAME="${YAML_DATASET_NAME_FROM_YAML}"
elif [ -n "${YAML_DATASET_REPO_ID}" ]; then
  YAML_DATASET_NAME_FROM_REPO="$(basename "${YAML_DATASET_REPO_ID}")"
  if [ -n "${ENV_DATASET_NAME}" ] && [ "${ENV_DATASET_NAME}" != "${YAML_DATASET_NAME_FROM_REPO}" ]; then
    echo "[ERROR]: DATASET_NAME mismatch between .env and YAML repo_id."
    echo "[ERROR]: .env DATASET_NAME=${ENV_DATASET_NAME}"
    echo "[ERROR]: YAML repo_id basename=${YAML_DATASET_NAME_FROM_REPO}"
    echo "[ERROR]: Use YAML as source of truth and unset DATASET_NAME in .env, or make them match."
    exit 1
  fi
  if [ -z "${DATASET_NAME:-}" ]; then
    DATASET_NAME="${YAML_DATASET_NAME_FROM_REPO}"
  fi
fi

openpi_set_default_path_vars
if [ -n "${YAML_DATASET_HF_HOME}" ]; then
  yaml_meta_info="${YAML_DATASET_HF_HOME}/${YAML_DATASET_REPO_ID}/meta/info.json"
  if [ -f "${yaml_meta_info}" ]; then
    if [ -n "${HF_LEROBOT_HOME:-}" ] && [ "${HF_LEROBOT_HOME}" != "${YAML_DATASET_HF_HOME}" ]; then
      echo "[WARN]: HF_LEROBOT_HOME (${HF_LEROBOT_HOME}) differs from YAML-derived root (${YAML_DATASET_HF_HOME})."
      echo "[WARN]: Overriding HF_LEROBOT_HOME with YAML-derived root because dataset metadata exists there."
    fi
    HF_LEROBOT_HOME="${YAML_DATASET_HF_HOME}"
  else
    echo "[WARN]: YAML-derived dataset metadata not found at ${yaml_meta_info}."
    echo "[WARN]: Keeping existing HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-<unset>}."
  fi
fi
openpi_export_standard_vars
openpi_require_vars WORKING_DIR DATA_DIR HF_LEROBOT_HOME CACHE_ROOT
openpi_warn_if_unset WANDB_API_KEY
openpi_setup_cache_env
openpi_log_standard_paths

if [ -z "${OPENPI_MPI_SHARED_TMPDIR}" ]; then
  OPENPI_MPI_SHARED_TMPDIR="${WORKING_DIR}"
fi
if ! mkdir -p "${OPENPI_MPI_SHARED_TMPDIR}" >/dev/null 2>&1; then
  echo "[ERROR]: OPENPI_MPI_SHARED_TMPDIR is not writable: ${OPENPI_MPI_SHARED_TMPDIR}"
  exit 1
fi

OPENPI_TRAIN_DISTRIBUTED_SCRIPT="${OPENPI_UTILS_SCRIPT_DIR}/script_train_distributedly.py"
if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  if [ -n "${CONFIG_NAME}" ]; then
    TRAIN_CONFIG_NAME="${CONFIG_NAME}"
  else
    TRAIN_CONFIG_NAME="$(basename "${OPENPI_CONFIG_YAML%.*}")"
  fi
else
  : "${CONFIG_NAME:=pi05_hsr_task689101112_level12_stationary_train}"
  TRAIN_CONFIG_NAME="${CONFIG_NAME}"
fi
if [ -z "${DATASET_NAME}" ]; then
  DATASET_NAME="${TRAIN_CONFIG_NAME}"
fi

if [ -n "${YAML_GPU_NUM_GPUS}" ]; then
  gpus_per_node="$(echo "${OPENPI_GRES}" | awk -F: '{print $NF}')"
  if [[ "${OPENPI_NUM_NODES}" =~ ^[0-9]+$ ]] && [[ "${OPENPI_MPI_TASKS_PER_NODE}" =~ ^[0-9]+$ ]] && [[ "${gpus_per_node}" =~ ^[0-9]+$ ]] && [[ "${YAML_GPU_NUM_GPUS}" =~ ^[0-9]+$ ]]; then
    if [ "${OPENPI_MPI_TASKS_PER_NODE}" -eq 1 ]; then
      expected_total_gpus=$((OPENPI_NUM_NODES * gpus_per_node))
    else
      expected_total_gpus=$((OPENPI_NUM_NODES * OPENPI_MPI_TASKS_PER_NODE))
      if [ "${OPENPI_MPI_TASKS_PER_NODE}" -gt "${gpus_per_node}" ]; then
        echo "[WARN]: OPENPI_MPI_TASKS_PER_NODE (${OPENPI_MPI_TASKS_PER_NODE}) > GPUs per node (${gpus_per_node})."
      fi
    fi
    if [ "${expected_total_gpus}" -ne "${YAML_GPU_NUM_GPUS}" ]; then
      echo "[WARN]: YAML gpu.num_gpus (${YAML_GPU_NUM_GPUS}) != expected launcher GPUs (${expected_total_gpus})."
      echo "[WARN]: expected = OPENPI_NUM_NODES(${OPENPI_NUM_NODES}) * ${OPENPI_MPI_TASKS_PER_NODE} task(s)/node"
      if [ "${OPENPI_MPI_TASKS_PER_NODE}" -eq 1 ]; then
        echo "[WARN]: (single task per node uses all GPUs from OPENPI_GRES=${OPENPI_GRES})."
      else
        echo "[WARN]: (multi-task per node assumes one GPU per task)."
      fi
    fi
  fi
fi

if [[ "${OPENPI_QSUB_SELECT}" != *"mpiprocs="* ]]; then
  echo "[WARN]: OPENPI_QSUB_SELECT does not include mpiprocs=."
  echo "[WARN]: For MPI jobs on ABCI, set OPENPI_QSUB_SELECT like: select=${OPENPI_NUM_NODES}:mpiprocs=${OPENPI_MPI_TASKS_PER_NODE}"
fi

echo "[INFO]: Config ownership:"
echo "[INFO]:   YAML (experiment semantics): ${OPENPI_CONFIG_YAML:-<none>}"
echo "[INFO]:   .env (launcher/runtime): OPENPI_NUM_NODES=${OPENPI_NUM_NODES}, OPENPI_MPI_TASKS_PER_NODE=${OPENPI_MPI_TASKS_PER_NODE}, OPENPI_GRES=${OPENPI_GRES}, OPENPI_RESUME=${OPENPI_RESUME}, OPENPI_OVERWRITE=${OPENPI_OVERWRITE}"
echo "[INFO]:   mpi module=${OPENPI_MPI_MODULE}, launcher=${OPENPI_MPI_LAUNCHER}, map_by=${OPENPI_MPI_MAP_BY}"
echo "[INFO]:   mpi shared tmpdir=${OPENPI_MPI_SHARED_TMPDIR}"
echo "[INFO]:   qsub args: OPENPI_QSUB_ARGS=${OPENPI_QSUB_ARGS}"
echo "[INFO]:   qsub select: OPENPI_QSUB_SELECT=${OPENPI_QSUB_SELECT}"
echo "[INFO]: Resolved dataset root: ${DATA_DIR}/${DATASET_NAME}"

if [ ! -f "${OPENPI_SCRIPT_RUN}" ]; then
  echo "[ERROR]: script_run.sh not found: ${OPENPI_SCRIPT_RUN}"
  exit 1
fi

if [ ! -f "${OPENPI_DISTRIBUTE_SCRIPT}" ]; then
  echo "[ERROR]: script_distribute_pbs_mpi_tasks.sh not found: ${OPENPI_DISTRIBUTE_SCRIPT}"
  exit 1
fi

if [ ! -f "${OPENPI_TRAIN_DISTRIBUTED_SCRIPT}" ]; then
  echo "[ERROR]: script_train_distributedly.py not found: ${OPENPI_TRAIN_DISTRIBUTED_SCRIPT}"
  exit 1
fi

default_slurm_preflight_script="${OPENPI_UTILS_SCRIPT_DIR}/script_check_parquet_path_across_nodes.sh"
if [ "${OPENPI_PREFLIGHT_ENABLE}" = "1" ] && [ "${OPENPI_PREFLIGHT_SCRIPT}" = "${default_slurm_preflight_script}" ]; then
  if ! command -v srun >/dev/null 2>&1; then
    echo "[WARN]: OPENPI_PREFLIGHT_SCRIPT uses srun but srun is not available in qsub mode."
    echo "[WARN]: preflight is disabled. Set OPENPI_PREFLIGHT_SCRIPT to an MPI-compatible checker to re-enable."
    OPENPI_PREFLIGHT_ENABLE="0"
  fi
fi

if [ "${OPENPI_PREFLIGHT_ENABLE}" = "1" ] && [ ! -f "${OPENPI_PREFLIGHT_SCRIPT}" ]; then
  echo "[ERROR]: preflight script not found: ${OPENPI_PREFLIGHT_SCRIPT}"
  exit 1
fi

train_args=()
if [ -n "${OPENPI_CONFIG_YAML}" ]; then
  train_args+=( "--config-yaml" "${OPENPI_CONFIG_YAML}" )
else
  train_args+=( "${CONFIG_NAME}" )
fi
train_args+=( "--exp-name" "${EXP_NAME}" )
if [ "${OPENPI_RESUME}" = "1" ] && [ "${OPENPI_OVERWRITE}" = "1" ]; then
  echo "[ERROR]: OPENPI_RESUME and OPENPI_OVERWRITE cannot both be 1."
  exit 1
fi

if [ "${OPENPI_RESUME}" = "1" ]; then
  train_args+=(--resume)
elif [ "${OPENPI_OVERWRITE}" = "1" ]; then
  train_args+=(--overwrite)
fi
if [ "$#" -gt 0 ]; then
  train_args+=("$@")
fi

qsub_args_arr=()
if [ -n "${OPENPI_QSUB_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  qsub_args_arr=( ${OPENPI_QSUB_ARGS} )
fi

extra_launch_arr=()
if [ -n "${OPENPI_EXTRA_LAUNCH_COMMAND:-}" ]; then
  # shellcheck disable=SC2206
  extra_launch_arr=( ${OPENPI_EXTRA_LAUNCH_COMMAND} )
fi

cd "${WORKING_DIR}"
PARENT_DATA_DIR=$(basename "${DATA_DIR}")
EPISODES_STATS_FILE="${DATA_DIR}/${DATASET_NAME}/meta/episodes_stats.jsonl"
NORM_STATS_OUTPUT_FILE="assets/${TRAIN_CONFIG_NAME}/${PARENT_DATA_DIR}/${DATASET_NAME}/norm_stats.json"
DATA_CHUNK_DIR="${DATA_DIR}/${DATASET_NAME}/data/"

echo "[INFO]: DATASET_NAME=${DATASET_NAME}"
if [ "${OPENPI_AGGREGATE_STATS_ENABLE}" = "1" ]; then
  echo "[INFO]: aggregate stats is enabled"
else
  echo "[INFO]: aggregate stats is disabled"
fi
if [ "${OPENPI_PREFLIGHT_ENABLE}" = "1" ]; then
  echo "[INFO]: preflight parquet check is enabled"
fi

aggregate_stats_cmd=(
  JAX_PLATFORMS=cpu
  uv
  run
  python
  scripts/aggregate_stats_simple.py
  --episodes-stats "${EPISODES_STATS_FILE}"
  --output-file "${NORM_STATS_OUTPUT_FILE}"
  --chunk-dir "${DATA_CHUNK_DIR}"
  --action-column "action.relative"
  --action-mode "relative"
)

distributed_launch_cmd=(
  bash
  "${OPENPI_DISTRIBUTE_SCRIPT}"
  --nodes "${OPENPI_NUM_NODES}"
  --tasks-per-node "${OPENPI_MPI_TASKS_PER_NODE}"
  --
)
if [ "${#extra_launch_arr[@]}" -gt 0 ]; then
  distributed_launch_cmd+=( "${extra_launch_arr[@]}" )
fi
distributed_launch_cmd+=(
  python
  "${OPENPI_TRAIN_DISTRIBUTED_SCRIPT}"
  "${train_args[@]}"
)

preflight_cmd=()
if [ "${OPENPI_PREFLIGHT_ENABLE}" = "1" ]; then
  preflight_cmd=(
    uv
    run
    bash
    "${OPENPI_PREFLIGHT_SCRIPT}"
    --dataset-root "${DATA_DIR}/${DATASET_NAME}"
    --nodes "${OPENPI_NUM_NODES}"
    --samples-per-node "${OPENPI_PREFLIGHT_SAMPLES_PER_NODE}"
    --retries "${OPENPI_PREFLIGHT_RETRIES}"
    --backoff-sec "${OPENPI_PREFLIGHT_BACKOFF_SEC}"
  )
  if [ "${OPENPI_PREFLIGHT_NO_PYARROW}" = "1" ]; then
    preflight_cmd+=(--no-pyarrow-check)
  fi
  if [ "${OPENPI_PREFLIGHT_CHECK_ALL_EPISODES}" = "1" ]; then
    preflight_cmd+=(--check-all-episodes)
  fi
fi

quote_command() {
  local quoted
  printf -v quoted '%q ' "$@"
  echo "${quoted% }"
}

aggregate_stats_cmd_str=$(quote_command "${aggregate_stats_cmd[@]}")
distributed_launch_cmd_str=$(quote_command "${distributed_launch_cmd[@]}")
cd_working_dir_cmd_str=$(quote_command cd "${WORKING_DIR}")
compute_node_launch_cmd="${cd_working_dir_cmd_str}"
if [ "${OPENPI_AGGREGATE_STATS_ENABLE}" = "1" ]; then
  compute_node_launch_cmd+=" && ${aggregate_stats_cmd_str}"
fi
if [ "${OPENPI_PREFLIGHT_ENABLE}" = "1" ]; then
  preflight_cmd_str=$(quote_command "${preflight_cmd[@]}")
  compute_node_launch_cmd+=" && ${preflight_cmd_str}"
fi
compute_node_launch_cmd+=" && ${distributed_launch_cmd_str}"

script_run_cmd_str=$(quote_command bash "${OPENPI_SCRIPT_RUN}" bash -lc "${compute_node_launch_cmd}")

job_script_file=$(mktemp "${TMPDIR:-/tmp}/openpi_qsub_job.XXXXXX.sh")
cleanup_job_script() {
  rm -f "${job_script_file}"
}
trap cleanup_job_script EXIT

{
  echo "#!/bin/bash"
  echo "set -e"
  echo "if command -v module >/dev/null 2>&1; then module load cuda; fi"
  echo "if command -v module >/dev/null 2>&1 && [ -n \"${OPENPI_MPI_MODULE}\" ]; then module load \"${OPENPI_MPI_MODULE}\"; fi"
  echo "if ! command -v ${OPENPI_MPI_LAUNCHER} >/dev/null 2>&1 && ! command -v mpiexec >/dev/null 2>&1; then"
  echo "  echo \"[ERROR]: MPI launcher not found (${OPENPI_MPI_LAUNCHER}/mpiexec).\""
  echo "  exit 1"
  echo "fi"
  echo "export OPENPI_MPI_LAUNCHER=$(printf '%q' "${OPENPI_MPI_LAUNCHER}")"
  echo "export OPENPI_MPI_MAP_BY=$(printf '%q' "${OPENPI_MPI_MAP_BY}")"
  echo "export OPENPI_MPI_SHARED_TMPDIR=$(printf '%q' "${OPENPI_MPI_SHARED_TMPDIR}")"
  echo "export WORKING_DIR=$(printf '%q' "${WORKING_DIR}")"
  echo "export DATA_DIR=$(printf '%q' "${DATA_DIR}")"
  echo "export DATASET_NAME=$(printf '%q' "${DATASET_NAME}")"
  echo "export HF_LEROBOT_HOME=$(printf '%q' "${HF_LEROBOT_HOME}")"
  echo "export CACHE_ROOT=$(printf '%q' "${CACHE_ROOT}")"
  echo "export CHECKPOINT_DIR=$(printf '%q' "${CHECKPOINT_DIR}")"
  echo "export OPENPI_CONFIG_YAML=$(printf '%q' "${OPENPI_CONFIG_YAML}")"
  echo "export OPENPI_DATASET_ROOT=$(printf '%q' "${DATA_DIR}/${DATASET_NAME}")"
  echo "export EXP_NAME=$(printf '%q' "${EXP_NAME}")"
  if [ -n "${OPENPI_MPI_ARGS:-}" ]; then
    echo "export OPENPI_MPI_ARGS=$(printf '%q' "${OPENPI_MPI_ARGS}")"
  fi
  if [ -n "${OPENPI_MPI_ASSIGN_UCX_NET_DEVICES:-}" ]; then
    echo "export OPENPI_MPI_ASSIGN_UCX_NET_DEVICES=$(printf '%q' "${OPENPI_MPI_ASSIGN_UCX_NET_DEVICES}")"
  fi
  if [ -n "${OPENPI_MPI_NDR_COUNT:-}" ]; then
    echo "export OPENPI_MPI_NDR_COUNT=$(printf '%q' "${OPENPI_MPI_NDR_COUNT}")"
  fi
  if [ -n "${UCX_MAX_RNDV_RAILS:-}" ]; then
    echo "export UCX_MAX_RNDV_RAILS=$(printf '%q' "${UCX_MAX_RNDV_RAILS}")"
  fi
  if [ -n "${UCX_MAX_EAGER_RAILS:-}" ]; then
    echo "export UCX_MAX_EAGER_RAILS=$(printf '%q' "${UCX_MAX_EAGER_RAILS}")"
  fi
  echo "${script_run_cmd_str}"
} > "${job_script_file}"
chmod +x "${job_script_file}"

cmd=(qsub)
if [ "${#qsub_args_arr[@]}" -gt 0 ]; then
  cmd+=( "${qsub_args_arr[@]}" )
fi
if [ -n "${OPENPI_QSUB_SELECT:-}" ]; then
  cmd+=( -l "${OPENPI_QSUB_SELECT}" )
fi
cmd+=( "${job_script_file}" )

echo "[INFO]: Running submit command (qsub + MPI):"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

echo "[INFO]: Finished at $(date)"
