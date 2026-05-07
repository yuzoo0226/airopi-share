#!/usr/bin/env bash

set -e
set -u
set -o pipefail

main() {
  local script_filepath
  script_filepath=$(realpath "${BASH_SOURCE[0]}")
  readonly script_filepath

  local script_filename
  script_filename=$(basename "${script_filepath}")
  readonly script_filename

  local script_dirpath
  script_dirpath=$(dirname "${script_filepath}")
  readonly script_dirpath

  # shellcheck source=/dev/null
  source "${script_dirpath}/include_ddp_utils.sh"

  local usage
  usage=$(cat << EOF_USAGE
Usage:
  ${script_filename} [OPTIONS] -- <command> [args...]

Options:
  --nodes N             Number of nodes to use.
                        Default: auto-detect unique hosts from PBS_NODEFILE.
  --tasks-per-node N    MPI ranks per node. Default: 1.
  --master-addr HOST    Override MASTER_ADDR. Default: first selected node.
  --master-port PORT    Override MASTER_PORT. Default: random non-used local port.
  -h, --help            Show this help.

Environment:
  OPENPI_MPI_LAUNCHER            MPI launcher command. Default: mpirun (fallback: mpiexec)
  OPENPI_MPI_ARGS                Extra args appended to MPI launcher.
  OPENPI_MPI_MAP_BY              Open MPI map policy (e.g., ppr:1:node).
                                 Default: ppr:<tasks-per-node>:node
  OPENPI_MPI_ASSIGN_UCX_NET_DEVICES
                                 If 1, assign UCX_NET_DEVICES per rank.
  OPENPI_MPI_NDR_COUNT           Number of HCAs for UCX_NET_DEVICES assignment. Default: 8.
EOF_USAGE
)

  local options
  options=$(getopt -o h --long help,nodes:,tasks-per-node:,master-addr:,master-port: -- "$@")
  if [ $? -ne 0 ]; then
    echo "${usage}"
    return 1
  fi

  eval set -- "${options}"

  local requested_nodes
  requested_nodes=""

  local tasks_per_node
  tasks_per_node=1

  local master_addr
  master_addr=""

  local master_port
  master_port=""

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --nodes)
        requested_nodes="$2"
        shift 2
        ;;
      --tasks-per-node)
        tasks_per_node="$2"
        shift 2
        ;;
      --master-addr)
        master_addr="$2"
        shift 2
        ;;
      --master-port)
        master_port="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      -h|--help)
        echo "${usage}"
        return 0
        ;;
      *)
        echo "[ERROR] Unknown option: $1"
        echo "${usage}"
        return 1
        ;;
    esac
  done

  if [ "$#" -eq 0 ]; then
    echo "[ERROR] Missing command after '--'."
    echo "${usage}"
    return 1
  fi

  if ! [[ "${tasks_per_node}" =~ ^[0-9]+$ ]] || [ "${tasks_per_node}" -le 0 ]; then
    echo "[ERROR] --tasks-per-node must be a positive integer. got=${tasks_per_node}"
    return 1
  fi

  if [ -n "${requested_nodes}" ]; then
    if ! [[ "${requested_nodes}" =~ ^[0-9]+$ ]] || [ "${requested_nodes}" -le 0 ]; then
      echo "[ERROR] --nodes must be a positive integer. got=${requested_nodes}"
      return 1
    fi
  fi

  local args
  args=("$@")
  readonly args

  local pbs_hostfile
  pbs_hostfile=""
  if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    pbs_hostfile="${PBS_NODEFILE}"
  fi

  local raw_nodes
  raw_nodes=()
  if [ -n "${pbs_hostfile}" ]; then
    while IFS= read -r node_name; do
      [ -z "${node_name}" ] && continue
      raw_nodes+=("${node_name}")
    done < "${pbs_hostfile}"
  else
    raw_nodes+=("$(_ddp_get_master_hostname)")
  fi

  if [ "${#raw_nodes[@]}" -eq 0 ]; then
    echo "[ERROR] Failed to determine nodes from PBS_NODEFILE or hostname."
    return 1
  fi

  local unique_nodes
  unique_nodes=()
  declare -A seen_nodes=()
  local node_name
  for node_name in "${raw_nodes[@]}"; do
    if [ -z "${seen_nodes[${node_name}]:-}" ]; then
      unique_nodes+=("${node_name}")
      seen_nodes["${node_name}"]=1
    fi
  done

  if [ "${#unique_nodes[@]}" -eq 0 ]; then
    echo "[ERROR] Failed to build unique node list."
    return 1
  fi

  local n_nodes
  if [ -n "${requested_nodes}" ]; then
    n_nodes="${requested_nodes}"
    if [ "${n_nodes}" -gt "${#unique_nodes[@]}" ]; then
      echo "[ERROR] Requested --nodes=${n_nodes}, but only ${#unique_nodes[@]} unique node(s) allocated."
      return 1
    fi
  else
    n_nodes="${#unique_nodes[@]}"
  fi
  readonly n_nodes

  local selected_nodes
  selected_nodes=("${unique_nodes[@]:0:${n_nodes}}")

  if [ -n "${master_addr}" ]; then
    :
  elif [ -n "${OPENPI_MASTER_ADDR:-}" ]; then
    master_addr="${OPENPI_MASTER_ADDR}"
  else
    master_addr="${selected_nodes[0]}"
  fi

  if [ -n "${master_port}" ]; then
    :
  elif [ -n "${OPENPI_MASTER_PORT:-}" ]; then
    master_port="${OPENPI_MASTER_PORT}"
  else
    master_port=$(python - <<'PY'
import random
import socket

for _ in range(64):
    p = random.randint(10000, 42767)
    with socket.socket() as s:
        try:
            s.bind(("", p))
        except OSError:
            continue
        print(p)
        break
PY
)
    if [ -z "${master_port}" ]; then
      master_port=$((10000 + RANDOM % 32768))
    fi
  fi

  if ! [[ "${master_port}" =~ ^[0-9]+$ ]] || [ "${master_port}" -le 0 ]; then
    echo "[ERROR] master port must be a positive integer. got=${master_port}"
    return 1
  fi

  local world_size
  world_size=$((n_nodes * tasks_per_node))
  readonly world_size

  local mpi_map_by
  mpi_map_by="${OPENPI_MPI_MAP_BY:-ppr:${tasks_per_node}:node}"

  local mpi_launcher
  mpi_launcher="${OPENPI_MPI_LAUNCHER:-mpirun}"
  if ! command -v "${mpi_launcher}" >/dev/null 2>&1; then
    if [ "${mpi_launcher}" = "mpirun" ] && command -v mpiexec >/dev/null 2>&1; then
      mpi_launcher="mpiexec"
    else
      echo "[ERROR] MPI launcher not found: ${mpi_launcher}"
      return 1
    fi
  fi

  local shared_tmp_root
  shared_tmp_root="${OPENPI_MPI_SHARED_TMPDIR:-${PBS_O_WORKDIR:-${PWD}}}"
  mkdir -p "${shared_tmp_root}"
  if [ ! -d "${shared_tmp_root}" ]; then
    echo "[ERROR] shared temp dir does not exist: ${shared_tmp_root}"
    return 1
  fi

  local hostfile
  hostfile=""
  local hostfile_is_temp
  hostfile_is_temp=0

  if [ -n "${pbs_hostfile}" ] && [ "${n_nodes}" -eq "${#unique_nodes[@]}" ]; then
    hostfile="${pbs_hostfile}"
  else
    hostfile=$(mktemp "${shared_tmp_root}/openpi_mpi_hostfile.XXXXXX")
    hostfile_is_temp=1
    if [ -n "${pbs_hostfile}" ]; then
      declare -A selected_lookup=()
      for node_name in "${selected_nodes[@]}"; do
        selected_lookup["${node_name}"]=1
      done
      while IFS= read -r node_name; do
        [ -z "${node_name}" ] && continue
        if [ -n "${selected_lookup[${node_name}]:-}" ]; then
          echo "${node_name}" >> "${hostfile}"
        fi
      done < "${pbs_hostfile}"
    else
      for node_name in "${selected_nodes[@]}"; do
        local i
        for ((i = 0; i < tasks_per_node; i++)); do
          echo "${node_name}" >> "${hostfile}"
        done
      done
    fi
  fi

  local worker_wrapper
  worker_wrapper=$(mktemp "${shared_tmp_root}/openpi_mpi_worker.XXXXXX.sh")
  readonly worker_wrapper

  cleanup_tmp() {
    if [ "${hostfile_is_temp}" -eq 1 ]; then
      rm -f "${hostfile}"
    fi
    rm -f "${worker_wrapper}"
  }
  trap cleanup_tmp RETURN

  cat > "${worker_wrapper}" << 'EOF_WORKER'
#!/usr/bin/env bash
set -e
set -u

rank="${RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${MV2_COMM_WORLD_RANK:-0}}}}}"
local_rank="${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${MV2_COMM_WORLD_LOCAL_RANK:-0}}}}"
world_size="${WORLD_SIZE:-${OMPI_COMM_WORLD_SIZE:-${PMI_SIZE:-${PMIX_SIZE:-${MV2_COMM_WORLD_SIZE:-1}}}}}"
local_world_size="${LOCAL_WORLD_SIZE:-${OMPI_COMM_WORLD_LOCAL_SIZE:-${MPI_LOCALNRANKS:-${MV2_COMM_WORLD_LOCAL_SIZE:-${OPENPI_MPI_TASKS_PER_NODE:-1}}}}}"

if [ -z "${OPENPI_MASTER_ADDR:-}" ] || [ -z "${OPENPI_MASTER_PORT:-}" ]; then
  echo "[ERROR] OPENPI_MASTER_ADDR/OPENPI_MASTER_PORT are required for MPI launcher."
  exit 1
fi

if [ "${OPENPI_MPI_TASKS_PER_NODE:-1}" -gt 1 ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${local_rank}"
  echo "[INFO][mpi] rank=${rank} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

if [ "${OPENPI_MPI_ASSIGN_UCX_NET_DEVICES:-0}" = "1" ]; then
  ndr_count="${OPENPI_MPI_NDR_COUNT:-8}"
  if [[ "${ndr_count}" =~ ^[0-9]+$ ]] && [ "${ndr_count}" -gt 0 ]; then
    ndr_index=$((rank % ndr_count + 1))
    export UCX_NET_DEVICES="mlx5_ibn${ndr_index}:1"
    echo "[INFO][mpi] rank=${rank} UCX_NET_DEVICES=${UCX_NET_DEVICES}"
  fi
fi

export RANK="${rank}"
export LOCAL_RANK="${local_rank}"
export WORLD_SIZE="${world_size}"
export LOCAL_WORLD_SIZE="${local_world_size}"
export MASTER_ADDR="${MASTER_ADDR:-${OPENPI_MASTER_ADDR}}"
export MASTER_PORT="${MASTER_PORT:-${OPENPI_MASTER_PORT}}"

echo "[INFO][mpi] rank=${RANK}/${WORLD_SIZE} local_rank=${LOCAL_RANK}/${LOCAL_WORLD_SIZE} host=$(hostname) master=${MASTER_ADDR}:${MASTER_PORT}"
exec "$@"
EOF_WORKER
  chmod +x "${worker_wrapper}"

  local mpi_extra_args_arr
  mpi_extra_args_arr=()
  if [ -n "${OPENPI_MPI_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    mpi_extra_args_arr=( ${OPENPI_MPI_ARGS} )
  fi

  export OPENPI_MASTER_ADDR="${master_addr}"
  export OPENPI_MASTER_PORT="${master_port}"
  export OPENPI_MPI_TASKS_PER_NODE="${tasks_per_node}"

  echo "[INFO][mpi] selected_nodes=${selected_nodes[*]}"
  echo "[INFO][mpi] nodes=${n_nodes} tasks_per_node=${tasks_per_node} world_size=${world_size}"
  echo "[INFO][mpi] hostfile=${hostfile}"
  echo "[INFO][mpi] MASTER_ADDR=${OPENPI_MASTER_ADDR} MASTER_PORT=${OPENPI_MASTER_PORT}"
  echo "[INFO][mpi] map_by=${mpi_map_by} launcher=${mpi_launcher}"

  local mpi_cmd
  mpi_cmd=(
    "${mpi_launcher}"
    -np "${world_size}"
    -hostfile "${hostfile}"
    -map-by "${mpi_map_by}"
  )
  if [ "${#mpi_extra_args_arr[@]}" -gt 0 ]; then
    mpi_cmd+=("${mpi_extra_args_arr[@]}")
  fi
  mpi_cmd+=(
    "${worker_wrapper}"
    "${args[@]}"
  )

  echo "[INFO][mpi] running command:"
  printf ' %q' "${mpi_cmd[@]}"
  echo
  "${mpi_cmd[@]}"
}

trap 'unset main' EXIT
main "$@"
