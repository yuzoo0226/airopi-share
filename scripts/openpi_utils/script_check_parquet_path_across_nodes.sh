#!/usr/bin/bash

set -e
set -u

function usage()
{
    cat << 'EOF'
Usage:
  script_check_parquet_path_across_nodes.sh [OPTIONS]

Options:
    --path PATH            Validate one specific parquet path (single-file mode).
                                                 If omitted, random sampling mode is used.
    --check-all-episodes   Validate all expected episode parquet paths derived from
                                                 meta/episodes.jsonl on every node.
    --dataset-root DIR     Dataset root directory containing data/chunk-*/episode_*.parquet
                                                 Default: /home/datasets/lerobot_datasets/airoa-hsr-all-v1.0-202504-202512-68tasks
    --samples-per-node N   Number of random parquet samples to validate on each node in sampling mode.
                                                 Default: 20
    --seed N               Random seed for sampling mode. Default: 42
  --nodes N              Number of nodes/tasks to check. Default: 8
  --partition NAME       Optional partition for srun
  --time LIMIT           Optional time limit for srun (e.g. 00:05:00)
  --no-pyarrow-check     Skip pyarrow open test
    --retries N            Parquet open retries (pyarrow mode). Default: 3
    --backoff-sec SEC      Base backoff seconds for retry (exponential). Default: 0.5
    --python-bin BIN       Python executable used on worker nodes. Default: OPENPI_PREFLIGHT_PYTHON or python
  -h, --help             Show this help

Examples:
  ./scripts/openpi_utils/script_check_parquet_path_across_nodes.sh
  ./scripts/openpi_utils/script_check_parquet_path_across_nodes.sh --path /path/to/file.parquet --nodes 8
    ./scripts/openpi_utils/script_check_parquet_path_across_nodes.sh --dataset-root /home/datasets/lerobot_datasets/my-dataset --samples-per-node 20 --nodes 8
EOF
}

function main()
{
    local target_path
        target_path=""

    local check_all_episodes
    check_all_episodes=0

        local dataset_root
        dataset_root="/home/datasets/lerobot_datasets/airoa-hsr-all-v1.0-202504-202512-68tasks"

        local samples_per_node
        samples_per_node=20

        local random_seed
        random_seed=42

    local nodes
    nodes=8

    local partition
    partition=""

    local time_limit
    time_limit=""

    local pyarrow_check
    pyarrow_check=1

    local parquet_open_retries
    parquet_open_retries=3

    local parquet_open_backoff_sec
    parquet_open_backoff_sec=0.5

    local python_bin
    python_bin="${OPENPI_PREFLIGHT_PYTHON:-python}"

    while [[ $# -gt 0 ]]
    do
        case "$1" in
            --path)
                target_path="$2"
                shift 2
                ;;
            --dataset-root)
                dataset_root="$2"
                shift 2
                ;;
            --check-all-episodes)
                check_all_episodes=1
                shift
                ;;
            --samples-per-node)
                samples_per_node="$2"
                shift 2
                ;;
            --seed)
                random_seed="$2"
                shift 2
                ;;
            --nodes)
                nodes="$2"
                shift 2
                ;;
            --partition)
                partition="$2"
                shift 2
                ;;
            --time)
                time_limit="$2"
                shift 2
                ;;
            --no-pyarrow-check)
                pyarrow_check=0
                shift
                ;;
            --retries)
                parquet_open_retries="$2"
                shift 2
                ;;
            --backoff-sec)
                parquet_open_backoff_sec="$2"
                shift 2
                ;;
            --python-bin)
                python_bin="$2"
                shift 2
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                echo "[ERROR] Unknown argument: $1"
                usage
                return 1
                ;;
        esac
    done

    if ! [[ "${nodes}" =~ ^[0-9]+$ ]] || [[ "${nodes}" -le 0 ]]
    then
        echo "[ERROR] --nodes must be a positive integer. got=${nodes}"
        return 1
    fi

    if ! [[ "${samples_per_node}" =~ ^[0-9]+$ ]] || [[ "${samples_per_node}" -le 0 ]]
    then
        echo "[ERROR] --samples-per-node must be a positive integer. got=${samples_per_node}"
        return 1
    fi

    if ! [[ "${parquet_open_retries}" =~ ^[0-9]+$ ]] || [[ "${parquet_open_retries}" -le 0 ]]
    then
        echo "[ERROR] --retries must be a positive integer. got=${parquet_open_retries}"
        return 1
    fi

    if ! command -v "${python_bin}" >/dev/null 2>&1
    then
        if command -v python3 >/dev/null 2>&1
        then
            python_bin="python3"
        elif command -v python >/dev/null 2>&1
        then
            python_bin="python"
        else
            echo "[ERROR] No usable python executable found. tried: ${python_bin}, python3, python"
            return 1
        fi
    fi

    local target_paths
    target_paths=""
    if [[ ${check_all_episodes} -eq 1 && -n "${target_path}" ]]
    then
        echo "[ERROR] --path and --check-all-episodes are mutually exclusive."
        return 1
    fi

    if [[ -n "${target_path}" ]]
    then
        target_paths="${target_path}"
    elif [[ ${check_all_episodes} -eq 1 ]]
    then
        if [[ ! -f "${dataset_root}/meta/episodes.jsonl" ]]
        then
            echo "[ERROR] episodes metadata not found: ${dataset_root}/meta/episodes.jsonl"
            return 1
        fi
    else
        if [[ ! -d "${dataset_root}/data" ]]
        then
            echo "[ERROR] dataset data dir not found: ${dataset_root}/data"
            return 1
        fi

        target_paths=$(DATASET_ROOT="${dataset_root}" SAMPLES_PER_NODE="${samples_per_node}" RANDOM_SEED="${random_seed}" python3 - << 'PY'
import os
import random
import sys
from pathlib import Path

dataset_root = Path(os.environ["DATASET_ROOT"])
samples_per_node = int(os.environ["SAMPLES_PER_NODE"])
random_seed = int(os.environ["RANDOM_SEED"])

data_dir = dataset_root / "data"
chunk_dirs = [p for p in data_dir.glob("chunk-*") if p.is_dir()]
if not chunk_dirs:
    raise SystemExit(f"[ERROR] no chunk-* directories found under {data_dir}")

rng = random.Random(random_seed)
rng.shuffle(chunk_dirs)

paths = []
for chunk in chunk_dirs:
    candidates = [p for p in chunk.glob("*.parquet") if p.is_file() or p.is_symlink()]
    if not candidates:
        continue
    paths.append(str(rng.choice(candidates)))
    if len(paths) >= samples_per_node:
        break

if not paths:
    raise SystemExit(f"[ERROR] no parquet files found under {data_dir}")

for p in paths:
    print(p)
PY
)

        if [[ -z "${target_paths}" ]]
        then
            echo "[ERROR] failed to sample parquet paths from ${dataset_root}/data"
            return 1
        fi
    fi

    local py_code
    py_code=$(cat << 'PY'
import json
import os
import socket
import sys
import time
from pathlib import Path

check_all_episodes = os.environ.get("CHECK_ALL_EPISODES", "0") == "1"
dataset_root = os.environ.get("DATASET_ROOT", "")
random_seed = int(os.environ.get("RANDOM_SEED", "42"))
samples_per_node = int(os.environ.get("SAMPLES_PER_NODE", "20"))
paths = [p for p in os.environ.get("TARGET_PATHS", "").splitlines() if p.strip()]
pyarrow_check = os.environ.get("PYARROW_CHECK", "1") == "1"
retries = int(os.environ.get("PARQUET_OPEN_RETRIES", "3"))
base_backoff_sec = float(os.environ.get("PARQUET_OPEN_BACKOFF_SEC", "0.5"))
host = socket.gethostname()

if check_all_episodes:
    root = Path(dataset_root)
    meta_path = root / "meta" / "episodes.jsonl"
    data_dir = root / "data"
    if not meta_path.is_file():
        print(f"host={host} status=NG reason=missing_meta path={meta_path}")
        sys.exit(2)
    if not data_dir.is_dir():
        print(f"host={host} status=NG reason=missing_data_dir path={data_dir}")
        sys.exit(2)

    built_paths = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                episode_index = int(obj["episode_index"])
            except Exception as e:
                print(
                    f"host={host} status=NG reason=invalid_episodes_jsonl_line "
                    f"line={lineno} error={type(e).__name__}: {e}"
                )
                sys.exit(2)

            chunk_id = episode_index // 1000
            built_paths.append(str(data_dir / f"chunk-{chunk_id}" / f"episode_{episode_index}.parquet"))

    paths = built_paths

if not paths:
    print(f"host={host} status=NG reason=no_target_paths")
    sys.exit(2)

failures = []

def append_failure(path: str, reason: str, realpath: str | None) -> None:
    failures.append({"path": path, "reason": reason, "realpath": realpath or "<unresolved>"})

def open_parquet_with_retry(path: str) -> tuple[bool, str]:
    import pyarrow.parquet as pq

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            pf = pq.ParquetFile(path)
            rows = pf.metadata.num_rows if pf.metadata is not None else "unknown"
            return True, f"rows={rows} attempt={attempt}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries:
                sleep_sec = base_backoff_sec * (2 ** (attempt - 1))
                print(
                    f"host={host} path={path} pyarrow_retry={attempt}/{retries} "
                    f"sleep={sleep_sec:.2f}s err={last_error}"
                )
                time.sleep(sleep_sec)
    return False, last_error

print("=" * 88)
print(
    f"host={host} pid={os.getpid()} total_paths={len(paths)} "
    f"pyarrow_check={pyarrow_check} check_all_episodes={check_all_episodes}"
)

for path in paths:
    print("-" * 88)
    print(f"host={host} path={path}")

    realpath = None
    try:
        st_link = os.lstat(path)
        print(f"lstat_type={st_link.st_mode:o}")
    except Exception as e:
        append_failure(path, f"lstat_error={type(e).__name__}: {e}", realpath)
        print(f"lstat_error={type(e).__name__}: {e}")
        continue

    exists = os.path.exists(path)
    islink = os.path.islink(path)
    isdir = os.path.isdir(path)
    isfile = os.path.isfile(path)
    print(f"exists={exists} islink={islink} isdir={isdir} isfile={isfile}")

    try:
        realpath = os.path.realpath(path)
        if islink:
            target = os.readlink(path)
            print(f"readlink={target}")
        print(f"realpath={realpath}")
    except Exception as e:
        print(f"readlink_error={type(e).__name__}: {e}")

    if not exists or isdir or not isfile:
        append_failure(
            path,
            f"invalid_path_state exists={exists} isdir={isdir} isfile={isfile}",
            realpath,
        )
        continue

    try:
        st_target = os.stat(path)
        print(f"stat_type={st_target.st_mode:o} size={st_target.st_size}")
    except Exception as e:
        append_failure(path, f"stat_error={type(e).__name__}: {e}", realpath)
        print(f"stat_error={type(e).__name__}: {e}")
        continue

    if pyarrow_check:
        ok, detail = open_parquet_with_retry(path)
        if ok:
            print(f"pyarrow_open=OK {detail}")
        else:
            append_failure(path, f"pyarrow_open=NG {detail}", realpath)
            print(f"pyarrow_open=NG {detail}")

if failures:
    print("=" * 88)
    print(f"host={host} status=NG failed_paths={len(failures)}")
    for item in failures:
        print(
            "failure "
            f"host={host} path={item['path']} realpath={item['realpath']} reason={item['reason']}"
        )
    sys.stdout.flush()
    sys.exit(2)

print("=" * 88)
print(f"host={host} status=OK checked_paths={len(paths)}")

sys.stdout.flush()
PY
)

    local srun_args
    srun_args=(
        --nodes "${nodes}"
        --ntasks "${nodes}"
        --ntasks-per-node 1
        --overlap
    )

    if [[ -n "${partition}" ]]
    then
        srun_args+=(--partition "${partition}")
    fi

    if [[ -n "${time_limit}" ]]
    then
        srun_args+=(--time "${time_limit}")
    fi

    echo "[INFO] Running node consistency check"
    if [[ -n "${target_path}" ]]
    then
        echo "[INFO] mode=single target_path=${target_path}"
    elif [[ ${check_all_episodes} -eq 1 ]]
    then
        echo "[INFO] mode=all-episodes dataset_root=${dataset_root}"
    else
        echo "[INFO] mode=sampling dataset_root=${dataset_root} samples_per_node=${samples_per_node} seed=${random_seed}"
    fi
    echo "[INFO] nodes=${nodes} pyarrow_check=${pyarrow_check}"
    echo "[INFO] retries=${parquet_open_retries} backoff_sec=${parquet_open_backoff_sec} python_bin=${python_bin}"
    if [[ ${check_all_episodes} -eq 1 ]]
    then
        echo "[INFO] sampled_paths_count=<all episodes from meta/episodes.jsonl>"
    else
        echo "[INFO] sampled_paths_count=$(echo "${target_paths}" | sed '/^$/d' | wc -l)"
    fi
    echo "[INFO] command: srun ${srun_args[*]} ${python_bin} -c '<inline_python>'"

    local preflight_log
    preflight_log=$(mktemp)
    trap 'rm -f "${preflight_log}"' RETURN

    set +e
    TARGET_PATHS="${target_paths}" \
    CHECK_ALL_EPISODES="${check_all_episodes}" \
    DATASET_ROOT="${dataset_root}" \
    SAMPLES_PER_NODE="${samples_per_node}" \
    RANDOM_SEED="${random_seed}" \
    PYARROW_CHECK="${pyarrow_check}" \
    PARQUET_OPEN_RETRIES="${parquet_open_retries}" \
    PARQUET_OPEN_BACKOFF_SEC="${parquet_open_backoff_sec}" \
    srun "${srun_args[@]}" "${python_bin}" -c "${py_code}" | tee "${preflight_log}"
    local srun_rc=$?
    set -e

    echo "[INFO] ===== preflight summary ====="
    local ok_count
    ok_count=$(grep -c 'status=OK' "${preflight_log}" || true)
    local ng_count
    ng_count=$(grep -c 'status=NG' "${preflight_log}" || true)
    echo "[INFO] host_status: OK=${ok_count} NG=${ng_count}"

    if grep -q '^failure ' "${preflight_log}"; then
        echo "[INFO] failing paths (host/path/realpath/reason):"
        grep '^failure ' "${preflight_log}" | sort -u
    fi

    if [[ ${srun_rc} -ne 0 ]]; then
        echo "[ERROR] preflight failed with exit code ${srun_rc}"
        return ${srun_rc}
    fi
}

trap 'unset -f main usage' EXIT
main "$@"
