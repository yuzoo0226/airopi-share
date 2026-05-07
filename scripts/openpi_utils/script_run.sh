#!/bin/bash


function main()
{

function log_header()
{
    cat << EOF | grep -v \#

================================= $@ =================================

EOF
}

function is_sensitive_env_name()
{
    local name_upper
    name_upper="${1^^}"
    case "${name_upper}" in
        *API_KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*PASSWD*|*PASS*|*CREDENTIAL*|*AUTH*|*COOKIE*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

function should_log_env_name()
{
    local name
    name="$1"
    case "${name}" in
        SLURM_*|CUDA_*|NCCL_*|OPENPI_*|HF_*|UV_*|XDG_*|JAX_*|XLA_*|OMP_*|MKL_*|WANDB_*|\
        WORKING_DIR|DATA_DIR|CACHE_ROOT|CHECKPOINT_DIR|CONFIG_NAME|EXP_NAME|DATASET_NAME|\
        PATH|PYTHONPATH|LD_LIBRARY_PATH|TMPDIR|PWD|USER|HOME|HOSTNAME|SHELL)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

function log_environment()
{
    log_header "Environment (safe)"
    local name
    local value
    local display_value
    local logged_count
    local redacted_count
    logged_count=0
    redacted_count=0

    while IFS='=' read -r name value
    do
        if ! should_log_env_name "${name}" && ! is_sensitive_env_name "${name}"
        then
            continue
        fi

        if is_sensitive_env_name "${name}"
        then
            display_value="<redacted>"
            redacted_count=$((redacted_count + 1))
        else
            display_value="${value}"
        fi
        printf '%s=%q\n' "${name}" "${display_value}"
        logged_count=$((logged_count + 1))
    done < <(env | LC_ALL=C sort)

    echo "Logged env vars: ${logged_count} (redacted: ${redacted_count})"
}

function log_env()
{
log_header "Information"
cat << EOF
Command: $@
Working dir: $(pwd)
Date: $(date)
EOF

log_header "Arguments"
local arg
for arg in "$@"
do 
    echo "$arg"
done 

log_environment

	local git_root
	git_root=$(pwd)				
readonly git_root

	local is_git_repo
	is_git_repo=0
	if git -C "${git_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1
	then
		is_git_repo=1
	fi


function run_git_command()
{
    log_header git "$@"
    git -C "${git_root}" "$@"

}


if [[ 1 -eq "${is_git_repo}" ]]
then
cat << EOF | grep -v \# | while read line ; do run_git_command ${line} ; done  
rev-parse HEAD
status --short
show HEAD --name-only
diff --name-only
EOF
else
	echo "[WARN] Working dir is not a git repository. Skipping git diagnostics."
fi

log_header "Output"
}
log_env "$@"
"$@"

}
trap "unset main" EXIT
main "$@"
