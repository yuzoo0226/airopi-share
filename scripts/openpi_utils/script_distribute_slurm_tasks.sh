#!/usr/bin/bash

set -e
set -u

# NOTE: While this script is using pipes,
# because the pipes include grep,
# which can return non-zero values
# if the patterns do not exist,
# -o pipefail is disabled.
# TODO: Remove all grep in order to add -o pipefail.
# set -o pipefail

function main()
{
    local script_filepath
    script_filepath=$(realpath "${BASH_SOURCE[0]}")
    readonly script_filepath

    local script_filename
    script_filename=$(basename "${script_filepath}")
    readonly script_filename

    local script_dirpath
    script_dirpath=$(dirname "${script_filepath}")
    readonly script_dirpath
	
	source "${script_dirpath}/include_ddp_utils.sh"

	local arg_spec
	arg_spec=$(cat << EOF | grep -v "\#" | tr "\n" ","
w_one_task_per_node:
EOF
)	
	readonly arg_spec
	
	local usage
	usage="${script_filename} -o h --long ${arg_spec}help"
	readonly usage

	local OPTIONS
	OPTIONS=$(getopt -n ${usage} -- "$@")
	readonly OPTIONS
	[[ $? -ne 0 ]] && return 

	eval set -- "$OPTIONS"
	
	local w_one_task_per_node
	
	w_one_task_per_node=1
	
	while [[ ! -z "$1" ]]
	do
		case "$1" in 
			--w_one_task_per_node )
				w_one_task_per_node="$2"
				shift 
			;;
			-- )
				shift 
				break
			;;
			"-h" | "--help" )
				echo "${usage}"
				return
			;;
		esac
		shift
	done
	
	readonly w_one_task_per_node
	
    local args
    args=()
    args+=("$@")
    readonly args

    echo "Selecting the master ... "
	
    local master_hostname
	master_hostname=$(_ddp_get_master_hostname)
    readonly master_hostname

    echo "Selecting a non-used port on ${master_hostname} ... "
    local master_port
	master_port=$(_ddp_get_port_on_host "${master_hostname}")
    readonly master_port

    if [[ -z "${master_port}" ]]
    then
        echo "[ERROR] Failed to find a non-used port. "
        return
    fi

	##############################################################

	local n_nodes
	n_nodes="${SLURM_NNODES}"
	readonly n_nodes
	
	local n_cpus_per_task
	n_cpus_per_task=1
	[[ -v SLURM_CPUS_ON_NODE ]] && n_cpus_per_task="${SLURM_CPUS_ON_NODE}"
	[[ -v SLURM_CPUS_PER_TASK ]] && n_cpus_per_task="${SLURM_CPUS_PER_TASK}"
	readonly n_cpus_per_task
	
	local n_gpus_per_task
	n_gpus_per_task=0
	
	local n_tasks_per_node
	n_tasks_per_node=1
	
	if [[ 0 -ne "${w_one_task_per_node}" ]]
	then
		[[ -v SLURM_GPUS_ON_NODE ]] && n_gpus_per_task="${SLURM_GPUS_ON_NODE}"
	else
		n_gpus_per_task=1
		[[ -v SLURM_NTASKS_PER_NODE ]] && n_tasks_per_node="${SLURM_NTASKS_PER_NODE}"
	fi
	readonly n_gpus_per_task
	readonly n_tasks_per_node
	
	local n_tasks
	n_tasks=$((${n_tasks_per_node} * ${n_nodes}))
	readonly n_tasks
	
	echo "Submitting ${n_tasks_per_node} task(s) w/ ${n_gpus_per_task} GPU(s) each to ${n_nodes} node(s) ... "
	
	local srun_common_args
	srun_common_args=()
	# NOTE: These arguments will be every for each task.
	# and therefore the value of --tasks-per-node is always 1.
	# NOTE: Do not specify #GPUs here!
	# Because slurm will only assign the first one,
	# pass all to let the task to decide.
	
	srun_common_args+=($(cat << EOF | grep -v \# | tr "\n" " "
-c ${n_cpus_per_task}
# TMP-DEL:	--gpus-per-task ${n_gpus_per_task}
--overlap
# TMP-MOD:	--tasks-per-node ${n_tasks_per_node}
--tasks-per-node 1
# TMP-MOD-END
-n 1
EOF
        ))

	local launched_pids
	launched_pids=()
	local launched_task_labels
	launched_task_labels=()
		
	function echo_and_export_env_var()
	{
		local name
		name="$1"
		shift

		local value
		value="$1"
		shift
		readonly value
		
		local command
		command="$name=$value"
		readonly command
		
		eval "${command}"
		export "$name"
		echo "Set: ${command}"
	}


	function submit_task()
	{
		local node_name
		node_name="$1"
		shift
		readonly node_name
		
		local global_rank
		global_rank="$1"
		shift
		readonly global_rank
		
		local local_rank
		local_rank="$1"
		shift
		readonly local_rank
		
		echo_and_export_env_var LOCAL_RANK "${local_rank}"
		echo_and_export_env_var RANK "${global_rank}"
		
		echo "Submitting ${global_rank}/${n_tasks} as ${local_rank}/${node_name} ... "
		local srun_rank_args
		srun_rank_args=($(cat << EOF | grep -v \# | tr "\n" " "
-w${node_name}            
EOF
            ))
		readonly srun_args

		_ddp_echo_and_run \
		srun \
			"${srun_common_args[@]}" \
			"${srun_rank_args[@]}" \
			"${args[@]}" \
            &

		local srun_pid
		srun_pid=$!
		local srun_task_label
		srun_task_label="rank=${global_rank} local_rank=${local_rank} node=${node_name}"
		launched_pids+=("${srun_pid}")
		launched_task_labels+=("${srun_task_label}")
		echo "Submitted PID=${srun_pid} (${srun_task_label})"
			
	}
	local master_node_name
	master_node_name="${SLURMD_NODENAME}"
	readonly master_node_name
	
	echo_and_export_env_var	LOCAL_WORLD_SIZE "${n_tasks_per_node}"
	echo_and_export_env_var	MASTER_ADDR "${master_node_name}"
	echo_and_export_env_var	MASTER_PORT "${master_port}"
	echo_and_export_env_var	WORLD_SIZE "${n_tasks}"
	
	local nodes_names
	nodes_names=(
	"${master_node_name}"
	)
	# scontrol may be permission-denied on some clusters; fall back to expanding SLURM_NODELIST manually.
	if scontrol show hostnames "${SLURM_NODELIST}" >/dev/null 2>&1; then
		nodes_names+=(
			$(scontrol show hostnames "${SLURM_NODELIST}" | grep -v -w "${master_node_name}" | tee)
		)
	else
		nodes_names+=(
			$(python3 -c "
import re, os
nl = os.environ['SLURM_NODELIST']
# Expand 'prefix[01-03,05]' into individual hostnames
parts = re.findall(r'([^\[,]+)\[([^\]]+)\]|([^\[,]+)', nl)
hosts = []
for prefix, ranges, single in parts:
    if single:
        hosts.append(single)
    else:
        for r in ranges.split(','):
            if '-' in r:
                lo, hi = r.split('-', 1)
                width = len(lo)
                for i in range(int(lo), int(hi) + 1):
                    hosts.append(f'{prefix}{str(i).zfill(width)}')
            else:
                hosts.append(f'{prefix}{r}')
master = os.environ.get('SLURMD_NODENAME', '')
for h in hosts:
    if h != master:
        print(h)
" | tee)
		)
	fi
	readonly nodes_names

	local global_rank=0

	local node_name
	for node_name in "${nodes_names[@]}"
	do
		local locali
		for locali in $(seq "${n_tasks_per_node}")
		do
			local local_rank
			local_rank=$((${locali} - 1))
			
			submit_task "${node_name}" "${global_rank}" "${local_rank}" 
			global_rank=$(("${global_rank}" + 1))
		done
	done

	local any_failed
	any_failed=0
	local first_failed_rc
	first_failed_rc=0
	local wait_index
	for wait_index in "${!launched_pids[@]}"
	do
		local wait_pid
		wait_pid="${launched_pids[${wait_index}]}"
		local task_label
		task_label="${launched_task_labels[${wait_index}]}"
		if wait "${wait_pid}"
		then
			echo "[OK] srun completed pid=${wait_pid} (${task_label})"
		else
			local wait_rc
			wait_rc=$?
			echo "[ERROR] srun failed pid=${wait_pid} rc=${wait_rc} (${task_label})"
			any_failed=1
			if [[ 0 -eq "${first_failed_rc}" ]]
			then
				first_failed_rc="${wait_rc}"
			fi
		fi
	done

	if [[ 0 -ne "${any_failed}" ]]
	then
		echo "[ERROR] One or more distributed tasks failed."
		return "${first_failed_rc}"
	fi
}
trap "unset main" EXIT
main "$@"
