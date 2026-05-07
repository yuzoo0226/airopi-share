#!/usr/bin/bash


function _ddp_echo_and_run()
{
	echo "$@"
	"$@"
}


function _ddp_get_master_hostname()
{
    hostname
}


function _ddp_get_port_on_host()
{
    local host_name
    host_name="$1"
    shift
    readonly host_name

    local port
    port=""

    local trial
    for trial in $(seq 10)
    do
        port=$((10000 + $RANDOM % 32768))

        local w_port 
	w_port=$(netstat ap | grep -w "${port}" | tee)
        [[ -z "${w_port}" ]] && break
        port=""
    done
    echo "${port}"
}


function _ddp_get_master_ip_address()
{
	local ip_prefix
	ip_prefix="$1"
	shift
	readonly ip_prefix
    
	if [[ -z "${ip_prefix}" ]] 
	then
		_ddp_get_master_hostname
		return
	fi
	local ip_tokens
	ip_tokens=($(ip addr | grep "inet ${ip_prefix}" | head -n 1))
	readonly ip_tokens
	
	echo "${ip_tokens[1]}" | cut -d / -f 1
}

