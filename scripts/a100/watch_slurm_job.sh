#!/usr/bin/env bash
# =============================================================================
#  Watch a Slurm job and say something the moment it stops being healthy.
#
#      scripts/a100/watch_slurm_job.sh <job_id> [interval_s]
#
#  A training job that dies on the cluster is quiet about it. squeue simply stops
#  listing the job, which looks identical to "finished successfully", and the
#  reason only shows up in sacct or at the end of the .out file. This prints one
#  line per state change and a clear verdict when the job leaves the queue, so a
#  failure is never mistaken for completion.
#
#  Emits on stdout, one event per line, and exits when the job reaches a terminal
#  state. Intended to be run under a monitor that turns each line into a
#  notification.
# =============================================================================
set -uo pipefail

JOB="${1:?usage: watch_slurm_job.sh <job_id> [interval_s]}"
INTERVAL="${2:-60}"

SSH_TARGET="${A100_SSH:-yano21@150.69.197.6}"
remote() { ssh -o BatchMode=yes -o ConnectTimeout=15 "${SSH_TARGET}" "$1" 2>/dev/null; }

# Terminal states Slurm can end a job in. COMPLETED is the only good one; the
# rest each need a different response, so name them rather than lumping them
# into "not running".
is_terminal() {
    case "$1" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) return 0 ;;
        *) return 1 ;;
    esac
}

# Ask Slurm where the job writes, rather than guessing. Globbing for
# slurm-<id>.out under the home directory misses it whenever the job was
# submitted from a subdirectory -- and a watcher that cannot find the output
# reports a healthy job as stalled, which is worse than not watching at all.
OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${SSH_TARGET}" \
      "scontrol show job ${JOB} 2>/dev/null | sed -n 's/.*StdOut=//p' | head -1" 2>/dev/null)
if [ -z "${OUT}" ]; then
    OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${SSH_TARGET}" \
          "find ~ -maxdepth 4 -name 'slurm-${JOB}.out' 2>/dev/null | head -1" 2>/dev/null)
fi
[ -n "${OUT}" ] && echo "output: ${OUT}" || echo "WARNING: cannot locate the job output; stall detection is off"

last_state=""
last_step=""
missing=0

echo "watching job ${JOB} on ${SSH_TARGET} every ${INTERVAL}s"

while true; do
    state=$(remote "squeue -h -j ${JOB} -o %T" | head -1)

    if [ -z "${state}" ]; then
        # Gone from the queue: ask sacct what actually happened. Retry once --
        # sacct can lag squeue by a few seconds.
        missing=$((missing + 1))
        acct=$(remote "sacct -n -X -j ${JOB} -o State%20,ExitCode,Elapsed,MaxRSS" | head -1)
        final=$(echo "${acct}" | awk '{print $1}')
        if [ -z "${final}" ] && [ "${missing}" -lt 3 ]; then
            sleep 5
            continue
        fi
        if [ "${final}" = "COMPLETED" ]; then
            echo "job ${JOB} COMPLETED  ${acct}"
        else
            echo "job ${JOB} ENDED ABNORMALLY: ${final:-unknown}  ${acct}"
            # The last lines of the job output usually name the cause.
            [ -n "${OUT}" ] && remote "tail -15 '${OUT}'" | sed 's/^/    | /'
        fi
        exit 0
    fi

    missing=0
    if [ "${state}" != "${last_state}" ]; then
        echo "job ${JOB} state: ${state}"
        last_state="${state}"
    fi

    # While it runs, surface progress so a silent stall is visible too: a job
    # that is RUNNING but has not written a step in a long while is as broken as
    # one that exited.
    # tqdm switches from "996it" to "1.61kit" past a thousand, so a pattern that
    # only matches the first form stops updating and every later poll looks like
    # no progress -- the watcher would call a healthy job stalled from step 1000
    # onwards.
    if [ "${state}" = "RUNNING" ] && [ -n "${OUT}" ]; then
        step=$(remote "grep -aoE 'Progress on: [0-9.]+k?it/[0-9.]+kit rate:[0-9.]+s/it' '${OUT}' 2>/dev/null | tail -1")
        if [ -n "${step}" ] && [ "${step}" != "${last_step}" ]; then
            echo "job ${JOB} ${step}"
            last_step="${step}"
            stalled=0
        else
            stalled=$((${stalled:-0} + 1))
            # INTERVAL * 10 with no new step
            if [ "${stalled}" -eq 10 ]; then
                echo "job ${JOB} STALLED: no new step for $((INTERVAL * 10))s while RUNNING"
            fi
        fi
    fi

    sleep "${INTERVAL}"
done
