#!/usr/bin/env bash
# 24/7 supervisor: restart the trainer on crash with restart-storm protection.
# Bails out if 3 restarts happen within 600s (mirrors the local watchdog).
set -uo pipefail

# v0.6.1: history + crash reason live on the persistent volume so pod
# restart/preemption doesn't reset the restart-storm counter and lose
# the diagnostic on the next boot.
PERSIST_DIR="${PERSIST_DIR:-/workspace/persistent}"
HISTORY_FILE="${PERSIST_DIR}/restart-history/history"
CRASH_REASON_FILE="${PERSIST_DIR}/CRASH_REASON.txt"
WINDOW_SECONDS=600
MAX_RESTARTS=3
BACKOFF_SECONDS=30

mkdir -p "$(dirname "${HISTORY_FILE}")"
touch "${HISTORY_FILE}"

ENTRYPOINT="${ENTRYPOINT:-/usr/local/bin/entrypoint.sh}"

log() { printf '[supervisor %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

child_pid=0
shutdown=0

forward_signal() {
    local sig="$1"
    shutdown=1
    log "received ${sig}, forwarding to child pid=${child_pid}"
    if [[ "${child_pid}" -gt 0 ]]; then
        kill -"${sig}" "${child_pid}" 2>/dev/null || true
    fi
}

trap 'forward_signal TERM' SIGTERM
trap 'forward_signal INT'  SIGINT

prune_history() {
    local now="$1"
    local cutoff=$(( now - WINDOW_SECONDS ))
    if [[ -s "${HISTORY_FILE}" ]]; then
        awk -v c="${cutoff}" '$1 >= c' "${HISTORY_FILE}" > "${HISTORY_FILE}.tmp" && \
            mv "${HISTORY_FILE}.tmp" "${HISTORY_FILE}"
    fi
}

while true; do
    now="$(date +%s)"
    prune_history "${now}"
    recent_count="$(wc -l < "${HISTORY_FILE}" | tr -d ' ')"

    if [[ "${recent_count}" -ge "${MAX_RESTARTS}" ]]; then
        log "FATAL: ${recent_count} restarts within ${WINDOW_SECONDS}s — giving up."
        {
            echo "Restart-storm: ${recent_count} restarts within ${WINDOW_SECONDS}s."
            echo "Last restart timestamps (epoch seconds):"
            cat "${HISTORY_FILE}"
            echo
            echo "Investigate trainer logs in this pod, then redeploy."
        } > "${CRASH_REASON_FILE}"
        exit 1
    fi

    log "launching trainer (restart count in window: ${recent_count}/${MAX_RESTARTS})"
    "${ENTRYPOINT}" &
    child_pid=$!
    log "child pid=${child_pid}"

    set +e
    wait "${child_pid}"
    rc=$?
    set -e
    child_pid=0

    if [[ "${shutdown}" -eq 1 ]]; then
        log "supervisor shutting down (child rc=${rc})"
        exit "${rc}"
    fi

    log "child exited rc=${rc}"
    echo "$(date +%s) ${rc}" >> "${HISTORY_FILE}"

    if [[ "${rc}" -eq 0 ]]; then
        log "trainer exited cleanly — stopping supervisor."
        exit 0
    fi

    log "backing off ${BACKOFF_SECONDS}s before restart..."
    sleep "${BACKOFF_SECONDS}"
done
