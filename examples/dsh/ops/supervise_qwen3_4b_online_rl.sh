#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 RUN_ROOT PYTHON_BIN COMMAND [ARG ...]" >&2
  exit 2
fi

RUN_ROOT="$1"
PYTHON_BIN="$2"
shift 2
MANIFEST_WRITER="${SCRIPT_DIR}/write_run_manifest.py"
READY_FILE="${RUN_ROOT}/supervisor-ready"
WORKER_PID_FILE="${RUN_ROOT}/worker.pid"
worker_pid=""
termination_signal=""

# shellcheck disable=SC2329
forward_signal() {
  termination_signal="$1"
  if [[ -n "${worker_pid}" ]] && kill -0 "${worker_pid}" 2>/dev/null; then
    kill -"${termination_signal}" "${worker_pid}" 2>/dev/null || true
  fi
}

write_manifest() {
  "${PYTHON_BIN}" "${MANIFEST_WRITER}" --run-root "${RUN_ROOT}" "$@"
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

printf '%s\n' "$$" > "${RUN_ROOT}/pid"
write_manifest --status running --pid "$$"

"$@" &
worker_pid=$!
printf '%s\n' "${worker_pid}" > "${WORKER_PID_FILE}"
write_manifest --status running --pid "$$" --worker-pid "${worker_pid}"
printf '%s\n' "$$" > "${READY_FILE}"

if [[ -n "${termination_signal}" ]]; then
  kill -"${termination_signal}" "${worker_pid}" 2>/dev/null || true
fi

set +e
wait "${worker_pid}"
worker_exit_code=$?
if [[ -n "${termination_signal}" ]]; then
  while kill -0 "${worker_pid}" 2>/dev/null; do
    wait "${worker_pid}"
    next_exit_code=$?
    if [[ "${next_exit_code}" -ne 127 ]]; then
      worker_exit_code="${next_exit_code}"
    fi
  done
  wait "${worker_pid}" 2>/dev/null
  next_exit_code=$?
  if [[ "${next_exit_code}" -ne 127 ]]; then
    worker_exit_code="${next_exit_code}"
  fi
fi
set -e

if [[ -n "${termination_signal}" ]]; then
  write_manifest --status interrupted --pid "$$" \
    --worker-pid "${worker_pid}" --exit-code "${worker_exit_code}" \
    --termination-signal "${termination_signal}"
  case "${termination_signal}" in
    TERM) exit 143 ;;
    INT) exit 130 ;;
    *) exit 1 ;;
  esac
fi

if [[ "${worker_exit_code}" -eq 0 ]]; then
  status=completed
else
  status=failed
fi
write_manifest --status "${status}" --pid "$$" \
  --worker-pid "${worker_pid}" --exit-code "${worker_exit_code}"
exit "${worker_exit_code}"
