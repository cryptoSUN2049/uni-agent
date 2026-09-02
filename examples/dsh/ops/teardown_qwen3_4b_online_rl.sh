#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
dsh_select_python
RUN_ROOT="${1:-${RUN_ROOT:-}}"
if [[ -z "${RUN_ROOT}" ]]; then
  echo "usage: $0 /absolute/path/to/run-root [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]]; then DRY_RUN=1; fi
PID=""
if [[ -f "${RUN_ROOT}/pid" ]]; then PID="$(tr -d '[:space:]' < "${RUN_ROOT}/pid")"; fi
if [[ -z "${PID}" ]]; then
  echo "no pid file in ${RUN_ROOT}; nothing to stop"
  exit 0
fi
if ! [[ "${PID}" =~ ^[0-9]+$ ]]; then
  echo "invalid pid file: ${RUN_ROOT}/pid" >&2
  exit 2
fi
if [[ ! -r "/proc/${PID}/cmdline" ]]; then
  echo "pid=${PID} is not running"
else
  CMDLINE="$(tr '\0' ' ' < "/proc/${PID}/cmdline")"
  if [[ "${CMDLINE}" != *"train_qwen3_4b_online_rl.sh"* \
    && "${CMDLINE}" != *"supervise_qwen3_4b_online_rl.sh"* \
    && "${CMDLINE}" != *"parallel_infer_verl.py"* ]]; then
    echo "refusing to stop pid=${PID}: command is not a known DSH run" >&2
    echo "${CMDLINE}" >&2
    exit 2
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "would send TERM to pid=${PID}"
    exit 0
  fi
  kill -TERM "${PID}" 2>/dev/null || true
  for _ in {1..30}; do
    if ! kill -0 "${PID}" 2>/dev/null; then break; fi
    sleep 1
  done
  if kill -0 "${PID}" 2>/dev/null; then
    echo "pid=${PID} did not exit after 30s; sending KILL" >&2
    kill -KILL "${PID}" 2>/dev/null || true
  fi
fi
if [[ "${RAY_STOP:-0}" == "1" ]]; then
  echo "stopping the local Ray cluster because RAY_STOP=1"
  "${PYTHON_BIN}" -m ray stop --force || true
else
  echo "Ray was not stopped; set RAY_STOP=1 after confirming this host is dedicated"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_ROOT}/teardown_at"
dsh_write_manifest "${RUN_ROOT}" --status torn_down --pid "${PID}" --command-file "${RUN_ROOT}/command.txt"
echo "teardown recorded in ${RUN_ROOT}/teardown_at"
