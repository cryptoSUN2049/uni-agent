#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
dsh_select_python
RUN_ROOT="${1:-${RUN_ROOT:-${HOME}/runs/dsh-evolution-v2-online-rl}}"

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "run root does not exist: ${RUN_ROOT}" >&2
  exit 2
fi
echo "run_root=${RUN_ROOT}"
if [[ -f "${RUN_ROOT}/run-manifest.json" ]]; then
  "${PYTHON_BIN}" - "${RUN_ROOT}/run-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ("status", "pid", "exit_code", "model_id", "uni_agent_sha", "verl_sha", "dsh_sha"):
    if key in value:
        print(f"{key}={value[key]}")
print(f"manifest_updated_at={value.get('updated_at', '')}")
print(f"dataset={value.get('dataset', {})}")
PY
fi
pid=""
if [[ -f "${RUN_ROOT}/pid" ]]; then
  pid="$(tr -d '[:space:]' < "${RUN_ROOT}/pid")"
fi
if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
  echo "process=running pid=${pid}"
  ps -p "${pid}" -o pid,ppid,stat,etime,%cpu,%mem,cmd || true
else
  echo "process=not-running pid=${pid:-unknown}"
fi
if [[ -f "${RUN_ROOT}/run.log" ]]; then
  echo "-- recent metrics/errors --"
  grep -E "Training Progress|global_step|Final validation|generate_sequences summary|run_task done|Saving checkpoint|Traceback|ERROR|segfault|No available memory" "${RUN_ROOT}/run.log" | tail -80 || true
fi
echo "-- checkpoints --"
find "${RUN_ROOT}/checkpoints" -type f -printf '%p %s bytes\\n' 2>/dev/null | sort | tail -40 || true
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "-- GPU processes --"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
fi
