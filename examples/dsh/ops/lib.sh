#!/usr/bin/env bash

# Shared, side-effect-free helpers for the human-operated DSH/VERL runbook.

DSH_OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSH_REPO_ROOT="$(cd "${DSH_OPS_DIR}/../../.." && pwd)"

dsh_select_python() {
  local venv_dir="${DSH_VENV:-${HOME}/dsh-online-rl-venv}"
  if [[ -z "${PYTHON_BIN:-}" && -x "${venv_dir}/bin/python" ]]; then
    export PYTHON_BIN="${venv_dir}/bin/python"
  fi
  : "${PYTHON_BIN:=python3}"
  if [[ -d "${venv_dir}/bin" ]]; then
    export PATH="${venv_dir}/bin:${PATH}"
  fi
  export PYTHONPATH="${DSH_REPO_ROOT}:${DSH_REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
}

dsh_require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file is missing: $1" >&2
    return 2
  fi
}

dsh_require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "required directory is missing: $1" >&2
    return 2
  fi
}

dsh_write_manifest() {
  local run_root="$1"
  shift
  "${PYTHON_BIN}" "${DSH_OPS_DIR}/write_run_manifest.py" --run-root "${run_root}" "$@"
}
