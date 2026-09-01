#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "usage: MODEL_LICENSE_APPROVED=1 $0 /absolute/path/to/global_step_N [--foreground]" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "checkpoint directory does not exist: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
export CHECKPOINT_PATH
export RUN_ROOT="${RUN_ROOT:-${HOME}/runs/dsh-evolution-v2-reload-$(date -u +%Y%m%dT%H%M%SZ)}"
export EXP_NAME="${EXP_NAME:-reload-$(basename "${CHECKPOINT_PATH}")}"
export RESUME_MODE=resume_path
export RESUME_FROM_PATH="${CHECKPOINT_PATH}"
export VAL_ONLY=True
export ROLLOUT_N="${ROLLOUT_N:-1}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export TEST_FREQ=1
exec "${SCRIPT_DIR}/launch_qwen3_4b_online_rl.sh" "${@:2}"
