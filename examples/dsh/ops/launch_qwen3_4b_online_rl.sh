#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
dsh_select_python

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3-4B}"
DATA_ROOT="${DATA_ROOT:-${HOME}/data/dsh-evolution-v2}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_ROOT}/holdout.parquet}"
TASK_CONFIG="${TASK_CONFIG:-${DSH_REPO_ROOT}/examples/dsh/evolution_task_config_v2_fast.yaml}"
RUN_ROOT="${RUN_ROOT:-${HOME}/runs/dsh-evolution-v2-online-rl}"
PROJECT_NAME="${PROJECT_NAME:-dsh-qwen3-4b-online-rl}"
EXP_NAME="${EXP_NAME:-expanded-v2}"
DSH_SHA="${DSH_SHA:-}"
MODE="${MODE:-detach}"
if [[ "${1:-}" == "--foreground" ]]; then
  MODE=foreground
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--foreground]" >&2
  exit 2
fi
if [[ "${MODE}" != "detach" && "${MODE}" != "foreground" ]]; then
  echo "MODE must be detach or foreground" >&2
  exit 2
fi

if [[ "${MODEL_LICENSE_APPROVED:-0}" != "1" ]]; then
  echo "set MODEL_LICENSE_APPROVED=1 after reviewing the Qwen3 license" >&2
  exit 2
fi
dsh_require_dir "${MODEL_PATH}"
dsh_require_file "${MODEL_PATH}/config.json"
dsh_require_file "${MODEL_PATH}/tokenizer_config.json"
dsh_require_file "${TRAIN_FILE}"
dsh_require_file "${TEST_FILE}"
dsh_require_file "${TASK_CONFIG}"
dsh_require_file "${DATA_ROOT}/manifest.json"

# The task runner resolves `runner_python: python`; put the same venv first in
# PATH so Ray workers do not silently fall back to /usr/bin/python.
if ! python -c 'import pydantic, deepseek_harness, deepseek_harness_runtime, transfer_queue, verl' >/dev/null 2>&1; then
  echo "the selected Python environment cannot import the DSH/VERL runtime; check DSH_VENV and PATH" >&2
  exit 2
fi

if [[ -e "${RUN_ROOT}" && "${ALLOW_REUSE:-0}" != "1" ]]; then
  if [[ -f "${RUN_ROOT}/run-manifest.json" || -f "${RUN_ROOT}/run.log" || -f "${RUN_ROOT}/pid" ]]; then
    echo "run root already contains an experiment: ${RUN_ROOT} (choose another RUN_ROOT or set ALLOW_REUSE=1)" >&2
    exit 2
  fi
fi
mkdir -p "${RUN_ROOT}"

# This allocator option conflicts with VERL's memory-pool path on the tested
# v0.9/torch 2.10 stack. Unset it explicitly rather than inheriting a shell
# setting from an unrelated job.
unset PYTORCH_CUDA_ALLOC_CONF

export MODEL_ID MODEL_PATH TRAIN_FILE TEST_FILE TASK_CONFIG RUN_ROOT PROJECT_NAME EXP_NAME
export MODEL_LICENSE_APPROVED LOW_VRAM=1
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-7168}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-4}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export TEST_FREQ="${TEST_FREQ:-4}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-16}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
export TOOL_PARSER="${TOOL_PARSER:-hermes}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.15}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
export ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-False}"
export ROLLOUT_CPU_OFFLOAD_GB="${ROLLOUT_CPU_OFFLOAD_GB:-8}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
export VAL_ONLY="${VAL_ONLY:-False}"
export PYTHON_BIN

LAUNCHER="${DSH_REPO_ROOT}/examples/dsh/train_qwen3_4b_online_rl.sh"
dsh_require_file "${LAUNCHER}"
COMMAND_FILE="${RUN_ROOT}/command.txt"
printf '%q ' env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH}" "PYTHON_BIN=${PYTHON_BIN}" \
  "MODEL_LICENSE_APPROVED=${MODEL_LICENSE_APPROVED}" "LOW_VRAM=${LOW_VRAM}" \
  "MODEL_ID=${MODEL_ID}" "MODEL_PATH=${MODEL_PATH}" "TRAIN_FILE=${TRAIN_FILE}" \
  "TEST_FILE=${TEST_FILE}" "TASK_CONFIG=${TASK_CONFIG}" "RUN_ROOT=${RUN_ROOT}" \
  "PROJECT_NAME=${PROJECT_NAME}" "EXP_NAME=${EXP_NAME}" "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}" \
  "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}" "PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  "ROLLOUT_N=${ROLLOUT_N}" "ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS}" \
  "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}" \
  "SAVE_FREQ=${SAVE_FREQ}" "TEST_FREQ=${TEST_FREQ}" "TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES}" \
  "VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}" "DATA_SHUFFLE=${DATA_SHUFFLE}" \
  "TOOL_PARSER=${TOOL_PARSER}" "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}" \
  "ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER}" "ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE}" \
  "ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON}" "ROLLOUT_CPU_OFFLOAD_GB=${ROLLOUT_CPU_OFFLOAD_GB}" \
  "ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD}" "ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD}" \
  "RESUME_MODE=${RESUME_MODE}" "RESUME_FROM_PATH=${RESUME_FROM_PATH}" "VAL_ONLY=${VAL_ONLY}" \
  "PYTORCH_CUDA_ALLOC_CONF=" bash "${LAUNCHER}" > "${COMMAND_FILE}"
printf '\n' >> "${COMMAND_FILE}"

MANIFEST_ARGS=(
  --command-file "${COMMAND_FILE}"
  --model-path "${MODEL_PATH}"
  --model-id "${MODEL_ID}"
  --train-file "${TRAIN_FILE}"
  --holdout-file "${TEST_FILE}"
  --task-config "${TASK_CONFIG}"
  --dataset-manifest "${DATA_ROOT}/manifest.json"
  --repo-root "${DSH_REPO_ROOT}"
  --verl-root "${DSH_REPO_ROOT}/verl"
)
if [[ -n "${DSH_SHA}" ]]; then
  MANIFEST_ARGS+=(--dsh-sha "${DSH_SHA}")
fi
dsh_write_manifest "${RUN_ROOT}" --status prepared "${MANIFEST_ARGS[@]}"

run_command=(env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH}" "PYTHON_BIN=${PYTHON_BIN}" \
  "MODEL_LICENSE_APPROVED=${MODEL_LICENSE_APPROVED}" "LOW_VRAM=${LOW_VRAM}" \
  "MODEL_ID=${MODEL_ID}" "MODEL_PATH=${MODEL_PATH}" "TRAIN_FILE=${TRAIN_FILE}" \
  "TEST_FILE=${TEST_FILE}" "TASK_CONFIG=${TASK_CONFIG}" "RUN_ROOT=${RUN_ROOT}" \
  "PROJECT_NAME=${PROJECT_NAME}" "EXP_NAME=${EXP_NAME}" "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}" \
  "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}" "PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  "ROLLOUT_N=${ROLLOUT_N}" "ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS}" \
  "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}" \
  "SAVE_FREQ=${SAVE_FREQ}" "TEST_FREQ=${TEST_FREQ}" "TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES}" \
  "VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}" "DATA_SHUFFLE=${DATA_SHUFFLE}" \
  "TOOL_PARSER=${TOOL_PARSER}" "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}" \
  "ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER}" "ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE}" \
  "ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON}" "ROLLOUT_CPU_OFFLOAD_GB=${ROLLOUT_CPU_OFFLOAD_GB}" \
  "ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD}" "ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD}" \
  "RESUME_MODE=${RESUME_MODE}" "RESUME_FROM_PATH=${RESUME_FROM_PATH}" "VAL_ONLY=${VAL_ONLY}" \
  "PYTORCH_CUDA_ALLOC_CONF=" bash "${LAUNCHER}")

if [[ "${MODE}" == "foreground" ]]; then
  dsh_write_manifest "${RUN_ROOT}" --status running --pid "$$" "${MANIFEST_ARGS[@]}"
  set +e
  "${run_command[@]}" "$@"
  code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then status=completed; else status=failed; fi
  dsh_write_manifest "${RUN_ROOT}" --status "${status}" --pid "$$" --exit-code "${code}" "${MANIFEST_ARGS[@]}"
  exit "$code"
fi

nohup "${run_command[@]}" > "${RUN_ROOT}/run.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "${RUN_ROOT}/pid"
dsh_write_manifest "${RUN_ROOT}" --status running --pid "$pid" "${MANIFEST_ARGS[@]}"
printf 'started pid=%s\nrun_root=%s\n' "$pid" "${RUN_ROOT}"
