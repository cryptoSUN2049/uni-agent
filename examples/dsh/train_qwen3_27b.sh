#!/usr/bin/env bash
# Thin DSH training launcher: reuse Uni-Agent's maintained VERL recipe.
#
# This script only selects the DSH task config and model/data defaults. The
# underlying recipe owns Ray, Gateway, optimizer, checkpoint, and distributed
# training flags; it must be run on a prepared GPU host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3-27B}"
TRAIN_FILE="${TRAIN_FILE:-${HOME}/data/dsh/verl-online-rl-seeds.parquet}"
TEST_FILE="${TEST_FILE:-${HOME}/data/dsh/verl-online-rl-held-out.parquet}"
TASK_CONFIG="${TASK_CONFIG:-examples/dsh/task_config.yaml}"

if [[ ! -f "${TASK_CONFIG}" ]]; then
    echo "missing DSH Task Config: ${TASK_CONFIG}" >&2
    exit 2
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "missing model path: ${MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" ]]; then
    echo "TRAIN_FILE and TEST_FILE must point at prepared Parquet releases" >&2
    exit 2
fi

exec env \
    MODEL_PATH="${MODEL_PATH}" \
    TRAIN_FILE="${TRAIN_FILE}" \
    TEST_FILE="${TEST_FILE}" \
    TASK_CONFIG="${TASK_CONFIG}" \
    bash examples/quickstart/training/train_qwen3_moe.sh
