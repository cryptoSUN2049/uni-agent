#!/usr/bin/env bash
# DSH Agent + VERL V1 sync GRPO/LoRA proof | Qwen3-4B | one update
#
# This is a deliberately small, dense-model recipe. It does not reuse the
# Qwen3-27B MoE launcher: that launcher has a different topology and model
# engine. Run from any directory; paths passed to Ray are made absolute where
# this script owns them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_ROOT="${REPO_ROOT}/verl"
cd "${REPO_ROOT}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_PATH="${MODEL_PATH:-}"
TRAIN_FILE="${TRAIN_FILE:-${HOME}/data/dsh/verl-online-rl-seeds.parquet}"
TEST_FILE="${TEST_FILE:-${HOME}/data/dsh/verl-online-rl-holdout.parquet}"
TASK_CONFIG="${TASK_CONFIG:-${REPO_ROOT}/examples/dsh/task_config.yaml}"
RUN_ROOT="${RUN_ROOT:-${HOME}/runs/dsh-qwen3-4b-online-rl}"
PROJECT_NAME="${PROJECT_NAME:-dsh-qwen3-4b-online-rl}"
EXP_NAME="${EXP_NAME:-one-update}"
CKPTS_DIR="${CKPTS_DIR:-${RUN_ROOT}/checkpoints/${PROJECT_NAME}/${EXP_NAME}}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${RUN_ROOT}/agent-logs/${PROJECT_NAME}/${EXP_NAME}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${RUN_ROOT}/rollouts/${PROJECT_NAME}/${EXP_NAME}}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${RUN_ROOT}/validation/${PROJECT_NAME}/${EXP_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_N="${ROLLOUT_N:-2}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
LOW_VRAM="${LOW_VRAM:-0}"
if [[ "${LOW_VRAM}" == "1" ]]; then
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
  LORA_RANK="${LORA_RANK:-4}"
  LORA_ALPHA="${LORA_ALPHA:-8}"
  ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
  ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
  ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
  ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-False}"
  ROLLOUT_CPU_OFFLOAD_GB="${ROLLOUT_CPU_OFFLOAD_GB:-8}"
  SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-True}"
else
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
  LORA_RANK="${LORA_RANK:-32}"
  LORA_ALPHA="${LORA_ALPHA:-32}"
  ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
  ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
  ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
  ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-True}"
  ROLLOUT_CPU_OFFLOAD_GB="${ROLLOUT_CPU_OFFLOAD_GB:-0}"
  SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-False}"
fi
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:-1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1}"
DATA_SHUFFLE="${DATA_SHUFFLE:-False}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-bfloat16}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-False}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-['model','optimizer','extra']}"
CHECKPOINT_LOAD_CONTENTS="${CHECKPOINT_LOAD_CONTENTS:-['model','optimizer','extra']}"
VAL_ONLY="${VAL_ONLY:-False}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

if [[ "${PRINT_COMMAND:-0}" == "1" ]]; then
  # PRINT_COMMAND is a dependency-free config gate used by CI and operators to
  # inspect the exact Hydra overrides without starting Ray or touching data.
  : "${MODEL_PATH:=/models/Qwen3-4B-snapshot}"
  COMMAND=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo
    "trainer.use_v1=True"
    "trainer.v1.trainer_mode=sync"
    "trainer.v1.sampler.sync_refill_failed_groups=True"
    "transfer_queue.enable=True"
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=False"
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    "data.prompt_key=prompt"
    "data.return_raw_chat=True"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
    "data.val_max_samples=${VAL_MAX_SAMPLES}"
    "data.shuffle=${DATA_SHUFFLE}"
    "++data.apply_chat_template_kwargs.enable_thinking=False"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "++actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
    "actor_rollout_ref.actor.fsdp_config.model_dtype=${ACTOR_MODEL_DTYPE}"
    "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
    "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
    "actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}"
    "actor_rollout_ref.rollout.layered_summon=${ROLLOUT_LAYERED_SUMMON}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.enable_prefix_caching=${ROLLOUT_ENABLE_PREFIX_CACHING}"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb=${ROLLOUT_CPU_OFFLOAD_GB}"
    "actor_rollout_ref.actor.checkpoint.save_contents=${CHECKPOINT_SAVE_CONTENTS}"
    "actor_rollout_ref.actor.checkpoint.load_contents=${CHECKPOINT_LOAD_CONTENTS}"
    "++actor_rollout_ref.actor.checkpoint.save_lora_only=${SAVE_LORA_ONLY}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER}"
    "++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${MODEL_ID}"
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True"
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.require_reward_post=True"
    "++actor_rollout_ref.rollout.custom.agent_framework.fail_on_rollout_error=True"
    "++actor_rollout_ref.rollout.custom.agent_framework.require_finished_episode=True"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.val_only=${VAL_ONLY}"
  )
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must point to a pinned local ${MODEL_ID} snapshot" >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/tokenizer_config.json" ]]; then
  echo "MODEL_PATH must contain config.json and tokenizer_config.json: ${MODEL_PATH}" >&2
  exit 2
fi
for required_path in "${TRAIN_FILE}" "${TEST_FILE}" "${TASK_CONFIG}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "required file is missing: ${required_path}" >&2
    exit 2
  fi
done
if [[ ! -d "${VERL_ROOT}/verl" ]]; then
  echo "VERL checkout is missing: ${VERL_ROOT}/verl" >&2
  exit 2
fi
if [[ "${MODEL_LICENSE_APPROVED:-0}" != "1" ]]; then
  echo "set MODEL_LICENSE_APPROVED=1 after reviewing the model license" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" - "${MODEL_PATH}" "${MODEL_ID}" <<'PY'
import json
import sys

model_path = sys.argv[1]
model_id = sys.argv[2]
with open(f"{model_path}/config.json", encoding="utf-8") as stream:
    config = json.load(stream)
if config.get("model_type") != "qwen3":
    raise SystemExit("MODEL_PATH must use a Qwen3 architecture")
if any(key in config for key in ("num_local_experts", "num_experts", "moe_intermediate_size")):
    raise SystemExit("MODEL_PATH must be a dense Qwen3 checkpoint, not a MoE checkpoint")
if model_id == "Qwen/Qwen3-4B" and (
    config.get("num_hidden_layers") != 36 or config.get("hidden_size") != 2560
):
    raise SystemExit("MODEL_PATH is not the expected Qwen/Qwen3-4B dense architecture")
try:
    import deepseek_harness  # noqa: F401
    import deepseek_harness_runtime  # noqa: F401
    import transfer_queue  # noqa: F401
    import verl  # noqa: F401
except ImportError as exc:
    raise SystemExit(f"online-RL runtime dependency is unavailable: {exc}") from exc
PY

mkdir -p "${CKPTS_DIR}" "${AGENT_LOG_DIR}" "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}"

COMMAND=(
  "${PYTHON_BIN}" -m verl.trainer.main_ppo
  trainer.use_v1=True
  trainer.v1.trainer_mode=sync
  trainer.v1.sampler.sync_refill_failed_groups=True
  transfer_queue.enable=True
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="${TRAIN_FILE}"
  data.val_files="${TEST_FILE}"
  data.prompt_key=prompt
  data.return_raw_chat=True
  data.filter_overlong_prompts=True
  data.truncation=error
  data.dataloader_num_workers=0
  data.shuffle="${DATA_SHUFFLE}"
  data.train_max_samples="${TRAIN_MAX_SAMPLES}"
  data.val_max_samples="${VAL_MAX_SAMPLES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  ++data.apply_chat_template_kwargs.enable_thinking=False
  actor_rollout_ref.model.path="${MODEL_PATH}"
  ++actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}"
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.lora_rank="${LORA_RANK}"
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.actor.optim.lr=1e-5
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}"
  actor_rollout_ref.actor.checkpoint.load_contents="${CHECKPOINT_LOAD_CONTENTS}"
  ++actor_rollout_ref.actor.checkpoint.save_lora_only="${SAVE_LORA_ONLY}"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon="${ROLLOUT_LAYERED_SUMMON}"
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}"
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb="${ROLLOUT_CPU_OFFLOAD_GB}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}"
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}"
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}"
  actor_rollout_ref.rollout.agent.num_workers=1
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}"
  ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}"
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}"
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}"
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${MODEL_ID}"
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.require_reward_post=True
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=True
  ++actor_rollout_ref.rollout.custom.agent_framework.fail_on_rollout_error=True
  ++actor_rollout_ref.rollout.custom.agent_framework.require_finished_episode=True
  trainer.logger="['console','file']"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXP_NAME}"
  trainer.val_before_train=True
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.total_epochs=1
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.default_local_dir="${CKPTS_DIR}"
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
  trainer.nnodes="${NNODES}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
)

if [[ "${RESUME_MODE}" == "resume_path" ]]; then
  if [[ -z "${RESUME_FROM_PATH}" ]]; then
    echo "RESUME_FROM_PATH is required when RESUME_MODE=resume_path" >&2
    exit 2
  fi
  COMMAND+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
elif [[ "${RESUME_MODE}" != "disable" && "${RESUME_MODE}" != "auto" ]]; then
  echo "RESUME_MODE must be disable, auto, or resume_path" >&2
  exit 2
fi
COMMAND+=("trainer.resume_mode=${RESUME_MODE}" "trainer.val_only=${VAL_ONLY}")

exec "${COMMAND[@]}" "$@"
