# Official DSH → Uni-Agent → VERL adapter

This directory is the smallest runnable S0 surface for the official DSH path. The `dsh` Agent launches the DSH Python SDK inside the same Uni-Agent Sandbox; the SDK sends model requests to the session-scoped Gateway, while `dsh_architecture` invokes a trusted verifier against the resulting trace and task envelope. `run_task(..., report_reward=True)` posts the scalar reward and a compact DSH receipt reference back to the Gateway, so VERL can keep the token trajectory and semantic lineage together.

The included `verifier.py` is a deterministic smoke fixture. It checks exact trace and envelope hashes and awards one only when `metadata.expected_response_sha256` matches the response. Replace it with a benchmark-specific verifier before any training release; never use a non-fresh or actor-supplied score as online reward.

The first training proof uses the official dense `Qwen/Qwen3-4B` checkpoint as the 3B-class target. The public Qwen release does not provide a `Qwen/Qwen3-3B` repository; set `MODEL_ID` and `MODEL_PATH` to a compatible private or mirrored 3B checkpoint when one is available. The model-size choice is an integration gate, not a claim that the policy already understands DSH or can perform self-evolution. The launcher fails closed on a missing DSH runtime, an unacknowledged reward POST, or any failed rollout session.

## Seed metadata

Rows produced from a released DSH task seed must carry the immutable identity fields below in `extra_info.tools_kwargs.task.metadata`:

```json
{
  "task_id": "dsh/architecture/intro",
  "task_version": "1",
  "split": "train",
  "environment_digest": "sha256:<64 lowercase hex digits>",
  "verifier_id": "dsh-fixture-verifier",
  "verifier_version": "1",
  "verifier_code_digest": "sha256:<64 lowercase hex digits>",
  "expected_response_sha256": "sha256:<64 lowercase hex digits>"
}
```

`verifier_command`, Sandbox controls, model credentials, and the short first-proof generation budget stay in the operator-owned Task Config. The checked-in config fixes `max_total_tokens=1024` and `max_tokens_per_turn=1024`; a sample row cannot replace them.

## Importing the M3 dual-backend release

The data-layer worktree's `dsh.dual-backend-training-release.v1` is a provider
projection, not a second task authority. After the release is marked
`eligible`, this adapter verifies the manifest hash and wraps
`verl-online-rl-seeds.jsonl` with Uni-Agent's `tools_kwargs.task` routing field:

```sh
python examples/dsh/prepare_dataset.py \
  --release-dir /absolute/path/to/dsh-dual-release \
  --output-dir /absolute/path/to/uni-agent-dsh-parquet \
  --verifier-code-digest sha256:<64 lowercase hex digits>
```

`held_out` is mapped to the official DSH `holdout` split inside the task
metadata, while the original row remains unchanged for VERL consumers. The
optional code-digest flag is required until the data-layer task registry emits
`verifier_code_digest`; it is an operator pin, not a value inferred from a
row. `analysis_only` releases are rejected. Rows with source tool schemas keep
those schemas as provenance; the selected DSH profile must expose the same
tools, and the verifier must compare the trace request header before awarding
fresh reward.

## S0 smoke

Install the matching DSH SDK/runtime in the Python environment used by the local Sandbox, then run the VERL inference entrypoint from this repository root. Use a local, pinned Qwen3-4B snapshot and the Hermes parser. The launcher disables Qwen3 thinking mode for this first short proof so the response-token budget remains bounded:

```sh
python examples/inference/parallel_infer_verl.py \
  --data-path /absolute/path/to/dsh.parquet \
  --model-path /absolute/path/to/Qwen3-4B \
  --task-config examples/dsh/task_config.yaml \
  --limit 1 --n 1 \
  --nnodes 1 --n-gpus-per-node 1 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 --tool-parser hermes \
  --gateway-count 1 --concurrency 1 --require-reward-post
```

This exercises Gateway token capture, the DSH SDK subprocess, the fresh verifier, and TransferQueue materialization. It is an integration smoke, not evidence of a parameter update or capability improvement.

## One-update online RL proof

The dedicated dense recipe uses VERL V1 sync training, two rollouts per prompt for GRPO, LoRA, and one training step. It does not reuse `train_qwen3_27b.sh`, which targets a different Qwen3 MoE/Megatron topology. Set `LOW_VRAM=1` for the tested 24 GB RTX 4090 profile; the profile uses BF16 actor weights, FSDP parameter offload, vLLM eager execution, an 8 GB CPU KV offload budget, short context, LoRA rank 4, and a LoRA-only checkpoint.

Before launching, prepare an `eligible` DSH seed Parquet and a held-out Parquet, install one consistent VERL/vLLM environment, and install both DSH SDK/runtime wheels in that same environment. This branch pins the VERL submodule at `483b8a009ba3a97563edee3a19887e4862b8094a` (`v0.9.0`), which is the tested runtime. The tested Linux environment used torch 2.10.0+cu128, vLLM 0.18.1, Ray 2.58.0, Transformers 4.57.6, TensorDict 0.10.0, and TransferQueue 0.1.11.dev0 from TransferQueue commit `06ad9c9022d49bb4791289edb3b50951fe665e9f`. The v0.9 package metadata asks for Transformers 5.5.3+, while vLLM 0.18.1 requires Transformers below 5; this tested lane therefore installs the explicit versions first and installs the local VERL and Uni-Agent sources with `--no-deps`. Treat that `--no-deps` choice as a pinned compatibility exception, and do not run `pip install -e verl` without it. The v0.9 checkout has no `manage_envs.py`; do not use the newer `verl/manage_envs.py sync` lock for this lane because it resolves a different CUDA/Torch/vLLM stack. Review the model license and set the explicit acknowledgement:

```sh
MODEL_LICENSE_APPROVED=1 \
LOW_VRAM=1 \
MODEL_ID=Qwen/Qwen3-4B \
MODEL_PATH=/absolute/path/to/Qwen3-4B \
TRAIN_FILE=/absolute/path/to/dsh-train.parquet \
TEST_FILE=/absolute/path/to/dsh-holdout.parquet \
bash examples/dsh/train_qwen3_4b_online_rl.sh
```

The script requires `config.json` and `tokenizer_config.json`, checks the Qwen3 dense architecture (and the official Qwen3-4B dimensions when the default model ID is used), and fails before Ray starts when the DSH runtime or TransferQueue is unavailable. It writes agent traces, rollout/validation dumps, and a VERL checkpoint under `RUN_ROOT`. A successful run must show non-empty `response_ids`/`response_mask`, a fresh reward acknowledgement, `trainer/global_step=1`, and a checkpoint containing the actor state.

The observed 4090 proof used torch 2.10.0+cu128, vLLM 0.18.1, Ray 2.58, Transformers 4.57.6, TensorDict 0.10.0, and the VERL source checkout at the pinned v0.9-compatible commit. The active DSH runtime came from the Linux source build, and the run used the Qwen3-4B revision recorded in the DSH worktree evidence. Keep these pins together; mixing the VERL source resolver with an older vLLM stack is not supported by this path.

The first bounded run completed one actor update and wrote readable LoRA model, optimizer, and extra-state files. Its DSH trajectories each had `finished=true`, five response tokens with a mask sum of five, and a fresh verifier receipt with `event_count=16`. The deterministic smoke verifier returned reward 1 for every sample, so GRPO advantages, policy loss, and gradient norm were all zero. This is a plumbing and checkpoint-reload proof, not evidence of policy improvement.

The checked-in Task Config selects the official `sdk-minimal` DSH profile for this bounded proof. This profile keeps the model-visible prompt below the trajectory capacity; a later full-`sdk` capability run must increase `MAX_PROMPT_LENGTH` after measuring its tool schema with the selected tokenizer.

To exercise checkpoint reload without another optimizer step, point the same launcher at the saved `global_step_1` directory and run validation only:

```sh
MODEL_LICENSE_APPROVED=1 \
LOW_VRAM=1 \
MODEL_ID=Qwen/Qwen3-4B \
MODEL_PATH=/absolute/path/to/Qwen3-4B \
TRAIN_FILE=/absolute/path/to/dsh-train.parquet \
TEST_FILE=/absolute/path/to/dsh-holdout.parquet \
RESUME_MODE=resume_path \
RESUME_FROM_PATH=/absolute/path/to/checkpoints/global_step_1 \
VAL_ONLY=True \
bash examples/dsh/train_qwen3_4b_online_rl.sh
```

This proves reload and held-out execution; it does not by itself prove capability improvement. Record the model/tokenizer/template, DSH runtime, verifier, Uni-Agent, VERL, and dependency revisions with the run artifacts before comparing against the frozen base.
