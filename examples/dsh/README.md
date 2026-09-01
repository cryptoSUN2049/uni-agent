# Official DSH → Uni-Agent → VERL adapter

This directory is the smallest runnable S0 surface for the official DSH path. The `dsh` Agent launches the DSH Python SDK inside the same Uni-Agent Sandbox; the SDK sends model requests to the session-scoped Gateway, while `dsh_architecture` invokes a trusted verifier against the resulting trace and task envelope. `run_task(..., report_reward=True)` posts the scalar reward and a compact DSH receipt reference back to the Gateway, so VERL can keep the token trajectory and semantic lineage together.

The included `verifier.py` is a deterministic smoke fixture. It checks exact trace and envelope hashes and awards one only when `metadata.expected_response_sha256` matches the response. Replace it with a benchmark-specific verifier before any training release; never use a non-fresh or actor-supplied score as online reward.

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

`verifier_command`, Sandbox controls, and model credentials stay in the operator-owned Task Config. A sample row cannot replace them.

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

Install the matching DSH SDK/runtime in the Python environment used by the local Sandbox, then run the existing VERL inference entrypoint from this repository root:

```sh
python examples/inference/parallel_infer_verl.py \
  --data-path /absolute/path/to/dsh.parquet \
  --model-path /absolute/path/to/Qwen3-27B \
  --task-config examples/dsh/task_config.yaml \
  --limit 1 --n 1
```

This exercises Gateway token capture, the DSH SDK subprocess, the fresh verifier, and TransferQueue materialization. It is an integration smoke, not evidence of a parameter update or capability improvement. For a GPU training run, use the existing VERL recipe with `TASK_CONFIG=examples/dsh/task_config.yaml` and pin the model, tokenizer, chat template, CUDA image, and release manifest first.
