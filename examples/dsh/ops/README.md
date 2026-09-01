# Qwen3-4B DSH online-RL 操作脚本

这些脚本把 RunPod 上的人工操作固化成可重放步骤。它们只负责启动、观察和
记录 Uni-Agent/VERL 作业；DSH runtime、verifier 和 reward 仍由
`evolution_task_config_v2_fast.yaml` 与固定的 task release 管理。

## 首次准备

在 Uni-Agent 仓库根目录执行：

```sh
export DSH_VENV=/root/dsh-online-rl-venv
export ENVIRONMENT_DIGEST=sha256:<pinned-dsh-runtime-digest>
examples/dsh/ops/prepare_qwen3_4b_data.sh
```

脚本会生成 `DATA_ROOT/train.parquet`（16 行）、`holdout.parquet`（8 行）和
`manifest.json`。Parquet 不提交 Git；manifest 记录 scenario、fixture、verifier
和数据文件 digest。

## 启动训练

先确认模型 license，再显式设置 `MODEL_LICENSE_APPROVED=1`：

```sh
export MODEL_LICENSE_APPROVED=1
export MODEL_PATH=/workspace/models/Qwen3-4B
export DATA_ROOT=/workspace/data/dsh-evolution-v2
export RUN_ROOT=/workspace/runs/dsh-evolution-v2-online-rl
examples/dsh/ops/launch_qwen3_4b_online_rl.sh
```

默认配置是 Qwen3-4B、Hermes parser、4 个 rollout/group、16 个 train row、8
个 holdout row 和 4 个 optimizer step。默认以 `nohup` 脱离 SSH；要在当前终端
查看退出码可加 `--foreground`。启动器会拒绝覆盖已有 run root，并把完整无密钥
命令写入 `command.txt`，把 PID、代码 revision、数据 digest 写入
`run-manifest.json`。

## 重启、查看、reload

电脑或 SSH 重启后，不要重新猜命令，先执行：

```sh
examples/dsh/ops/status_qwen3_4b_online_rl.sh /workspace/runs/dsh-evolution-v2-online-rl
```

如果作业已结束，先从 status 输出找到 checkpoint，再以独立 run root 做验证：

```sh
MODEL_LICENSE_APPROVED=1 MODEL_PATH=/workspace/models/Qwen3-4B \\
  DATA_ROOT=/workspace/data/dsh-evolution-v2 \\
  examples/dsh/ops/reload_qwen3_4b_checkpoint.sh \\
  /workspace/runs/dsh-evolution-v2-online-rl/checkpoints/dsh-qwen3-4b-online-rl/expanded-v2/global_step_4
```

reload 默认只跑一个 rollout 和 8 个 holdout row；它证明 checkpoint 可加载并能
执行验证，不等于证明能力提升。

## 安全 teardown

```sh
examples/dsh/ops/teardown_qwen3_4b_online_rl.sh \\
  /workspace/runs/dsh-evolution-v2-online-rl --dry-run
RAY_STOP=1 examples/dsh/ops/teardown_qwen3_4b_online_rl.sh \\
  /workspace/runs/dsh-evolution-v2-online-rl
```

脚本只接受 run manifest 中的已知训练/推理进程，并默认不停止整台机器的 Ray
集群；确认服务器没有其他作业后才设置 `RAY_STOP=1`。GPU 为空、`teardown_at`
和 manifest 状态写入后，才算清理完成。

## 已知坑

- Task config 使用 `runner_python: python`，必须让 `$DSH_VENV/bin` 位于 `PATH`；否则 Ray worker 可能落到 `/usr/bin/python` 并报 `No module named pydantic`。
- 测试过的 VERL/torch memory-pool 路径不能继承 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`；启动脚本会主动 unset。
- `GPU_MEMORY_UTILIZATION=0.15` 只适用于带 eager、CPU offload 和 free-cache 的训练 launcher；直接用 inference 脚本时应按显存重新选择值。
- API key 不参与 SSH 或本地 DSH rollout。不要把 RunPod/PAT 写入仓库、manifest、command 或 `.env`；泄露过的 key 应先撤销并轮换。
