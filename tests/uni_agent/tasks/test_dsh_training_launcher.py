import os
import subprocess
from pathlib import Path

SCRIPT = Path("examples/dsh/train_qwen25_3b_online_rl.sh")


def test_qwen25_3b_launcher_has_valid_shell_syntax():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_qwen25_3b_launcher_prints_strict_online_rl_contract():
    env = {
        **os.environ,
        "PRINT_COMMAND": "1",
        "MODEL_PATH": "/models/Qwen2.5-3B-Instruct-snapshot",
        "TRAIN_FILE": "/data/dsh-train.parquet",
        "TEST_FILE": "/data/dsh-validation.parquet",
    }
    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    command = result.stdout
    for expected in (
        "Qwen/Qwen2.5-3B-Instruct",
        "trainer.use_v1=True",
        "trainer.v1.trainer_mode=sync",
        "trainer.v1.sampler.sync_refill_failed_groups=True",
        "algorithm.adv_estimator=grpo",
        "actor_rollout_ref.model.lora_rank=32",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.multi_turn.format=hermes",
        "uni_agent.framework.entry.AgentFrameworkRolloutAdapter",
        "runner_kwargs.report_reward=True",
        "runner_kwargs.require_reward_post=True",
        "fail_on_rollout_error=True",
        "trainer.save_freq=1",
        "trainer.total_training_steps=1",
    ):
        assert expected in command
