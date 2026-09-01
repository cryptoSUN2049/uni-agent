import pytest

from uni_agent.tasks import TaskResult
from uni_agent.tasks.base import build_reward_info


def test_task_reward_info_preserves_dsh_lineage_for_gateway_trajectory() -> None:
    result = TaskResult(
        reward=0.75,
        accuracy=0.5,
        finished=True,
        reward_info={
            "dsh": {
                "trace_sha256": "sha256:" + "a" * 64,
                "receipt_sha256": "sha256:" + "b" * 64,
                "freshness": "fresh",
            }
        },
    )

    assert build_reward_info(result) == {
        "reward": 0.75,
        "acc": 0.5,
        "finished": True,
        "dsh": {
            "trace_sha256": "sha256:" + "a" * 64,
            "receipt_sha256": "sha256:" + "b" * 64,
            "freshness": "fresh",
        },
    }


def test_task_reward_info_rejects_reserved_or_nonfinite_metadata() -> None:
    with pytest.raises(ValueError, match="cannot overwrite"):
        build_reward_info(TaskResult(reward=1.0, reward_info={"reward": 2.0}))
    with pytest.raises(ValueError, match="strict-JSON"):
        build_reward_info(TaskResult(reward=float("nan")))
