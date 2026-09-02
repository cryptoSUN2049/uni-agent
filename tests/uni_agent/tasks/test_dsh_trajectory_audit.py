from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from uni_agent.gateway.session import Trajectory
from uni_agent.tasks.dsh.trajectory_audit import TrajectoryAuditError, validate_trajectories


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _valid_trajectory(
    tmp_path: Path,
    *,
    split: str = "train",
    eligible: bool = True,
) -> tuple[Trajectory, Path, Path]:
    gateway_session_id = "session-sample-0-rollout-0-audit"
    dsh_session_id = f"dsh-{gateway_session_id}"
    trace_root = tmp_path / "traces"
    result_root = tmp_path / "results"
    trace_bytes = b'{"type":"turn/end","data":{"turn":1}}\n'
    trace_sha256 = _digest(trace_bytes)
    trace_dir = trace_root / hashlib.sha256(gateway_session_id.encode()).hexdigest()[:24]
    trace_dir.mkdir(parents=True)
    (trace_dir / "session.jsonl").write_bytes(trace_bytes)

    identity = {
        "task_id": "dsh/harness-evolution/audit-01",
        "task_version": "1",
        "split": split,
        "environment_digest": _digest(b"environment"),
        "verifier_id": "dsh-harness-evolution-verifier",
        "verifier_version": "1",
        "verifier_code_digest": _digest(b"verifier bundle"),
    }
    envelope = {
        "schema": "dsh.uni-agent.task-result.v1",
        "metadata": dict(identity),
        "finished": True,
        "dsh": {
            "dsh_session_id": dsh_session_id,
            "trace_sha256": trace_sha256,
        },
    }
    envelope_bytes = _canonical_json_bytes(envelope)
    artifact_sha256 = _digest(envelope_bytes)
    artifact_key = hashlib.sha256(f"{dsh_session_id}\0{trace_sha256}".encode()).hexdigest()[:24]
    result_dir = result_root / artifact_key
    result_dir.mkdir(parents=True)
    (result_dir / "agent-result.json").write_bytes(envelope_bytes)

    receipt_body = {
        "schema": "dsh.verifier-receipt.v1",
        "task_id": identity["task_id"],
        "task_version": identity["task_version"],
        "dsh_session_id": dsh_session_id,
        "trace_sha256": trace_sha256,
        "artifact_sha256": artifact_sha256,
        "environment_digest": identity["environment_digest"],
        "verifier": {
            "id": identity["verifier_id"],
            "version": identity["verifier_version"],
            "code_digest": identity["verifier_code_digest"],
        },
        "issued_at": "2026-09-02T00:00:00Z",
        "fresh": True,
        "eligible": eligible,
        "issuer": {"kind": "trusted-verifier", "id": "uni-agent-dsh"},
        "reward": 0.75,
        "accuracy": 1.0,
        "finished": True,
        "evidence": ["trace"],
    }
    receipt_id = _digest(_canonical_json_bytes(receipt_body))
    receipt = {"receipt_id": receipt_id, **receipt_body}
    (result_dir / "verifier-receipt.json").write_bytes(_canonical_json_bytes(receipt))

    reward_info = {
        "reward": 0.75,
        "verifier_reward": 0.75,
        "finished": True,
        "dsh": {
            "schema": "dsh.verifier-receipt.v1",
            "receipt_sha256": receipt_id,
            "freshness": "fresh",
            "eligible": eligible,
            "rollout_id": gateway_session_id,
            "task_id": identity["task_id"],
            "task_version": identity["task_version"],
            "split": identity["split"],
            "dsh_session_id": dsh_session_id,
            "trace_sha256": trace_sha256,
            "artifact_sha256": artifact_sha256,
            "environment_digest": identity["environment_digest"],
            "verifier_id": identity["verifier_id"],
            "verifier_version": identity["verifier_version"],
            "verifier_code_digest": identity["verifier_code_digest"],
            "event_count": 1,
        },
    }
    trajectory = Trajectory(
        prompt_ids=[10, 11],
        response_ids=[20, 21, 22],
        response_mask=[1, 0, 1],
        response_logprobs=[-0.1, 0.0, -0.2],
        reward_info=reward_info,
        num_turns=1,
    )
    return trajectory, trace_root, result_root


def test_validate_trajectories_accepts_fresh_hash_bound_dsh_artifacts(tmp_path: Path) -> None:
    trajectory, trace_root, result_root = _valid_trajectory(tmp_path)

    result = validate_trajectories(
        (trajectory,),
        context={
            "partition_id": "train",
            "gateway_session_id": "session-sample-0-rollout-0-audit",
        },
        trace_root=str(trace_root),
        result_root=str(result_root),
    )

    assert result == [trajectory]


def test_validate_trajectories_rejects_verifier_ineligible_episode(tmp_path: Path) -> None:
    trajectory, trace_root, result_root = _valid_trajectory(tmp_path, eligible=False)

    with pytest.raises(TrajectoryAuditError, match="eligible=true"):
        validate_trajectories(
            (trajectory,),
            context={
                "partition_id": "train",
                "gateway_session_id": "session-sample-0-rollout-0-audit",
            },
            trace_root=str(trace_root),
            result_root=str(result_root),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda trajectory: trajectory.response_mask.__setitem__(slice(None), [0, 0, 0]), "non-empty response mask"),
        (lambda trajectory: trajectory.response_logprobs.__setitem__(0, float("nan")), "finite log probabilities"),
        (lambda trajectory: trajectory.response_mask.pop(), "response mask must align"),
        (lambda trajectory: trajectory.reward_info.__setitem__("finished", False), "finished=true"),
        (lambda trajectory: trajectory.reward_info["dsh"].__setitem__("event_count", 2), "event_count"),
        (
            lambda trajectory: trajectory.reward_info["dsh"].__setitem__("task_id", "dsh/other"),
            "task_id",
        ),
        (
            lambda trajectory: trajectory.reward_info["dsh"].__setitem__(
                "environment_digest", _digest(b"other environment")
            ),
            "environment_digest",
        ),
    ],
    ids=[
        "empty-mask",
        "nonfinite-logprob",
        "length-mismatch",
        "unfinished",
        "event-count-mismatch",
        "task-identity-mismatch",
        "environment-identity-mismatch",
    ],
)
def test_validate_trajectories_rejects_invalid_training_evidence(tmp_path: Path, mutate, message: str) -> None:
    trajectory, trace_root, result_root = _valid_trajectory(tmp_path)
    mutate(trajectory)

    with pytest.raises(TrajectoryAuditError, match=message):
        validate_trajectories(
            (trajectory,),
            context={
                "partition_id": "train",
                "gateway_session_id": "session-sample-0-rollout-0-audit",
            },
            trace_root=str(trace_root),
            result_root=str(result_root),
        )


def test_validate_trajectories_rejects_tampered_artifact_bytes(tmp_path: Path) -> None:
    trajectory, trace_root, result_root = _valid_trajectory(tmp_path)
    artifact_key = hashlib.sha256(
        f"{trajectory.reward_info['dsh']['dsh_session_id']}\0{trajectory.reward_info['dsh']['trace_sha256']}".encode()
    ).hexdigest()[:24]
    (result_root / artifact_key / "agent-result.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(TrajectoryAuditError, match="artifact_sha256"):
        validate_trajectories(
            (trajectory,),
            context={
                "partition_id": "train",
                "gateway_session_id": "session-sample-0-rollout-0-audit",
            },
            trace_root=str(trace_root),
            result_root=str(result_root),
        )


def test_validate_trajectories_rejects_tampered_receipt_bytes(tmp_path: Path) -> None:
    trajectory, trace_root, result_root = _valid_trajectory(tmp_path)
    artifact_key = hashlib.sha256(
        f"{trajectory.reward_info['dsh']['dsh_session_id']}\0{trajectory.reward_info['dsh']['trace_sha256']}".encode()
    ).hexdigest()[:24]
    receipt_path = result_root / artifact_key / "verifier-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["reward"] = 0.5
    receipt_path.write_bytes(_canonical_json_bytes(receipt))

    with pytest.raises(TrajectoryAuditError, match="receipt_sha256"):
        validate_trajectories(
            (trajectory,),
            context={
                "partition_id": "train",
                "gateway_session_id": "session-sample-0-rollout-0-audit",
            },
            trace_root=str(trace_root),
            result_root=str(result_root),
        )


def test_validate_trajectories_rejects_train_partition_with_holdout_identity(tmp_path: Path) -> None:
    trajectory, trace_root, result_root = _valid_trajectory(tmp_path, split="holdout")

    with pytest.raises(TrajectoryAuditError, match="partition train requires split=train"):
        validate_trajectories(
            (trajectory,),
            context={
                "partition_id": "train",
                "gateway_session_id": "session-sample-0-rollout-0-audit",
            },
            trace_root=str(trace_root),
            result_root=str(result_root),
        )


def test_validate_trajectories_rejects_receipt_from_another_gateway_session(tmp_path: Path) -> None:
    trajectory, trace_root, result_root = _valid_trajectory(tmp_path)

    with pytest.raises(TrajectoryAuditError, match="gateway_session_id"):
        validate_trajectories(
            (trajectory,),
            context={"partition_id": "train", "gateway_session_id": "session-other"},
            trace_root=str(trace_root),
            result_root=str(result_root),
        )
