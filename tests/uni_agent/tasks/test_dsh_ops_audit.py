from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from examples.dsh.ops.audit_qwen3_4b_online_rl import audit_trajectory_groups
from tests.uni_agent.tasks.test_dsh_trajectory_audit import _valid_trajectory


def _write_run(tmp_path: Path, *, consume: bool = True, consumed_step: int = 1) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    agent_log_root = run_root / "agent-logs/project/experiment"
    rollout_root = run_root / "rollouts/project/experiment"
    validation_root = run_root / "validation/project/experiment"
    session_dir = agent_log_root / "step_1/session-sample-0-rollout-0-audit"
    session_dir.mkdir(parents=True)
    rollout_root.mkdir(parents=True)
    validation_root.mkdir(parents=True)

    trajectory, trace_root, result_root = _valid_trajectory(run_root)
    dsh = trajectory.reward_info["dsh"]
    scored = replace(
        trajectory,
        reward_score=0.75,
        extra_fields={"reward_extra_info": {"verifier_reward": 0.75, "dsh": dict(dsh)}},
    )
    transfer_key = "group-uid_0_0"
    npz_path = session_dir / "trajectory.npz"
    np.savez_compressed(
        npz_path,
        traj0_prompt_ids=np.asarray(scored.prompt_ids, dtype=np.int32),
        traj0_response_ids=np.asarray(scored.response_ids, dtype=np.int32),
        traj0_response_mask=np.asarray(scored.response_mask, dtype=np.int8),
        traj0_response_logprobs=np.asarray(scored.response_logprobs, dtype=np.float32),
    )
    dump = {
        "schema": "uni-agent.trajectory-dump.v2",
        "session_id": dsh["rollout_id"],
        "gateway_session_id": dsh["rollout_id"],
        "partition_id": "train",
        "global_steps": 1,
        "group_uid": "group-uid",
        "group_size": 1,
        "sample_index": 0,
        "session_index": 0,
        "trajectory_npz_sha256": "sha256:" + hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "num_trajectories": 1,
        "trajectories": [
            {
                "trajectory_index": 0,
                "transfer_queue_key": transfer_key,
                "num_turns": scored.num_turns,
                "finished": True,
                "reward_score": scored.reward_score,
                "reward_info": scored.reward_info,
                "reward_extra_info": scored.extra_fields["reward_extra_info"],
                "materialization_reason": None,
                "prompt_len": len(scored.prompt_ids),
                "response_len": len(scored.response_ids),
                "model_token_count": sum(scored.response_mask),
                "has_routed_experts": False,
                "has_logprobs": True,
            }
        ],
    }
    (session_dir / "trajectory.json").write_text(
        json.dumps(dump, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    if consume:
        (rollout_root / f"{consumed_step}.jsonl").write_text(
            json.dumps({"uid": transfer_key, "step": consumed_step}) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema": "dsh.online-rl.run-manifest.v1",
        "status": "completed",
        "paths": {
            "agent_log_dir": str(agent_log_root),
            "rollout_data_dir": str(rollout_root),
            "validation_data_dir": str(validation_root),
            "trace_root": str(trace_root),
            "result_root": str(result_root),
        },
        "rollout": {"train_n": 1, "validation_n": 1},
    }
    (run_root / "run-manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return run_root, session_dir


def test_audit_reports_hash_bound_group_as_consumed_and_eligible(tmp_path: Path) -> None:
    run_root, _session_dir = _write_run(tmp_path)

    report = audit_trajectory_groups(run_root, partition="train")

    assert report["eligible"] is True
    assert report["summary"] == {
        "groups": 1,
        "eligible_groups": 1,
        "rejected_groups": 0,
        "variance_groups": 0,
        "legacy_unjoinable_files": 0,
        "unexpected_consumed_rows": 0,
        "run_completed": True,
    }
    assert report["groups"][0]["status"] == "eligible-and-consumed"
    assert report["groups"][0]["transfer_queue_keys"] == ["group-uid_0_0"]


def test_audit_reports_preoptimizer_group_as_not_yet_consumed(tmp_path: Path) -> None:
    run_root, _session_dir = _write_run(tmp_path, consume=False)

    report = audit_trajectory_groups(run_root, partition="train")

    assert report["eligible"] is False
    assert report["groups"][0]["status"] == "eligible-not-consumed"
    assert report["groups"][0]["missing_consumed_keys"] == ["group-uid_0_0"]


def test_audit_requires_completed_run_manifest(tmp_path: Path) -> None:
    run_root, _session_dir = _write_run(tmp_path)
    manifest_path = run_root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    report = audit_trajectory_groups(run_root, partition="train")

    assert report["eligible"] is False
    assert report["run_status"] == "running"
    assert report["summary"]["run_completed"] is False


def test_audit_rejects_nonfinite_dump_and_unmatched_optimizer_row(tmp_path: Path) -> None:
    run_root, session_dir = _write_run(tmp_path)
    with np.load(session_dir / "trajectory.npz") as arrays:
        values = {name: arrays[name] for name in arrays.files}
    values["traj0_response_logprobs"] = np.asarray([float("nan"), 0.0, -0.2], dtype=np.float32)
    npz_path = session_dir / "trajectory.npz"
    np.savez_compressed(npz_path, **values)
    dump_path = session_dir / "trajectory.json"
    dump = json.loads(dump_path.read_text(encoding="utf-8"))
    dump["trajectory_npz_sha256"] = "sha256:" + hashlib.sha256(npz_path.read_bytes()).hexdigest()
    dump_path.write_text(json.dumps(dump, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    rollout_path = run_root / "rollouts/project/experiment/1.jsonl"
    with rollout_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"uid": "unknown_0_0", "step": 1}) + "\n")

    report = audit_trajectory_groups(run_root, partition="train")

    assert report["eligible"] is False
    assert report["groups"][0]["status"] == "rejected"
    assert any("finite log probabilities" in reason for reason in report["groups"][0]["reasons"])
    assert report["unexpected_consumed"] == {"train": [{"global_steps": 1, "transfer_queue_key": "unknown_0_0"}]}


def test_audit_rejects_npz_bytes_that_do_not_match_dump_digest(tmp_path: Path) -> None:
    run_root, session_dir = _write_run(tmp_path)
    with np.load(session_dir / "trajectory.npz") as arrays:
        values = {name: arrays[name] for name in arrays.files}
    values["traj0_prompt_ids"] = np.asarray([901, 902], dtype=np.int32)
    np.savez_compressed(session_dir / "trajectory.npz", **values)

    report = audit_trajectory_groups(run_root, partition="train")

    assert report["eligible"] is False
    assert "trajectory_npz_sha256_mismatch" in report["groups"][0]["reasons"]


def test_audit_joins_consumption_by_global_step_and_transfer_queue_key(tmp_path: Path) -> None:
    run_root, _session_dir = _write_run(tmp_path, consumed_step=2)

    report = audit_trajectory_groups(run_root, partition="train")

    assert report["eligible"] is False
    assert report["groups"][0]["status"] == "eligible-not-consumed"
    assert report["groups"][0]["missing_consumed_keys"] == ["group-uid_0_0"]
    assert report["unexpected_consumed"] == {"train": [{"global_steps": 2, "transfer_queue_key": "group-uid_0_0"}]}


def test_audit_marks_pre_crosswalk_dump_as_legacy_unjoinable(tmp_path: Path) -> None:
    run_root, session_dir = _write_run(tmp_path)
    dump_path = session_dir / "trajectory.json"
    dump = json.loads(dump_path.read_text(encoding="utf-8"))
    dump.pop("schema")
    dump_path.write_text(json.dumps(dump) + "\n", encoding="utf-8")

    report = audit_trajectory_groups(run_root, partition="train")

    assert report["eligible"] is False
    assert report["summary"]["legacy_unjoinable_files"] == 1
    assert report["summary"]["unexpected_consumed_rows"] == 1
