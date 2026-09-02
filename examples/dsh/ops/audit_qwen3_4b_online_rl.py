#!/usr/bin/env python3
"""Audit persisted DSH trajectory groups against their trusted artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from uni_agent.framework.trajectory_identity import trajectory_tq_key
from uni_agent.gateway.session import Trajectory
from uni_agent.tasks.dsh.trajectory_audit import TrajectoryAuditError, validate_trajectory

_DUMP_SCHEMA = "uni-agent.trajectory-dump.v2"
_REPORT_SCHEMA = "dsh.trajectory-group-audit.v1"


class AuditInputError(ValueError):
    """The run manifest or its declared audit inputs are unavailable."""


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"invalid JSON constant: {constant}")


def _strict_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AuditInputError(f"{label} is not readable: {path}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AuditInputError(f"{label} is not strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuditInputError(f"{label} must be a JSON object: {path}")
    return value, raw


def _manifest_path(manifest: dict[str, Any], name: str) -> Path:
    paths = manifest.get("paths")
    if not isinstance(paths, dict):
        raise AuditInputError("run manifest is missing paths")
    value = paths.get(name)
    if not isinstance(value, str) or not value:
        raise AuditInputError(f"run manifest is missing paths.{name}")
    return Path(value)


def _expected_rollouts(manifest: dict[str, Any], partition: str) -> int:
    rollout = manifest.get("rollout")
    if not isinstance(rollout, dict):
        raise AuditInputError("run manifest is missing rollout counts")
    field = "train_n" if partition == "train" else "validation_n"
    value = rollout.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AuditInputError(f"run manifest rollout.{field} must be a positive integer")
    return value


def _consumed_keys(path: Path) -> list[tuple[int, str]]:
    keys: list[tuple[int, str]] = []
    if not path.is_dir():
        return keys
    for jsonl_path in sorted(path.rglob("*.jsonl")):
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise AuditInputError(f"trainer output is not readable: {jsonl_path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise AuditInputError(f"trainer output line is not JSON: {jsonl_path}:{line_number}") from exc
            uid = row.get("uid") if isinstance(row, dict) else None
            if not isinstance(uid, str) or not uid:
                raise AuditInputError(f"trainer output line is missing uid: {jsonl_path}:{line_number}")
            step = row.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise AuditInputError(f"trainer output line has invalid step: {jsonl_path}:{line_number}")
            keys.append((step, uid))
    return keys


def _load_dump_trajectory(
    *,
    npz_bytes: bytes,
    trajectory_meta: dict[str, Any],
    trajectory_index: int,
) -> Trajectory:
    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as arrays:
            prefix = f"traj{trajectory_index}_"
            prompt_ids = arrays[prefix + "prompt_ids"].tolist()
            response_ids = arrays[prefix + "response_ids"].tolist()
            response_mask = arrays[prefix + "response_mask"].tolist()
            logprob_key = prefix + "response_logprobs"
            response_logprobs = arrays[logprob_key].tolist() if logprob_key in arrays else None
    except (OSError, ValueError, KeyError) as exc:
        raise TrajectoryAuditError("trajectory NPZ is incomplete or unreadable") from exc

    reward_info = trajectory_meta.get("reward_info")
    if not isinstance(reward_info, dict):
        raise TrajectoryAuditError("trajectory metadata reward_info must be an object")
    reward_extra_info = trajectory_meta.get("reward_extra_info")
    if not isinstance(reward_extra_info, dict):
        raise TrajectoryAuditError("trajectory metadata reward_extra_info must be an object")
    reward_score = trajectory_meta.get("reward_score")
    if isinstance(reward_score, bool) or not isinstance(reward_score, int | float) or not math.isfinite(reward_score):
        raise TrajectoryAuditError("trajectory metadata reward_score must be finite")
    if reward_score != reward_info.get("reward") or reward_score != reward_extra_info.get("verifier_reward"):
        raise TrajectoryAuditError("trajectory metadata reward projection does not match verifier reward")
    if reward_info.get("dsh") != reward_extra_info.get("dsh"):
        raise TrajectoryAuditError("trajectory metadata DSH lineage projection does not match reward_info")
    if trajectory_meta.get("prompt_len") != len(prompt_ids):
        raise TrajectoryAuditError("trajectory metadata prompt_len does not match NPZ")
    if trajectory_meta.get("response_len") != len(response_ids):
        raise TrajectoryAuditError("trajectory metadata response_len does not match NPZ")
    if trajectory_meta.get("model_token_count") != sum(response_mask):
        raise TrajectoryAuditError("trajectory metadata model_token_count does not match NPZ")
    if trajectory_meta.get("has_logprobs") is not True:
        raise TrajectoryAuditError("trajectory metadata must declare has_logprobs=true")

    return Trajectory(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        reward_info=reward_info,
        reward_score=float(reward_score),
        num_turns=int(trajectory_meta.get("num_turns", 0)),
        extra_fields={"reward_extra_info": reward_extra_info},
    )


def _audit_dump(
    *,
    dump_path: Path,
    meta: dict[str, Any],
    trace_root: Path,
    result_root: Path,
) -> tuple[list[str], list[str], list[float], list[str]]:
    reasons: list[str] = []
    transfer_keys: list[str] = []
    rewards: list[float] = []
    receipt_ids: list[str] = []
    trajectories = meta.get("trajectories")
    if not isinstance(trajectories, list) or meta.get("num_trajectories") != len(trajectories):
        return ["trajectory_count_mismatch"], transfer_keys, rewards, receipt_ids
    npz_path = dump_path.parent / "trajectory.npz"
    try:
        npz_bytes = npz_path.read_bytes()
    except OSError:
        return ["trajectory_npz_unreadable"], transfer_keys, rewards, receipt_ids
    expected_npz_digest = meta.get("trajectory_npz_sha256")
    actual_npz_digest = "sha256:" + hashlib.sha256(npz_bytes).hexdigest()
    if expected_npz_digest != actual_npz_digest:
        return ["trajectory_npz_sha256_mismatch"], transfer_keys, rewards, receipt_ids
    session_index = meta.get("session_index")
    valid_session_index = (
        session_index if not isinstance(session_index, bool) and isinstance(session_index, int) else -1
    )
    for index, raw_trajectory_meta in enumerate(trajectories):
        if not isinstance(raw_trajectory_meta, dict):
            reasons.append(f"trajectory_{index}:metadata_not_object")
            continue
        expected_key = trajectory_tq_key(str(meta.get("group_uid")), valid_session_index, index)
        if raw_trajectory_meta.get("trajectory_index") != index:
            reasons.append(f"trajectory_{index}:trajectory_index_mismatch")
        if raw_trajectory_meta.get("transfer_queue_key") != expected_key:
            reasons.append(f"trajectory_{index}:transfer_queue_key_mismatch")
        else:
            transfer_keys.append(expected_key)
        try:
            trajectory = _load_dump_trajectory(
                npz_bytes=npz_bytes,
                trajectory_meta=raw_trajectory_meta,
                trajectory_index=index,
            )
            validate_trajectory(
                trajectory,
                partition_id=str(meta.get("partition_id")),
                gateway_session_id=str(meta.get("gateway_session_id")),
                trace_root=str(trace_root),
                result_root=str(result_root),
            )
        except (TrajectoryAuditError, TypeError, ValueError) as exc:
            reasons.append(f"trajectory_{index}:{exc}")
            continue
        rewards.append(float(trajectory.reward_score))
        receipt_ids.append(str(trajectory.reward_info["dsh"]["receipt_sha256"]))
    return reasons, transfer_keys, rewards, receipt_ids


def audit_trajectory_groups(run_root: Path, *, partition: str | None = None) -> dict[str, Any]:
    """Return a deterministic audit report for one manifest-declared run."""
    manifest, manifest_bytes = _strict_json_object(run_root / "run-manifest.json", label="run manifest")
    agent_log_root = _manifest_path(manifest, "agent_log_dir")
    trace_root = _manifest_path(manifest, "trace_root")
    result_root = _manifest_path(manifest, "result_root")
    if not agent_log_root.is_dir():
        raise AuditInputError(f"agent log directory is not readable: {agent_log_root}")

    requested_partitions = [partition] if partition else ["train", "val"]
    expected_by_partition = {name: _expected_rollouts(manifest, name) for name in requested_partitions}
    consumed_by_partition: dict[str, list[tuple[int, str]]] = {}
    if "train" in requested_partitions:
        consumed_by_partition["train"] = _consumed_keys(_manifest_path(manifest, "rollout_data_dir"))
    if "val" in requested_partitions:
        consumed_by_partition["val"] = _consumed_keys(_manifest_path(manifest, "validation_data_dir"))

    groups: dict[tuple[str, int, str], dict[str, Any]] = {}
    legacy_files: list[str] = []
    receipt_groups: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    for dump_path in sorted(agent_log_root.rglob("trajectory.json")):
        try:
            meta, _raw = _strict_json_object(dump_path, label="trajectory dump")
        except AuditInputError as exc:
            legacy_files.append(f"{dump_path.relative_to(agent_log_root)}:{exc}")
            continue
        if meta.get("schema") != _DUMP_SCHEMA:
            legacy_files.append(str(dump_path.relative_to(agent_log_root)))
            continue
        dump_partition = meta.get("partition_id")
        if dump_partition not in requested_partitions:
            continue
        group_uid = meta.get("group_uid")
        global_steps = meta.get("global_steps")
        if not isinstance(group_uid, str) or not group_uid:
            legacy_files.append(f"{dump_path.relative_to(agent_log_root)}:missing_group_uid")
            continue
        valid_global_steps = (
            global_steps
            if not isinstance(global_steps, bool) and isinstance(global_steps, int) and global_steps >= 0
            else -1
        )
        key = (str(dump_partition), valid_global_steps, group_uid)
        group = groups.setdefault(
            key,
            {
                "partition_id": dump_partition,
                "global_steps": global_steps,
                "group_uid": group_uid,
                "expected_sessions": expected_by_partition[str(dump_partition)],
                "session_indexes": [],
                "gateway_session_ids": [],
                "transfer_queue_keys": [],
                "reward_values": [],
                "reasons": [],
            },
        )
        if valid_global_steps < 0:
            group["reasons"].append("invalid_global_steps")
        if meta.get("group_size") != group["expected_sessions"]:
            group["reasons"].append("group_size_mismatch")
        session_index = meta.get("session_index")
        gateway_session_id = meta.get("gateway_session_id")
        if isinstance(session_index, bool) or not isinstance(session_index, int) or session_index < 0:
            group["reasons"].append("invalid_session_index")
        else:
            group["session_indexes"].append(session_index)
        if not isinstance(gateway_session_id, str) or meta.get("session_id") != gateway_session_id:
            group["reasons"].append("gateway_session_id_mismatch")
        else:
            group["gateway_session_ids"].append(gateway_session_id)
        reasons, transfer_keys, rewards, receipt_ids = _audit_dump(
            dump_path=dump_path,
            meta=meta,
            trace_root=trace_root,
            result_root=result_root,
        )
        group["reasons"].extend(reasons)
        group["transfer_queue_keys"].extend(transfer_keys)
        group["reward_values"].extend(rewards)
        for receipt_id in receipt_ids:
            receipt_groups[receipt_id].add(key)

    for receipt_id, owner_groups in receipt_groups.items():
        if len(owner_groups) > 1:
            for key in owner_groups:
                groups[key]["reasons"].append(f"duplicate_receipt:{receipt_id}")

    report_groups: list[dict[str, Any]] = []
    known_keys: dict[str, set[tuple[int, str]]] = {name: set() for name in requested_partitions}
    for key in sorted(groups):
        group = groups[key]
        expected_sessions = group.pop("expected_sessions")
        session_indexes = sorted(group.pop("session_indexes"))
        gateway_session_ids = group.pop("gateway_session_ids")
        transfer_keys = group.pop("transfer_queue_keys")
        reward_values = group.pop("reward_values")
        reasons = group.pop("reasons")
        if session_indexes != list(range(expected_sessions)):
            reasons.append("group_size_mismatch")
        if len(set(gateway_session_ids)) != len(gateway_session_ids):
            reasons.append("duplicate_gateway_session_id")
        if len(set(transfer_keys)) != len(transfer_keys):
            reasons.append("duplicate_transfer_queue_key")
        group_step = key[1]
        joined_keys = [(group_step, transfer_key) for transfer_key in transfer_keys]
        known_keys[str(group["partition_id"])].update(joined_keys)
        consumed = Counter(consumed_by_partition[str(group["partition_id"])])
        duplicate_consumed = sorted(
            transfer_key for step, transfer_key in joined_keys if consumed[(step, transfer_key)] > 1
        )
        if duplicate_consumed:
            reasons.append("duplicate_optimizer_row")
        missing_consumed = sorted(
            transfer_key for step, transfer_key in joined_keys if consumed[(step, transfer_key)] == 0
        )
        reasons = sorted(set(reasons))
        eligible = not reasons
        if not eligible:
            status = "rejected"
        elif missing_consumed:
            status = "eligible-not-consumed"
        else:
            status = "eligible-and-consumed"
        report_groups.append(
            {
                **group,
                "expected_sessions": expected_sessions,
                "observed_sessions": len(session_indexes),
                "eligible": eligible,
                "status": status,
                "has_reward_variance": len(set(reward_values)) > 1,
                "reward_values": reward_values,
                "transfer_queue_keys": transfer_keys,
                "missing_consumed_keys": missing_consumed,
                "reasons": reasons,
            }
        )

    unexpected_consumed = {
        name: [
            {"global_steps": step, "transfer_queue_key": transfer_key}
            for step, transfer_key in sorted(set(values) - known_keys[name])
        ]
        for name, values in consumed_by_partition.items()
    }
    rejected_groups = sum(not group["eligible"] for group in report_groups)
    eligible_groups = len(report_groups) - rejected_groups
    run_status = manifest.get("status")
    run_completed = run_status == "completed"
    overall_eligible = (
        run_completed
        and bool(report_groups)
        and rejected_groups == 0
        and all(group["status"] == "eligible-and-consumed" for group in report_groups)
        and not legacy_files
        and not any(unexpected_consumed.values())
    )
    return {
        "schema": _REPORT_SCHEMA,
        "run_manifest_sha256": "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        "run_status": run_status,
        "eligible": overall_eligible,
        "summary": {
            "groups": len(report_groups),
            "eligible_groups": eligible_groups,
            "rejected_groups": rejected_groups,
            "variance_groups": sum(group["has_reward_variance"] for group in report_groups if group["eligible"]),
            "legacy_unjoinable_files": len(legacy_files),
            "unexpected_consumed_rows": sum(len(values) for values in unexpected_consumed.values()),
            "run_completed": run_completed,
        },
        "groups": report_groups,
        "legacy_unjoinable": legacy_files,
        "unexpected_consumed": unexpected_consumed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--partition", choices=("train", "val"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Write the canonical audit report and fail when any group is ineligible."""
    args = _parser().parse_args()
    try:
        report = audit_trajectory_groups(args.run_root, partition=args.partition)
    except AuditInputError as exc:
        print(f"DSH trajectory audit input error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    output = args.output or args.run_root / "trajectory-audit.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    print(output)
    if not report["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
