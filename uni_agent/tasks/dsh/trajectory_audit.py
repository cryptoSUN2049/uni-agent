"""Fail-closed admission checks for verifier-backed DSH trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from uni_agent.gateway.session import Trajectory

_DEFAULT_TRACE_ROOT = "/tmp/uni-agent-dsh/artifacts"
_DEFAULT_RESULT_ROOT = "/tmp/uni-agent-dsh-task/results"
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$", re.ASCII)


class TrajectoryAuditError(ValueError):
    """A trajectory cannot be admitted because its training evidence is invalid."""


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"invalid JSON constant: {constant}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TrajectoryAuditError(f"evidence is not canonical JSON: {exc}") from exc
    return (payload + "\n").encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrajectoryAuditError(f"{field} must be an object")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrajectoryAuditError(f"{field} must be a non-empty string")
    return value


def _require_digest(value: object, *, field: str) -> str:
    digest = _require_string(value, field=field)
    if _HASH_PATTERN.fullmatch(digest) is None:
        raise TrajectoryAuditError(f"{field} must match sha256:<64 lowercase hex digits>")
    return digest


def _require_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TrajectoryAuditError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise TrajectoryAuditError(f"{field} must be a finite number")
    return number


def _load_json_object(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TrajectoryAuditError(f"{field} is not readable: {path}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrajectoryAuditError(f"{field} is not strict JSON: {path}") from exc
    return _require_object(value, field=field), raw


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise TrajectoryAuditError(f"{field} does not match trusted DSH evidence")


def _validate_token_evidence(trajectory: Trajectory) -> None:
    response_length = len(trajectory.response_ids)
    if response_length == 0:
        raise TrajectoryAuditError("trajectory must contain response tokens")
    if len(trajectory.response_mask) != response_length:
        raise TrajectoryAuditError("response mask must align with response tokens")
    if any(value not in (0, 1) for value in trajectory.response_mask):
        raise TrajectoryAuditError("response mask must contain only 0 or 1")
    if not any(trajectory.response_mask):
        raise TrajectoryAuditError("trajectory must have a non-empty response mask")
    if trajectory.response_logprobs is None or len(trajectory.response_logprobs) != response_length:
        raise TrajectoryAuditError("rollout log probabilities must align with response tokens")
    if any(not math.isfinite(float(value)) for value in trajectory.response_logprobs):
        raise TrajectoryAuditError("trajectory must have finite log probabilities")


def _validate_trace(*, dsh: dict[str, Any], trace_root: Path) -> None:
    rollout_id = _require_string(dsh.get("rollout_id"), field="reward_info.dsh.rollout_id")
    trace_sha256 = _require_digest(dsh.get("trace_sha256"), field="reward_info.dsh.trace_sha256")
    trace_key = hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()[:24]
    trace_path = trace_root / trace_key / "session.jsonl"
    try:
        trace_bytes = trace_path.read_bytes()
    except OSError as exc:
        raise TrajectoryAuditError(f"DSH trace is not readable: {trace_path}") from exc
    if _digest(trace_bytes) != trace_sha256:
        raise TrajectoryAuditError("reward_info.dsh.trace_sha256 does not match DSH trace bytes")

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(trace_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrajectoryAuditError(f"DSH trace line {line_number} is not valid JSON") from exc
        events.append(_require_object(event, field=f"DSH trace line {line_number}"))
    if not events or events[-1].get("type") != "turn/end":
        raise TrajectoryAuditError("DSH trace must end with turn/end")
    event_count = dsh.get("event_count")
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count != len(events):
        raise TrajectoryAuditError("reward_info.dsh.event_count does not match DSH trace")


def _validate_partition(*, dsh: dict[str, Any], partition_id: str) -> None:
    split = _require_string(dsh.get("split"), field="reward_info.dsh.split")
    if partition_id == "train" and split != "train":
        raise TrajectoryAuditError("partition train requires split=train")
    if partition_id == "val" and split == "train":
        raise TrajectoryAuditError("partition val cannot use split=train")


def _validate_result_artifacts(
    *,
    reward_info: dict[str, Any],
    dsh: dict[str, Any],
    result_root: Path,
) -> None:
    reward = _require_finite(reward_info.get("reward"), field="reward_info.reward")
    verifier_reward = _require_finite(
        reward_info.get("verifier_reward"),
        field="reward_info.verifier_reward",
    )
    _require_equal(verifier_reward, reward, field="reward_info.verifier_reward")

    dsh_session_id = _require_string(dsh.get("dsh_session_id"), field="reward_info.dsh.dsh_session_id")
    rollout_id = _require_string(dsh.get("rollout_id"), field="reward_info.dsh.rollout_id")
    _require_equal(dsh_session_id, f"dsh-{rollout_id}", field="reward_info.dsh.dsh_session_id")
    trace_sha256 = _require_digest(dsh.get("trace_sha256"), field="reward_info.dsh.trace_sha256")
    artifact_sha256 = _require_digest(dsh.get("artifact_sha256"), field="reward_info.dsh.artifact_sha256")
    receipt_sha256 = _require_digest(dsh.get("receipt_sha256"), field="reward_info.dsh.receipt_sha256")
    environment_digest = _require_digest(
        dsh.get("environment_digest"),
        field="reward_info.dsh.environment_digest",
    )
    verifier_code_digest = _require_digest(
        dsh.get("verifier_code_digest"),
        field="reward_info.dsh.verifier_code_digest",
    )
    artifact_key = hashlib.sha256(f"{dsh_session_id}\0{trace_sha256}".encode()).hexdigest()[:24]
    result_dir = result_root / artifact_key

    envelope, envelope_bytes = _load_json_object(result_dir / "agent-result.json", field="DSH result envelope")
    if _digest(envelope_bytes) != artifact_sha256:
        raise TrajectoryAuditError("reward_info.dsh.artifact_sha256 does not match result envelope bytes")
    if envelope.get("schema") != "dsh.uni-agent.task-result.v1":
        raise TrajectoryAuditError("DSH result envelope has the wrong schema")
    if envelope.get("finished") is not True:
        raise TrajectoryAuditError("DSH result envelope must declare finished=true")
    envelope_dsh = _require_object(envelope.get("dsh"), field="DSH result envelope dsh")
    _require_equal(envelope_dsh.get("dsh_session_id"), dsh_session_id, field="result envelope dsh_session_id")
    _require_equal(envelope_dsh.get("trace_sha256"), trace_sha256, field="result envelope trace_sha256")
    metadata = _require_object(envelope.get("metadata"), field="DSH result envelope metadata")

    receipt, receipt_bytes = _load_json_object(
        result_dir / "verifier-receipt.json",
        field="DSH verifier receipt",
    )
    if receipt_bytes != _canonical_json_bytes(receipt):
        raise TrajectoryAuditError("DSH verifier receipt is not canonical JSON")
    if receipt.get("schema") != "dsh.verifier-receipt.v1" or dsh.get("schema") != receipt.get("schema"):
        raise TrajectoryAuditError("DSH verifier receipt has the wrong schema")
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if _digest(_canonical_json_bytes(receipt_body)) != receipt_sha256:
        raise TrajectoryAuditError("reward_info.dsh.receipt_sha256 does not match verifier receipt")
    _require_equal(receipt.get("receipt_id"), receipt_sha256, field="verifier receipt receipt_id")
    if receipt.get("fresh") is not True or dsh.get("freshness") != "fresh":
        raise TrajectoryAuditError("DSH verifier receipt must be fresh")
    _require_equal(dsh.get("eligible"), receipt.get("eligible"), field="reward_info.dsh.eligible")
    if receipt.get("eligible") is not True:
        raise TrajectoryAuditError("DSH verifier receipt must declare eligible=true")
    if receipt.get("finished") is not True:
        raise TrajectoryAuditError("DSH verifier receipt must declare finished=true")
    _require_equal(receipt.get("reward"), reward, field="verifier receipt reward")
    _require_equal(receipt.get("dsh_session_id"), dsh_session_id, field="verifier receipt dsh_session_id")
    _require_equal(receipt.get("trace_sha256"), trace_sha256, field="verifier receipt trace_sha256")
    _require_equal(receipt.get("artifact_sha256"), artifact_sha256, field="verifier receipt artifact_sha256")
    _require_equal(receipt.get("environment_digest"), environment_digest, field="verifier receipt environment_digest")

    verifier = _require_object(receipt.get("verifier"), field="DSH verifier receipt verifier")
    identity_fields = (
        ("task_id", receipt.get("task_id"), metadata.get("task_id")),
        ("task_version", receipt.get("task_version"), metadata.get("task_version")),
        ("split", dsh.get("split"), metadata.get("split")),
        ("environment_digest", environment_digest, metadata.get("environment_digest")),
        ("verifier_id", verifier.get("id"), metadata.get("verifier_id")),
        ("verifier_version", verifier.get("version"), metadata.get("verifier_version")),
        ("verifier_code_digest", verifier.get("code_digest"), metadata.get("verifier_code_digest")),
    )
    for field, receipt_value, envelope_value in identity_fields:
        _require_equal(dsh.get(field), receipt_value, field=f"reward_info.dsh.{field}")
        _require_equal(receipt_value, envelope_value, field=f"result envelope metadata.{field}")
    _require_equal(verifier.get("code_digest"), verifier_code_digest, field="verifier receipt code_digest")


def validate_trajectory(
    trajectory: Trajectory,
    *,
    partition_id: str,
    gateway_session_id: str,
    trace_root: str = _DEFAULT_TRACE_ROOT,
    result_root: str = _DEFAULT_RESULT_ROOT,
) -> None:
    """Validate one DSH trajectory and every file that establishes its lineage."""
    _validate_token_evidence(trajectory)
    reward_info = _require_object(trajectory.reward_info, field="reward_info")
    if reward_info.get("finished") is not True:
        raise TrajectoryAuditError("reward_info must declare finished=true")
    dsh = _require_object(reward_info.get("dsh"), field="reward_info.dsh")
    _validate_partition(dsh=dsh, partition_id=partition_id)
    _require_equal(
        dsh.get("rollout_id"),
        _require_string(gateway_session_id, field="context.gateway_session_id"),
        field="reward_info.dsh.rollout_id for context.gateway_session_id",
    )
    _validate_trace(dsh=dsh, trace_root=Path(trace_root))
    _validate_result_artifacts(reward_info=reward_info, dsh=dsh, result_root=Path(result_root))


def validate_trajectories(
    trajectories: tuple[Trajectory, ...],
    *,
    context: Mapping[str, object],
    trace_root: str = _DEFAULT_TRACE_ROOT,
    result_root: str = _DEFAULT_RESULT_ROOT,
) -> list[Trajectory]:
    """Validate a finalized DSH session before it can enter TransferQueue."""
    if not trajectories:
        raise TrajectoryAuditError("DSH trajectory audit requires at least one trajectory")
    partition_id = context.get("partition_id") if isinstance(context, Mapping) else None
    if partition_id not in {"train", "val"}:
        raise TrajectoryAuditError("trajectory audit context.partition_id must be train or val")
    gateway_session_id = context.get("gateway_session_id")
    if not isinstance(gateway_session_id, str) or not gateway_session_id:
        raise TrajectoryAuditError("trajectory audit context.gateway_session_id must be a non-empty string")
    for index, trajectory in enumerate(trajectories):
        try:
            validate_trajectory(
                trajectory,
                partition_id=partition_id,
                gateway_session_id=gateway_session_id,
                trace_root=trace_root,
                result_root=result_root,
            )
        except TrajectoryAuditError as exc:
            raise TrajectoryAuditError(f"trajectory {index}: {exc}") from exc
    return list(trajectories)
