"""Run a DeepSeek Harness episode and score it with a trusted verifier.

The task keeps reward computation outside the DSH Agent process. The Agent
produces a semantic trace reference; this task writes a minimal result envelope
inside the same Sandbox and invokes an operator-owned verifier command. The
verifier emits exactly one JSON object on stdout, which becomes the Uni-Agent
TaskResult and, when enabled by the framework runner, the VERL reward.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from pydantic import Field, field_validator

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$", re.ASCII)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$", re.ASCII)
_SPLITS = {"train", "validation", "test", "holdout"}
_INFO_KEYS = (
    "adapter",
    "dsh_session_id",
    "gateway_session_id",
    "trace_sha256",
    "trace_path",
    "event_count",
    "finish_reason",
    "keep_trace",
)


class DshArchitectureTaskConfig(TaskConfig):
    """Configuration for one verifier-backed DSH architecture episode."""

    task_config_only_fields = frozenset(
        {
            "agent",
            "require_trace",
            "result_root",
            "sandbox",
            "verifier_command",
            "verifier_timeout",
            "workdir",
            "environment_digest",
            "verifier_id",
            "verifier_version",
            "verifier_code_digest",
        }
    )
    name: str = "dsh_architecture"
    verifier_command: list[str] = Field(
        min_length=1,
        description="Trusted executable argv; it must print one JSON object containing a finite reward.",
    )
    verifier_timeout: float = Field(default=300.0, gt=0, description="Verifier wall-clock cap in seconds.")
    result_root: str = Field(
        default="/tmp/uni-agent-dsh-task/results",
        min_length=1,
        description="Absolute Sandbox directory for collision-safe Agent result envelopes.",
    )
    workdir: str | None = Field(
        default=None,
        description="Optional absolute working directory for both the DSH Agent and verifier.",
    )
    require_trace: bool = Field(
        default=True,
        description="Require the DSH Agent to persist a readable canonical event trace.",
    )
    environment_digest: str | None = Field(
        default=None,
        description="Optional operator-pinned Sandbox/environment digest; otherwise read from task metadata.",
    )
    verifier_id: str | None = Field(
        default=None,
        description="Optional operator-pinned verifier identifier; otherwise read from task metadata.",
    )
    verifier_version: str | None = Field(
        default=None,
        description="Optional operator-pinned verifier version; otherwise read from task metadata.",
    )
    verifier_code_digest: str | None = Field(
        default=None,
        description="Optional operator-pinned verifier code digest; otherwise read from task metadata.",
    )

    @field_validator("verifier_command")
    @classmethod
    def _validate_verifier_command(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("dsh_architecture verifier_command must contain non-empty argv entries")
        return value

    @field_validator("result_root", "workdir")
    @classmethod
    def _validate_absolute_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("dsh_architecture paths must not be blank")
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"dsh_architecture path must be absolute and traversal-free, got {value!r}")
        return value

    @field_validator("environment_digest", "verifier_code_digest")
    @classmethod
    def _validate_optional_digest(cls, value: str | None) -> str | None:
        if value is not None and _HASH_PATTERN.fullmatch(value) is None:
            raise ValueError("dsh_architecture identity digests must match sha256:<64 lowercase hex digits>")
        return value

    @field_validator("verifier_id")
    @classmethod
    def _validate_optional_identifier(cls, value: str | None) -> str | None:
        if value is not None and _IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise ValueError("dsh_architecture verifier_id has an invalid identifier")
        return value

    @field_validator("verifier_version")
    @classmethod
    def _validate_optional_version(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("dsh_architecture verifier_version must not be blank")
        return value


def _canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON value deterministically for envelopes and audit hashes."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DSH task value is not JSON serializable: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _required_info(info: dict[str, Any], key: str) -> str:
    value = info.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"dsh_architecture Agent result is missing string info[{key!r}]")
    return value


def _identity_value(
    config_value: str | None,
    metadata: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
) -> str:
    """Resolve an operator pin and dataset identity, rejecting disagreements."""
    metadata_values = [metadata.get(name) for name in (key, *aliases) if metadata.get(name) is not None]
    string_values = [value for value in metadata_values if isinstance(value, str)]
    if len(set(string_values)) > 1 or len(string_values) != len(metadata_values):
        raise RuntimeError(f"dsh_architecture metadata has conflicting {key} values")
    metadata_value = metadata_values[0] if metadata_values else None
    if config_value is not None and metadata_value is not None and config_value != metadata_value:
        raise RuntimeError(f"dsh_architecture operator pin {key} disagrees with task metadata")
    value = config_value if config_value is not None else metadata_value
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"dsh_architecture requires identity field {key!r}")
    return value


def _task_identity(config: DshArchitectureTaskConfig) -> dict[str, str]:
    """Resolve the immutable task, environment, and verifier identity."""
    metadata = config.metadata
    task_id = metadata.get("task_id")
    task_version = metadata.get("task_version")
    split = metadata.get("split")
    if not isinstance(task_id, str) or _IDENTIFIER_PATTERN.fullmatch(task_id) is None:
        raise RuntimeError("dsh_architecture metadata.task_id has an invalid identifier")
    if not isinstance(task_version, str) or not task_version.strip():
        raise RuntimeError("dsh_architecture metadata.task_version must be non-empty")
    if split not in _SPLITS:
        raise RuntimeError("dsh_architecture metadata.split must be train, validation, test, or holdout")
    environment_digest = _identity_value(
        config.environment_digest,
        metadata,
        "environment_digest",
        aliases=("environment_ref",),
    )
    verifier_id = _identity_value(config.verifier_id, metadata, "verifier_id")
    verifier_version = _identity_value(config.verifier_version, metadata, "verifier_version")
    verifier_code_digest = _identity_value(config.verifier_code_digest, metadata, "verifier_code_digest")
    if _HASH_PATTERN.fullmatch(environment_digest) is None:
        raise RuntimeError("dsh_architecture environment_digest must be sha256:<64 lowercase hex digits>")
    if _IDENTIFIER_PATTERN.fullmatch(verifier_id) is None:
        raise RuntimeError("dsh_architecture verifier_id has an invalid identifier")
    if _HASH_PATTERN.fullmatch(verifier_code_digest) is None:
        raise RuntimeError("dsh_architecture verifier_code_digest must be sha256:<64 lowercase hex digits>")
    return {
        "task_id": task_id,
        "task_version": task_version,
        "split": split,
        "environment_digest": environment_digest,
        "verifier_id": verifier_id,
        "verifier_version": verifier_version,
        "verifier_code_digest": verifier_code_digest,
    }


def _issued_at(value: object) -> str:
    """Validate a verifier timestamp or provide the current UTC issuance time."""
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("dsh_architecture verifier field 'issued_at' must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("dsh_architecture verifier field 'issued_at' must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise RuntimeError("dsh_architecture verifier field 'issued_at' must use UTC")
    return value


def _evidence(value: object) -> list[str]:
    """Validate the verifier evidence list bound into a fresh receipt."""
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise RuntimeError("dsh_architecture verifier field 'evidence' must be a non-empty string list")
    return list(value)


def _artifact_key(info: dict[str, Any]) -> str:
    """Derive a collision-resistant, non-secret result directory key."""
    session_id = _required_info(info, "dsh_session_id")
    trace_sha256 = _required_info(info, "trace_sha256")
    if _HASH_PATTERN.fullmatch(trace_sha256) is None:
        raise RuntimeError("dsh_architecture Agent result has an invalid trace_sha256")
    return hashlib.sha256(f"{session_id}\x00{trace_sha256}".encode()).hexdigest()[:24]


def _agent_envelope(config: DshArchitectureTaskConfig, agent_result: Any, result_path: str) -> dict[str, Any]:
    """Build verifier input without copying model credentials or transcripts."""
    info = agent_result.info
    selected_info = {key: info[key] for key in _INFO_KEYS if key in info}
    trace_path = selected_info.get("trace_path")
    if config.require_trace:
        if selected_info.get("keep_trace") is not True:
            raise RuntimeError("dsh_architecture requires keep_trace=true from the DSH Agent")
        if not isinstance(trace_path, str) or not trace_path:
            raise RuntimeError("dsh_architecture requires a persisted trace_path")
    response = agent_result.output.get("response") if isinstance(agent_result.output, dict) else None
    if not isinstance(response, str):
        raise RuntimeError("dsh_architecture Agent result must contain a string output.response")
    envelope = {
        "schema": "dsh.uni-agent.task-result.v1",
        "task_name": config.name,
        "prompt": config.prompt,
        "metadata": config.metadata,
        "response": response,
        "finished": agent_result.finished,
        "dsh": selected_info,
        "result_path": result_path,
    }
    _canonical_json_bytes(envelope)
    return envelope


def _parse_verifier_output(stdout: str) -> dict[str, Any]:
    """Parse the verifier's single-object stdout protocol."""
    text = stdout.strip()
    if not text:
        raise RuntimeError("dsh_architecture verifier emitted no JSON result")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("dsh_architecture verifier stdout must be exactly one JSON object") from exc
    if not isinstance(value, dict):
        raise RuntimeError("dsh_architecture verifier result must be a JSON object")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"dsh_architecture verifier field {field!r} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"dsh_architecture verifier field {field!r} must be a finite number")
    return number


def _task_result(
    verifier_result: dict[str, Any],
    agent_result: Any,
    *,
    identity: dict[str, str],
    artifact_sha256: str,
    verifier_command: list[str],
    verifier_stdout: str,
) -> tuple[TaskResult, dict[str, Any]]:
    """Convert a validated verifier object into the framework result."""
    reward = _finite_number(verifier_result.get("reward"), field="reward")
    raw_accuracy = verifier_result.get("accuracy")
    accuracy = None if raw_accuracy is None else _finite_number(raw_accuracy, field="accuracy")
    raw_finished = verifier_result.get("finished", agent_result.finished)
    if raw_finished is not None and type(raw_finished) is not bool:
        raise RuntimeError("dsh_architecture verifier field 'finished' must be boolean or null")
    extra_info = verifier_result.get("extra_info", {})
    if not isinstance(extra_info, dict):
        raise RuntimeError("dsh_architecture verifier field 'extra_info' must be an object")
    if verifier_result.get("fresh") is not True:
        raise RuntimeError("dsh_architecture verifier must declare fresh=true")
    evidence = _evidence(verifier_result.get("evidence"))
    issued_at = _issued_at(verifier_result.get("issued_at"))
    command_sha256 = f"sha256:{hashlib.sha256(_canonical_json_bytes(verifier_command)).hexdigest()}"
    stdout_sha256 = f"sha256:{hashlib.sha256(verifier_stdout.encode('utf-8')).hexdigest()}"
    receipt_without_id = {
        "schema": "dsh.verifier-receipt.v1",
        "task_id": identity["task_id"],
        "task_version": identity["task_version"],
        "dsh_session_id": _required_info(agent_result.info, "dsh_session_id"),
        "trace_sha256": _required_info(agent_result.info, "trace_sha256"),
        "artifact_sha256": artifact_sha256,
        "environment_digest": identity["environment_digest"],
        "verifier": {
            "id": identity["verifier_id"],
            "version": identity["verifier_version"],
            "code_digest": identity["verifier_code_digest"],
        },
        "issued_at": issued_at,
        "fresh": True,
        "issuer": {"kind": "trusted-verifier", "id": "uni-agent-dsh"},
        "reward": reward,
        **({"accuracy": accuracy} if accuracy is not None else {}),
        **({"finished": raw_finished} if raw_finished is not None else {}),
        "evidence": evidence,
    }
    receipt_id = f"sha256:{hashlib.sha256(_canonical_json_bytes(receipt_without_id)).hexdigest()}"
    receipt = {"receipt_id": receipt_id, **receipt_without_id}
    reward_info = {
        "dsh": {
            "schema": "dsh.verifier-receipt.v1",
            "receipt_sha256": receipt_id,
            "freshness": "fresh",
            "rollout_id": _required_info(agent_result.info, "gateway_session_id"),
            "task_id": identity["task_id"],
            "task_version": identity["task_version"],
            "split": identity["split"],
            "dsh_session_id": receipt["dsh_session_id"],
            "trace_sha256": receipt["trace_sha256"],
            "artifact_sha256": artifact_sha256,
            "environment_digest": identity["environment_digest"],
            "verifier_id": identity["verifier_id"],
            "verifier_version": identity["verifier_version"],
            "event_count": agent_result.info.get("event_count", 0),
        }
    }
    return TaskResult(
        reward=reward,
        accuracy=accuracy,
        finished=raw_finished,
        extra_info={
            "agent": dict(agent_result.info),
            "verifier": {
                "command_sha256": command_sha256,
                "stdout_sha256": stdout_sha256,
                "extra_info": extra_info,
                "receipt_sha256": receipt_id,
            },
        },
        reward_info=reward_info,
    ), receipt


@register_task("dsh_architecture")
class DshArchitectureTask(Task):
    """Score DSH architecture and self-evolution tasks with an operator verifier."""

    name = "dsh_architecture"
    config_model = DshArchitectureTaskConfig

    async def run(self) -> TaskResult:
        cfg: DshArchitectureTaskConfig = self.config  # type: ignore[assignment]
        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            agent_result = await agent.run(
                sandbox=sandbox,
                messages=cfg.prompt,
                workdir=cfg.workdir,
            )
            result_path = f"{cfg.result_root.rstrip('/')}/{_artifact_key(agent_result.info)}/agent-result.json"
            envelope = _agent_envelope(cfg, agent_result, result_path)
            envelope_bytes = _canonical_json_bytes(envelope)
            await sandbox.write_file(result_path, envelope_bytes)
            protect_result = await sandbox.exec(["chmod", "600", result_path], timeout=10)
            if protect_result.exit_code != 0:
                raise RuntimeError("dsh_architecture could not protect its result envelope")
            artifact_sha256 = f"sha256:{hashlib.sha256(envelope_bytes).hexdigest()}"
            receipt_path = f"{result_path.rsplit('/', 1)[0]}/verifier-receipt.json"
            identity = _task_identity(cfg)

            trace_path = agent_result.info.get("trace_path")
            if cfg.require_trace:
                assert isinstance(trace_path, str)
                trace_probe = await sandbox.exec(["test", "-f", trace_path], timeout=10)
                if trace_probe.exit_code != 0:
                    raise RuntimeError(f"dsh_architecture trace artifact is not readable: {trace_path!r}")
                trace_bytes = await sandbox.read_file(trace_path)
                actual_trace_sha256 = f"sha256:{hashlib.sha256(trace_bytes).hexdigest()}"
                expected_trace_sha256 = _required_info(agent_result.info, "trace_sha256")
                if actual_trace_sha256 != expected_trace_sha256:
                    raise RuntimeError("dsh_architecture trace bytes do not match trace_sha256")

            env = {
                "DSH_TASK_RESULT_PATH": result_path,
                "DSH_DSH_SESSION_ID": _required_info(agent_result.info, "dsh_session_id"),
                "DSH_TRACE_SHA256": _required_info(agent_result.info, "trace_sha256"),
                "DSH_ARTIFACT_SHA256": artifact_sha256,
                "DSH_TRACE_PATH": str(agent_result.info.get("trace_path") or ""),
                "DSH_TASK_NAME": cfg.name,
                "DSH_TASK_ID": identity["task_id"],
                "DSH_TASK_VERSION": identity["task_version"],
                "DSH_TASK_SPLIT": identity["split"],
                "DSH_ENVIRONMENT_DIGEST": identity["environment_digest"],
                "DSH_VERIFIER_ID": identity["verifier_id"],
                "DSH_VERIFIER_VERSION": identity["verifier_version"],
                "DSH_VERIFIER_CODE_DIGEST": identity["verifier_code_digest"],
            }
            verification = await sandbox.exec(
                list(cfg.verifier_command),
                timeout=cfg.verifier_timeout,
                workdir=cfg.workdir,
                env=env,
            )
            if verification.exit_code == -1:
                raise TimeoutError(f"dsh_architecture verifier exceeded {cfg.verifier_timeout:g}s")
            if verification.exit_code != 0:
                detail = (verification.stderr or verification.stdout or "verifier failed").strip()[-2000:]
                raise RuntimeError(f"dsh_architecture verifier exited {verification.exit_code}: {detail}")
            verifier_result = _parse_verifier_output(verification.stdout)
            result, receipt = _task_result(
                verifier_result,
                agent_result,
                identity=identity,
                artifact_sha256=artifact_sha256,
                verifier_command=list(cfg.verifier_command),
                verifier_stdout=verification.stdout,
            )
            await sandbox.write_file(receipt_path, _canonical_json_bytes(receipt))
            protect_receipt = await sandbox.exec(["chmod", "600", receipt_path], timeout=10)
            if protect_receipt.exit_code != 0:
                raise RuntimeError("dsh_architecture could not protect its verifier receipt")
            logger.info(
                "dsh_architecture task complete: session=%s reward=%s",
                agent_result.info.get("dsh_session_id"),
                result.reward,
            )
            return result
