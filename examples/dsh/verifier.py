#!/usr/bin/env python3
"""Deterministic S0 verifier for the Uni-Agent DSH adapter.

This verifier is intentionally a smoke fixture, not a benchmark judge. It
checks that the task envelope and semantic trace are the exact artifacts named
by the task process, then awards ``1`` only when the response hash in the
released seed matches. A real DSH benchmark should replace this command with
its trusted verifier while preserving the same stdout protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$", re.ASCII)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.strip():
        raise RuntimeError(f"missing verifier environment variable {name}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _load_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"verifier input {path} must be a JSON object")
    return value, raw


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a sha256 digest")
    return value


def verify() -> dict[str, Any]:
    """Validate one DSH task envelope and return the fresh score object."""
    result_path = Path(_required_env("DSH_TASK_RESULT_PATH"))
    envelope, envelope_bytes = _load_object(result_path)
    if envelope.get("schema") != "dsh.uni-agent.task-result.v1":
        raise RuntimeError("verifier input has the wrong task-result schema")

    expected_artifact = _require_digest(_required_env("DSH_ARTIFACT_SHA256"), label="DSH_ARTIFACT_SHA256")
    actual_artifact = _sha256_bytes(envelope_bytes)
    if actual_artifact != expected_artifact:
        raise RuntimeError("task envelope hash does not match DSH_ARTIFACT_SHA256")

    dsh = envelope.get("dsh")
    if not isinstance(dsh, dict):
        raise RuntimeError("verifier input is missing the DSH trace metadata")
    expected_dsh_session = _required_env("DSH_DSH_SESSION_ID")
    if dsh.get("dsh_session_id") != expected_dsh_session:
        raise RuntimeError("task envelope DSH session id does not match DSH_DSH_SESSION_ID")
    expected_trace = _require_digest(_required_env("DSH_TRACE_SHA256"), label="DSH_TRACE_SHA256")
    if dsh.get("trace_sha256") != expected_trace:
        raise RuntimeError("task envelope trace hash does not match DSH_TRACE_SHA256")
    trace_path = Path(_required_env("DSH_TRACE_PATH"))
    trace_bytes = trace_path.read_bytes()
    if _sha256_bytes(trace_bytes) != expected_trace:
        raise RuntimeError("DSH trace bytes do not match DSH_TRACE_SHA256")
    trace_events = [json.loads(line) for line in trace_bytes.decode("utf-8").splitlines() if line.strip()]
    if not trace_events or not any(
        isinstance(event, dict) and event.get("type") == "turn/end" for event in trace_events
    ):
        raise RuntimeError("DSH trace must contain a terminal turn/end event")

    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("verifier input metadata must be an object")
    for env_name, metadata_name in (
        ("DSH_TASK_ID", "task_id"),
        ("DSH_TASK_VERSION", "task_version"),
        ("DSH_ENVIRONMENT_DIGEST", "environment_digest"),
        ("DSH_VERIFIER_ID", "verifier_id"),
        ("DSH_VERIFIER_VERSION", "verifier_version"),
        ("DSH_VERIFIER_CODE_DIGEST", "verifier_code_digest"),
    ):
        expected_value = _required_env(env_name)
        if metadata.get(metadata_name) != expected_value:
            raise RuntimeError(f"task metadata.{metadata_name} does not match {env_name}")
        if metadata_name.endswith("digest"):
            _require_digest(expected_value, label=env_name)

    expected_response = metadata.get("expected_response_sha256")
    if not isinstance(expected_response, str):
        raise RuntimeError("smoke seed metadata.expected_response_sha256 is required")
    _require_digest(expected_response, label="metadata.expected_response_sha256")
    response = envelope.get("response")
    if not isinstance(response, str):
        raise RuntimeError("verifier input response must be a string")
    actual_response = _sha256_bytes(response.encode("utf-8"))
    matched = actual_response == expected_response
    finished = envelope.get("finished")
    if finished is not None and type(finished) is not bool:
        raise RuntimeError("verifier input finished must be boolean or null")
    return {
        "reward": 1.0 if matched else 0.0,
        "accuracy": 1.0 if matched else 0.0,
        "eligible": True,
        "finished": finished,
        "fresh": True,
        "issued_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "evidence": [
            f"trace_sha256:{expected_trace}",
            f"artifact_sha256:{expected_artifact}",
            f"response_sha256:{actual_response}",
            f"environment_digest:{_required_env('DSH_ENVIRONMENT_DIGEST')}",
        ],
        "extra_info": {
            "smoke": True,
            "expected_response_sha256": expected_response,
            "actual_response_sha256": actual_response,
            "matched": matched,
        },
    }


def main() -> None:
    """Print exactly one JSON result or fail on stderr."""
    try:
        result = verify()
    except Exception as exc:  # noqa: BLE001 - CLI must report a concise failure
        print(f"dsh smoke verifier failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
