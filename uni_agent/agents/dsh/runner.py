"""Sandbox-side helper that launches the DeepSeek Harness Python SDK.

The module is intentionally optional: importing it does not require the DSH SDK.
The selected rollout image must install ``deepseek-harness-sdk`` (and its bundled
runtime) before invoking this helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


def _env(name: str, *, required: bool = True) -> str | None:
    value = os.environ.get(name)
    if required and (value is None or not value.strip()):
        raise RuntimeError(f"missing required DSH adapter environment variable {name}")
    return value


def _canonical_event_bytes(events: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for event in events
    )


def _write_private(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(content)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _patches_from_env() -> tuple[str, ...]:
    """Decode ordered profile patch paths across the helper process boundary."""
    raw = os.environ.get("DSH_UA_PATCHES", "[]")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("DSH_UA_PATCHES must be a JSON array") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise RuntimeError("DSH_UA_PATCHES must be a JSON array of non-empty paths")
    for item in value:
        if ".." in PurePosixPath(item).parts:
            raise RuntimeError("DSH_UA_PATCHES paths must not contain traversal")
    return tuple(value)


def _patches_digest(patches: tuple[str, ...]) -> str:
    """Return a non-secret identity for the ordered profile patch stack."""
    encoded = json.dumps(list(patches), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def run(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Run one DSH SDK session and persist its semantic event JSONL."""
    try:
        # Validate process-boundary controls before importing the optional SDK;
        # malformed operator configuration should fail deterministically even
        # in a minimal test image that does not install the SDK.
        patches = _patches_from_env()
        profile = os.environ.get("DSH_UA_PROFILE") or "sdk"
        if not profile.strip():
            raise RuntimeError("DSH_UA_PROFILE must be non-empty")
        from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
            raise RuntimeError("DSH adapter input must contain a non-empty string prompt")
        dsh_session_id = payload.get("session_id")
        if not isinstance(dsh_session_id, str) or not dsh_session_id:
            raise RuntimeError("DSH adapter input must contain a session_id")
        trace_path = Path(_env("DSH_UA_TRACE_PATH") or "")
        keep_trace = (_env("DSH_UA_KEEP_TRACE") or "1") == "1"
        model = _env("DSH_UA_MODEL")
        cwd = _env("DSH_UA_CWD")
        config = DeepSeekHarnessConfig(
            provider=_env("DSH_UA_PROVIDER") or "deepseek-official",
            model=model or "",
            reasoning_effort=_env("DSH_UA_REASONING_EFFORT", required=False),
            max_tokens=int(_env("DSH_UA_MAX_TOKENS", required=False) or "0") or None,
            cwd=cwd or "/workspace",
            runtime_cwd=cwd or "/workspace",
            profile=profile,
            patches=patches,
            dsh_home=_env("DSH_UA_HOME"),
            base_url=_env("DSH_UA_BASE_URL"),
            api_key=_env("DSH_UA_API_KEY") or "EMPTY",
        )
        with DeepSeekHarness(config) as harness:
            result = harness.run(payload["prompt"], session_id=dsh_session_id)
        event_bytes = _canonical_event_bytes(result.events)
        trace_sha256 = f"sha256:{hashlib.sha256(event_bytes).hexdigest()}"
        if keep_trace:
            _write_private(trace_path, event_bytes)
        output = {
            "schema": "dsh.uni-agent.dsh-run.v1",
            "dsh_session_id": result.session_id,
            "trace_sha256": trace_sha256,
            "trace_path": str(trace_path),
            "event_count": len(result.events),
            "finish_reason": result.finish_reason,
            "final_response": result.final_response,
            "trace_persisted": keep_trace,
            "profile": profile,
            "patches_sha256": _patches_digest(patches),
        }
        _write_private(output_path, (json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        return output
    finally:
        try:
            input_path.unlink()
        except OSError:
            pass


def main() -> None:
    """Parse helper paths and run one DSH session."""
    parser = argparse.ArgumentParser(description="Run DeepSeek Harness inside a Uni-Agent Sandbox")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
