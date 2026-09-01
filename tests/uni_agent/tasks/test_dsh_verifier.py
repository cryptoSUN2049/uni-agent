import hashlib
import json
from pathlib import Path

from examples.dsh.verifier import verify


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def test_smoke_verifier_checks_exact_trace_and_scores_response(monkeypatch, tmp_path: Path) -> None:
    trace = b'{"type":"session/start"}\n{"type":"turn/end"}\n'
    trace_path = tmp_path / "session.jsonl"
    trace_path.write_bytes(trace)
    envelope = {
        "schema": "dsh.uni-agent.task-result.v1",
        "metadata": {
            "task_id": "dsh/architecture/intro",
            "task_version": "1",
            "environment_digest": _digest(b"environment"),
            "verifier_id": "dsh-fixture-verifier",
            "verifier_version": "1",
            "verifier_code_digest": _digest(b"verifier"),
            "expected_response_sha256": _digest(b"answer"),
        },
        "response": "answer",
        "finished": True,
        "dsh": {"dsh_session_id": "dsh-session-1", "trace_sha256": _digest(trace)},
    }
    envelope_path = tmp_path / "agent-result.json"
    envelope_bytes = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    envelope_path.write_bytes(envelope_bytes)
    env = {
        "DSH_TASK_RESULT_PATH": str(envelope_path),
        "DSH_DSH_SESSION_ID": "dsh-session-1",
        "DSH_ARTIFACT_SHA256": _digest(envelope_bytes),
        "DSH_TRACE_SHA256": _digest(trace),
        "DSH_TRACE_PATH": str(trace_path),
        "DSH_TASK_ID": "dsh/architecture/intro",
        "DSH_TASK_VERSION": "1",
        "DSH_ENVIRONMENT_DIGEST": _digest(b"environment"),
        "DSH_VERIFIER_ID": "dsh-fixture-verifier",
        "DSH_VERIFIER_VERSION": "1",
        "DSH_VERIFIER_CODE_DIGEST": _digest(b"verifier"),
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    result = verify()

    assert result["reward"] == 1.0
    assert result["fresh"] is True
    assert result["extra_info"]["matched"] is True


def test_smoke_verifier_rejects_trace_hash_mismatch(monkeypatch, tmp_path: Path) -> None:
    trace_path = tmp_path / "session.jsonl"
    trace_path.write_bytes(b"actual\n")
    envelope = {
        "schema": "dsh.uni-agent.task-result.v1",
        "metadata": {
            "task_id": "dsh/architecture/intro",
            "task_version": "1",
            "expected_response_sha256": _digest(b"answer"),
        },
        "response": "answer",
        "finished": True,
        "dsh": {"dsh_session_id": "dsh-session-1", "trace_sha256": _digest(b"different\n")},
    }
    envelope_path = tmp_path / "agent-result.json"
    envelope_bytes = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    envelope_path.write_bytes(envelope_bytes)
    monkeypatch.setenv("DSH_TASK_RESULT_PATH", str(envelope_path))
    monkeypatch.setenv("DSH_DSH_SESSION_ID", "dsh-session-1")
    monkeypatch.setenv("DSH_ARTIFACT_SHA256", _digest(envelope_bytes))
    monkeypatch.setenv("DSH_TRACE_SHA256", _digest(b"different\n"))
    monkeypatch.setenv("DSH_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("DSH_TASK_ID", "dsh/architecture/intro")
    monkeypatch.setenv("DSH_TASK_VERSION", "1")
    monkeypatch.setenv("DSH_ENVIRONMENT_DIGEST", _digest(b"environment"))
    monkeypatch.setenv("DSH_VERIFIER_ID", "dsh-fixture-verifier")
    monkeypatch.setenv("DSH_VERIFIER_VERSION", "1")
    monkeypatch.setenv("DSH_VERIFIER_CODE_DIGEST", _digest(b"verifier"))

    try:
        verify()
    except RuntimeError as exc:
        assert "trace bytes" in str(exc)
    else:
        raise AssertionError("expected trace mismatch")
