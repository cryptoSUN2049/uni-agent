from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from uni_agent.agents.dsh import runner


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def test_runner_persists_canonical_trace_and_passes_gateway_config(monkeypatch, tmp_path: Path) -> None:
    events = [{"type": "session/start", "data": {"id": "dsh-1"}}, {"type": "turn/end", "data": {}}]
    calls: dict[str, object] = {}

    class FakeHarness:
        def __init__(self, config) -> None:
            calls["config"] = config

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def run(self, prompt, *, session_id):
            calls["prompt"] = prompt
            calls["session_id"] = session_id
            return SimpleNamespace(
                session_id=session_id,
                final_response="done",
                finish_reason="completed",
                events=events,
            )

    class FakeConfig:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "deepseek_harness",
        SimpleNamespace(DeepSeekHarness=FakeHarness, DeepSeekHarnessConfig=FakeConfig),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "nested" / "result.json"
    trace_path = tmp_path / "trace" / "session.jsonl"
    input_path.write_text(json.dumps({"prompt": "learn DSH", "session_id": "dsh-1"}), encoding="utf-8")
    env = {
        "DSH_UA_BASE_URL": "http://127.0.0.1:1234/sessions/s1/v1",
        "DSH_UA_API_KEY": "EMPTY",
        "DSH_UA_MODEL": "Qwen3-27B",
        "DSH_UA_PROVIDER": "deepseek-official",
        "DSH_UA_PROFILE": "sdk",
        "DSH_UA_HOME": str(tmp_path / "home"),
        "DSH_UA_CWD": str(tmp_path),
        "DSH_UA_TRACE_PATH": str(trace_path),
        "DSH_UA_KEEP_TRACE": "1",
        "DSH_UA_MAX_TOKENS": "4096",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    result = runner.run(input_path, output_path)

    trace_bytes = b"".join(
        (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for event in events
    )
    assert result["trace_sha256"] == _digest(trace_bytes)
    assert trace_path.read_bytes() == trace_bytes
    assert json.loads(output_path.read_text(encoding="utf-8"))["final_response"] == "done"
    assert trace_path.stat().st_mode & 0o777 == 0o600
    assert output_path.stat().st_mode & 0o777 == 0o600
    config = calls["config"]
    assert config.model == "Qwen3-27B"
    assert config.base_url == env["DSH_UA_BASE_URL"]
    assert calls["prompt"] == "learn DSH"
    assert calls["session_id"] == "dsh-1"
    assert not input_path.exists()


def test_runner_rejects_invalid_input_before_starting_harness(monkeypatch, tmp_path: Path) -> None:
    class ShouldNotStart:
        def __init__(self, _config) -> None:
            raise AssertionError("harness should not start")

    monkeypatch.setitem(
        sys.modules,
        "deepseek_harness",
        SimpleNamespace(DeepSeekHarness=ShouldNotStart, DeepSeekHarnessConfig=object),
    )
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"prompt": "", "session_id": "dsh-1"}), encoding="utf-8")

    try:
        runner.run(input_path, tmp_path / "result.json")
    except RuntimeError as exc:
        assert "non-empty string prompt" in str(exc)
    else:
        raise AssertionError("expected invalid input failure")
