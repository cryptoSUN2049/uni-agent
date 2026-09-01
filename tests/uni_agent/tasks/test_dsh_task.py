from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from uni_agent.agents.base import AgentConfig, AgentResult
from uni_agent.sandbox.base import ExecResult
from uni_agent.tasks import TaskConfigResolver
from uni_agent.tasks.dsh.task import (
    DshArchitectureTask,
    DshArchitectureTaskConfig,
    _parse_verifier_output,
)


class _FakeSandbox:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.trace_bytes = b'{"type":"session/start"}\n'
        self.writes: dict[str, bytes] = {}
        self.calls: list[dict] = []

    async def __aenter__(self) -> _FakeSandbox:
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.writes[path] = content.encode() if isinstance(content, str) else content

    async def read_file(self, path: str) -> bytes:
        if path.endswith(".jsonl"):
            return self.trace_bytes
        return self.writes[path]

    async def exec(self, argv, *, timeout=None, workdir=None, env=None) -> ExecResult:
        self.calls.append({"argv": argv, "timeout": timeout, "workdir": workdir, "env": env})
        if argv[:2] == ["test", "-f"]:
            return ExecResult(exit_code=0, stdout="", stderr="")
        if argv[:1] == ["chmod"]:
            return ExecResult(exit_code=0, stdout="", stderr="")
        return ExecResult(exit_code=0, stdout=self.stdout, stderr="")


class _FakeAgent:
    async def run(self, *, sandbox, messages, workdir=None) -> AgentResult:
        return AgentResult(
            output={"response": "DSH answer"},
            info={
                "adapter": "uni-agent-dsh",
                "dsh_session_id": "dsh-session-1",
                "gateway_session_id": "session-1",
                "trace_sha256": "sha256:a8086182c96628a0d7b3d9359662e64ac83ad11d4f7a98453fa4ee28071a9ac8",
                "trace_path": "/tmp/traces/session-1.jsonl",
                "event_count": 4,
                "finish_reason": "completed",
                "keep_trace": True,
            },
            finished=True,
        )


class _HarnessTask(DshArchitectureTask):
    def __init__(self, config, sandbox, agent) -> None:
        super().__init__(config)
        self._sandbox = sandbox
        self._agent = agent

    def build_sandbox(self):
        return self._sandbox

    def build_agent(self):
        return self._agent


def _config(**overrides) -> DshArchitectureTaskConfig:
    values = {
        "verifier_command": ["verify-dsh", "--json"],
        "sandbox": {"provider": "local"},
        "agent": AgentConfig(name="fake"),
        "prompt": [{"role": "user", "content": "Explain the plugin seam."}],
        "metadata": {
            "task_id": "dsh/architecture/intro",
            "task_version": "1",
            "split": "train",
            "environment_digest": "sha256:" + "b" * 64,
            "verifier_id": "dsh-fixture-verifier",
            "verifier_version": "1",
            "verifier_code_digest": "sha256:" + "c" * 64,
        },
        "result_root": "/tmp/results",
    }
    values.update(overrides)
    return DshArchitectureTaskConfig(**values)


def test_config_rejects_untrusted_path_and_empty_verifier_argv() -> None:
    with pytest.raises(ValueError, match="verifier_command"):
        _config(verifier_command=[])
    with pytest.raises(ValueError, match="absolute"):
        _config(result_root="relative/results")
    with pytest.raises(ValueError, match="traversal"):
        _config(workdir="/tmp/../workspace")


def test_task_writes_minimal_envelope_and_returns_verifier_reward() -> None:
    sandbox = _FakeSandbox(
        json.dumps(
            {
                "reward": 0.75,
                "accuracy": 0.5,
                "fresh": True,
                "issued_at": "2026-09-01T00:00:00Z",
                "evidence": ["answer.json"],
                "extra_info": {"rubric": "all"},
            }
        )
    )
    result = asyncio.run(_HarnessTask(_config(), sandbox, _FakeAgent()).run())

    assert result.reward == 0.75
    assert result.accuracy == 0.5
    assert result.finished is True
    assert result.extra_info["verifier"]["extra_info"] == {"rubric": "all"}
    assert len(sandbox.writes) == 2
    envelope_path = next(path for path in sandbox.writes if path.endswith("agent-result.json"))
    envelope = json.loads(sandbox.writes[envelope_path])
    assert envelope["schema"] == "dsh.uni-agent.task-result.v1"
    assert envelope["response"] == "DSH answer"
    assert "api_key" not in json.dumps(envelope)
    assert result.reward_info is not None
    dsh_info = result.reward_info["dsh"]
    assert dsh_info["freshness"] == "fresh"
    assert dsh_info["task_id"] == "dsh/architecture/intro"
    assert dsh_info["receipt_sha256"].startswith("sha256:")
    assert "receipt_path" not in dsh_info
    receipt_paths = [path for path in sandbox.writes if path.endswith("verifier-receipt.json")]
    assert len(receipt_paths) == 1
    receipt = json.loads(sandbox.writes[receipt_paths[0]])
    assert receipt["schema"] == "dsh.verifier-receipt.v1"
    assert receipt["fresh"] is True
    assert receipt["artifact_sha256"].startswith("sha256:")
    verifier_call = next(call for call in sandbox.calls if call["argv"] == ["verify-dsh", "--json"])
    assert verifier_call["argv"] == ["verify-dsh", "--json"]
    assert verifier_call["env"]["DSH_DSH_SESSION_ID"] == "dsh-session-1"
    assert verifier_call["env"]["DSH_TRACE_SHA256"].startswith("sha256:")
    assert verifier_call["env"]["DSH_TASK_ID"] == "dsh/architecture/intro"
    assert verifier_call["env"]["DSH_ENVIRONMENT_DIGEST"].startswith("sha256:")


def test_task_rejects_nonfinite_or_nonobject_verifier_results() -> None:
    for stdout, message in [
        ('{"reward": NaN}', "finite"),
        ('{"reward": 1, "extra_info": []}', "extra_info"),
        ("[]", "JSON object"),
    ]:
        sandbox = _FakeSandbox(stdout)
        with pytest.raises(RuntimeError, match=message):
            asyncio.run(_HarnessTask(_config(), sandbox, _FakeAgent()).run())


def test_task_rejects_historical_verifier_reward() -> None:
    sandbox = _FakeSandbox(json.dumps({"reward": 1.0, "fresh": False, "evidence": ["old.json"]}))
    with pytest.raises(RuntimeError, match="fresh=true"):
        asyncio.run(_HarnessTask(_config(), sandbox, _FakeAgent()).run())


def test_task_rejects_trace_bytes_that_do_not_match_agent_hash() -> None:
    sandbox = _FakeSandbox(json.dumps({"reward": 1.0, "fresh": True, "evidence": ["ok"]}))
    sandbox.trace_bytes = b"tampered trace\n"
    with pytest.raises(RuntimeError, match="trace bytes"):
        asyncio.run(_HarnessTask(_config(), sandbox, _FakeAgent()).run())


def test_task_rejects_operator_and_metadata_identity_mismatch() -> None:
    config = _config(verifier_id="other-verifier")
    sandbox = _FakeSandbox(json.dumps({"reward": 1.0, "fresh": True, "evidence": ["answer.json"]}))
    with pytest.raises(RuntimeError, match="operator pin verifier_id"):
        asyncio.run(_HarnessTask(config, sandbox, _FakeAgent()).run())


def test_verifier_stdout_is_one_json_object_without_log_lines() -> None:
    with pytest.raises(RuntimeError, match="exactly one JSON object"):
        _parse_verifier_output('debug line\n{"reward": 1}')


def test_dataset_row_cannot_override_operator_runtime_controls() -> None:
    resolver = TaskConfigResolver(
        {
            "dsh_architecture": {
                "name": "dsh_architecture",
                "sandbox": {"provider": "local"},
                "agent": {"name": "dsh"},
                "verifier_command": ["trusted-verifier"],
                "verifier_timeout": 30,
                "result_root": "/tmp/results",
                "workdir": None,
                "require_trace": True,
                "environment_digest": "sha256:" + "b" * 64,
                "verifier_id": "dsh-fixture-verifier",
                "verifier_version": "1",
                "verifier_code_digest": "sha256:" + "c" * 64,
            }
        }
    )
    with pytest.raises(ValueError, match="task-config-only"):
        resolver.resolve(
            {
                "name": "dsh_architecture",
                "metadata": {},
                "verifier_command": ["sample-controlled-verifier"],
            }
        )

    resolved = resolver.resolve({"name": "dsh_architecture", "metadata": {"task_id": "safe"}})
    assert resolved["verifier_command"] == ["trusted-verifier"]


def test_example_task_config_contains_all_operator_only_keys() -> None:
    resolver = TaskConfigResolver.from_file("examples/dsh/task_config.yaml")

    resolved = resolver.resolve(
        {
            "name": "dsh_architecture",
            "metadata": {
                "task_id": "dsh/architecture/intro",
                "task_version": "1",
                "split": "train",
                "environment_digest": "sha256:" + "b" * 64,
                "verifier_id": "dsh-fixture-verifier",
                "verifier_version": "1",
                "verifier_code_digest": "sha256:" + "c" * 64,
            },
        }
    )

    assert resolved["workdir"] is None
    assert resolved["agent"]["model"]["max_total_tokens"] == 1024
    assert resolved["agent"]["model"]["max_tokens_per_turn"] == 1024
    expected_digest = "sha256:" + hashlib.sha256(Path("examples/dsh/verifier.py").read_bytes()).hexdigest()
    assert resolved["verifier_code_digest"] == expected_digest
