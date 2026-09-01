from __future__ import annotations

import asyncio
import json

import pytest

from uni_agent.agents import get_agent_cls
from uni_agent.agents.base import ModelConfig
from uni_agent.agents.dsh.agent import (
    DshAgent,
    DshAgentConfig,
    _run_key,
    extract_gateway_session_id,
    prompt_from_messages,
)
from uni_agent.sandbox.base import ExecResult


class _FakeSandbox:
    def __init__(self, *, exit_code: int = 0, output: dict | None = None, stderr: str = "") -> None:
        self.exit_code = exit_code
        self.output = output
        self.stderr = stderr
        self.writes: dict[str, bytes] = {}
        self.exec_calls: list[dict] = []
        self.reads: list[str] = []

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.writes[path] = content.encode() if isinstance(content, str) else content

    async def exec(self, argv, *, timeout=None, workdir=None, env=None) -> ExecResult:
        self.exec_calls.append({"argv": argv, "timeout": timeout, "workdir": workdir, "env": env})
        if argv[:1] == ["chmod"]:
            return ExecResult(exit_code=0, stdout="", stderr="")
        if argv[:2] != ["rm", "-f"] and "--output" in argv and self.output is not None:
            output_path = argv[argv.index("--output") + 1]
            self.writes[output_path] = (json.dumps(self.output) + "\n").encode()
        return ExecResult(exit_code=self.exit_code, stdout="", stderr=self.stderr)

    async def read_file(self, path: str) -> bytes:
        self.reads.append(path)
        return self.writes[path]


def _config(*, model: ModelConfig | None = None, **kwargs) -> DshAgentConfig:
    return DshAgentConfig(
        model=model
        or ModelConfig(
            base_url="http://gateway.example/sessions/abc123/v1",
            api_key="gateway-key",
            model_name="Qwen3-27B",
        ),
        **kwargs,
    )


def _helper_result() -> dict:
    return {
        "schema": "dsh.uni-agent.dsh-run.v1",
        "dsh_session_id": "dsh-abc123",
        "trace_sha256": "sha256:" + "a" * 64,
        "trace_path": f"/tmp/uni-agent-dsh/artifacts/{_run_key('abc123')}/session.jsonl",
        "event_count": 12,
        "finish_reason": "completed",
        "final_response": "done",
        "trace_persisted": True,
    }


def test_gateway_url_requires_session_scope_and_allows_prefix() -> None:
    assert extract_gateway_session_id("https://host/api/sessions/run-1/v1/") == "run-1"
    with pytest.raises(ValueError, match="session-scoped"):
        extract_gateway_session_id("https://host/v1")
    with pytest.raises(ValueError, match="query"):
        extract_gateway_session_id("https://host/sessions/run/v1?token=secret")


def test_config_rejects_traversal_in_artifact_roots() -> None:
    with pytest.raises(ValueError, match="absolute"):
        _config(artifact_root="relative")
    with pytest.raises(ValueError, match="traversal"):
        _config(dsh_home_root="/tmp/../unsafe")
    with pytest.raises(ValueError, match="traversal"):
        _config(patches=["../unsafe.patch.yml"])


def test_prompt_conversion_preserves_user_and_marks_system() -> None:
    assert prompt_from_messages([{"role": "user", "content": "learn DSH"}]) == "learn DSH"
    assert (
        prompt_from_messages(
            [
                {"role": "system", "content": "be precise"},
                {"role": "user", "content": [{"type": "text", "text": "learn DSH"}]},
            ]
        )
        == "[System instructions]\nbe precise\n\n[User task]\nlearn DSH"
    )
    with pytest.raises(ValueError, match="exactly one user"):
        prompt_from_messages([{"role": "assistant", "content": "old"}])
    with pytest.raises(ValueError, match="cannot replay"):
        prompt_from_messages([{"role": "user", "content": "x"}, {"role": "tool", "content": "y"}])


def test_registry_loads_dsh_lazily() -> None:
    assert get_agent_cls("dsh") is DshAgent


def test_run_launches_helper_inside_sandbox_and_returns_trace_correlation() -> None:
    output = _helper_result()
    sandbox = _FakeSandbox(output=output)
    result = asyncio.run(
        DshAgent(_config()).run(
            sandbox=sandbox,
            messages=[{"role": "user", "content": "explain SessionEvent"}],
            workdir="/testbed",
        )
    )
    assert result.finished is True
    assert result.output == {"response": "done"}
    assert result.info["dsh_session_id"] == "dsh-abc123"
    assert result.info["trace_sha256"] == "sha256:" + "a" * 64
    assert result.info["trace_path"].endswith("/session.jsonl")
    call = next(call for call in sandbox.exec_calls if call["argv"][0:2] == ["python", "-m"])
    assert call["argv"][:4] == ["python", "-m", "uni_agent.agents.dsh.runner", "--input"]
    assert call["workdir"] == "/testbed"
    assert call["timeout"] == 1800.0
    assert call["env"]["DSH_UA_BASE_URL"] == "http://gateway.example/sessions/abc123/v1"
    assert json.loads(call["env"]["DSH_UA_PATCHES"]) == []
    assert "api_key" not in result.info
    assert "base_url" not in result.info
    assert any(call["argv"][:2] == ["rm", "-f"] for call in sandbox.exec_calls)


def test_run_passes_ordered_profile_patches_to_helper() -> None:
    output = _helper_result()
    output["profile"] = "sdk-minimal"
    output["patches_sha256"] = "sha256:" + "d" * 64
    sandbox = _FakeSandbox(output=output)
    result = asyncio.run(
        DshAgent(
            _config(
                profile="sdk-minimal",
                patches=["examples/dsh/evolution.patch.yml", "/opt/dsh/last.patch.yml"],
            )
        ).run(sandbox=sandbox, messages=[{"role": "user", "content": "x"}])
    )
    call = next(call for call in sandbox.exec_calls if call["argv"][0:2] == ["python", "-m"])
    assert json.loads(call["env"]["DSH_UA_PATCHES"]) == [
        "examples/dsh/evolution.patch.yml",
        "/opt/dsh/last.patch.yml",
    ]
    assert result.info["profile"] == "sdk-minimal"
    assert result.info["patches_sha256"] == "sha256:" + "d" * 64


def test_run_uses_per_turn_token_cap_before_episode_budget() -> None:
    sandbox = _FakeSandbox(output=_helper_result())
    model = ModelConfig(
        base_url="http://gateway.example/sessions/abc123/v1",
        api_key="gateway-key",
        model_name="Qwen3-27B",
        max_total_tokens=4096,
        max_tokens_per_turn=1024,
    )

    asyncio.run(DshAgent(_config(model=model)).run(sandbox=sandbox, messages=[{"role": "user", "content": "x"}]))

    call = next(call for call in sandbox.exec_calls if call["argv"][0:2] == ["python", "-m"])
    assert call["env"]["DSH_UA_MAX_TOKENS"] == "1024"


def test_run_falls_back_to_episode_budget_when_no_per_turn_cap() -> None:
    sandbox = _FakeSandbox(output=_helper_result())
    model = ModelConfig(
        base_url="http://gateway.example/sessions/abc123/v1",
        api_key="gateway-key",
        model_name="Qwen3-27B",
        max_total_tokens=4096,
    )

    asyncio.run(DshAgent(_config(model=model)).run(sandbox=sandbox, messages=[{"role": "user", "content": "x"}]))

    call = next(call for call in sandbox.exec_calls if call["argv"][0:2] == ["python", "-m"])
    assert call["env"]["DSH_UA_MAX_TOKENS"] == "4096"


def test_run_rejects_bad_helper_result_and_cleans_input() -> None:
    bad = _helper_result()
    bad["trace_sha256"] = "not-a-hash"
    sandbox = _FakeSandbox(output=bad)
    with pytest.raises(RuntimeError, match="invalid trace_sha256"):
        asyncio.run(DshAgent(_config()).run(sandbox=sandbox, messages=[{"role": "user", "content": "x"}]))
    assert any(call["argv"][:2] == ["rm", "-f"] for call in sandbox.exec_calls)


def test_run_rejects_helper_session_lineage_mismatch() -> None:
    bad = _helper_result()
    bad["dsh_session_id"] = "dsh-other-session"
    sandbox = _FakeSandbox(output=bad)
    with pytest.raises(RuntimeError, match="does not match"):
        asyncio.run(DshAgent(_config()).run(sandbox=sandbox, messages=[{"role": "user", "content": "x"}]))


def test_run_allows_explicit_trace_opt_out() -> None:
    output = _helper_result()
    output["trace_persisted"] = False
    sandbox = _FakeSandbox(output=output)
    result = asyncio.run(
        DshAgent(_config(keep_trace=False)).run(sandbox=sandbox, messages=[{"role": "user", "content": "x"}])
    )
    assert result.info["keep_trace"] is False


def test_run_surfaces_timeout_and_helper_failure_without_endpoint_in_error() -> None:
    timed_out = _FakeSandbox(exit_code=-1)
    with pytest.raises(TimeoutError, match="exceeded"):
        asyncio.run(DshAgent(_config()).run(sandbox=timed_out, messages=[{"role": "user", "content": "x"}]))
    failed = _FakeSandbox(exit_code=7, stderr="dependency missing")
    with pytest.raises(RuntimeError, match="dependency missing") as error:
        asyncio.run(DshAgent(_config()).run(sandbox=failed, messages=[{"role": "user", "content": "x"}]))
    assert "gateway.example" not in str(error.value)
