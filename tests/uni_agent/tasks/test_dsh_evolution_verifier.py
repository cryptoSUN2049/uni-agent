from __future__ import annotations

import hashlib
import json
from pathlib import Path

from examples.dsh.evolution_verifier import verify
from examples.dsh.prepare_evolution_dataset import _host_code, build_evolution_rows


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _event(event_type: str, data: dict) -> dict:
    return {"type": event_type, "data": data}


def _call(call_id: str, name: str, arguments: dict, *, seq: int) -> dict:
    return _event(
        "tool/call",
        {"turn": seq, "step": 1, "callId": call_id, "name": name, "arguments": json.dumps(arguments)},
    )


def _result(call_id: str, text: str, *, error: bool = False) -> dict:
    return _event(
        "tool/result",
        {
            "message": {
                "source": {"kind": "tool", "callId": call_id},
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": call_id,
                        "content": [{"type": "text", "text": text}],
                        "isError": error,
                    }
                ],
            }
        },
    )


def _episode(tmp_path: Path, *, candidate_output: str, shell: bool = False) -> tuple[dict, bytes, dict[str, str]]:
    fixture_bytes = b'{"schema":"dsh.evolution.fixture.v1","operation":"normalize_whitespace","input":"  a   b  "}\n'
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_bytes(fixture_bytes)
    expected_digest = _digest(b"a b")
    code = (
        "return {name:'candidate', inject:['tools'], apply(ctx){"
        "harness.registerTool(ctx, harness.defineTool({"
        "name:'normalize_payload',parameters:{text:{type:'string',required:true}},"
        "output:{schema:{type:'string'},render(_a,v){return [{type:'text',text:v}]}},"
        "async execute(args){return args.text}}))}}"
    )
    events = [
        _call("c1", "str_replace_editor", {"command": "view", "path": str(fixture_path)}, seq=1),
        _result("c1", fixture_bytes.decode()),
        _call("c2", "cordis_inspect_list", {}, seq=2),
        _result("c2", "inspect"),
        _call(
            "c3",
            "cordis_define",
            {
                "plugin": {"kind": "new", "idPrefix": "evo"},
                "name": "normalize",
                "purpose": "normalize text",
                "code": {"host": code},
            },
            seq=3,
        ),
        _result("c3", "Defined evo-1/pkg-1 (normalize); it is not running yet."),
        _call("c4", "cordis_run", {"pluginId": "evo-1", "packageId": "pkg-1", "mode": "run"}, seq=4),
        _result("c4", "evo-1/pkg-1 is running (run-1)."),
        _call("c5", "normalize_payload", {"text": "  a   b  "}, seq=5),
        _result("c5", candidate_output),
        _call("c6", "cordis_stop", {"pluginId": "evo-1"}, seq=6),
        _result("c6", "Dynamic Plugin evo-1 is stopped"),
        _call("c7", "cordis_undefine", {"pluginId": "evo-1"}, seq=7),
        _result("c7", "Removed dynamic Plugin evo-1"),
        _event("turn/end", {"turn": 7, "reason": {"kind": "completed"}}),
    ]
    if shell:
        events.insert(1, _call("bad", "bash", {"command": "echo unsafe"}, seq=1))
    trace_bytes = b"".join(
        (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode() for event in events
    )
    trace_path = tmp_path / "session.jsonl"
    trace_path.write_bytes(trace_bytes)
    metadata = {
        "task_id": "dsh/harness-evolution/normalize-basic-01",
        "task_version": "1",
        "split": "train",
        "scenario_id": "normalize-basic-01",
        "fixture_path": fixture_path.name,
        "fixture_digest": _digest(fixture_bytes),
        "operation": "normalize_whitespace",
        "candidate_tool_name": "normalize_payload",
        "variant": "basic",
        "environment_digest": _digest(b"environment"),
        "verifier_id": "dsh-harness-evolution-verifier",
        "verifier_version": "1",
        "verifier_code_digest": _digest(b"verifier"),
        "profile": "sdk-minimal",
        "patches_sha256": _digest(b"patches"),
    }
    dsh = {
        "dsh_session_id": "dsh-session-1",
        "trace_sha256": _digest(trace_bytes),
        "profile": "sdk-minimal",
        "patches_sha256": _digest(b"patches"),
    }
    envelope = {
        "schema": "dsh.uni-agent.task-result.v1",
        "metadata": metadata,
        "response": json.dumps(
            {
                "status": "promote",
                "plugin_id": "evo-1",
                "package_id": "pkg-1",
                "result_digest": expected_digest,
                "evidence": ["inspected", "ran", "cleaned"],
            },
            separators=(",", ":"),
        ),
        "finished": True,
        "dsh": dsh,
    }
    envelope_bytes = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    envelope_path = tmp_path / "agent-result.json"
    envelope_path.write_bytes(envelope_bytes)
    env = {
        "DSH_TASK_RESULT_PATH": str(envelope_path),
        "DSH_DSH_SESSION_ID": "dsh-session-1",
        "DSH_ARTIFACT_SHA256": _digest(envelope_bytes),
        "DSH_TRACE_SHA256": _digest(trace_bytes),
        "DSH_TRACE_PATH": str(trace_path),
        "DSH_TASK_ID": metadata["task_id"],
        "DSH_TASK_VERSION": "1",
        "DSH_ENVIRONMENT_DIGEST": metadata["environment_digest"],
        "DSH_VERIFIER_ID": metadata["verifier_id"],
        "DSH_VERIFIER_VERSION": "1",
        "DSH_VERIFIER_CODE_DIGEST": metadata["verifier_code_digest"],
        "DSH_TASK_WORKDIR": str(tmp_path),
    }
    return envelope, envelope_bytes, env


def _set_env(monkeypatch, values: dict[str, str]) -> None:
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_evolution_verifier_scores_independent_behavior_and_lifecycle(monkeypatch, tmp_path: Path) -> None:
    _envelope, _bytes, env = _episode(tmp_path, candidate_output="a b")
    _set_env(monkeypatch, env)
    result = verify()
    assert result["reward"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["extra_info"]["eligible"] is True
    assert result["extra_info"]["components"]["behavior"] == 1.0


def test_evolution_verifier_caps_wrong_behavior_but_keeps_fresh_zero_or_partial_reward(
    monkeypatch, tmp_path: Path
) -> None:
    _envelope, _bytes, env = _episode(tmp_path, candidate_output="wrong")
    _set_env(monkeypatch, env)
    result = verify()
    assert 0.0 < result["reward"] <= 0.40
    assert result["accuracy"] == 0.0
    assert result["fresh"] is True


def test_evolution_verifier_hard_vetoes_shell_access(monkeypatch, tmp_path: Path) -> None:
    _envelope, _bytes, env = _episode(tmp_path, candidate_output="a b", shell=True)
    _set_env(monkeypatch, env)
    result = verify()
    assert result["reward"] == 0.0
    assert result["extra_info"]["eligible"] is False
    assert "shell_access" in result["extra_info"]["hard_veto"]


def test_evolution_verifier_derives_digest_when_report_omits_it(monkeypatch, tmp_path: Path) -> None:
    envelope, _bytes, env = _episode(tmp_path, candidate_output="a b")
    report = json.loads(envelope["response"])
    report.pop("result_digest")
    envelope["response"] = json.dumps(report, separators=(",", ":"))
    result_path = Path(env["DSH_TASK_RESULT_PATH"])
    envelope_bytes = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    result_path.write_bytes(envelope_bytes)
    env["DSH_ARTIFACT_SHA256"] = _digest(envelope_bytes)
    _set_env(monkeypatch, env)
    result = verify()
    assert result["accuracy"] == 1.0
    assert result["extra_info"]["expected_result_digest"].startswith("sha256:")


def test_evolution_verifier_supports_trim_bootstrap_operation() -> None:
    code = _host_code(candidate_tool_name="trim_payload", operation="trim")
    assert "execute:a=>a.text.trim()" in code
    assert code.endswith("}))}}")


def test_evolution_dataset_builder_binds_fixture_and_patch_identity(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenarios.jsonl"
    fixture_root = tmp_path / "root"
    fixture = fixture_root / "examples/dsh/fixtures/evolution/one.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        '{"schema":"dsh.evolution.fixture.v1","operation":"mask_digits","input":"v2"}\n',
        encoding="utf-8",
    )
    scenario_path.write_text(
        json.dumps(
            {
                "scenario_id": "one",
                "task_id": "dsh/harness-evolution/one",
                "task_version": "1",
                "split": "train",
                "fixture_path": "examples/dsh/fixtures/evolution/one.json",
                "candidate_tool_name": "mask_payload",
                "variant": "basic",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = build_evolution_rows(
        scenario_path,
        fixture_root=fixture_root,
        environment_digest=_digest(b"environment"),
        verifier_id="dsh-harness-evolution-verifier",
        verifier_version="1",
        verifier_code_digest=_digest(b"verifier"),
        patches=["examples/dsh/evolution.patch.yml"],
    )
    task = rows["train"][0]["extra_info"]["tools_kwargs"]["task"]
    prompt = rows["train"][0]["prompt"][0]["content"]
    assert "Call exactly one tool in each assistant turn" in prompt
    assert "never batch or parallelize tool calls" in prompt
    assert task["metadata"]["fixture_digest"] == _digest(fixture.read_bytes())
    assert task["metadata"]["candidate_scope"] == "session-local-host-only"
    assert "sha256:" in task["metadata"]["patches_sha256"]
    assert "v2" not in rows["train"][0]["prompt"][0]["content"]


def test_evolution_task_config_selects_dynamic_profile_and_verifier() -> None:
    from uni_agent.tasks import TaskConfigResolver

    resolved = TaskConfigResolver.from_file("examples/dsh/evolution_task_config.yaml").resolve(
        {
            "name": "dsh_architecture",
            "metadata": {
                "task_id": "dsh/harness-evolution/normalize-basic-01",
                "task_version": "1",
                "split": "train",
                "environment_digest": _digest(b"environment"),
                "verifier_id": "dsh-harness-evolution-verifier",
                "verifier_version": "1",
                "verifier_code_digest": "sha256:6715dbe97e671ca9151a68ec7676bf00a35072f591b1dfacd3814db1b773e991",
            },
        }
    )
    assert resolved["agent"]["profile"] == "sdk-minimal"
    assert resolved["agent"]["patches"] == ["examples/dsh/evolution.patch.yml"]
    assert resolved["agent"]["reasoning_effort"] == "off"
    assert resolved["verifier_command"][-1] == "examples.dsh.evolution_verifier"
