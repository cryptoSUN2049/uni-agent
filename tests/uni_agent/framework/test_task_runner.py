import pytest

from uni_agent.framework import task_runner
from uni_agent.framework.task_runner import (
    _extract_upstream,
    _inject_dsh_artifact_roots,
    _inject_gateway_tunnel,
    _reward_info_from_result,
    _rewrite_gateway_url,
)
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskConfig, TaskResult


def test_rewrite_gateway_url_replaces_host_with_tunnel_port():
    assert _rewrite_gateway_url("http://gateway.example:40169/sessions/abc/v1", 38197) == (
        "http://127.0.0.1:38197/sessions/abc/v1"
    )


def test_rewrite_gateway_url_custom_proxy_port():
    assert _rewrite_gateway_url("http://gateway:8000/v1", 4242) == "http://127.0.0.1:4242/v1"


def test_extract_upstream_returns_host_port():
    assert _extract_upstream("http://gateway.example:40169/sessions/abc/v1") == "gateway.example:40169"


def test_extract_upstream_none_without_port():
    assert _extract_upstream("http://gateway/v1") is None


def test_inject_gateway_tunnel_rewrites_upstream_and_base_url():
    task = {
        "sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197, "image": "x"}},
        "agent": {"step_limit": 10},
    }
    merged = _inject_gateway_tunnel(task, "http://gateway.example:40169/sessions/abc/v1")

    assert merged["sandbox"]["sandbox_kwargs"]["upstream"] == "gateway.example:40169"
    assert merged["sandbox"]["sandbox_kwargs"]["proxy_port"] == 38197
    # The agent receives the tunnel-rewritten base_url; unrelated keys are preserved.
    assert merged["agent"]["model"]["base_url"] == "http://127.0.0.1:38197/sessions/abc/v1"
    assert merged["agent"]["step_limit"] == 10


def test_inject_gateway_tunnel_raises_without_port():
    task = {"sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="cannot derive gateway tunnel upstream"):
        _inject_gateway_tunnel(task, "http://gateway.example/v1")


def test_inject_gateway_tunnel_rejects_non_yuanrong_sandbox():
    task = {"sandbox": {"provider": "local", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="supported only on 'openyuanrong'"):
        _inject_gateway_tunnel(task, "http://gateway.example:40169/v1")


def test_inject_dsh_artifact_roots_preserves_unrelated_task_config():
    task = {
        "name": "dsh_architecture",
        "agent": {"name": "dsh", "step_limit": 10},
        "result_root": "/tmp/old-results",
    }

    merged = _inject_dsh_artifact_roots(
        task,
        trace_root="/workspace/run/artifacts/traces",
        result_root="/workspace/run/artifacts/results",
    )

    assert merged["agent"] == {
        "name": "dsh",
        "step_limit": 10,
        "artifact_root": "/workspace/run/artifacts/traces",
    }
    assert merged["result_root"] == "/workspace/run/artifacts/results"


@pytest.mark.parametrize(
    ("trace_root", "result_root", "message"),
    [
        ("/workspace/traces", None, "configured together"),
        ("relative/traces", "/workspace/results", "absolute traversal-free"),
        ("/workspace/traces", "/workspace/../results", "absolute traversal-free"),
    ],
)
def test_inject_dsh_artifact_roots_rejects_incomplete_or_unsafe_paths(trace_root, result_root, message):
    with pytest.raises(ValueError, match=message):
        _inject_dsh_artifact_roots(
            {"name": "dsh_architecture", "agent": {"name": "dsh"}},
            trace_root=trace_root,
            result_root=result_root,
        )


def test_task_result_positional_field_order():
    result = TaskResult(0.5, 1.0, False, {"reason": "limit"})

    assert result.reward == 0.5
    assert result.accuracy == 1.0
    assert result.finished is False
    assert result.extra_info == {"reason": "limit"}


def test_reward_info_omits_unknown_agent_completion():
    result = TaskResult(reward=0.5, accuracy=1.0)

    assert _reward_info_from_result(result) == {
        "reward": 0.5,
        "acc": 1.0,
    }


@pytest.mark.parametrize("finished", [True, False])
def test_reward_info_forwards_agent_completion(finished):
    result = TaskResult(reward=0.0, finished=finished)

    assert _reward_info_from_result(result) == {
        "reward": 0.0,
        "finished": finished,
    }


def test_reward_info_rejects_non_boolean_agent_completion():
    result = TaskResult(reward=0.0, finished=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finished must be a bool or None"):
        _reward_info_from_result(result)


@pytest.mark.asyncio
async def test_run_task_binds_raw_prompt_to_sample_task_config(monkeypatch, tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(
        """
- name: test_task
""".strip()
    )
    captured = {}

    class _FakeTask:
        def __init__(self, config):
            self.config = TaskConfig(
                name=config["name"],
                sandbox={"provider": "local"},
                prompt=config["prompt"],
                metadata=config["metadata"],
            )

        async def run(self):
            captured["config"] = self.config
            return TaskResult(reward=1.0, accuracy=1.0, finished=True)

    monkeypatch.setattr(task_runner, "get_task", _FakeTask)
    source_prompt = [{"role": "user", "content": "Canonical source problem"}]

    await task_runner.run_task(
        session=SessionHandle(
            session_id="test-session",
            base_url="http://gateway/sessions/test/v1",
            reward_info_url=None,
        ),
        raw_prompt=source_prompt,
        tools_kwargs={
            "task": {
                "name": "test_task",
                "metadata": {"problem_statement": "METADATA PROBLEM"},
            }
        },
        task_config_path=str(config_path),
    )

    assert captured["config"].prompt == source_prompt


def _patch_fake_task(monkeypatch, tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text("- name: test_task")
    captured = {}

    class _FakeTask:
        def __init__(self, config):
            self.config = config

        async def run(self):
            captured["ran"] = True
            return TaskResult(reward=1.0, accuracy=1.0, finished=True)

    monkeypatch.setattr(task_runner, "get_task", _FakeTask)
    return config_path, captured


def _runner_kwargs():
    return {
        "raw_prompt": [{"role": "user", "content": "hello"}],
        "tools_kwargs": {"task": {"name": "test_task", "metadata": {}}},
    }


@pytest.mark.asyncio
async def test_run_task_requires_report_reward_when_ack_is_required():
    with pytest.raises(ValueError, match="requires report_reward=True"):
        await task_runner.run_task(
            session=SessionHandle(session_id="test-session"),
            report_reward=False,
            require_reward_post=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("report_reward", "require_reward_post", "message"),
    [
        ("true", False, "report_reward must be a bool"),
        (False, "true", "require_reward_post must be a bool"),
    ],
)
async def test_run_task_rejects_non_boolean_reward_flags(report_reward, require_reward_post, message):
    with pytest.raises(ValueError, match=message):
        await task_runner.run_task(
            session=SessionHandle(session_id="test-session"),
            report_reward=report_reward,
            require_reward_post=require_reward_post,
        )


@pytest.mark.asyncio
async def test_run_task_ack_required_rejects_missing_reward_endpoint(monkeypatch, tmp_path):
    config_path, captured = _patch_fake_task(monkeypatch, tmp_path)
    kwargs = _runner_kwargs()
    kwargs["task_config_path"] = str(config_path)

    with pytest.raises(RuntimeError, match="reward-info endpoint"):
        await task_runner.run_task(
            session=SessionHandle(session_id="test-session", base_url="http://gateway/v1"),
            report_reward=True,
            require_reward_post=True,
            **kwargs,
        )
    assert captured["ran"] is True


@pytest.mark.asyncio
async def test_run_task_ack_required_rejects_failed_reward_post(monkeypatch, tmp_path):
    config_path, _ = _patch_fake_task(monkeypatch, tmp_path)
    kwargs = _runner_kwargs()
    kwargs["task_config_path"] = str(config_path)

    async def _failed_post(_url, _result):
        return False

    monkeypatch.setattr(task_runner, "_post_reward_info", _failed_post)
    with pytest.raises(RuntimeError, match="reward acknowledgement"):
        await task_runner.run_task(
            session=SessionHandle(
                session_id="test-session",
                base_url="http://gateway/v1",
                reward_info_url="http://gateway/reward",
            ),
            report_reward=True,
            require_reward_post=True,
            **kwargs,
        )


@pytest.mark.asyncio
async def test_run_task_ack_required_accepts_successful_reward_post(monkeypatch, tmp_path):
    config_path, _ = _patch_fake_task(monkeypatch, tmp_path)
    kwargs = _runner_kwargs()
    kwargs["task_config_path"] = str(config_path)
    seen = {}

    async def _successful_post(url, result):
        seen["url"] = url
        seen["reward"] = result.reward
        return True

    monkeypatch.setattr(task_runner, "_post_reward_info", _successful_post)
    result = await task_runner.run_task(
        session=SessionHandle(
            session_id="test-session",
            base_url="http://gateway/v1",
            reward_info_url="http://gateway/reward",
        ),
        report_reward=True,
        require_reward_post=True,
        **kwargs,
    )

    assert result.reward == 1.0
    assert seen == {"url": "http://gateway/reward", "reward": 1.0}
