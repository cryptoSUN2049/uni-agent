"""Run the official DeepSeek Harness inside a Uni-Agent task Sandbox.

The adapter deliberately keeps the Harness process in the same Sandbox as the
task verifier. Uni-Agent's Gateway remains the model endpoint, so its token
buffers and response masks are the source used by VERL; this module only adds
the semantic DSH trace and a safe correlation record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import Field, field_validator

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$", re.ASCII)
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._~-]+$", re.ASCII)
_DEFAULT_DSH_HOME_ROOT = "/tmp/uni-agent-dsh/home"
_DEFAULT_ARTIFACT_ROOT = "/tmp/uni-agent-dsh/artifacts"
_DEFAULT_WORKDIR = "/workspace"
_RUNNER_MODULE = "uni_agent.agents.dsh.runner"


class DshAgentConfig(AgentConfig):
    """Launch settings for the DSH SDK helper used by the online rollout."""

    name: str = "dsh"
    runner_python: str = Field(default="python", min_length=1, description="Python executable in the task image.")
    runner_module: str = Field(default=_RUNNER_MODULE, min_length=1, description="Installed DSH helper module.")
    dsh_home_root: str = Field(default=_DEFAULT_DSH_HOME_ROOT, min_length=1, description="Per-session DSH home root.")
    artifact_root: str = Field(
        default=_DEFAULT_ARTIFACT_ROOT, min_length=1, description="Semantic trace artifact root."
    )
    default_workdir: str = Field(
        default=_DEFAULT_WORKDIR, min_length=1, description="Fallback workspace when Task has no workdir."
    )
    profile: str = Field(default="sdk", min_length=1, description="Official DSH profile used by the SDK.")
    provider: str = Field(default="deepseek-official", min_length=1, description="DSH provider name.")
    reasoning_effort: str | None = Field(default=None, description="Optional DSH reasoning effort.")
    run_timeout: float = Field(default=1800.0, gt=0, description="Wall-clock cap for the DSH helper process.")
    keep_trace: bool = Field(default=True, description="Persist the canonical DSH event JSONL in the Sandbox.")
    runner_args: list[str] = Field(default_factory=list, description="Extra argv appended to the helper invocation.")

    @field_validator(
        "runner_python", "runner_module", "dsh_home_root", "artifact_root", "default_workdir", "profile", "provider"
    )
    @classmethod
    def _reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("DSH adapter paths and names must not be blank")
        return value

    @field_validator("dsh_home_root", "artifact_root")
    @classmethod
    def _validate_root_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("DSH adapter artifact roots must be absolute and traversal-free")
        return value

    @field_validator("default_workdir")
    @classmethod
    def _validate_workdir(cls, value: str) -> str:
        if value != "." and ".." in PurePosixPath(value).parts:
            raise ValueError("DSH adapter default_workdir must not contain traversal")
        return value


def extract_gateway_session_id(base_url: str) -> str:
    """Extract and validate the session ID from a Uni-Agent Gateway URL."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("dsh adapter requires an absolute Gateway URL without query or fragment")
    segments = [segment for segment in parsed.path.split("/") if segment]
    try:
        session_index = segments.index("sessions")
    except ValueError:
        session_index = -1
    if session_index < 0 or len(segments) != session_index + 3 or segments[-1] != "v1":
        raise ValueError("dsh adapter requires a session-scoped Gateway URL ending in /sessions/<id>/v1")
    session_id = segments[session_index + 1]
    if _SESSION_ID_PATTERN.fullmatch(session_id) is None:
        raise ValueError("Gateway session ID contains unsafe characters")
    return session_id


def _run_key(session_id: str) -> str:
    """Return a path-safe, non-secret identifier for one Gateway session."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]


def _text_content(value: object, *, label: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError(f"dsh adapter {label} content must be text or text blocks")
    parts: list[str] = []
    for index, block in enumerate(value):
        if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise ValueError(f"dsh adapter {label} block {index} must be a text block")
        parts.append(block["text"])
    return "".join(parts)


def prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """Convert the agent-neutral one-turn prompt to the DSH SDK text input."""
    if not messages:
        raise ValueError("dsh adapter requires at least one message")
    users = [message for message in messages if message.get("role") == "user"]
    if len(users) != 1:
        raise ValueError(f"dsh adapter requires exactly one user message, got {len(users)}")
    unsupported = [message.get("role") for message in messages if message.get("role") not in {"system", "user"}]
    if unsupported:
        raise ValueError(f"dsh adapter cannot replay prior {unsupported[0]!r} messages into a fresh DSH session")
    user_text = _text_content(users[0].get("content"), label="user")
    if not user_text.strip():
        raise ValueError("dsh adapter user message must not be blank")
    system_texts = [
        _text_content(message.get("content"), label="system") for message in messages if message.get("role") == "system"
    ]
    if not system_texts:
        return user_text
    return "[System instructions]\n" + "\n\n".join(system_texts) + "\n\n[User task]\n" + user_text


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _require_result(
    value: object,
    *,
    expected_trace_path: str,
    expected_dsh_session_id: str,
    require_trace: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("dsh adapter helper returned a non-object result")
    if value.get("schema") != "dsh.uni-agent.dsh-run.v1":
        raise RuntimeError("dsh adapter helper returned an unknown result schema")
    for key in ("dsh_session_id", "trace_sha256", "event_count", "final_response"):
        if key not in value:
            raise RuntimeError(f"dsh adapter helper result is missing {key}")
    if not isinstance(value["dsh_session_id"], str) or not value["dsh_session_id"]:
        raise RuntimeError("dsh adapter helper returned an invalid dsh_session_id")
    if value["dsh_session_id"] != expected_dsh_session_id:
        raise RuntimeError("dsh adapter helper dsh_session_id does not match the requested session")
    if not isinstance(value["trace_sha256"], str) or _HASH_PATTERN.fullmatch(value["trace_sha256"]) is None:
        raise RuntimeError("dsh adapter helper returned an invalid trace_sha256")
    if not isinstance(value["event_count"], int) or value["event_count"] < 0:
        raise RuntimeError("dsh adapter helper returned an invalid event_count")
    if not isinstance(value["final_response"], str):
        raise RuntimeError("dsh adapter helper returned an invalid final_response")
    if value.get("trace_path") != expected_trace_path:
        raise RuntimeError("dsh adapter helper trace_path does not match the requested artifact path")
    if require_trace and value.get("trace_persisted") is not True:
        raise RuntimeError("dsh adapter helper did not persist the canonical trace")
    finish_reason = value.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise RuntimeError("dsh adapter helper returned an invalid finish_reason")
    return value


@register_agent("dsh")
class DshAgent(Agent):
    """Black-box DSH runner whose model calls terminate at the Uni-Agent Gateway."""

    config_model = DshAgentConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: DshAgentConfig = self.config  # type: ignore[assignment]
        base_url = cfg.model.base_url
        if not base_url:
            raise ValueError("dsh adapter: config.model.base_url is not set")
        if cfg.model.model_name is None:
            raise ValueError("dsh adapter: config.model.model_name is required")
        if cfg.model.api_key.strip() == "":
            raise ValueError("dsh adapter: config.model.api_key must be non-empty")
        gateway_session_id = extract_gateway_session_id(base_url)
        prompt = prompt_from_messages(messages)
        key = _run_key(gateway_session_id)
        artifact_dir = f"{cfg.artifact_root.rstrip('/')}/{key}"
        input_path = f"{artifact_dir}/input.json"
        output_path = f"{artifact_dir}/result.json"
        trace_path = f"{artifact_dir}/session.jsonl"
        dsh_session_id = f"dsh-{gateway_session_id}"
        payload = {"prompt": prompt, "session_id": dsh_session_id}
        await sandbox.write_file(input_path, _json_bytes(payload))
        chmod_result = await sandbox.exec(["chmod", "600", input_path], timeout=10)
        if chmod_result.exit_code != 0:
            await self._cleanup_input(sandbox, input_path)
            raise RuntimeError("dsh adapter could not protect its temporary input file")

        effective_workdir = workdir or cfg.default_workdir
        env = {
            "DSH_UA_BASE_URL": base_url,
            "DSH_UA_SESSION_ID": gateway_session_id,
            "DSH_UA_API_KEY": cfg.model.api_key,
            "DSH_UA_MODEL": cfg.model.model_name or "",
            "DSH_UA_PROVIDER": cfg.provider,
            "DSH_UA_PROFILE": cfg.profile,
            "DSH_UA_HOME": f"{cfg.dsh_home_root.rstrip('/')}/{key}",
            "DSH_UA_CWD": effective_workdir,
            "DSH_UA_TRACE_PATH": trace_path,
            "DSH_UA_KEEP_TRACE": "1" if cfg.keep_trace else "0",
        }
        if cfg.reasoning_effort is not None:
            env["DSH_UA_REASONING_EFFORT"] = cfg.reasoning_effort
        if cfg.model.max_total_tokens is not None:
            env["DSH_UA_MAX_TOKENS"] = str(cfg.model.max_total_tokens)
        argv = [
            cfg.runner_python,
            "-m",
            cfg.runner_module,
            "--input",
            input_path,
            "--output",
            output_path,
            *cfg.runner_args,
        ]
        result = await sandbox.exec(argv, timeout=cfg.run_timeout, workdir=effective_workdir, env=env)
        if result.exit_code == -1:
            await self._cleanup_input(sandbox, input_path)
            raise TimeoutError(f"dsh adapter helper exceeded {cfg.run_timeout:g}s")
        if result.exit_code != 0:
            await self._cleanup_input(sandbox, input_path)
            detail = (result.stderr or result.stdout or "helper failed").strip()[-2000:]
            raise RuntimeError(f"dsh adapter helper exited {result.exit_code}: {detail}")
        try:
            raw_result = json.loads((await sandbox.read_file(output_path)).decode("utf-8"))
            helper_result = _require_result(
                raw_result,
                expected_trace_path=trace_path,
                expected_dsh_session_id=dsh_session_id,
                require_trace=cfg.keep_trace,
            )
        finally:
            await self._cleanup_input(sandbox, input_path)

        finish_reason = helper_result.get("finish_reason")
        finished = finish_reason == "completed"
        info = {
            "adapter": "uni-agent-dsh",
            "dsh_session_id": helper_result["dsh_session_id"],
            "gateway_session_id": gateway_session_id,
            "trace_sha256": helper_result["trace_sha256"],
            "trace_path": helper_result["trace_path"],
            "event_count": helper_result["event_count"],
            "finish_reason": finish_reason,
            "keep_trace": cfg.keep_trace,
        }
        logger.info(
            "dsh adapter complete: gateway_session=%s events=%s finish_reason=%s",
            gateway_session_id,
            helper_result["event_count"],
            finish_reason,
        )
        return AgentResult(
            output={"response": helper_result["final_response"]},
            transcript=list(messages),
            info=info,
            finished=finished,
        )

    @staticmethod
    async def _cleanup_input(sandbox: Sandbox, input_path: str) -> None:
        """Best-effort removal of the prompt file after the helper exits."""
        try:
            await sandbox.exec(["rm", "-f", input_path], timeout=10)
        except Exception:
            logger.warning("dsh adapter could not remove its temporary input file", exc_info=True)
