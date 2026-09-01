from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.dsh.prepare_dataset import (
    adapt_verl_online_rl_seed,
    build_rows,
    build_rows_from_verl_seed_file,
    load_dual_backend_manifest,
    load_released_manifest,
)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _seed(task_id: str = "dsh/architecture/intro", split: str = "train") -> dict:
    return {
        "schema": "dsh.uni-agent.task-seed.v1",
        "task_id": task_id,
        "task_version": "1",
        "split": split,
        "prompt": [{"role": "user", "content": "Explain DSH plugins."}],
        "metadata": {"rubric_id": "plugin-basics"},
    }


def _manifest(task_id: str = "dsh/architecture/intro", split: str = "train") -> dict:
    return {
        "schema": "dsh.training-release.v1",
        "status": "released",
        "tasks": [{"task_id": task_id, "task_version": "1", "split": split}],
        "files": [],
    }


def test_build_rows_keeps_runtime_controls_out_of_dataset_rows(tmp_path: Path) -> None:
    seed_path = tmp_path / "task-seeds.jsonl"
    seed_path.write_text(json.dumps(_seed()) + "\n", encoding="utf-8")
    rows = build_rows(_manifest(), seed_path)

    row = rows["train"][0]
    assert row["prompt"][0]["content"] == "Explain DSH plugins."
    task = row["extra_info"]["tools_kwargs"]["task"]
    assert task["name"] == "dsh_architecture"
    assert task["metadata"]["task_id"] == "dsh/architecture/intro"
    assert "verifier_command" not in task


def test_manifest_file_hash_is_checked_before_seed_conversion(tmp_path: Path) -> None:
    seed_bytes = (json.dumps(_seed()) + "\n").encode("utf-8")
    seed_path = tmp_path / "task-seeds.jsonl"
    seed_path.write_bytes(seed_bytes)
    manifest = _manifest()
    manifest["files"] = [{"path": "task-seeds.jsonl", "sha256": _sha256(seed_bytes), "records": 1}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded, files = load_released_manifest(tmp_path)
    assert loaded["status"] == "released"
    assert files["task-seeds.jsonl"]["records"] == 1

    seed_path.write_text(seed_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_released_manifest(tmp_path)


def test_build_rows_rejects_seed_not_listed_in_release(tmp_path: Path) -> None:
    seed_path = tmp_path / "task-seeds.jsonl"
    seed_path.write_text(json.dumps(_seed("dsh/architecture/other")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not listed"):
        build_rows(_manifest(), seed_path)


def _verl_seed(split: str = "train") -> dict:
    return {
        "data_source": "dsh/repository",
        "prompt": [{"role": "user", "content": "Repair the failing test."}],
        "ability": "dsh-agent",
        "reward_model": {"style": "dsh_verifier", "verifier_ref": "dsh.repo-tests@1"},
        "extra_info": {
            "task_id": "repo/repair-test-v1",
            "task_version": "1.0.0",
            "split": split,
            "environment_ref": _sha256(b"environment"),
            "verifier_id": "dsh.repo-tests",
            "verifier_version": "1.0.0",
            "seed_id": "dsh-seed:repo/repair-test-v1@1.0.0",
            "budget": {"max_turns": 8, "max_tokens": 1000, "max_tool_calls": 4, "max_wall_time_ms": 10000},
        },
        "agent_name": "dsh",
    }


def test_adapter_wraps_m3_verl_seed_for_uni_agent_runner() -> None:
    row = adapt_verl_online_rl_seed(_verl_seed())

    task = row["extra_info"]["tools_kwargs"]["task"]
    assert task == {
        "name": "dsh_architecture",
        "metadata": {
            "task_id": "repo/repair-test-v1",
            "task_version": "1.0.0",
            "split": "train",
            "environment_digest": _sha256(b"environment"),
            "environment_ref": _sha256(b"environment"),
            "verifier_id": "dsh.repo-tests",
            "verifier_version": "1.0.0",
            "seed_id": "dsh-seed:repo/repair-test-v1@1.0.0",
            "budget": {"max_turns": 8, "max_tokens": 1000, "max_tool_calls": 4, "max_wall_time_ms": 10000},
        },
    }
    assert row["extra_info"]["task_id"] == "repo/repair-test-v1"
    assert row["extra_info"]["tools_kwargs"]["task"]["name"] == "dsh_architecture"


def test_adapter_maps_m3_held_out_split_and_rejects_non_dsh_agent() -> None:
    row = adapt_verl_online_rl_seed(_verl_seed("held_out"))
    assert row["extra_info"]["tools_kwargs"]["task"]["metadata"]["split"] == "holdout"
    with pytest.raises(ValueError, match="agent_name"):
        adapt_verl_online_rl_seed({**_verl_seed(), "agent_name": "react"})


def test_adapter_can_bind_operator_verifier_code_digest() -> None:
    digest = _sha256(b"verifier-code")
    row = adapt_verl_online_rl_seed(_verl_seed(), verifier_code_digest=digest)
    assert row["extra_info"]["tools_kwargs"]["task"]["metadata"]["verifier_code_digest"] == digest
    with pytest.raises(ValueError, match="conflicts"):
        adapt_verl_online_rl_seed(
            {
                **_verl_seed(),
                "extra_info": {**_verl_seed()["extra_info"], "verifier_code_digest": _sha256(b"other")},
            },
            verifier_code_digest=digest,
        )


def test_adapter_drops_untrusted_nested_runner_controls() -> None:
    source = _verl_seed()
    source["extra_info"]["tools_kwargs"] = {"task": {"verifier_command": ["evil"]}, "other": "drop"}

    row = adapt_verl_online_rl_seed(source)

    tools_kwargs = row["extra_info"]["tools_kwargs"]
    assert list(tools_kwargs) == ["task"]
    assert tools_kwargs["task"]["name"] == "dsh_architecture"
    assert "verifier_command" not in tools_kwargs["task"]["metadata"]


def test_dual_backend_manifest_selects_and_hash_checks_verl_seed_file(tmp_path: Path) -> None:
    seed_bytes = (json.dumps(_verl_seed()) + "\n").encode("utf-8")
    seed_path = tmp_path / "verl-online-rl-seeds.jsonl"
    seed_path.write_bytes(seed_bytes)
    manifest = {
        "schema": "dsh.dual-backend-training-release.v1",
        "eligibility": {"status": "eligible", "reasons": []},
        "backends": {"verl": {"onlineRlSeed": {"path": seed_path.name, "sha256": _sha256(seed_bytes)}}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded, selected = load_dual_backend_manifest(tmp_path)
    assert loaded["schema"] == "dsh.dual-backend-training-release.v1"
    assert selected == seed_path
    rows = build_rows_from_verl_seed_file(selected)
    assert rows["train"][0]["extra_info"]["tools_kwargs"]["task"]["name"] == "dsh_architecture"

    seed_path.write_bytes(seed_bytes + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dual_backend_manifest(tmp_path)


def test_dual_backend_manifest_rejects_analysis_only_release(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "dsh.dual-backend-training-release.v1",
                "eligibility": {"status": "analysis_only", "reasons": ["GPU_RUNTIME_PENDING"]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not eligible"):
        load_dual_backend_manifest(tmp_path)
