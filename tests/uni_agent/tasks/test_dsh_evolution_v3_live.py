from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.dsh.evolution_v3_catalog import FAMILY_IDS
from examples.dsh.evolution_v3_live import (
    build_live_task_rows,
    load_live_scenarios,
    write_live_contract_bundle,
)
from uni_agent.tasks import TaskConfigResolver

SCENARIO_PATH = Path("examples/dsh/evolution_v3_live_scenarios.jsonl")
ENVIRONMENT_DIGEST = "sha256:" + "e" * 64
VERIFIER_DIGEST = "sha256:" + "f" * 64


def _rows() -> list[dict]:
    scenarios = load_live_scenarios(SCENARIO_PATH, repository_root=Path.cwd())
    return build_live_task_rows(
        scenarios,
        repository_root=Path.cwd(),
        environment_digest=ENVIRONMENT_DIGEST,
        verifier_code_digest=VERIFIER_DIGEST,
        profile="sdk-minimal",
        patches=["examples/dsh/evolution.patch.yml"],
    )


def test_live_scenarios_cover_every_family_with_physical_fixtures() -> None:
    scenarios = load_live_scenarios(SCENARIO_PATH, repository_root=Path.cwd())

    assert tuple(scenario["family_id"] for scenario in scenarios) == FAMILY_IDS
    assert len({scenario["task_id"] for scenario in scenarios}) == 8
    assert all((Path.cwd() / scenario["fixture_path"]).is_file() for scenario in scenarios)


def test_live_scenarios_reject_family_verifier_mismatch(tmp_path: Path) -> None:
    scenarios = load_live_scenarios(SCENARIO_PATH, repository_root=Path.cwd())
    scenarios[0]["verifier_kind"] = "timeout_cleanup"
    path = tmp_path / "mismatched.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in scenarios), encoding="utf-8")

    with pytest.raises(ValueError, match="verifier_kind does not match family_id"):
        load_live_scenarios(path, repository_root=Path.cwd())


def test_live_rows_are_executable_dsh_tasks_without_trusted_rubrics() -> None:
    rows = _rows()
    resolver = TaskConfigResolver.from_file("examples/dsh/evolution_task_config_v3_live.yaml")

    assert len(rows) == 8
    for row in rows:
        task = row["extra_info"]["tools_kwargs"]["task"]
        metadata = task["metadata"]
        assert task["name"] == "dsh_architecture"
        assert metadata["split"] == "test"
        assert metadata["fixture_digest"].startswith("sha256:")
        assert metadata["scenario_contract_sha256"].startswith("sha256:")
        rendered = json.dumps(row, sort_keys=True)
        for forbidden in ("verifier_spec", "success_observation", "failure_observation", '"live"'):
            assert forbidden not in rendered
        resolved = resolver.resolve(task)
        assert resolved["agent"]["profile"] == "sdk-minimal"
        assert resolved["metadata"]["verifier_code_digest"] == VERIFIER_DIGEST


def test_diagnostic_prompt_uses_the_scenario_missing_provider() -> None:
    scenarios = load_live_scenarios(SCENARIO_PATH, repository_root=Path.cwd())
    diagnostic = next(row for row in scenarios if row["family_id"] == "diagnostic-recovery")
    diagnostic["live"]["missing_provider"] = "dshV3TestMissing"

    rows = build_live_task_rows(
        scenarios,
        repository_root=Path.cwd(),
        environment_digest=ENVIRONMENT_DIGEST,
        verifier_code_digest=VERIFIER_DIGEST,
        profile="sdk-minimal",
        patches=["examples/dsh/evolution.patch.yml"],
    )
    row = next(
        item
        for item in rows
        if item["extra_info"]["tools_kwargs"]["task"]["metadata"]["family_id"] == "diagnostic-recovery"
    )

    prompt = row["prompt"][0]["content"]
    assert prompt.count("dshV3TestMissing") == 2
    assert "waitingFor=dshV3TestMissing" in prompt


def test_live_contract_bundle_is_deterministic_but_blocked_until_process_smoke(tmp_path: Path) -> None:
    scenarios = load_live_scenarios(SCENARIO_PATH, repository_root=Path.cwd())
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = write_live_contract_bundle(
        scenarios,
        scenario_path=SCENARIO_PATH,
        repository_root=Path.cwd(),
        output_dir=first,
        environment_digest=ENVIRONMENT_DIGEST,
        profile="sdk-minimal",
        patches=["examples/dsh/evolution.patch.yml"],
    )
    second_manifest = write_live_contract_bundle(
        scenarios,
        scenario_path=SCENARIO_PATH,
        repository_root=Path.cwd(),
        output_dir=second,
        environment_digest=ENVIRONMENT_DIGEST,
        profile="sdk-minimal",
        patches=["examples/dsh/evolution.patch.yml"],
    )

    first_files = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files
    assert first_manifest == second_manifest
    assert first_manifest["schema"] == "dsh.evolution.live-contract-bundle.v1"
    assert first_manifest["status"] == "blocked"
    assert first_manifest["eligibility"] == {
        "online_rl": {"status": "blocked", "reasons": ["LIVE_FAMILY_SMOKE_PENDING"]}
    }
    assert first_manifest["counts"] == {"families": 8, "task_rows": 8}
    assert first_manifest["release_id"].startswith("sha256:")
    assert set(first_files) == {Path("live-task-rows.jsonl"), Path("manifest.json")}
    assert {entry["path"] for entry in first_manifest["verifier_bundle"]["files"]} == {
        "examples/dsh/evolution_verifier.py",
        "examples/dsh/evolution_v3_catalog.py",
        "examples/dsh/evolution_v3_live.py",
        "examples/dsh/evolution_v3_live_scenarios.jsonl",
        "examples/dsh/evolution_v3_live_verifier.py",
        "examples/dsh/evolution_v3_verifier.py",
        "examples/dsh/verifier.py",
    }
    resolver = TaskConfigResolver.from_file("examples/dsh/evolution_task_config_v3_live.yaml")
    assert (
        resolver.defaults_by_name["dsh_architecture"]["verifier_code_digest"]
        == first_manifest["verifier_bundle"]["sha256"]
    )
