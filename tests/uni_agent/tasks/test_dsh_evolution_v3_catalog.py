from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from examples.dsh.evolution_v3_catalog import (
    FAMILY_IDS,
    build_cpu_matrix,
    build_task_seeds,
    load_catalog,
    write_cpu_release,
)
from examples.dsh.evolution_v3_verifier import verify_matrix, verify_matrix_case

CATALOG_PATH = Path("examples/dsh/evolution_v3_catalog.json")
VERIFIER_BUNDLE = (
    CATALOG_PATH,
    Path("examples/dsh/evolution_v3_catalog.py"),
    Path("examples/dsh/evolution_v3_verifier.py"),
)


def test_v3_catalog_defines_the_eight_approved_families_once() -> None:
    catalog = load_catalog(CATALOG_PATH)

    assert catalog["schema"] == "dsh.evolution.catalog.v3"
    assert tuple(family["family_id"] for family in catalog["families"]) == FAMILY_IDS
    assert len(set(FAMILY_IDS)) == 8
    assert all(family["scenario_id"].startswith(f"{family['family_id']}-") for family in catalog["families"])
    assert all(not Path(family["fixture_ref"]).is_absolute() for family in catalog["families"])


def test_v3_policy_seeds_exclude_trusted_answers_and_final_holdout() -> None:
    seeds = build_task_seeds(load_catalog(CATALOG_PATH))

    assert len(seeds) == 8
    assert len({seed["task_id"] for seed in seeds}) == 8
    for seed in seeds:
        assert seed["split"] == "cpu_check"
        assert set(seed) == {"schema", "task_id", "task_version", "split", "prompt", "metadata"}
        assert "verifier_spec" not in seed["metadata"]
        assert "success_observation" not in seed["metadata"]
        assert "failure_observation" not in seed["metadata"]
        assert "final_holdout" not in json.dumps(seed, sort_keys=True)


def test_v3_cpu_matrix_distinguishes_success_failure_and_tamper() -> None:
    matrix = build_cpu_matrix(load_catalog(CATALOG_PATH))

    assert len(matrix) == 24
    assert len({case["case_id"] for case in matrix}) == 24
    assert Counter((case["family_id"], case["case_kind"]) for case in matrix) == Counter(
        (family_id, case_kind) for family_id in FAMILY_IDS for case_kind in ("success", "failure", "tamper")
    )

    for case in matrix:
        result = verify_matrix_case(case)
        if case["case_kind"] == "success":
            assert result["eligible"] is True
            assert result["passed"] is True
            assert result["reward"] == 1.0
            assert result["reasons"] == []
        elif case["case_kind"] == "failure":
            assert result["eligible"] is True
            assert result["passed"] is False
            assert result["reward"] == 0.0
            assert result["reasons"]
            assert "observation_digest_mismatch" not in result["reasons"]
        else:
            assert result["eligible"] is False
            assert result["passed"] is False
            assert result["reward"] == 0.0
            assert result["reasons"] == ["observation_digest_mismatch"]

    assert verify_matrix(matrix) == {
        "cases": 24,
        "eligible": 16,
        "passed": 8,
        "rejected": 8,
    }


def test_v3_cpu_matrix_rejects_an_incorrect_expected_outcome() -> None:
    matrix = build_cpu_matrix(load_catalog(CATALOG_PATH))
    matrix[0]["case_kind"] = "failure"

    with pytest.raises(ValueError, match="expected outcome"):
        verify_matrix(matrix)


def test_v3_cpu_matrix_rejects_a_non_string_identity() -> None:
    matrix = build_cpu_matrix(load_catalog(CATALOG_PATH))
    matrix[0]["family_id"] = []

    with pytest.raises(ValueError, match="family_id"):
        verify_matrix(matrix)


def test_v3_cpu_release_is_byte_deterministic_and_hash_bound(tmp_path: Path) -> None:
    catalog = load_catalog(CATALOG_PATH)
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_cpu_release(
        catalog,
        catalog_path=CATALOG_PATH,
        verifier_bundle=VERIFIER_BUNDLE,
        repository_root=Path.cwd(),
        output_dir=first,
    )
    write_cpu_release(
        catalog,
        catalog_path=CATALOG_PATH,
        verifier_bundle=VERIFIER_BUNDLE,
        repository_root=Path.cwd(),
        output_dir=second,
    )

    first_files = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files
    assert set(first_files) == {Path("cpu-matrix.jsonl"), Path("manifest.json"), Path("task-seeds.jsonl")}

    manifest = json.loads(first_files[Path("manifest.json")])
    assert manifest["schema"] == "dsh.evolution.cpu-release.v1"
    assert manifest["counts"] == {"families": 8, "matrix_cases": 24, "task_seeds": 8}
    assert manifest["catalog"]["path"] == CATALOG_PATH.as_posix()
    assert manifest["verifier_bundle"]["sha256"].startswith("sha256:")
    assert all(not Path(entry["path"]).is_absolute() for entry in manifest["verifier_bundle"]["files"])
    assert all(entry["sha256"].startswith("sha256:") for entry in manifest["files"])


def test_v3_catalog_rejects_duplicate_family_identity(tmp_path: Path) -> None:
    value = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    value["families"].append(value["families"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="family identity"):
        load_catalog(path)
