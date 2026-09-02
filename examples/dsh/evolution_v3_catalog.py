"""Validate the canonical DSH v3 task catalog and generate CPU-only evidence.

The catalog is the single authored source for the initial eight task families.
Policy-facing seeds omit verifier specifications and expected observations. The
separate CPU matrix exercises success, valid policy failure, and invalid
artifact tampering before any Parquet projection or GPU calibration exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

FAMILY_IDS = (
    "runtime-grounding",
    "lifecycle-composition",
    "multi-step-configuration",
    "diagnostic-recovery",
    "timeout-cleanup",
    "permission-abstention",
    "reward-hacking-resistance",
    "transfer-composition",
)

_VERIFIER_KINDS = {
    "diagnostic_recovery",
    "lifecycle_composition",
    "multi_step_configuration",
    "permission_abstention",
    "reward_hacking_resistance",
    "runtime_grounding",
    "timeout_cleanup",
    "transfer_composition",
}
_SHA256_PREFIX = "sha256:"


def _digest_bytes(value: bytes) -> str:
    """Return the canonical SHA-256 spelling used by DSH artifacts."""
    return _SHA256_PREFIX + hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict JSON deterministically with one trailing newline."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not strict JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Encode an ordered list as canonical JSON Lines."""
    return b"".join(canonical_json_bytes(row) for row in rows)


def _reject_nonfinite(value: str) -> None:
    """Reject JSON constants that Python otherwise accepts as floats."""
    raise ValueError(f"catalog contains non-finite JSON constant {value}")


def _required_string(value: object, *, label: str) -> str:
    """Return a non-empty string or reject the catalog field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_object(value: object, *, label: str) -> dict[str, Any]:
    """Return a JSON object or reject the catalog field."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_budget(value: object, *, label: str) -> None:
    """Require a non-empty object whose numeric limits are non-negative."""
    budget = _required_object(value, label=label)
    if not budget:
        raise ValueError(f"{label} must not be empty")
    for key, item in budget.items():
        if key == "protected_paths":
            if not isinstance(item, list) or any(not isinstance(path, str) or not path for path in item):
                raise ValueError(f"{label}.protected_paths must contain non-empty strings")
        elif type(item) is not int or item < 0:
            raise ValueError(f"{label}.{key} must be a non-negative integer")


def _validate_family(family: object, *, index: int) -> dict[str, Any]:
    """Validate one authored family without adding defaults."""
    value = _required_object(family, label=f"families[{index}]")
    for key in ("family_id", "scenario_id", "task_id", "task_version", "fixture_ref", "verifier_kind", "prompt"):
        _required_string(value.get(key), label=f"families[{index}].{key}")
    fixture_ref = Path(value["fixture_ref"])
    if fixture_ref.is_absolute() or ".." in fixture_ref.parts:
        raise ValueError(f"families[{index}].fixture_ref must be relative and traversal-free")
    if value["verifier_kind"] not in _VERIFIER_KINDS:
        raise ValueError(f"families[{index}].verifier_kind is not allowlisted")
    for reserved in ("verifier_spec", "success_observation", "failure_observation"):
        if reserved in value["prompt"]:
            raise ValueError(f"families[{index}].prompt exposes trusted field {reserved}")
    _validate_budget(value.get("mutation_budget"), label=f"families[{index}].mutation_budget")
    _validate_budget(value.get("token_budget"), label=f"families[{index}].token_budget")
    _required_object(value.get("verifier_spec"), label=f"families[{index}].verifier_spec")
    _required_object(value.get("success_observation"), label=f"families[{index}].success_observation")
    _required_object(value.get("failure_observation"), label=f"families[{index}].failure_observation")
    return value


def load_catalog(path: Path) -> dict[str, Any]:
    """Load and validate the complete canonical v3 catalog."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as exc:
        raise ValueError(f"catalog is not valid JSON: {exc}") from exc
    catalog = _required_object(value, label="catalog")
    if catalog.get("schema") != "dsh.evolution.catalog.v3":
        raise ValueError("catalog has the wrong schema")
    _required_string(catalog.get("catalog_id"), label="catalog.catalog_id")
    _required_string(catalog.get("catalog_version"), label="catalog.catalog_version")
    families = catalog.get("families")
    if not isinstance(families, list):
        raise ValueError("catalog.families must be an array")
    validated = [_validate_family(family, index=index) for index, family in enumerate(families)]
    family_ids = [family["family_id"] for family in validated]
    task_ids = [family["task_id"] for family in validated]
    scenario_ids = [family["scenario_id"] for family in validated]
    if (
        len(set(family_ids)) != len(family_ids)
        or len(set(task_ids)) != len(task_ids)
        or len(set(scenario_ids)) != len(scenario_ids)
    ):
        raise ValueError("catalog family identity fields must be unique")
    if tuple(family_ids) != FAMILY_IDS:
        raise ValueError(f"catalog families must be exactly {FAMILY_IDS!r}")
    return catalog


def build_task_seeds(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Project policy-facing CPU seeds without trusted verifier answers."""
    seeds: list[dict[str, Any]] = []
    for family in catalog["families"]:
        seeds.append(
            {
                "schema": "dsh.evolution.cpu-task-seed.v1",
                "task_id": family["task_id"],
                "task_version": family["task_version"],
                "split": "cpu_check",
                "prompt": [{"role": "user", "content": family["prompt"]}],
                "metadata": {
                    "catalog_id": catalog["catalog_id"],
                    "catalog_version": catalog["catalog_version"],
                    "family_id": family["family_id"],
                    "scenario_id": family["scenario_id"],
                    "fixture_ref": family["fixture_ref"],
                    "mutation_budget": copy.deepcopy(family["mutation_budget"]),
                    "token_budget": copy.deepcopy(family["token_budget"]),
                    "verifier_kind": family["verifier_kind"],
                },
            }
        )
    return seeds


def _matrix_case(
    family: dict[str, Any],
    *,
    case_kind: str,
    observation: dict[str, Any],
    declared_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one hash-bound CPU verifier case."""
    declared = observation if declared_observation is None else declared_observation
    return {
        "schema": "dsh.evolution.cpu-case.v1",
        "case_id": f"{family['family_id']}--{case_kind}",
        "case_kind": case_kind,
        "family_id": family["family_id"],
        "scenario_id": family["scenario_id"],
        "verifier_kind": family["verifier_kind"],
        "verifier_spec": copy.deepcopy(family["verifier_spec"]),
        "observation": copy.deepcopy(observation),
        "observation_sha256": _digest_bytes(canonical_json_bytes(declared)),
    }


def build_cpu_matrix(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Build success, valid-failure, and tamper cases for every family."""
    matrix: list[dict[str, Any]] = []
    for family in catalog["families"]:
        success = copy.deepcopy(family["success_observation"])
        failure = copy.deepcopy(family["failure_observation"])
        tampered = copy.deepcopy(success)
        tampered["_tampered"] = True
        matrix.extend(
            (
                _matrix_case(family, case_kind="success", observation=success),
                _matrix_case(family, case_kind="failure", observation=failure),
                _matrix_case(
                    family,
                    case_kind="tamper",
                    observation=tampered,
                    declared_observation=success,
                ),
            )
        )
    return matrix


def _repository_file(path: Path, *, repository_root: Path) -> tuple[str, bytes]:
    """Read one verifier-bundle file and return its stable repository path."""
    root = repository_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"verifier bundle path escapes repository root: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"verifier bundle file is missing: {relative.as_posix()}")
    return relative.as_posix(), resolved.read_bytes()


def _bundle_identity(paths: tuple[Path, ...], *, repository_root: Path) -> dict[str, Any]:
    """Hash the complete ordered verifier bundle, including repository paths."""
    entries = []
    seen: set[str] = set()
    for path in paths:
        relative, raw = _repository_file(path, repository_root=repository_root)
        if relative in seen:
            continue
        seen.add(relative)
        entries.append({"path": relative, "sha256": _digest_bytes(raw)})
    entries.sort(key=lambda item: item["path"])
    if not entries:
        raise ValueError("verifier bundle must contain at least one file")
    return {"sha256": _digest_bytes(canonical_json_bytes(entries)), "files": entries}


def _atomic_write(path: Path, data: bytes) -> None:
    """Write one generated artifact through a same-directory replacement."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_cpu_release(
    catalog: dict[str, Any],
    *,
    catalog_path: Path,
    verifier_bundle: tuple[Path, ...],
    repository_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write deterministic CPU seeds, matrix cases, and their manifest."""
    output_dir.mkdir(parents=True, exist_ok=False)
    seeds = build_task_seeds(catalog)
    matrix = build_cpu_matrix(catalog)
    seed_bytes = _jsonl_bytes(seeds)
    matrix_bytes = _jsonl_bytes(matrix)
    _atomic_write(output_dir / "task-seeds.jsonl", seed_bytes)
    _atomic_write(output_dir / "cpu-matrix.jsonl", matrix_bytes)

    catalog_relative, catalog_bytes = _repository_file(catalog_path, repository_root=repository_root)
    manifest = {
        "schema": "dsh.evolution.cpu-release.v1",
        "release_id": f"{catalog['catalog_id']}@{catalog['catalog_version']}-cpu-matrix",
        "catalog": {"path": catalog_relative, "sha256": _digest_bytes(catalog_bytes)},
        "verifier_bundle": _bundle_identity(verifier_bundle, repository_root=repository_root),
        "counts": {"families": len(catalog["families"]), "matrix_cases": len(matrix), "task_seeds": len(seeds)},
        "files": [
            {"path": "cpu-matrix.jsonl", "records": len(matrix), "sha256": _digest_bytes(matrix_bytes)},
            {"path": "task-seeds.jsonl", "records": len(seeds), "sha256": _digest_bytes(seed_bytes)},
        ],
    }
    _atomic_write(output_dir / "manifest.json", canonical_json_bytes(manifest))
    return manifest


def main() -> None:
    """Generate the v3 CPU release from the canonical checked-in catalog."""
    parser = argparse.ArgumentParser(description="Generate the DSH v3 CPU verifier matrix")
    parser.add_argument("--catalog", type=Path, default=Path("examples/dsh/evolution_v3_catalog.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    module_path = Path(__file__)
    bundle = (args.catalog, module_path, module_path.with_name("evolution_v3_verifier.py"))
    manifest = write_cpu_release(
        catalog,
        catalog_path=args.catalog,
        verifier_bundle=bundle,
        repository_root=args.repository_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
