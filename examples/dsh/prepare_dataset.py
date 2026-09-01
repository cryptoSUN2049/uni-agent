"""Convert released DSH task seeds to Uni-Agent Parquet rows.

The converter accepts the official DSH release manifest or the data-layer
dual-backend release manifest. Verifier and Sandbox settings stay in the
operator-owned Task Config YAML; dataset rows carry prompt and task metadata
only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$", re.ASCII)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$", re.ASCII)
_SPLITS = ("train", "validation", "test", "holdout")
_DUAL_SPLITS = ("train", "validation", "test", "held_out")
_SEED_KEYS = {"schema", "task_id", "task_version", "split", "prompt", "metadata"}


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file {path} must contain an object")
    return value


def _load_jsonl_object(line: str, *, row_number: int, source: Path) -> dict[str, Any]:
    """Decode one object from a release JSONL file with a useful location error."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} row {row_number} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} row {row_number} must be a JSON object")
    return value


def _safe_relative(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"release file path must be relative and traversal-free: {path_value!r}")
    return path


def load_released_manifest(release_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load a released manifest and index its exact file hashes."""
    manifest_path = release_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "dsh.training-release.v1":
        raise ValueError("manifest has the wrong schema")
    if manifest.get("status") != "released":
        raise ValueError("manifest status must be released")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest.files must be a non-empty list")
    index: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("manifest.files entries must contain a path")
        path = _safe_relative(entry["path"])
        digest = entry.get("sha256")
        if not isinstance(digest, str) or _HASH_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"manifest file {path} has an invalid sha256")
        key = path.as_posix()
        if key in index:
            raise ValueError(f"manifest contains duplicate file {key}")
        index[key] = entry
        actual = release_dir / path
        if not actual.is_file():
            raise ValueError(f"manifest-listed file is missing: {actual}")
        if _sha256(actual) != digest:
            raise ValueError(f"manifest hash mismatch for {key}")
    return manifest, index


def load_dual_backend_manifest(release_dir: Path) -> tuple[dict[str, Any], Path]:
    """Resolve the eligible VERL Online RL seed artifact from an M3 release.

    The data-layer worktree owns the dual-backend manifest and file projection;
    this adapter only verifies the selected bytes before adding Uni-Agent's
    ``tools_kwargs.task`` envelope. Analysis-only releases are intentionally not
    accepted as training input.
    """
    manifest = _load_json(release_dir / "manifest.json")
    if manifest.get("schema") != "dsh.dual-backend-training-release.v1":
        raise ValueError("manifest has the wrong dual-backend schema")
    eligibility = manifest.get("eligibility")
    if not isinstance(eligibility, dict) or eligibility.get("status") != "eligible":
        reasons = eligibility.get("reasons", []) if isinstance(eligibility, dict) else []
        raise ValueError(f"dual-backend release is not eligible: {reasons}")
    try:
        file_entry = manifest["backends"]["verl"]["onlineRlSeed"]
        relative = _safe_relative(file_entry["path"])
        expected = file_entry["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("dual-backend manifest is missing backends.verl.onlineRlSeed") from exc
    if not isinstance(expected, str) or _HASH_PATTERN.fullmatch(expected) is None:
        raise ValueError("dual-backend VERL seed artifact has an invalid sha256")
    selected = release_dir / relative
    if not selected.is_file():
        raise ValueError(f"dual-backend VERL seed artifact is missing: {selected}")
    if _sha256(selected) != expected:
        raise ValueError(f"manifest hash mismatch for {relative.as_posix()}")
    return manifest, selected


def _validate_prompt(value: object, *, row_number: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"seed row {row_number} prompt must be a non-empty list")
    users = 0
    prompt: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict) or set(message) - {"role", "content"}:
            raise ValueError(f"seed row {row_number} prompt[{index}] has unknown fields")
        role = message.get("role")
        if role not in {"system", "user"}:
            raise ValueError(f"seed row {row_number} prompt[{index}] role must be system or user")
        content = message.get("content")
        if isinstance(content, str):
            if not content.strip():
                raise ValueError(f"seed row {row_number} prompt[{index}] content is blank")
        elif isinstance(content, list):
            if not content or any(
                not isinstance(block, dict)
                or block.get("type") != "text"
                or not isinstance(block.get("text"), str)
                or not block["text"].strip()
                for block in content
            ):
                raise ValueError(f"seed row {row_number} prompt[{index}] has invalid text blocks")
        else:
            raise ValueError(f"seed row {row_number} prompt[{index}] content must be text")
        users += role == "user"
        prompt.append({"role": role, "content": content})
    if users != 1:
        raise ValueError(f"seed row {row_number} must contain exactly one user message")
    return prompt


def adapt_verl_online_rl_seed(
    row: dict[str, Any],
    *,
    row_number: int = 1,
    verifier_code_digest: str | None = None,
) -> dict[str, Any]:
    """Adapt one M3 native VERL seed row to the Uni-Agent task envelope.

    M3 keeps provider-neutral DSH identity in VERL ``extra_info``. Uni-Agent's
    runner receives the same facts through ``extra_info.tools_kwargs.task``;
    this function adds that routing envelope while retaining the original
    fields for VERL reward/evaluation consumers. It does not create a new data
    authority or fabricate a trajectory.
    """
    if row.get("agent_name") != "dsh":
        raise ValueError(f"VERL seed row {row_number} agent_name must be 'dsh'")
    if row.get("ability") != "dsh-agent":
        raise ValueError(f"VERL seed row {row_number} ability must be 'dsh-agent'")
    if not isinstance(row.get("reward_model"), dict) or row["reward_model"].get("style") != "dsh_verifier":
        raise ValueError(f"VERL seed row {row_number} reward_model must use dsh_verifier")
    prompt = _validate_prompt(row.get("prompt"), row_number=row_number)
    source_info = row.get("extra_info")
    if not isinstance(source_info, dict):
        raise ValueError(f"VERL seed row {row_number} extra_info must be an object")
    task_id = source_info.get("task_id")
    task_version = source_info.get("task_version")
    split = source_info.get("split")
    environment_ref = source_info.get("environment_ref")
    verifier_id = source_info.get("verifier_id")
    verifier_version = source_info.get("verifier_version")
    seed_id = source_info.get("seed_id")
    if verifier_code_digest is not None and _HASH_PATTERN.fullmatch(verifier_code_digest) is None:
        raise ValueError("verifier_code_digest override must be a sha256 digest")
    if not isinstance(task_id, str) or _ID_PATTERN.fullmatch(task_id) is None:
        raise ValueError(f"VERL seed row {row_number} task_id is invalid")
    if not isinstance(task_version, str) or not task_version.strip():
        raise ValueError(f"VERL seed row {row_number} task_version is invalid")
    if split not in _DUAL_SPLITS:
        raise ValueError(f"VERL seed row {row_number} split is invalid")
    if not isinstance(environment_ref, str) or _HASH_PATTERN.fullmatch(environment_ref) is None:
        raise ValueError(f"VERL seed row {row_number} environment_ref must be a sha256 digest")
    if not isinstance(verifier_id, str) or _ID_PATTERN.fullmatch(verifier_id) is None:
        raise ValueError(f"VERL seed row {row_number} verifier_id is invalid")
    if not isinstance(verifier_version, str) or not verifier_version.strip():
        raise ValueError(f"VERL seed row {row_number} verifier_version is invalid")
    if not isinstance(seed_id, str) or not seed_id.strip():
        raise ValueError(f"VERL seed row {row_number} seed_id is invalid")

    normalized_split = "holdout" if split == "held_out" else split
    metadata: dict[str, Any] = {
        "task_id": task_id,
        "task_version": task_version,
        "split": normalized_split,
        "environment_digest": environment_ref,
        "environment_ref": environment_ref,
        "verifier_id": verifier_id,
        "verifier_version": verifier_version,
        "seed_id": seed_id,
    }
    if split == "held_out":
        metadata["source_split"] = split
    for key in ("budget", "solution", "expected_response_sha256", "verifier_code_digest", "source_trace_sha256"):
        if key not in source_info:
            continue
        value = source_info[key]
        if key.endswith("digest") or key == "source_trace_sha256":
            if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError(f"VERL seed row {row_number} {key} must be a sha256 digest")
        elif key == "solution" and (not isinstance(value, str) or not value):
            raise ValueError(f"VERL seed row {row_number} solution must be non-empty when present")
        elif key == "budget" and not isinstance(value, dict):
            raise ValueError(f"VERL seed row {row_number} budget must be an object")
        elif key == "expected_response_sha256" and (
            not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None
        ):
            raise ValueError(f"VERL seed row {row_number} expected_response_sha256 must be a sha256 digest")
        metadata[key] = value
    if verifier_code_digest is not None:
        existing_digest = metadata.get("verifier_code_digest")
        if existing_digest is not None and existing_digest != verifier_code_digest:
            raise ValueError(f"VERL seed row {row_number} verifier_code_digest conflicts with override")
        metadata["verifier_code_digest"] = verifier_code_digest

    # Tool schemas are owned by the DSH profile. Keep them in provenance so a
    # verifier can compare the profile's actual request header, rather than
    # silently discarding a model-visible input at the projection boundary.
    tools = row.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise ValueError(f"VERL seed row {row_number} tools must be an object list")
        metadata["source_tools"] = tools

    original_extra = dict(row["extra_info"])
    existing_tools_kwargs = original_extra.get("tools_kwargs")
    if existing_tools_kwargs is not None and not isinstance(existing_tools_kwargs, dict):
        raise ValueError(f"VERL seed row {row_number} extra_info.tools_kwargs must be an object")
    # Do not carry arbitrary nested runner controls from a dataset projection.
    original_extra["tools_kwargs"] = {"task": {"name": "dsh_architecture", "metadata": metadata}}
    adapted = dict(row)
    adapted["prompt"] = prompt
    adapted["extra_info"] = original_extra
    return adapted


def build_rows_from_verl_seed_file(
    seed_path: Path,
    *,
    max_records: int | None = None,
    verifier_code_digest: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build Uni-Agent rows from M3's ``verl-online-rl-seeds.jsonl`` projection."""
    rows_by_split = {split: [] for split in _SPLITS}
    seen: set[tuple[str, str]] = set()
    with seed_path.open(encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            source_row = _load_jsonl_object(line, row_number=row_number, source=seed_path)
            row = adapt_verl_online_rl_seed(
                source_row,
                row_number=row_number,
                verifier_code_digest=verifier_code_digest,
            )
            task = row["extra_info"]["tools_kwargs"]["task"]
            metadata = task["metadata"]
            identity = (metadata["task_id"], metadata["task_version"])
            if identity in seen:
                raise ValueError(f"VERL seed row {row_number} duplicates task {identity[0]}@{identity[1]}")
            seen.add(identity)
            split = metadata["split"]
            if max_records is not None and len(rows_by_split[split]) >= max_records:
                continue
            rows_by_split[split].append(row)
    return {split: rows for split, rows in rows_by_split.items() if rows}


def build_rows(
    manifest: dict[str, Any],
    seed_path: Path,
    *,
    max_records: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Validate task seeds and build the rows consumed by the Uni-Agent runner."""
    manifest_tasks = manifest.get("tasks")
    if not isinstance(manifest_tasks, list):
        raise ValueError("manifest.tasks must be a list")
    task_index: dict[tuple[str, str], dict[str, Any]] = {}
    for task in manifest_tasks:
        if not isinstance(task, dict):
            raise ValueError("manifest.tasks entries must be objects")
        identity = (task.get("task_id"), task.get("task_version"))
        if not isinstance(identity[0], str) or not isinstance(identity[1], str):
            raise ValueError("manifest task identity is incomplete")
        if identity in task_index:
            raise ValueError(f"manifest contains duplicate task {identity[0]}@{identity[1]}")
        task_index[identity] = task

    rows_by_split = {split: [] for split in _SPLITS}
    seen: set[tuple[str, str]] = set()
    with seed_path.open(encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                seed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"seed row {row_number} is not valid JSON") from exc
            if not isinstance(seed, dict) or set(seed) != _SEED_KEYS:
                raise ValueError(f"seed row {row_number} must contain exactly {sorted(_SEED_KEYS)}")
            if seed.get("schema") != "dsh.uni-agent.task-seed.v1":
                raise ValueError(f"seed row {row_number} has the wrong schema")
            task_id = seed.get("task_id")
            task_version = seed.get("task_version")
            split = seed.get("split")
            if (
                not isinstance(task_id, str)
                or _ID_PATTERN.fullmatch(task_id) is None
                or not isinstance(task_version, str)
                or not task_version
                or split not in _SPLITS
            ):
                raise ValueError(f"seed row {row_number} has an invalid task identity or split")
            identity = (task_id, task_version)
            if identity in seen:
                raise ValueError(f"seed row {row_number} duplicates task {task_id}@{task_version}")
            seen.add(identity)
            manifest_task = task_index.get(identity)
            if manifest_task is None:
                raise ValueError(f"seed row {row_number} is not listed in the release manifest")
            if manifest_task.get("split") != split:
                raise ValueError(f"seed row {row_number} split disagrees with the release manifest")
            metadata = seed.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"seed row {row_number} metadata must be an object")
            prompt = _validate_prompt(seed.get("prompt"), row_number=row_number)
            row_metadata = dict(metadata)
            for key, expected in {
                "task_id": task_id,
                "task_version": task_version,
                "split": split,
            }.items():
                if key in row_metadata and row_metadata[key] != expected:
                    raise ValueError(f"seed row {row_number} metadata.{key} conflicts with identity")
                row_metadata[key] = expected
            if max_records is not None and len(rows_by_split[split]) >= max_records:
                continue
            rows_by_split[split].append(
                {
                    "data_source": "dsh/" + task_id,
                    "prompt": prompt,
                    "extra_info": {
                        "tools_kwargs": {
                            "task": {
                                "name": "dsh_architecture",
                                "metadata": row_metadata,
                            }
                        }
                    },
                }
            )
    return {split: rows for split, rows in rows_by_split.items() if rows}


def write_parquet(rows_by_split: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    """Write one Parquet file per split using the optional datasets dependency."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("install the Uni-Agent dataset dependencies before writing Parquet") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        Dataset.from_list(rows).to_parquet(str(output_dir / (split + ".parquet")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a released DSH or dual-backend seed release to Uni-Agent Parquet"
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--seed-file", default="task-seeds.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--verifier-code-digest",
        default=None,
        help="Optional operator-pinned verifier code digest for dual-backend rows that omit it.",
    )
    args = parser.parse_args()
    if args.max_records is not None and args.max_records <= 0:
        raise SystemExit("--max-records must be positive")
    manifest_path = args.release_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") == "dsh.dual-backend-training-release.v1":
        _, seed_path = load_dual_backend_manifest(args.release_dir)
        rows_by_split = build_rows_from_verl_seed_file(
            seed_path,
            max_records=args.max_records,
            verifier_code_digest=args.verifier_code_digest,
        )
    else:
        manifest, file_index = load_released_manifest(args.release_dir)
        seed_rel = _safe_relative(args.seed_file)
        if seed_rel.as_posix() not in file_index:
            raise SystemExit(f"seed file is not listed in manifest.files: {seed_rel}")
        rows_by_split = build_rows(manifest, args.release_dir / seed_rel, max_records=args.max_records)
    if not rows_by_split:
        raise SystemExit("seed file contains no task rows")
    write_parquet(rows_by_split, args.output_dir)
    print(f"wrote splits: {', '.join(f'{key}={len(value)}' for key, value in rows_by_split.items())}")


if __name__ == "__main__":
    main()
