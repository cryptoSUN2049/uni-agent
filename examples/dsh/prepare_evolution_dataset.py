"""Build Uni-Agent Parquet rows for the bounded DSH evolution task family.

Scenario JSONL is a release input, while the verifier and profile patch remain
operator-owned files. The builder verifies fixture bytes and records their
digests in each row; it never embeds an expected answer or trainer control in
the prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .prepare_dataset import _ID_PATTERN, _SPLITS, write_parquet
except ImportError:  # pragma: no cover - direct ``python examples/...`` invocation
    from examples.dsh.prepare_dataset import _ID_PATTERN, _SPLITS, write_parquet


def _digest_bytes(value: bytes) -> str:
    """Return the repository's canonical SHA-256 spelling."""
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _patches_digest(patches: list[str]) -> str:
    """Return the same ordered patch-stack identity emitted by the DSH runner."""
    encoded = json.dumps(patches, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _digest_bytes(encoded)


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load strict scenario objects from JSONL."""
    scenarios: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"scenario row {line_number} is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"scenario row {line_number} must be an object")
            scenarios.append(value)
    if not scenarios:
        raise ValueError("scenario file contains no rows")
    return scenarios


def _host_code(*, candidate_tool_name: str, operation: str) -> str:
    """Return the smallest valid host package body for one allowlisted operation."""
    operation_bodies = {
        "trim": "a.text.trim()",
        "normalize_whitespace": "a.text.replace(/\\s+/g,' ').trim()",
        "redact_email": ("a.text.replace(new RegExp(\"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\", 'g'), '<EMAIL>')"),
        "mask_digits": "a.text.replace(/[0-9]/g, '#')",
    }
    return (
        "return {inject:['tools'],apply(ctx){return harness.registerTool(ctx,"
        "harness.defineTool({name:'"
        + candidate_tool_name
        + "',description:'transform',parameters:{text:{type:'string',required:true}},"
        "output:{schema:{type:'string'},render:(_a,v)=>[{type:'text',text:v}]},"
        "execute:a=>" + operation_bodies[operation] + "}))}}"
    )


def _prompt(*, fixture_path: str, candidate_tool_name: str, operation: str) -> list[dict[str, str]]:
    """Render a bounded task instruction without revealing the reference output."""
    code = _host_code(candidate_tool_name=candidate_tool_name, operation=operation)
    define_arguments = json.dumps(
        {
            "plugin": {"kind": "new", "idPrefix": "evo"},
            "name": "Bounded transform",
            "purpose": "Apply the fixture transformation",
            "code": {"host": code},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {
            "role": "user",
            "content": (
                "Run one bounded DSH dynamic-plugin episode. Call exactly one tool in each assistant turn, wait "
                "for its result, and never batch or parallelize tool calls. The fixture is immutable at the absolute "
                "path "
                f'`{fixture_path}`. First call `str_replace_editor` with `{{"command":"view","path":'
                f"{json.dumps(fixture_path)}}}`; never edit it, call bash, or use filesystem APIs. Next call "
                "`cordis_inspect_list` once. Then call `cordis_define` with this exact JSON arguments object; "
                "copy it exactly and do not add a client half:\n"
                f"{define_arguments}\n"
                "The host code is intentionally complete and syntactically valid; preserve every quote, brace, "
                "and parenthesis. The operation is "
                f"`{operation}` and the candidate tool is `{candidate_tool_name}`. After define succeeds, read "
                "the returned separate `pluginId` and `packageId`. Because this is a fresh session with the fixed "
                "idPrefix `evo`, the first successful result is displayed as `Defined evo-1/pkg-1`: pass "
                "`pluginId`=`evo-1` and `packageId`=`pkg-1` (the suffix only), never `evo-1/pkg-1`. Then call "
                "`cordis_run` with those IDs and "
                '`{"mode":"run"}`. After a successful run, the very next action MUST be one call to the visible '
                f'`{candidate_tool_name}` tool with arguments `{{"text": <the exact input string read from the '
                "fixture>}}`. A plain answer at this point "
                "is invalid. After the candidate result, call `cordis_stop` and then `cordis_undefine`; do not "
                "emit a final answer until both cleanup results succeed. Make one define attempt only. Finish "
                'with exactly one JSON object and no prose: {"status":"promote","plugin_id":"<returned id>",'
                '"package_id":"<returned id>","evidence":["inspected","ran","transformed","cleaned"]}. '
                "The verifier computes the result independently; never invent a digest."
            ),
        }
    ]


def build_evolution_rows(
    scenario_path: Path,
    *,
    fixture_root: Path,
    environment_digest: str,
    verifier_id: str,
    verifier_version: str,
    verifier_code_digest: str,
    profile: str = "sdk-minimal",
    patches: list[str] | None = None,
    max_records: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build split rows after validating scenario identity and fixture hashes."""
    for label, digest in {
        "environment_digest": environment_digest,
        "verifier_code_digest": verifier_code_digest,
    }.items():
        if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError(f"{label} must be a sha256 digest")
    if not verifier_id.strip() or not verifier_version.strip() or not profile.strip():
        raise ValueError("verifier_id, verifier_version, and profile must be non-empty")
    ordered_patches = list(patches or [])
    if any(not isinstance(path, str) or not path.strip() for path in ordered_patches):
        raise ValueError("patches must contain non-empty paths")
    patch_digest = _patches_digest(ordered_patches)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in _SPLITS}
    seen: set[tuple[str, str]] = set()
    for row_number, scenario in enumerate(_load_scenarios(scenario_path), start=1):
        scenario_id = scenario.get("scenario_id")
        task_id = scenario.get("task_id")
        task_version = scenario.get("task_version")
        split = scenario.get("split")
        fixture_value = scenario.get("fixture_path")
        candidate_tool_name = scenario.get("candidate_tool_name")
        variant = scenario.get("variant", "basic")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError(f"scenario row {row_number} scenario_id must be non-empty")
        if not isinstance(task_id, str) or _ID_PATTERN.fullmatch(task_id) is None:
            raise ValueError(f"scenario row {row_number} task_id is invalid")
        if not isinstance(task_version, str) or not task_version.strip() or split not in _SPLITS:
            raise ValueError(f"scenario row {row_number} has an invalid task version or split")
        if not isinstance(fixture_value, str) or not fixture_value.strip() or not isinstance(candidate_tool_name, str):
            raise ValueError(f"scenario row {row_number} fixture_path and candidate_tool_name are required")
        if not candidate_tool_name.isidentifier() or candidate_tool_name in {"bash", "write", "edit"}:
            raise ValueError(f"scenario row {row_number} candidate_tool_name is invalid")
        if variant not in {"basic", "repair"}:
            raise ValueError(f"scenario row {row_number} variant is invalid")
        identity = (task_id, task_version)
        if identity in seen:
            raise ValueError(f"scenario row {row_number} duplicates {task_id}@{task_version}")
        seen.add(identity)
        fixture_rel = Path(fixture_value)
        if fixture_rel.is_absolute() or ".." in fixture_rel.parts:
            raise ValueError(f"scenario row {row_number} fixture_path must be relative and traversal-free")
        fixture_path = (fixture_root / fixture_rel).resolve()
        try:
            fixture_path.relative_to(fixture_root.resolve())
        except ValueError as exc:
            raise ValueError(f"scenario row {row_number} fixture_path escapes fixture_root") from exc
        if not fixture_path.is_file():
            raise ValueError(f"scenario row {row_number} fixture is missing: {fixture_path}")
        fixture_bytes = fixture_path.read_bytes()
        fixture = json.loads(fixture_bytes.decode("utf-8"))
        if not isinstance(fixture, dict) or fixture.get("schema") != "dsh.evolution.fixture.v1":
            raise ValueError(f"scenario row {row_number} fixture has the wrong schema")
        operation = fixture.get("operation")
        if operation not in {"trim", "normalize_whitespace", "redact_email", "mask_digits"}:
            raise ValueError(f"scenario row {row_number} fixture operation is not allowlisted")
        fixture_input = fixture.get("input")
        if not isinstance(fixture_input, str):
            raise ValueError(f"scenario row {row_number} fixture input must be a string")
        metadata = {
            "task_id": task_id,
            "task_version": task_version,
            "split": split,
            "scenario_id": scenario_id,
            "fixture_path": fixture_value,
            "fixture_digest": _digest_bytes(fixture_bytes),
            "operation": operation,
            "candidate_tool_name": candidate_tool_name,
            "variant": variant,
            "candidate_scope": "session-local-host-only",
            "mutation_budget": 0,
            "environment_digest": environment_digest,
            "verifier_id": verifier_id,
            "verifier_version": verifier_version,
            "verifier_code_digest": verifier_code_digest,
            "profile": profile,
            "patches_sha256": patch_digest,
        }
        if max_records is not None and len(rows_by_split[split]) >= max_records:
            continue
        rows_by_split[split].append(
            {
                "data_source": f"dsh/harness-evolution/{scenario_id}",
                "prompt": _prompt(
                    fixture_path=str(fixture_path),
                    candidate_tool_name=candidate_tool_name,
                    operation=operation,
                ),
                "extra_info": {"tools_kwargs": {"task": {"name": "dsh_architecture", "metadata": metadata}}},
            }
        )
    return {split: rows for split, rows in rows_by_split.items() if rows}


def main() -> None:
    """Build Parquet files from the scenario release input."""
    parser = argparse.ArgumentParser(description="Build DSH Harness evolution Uni-Agent Parquet rows")
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment-digest", required=True)
    parser.add_argument("--verifier-id", default="dsh-harness-evolution-verifier")
    parser.add_argument("--verifier-version", default="1")
    parser.add_argument("--verifier-code-digest", required=True)
    parser.add_argument("--profile", default="sdk-minimal")
    parser.add_argument("--patch", action="append", default=[])
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    if args.max_records is not None and args.max_records <= 0:
        raise SystemExit("--max-records must be positive")
    rows = build_evolution_rows(
        args.scenario_file,
        fixture_root=args.fixture_root,
        environment_digest=args.environment_digest,
        verifier_id=args.verifier_id,
        verifier_version=args.verifier_version,
        verifier_code_digest=args.verifier_code_digest,
        profile=args.profile,
        patches=args.patch,
        max_records=args.max_records,
    )
    write_parquet(rows, args.output_dir)
    print("wrote splits: " + ", ".join(f"{key}={len(value)}" for key, value in rows.items()))


if __name__ == "__main__":
    main()
