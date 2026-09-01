"""Create the immutable manifest for the expanded DSH evolution Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment-digest", required=True)
    parser.add_argument("--verifier-code-digest", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(args.scenario_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"scenario row {line_number} is not an object")
        rows.append(value)
    identities = [(row.get("task_id"), row.get("task_version")) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("scenario task identities are not unique")
    counts = {split: sum(row.get("split") == split for row in rows) for split in ("train", "holdout")}
    if counts != {"train": 16, "holdout": 8}:
        raise ValueError(f"expanded corpus must contain train=16 and holdout=8, got {counts}")
    fixtures: dict[str, dict[str, str]] = {}
    operations = {name: 0 for name in ("trim", "normalize_whitespace", "redact_email", "mask_digits")}
    for row in rows:
        fixture = (args.repo_root / str(row["fixture_path"])).resolve()
        fixture.relative_to(args.repo_root.resolve())
        fixture_value = json.loads(fixture.read_text(encoding="utf-8"))
        operation = fixture_value.get("operation")
        if operation not in operations:
            raise ValueError(f"unsupported operation in {fixture}")
        operations[operation] += 1
        fixtures[str(row["scenario_id"])] = {
            "path": str(row["fixture_path"]),
            "sha256": _sha256(fixture),
        }
    files: dict[str, dict[str, object]] = {}
    for name, expected_records in (("train.parquet", 16), ("holdout.parquet", 8)):
        path = args.output_dir / name
        digest = _sha256(path)
        try:
            import pyarrow.parquet as parquet

            records = parquet.read_metadata(path).num_rows
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to write a dataset manifest") from exc
        if records != expected_records:
            raise ValueError(f"{name} has {records} rows, expected {expected_records}")
        files[name] = {"sha256": digest, "records": records}
    manifest = {
        "schema": "dsh.evolution-dataset-manifest.v1",
        "dataset_id": "dsh-harness-evolution-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "cryptoSUN2049/uni-agent",
            "branch": "dsh-adapter",
            "commit": _git_revision(args.repo_root),
            "scenario_file": str(args.scenario_file.relative_to(args.repo_root)),
            "scenario_file_sha256": _sha256(args.scenario_file),
        },
        "task": {
            "task_name": "dsh_architecture",
            "profile": "sdk-minimal",
            "patches": ["examples/dsh/evolution.patch.yml"],
            "environment_digest": args.environment_digest,
            "verifier_id": "dsh-harness-evolution-verifier",
            "verifier_version": "1",
            "verifier_code_digest": args.verifier_code_digest,
        },
        "counts": counts,
        "operations": operations,
        "fixtures": fixtures,
        "files": files,
        "status": "generated",
    }
    target = args.output_dir / "manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(target)


if __name__ == "__main__":
    main()
