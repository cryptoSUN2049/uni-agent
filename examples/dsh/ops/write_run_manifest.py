"""Write a restart-safe, non-secret manifest for a DSH online-RL run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_STATUSES = ("prepared", "running", "completed", "failed", "interrupted", "torn_down")
TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted", "torn_down"})


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--status", choices=RUN_STATUSES, required=True)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--worker-pid", type=int, default=None)
    parser.add_argument("--exit-code", type=int, default=None)
    parser.add_argument("--termination-signal")
    parser.add_argument("--command-file", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--holdout-file", type=Path)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--verl-root", type=Path)
    parser.add_argument("--dsh-sha")
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_root / "run-manifest.json"
    manifest = _read_json(manifest_path) or {"schema": "dsh.online-rl.run-manifest.v1"}
    updated_at = datetime.now(timezone.utc).isoformat()
    manifest.update(
        {
            "schema": "dsh.online-rl.run-manifest.v1",
            "updated_at": updated_at,
            "status": args.status,
            "run_root": str(args.run_root),
        }
    )
    if args.status == "prepared":
        for key in ("pid", "worker_pid", "started_at", "finished_at", "exit_code", "termination_signal"):
            manifest.pop(key, None)
    elif args.status == "running":
        manifest.setdefault("started_at", updated_at)
        for key in ("finished_at", "exit_code", "termination_signal"):
            manifest.pop(key, None)
    elif args.status in TERMINAL_STATUSES:
        manifest.setdefault("finished_at", updated_at)
    if args.pid is not None:
        manifest["pid"] = args.pid
    if args.worker_pid is not None:
        manifest["worker_pid"] = args.worker_pid
    if args.exit_code is not None:
        manifest["exit_code"] = args.exit_code
    if args.termination_signal:
        manifest["termination_signal"] = args.termination_signal
    if args.command_file:
        manifest["command_file"] = str(args.command_file)
        if args.command_file.is_file():
            manifest["command"] = args.command_file.read_text(encoding="utf-8").strip()
    if args.model_path:
        manifest["model"] = str(args.model_path)
        digest = _sha256(args.model_path / "config.json")
        if digest:
            manifest.setdefault("sha256", {})["model_config"] = digest
    if args.model_id:
        manifest["model_id"] = args.model_id
    for name, path in (
        ("train_parquet", args.train_file),
        ("holdout_parquet", args.holdout_file),
        ("task_config", args.task_config),
        ("dataset_manifest", args.dataset_manifest),
    ):
        if path:
            manifest.setdefault("sha256", {})[name] = _sha256(path)
            manifest.setdefault("paths", {})[name] = str(path)
    if args.dataset_manifest:
        dataset = _read_json(args.dataset_manifest)
        if dataset:
            manifest["dataset"] = {
                "dataset_id": dataset.get("dataset_id"),
                "counts": dataset.get("counts"),
                "source_commit": dataset.get("source", {}).get("commit")
                if isinstance(dataset.get("source"), dict)
                else None,
            }
    if args.repo_root:
        manifest["uni_agent_sha"] = _git_revision(args.repo_root)
    if args.verl_root:
        manifest["verl_sha"] = _git_revision(args.verl_root)
    if args.dsh_sha:
        manifest["dsh_sha"] = args.dsh_sha
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
