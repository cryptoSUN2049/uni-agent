import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "examples/dsh/ops/launch_qwen3_4b_online_rl.sh"
TRAINING_LAUNCHER = REPO_ROOT / "examples/dsh/train_qwen3_4b_online_rl.sh"
SUPERVISOR = REPO_ROOT / "examples/dsh/ops/supervise_qwen3_4b_online_rl.sh"
TEARDOWN = REPO_ROOT / "examples/dsh/ops/teardown_qwen3_4b_online_rl.sh"
STATUS = REPO_ROOT / "examples/dsh/ops/status_qwen3_4b_online_rl.sh"


def _start_signal_aware_supervisor(tmp_path: Path) -> tuple[subprocess.Popen[str], Path, Path]:
    run_root = tmp_path / "run"
    run_root.mkdir()
    child_ready = tmp_path / "child-ready"
    child_stopped = tmp_path / "child-stopped"
    child_script = tmp_path / "child.py"
    child_script.write_text(
        """import signal
import sys
import time
from pathlib import Path

ready = Path(sys.argv[1])
stopped = Path(sys.argv[2])

def stop(signum, frame):
    del signum, frame
    stopped.write_text("stopped\\n", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
ready.write_text("ready\\n", encoding="utf-8")
while True:
    time.sleep(0.05)
""",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [
            "/bin/bash",
            str(SUPERVISOR),
            str(run_root),
            sys.executable,
            sys.executable,
            str(child_script),
            str(child_ready),
            str(child_stopped),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if child_ready.exists() and (run_root / "supervisor-ready").exists():
            return process, run_root, child_stopped
        if process.poll() is not None:
            raise AssertionError(f"supervisor exited early: {process.communicate()}")
        time.sleep(0.01)
    process.kill()
    process.wait(timeout=5)
    raise AssertionError("supervisor or child did not become ready")


def test_detached_launcher_finalizes_manifest_after_training_exits(tmp_path: Path):
    model_path = tmp_path / "model"
    data_root = tmp_path / "data"
    fake_venv_bin = tmp_path / "venv/bin"
    run_root = tmp_path / "run"
    model_path.mkdir()
    data_root.mkdir()
    fake_venv_bin.mkdir(parents=True)

    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    (model_path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (data_root / "train.parquet").write_bytes(b"train")
    (data_root / "holdout.parquet").write_bytes(b"holdout")
    (data_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    task_config = tmp_path / "task.yaml"
    task_config.write_text("tasks: []\n", encoding="utf-8")

    fake_python = fake_venv_bin / "python"
    fake_python.write_text(
        """#!/bin/sh
if [ "$1" = "-c" ]; then
  exit 0
fi
exec "$REAL_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    child_finished = tmp_path / "child-finished"
    fake_bash = fake_venv_bin / "bash"
    fake_bash.write_text(
        """#!/bin/sh
if [ "$1" != "$TRAINING_LAUNCHER" ]; then
  exec /bin/bash "$@"
fi
until grep -q '"status": "running"' "$RUN_ROOT/run-manifest.json" 2>/dev/null; do
  sleep 0.01
done
trap ': > "$CHILD_FINISHED"' EXIT
exit "$TRAINING_EXIT_CODE"
""",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    env = {
        **os.environ,
        "MODEL_LICENSE_APPROVED": "1",
        "MODEL_PATH": str(model_path),
        "DATA_ROOT": str(data_root),
        "TASK_CONFIG": str(task_config),
        "RUN_ROOT": str(run_root),
        "DSH_VENV": str(fake_venv_bin.parent),
        "PYTHON_BIN": sys.executable,
        "REAL_PYTHON": sys.executable,
        "TRAINING_LAUNCHER": str(TRAINING_LAUNCHER),
        "TRAINING_EXIT_CODE": "23",
        "CHILD_FINISHED": str(child_finished),
    }
    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    pid = int((run_root / "pid").read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process_state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if child_finished.exists() and (not process_state or process_state.startswith("Z")):
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"detached training process {pid} did not exit")

    manifest = json.loads((run_root / "run-manifest.json").read_text(encoding="utf-8"))
    terminal_state = {
        "status": manifest.get("status"),
        "exit_code": manifest.get("exit_code"),
        "has_finished_at": bool(manifest.get("finished_at")),
    }
    assert terminal_state == {"status": "failed", "exit_code": 23, "has_finished_at": True}


def test_supervisor_forwards_term_and_records_interruption(tmp_path: Path):
    process, run_root, child_stopped = _start_signal_aware_supervisor(tmp_path)
    try:
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode == 143, (stdout, stderr)
    assert child_stopped.is_file()
    manifest = json.loads((run_root / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["exit_code"] == 0
    assert manifest["termination_signal"] == "TERM"
    assert manifest["finished_at"]


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="teardown validates Linux /proc command lines")
def test_teardown_stops_supervisor_and_preserves_terminal_evidence(tmp_path: Path):
    process, run_root, child_stopped = _start_signal_aware_supervisor(tmp_path)
    env = {**os.environ, "PYTHON_BIN": sys.executable, "DSH_VENV": str(tmp_path / "missing-venv")}
    teardown = subprocess.Popen(
        ["/bin/bash", str(TEARDOWN), str(run_root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        supervisor_stdout, supervisor_stderr = process.communicate(timeout=5)
        teardown_stdout, teardown_stderr = teardown.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if teardown.poll() is None:
            teardown.kill()
            teardown.wait(timeout=5)

    assert process.returncode == 143, (supervisor_stdout, supervisor_stderr)
    assert teardown.returncode == 0, (teardown_stdout, teardown_stderr)
    assert child_stopped.is_file()
    manifest = json.loads((run_root / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "torn_down"
    assert manifest["exit_code"] == 0
    assert manifest["termination_signal"] == "TERM"
    assert manifest["finished_at"]


def test_supervisor_records_an_immediate_worker_failure(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()

    result = subprocess.run(
        [
            "/bin/bash",
            str(SUPERVISOR),
            str(run_root),
            sys.executable,
            "/bin/sh",
            "-c",
            "exit 31",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 31, result.stderr
    manifest = json.loads((run_root / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 31
    assert manifest["finished_at"]


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (
            "tool_parser_attempt backend=vllm parser=hermes\n",
            "parser_attempts=1 parser_recovered_calls=0 parser_rejected_calls=0 "
            "parser_malformed_attempts=0 parser_malformed_rate=0.000000",
        ),
        (
            "tool_parser_attempt backend=vllm parser=hermes\n"
            "tool_parser_recovery backend=vllm parser=hermes recovered_calls=2\n"
            "tool_parser_attempt backend=vllm parser=hermes\n"
            "tool_parser_rejection backend=vllm parser=hermes rejected_calls=1\n"
            "tool_parser_attempt backend=vllm parser=hermes\n",
            "parser_attempts=3 parser_recovered_calls=2 parser_rejected_calls=1 "
            "parser_malformed_attempts=2 parser_malformed_rate=0.666667",
        ),
        (
            "Error in extracting tool call from response.\n",
            "parser_attempts=0 parser_recovered_calls=0 parser_rejected_calls=0 "
            "parser_malformed_attempts=0 parser_malformed_rate=unavailable "
            "parser_telemetry_complete=false parser_unmatched_legacy_errors=1",
        ),
    ],
    ids=["zero-malformed", "recovered-and-rejected", "legacy-incomplete"],
)
def test_status_reports_structured_parser_counters(tmp_path: Path, events: str, expected: str):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run.log").write_text(events, encoding="utf-8")
    env = {**os.environ, "PYTHON_BIN": sys.executable, "DSH_VENV": str(tmp_path / "missing-venv")}

    result = subprocess.run(
        ["/bin/bash", str(STATUS), str(run_root)],
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
