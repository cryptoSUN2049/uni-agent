"""Summarize content-free Tool parser markers from a run log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_RECOVERY_PATTERN = re.compile(r"\btool_parser_recovery\b[^\r\n]*\brecovered_calls=(\d+)\b")
_REJECTION_PATTERN = re.compile(r"\btool_parser_rejection\b[^\r\n]*\brejected_calls=(\d+)\b")


def summarize(log_path: Path) -> dict[str, int | float | bool | None]:
    """Return parser counters without retaining or returning model content."""
    attempts = 0
    recovered_calls = 0
    rejected_calls = 0
    recovered_attempts = 0
    rejected_attempts = 0
    legacy_errors = 0
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "tool_parser_attempt " in line:
                    attempts += 1
                if "Error in extracting tool call from response." in line:
                    legacy_errors += 1
                recovery = _RECOVERY_PATTERN.search(line)
                if recovery:
                    recovered_attempts += 1
                    recovered_calls += int(recovery.group(1))
                rejection = _REJECTION_PATTERN.search(line)
                if rejection:
                    rejected_attempts += 1
                    rejected_calls += int(rejection.group(1))

    malformed_attempts = recovered_attempts + rejected_attempts
    unmatched_legacy_errors = max(legacy_errors - malformed_attempts, 0)
    telemetry_complete = attempts >= malformed_attempts and unmatched_legacy_errors == 0
    malformed_rate = malformed_attempts / attempts if telemetry_complete and attempts else None
    if telemetry_complete and not attempts:
        malformed_rate = 0.0
    return {
        "parser_attempts": attempts,
        "parser_recovered_calls": recovered_calls,
        "parser_rejected_calls": rejected_calls,
        "parser_malformed_attempts": malformed_attempts,
        "parser_malformed_rate": malformed_rate,
        "parser_telemetry_complete": telemetry_complete,
        "parser_unmatched_legacy_errors": unmatched_legacy_errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = summarize(args.log_path)
    if args.json:
        print(json.dumps(summary, sort_keys=True))
        return
    malformed_rate = summary["parser_malformed_rate"]
    rendered_rate = "unavailable" if malformed_rate is None else f"{malformed_rate:.6f}"
    print(
        f"parser_attempts={summary['parser_attempts']} "
        f"parser_recovered_calls={summary['parser_recovered_calls']} "
        f"parser_rejected_calls={summary['parser_rejected_calls']} "
        f"parser_malformed_attempts={summary['parser_malformed_attempts']} "
        f"parser_malformed_rate={rendered_rate} "
        f"parser_telemetry_complete={str(summary['parser_telemetry_complete']).lower()} "
        f"parser_unmatched_legacy_errors={summary['parser_unmatched_legacy_errors']}"
    )


if __name__ == "__main__":
    main()
