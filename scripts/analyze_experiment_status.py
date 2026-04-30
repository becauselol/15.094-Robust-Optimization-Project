#!/usr/bin/env python3
"""
Summarize experiment job status and parameter patterns from an experiment folder.

Usage:
    python scripts/analyze_experiment_status.py experiments/2026-05-01_subsample_40
    python scripts/analyze_experiment_status.py experiments/2026-05-01_subsample_40 --show-jobs
    python scripts/analyze_experiment_status.py experiments/2026-05-01_subsample_40 \
        --group-by model.type parameters.in_vehicle_time_weight parameters.k
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RUN_JOB_RE = re.compile(r"_job(\d+)_")
DEFAULT_GROUP_BY = ["model.type", "parameters.in_vehicle_time_weight"]


def get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def load_job_config(config_dir: Path, job_id: int) -> dict[str, Any]:
    job_path = config_dir / f"job_{job_id}.toml"
    return tomllib.loads(job_path.read_text())


def collect_run_attempts(runs_dir: Path) -> dict[int, list[str]]:
    attempts: dict[int, list[str]] = defaultdict(list)
    if not runs_dir.exists():
        return attempts

    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_JOB_RE.search(child.name)
        if match:
            attempts[int(match.group(1))].append(child.name)
    return attempts


def build_rows(exp_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, list[str]]]:
    state_path = exp_dir / "monitor_state.json"
    config_dir = exp_dir / "config"
    runs_dir = exp_dir / "runs"

    monitor = json.loads(state_path.read_text())
    jobs = monitor["jobs"]
    run_attempts = collect_run_attempts(runs_dir)

    rows: list[dict[str, Any]] = []
    for job_id_text, state in sorted(jobs.items(), key=lambda kv: int(kv[0])):
        job_id = int(job_id_text)
        config = load_job_config(config_dir, job_id)
        row = {
            "job_id": job_id,
            "status": state["status"],
            "retry_count": state.get("retry_count", 0),
            "array_job_id": state.get("array_job_id"),
            "attempt_count": len(run_attempts.get(job_id, [])),
            "run_dirs": run_attempts.get(job_id, []),
            "integrality_gap": state.get("integrality_gap"),
            "no_restart_reason": state.get("no_restart_reason"),
            "config": config,
        }
        rows.append(row)
    return rows, monitor, run_attempts


def print_overview(rows: list[dict[str, Any]], run_attempts: dict[int, list[str]], metadata: dict[str, Any]) -> None:
    status_counts = Counter(row["status"] for row in rows)
    retried = [job_id for job_id, attempts in run_attempts.items() if len(attempts) > 1]

    print("Overview")
    print(f"  total jobs in monitor state: {len(rows)}")
    if "total_jobs" in metadata:
        print(f"  total jobs in setup metadata: {metadata['total_jobs']}")
    print(f"  unique jobs with run dirs: {len(run_attempts)}")
    print(f"  total run dirs: {sum(len(v) for v in run_attempts.values())}")
    print(f"  jobs with multiple attempts: {len(retried)}")
    print()

    print("Status counts")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print()


def print_status_job_lists(rows: list[dict[str, Any]]) -> None:
    by_status: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_status[row["status"]].append(row["job_id"])

    print("Jobs by status")
    for status in sorted(by_status):
        job_list = ", ".join(str(job_id) for job_id in by_status[status])
        print(f"  {status}: {job_list}")
    print()


def print_group_summary(rows: list[dict[str, Any]], group_by: list[str]) -> None:
    grouped: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = tuple(format_value(get_nested(row["config"], field)) for field in group_by)
        grouped[key][row["status"]] += 1

    print("Grouped status summary")
    print(f"  grouped by: {', '.join(group_by)}")
    for key in sorted(grouped):
        counts = grouped[key]
        label = ", ".join(f"{field}={value}" for field, value in zip(group_by, key))
        count_text = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
        print(f"  {label}: {count_text}")
    print()


def print_completion_patterns(rows: list[dict[str, Any]], fields: list[str]) -> None:
    summaries: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        for field in fields:
            value = format_value(get_nested(row["config"], field))
            summaries[(field, value)][row["status"]] += 1

    print("Completion patterns")
    print("  value groups sorted by completion rate")
    sortable = []
    for (field, value), counts in summaries.items():
        total = sum(counts.values())
        completed = counts.get("completed", 0)
        completion_rate = completed / total if total else 0.0
        sortable.append((completion_rate, field, value, counts, total))

    for completion_rate, field, value, counts, total in sorted(
        sortable, key=lambda item: (-item[0], item[1], item[2])
    ):
        count_text = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
        print(f"  {field}={value}: completion_rate={completion_rate:.2%} ({count_text}; total={total})")
    print()


def print_timeout_details(rows: list[dict[str, Any]]) -> None:
    timeout_rows = [row for row in rows if row["status"] == "no_restart"]
    if not timeout_rows:
        return

    print("No-restart details")
    for row in timeout_rows:
        gap = row["integrality_gap"] or "-"
        reason = row["no_restart_reason"] or "-"
        print(f"  job {row['job_id']}: gap={gap}, reason={reason}")
    print()


def print_attempt_details(run_attempts: dict[int, list[str]]) -> None:
    retried = {job_id: attempts for job_id, attempts in run_attempts.items() if len(attempts) > 1}
    if not retried:
        return

    print("Retried jobs")
    for job_id in sorted(retried):
        attempts = ", ".join(retried[job_id])
        print(f"  job {job_id}: {attempts}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=DEFAULT_GROUP_BY,
        help="Dotted config fields used for grouped status summary",
    )
    parser.add_argument(
        "--pattern-fields",
        nargs="+",
        default=["model.type", "parameters.in_vehicle_time_weight", "parameters.k"],
        help="Dotted config fields ranked by completion rate",
    )
    parser.add_argument(
        "--show-jobs",
        action="store_true",
        help="Print job ids grouped by status",
    )
    parser.add_argument(
        "--show-attempts",
        action="store_true",
        help="Print run directory names for jobs with multiple attempts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir

    if not exp_dir.exists():
        print(f"experiment directory not found: {exp_dir}", file=sys.stderr)
        return 1

    metadata = {}
    setup_metadata_path = exp_dir / "setup_metadata.json"
    if setup_metadata_path.exists():
        metadata = json.loads(setup_metadata_path.read_text())

    rows, _, run_attempts = build_rows(exp_dir)

    print_overview(rows, run_attempts, metadata)
    if args.show_jobs:
        print_status_job_lists(rows)
    print_group_summary(rows, args.group_by)
    print_completion_patterns(rows, args.pattern_fields)
    print_timeout_details(rows)
    if args.show_attempts:
        print_attempt_details(run_attempts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
