#!/usr/bin/env python3
"""
Compare weighted_total_cost between nominal and robust runs without aggregation.

The script scans runs/**/metrics.json inside an experiment directory and reports
one row per comparable nominal/robust pair, matched on:
  - k
  - in_vehicle_time_weight

For robust runs, only rows with Q_cap_quantile == q_high_quantile are kept.
April and May are reported separately:
  - April = in_sample_direct.weighted_total_cost
  - May   = direct_backtest.weighted_total_cost

Usage:
    python scripts/compare_weighted_total_cost.py experiments/2026-05-01_subsample_40
    python scripts/compare_weighted_total_cost.py experiments/2026-05-01_subsample_40 --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    return json.loads(metrics_path.read_text())


def get_weighted_total_cost(metrics: dict[str, Any], month: str) -> float | None:
    if month == "April":
        block = metrics.get("in_sample_direct", {})
    elif month == "May":
        block = metrics.get("direct_backtest", {})
    else:
        raise ValueError(f"unexpected month: {month}")
    return block.get("weighted_total_cost")


def iter_metric_files(exp_dir: Path) -> list[Path]:
    return sorted(exp_dir.glob("runs/*/metrics.json"))


def collect_runs(exp_dir: Path) -> tuple[dict[tuple[int, float], list[dict[str, Any]]], dict[tuple[int, float], list[dict[str, Any]]]]:
    nominal_by_key: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    robust_by_key: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)

    for metrics_path in iter_metric_files(exp_dir):
        metrics = load_metrics(metrics_path)
        model_type = metrics.get("model_type")
        k = metrics.get("k")
        in_vehicle_time_weight = metrics.get("in_vehicle_time_weight")
        if k is None or in_vehicle_time_weight is None or model_type is None:
            continue

        record = {
            "job_id": metrics.get("job_id"),
            "metrics_path": str(metrics_path),
            "run_dir": str(metrics_path.parent),
            "timestamp": metrics.get("timestamp"),
            "k": k,
            "in_vehicle_time_weight": in_vehicle_time_weight,
            "april_weighted_total_cost": get_weighted_total_cost(metrics, "April"),
            "may_weighted_total_cost": get_weighted_total_cost(metrics, "May"),
            "Q_cap_quantile": metrics.get("Q_cap_quantile"),
            "q_high_quantile": metrics.get("q_high_quantile"),
        }
        key = (k, in_vehicle_time_weight)

        if model_type == "NominalModel":
            nominal_by_key[key].append(record)
        elif model_type == "RobustTotalDemandCapModel":
            q_cap = record["Q_cap_quantile"]
            q_high = record["q_high_quantile"]
            if q_cap is None or q_high is None or q_cap != q_high:
                continue
            robust_by_key[key].append(record)

    return nominal_by_key, robust_by_key


def build_rows(exp_dir: Path) -> list[dict[str, Any]]:
    nominal_by_key, robust_by_key = collect_runs(exp_dir)
    rows: list[dict[str, Any]] = []

    for key in sorted(robust_by_key):
        nominal_runs = nominal_by_key.get(key, [])
        robust_runs = sorted(
            robust_by_key[key],
            key=lambda row: (
                row["Q_cap_quantile"],
                row["job_id"] if row["job_id"] is not None else -1,
                row["timestamp"] or "",
            ),
        )
        if not nominal_runs:
            continue

        nominal_runs = sorted(
            nominal_runs,
            key=lambda row: (
                row["job_id"] if row["job_id"] is not None else -1,
                row["timestamp"] or "",
            ),
        )

        for nominal in nominal_runs:
            for robust in robust_runs:
                for month, nominal_cost_key, robust_cost_key in [
                    ("April", "april_weighted_total_cost", "april_weighted_total_cost"),
                    ("May", "may_weighted_total_cost", "may_weighted_total_cost"),
                ]:
                    nominal_cost = nominal[nominal_cost_key]
                    robust_cost = robust[robust_cost_key]
                    if nominal_cost is None or robust_cost is None:
                        continue
                    rows.append(
                        {
                            "k": key[0],
                            "in_vehicle_time_weight": key[1],
                            "month": month,
                            "quantile": robust["Q_cap_quantile"],
                            "nominal_job_id": nominal["job_id"],
                            "robust_job_id": robust["job_id"],
                            "nominal_weighted_total_cost": nominal_cost,
                            "robust_weighted_total_cost": robust_cost,
                            "robust_minus_nominal": robust_cost - nominal_cost,
                            "robust_over_nominal": (
                                robust_cost / nominal_cost if nominal_cost != 0 else None
                            ),
                            "nominal_metrics_path": nominal["metrics_path"],
                            "robust_metrics_path": robust["metrics_path"],
                        }
                    )

    return rows


def print_tsv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "k",
        "in_vehicle_time_weight",
        "month",
        "quantile",
        "nominal_job_id",
        "robust_job_id",
        "nominal_weighted_total_cost",
        "robust_weighted_total_cost",
        "robust_minus_nominal",
        "robust_over_nominal",
        "nominal_metrics_path",
        "robust_metrics_path",
    ]
    print("\t".join(fieldnames))
    for row in rows:
        print(
            "\t".join(
                "" if row[field] is None else str(row[field])
                for field in fieldnames
            )
        )


def print_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "k",
        "in_vehicle_time_weight",
        "month",
        "quantile",
        "nominal_job_id",
        "robust_job_id",
        "nominal_weighted_total_cost",
        "robust_weighted_total_cost",
        "robust_minus_nominal",
        "robust_over_nominal",
        "nominal_metrics_path",
        "robust_metrics_path",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    parser.add_argument(
        "--format",
        choices=["tsv", "csv"],
        default="tsv",
        help="Output format",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir
    if not exp_dir.exists():
        print(f"experiment directory not found: {exp_dir}", file=sys.stderr)
        return 1

    rows = build_rows(exp_dir)
    if args.format == "csv":
        print_csv(rows)
    else:
        print_tsv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
