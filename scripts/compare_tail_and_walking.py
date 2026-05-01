#!/usr/bin/env python3
"""
Compare nominal vs robust right-tail costs and walking-violation rates.

Matches runs on:
  - k
  - in_vehicle_time_weight

For robust runs, prefers `demand_quantile` and falls back to the older
`Q_cap_quantile == q_high_quantile` convention.

Reported metrics:
  - realized order-cost quantiles: p90, p95, p99 for April and May
  - realized walking-violation rate: April and May
  - theoretical walking-violation rate over all possible ODs:
      * scenario-specific (period_1 ... period_4)
      * average across scenarios
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from compare_weighted_total_cost import (
    compute_order_costs,
    compute_best_feasible_pair_cost,
    get_full_station_path,
    get_max_walking_distance,
    get_realized_walking_violation_rate,
    iter_metric_files,
    load_metrics,
    load_period_map,
    load_routing_costs,
    load_station_coords,
)


def get_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def collect_runs(exp_dir: Path) -> tuple[dict[tuple[int, float], list[dict[str, Any]]], dict[tuple[int, float], list[dict[str, Any]]]]:
    nominal_by_key: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    robust_by_key: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)

    for metrics_path in iter_metric_files(exp_dir):
        metrics = load_metrics(metrics_path)
        model_type = metrics.get("model_type")
        k = metrics.get("k")
        lam = metrics.get("in_vehicle_time_weight")
        if model_type is None or k is None or lam is None:
            continue

        record = {
            "job_id": metrics.get("job_id"),
            "metrics_path": str(metrics_path),
            "run_dir": metrics_path.parent,
            "timestamp": metrics.get("timestamp"),
            "k": k,
            "in_vehicle_time_weight": lam,
            "demand_quantile": metrics.get("demand_quantile"),
            "Q_cap_quantile": metrics.get("Q_cap_quantile"),
            "q_high_quantile": metrics.get("q_high_quantile"),
        }
        key = (k, lam)
        if model_type == "NominalModel":
            nominal_by_key[key].append(record)
        elif model_type == "RobustTotalDemandCapModel":
            quantile = record["demand_quantile"]
            if quantile is None:
                q_cap = record["Q_cap_quantile"]
                q_high = record["q_high_quantile"]
                if q_cap is None or q_high is None or q_cap != q_high:
                    continue
                quantile = q_cap
            record["comparison_quantile"] = quantile
            robust_by_key[key].append(record)
    return nominal_by_key, robust_by_key


def theoretical_walking_violation_rates(run_dir: Path) -> dict[str, float | None]:
    metrics = load_metrics(run_dir / "metrics.json")
    max_walking_distance = get_max_walking_distance(run_dir)
    lambda_val = float(metrics.get("in_vehicle_time_weight", 0.0))

    active_schedule_path = run_dir / "backtest" / "active_station_schedule.csv"
    station_path = get_full_station_path(run_dir)
    segment_path = run_dir / "backtest" / "simulation_inputs" / "segment.csv"
    if not active_schedule_path.exists() or not station_path.exists() or not segment_path.exists():
        return {}

    coords = load_station_coords(station_path)
    station_ids = sorted(coords)
    routing_costs = load_routing_costs(segment_path, set(coords))
    period_map = load_period_map(active_schedule_path)

    rates: dict[str, float | None] = {}
    period_rates: list[float] = []

    for idx, (period_start_hour, active_ids) in enumerate(period_map, start=1):
        label = f"period_{idx}"
        if not active_ids:
            rates[label] = None
            continue
        total_ods = 0
        violating_ods = 0
        for origin_id in station_ids:
            for dest_id in station_ids:
                if origin_id == dest_id:
                    continue
                total_ods += 1
                feasible_pair = compute_best_feasible_pair_cost(
                    origin_id,
                    dest_id,
                    active_ids,
                    coords,
                    routing_costs,
                    lambda_val,
                    max_walking_distance,
                )
                if feasible_pair is None:
                    violating_ods += 1

        rate = violating_ods / total_ods if total_ods > 0 else None
        rates[label] = rate
        if rate is not None:
            period_rates.append(rate)

    rates["scenario_avg"] = sum(period_rates) / len(period_rates) if period_rates else None
    return rates


def build_run_summary(record: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(record["run_dir"])
    lambda_val = float(record["in_vehicle_time_weight"])
    summary = dict(record)

    for month in ("April", "May"):
        order_costs = compute_order_costs(run_dir, month, lambda_val)
        summary[f"{month.lower()}_p90"] = get_quantile(order_costs, 0.90)
        summary[f"{month.lower()}_p95"] = get_quantile(order_costs, 0.95)
        summary[f"{month.lower()}_p99"] = get_quantile(order_costs, 0.99)
        summary[f"{month.lower()}_walking_violation_rate"] = get_realized_walking_violation_rate(run_dir, month)

    theo = theoretical_walking_violation_rates(run_dir)
    for label, value in theo.items():
        summary[f"theoretical_walking_violation_rate_{label}"] = value
    return summary


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def build_rows(exp_dir: Path) -> list[dict[str, Any]]:
    nominal_by_key, robust_by_key = collect_runs(exp_dir)
    rows: list[dict[str, Any]] = []

    for key in sorted(robust_by_key):
        nominal_runs = sorted(
            nominal_by_key.get(key, []),
            key=lambda row: (row["job_id"] if row["job_id"] is not None else -1, row["timestamp"] or ""),
        )
        robust_runs = sorted(
            robust_by_key[key],
            key=lambda row: (row["comparison_quantile"], row["job_id"] if row["job_id"] is not None else -1, row["timestamp"] or ""),
        )
        if not nominal_runs:
            continue

        nominal_summaries = [build_run_summary(row) for row in nominal_runs]
        robust_summaries = [build_run_summary(row) for row in robust_runs]

        for nominal in nominal_summaries:
            for robust in robust_summaries:
                row = {
                    "k": key[0],
                    "in_vehicle_time_weight": key[1],
                    "quantile": robust["comparison_quantile"],
                    "nominal_job_id": nominal["job_id"],
                    "robust_job_id": robust["job_id"],
                    "nominal_metrics_path": nominal["metrics_path"],
                    "robust_metrics_path": robust["metrics_path"],
                }

                for month in ("april", "may"):
                    for q in ("p90", "p95", "p99"):
                        n_key = f"{month}_{q}"
                        r_key = f"{month}_{q}"
                        row[f"nominal_{month}_{q}"] = nominal[n_key]
                        row[f"robust_{month}_{q}"] = robust[r_key]
                        row[f"robust_over_nominal_{month}_{q}"] = ratio(robust[r_key], nominal[n_key])
                        row[f"robust_minus_nominal_{month}_{q}"] = diff(robust[r_key], nominal[n_key])

                    n_v = nominal[f"{month}_walking_violation_rate"]
                    r_v = robust[f"{month}_walking_violation_rate"]
                    row[f"nominal_{month}_walking_violation_rate"] = n_v
                    row[f"robust_{month}_walking_violation_rate"] = r_v
                    row[f"robust_over_nominal_{month}_walking_violation_rate"] = ratio(r_v, n_v)
                    row[f"robust_minus_nominal_{month}_walking_violation_rate"] = diff(r_v, n_v)

                for label in ("period_1", "period_2", "period_3", "period_4", "scenario_avg"):
                    n_key = f"theoretical_walking_violation_rate_{label}"
                    r_key = f"theoretical_walking_violation_rate_{label}"
                    n_v = nominal.get(n_key)
                    r_v = robust.get(r_key)
                    row[f"nominal_theoretical_walking_violation_rate_{label}"] = n_v
                    row[f"robust_theoretical_walking_violation_rate_{label}"] = r_v
                    row[f"robust_over_nominal_theoretical_walking_violation_rate_{label}"] = ratio(r_v, n_v)
                    row[f"robust_minus_nominal_theoretical_walking_violation_rate_{label}"] = diff(r_v, n_v)

                rows.append(row)

    return rows


FIELDNAMES = [
    "k",
    "in_vehicle_time_weight",
    "quantile",
    "nominal_job_id",
    "robust_job_id",
    "nominal_april_p90",
    "robust_april_p90",
    "robust_over_nominal_april_p90",
    "nominal_april_p95",
    "robust_april_p95",
    "robust_over_nominal_april_p95",
    "nominal_april_p99",
    "robust_april_p99",
    "robust_over_nominal_april_p99",
    "nominal_may_p90",
    "robust_may_p90",
    "robust_over_nominal_may_p90",
    "nominal_may_p95",
    "robust_may_p95",
    "robust_over_nominal_may_p95",
    "nominal_may_p99",
    "robust_may_p99",
    "robust_over_nominal_may_p99",
    "nominal_april_walking_violation_rate",
    "robust_april_walking_violation_rate",
    "robust_over_nominal_april_walking_violation_rate",
    "nominal_may_walking_violation_rate",
    "robust_may_walking_violation_rate",
    "robust_over_nominal_may_walking_violation_rate",
    "nominal_theoretical_walking_violation_rate_period_1",
    "robust_theoretical_walking_violation_rate_period_1",
    "robust_over_nominal_theoretical_walking_violation_rate_period_1",
    "nominal_theoretical_walking_violation_rate_period_2",
    "robust_theoretical_walking_violation_rate_period_2",
    "robust_over_nominal_theoretical_walking_violation_rate_period_2",
    "nominal_theoretical_walking_violation_rate_period_3",
    "robust_theoretical_walking_violation_rate_period_3",
    "robust_over_nominal_theoretical_walking_violation_rate_period_3",
    "nominal_theoretical_walking_violation_rate_period_4",
    "robust_theoretical_walking_violation_rate_period_4",
    "robust_over_nominal_theoretical_walking_violation_rate_period_4",
    "nominal_theoretical_walking_violation_rate_scenario_avg",
    "robust_theoretical_walking_violation_rate_scenario_avg",
    "robust_over_nominal_theoretical_walking_violation_rate_scenario_avg",
    "nominal_metrics_path",
    "robust_metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    parser.add_argument("--format", choices=["tsv", "csv"], default="tsv")
    return parser.parse_args()


def print_rows(rows: list[dict[str, Any]], fmt: str) -> None:
    if fmt == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in FIELDNAMES} for row in rows])
        return

    print("\t".join(FIELDNAMES))
    for row in rows:
        print("\t".join("" if row.get(field) is None else str(row.get(field)) for field in FIELDNAMES))


def main() -> int:
    args = parse_args()
    if not args.exp_dir.exists():
        print(f"experiment directory not found: {args.exp_dir}", file=sys.stderr)
        return 1
    rows = build_rows(args.exp_dir)
    print_rows(rows, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
