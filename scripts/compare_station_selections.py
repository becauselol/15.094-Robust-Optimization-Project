#!/usr/bin/env python3
"""
Compare nominal vs robust station selections and scenario activations.

The script scans runs/* and matches nominal vs robust runs on:
  - k
  - in_vehicle_time_weight

For robust runs, only comparisons with Q_cap_quantile == q_high_quantile are kept.

For each comparable pair, it computes:
  - whether y_j selected stations are exactly the same
  - Jaccard similarity on the y_j selected-station sets
  - whether z_js station-scenario activations are exactly the same
  - Jaccard similarity on the z_js activation sets
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


def iter_metric_files(exp_dir: Path) -> list[Path]:
    return sorted(exp_dir.glob("runs/*/metrics.json"))


def parse_station_selection(run_dir: Path) -> set[int]:
    path = run_dir / "variable_exports" / "station_selection.csv"
    selected: set[int] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("selected", "").strip()
            if value in {"1", "1.0"}:
                selected.add(int(row["station_id"]))
    return selected


def parse_scenario_activation(run_dir: Path) -> set[tuple[int, str]]:
    path = run_dir / "variable_exports" / "scenario_activation.csv"
    active: set[tuple[int, str]] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("value", "").strip()
            if value in {"1", "1.0"}:
                active.add((int(row["station_id"]), row["scenario_label"]))
    return active


def jaccard(a: set[Any], b: set[Any]) -> float | None:
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


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

        run_dir = metrics_path.parent
        record = {
            "job_id": metrics.get("job_id"),
            "metrics_path": str(metrics_path),
            "run_dir": str(run_dir),
            "timestamp": metrics.get("timestamp"),
            "k": k,
            "in_vehicle_time_weight": in_vehicle_time_weight,
            "Q_cap_quantile": metrics.get("Q_cap_quantile"),
            "q_high_quantile": metrics.get("q_high_quantile"),
            "y_j": parse_station_selection(run_dir),
            "z_js": parse_scenario_activation(run_dir),
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
        nominal_runs = sorted(
            nominal_by_key.get(key, []),
            key=lambda row: (row["job_id"] if row["job_id"] is not None else -1, row["timestamp"] or ""),
        )
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

        for nominal in nominal_runs:
            for robust in robust_runs:
                y_nom = nominal["y_j"]
                y_rob = robust["y_j"]
                z_nom = nominal["z_js"]
                z_rob = robust["z_js"]
                rows.append(
                    {
                        "k": key[0],
                        "in_vehicle_time_weight": key[1],
                        "quantile": robust["Q_cap_quantile"],
                        "nominal_job_id": nominal["job_id"],
                        "robust_job_id": robust["job_id"],
                        "y_j_exact_same": y_nom == y_rob,
                        "y_j_jaccard": jaccard(y_nom, y_rob),
                        "y_j_intersection_size": len(y_nom & y_rob),
                        "y_j_union_size": len(y_nom | y_rob),
                        "z_js_exact_same": z_nom == z_rob,
                        "z_js_jaccard": jaccard(z_nom, z_rob),
                        "z_js_intersection_size": len(z_nom & z_rob),
                        "z_js_union_size": len(z_nom | z_rob),
                        "nominal_run_dir": nominal["run_dir"],
                        "robust_run_dir": robust["run_dir"],
                    }
                )

    return rows


def print_tsv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "k",
        "in_vehicle_time_weight",
        "quantile",
        "nominal_job_id",
        "robust_job_id",
        "y_j_exact_same",
        "y_j_jaccard",
        "y_j_intersection_size",
        "y_j_union_size",
        "z_js_exact_same",
        "z_js_jaccard",
        "z_js_intersection_size",
        "z_js_union_size",
        "nominal_run_dir",
        "robust_run_dir",
    ]
    print("\t".join(fieldnames))
    for row in rows:
        print("\t".join("" if row[field] is None else str(row[field]) for field in fieldnames))


def print_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "k",
        "in_vehicle_time_weight",
        "quantile",
        "nominal_job_id",
        "robust_job_id",
        "y_j_exact_same",
        "y_j_jaccard",
        "y_j_intersection_size",
        "y_j_union_size",
        "z_js_exact_same",
        "z_js_jaccard",
        "z_js_intersection_size",
        "z_js_union_size",
        "nominal_run_dir",
        "robust_run_dir",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    parser.add_argument("--format", choices=["tsv", "csv"], default="tsv")
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
