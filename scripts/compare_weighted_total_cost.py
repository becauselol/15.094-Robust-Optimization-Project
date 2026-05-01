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
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import pstdev
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


def get_mean_weighted_cost_per_order(metrics: dict[str, Any], month: str) -> float | None:
    if month == "April":
        block = metrics.get("in_sample_direct", {})
    elif month == "May":
        block = metrics.get("direct_backtest", {})
    else:
        raise ValueError(f"unexpected month: {month}")
    return block.get("mean_weighted_cost_per_order")


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.asin(math.sqrt(a))


def load_station_coords(station_path: Path) -> dict[int, tuple[float, float]]:
    coords: dict[int, tuple[float, float]] = {}
    with station_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_id = int(row.get("id", row.get("station_id")))
            lon = float(row.get("lon", row.get("station_lon")))
            lat = float(row.get("lat", row.get("station_lat")))
            coords[station_id] = (lat, lon)
    return coords


def load_routing_costs(segment_path: Path, station_ids: set[int]) -> dict[tuple[int, int], float]:
    ordered_ids = sorted(station_ids)
    inf = float("inf")
    dists: dict[tuple[int, int], float] = {
        (i, j): (0.0 if i == j else inf) for i in ordered_ids for j in ordered_ids
    }
    with segment_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = int(row["from_station"])
            j = int(row["to_station"])
            if i in station_ids and j in station_ids:
                dists[(i, j)] = min(dists[(i, j)], float(row["seg_time"]))

    for k in ordered_ids:
        for i in ordered_ids:
            dik = dists[(i, k)]
            if not math.isfinite(dik):
                continue
            for j in ordered_ids:
                alt = dik + dists[(k, j)]
                if alt < dists[(i, j)]:
                    dists[(i, j)] = alt
    return dists


def compute_order_costs(run_dir: Path, month: str, lambda_val: float) -> list[float]:
    if month == "April":
        transform_dir = run_dir / "backtest" / "transform_in_sample"
    elif month == "May":
        transform_dir = run_dir / "backtest" / "transform"
    else:
        raise ValueError(f"unexpected month: {month}")

    manifest_path = transform_dir / "daily_manifest.csv"
    station_path = run_dir / "backtest" / "simulation_inputs" / "station.csv"
    segment_path = run_dir / "backtest" / "simulation_inputs" / "segment.csv"
    if not manifest_path.exists() or not station_path.exists() or not segment_path.exists():
        return []

    coords = load_station_coords(station_path)
    routing_costs = load_routing_costs(segment_path, set(coords))
    order_costs: list[float] = []

    with manifest_path.open(newline="") as f:
        manifest_reader = csv.DictReader(f)
        for manifest_row in manifest_reader:
            orders_path = Path(manifest_row["orders_file"])
            if not orders_path.exists():
                orders_path = transform_dir / "daily_orders" / orders_path.name
            if not orders_path.exists():
                continue
            with orders_path.open(newline="") as orders_f:
                orders_reader = csv.DictReader(orders_f)
                for row in orders_reader:
                    pickup_id = int(float(row.get("assigned_pickup_id", "0") or 0))
                    dropoff_id = int(float(row.get("assigned_dropoff_id", "0") or 0))
                    if pickup_id == 0 or dropoff_id == 0:
                        continue
                    origin_id = int(row["origin_station_id"])
                    dest_id = int(row["destination_station_id"])
                    if (
                        origin_id not in coords
                        or dest_id not in coords
                        or pickup_id not in coords
                        or dropoff_id not in coords
                    ):
                        continue
                    route_cost = routing_costs.get((pickup_id, dropoff_id), float("inf"))
                    if not math.isfinite(route_cost):
                        continue
                    origin_lat, origin_lon = coords[origin_id]
                    pickup_lat, pickup_lon = coords[pickup_id]
                    dropoff_lat, dropoff_lon = coords[dropoff_id]
                    dest_lat, dest_lon = coords[dest_id]
                    walk_cost = (
                        haversine_meters(origin_lat, origin_lon, pickup_lat, pickup_lon) / 1.4
                        + haversine_meters(dropoff_lat, dropoff_lon, dest_lat, dest_lon) / 1.4
                    )
                    order_costs.append(walk_cost + lambda_val * route_cost)
    return order_costs


def compute_daily_weighted_total_costs(run_dir: Path, month: str, lambda_val: float) -> list[float]:
    if month == "April":
        transform_dir = run_dir / "backtest" / "transform_in_sample"
    elif month == "May":
        transform_dir = run_dir / "backtest" / "transform"
    else:
        raise ValueError(f"unexpected month: {month}")

    manifest_path = transform_dir / "daily_manifest.csv"
    station_path = run_dir / "backtest" / "simulation_inputs" / "station.csv"
    segment_path = run_dir / "backtest" / "simulation_inputs" / "segment.csv"
    if not manifest_path.exists() or not station_path.exists() or not segment_path.exists():
        return []

    coords = load_station_coords(station_path)
    routing_costs = load_routing_costs(segment_path, set(coords))
    daily_costs: list[float] = []

    with manifest_path.open(newline="") as f:
        manifest_reader = csv.DictReader(f)
        for manifest_row in manifest_reader:
            orders_path = Path(manifest_row["orders_file"])
            if not orders_path.exists():
                orders_path = transform_dir / "daily_orders" / orders_path.name
            if not orders_path.exists():
                continue
            total_cost = 0.0
            with orders_path.open(newline="") as orders_f:
                orders_reader = csv.DictReader(orders_f)
                for row in orders_reader:
                    pickup_id = int(float(row.get("assigned_pickup_id", "0") or 0))
                    dropoff_id = int(float(row.get("assigned_dropoff_id", "0") or 0))
                    if pickup_id == 0 or dropoff_id == 0:
                        continue
                    origin_id = int(row["origin_station_id"])
                    dest_id = int(row["destination_station_id"])
                    if (
                        origin_id not in coords
                        or dest_id not in coords
                        or pickup_id not in coords
                        or dropoff_id not in coords
                    ):
                        continue
                    route_cost = routing_costs.get((pickup_id, dropoff_id), float("inf"))
                    if not math.isfinite(route_cost):
                        continue
                    origin_lat, origin_lon = coords[origin_id]
                    pickup_lat, pickup_lon = coords[pickup_id]
                    dropoff_lat, dropoff_lon = coords[dropoff_id]
                    dest_lat, dest_lon = coords[dest_id]
                    walk_cost = (
                        haversine_meters(origin_lat, origin_lon, pickup_lat, pickup_lon) / 1.4
                        + haversine_meters(dropoff_lat, dropoff_lon, dest_lat, dest_lon) / 1.4
                    )
                    total_cost += walk_cost + lambda_val * route_cost
            daily_costs.append(total_cost)
    return daily_costs


def get_daily_cost_std(run_dir: Path, month: str, lambda_val: float) -> float | None:
    daily_costs = compute_daily_weighted_total_costs(run_dir, month, lambda_val)
    if not daily_costs:
        return None
    return pstdev(daily_costs)


def get_order_cost_std(run_dir: Path, month: str, lambda_val: float) -> float | None:
    order_costs = compute_order_costs(run_dir, month, lambda_val)
    if not order_costs:
        return None
    return pstdev(order_costs)


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
            "april_mean_cost_per_order": get_mean_weighted_cost_per_order(metrics, "April"),
            "may_mean_cost_per_order": get_mean_weighted_cost_per_order(metrics, "May"),
            "april_order_cost_std": get_order_cost_std(metrics_path.parent, "April", float(in_vehicle_time_weight)),
            "may_order_cost_std": get_order_cost_std(metrics_path.parent, "May", float(in_vehicle_time_weight)),
            "april_daily_cost_std": get_daily_cost_std(metrics_path.parent, "April", float(in_vehicle_time_weight)),
            "may_daily_cost_std": get_daily_cost_std(metrics_path.parent, "May", float(in_vehicle_time_weight)),
            "demand_quantile": metrics.get("demand_quantile"),
            "Q_cap_quantile": metrics.get("Q_cap_quantile"),
            "q_high_quantile": metrics.get("q_high_quantile"),
        }
        key = (k, in_vehicle_time_weight)

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
            if quantile is None:
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
                row["comparison_quantile"],
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
                for month, nominal_cost_key, robust_cost_key, nominal_mean_key, robust_mean_key, nominal_order_std_key, robust_order_std_key, nominal_daily_std_key, robust_daily_std_key in [
                    ("April", "april_weighted_total_cost", "april_weighted_total_cost", "april_mean_cost_per_order", "april_mean_cost_per_order", "april_order_cost_std", "april_order_cost_std", "april_daily_cost_std", "april_daily_cost_std"),
                    ("May", "may_weighted_total_cost", "may_weighted_total_cost", "may_mean_cost_per_order", "may_mean_cost_per_order", "may_order_cost_std", "may_order_cost_std", "may_daily_cost_std", "may_daily_cost_std"),
                ]:
                    nominal_cost = nominal[nominal_cost_key]
                    robust_cost = robust[robust_cost_key]
                    nominal_mean = nominal[nominal_mean_key]
                    robust_mean = robust[robust_mean_key]
                    nominal_order_std = nominal[nominal_order_std_key]
                    robust_order_std = robust[robust_order_std_key]
                    nominal_daily_std = nominal[nominal_daily_std_key]
                    robust_daily_std = robust[robust_daily_std_key]
                    if nominal_cost is None or robust_cost is None:
                        continue
                    rows.append(
                        {
                            "k": key[0],
                            "in_vehicle_time_weight": key[1],
                            "month": month,
                            "quantile": robust["comparison_quantile"],
                            "nominal_job_id": nominal["job_id"],
                            "robust_job_id": robust["job_id"],
                            "nominal_weighted_total_cost": nominal_cost,
                            "robust_weighted_total_cost": robust_cost,
                            "robust_minus_nominal": robust_cost - nominal_cost,
                            "robust_over_nominal": (
                                robust_cost / nominal_cost if nominal_cost != 0 else None
                            ),
                            "nominal_mean_cost_per_order": nominal_mean,
                            "robust_mean_cost_per_order": robust_mean,
                            "robust_minus_nominal_mean_cost_per_order": (
                                robust_mean - nominal_mean
                                if nominal_mean is not None and robust_mean is not None
                                else None
                            ),
                            "robust_over_nominal_mean_cost_per_order": (
                                robust_mean / nominal_mean
                                if nominal_mean not in (None, 0) and robust_mean is not None
                                else None
                            ),
                            "nominal_order_cost_std": nominal_order_std,
                            "robust_order_cost_std": robust_order_std,
                            "robust_minus_nominal_order_cost_std": (
                                robust_order_std - nominal_order_std
                                if nominal_order_std is not None and robust_order_std is not None
                                else None
                            ),
                            "robust_over_nominal_order_cost_std": (
                                robust_order_std / nominal_order_std
                                if nominal_order_std not in (None, 0) and robust_order_std is not None
                                else None
                            ),
                            "nominal_daily_cost_std": nominal_daily_std,
                            "robust_daily_cost_std": robust_daily_std,
                            "robust_minus_nominal_daily_std": (
                                robust_daily_std - nominal_daily_std
                                if nominal_daily_std is not None and robust_daily_std is not None
                                else None
                            ),
                            "robust_over_nominal_daily_std": (
                                robust_daily_std / nominal_daily_std
                                if nominal_daily_std not in (None, 0) and robust_daily_std is not None
                                else None
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
        "nominal_mean_cost_per_order",
        "robust_mean_cost_per_order",
        "robust_minus_nominal_mean_cost_per_order",
        "robust_over_nominal_mean_cost_per_order",
        "nominal_order_cost_std",
        "robust_order_cost_std",
        "robust_minus_nominal_order_cost_std",
        "robust_over_nominal_order_cost_std",
        "nominal_daily_cost_std",
        "robust_daily_cost_std",
        "robust_minus_nominal_daily_std",
        "robust_over_nominal_daily_std",
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
        "nominal_mean_cost_per_order",
        "robust_mean_cost_per_order",
        "robust_minus_nominal_mean_cost_per_order",
        "robust_over_nominal_mean_cost_per_order",
        "nominal_order_cost_std",
        "robust_order_cost_std",
        "robust_minus_nominal_order_cost_std",
        "robust_over_nominal_order_cost_std",
        "nominal_daily_cost_std",
        "robust_daily_cost_std",
        "robust_minus_nominal_daily_std",
        "robust_over_nominal_daily_std",
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
