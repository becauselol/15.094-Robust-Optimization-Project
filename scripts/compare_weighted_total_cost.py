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
    python scripts/compare_weighted_total_cost.py experiments/2026-05-01_subsample_40 --summary-mode full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev
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


def get_max_walking_distance(run_dir: Path) -> float:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_metrics(metrics_path)
        value = metrics.get("max_walking_distance")
        if value is not None:
            return float(value)
    return float("inf")


def get_full_station_path(run_dir: Path) -> Path:
    cluster_path = run_dir / "backtest" / "cluster_stations.csv"
    if cluster_path.exists():
        return cluster_path
    return run_dir / "backtest" / "simulation_inputs" / "station.csv"


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


def parse_active_station_ids(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(part) for part in raw.split()]


def load_period_map(active_schedule_path: Path) -> list[tuple[int, list[int]]]:
    seen: dict[int, list[int]] = {}
    with active_schedule_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hour = int(row["period_start_hour"])
            if hour not in seen:
                seen[hour] = parse_active_station_ids(row["active_station_ids"])
    return sorted(seen.items(), key=lambda item: item[0])


def compute_best_feasible_pair_cost(
    origin_id: int,
    dest_id: int,
    active_ids: list[int],
    coords: dict[int, tuple[float, float]],
    routing_costs: dict[tuple[int, int], float],
    lambda_val: float,
    max_walking_distance: float,
) -> tuple[float, float, float] | None:
    origin_lat, origin_lon = coords[origin_id]
    dest_lat, dest_lon = coords[dest_id]
    best_cost = float("inf")
    best_wj = float("inf")
    best_wk = float("inf")

    for pickup_id in active_ids:
        pickup_lat, pickup_lon = coords[pickup_id]
        walk_pickup = haversine_meters(origin_lat, origin_lon, pickup_lat, pickup_lon) / 1.4
        if walk_pickup > max_walking_distance:
            continue
        for dropoff_id in active_ids:
            dropoff_lat, dropoff_lon = coords[dropoff_id]
            walk_dropoff = haversine_meters(dropoff_lat, dropoff_lon, dest_lat, dest_lon) / 1.4
            if walk_dropoff > max_walking_distance:
                continue
            route_cost = 0.0 if pickup_id == dropoff_id else routing_costs.get((pickup_id, dropoff_id), float("inf"))
            if not math.isfinite(route_cost):
                continue
            total_cost = walk_pickup + walk_dropoff + lambda_val * route_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_wj = walk_pickup
                best_wk = walk_dropoff

    if not math.isfinite(best_cost):
        return None
    return best_cost, best_wj, best_wk


@lru_cache(maxsize=None)
def compute_theoretical_od_metrics_cached(run_dir_str: str, lambda_val: float) -> dict[str, float | None]:
    run_dir = Path(run_dir_str)
    active_schedule_path = run_dir / "backtest" / "active_station_schedule.csv"
    station_path = get_full_station_path(run_dir)
    segment_path = run_dir / "backtest" / "simulation_inputs" / "segment.csv"
    if not active_schedule_path.exists() or not station_path.exists() or not segment_path.exists():
        return {
            "mean_od_mean_cost_theoretical": None,
            "std_od_mean_cost_theoretical": None,
        }

    coords = load_station_coords(station_path)
    station_ids = sorted(coords)
    routing_costs = load_routing_costs(segment_path, set(coords))
    period_map = load_period_map(active_schedule_path)
    max_walking_distance = get_max_walking_distance(run_dir)
    if not period_map:
        return {
            "mean_od_mean_cost_theoretical": None,
            "std_od_mean_cost_theoretical": None,
        }

    od_costs: dict[tuple[int, int], list[float]] = defaultdict(list)
    for period_start_hour, active_ids in period_map:
        if not active_ids:
            continue
        for origin_id in station_ids:
            origin_lat, origin_lon = coords[origin_id]
            for dest_id in station_ids:
                if origin_id == dest_id:
                    continue
                dest_lat, dest_lon = coords[dest_id]
                _ = (origin_lat, origin_lon, dest_lat, dest_lon)
                result = compute_best_feasible_pair_cost(
                    origin_id,
                    dest_id,
                    active_ids,
                    coords,
                    routing_costs,
                    lambda_val,
                    max_walking_distance,
                )
                if result is not None:
                    best_cost, _, _ = result
                    od_costs[(origin_id, dest_id)].append(best_cost)

    if not od_costs:
        return {
            "mean_od_mean_cost_theoretical": None,
            "std_od_mean_cost_theoretical": None,
        }

    od_means = [mean(costs) for costs in od_costs.values() if costs]
    return {
        "mean_od_mean_cost_theoretical": mean(od_means) if od_means else None,
        "std_od_mean_cost_theoretical": pstdev(od_means) if od_means else None,
    }


def compute_order_costs(run_dir: Path, month: str, lambda_val: float) -> list[float]:
    if month == "April":
        transform_dir = run_dir / "backtest" / "transform_in_sample"
    elif month == "May":
        transform_dir = run_dir / "backtest" / "transform"
    else:
        raise ValueError(f"unexpected month: {month}")

    manifest_path = transform_dir / "daily_manifest.csv"
    station_path = get_full_station_path(run_dir)
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


def get_order_cost_mean(run_dir: Path, month: str, lambda_val: float) -> float | None:
    order_costs = compute_order_costs(run_dir, month, lambda_val)
    if not order_costs:
        return None
    return mean(order_costs)


def get_order_cost_total(run_dir: Path, month: str, lambda_val: float) -> float | None:
    order_costs = compute_order_costs(run_dir, month, lambda_val)
    if not order_costs:
        return None
    return float(sum(order_costs))


def get_realized_walking_violation_rate(run_dir: Path, month: str) -> float | None:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_metrics(metrics_path)
        if month == "April":
            block = metrics.get("in_sample_direct", {})
        elif month == "May":
            block = metrics.get("direct_backtest", {})
        else:
            raise ValueError(f"unexpected month: {month}")

        violations = block.get("orders_walking_violation")
        total_orders = block.get("total_orders")
        if violations is not None and total_orders not in (None, 0):
            return float(violations) / float(total_orders)

    if month == "April":
        transform_dir = run_dir / "backtest" / "transform_in_sample"
    elif month == "May":
        transform_dir = run_dir / "backtest" / "transform"
    else:
        raise ValueError(f"unexpected month: {month}")

    manifest_path = transform_dir / "daily_manifest.csv"
    station_path = get_full_station_path(run_dir)
    if not manifest_path.exists() or not station_path.exists():
        return None

    coords = load_station_coords(station_path)
    max_walking_distance = get_max_walking_distance(run_dir)
    total_assigned = 0
    violations = 0

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
                    origin_lat, origin_lon = coords[origin_id]
                    pickup_lat, pickup_lon = coords[pickup_id]
                    dropoff_lat, dropoff_lon = coords[dropoff_id]
                    dest_lat, dest_lon = coords[dest_id]
                    walk_pickup = haversine_meters(origin_lat, origin_lon, pickup_lat, pickup_lon) / 1.4
                    walk_dropoff = haversine_meters(dropoff_lat, dropoff_lon, dest_lat, dest_lon) / 1.4
                    total_assigned += 1
                    if walk_pickup > max_walking_distance or walk_dropoff > max_walking_distance:
                        violations += 1

    if total_assigned == 0:
        return None
    return violations / total_assigned


def compute_daily_weighted_total_costs(run_dir: Path, month: str, lambda_val: float) -> list[float]:
    if month == "April":
        transform_dir = run_dir / "backtest" / "transform_in_sample"
    elif month == "May":
        transform_dir = run_dir / "backtest" / "transform"
    else:
        raise ValueError(f"unexpected month: {month}")

    manifest_path = transform_dir / "daily_manifest.csv"
    station_path = get_full_station_path(run_dir)
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


def compute_od_costs(run_dir: Path, month: str, lambda_val: float) -> dict[tuple[int, int], list[float]]:
    if month == "April":
        transform_dir = run_dir / "backtest" / "transform_in_sample"
    elif month == "May":
        transform_dir = run_dir / "backtest" / "transform"
    else:
        raise ValueError(f"unexpected month: {month}")

    manifest_path = transform_dir / "daily_manifest.csv"
    station_path = get_full_station_path(run_dir)
    segment_path = run_dir / "backtest" / "simulation_inputs" / "segment.csv"
    if not manifest_path.exists() or not station_path.exists() or not segment_path.exists():
        return {}

    coords = load_station_coords(station_path)
    routing_costs = load_routing_costs(segment_path, set(coords))
    od_costs: dict[tuple[int, int], list[float]] = defaultdict(list)

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
                    od_costs[(origin_id, dest_id)].append(walk_cost + lambda_val * route_cost)
    return dict(od_costs)


def get_od_unweighted_metrics(run_dir: Path, month: str, lambda_val: float) -> dict[str, float | None]:
    od_costs = compute_od_costs(run_dir, month, lambda_val)
    if not od_costs:
        return {
            "mean_od_mean_cost_unweighted": None,
            "std_od_mean_cost_unweighted": None,
        }

    od_means = [mean(costs) for costs in od_costs.values() if costs]
    return {
        "mean_od_mean_cost_unweighted": mean(od_means) if od_means else None,
        "std_od_mean_cost_unweighted": pstdev(od_means) if od_means else None,
    }


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
            "april_weighted_total_cost": get_order_cost_total(metrics_path.parent, "April", float(in_vehicle_time_weight)),
            "may_weighted_total_cost": get_order_cost_total(metrics_path.parent, "May", float(in_vehicle_time_weight)),
            "april_mean_cost_per_order": get_order_cost_mean(metrics_path.parent, "April", float(in_vehicle_time_weight)),
            "may_mean_cost_per_order": get_order_cost_mean(metrics_path.parent, "May", float(in_vehicle_time_weight)),
            "april_order_cost_std": get_order_cost_std(metrics_path.parent, "April", float(in_vehicle_time_weight)),
            "may_order_cost_std": get_order_cost_std(metrics_path.parent, "May", float(in_vehicle_time_weight)),
            "april_daily_cost_std": get_daily_cost_std(metrics_path.parent, "April", float(in_vehicle_time_weight)),
            "may_daily_cost_std": get_daily_cost_std(metrics_path.parent, "May", float(in_vehicle_time_weight)),
            "demand_quantile": metrics.get("demand_quantile"),
            "Q_cap_quantile": metrics.get("Q_cap_quantile"),
            "q_high_quantile": metrics.get("q_high_quantile"),
        }
        april_od_metrics = get_od_unweighted_metrics(metrics_path.parent, "April", float(in_vehicle_time_weight))
        may_od_metrics = get_od_unweighted_metrics(metrics_path.parent, "May", float(in_vehicle_time_weight))
        theoretical_od_metrics = compute_theoretical_od_metrics_cached(str(metrics_path.parent), float(in_vehicle_time_weight))
        record.update({
            "april_mean_od_mean_cost_unweighted": april_od_metrics["mean_od_mean_cost_unweighted"],
            "april_std_od_mean_cost_unweighted": april_od_metrics["std_od_mean_cost_unweighted"],
            "may_mean_od_mean_cost_unweighted": may_od_metrics["mean_od_mean_cost_unweighted"],
            "may_std_od_mean_cost_unweighted": may_od_metrics["std_od_mean_cost_unweighted"],
            "theoretical_mean_od_mean_cost_unweighted": theoretical_od_metrics["mean_od_mean_cost_theoretical"],
            "theoretical_std_od_mean_cost_unweighted": theoretical_od_metrics["std_od_mean_cost_theoretical"],
        })
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
                for month, nominal_cost_key, robust_cost_key, nominal_mean_key, robust_mean_key, nominal_order_std_key, robust_order_std_key, nominal_daily_std_key, robust_daily_std_key, nominal_od_mean_key, robust_od_mean_key, nominal_od_std_key, robust_od_std_key in [
                    ("April", "april_weighted_total_cost", "april_weighted_total_cost", "april_mean_cost_per_order", "april_mean_cost_per_order", "april_order_cost_std", "april_order_cost_std", "april_daily_cost_std", "april_daily_cost_std", "april_mean_od_mean_cost_unweighted", "april_mean_od_mean_cost_unweighted", "april_std_od_mean_cost_unweighted", "april_std_od_mean_cost_unweighted"),
                    ("May", "may_weighted_total_cost", "may_weighted_total_cost", "may_mean_cost_per_order", "may_mean_cost_per_order", "may_order_cost_std", "may_order_cost_std", "may_daily_cost_std", "may_daily_cost_std", "may_mean_od_mean_cost_unweighted", "may_mean_od_mean_cost_unweighted", "may_std_od_mean_cost_unweighted", "may_std_od_mean_cost_unweighted"),
                ]:
                    nominal_cost = nominal[nominal_cost_key]
                    robust_cost = robust[robust_cost_key]
                    nominal_mean = nominal[nominal_mean_key]
                    robust_mean = robust[robust_mean_key]
                    nominal_order_std = nominal[nominal_order_std_key]
                    robust_order_std = robust[robust_order_std_key]
                    nominal_daily_std = nominal[nominal_daily_std_key]
                    robust_daily_std = robust[robust_daily_std_key]
                    nominal_od_mean = nominal[nominal_od_mean_key]
                    robust_od_mean = robust[robust_od_mean_key]
                    nominal_od_std = nominal[nominal_od_std_key]
                    robust_od_std = robust[robust_od_std_key]
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
                            "nominal_mean_od_mean_cost_unweighted": nominal_od_mean,
                            "robust_mean_od_mean_cost_unweighted": robust_od_mean,
                            "robust_minus_nominal_mean_od_mean_cost_unweighted": (
                                robust_od_mean - nominal_od_mean
                                if nominal_od_mean is not None and robust_od_mean is not None
                                else None
                            ),
                            "robust_over_nominal_mean_od_mean_cost_unweighted": (
                                robust_od_mean / nominal_od_mean
                                if nominal_od_mean not in (None, 0) and robust_od_mean is not None
                                else None
                            ),
                            "nominal_std_od_mean_cost_unweighted": nominal_od_std,
                            "robust_std_od_mean_cost_unweighted": robust_od_std,
                            "robust_minus_nominal_std_od_mean_cost_unweighted": (
                                robust_od_std - nominal_od_std
                                if nominal_od_std is not None and robust_od_std is not None
                                else None
                            ),
                            "robust_over_nominal_std_od_mean_cost_unweighted": (
                                robust_od_std / nominal_od_std
                                if nominal_od_std not in (None, 0) and robust_od_std is not None
                                else None
                            ),
                            "nominal_theoretical_mean_od_mean_cost_unweighted": nominal["theoretical_mean_od_mean_cost_unweighted"],
                            "robust_theoretical_mean_od_mean_cost_unweighted": robust["theoretical_mean_od_mean_cost_unweighted"],
                            "robust_minus_nominal_theoretical_mean_od_mean_cost_unweighted": (
                                robust["theoretical_mean_od_mean_cost_unweighted"] - nominal["theoretical_mean_od_mean_cost_unweighted"]
                                if nominal["theoretical_mean_od_mean_cost_unweighted"] is not None and robust["theoretical_mean_od_mean_cost_unweighted"] is not None
                                else None
                            ),
                            "robust_over_nominal_theoretical_mean_od_mean_cost_unweighted": (
                                robust["theoretical_mean_od_mean_cost_unweighted"] / nominal["theoretical_mean_od_mean_cost_unweighted"]
                                if nominal["theoretical_mean_od_mean_cost_unweighted"] not in (None, 0) and robust["theoretical_mean_od_mean_cost_unweighted"] is not None
                                else None
                            ),
                            "nominal_theoretical_std_od_mean_cost_unweighted": nominal["theoretical_std_od_mean_cost_unweighted"],
                            "robust_theoretical_std_od_mean_cost_unweighted": robust["theoretical_std_od_mean_cost_unweighted"],
                            "robust_minus_nominal_theoretical_std_od_mean_cost_unweighted": (
                                robust["theoretical_std_od_mean_cost_unweighted"] - nominal["theoretical_std_od_mean_cost_unweighted"]
                                if nominal["theoretical_std_od_mean_cost_unweighted"] is not None and robust["theoretical_std_od_mean_cost_unweighted"] is not None
                                else None
                            ),
                            "robust_over_nominal_theoretical_std_od_mean_cost_unweighted": (
                                robust["theoretical_std_od_mean_cost_unweighted"] / nominal["theoretical_std_od_mean_cost_unweighted"]
                                if nominal["theoretical_std_od_mean_cost_unweighted"] not in (None, 0) and robust["theoretical_std_od_mean_cost_unweighted"] is not None
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
    fieldnames = FULL_FIELDNAMES if CURRENT_SUMMARY_MODE == "full" else COMPACT_FIELDNAMES
    print("\t".join(fieldnames))
    for row in rows:
        print(
            "\t".join(
                "" if row.get(field) is None else str(row.get(field))
                for field in fieldnames
            )
        )


def print_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = FULL_FIELDNAMES if CURRENT_SUMMARY_MODE == "full" else COMPACT_FIELDNAMES
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows([{field: row.get(field) for field in fieldnames} for row in rows])


FULL_FIELDNAMES = [
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
        "nominal_mean_od_mean_cost_unweighted",
        "robust_mean_od_mean_cost_unweighted",
        "robust_minus_nominal_mean_od_mean_cost_unweighted",
        "robust_over_nominal_mean_od_mean_cost_unweighted",
        "nominal_std_od_mean_cost_unweighted",
        "robust_std_od_mean_cost_unweighted",
        "robust_minus_nominal_std_od_mean_cost_unweighted",
        "robust_over_nominal_std_od_mean_cost_unweighted",
        "nominal_theoretical_mean_od_mean_cost_unweighted",
        "robust_theoretical_mean_od_mean_cost_unweighted",
        "robust_minus_nominal_theoretical_mean_od_mean_cost_unweighted",
        "robust_over_nominal_theoretical_mean_od_mean_cost_unweighted",
        "nominal_theoretical_std_od_mean_cost_unweighted",
        "robust_theoretical_std_od_mean_cost_unweighted",
        "robust_minus_nominal_theoretical_std_od_mean_cost_unweighted",
        "robust_over_nominal_theoretical_std_od_mean_cost_unweighted",
        "nominal_daily_cost_std",
        "robust_daily_cost_std",
        "robust_minus_nominal_daily_std",
        "robust_over_nominal_daily_std",
        "nominal_metrics_path",
        "robust_metrics_path",
    ]


COMPACT_FIELDNAMES = [
        "k",
        "in_vehicle_time_weight",
        "month",
        "quantile",
        "nominal_job_id",
        "robust_job_id",
        "nominal_mean_cost_per_order",
        "robust_mean_cost_per_order",
        "robust_over_nominal_mean_cost_per_order",
        "nominal_order_cost_std",
        "robust_order_cost_std",
        "robust_over_nominal_order_cost_std",
        "nominal_theoretical_mean_od_mean_cost_unweighted",
        "robust_theoretical_mean_od_mean_cost_unweighted",
        "robust_over_nominal_theoretical_mean_od_mean_cost_unweighted",
        "nominal_theoretical_std_od_mean_cost_unweighted",
        "robust_theoretical_std_od_mean_cost_unweighted",
        "robust_over_nominal_theoretical_std_od_mean_cost_unweighted",
        "nominal_metrics_path",
        "robust_metrics_path",
]


CURRENT_SUMMARY_MODE = "compact"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    parser.add_argument(
        "--format",
        choices=["tsv", "csv"],
        default="tsv",
        help="Output format",
    )
    parser.add_argument(
        "--summary-mode",
        choices=["compact", "full"],
        default="compact",
        help="Column set to print",
    )
    return parser.parse_args()


def main() -> int:
    global CURRENT_SUMMARY_MODE

    args = parse_args()
    CURRENT_SUMMARY_MODE = args.summary_mode
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
