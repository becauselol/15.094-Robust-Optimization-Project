#!/usr/bin/env python3
"""
Plot nominal vs robust empirical CDFs of realized order costs by scenario.

For each matched nominal/robust pair, this script groups reconstructed order
costs by period scenario (`period_1` ... `period_4`) and writes one 4-panel CDF
figure for April and one for May to:

    experiments/<name>/figures/order_cost_cdfs_by_scenario/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_weighted_total_cost import (
    get_full_station_path,
    haversine_meters,
    iter_metric_files,
    load_metrics,
    load_routing_costs,
    load_station_coords,
)


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


def pair_runs(exp_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    nominal_by_key, robust_by_key = collect_runs(exp_dir)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
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
        for nominal in nominal_runs:
            for robust in robust_runs:
                pairs.append((nominal, robust))
    return pairs


def period_label_from_hour(hour: int) -> str | None:
    if 6 <= hour < 10:
        return "period_1"
    if 10 <= hour < 15:
        return "period_2"
    if 15 <= hour < 20:
        return "period_3"
    if 20 <= hour < 24:
        return "period_4"
    return None


def compute_order_costs_by_scenario(run_dir: Path, month: str, lambda_val: float) -> dict[str, list[float]]:
    if month == "April":
        transform_dir = run_dir / "backtest" / "transform_in_sample"
    elif month == "May":
        transform_dir = run_dir / "backtest" / "transform"
    else:
        raise ValueError(f"unexpected month: {month}")

    manifest_path = transform_dir / "daily_manifest.csv"
    station_path = get_full_station_path(run_dir)
    segment_path = run_dir / "backtest" / "simulation_inputs" / "segment.csv"
    coords = load_station_coords(station_path)
    routing_costs = load_routing_costs(segment_path, set(coords))
    result: dict[str, list[float]] = defaultdict(list)

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
                    route_cost = routing_costs.get((pickup_id, dropoff_id), float("inf"))
                    if route_cost == float("inf"):
                        continue
                    origin_lat, origin_lon = coords[origin_id]
                    pickup_lat, pickup_lon = coords[pickup_id]
                    dropoff_lat, dropoff_lon = coords[dropoff_id]
                    dest_lat, dest_lon = coords[dest_id]
                    walk_cost = (
                        haversine_meters(origin_lat, origin_lon, pickup_lat, pickup_lon) / 1.4
                        + haversine_meters(dropoff_lat, dropoff_lon, dest_lat, dest_lon) / 1.4
                    )
                    order_time = datetime.strptime(row["order_time"], "%Y-%m-%d %H:%M:%S")
                    label = period_label_from_hour(order_time.hour)
                    if label is None:
                        continue
                    result[label].append(walk_cost + lambda_val * route_cost)
    return dict(result)


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(arr) + 1, dtype=float) / len(arr)
    return arr, y


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def figure_name(nominal: dict[str, Any], robust: dict[str, Any], month: str) -> str:
    return safe_slug(
        f"{month}_k{nominal['k']}_lambda{nominal['in_vehicle_time_weight']}_nom{nominal['job_id']}_rob{robust['job_id']}_q{robust['comparison_quantile']}"
    ) + ".png"


def plot_pair_month(nominal: dict[str, Any], robust: dict[str, Any], month: str, out_dir: Path) -> dict[str, Any]:
    lambda_val = float(nominal["in_vehicle_time_weight"])
    nom = compute_order_costs_by_scenario(nominal["run_dir"], month, lambda_val)
    rob = compute_order_costs_by_scenario(robust["run_dir"], month, lambda_val)
    labels = ["period_1", "period_2", "period_3", "period_4"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, label in zip(axes, labels):
        nom_costs = nom.get(label, [])
        rob_costs = rob.get(label, [])
        if nom_costs:
            x_nom, y_nom = ecdf(nom_costs)
            ax.plot(x_nom, y_nom, color="#1f77b4", linewidth=2, label=f"Nominal (n={len(nom_costs)})")
        if rob_costs:
            x_rob, y_rob = ecdf(rob_costs)
            ax.plot(x_rob, y_rob, color="#d95f02", linewidth=2, label=f"Robust (n={len(rob_costs)})")
        ax.set_title(label)
        ax.set_xlabel("Order weighted cost")
        ax.set_ylabel("Empirical CDF")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(
        f"{month} order-cost CDFs by scenario | k={nominal['k']} | lambda={nominal['in_vehicle_time_weight']} | "
        f"nominal job {nominal['job_id']} vs robust job {robust['job_id']} | q={robust['comparison_quantile']}",
        fontsize=12,
    )
    out_path = out_dir / figure_name(nominal, robust, month)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "month": month,
        "k": nominal["k"],
        "in_vehicle_time_weight": nominal["in_vehicle_time_weight"],
        "quantile": robust["comparison_quantile"],
        "nominal_job_id": nominal["job_id"],
        "robust_job_id": robust["job_id"],
        "figure_path": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir
    out_dir = exp_dir / "figures" / "order_cost_cdfs_by_scenario"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for nominal, robust in pair_runs(exp_dir):
        rows.append(plot_pair_month(nominal, robust, "April", out_dir))
        rows.append(plot_pair_month(nominal, robust, "May", out_dir))

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["month", "k", "in_vehicle_time_weight", "quantile", "nominal_job_id", "robust_job_id", "figure_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} order-cost CDF-by-scenario figures to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
