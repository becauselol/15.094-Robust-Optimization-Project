#!/usr/bin/env python3
"""
Plot nominal vs robust empirical CDFs of theoretical OD costs by scenario.

For each matched nominal/robust pair, this script computes the implied
minimum-cost assignment for every theoretical OD pair in each scenario period
and writes a 4-panel CDF comparison figure to:

    experiments/<name>/figures/theoretical_od_cdfs_by_scenario/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_weighted_total_cost import (
    compute_best_feasible_pair_cost,
    get_full_station_path,
    get_max_walking_distance,
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


def load_scenario_periods(run_dir: Path) -> list[tuple[str, list[int]]]:
    active_schedule_path = run_dir / "backtest" / "active_station_schedule.csv"
    schedule = {}
    with active_schedule_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["scenario_label"]
            if label not in schedule:
                from compare_weighted_total_cost import parse_active_station_ids

                schedule[label] = parse_active_station_ids(row["active_station_ids"])
    ordered = sorted(schedule.items(), key=lambda item: item[0])
    return ordered


def theoretical_costs_by_scenario(run_dir: Path, lambda_val: float) -> dict[str, list[float]]:
    station_path = get_full_station_path(run_dir)
    segment_path = run_dir / "backtest" / "simulation_inputs" / "segment.csv"
    coords = load_station_coords(station_path)
    routing_costs = load_routing_costs(segment_path, set(coords))
    station_ids = sorted(coords)
    max_walking_distance = get_max_walking_distance(run_dir)
    result: dict[str, list[float]] = {}
    for label, active_ids in load_scenario_periods(run_dir):
        costs: list[float] = []
        for origin_id in station_ids:
            for dest_id in station_ids:
                if origin_id == dest_id:
                    continue
                result_pair = compute_best_feasible_pair_cost(
                    origin_id,
                    dest_id,
                    active_ids,
                    coords,
                    routing_costs,
                    lambda_val,
                    max_walking_distance,
                )
                if result_pair is not None:
                    best_cost, _, _ = result_pair
                    costs.append(best_cost)
        result[label] = costs
    return result


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(arr) + 1, dtype=float) / len(arr)
    return arr, y


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def figure_name(nominal: dict[str, Any], robust: dict[str, Any]) -> str:
    return safe_slug(
        f"k{nominal['k']}_lambda{nominal['in_vehicle_time_weight']}_nom{nominal['job_id']}_rob{robust['job_id']}_q{robust['comparison_quantile']}"
    ) + ".png"


def plot_pair(nominal: dict[str, Any], robust: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    lambda_val = float(nominal["in_vehicle_time_weight"])
    nom = theoretical_costs_by_scenario(nominal["run_dir"], lambda_val)
    rob = theoretical_costs_by_scenario(robust["run_dir"], lambda_val)
    scenario_labels = sorted(set(nom) | set(rob))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, label in zip(axes, scenario_labels):
        nom_costs = nom.get(label, [])
        rob_costs = rob.get(label, [])
        if nom_costs:
            x_nom, y_nom = ecdf(nom_costs)
            ax.plot(x_nom, y_nom, color="#1f77b4", linewidth=2, label=f"Nominal (n={len(nom_costs)})")
        if rob_costs:
            x_rob, y_rob = ecdf(rob_costs)
            ax.plot(x_rob, y_rob, color="#d95f02", linewidth=2, label=f"Robust (n={len(rob_costs)})")
        ax.set_title(label)
        ax.set_xlabel("Theoretical OD weighted cost")
        ax.set_ylabel("Empirical CDF")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"Theoretical OD CDFs by scenario | k={nominal['k']} | lambda={nominal['in_vehicle_time_weight']} | "
        f"nominal job {nominal['job_id']} vs robust job {robust['job_id']} | q={robust['comparison_quantile']}",
        fontsize=12,
    )
    out_path = out_dir / figure_name(nominal, robust)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
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
    out_dir = exp_dir / "figures" / "theoretical_od_cdfs_by_scenario"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [plot_pair(nominal, robust, out_dir) for nominal, robust in pair_runs(exp_dir)]
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["k", "in_vehicle_time_weight", "quantile", "nominal_job_id", "robust_job_id", "figure_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} theoretical OD CDF figures to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
