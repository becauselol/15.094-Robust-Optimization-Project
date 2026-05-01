#!/usr/bin/env python3
"""
Plot nominal vs robust z_js activations by time period.

For each matched nominal/robust pair in an experiment, generate one figure with
8 panels:
  - top row: nominal periods 1-4
  - bottom row: robust periods 1-4

Output goes to:
  experiments/<name>/figures/activation_comparisons/
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

from compare_weighted_total_cost import get_full_station_path, load_station_coords


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    import json

    return json.loads(metrics_path.read_text())


def iter_metric_files(exp_dir: Path) -> list[Path]:
    return sorted(exp_dir.glob("runs/*/metrics.json"))


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


def load_station_points(run_dir: Path) -> list[dict[str, float | int]]:
    coords = load_station_coords(get_full_station_path(run_dir))
    return [
        {"station_id": station_id, "lon": lon, "lat": lat}
        for station_id, (lat, lon) in sorted(coords.items())
    ]


def load_period_activations(run_dir: Path) -> dict[str, set[int]]:
    path = run_dir / "variable_exports" / "scenario_activation.csv"
    by_period: dict[str, set[int]] = defaultdict(set)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["value"]) >= 0.5:
                by_period[row["scenario_label"]].add(int(row["station_id"]))
    return dict(by_period)


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def figure_name(nominal: dict[str, Any], robust: dict[str, Any]) -> str:
    return safe_slug(
        f"k{nominal['k']}_lambda{nominal['in_vehicle_time_weight']}_nom{nominal['job_id']}_rob{robust['job_id']}_q{robust['comparison_quantile']}"
    ) + ".png"


def plot_pair(nominal: dict[str, Any], robust: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    stations = load_station_points(nominal["run_dir"])
    nom_by_period = load_period_activations(nominal["run_dir"])
    rob_by_period = load_period_activations(robust["run_dir"])
    periods = ["period_1", "period_2", "period_3", "period_4"]

    lons = [row["lon"] for row in stations]
    lats = [row["lat"] for row in stations]
    xmin, xmax = min(lons), max(lons)
    ymin, ymax = min(lats), max(lats)
    xpad = (xmax - xmin) * 0.05
    ypad = (ymax - ymin) * 0.05

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    base_color = "#d0d0d0"
    nom_color = "#1f77b4"
    rob_color = "#d95f02"

    for col, period in enumerate(periods):
        for row_idx, (label, active_ids, color) in enumerate(
            [
                ("Nominal", nom_by_period.get(period, set()), nom_color),
                ("Robust", rob_by_period.get(period, set()), rob_color),
            ]
        ):
            ax = axes[row_idx, col]
            ax.scatter(lons, lats, s=18, c=base_color, alpha=0.8, linewidths=0)
            active_points = [row for row in stations if row["station_id"] in active_ids]
            if active_points:
                ax.scatter(
                    [row["lon"] for row in active_points],
                    [row["lat"] for row in active_points],
                    s=45,
                    c=color,
                    edgecolors="black",
                    linewidths=0.4,
                )
            ax.set_title(f"{label} {period}")
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"z_js activations | k={nominal['k']} | lambda={nominal['in_vehicle_time_weight']} | "
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
    out_dir = exp_dir / "figures" / "activation_comparisons"
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

    print(f"Wrote {len(rows)} activation comparison figures to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
