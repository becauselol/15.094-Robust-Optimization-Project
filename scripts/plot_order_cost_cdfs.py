#!/usr/bin/env python3
"""
Generate nominal-vs-robust empirical CDF comparisons of order-level weighted cost.

The script matches runs on:
  - k
  - in_vehicle_time_weight

For robust runs, `demand_quantile` is preferred. Older runs using matching
`Q_cap_quantile == q_high_quantile` are also supported.

For each nominal/robust pair, it reconstructs order-level weighted costs for
April and May and writes two-panel CDF comparison figures to:

    experiments/<name>/figures/order_cost_cdfs/

Usage:
    python3 scripts/plot_order_cost_cdfs.py experiments/2026-04-30_sparse_subsample_40
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

from compare_weighted_total_cost import compute_order_costs, iter_metric_files, load_metrics


def safe_slug(value: str) -> str:
    chars = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


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
            "model_type": model_type,
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
            key=lambda row: (
                row["comparison_quantile"],
                row["job_id"] if row["job_id"] is not None else -1,
                row["timestamp"] or "",
            ),
        )
        if not nominal_runs:
            continue
        for nominal in nominal_runs:
            for robust in robust_runs:
                pairs.append((nominal, robust))
    return pairs


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(arr) + 1, dtype=float) / len(arr)
    return arr, y


def build_figure_name(nominal: dict[str, Any], robust: dict[str, Any]) -> str:
    return safe_slug(
        f"k{nominal['k']}_lambda{nominal['in_vehicle_time_weight']}"
        f"_nom{nominal['job_id']}_rob{robust['job_id']}_q{robust['comparison_quantile']}"
    ) + ".png"


def plot_pair(nominal: dict[str, Any], robust: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    lambda_val = float(nominal["in_vehicle_time_weight"])
    nominal_april = compute_order_costs(nominal["run_dir"], "April", lambda_val)
    nominal_may = compute_order_costs(nominal["run_dir"], "May", lambda_val)
    robust_april = compute_order_costs(robust["run_dir"], "April", lambda_val)
    robust_may = compute_order_costs(robust["run_dir"], "May", lambda_val)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    panel_data = [
        ("April", nominal_april, robust_april),
        ("May", nominal_may, robust_may),
    ]
    colors = {"Nominal": "#1f77b4", "Robust": "#d95f02"}

    for ax, (month, nom_costs, rob_costs) in zip(axes, panel_data):
        if nom_costs:
            x_nom, y_nom = ecdf(nom_costs)
            ax.plot(x_nom, y_nom, label=f"Nominal (n={len(nom_costs)})", color=colors["Nominal"], linewidth=2)
        if rob_costs:
            x_rob, y_rob = ecdf(rob_costs)
            ax.plot(x_rob, y_rob, label=f"Robust (n={len(rob_costs)})", color=colors["Robust"], linewidth=2)
        ax.set_title(month)
        ax.set_xlabel("Order weighted cost")
        ax.set_ylabel("Empirical CDF")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(
        f"k={nominal['k']} | lambda={nominal['in_vehicle_time_weight']} | "
        f"nominal job {nominal['job_id']} vs robust job {robust['job_id']} | "
        f"q={robust['comparison_quantile']}",
        fontsize=12,
    )

    out_path = out_dir / build_figure_name(nominal, robust)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "k": nominal["k"],
        "in_vehicle_time_weight": nominal["in_vehicle_time_weight"],
        "quantile": robust["comparison_quantile"],
        "nominal_job_id": nominal["job_id"],
        "robust_job_id": robust["job_id"],
        "figure_path": str(out_path),
        "april_nominal_orders": len(nominal_april),
        "april_robust_orders": len(robust_april),
        "may_nominal_orders": len(nominal_may),
        "may_robust_orders": len(robust_may),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir
    out_dir = exp_dir / "figures" / "order_cost_cdfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [plot_pair(nominal, robust, out_dir) for nominal, robust in pair_runs(exp_dir)]

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "in_vehicle_time_weight",
                "quantile",
                "nominal_job_id",
                "robust_job_id",
                "figure_path",
                "april_nominal_orders",
                "april_robust_orders",
                "may_nominal_orders",
                "may_robust_orders",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} CDF figures to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
