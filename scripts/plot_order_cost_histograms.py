#!/usr/bin/env python3
"""
Generate per-run histograms of order-level weighted costs.

For each run under an experiment directory, this script reconstructs the
weighted cost of each realized order for April (`transform_in_sample`) and May
(`transform`), then writes a two-panel histogram figure to:

    experiments/<name>/figures/order_cost_histograms/

Usage:
    python3 scripts/plot_order_cost_histograms.py experiments/2026-04-30_sparse_subsample_40
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_weighted_total_cost import compute_order_costs, iter_metric_files, load_metrics


def safe_slug(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def figure_name(metrics: dict, run_dir: Path) -> str:
    job_id = metrics.get("job_id", "unknown")
    model_type = metrics.get("model_type", "unknown")
    k = metrics.get("k", "na")
    lam = metrics.get("in_vehicle_time_weight", "na")
    quantile = metrics.get("demand_quantile", metrics.get("Q_cap_quantile", "na"))
    return safe_slug(
        f"job{job_id}_{model_type}_k{k}_lambda{lam}_q{quantile}_{run_dir.name}"
    ) + ".png"


def plot_run_histogram(metrics_path: Path, out_dir: Path, bins: int) -> dict[str, str | int | float]:
    metrics = load_metrics(metrics_path)
    run_dir = metrics_path.parent
    lambda_val = float(metrics.get("in_vehicle_time_weight", 0.0))
    april_costs = compute_order_costs(run_dir, "April", lambda_val)
    may_costs = compute_order_costs(run_dir, "May", lambda_val)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    months = [("April", april_costs), ("May", may_costs)]
    colors = {"April": "#1f77b4", "May": "#d95f02"}

    for ax, (month, costs) in zip(axes, months):
        if costs:
            ax.hist(costs, bins=bins, color=colors[month], alpha=0.8, edgecolor="white")
            ax.axvline(sum(costs) / len(costs), color="black", linestyle="--", linewidth=1.2)
            ax.set_title(f"{month} | n={len(costs)}")
        else:
            ax.text(0.5, 0.5, "No reconstructed orders", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{month} | n=0")
        ax.set_xlabel("Order weighted cost")
        ax.set_ylabel("Count")

    fig.suptitle(
        f"job {metrics.get('job_id')} | {metrics.get('model_type')} | "
        f"k={metrics.get('k')} | lambda={metrics.get('in_vehicle_time_weight')}",
        fontsize=12,
    )

    out_path = out_dir / figure_name(metrics, run_dir)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "job_id": metrics.get("job_id"),
        "model_type": metrics.get("model_type"),
        "k": metrics.get("k"),
        "in_vehicle_time_weight": metrics.get("in_vehicle_time_weight"),
        "demand_quantile": metrics.get("demand_quantile", metrics.get("Q_cap_quantile")),
        "run_dir": str(run_dir),
        "figure_path": str(out_path),
        "april_orders": len(april_costs),
        "may_orders": len(may_costs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins per panel")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir
    if not exp_dir.exists():
        print(f"experiment directory not found: {exp_dir}", file=sys.stderr)
        return 1

    out_dir = exp_dir / "figures" / "order_cost_histograms"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_path in iter_metric_files(exp_dir):
        rows.append(plot_run_histogram(metrics_path, out_dir, args.bins))

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job_id",
                "model_type",
                "k",
                "in_vehicle_time_weight",
                "demand_quantile",
                "run_dir",
                "figure_path",
                "april_orders",
                "may_orders",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} histogram figures to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
