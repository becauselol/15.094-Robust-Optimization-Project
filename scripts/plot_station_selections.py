#!/usr/bin/env python3
"""
Compare station selections and per-period activations for three models
on the sparse 40-station network.

Produces two figures:
  1. <output_dir>/station_selections_built.png  — 1×3 panels, built stations only
  2. <output_dir>/station_selections_activations.png — 3×4 panels, activations per period

Usage:
    python scripts/plot_station_selections.py <exp_dir> <k> [output_dir] [quantile]

Looks for runs with the given k and models:
  NominalModel (q=None), NominalFeasibleModel (q=None), RobustTotalDemandCapModel (q=0.95 default)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PERIOD_LABELS = {
    "period_1": "Morning\n06–10h",
    "period_2": "Afternoon\n10–15h",
    "period_3": "Evening\n15–20h",
    "period_4": "Night\n20–24h",
}
PERIODS = ["period_1", "period_2", "period_3", "period_4"]

MODEL_COLORS = {
    "Nominal":            "#1f77b4",   # blue
    "Nominal (feasible)": "#ff7f0e",   # orange
    "Robust q=0.95":      "#d62728",   # red
}

MODEL_ORDER = ["Nominal", "Nominal (feasible)", "Robust q=0.95"]

MODEL_SLUG = {
    "Nominal":            "nominal",
    "Nominal (feasible)": "feasible",
    "Robust q=0.95":      "robust_q095",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_stations(run_dir: Path) -> dict[int, dict]:
    """Load WGS84 station coordinates from backtest/cluster_stations.csv."""
    path = run_dir / "backtest" / "cluster_stations.csv"
    stations = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = int(row["id"])
            stations[sid] = {
                "id":   sid,
                "name": row.get("station_name", ""),
                "lon":  float(row["lon"]),
                "lat":  float(row["lat"]),
            }
    return stations


def load_built(run_dir: Path) -> set[int]:
    """Load station IDs with selected=1 from variable_exports/station_selection.csv."""
    path = run_dir / "variable_exports" / "station_selection.csv"
    built = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["selected"]) >= 0.5:
                built.add(int(row["station_id"]))
    return built


def load_activations(run_dir: Path) -> dict[str, set[int]]:
    """Load per-period active station IDs from variable_exports/scenario_activation.csv."""
    path = run_dir / "variable_exports" / "scenario_activation.csv"
    by_period: dict[str, set[int]] = {p: set() for p in PERIODS}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["value"]) >= 0.5:
                label = row["scenario_label"]
                if label in by_period:
                    by_period[label].add(int(row["station_id"]))
    return by_period


def find_run(exp_dir: Path, model_type: str, k: int, quantile=None) -> Path | None:
    """Return the first matching run directory."""
    runs_dir = exp_dir / "runs"
    for run_name in sorted(runs_dir.iterdir()):
        metrics_path = run_name / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        if m.get("model_type") != model_type:
            continue
        if int(m.get("k", -1)) != k:
            continue
        if quantile is not None:
            q = m.get("demand_quantile")
            if q is None or abs(float(q) - quantile) > 1e-9:
                continue
        return run_name
    return None


# ── plotting helpers ───────────────────────────────────────────────────────────

def _scatter_stations(ax, all_stations, built, active, color):
    sids  = sorted(all_stations)
    lons  = [all_stations[s]["lon"] for s in sids]
    lats  = [all_stations[s]["lat"] for s in sids]

    not_built      = [s for s in sids if s not in built]
    built_inactive = [s for s in sids if s in built and s not in active]
    active_ids     = [s for s in sids if s in active]

    def xs(ids): return [all_stations[s]["lon"] for s in ids]
    def ys(ids): return [all_stations[s]["lat"] for s in ids]

    # not built — small light gray
    if not_built:
        ax.scatter(xs(not_built), ys(not_built),
                   s=22, c="#dddddd", edgecolors="none", zorder=2)

    # built but inactive — medium gray (de-emphasised)
    if built_inactive:
        ax.scatter(xs(built_inactive), ys(built_inactive),
                   s=40, c="#aaaaaa", edgecolors="none", zorder=3)

    # active — large colored (prominent)
    if active_ids:
        ax.scatter(xs(active_ids), ys(active_ids),
                   s=90, c=color, edgecolors="white",
                   linewidths=0.7, zorder=4)

    xpad = (max(lons) - min(lons)) * 0.06
    ypad = (max(lats) - min(lats)) * 0.08
    ax.set_xlim(min(lons) - xpad, max(lons) + xpad)
    ax.set_ylim(min(lats) - ypad, max(lats) + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)


# ── figure 1: built stations (1×3) ────────────────────────────────────────────

def plot_built(model_data: dict, k: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Built station selections — $k={k}$",
                 fontsize=12, y=1.01)

    for ax, label in zip(axes, MODEL_ORDER):
        d = model_data[label]
        color = MODEL_COLORS[label]
        _scatter_stations(ax, d["stations"], d["built"], d["built"], color)
        n_unique = len(d["built"])
        ax.set_title(f"{label}\n({n_unique} built)", fontsize=10)

    # legend
    legend_handles = [
        mpatches.Patch(facecolor="#dddddd", edgecolor="none", label=f"Not built ({40 - k})"),
        mpatches.Patch(facecolor="#555555", edgecolor="none", label="Built (colour per model)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── figure 2: activations per period (3×4) ────────────────────────────────────

def plot_activations(model_data: dict, k: int, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle(
        f"Per-period station activations — $k={k}$",
        fontsize=11, y=1.01,
    )

    for row_idx, label in enumerate(MODEL_ORDER):
        d = model_data[label]
        color = MODEL_COLORS[label]
        for col_idx, period in enumerate(PERIODS):
            ax = axes[row_idx][col_idx]
            active = d["activations"].get(period, set())
            _scatter_stations(ax, d["stations"], d["built"], active, color)
            if row_idx == 0:
                ax.set_title(PERIOD_LABELS[period], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, labelpad=6)

    # shared legend
    legend_handles = [
        mpatches.Patch(facecolor="#dddddd", edgecolor="none", label="Not built"),
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="none", label="Built, inactive"),
        mpatches.Patch(facecolor=MODEL_COLORS["Nominal"],
                       edgecolor="white", label="Active (Nominal)"),
        mpatches.Patch(facecolor=MODEL_COLORS["Nominal (feasible)"],
                       edgecolor="white", label="Active (Nominal (feasible))"),
        mpatches.Patch(facecolor=MODEL_COLORS["Robust q=0.95"],
                       edgecolor="white", label="Active (Robust q=0.95)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── individual slides: one file per model (built) ────────────────────────────

def plot_built_individual(model_data: dict, k: int, out_dir: Path) -> None:
    legend_handles = [
        mpatches.Patch(facecolor="#dddddd", edgecolor="none", label="Not built"),
        mpatches.Patch(facecolor="#555555", edgecolor="none", label="Built"),
    ]
    for label in MODEL_ORDER:
        d = model_data[label]
        color = MODEL_COLORS[label]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        _scatter_stations(ax, d["stations"], d["built"], d["built"], color)
        n_unique = len(d["built"])
        ax.set_title(f"{label} — built stations  ($k={k}$, {n_unique} built)", fontsize=10)
        n_total = len(d["stations"])
        legend_handles_m = [
            mpatches.Patch(facecolor="#dddddd", edgecolor="none", label=f"Not built ({n_total - n_unique})"),
            mpatches.Patch(facecolor=color, edgecolor="white", label=f"Built ({n_unique})"),
        ]
        ax.legend(handles=legend_handles_m, loc="lower right", fontsize=8.5,
                  frameon=True, framealpha=0.85)
        fig.tight_layout()
        slug = MODEL_SLUG[label]
        out_path = out_dir / f"built_{slug}_k{k}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


# ── individual slides: one file per (model, period) ──────────────────────────

def plot_activations_individual(model_data: dict, k: int, out_dir: Path) -> None:
    for label in MODEL_ORDER:
        d = model_data[label]
        color = MODEL_COLORS[label]
        slug = MODEL_SLUG[label]
        for period in PERIODS:
            active = d["activations"].get(period, set())
            n_active = len(active)
            fig, ax = plt.subplots(figsize=(5.5, 5))
            _scatter_stations(ax, d["stations"], d["built"], active, color)
            period_str = PERIOD_LABELS[period].replace("\n", " ")
            ax.set_title(f"{label}\n{period_str}  ($k={k}$, {n_active} active)", fontsize=10)
            legend_handles = [
                mpatches.Patch(facecolor="#dddddd", edgecolor="none", label="Not built"),
                mpatches.Patch(facecolor="#aaaaaa", edgecolor="none", label="Built, inactive"),
                mpatches.Patch(facecolor=color, edgecolor="white", label=f"Active ({n_active})"),
            ]
            ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
                      frameon=True, framealpha=0.85)
            fig.tight_layout()
            out_path = out_dir / f"activation_{slug}_{period}_k{k}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/plot_station_selections.py <exp_dir> <k> [output_dir] [quantile]",
              file=sys.stderr)
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    k = int(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 \
              else exp_dir / "analysis" / "station_selections"
    robust_q = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
    robust_label = f"Robust q={robust_q:.2f}".rstrip("0").rstrip(".")

    # Remap MODEL_ORDER/MODEL_COLORS/MODEL_SLUG for the chosen quantile label
    global MODEL_ORDER, MODEL_COLORS, MODEL_SLUG
    MODEL_ORDER = ["Nominal", "Nominal (feasible)", robust_label]
    MODEL_COLORS[robust_label] = MODEL_COLORS.pop("Robust q=0.95", "#d62728")
    MODEL_SLUG[robust_label] = f"robust_q{int(robust_q * 100):03d}"

    targets = [
        ("Nominal",            "NominalModel",              None),
        ("Nominal (feasible)", "NominalFeasibleModel",      None),
        (robust_label,         "RobustTotalDemandCapModel", robust_q),
    ]

    model_data = {}
    for label, mtype, q in targets:
        run = find_run(exp_dir, mtype, k, q)
        if run is None:
            print(f"WARNING: could not find run for {label} (k={k}, q={q})", file=sys.stderr)
            continue
        print(f"  {label}: {run.name}")
        model_data[label] = {
            "stations":    load_stations(run),
            "built":       load_built(run),
            "activations": load_activations(run),
        }

    if len(model_data) < len(targets):
        missing = [l for l, *_ in targets if l not in model_data]
        print(f"ERROR: missing runs for {missing}", file=sys.stderr)
        sys.exit(1)

    plot_built(model_data, k, out_dir / f"station_selections_built_k{k}.png")
    plot_activations(model_data, k, out_dir / f"station_selections_activations_k{k}.png")

    ind_dir = out_dir / "individual"
    plot_built_individual(model_data, k, ind_dir)
    plot_activations_individual(model_data, k, ind_dir)


if __name__ == "__main__":
    main()
