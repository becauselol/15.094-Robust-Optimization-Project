#!/usr/bin/env python3
"""
Render the sparse-network results table as a presentation-friendly PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT = Path("presentation/img/sparse_results_table.png")


def build_dataframe() -> pd.DataFrame:
    rows = [
        ["10", "--", "Nominal", "4446.97", "4305.57", "851.46", "1092.41", "227"],
        ["10", "--", "Smoothed", "4477.06", "4326.80", "860.26", "1108.36", "0"],
        ["10", "0.90", "Robust", "4505.20", "4363.35", "845.97", "1100.00", "0"],
        ["10", "0.95", "Robust", "4521.04", "4392.62", "836.33", "1094.39", "0"],
        ["10", "0.99", "Robust", "4507.51", "4368.66", "840.04", "1089.52", "0"],
        ["15", "--", "Nominal", "4182.48", "4031.55", "881.21", "1082.65", "14"],
        ["15", "--", "Smoothed", "4183.29", "4030.61", "882.52", "1082.23", "0"],
        ["15", "0.90", "Robust", "4234.00", "4080.66", "853.99", "1059.06", "0"],
        ["15", "0.95", "Robust", "4235.88", "4089.67", "848.30", "1058.01", "0"],
        ["15", "0.99", "Robust", "4236.58", "4073.98", "855.82", "1067.54", "0"],
        ["20", "--", "Nominal", "4136.13", "3972.53", "901.73", "1094.37", "0"],
        ["20", "--", "Smoothed", "4136.17", "3972.66", "901.21", "1093.63", "0"],
        ["20", "0.90", "Robust", "4149.90", "3985.01", "886.97", "1082.80", "0"],
        ["20", "0.95", "Robust", "4155.78", "3998.38", "883.00", "1077.17", "0"],
        ["20", "0.99", "Robust", "4148.22", "3984.41", "887.77", "1079.39", "0"],
    ]
    cols = ["k", "q", "Model", "April mean", "May mean", "April std", "May std", "Uncovered"]
    return pd.DataFrame(rows, columns=cols)


def render_table(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6.8))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    header_color = "#E9EEF4"
    group_colors = {"10": "#FFF7EC", "15": "#F4F8F0", "20": "#EEF5FB"}

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#B8C2CC")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
        else:
            k = df.iloc[r - 1, 0]
            cell.set_facecolor(group_colors.get(k, "white"))

    ax.set_title("Sparse-Network Three-Model Comparison", fontsize=16, pad=20)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    render_table(build_dataframe(), OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
