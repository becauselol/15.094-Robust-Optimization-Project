#!/usr/bin/env python3
"""
Plot the 40-station Zhuzhou optimization instance as a clean geographic scatter.

Usage:
    python scripts/plot_sparse_station_map.py
    python scripts/plot_sparse_station_map.py <station_csv> <output_prefix>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_STATION_CSV = Path("Data/zhuzhou_data_40_sparse_v3/station.csv")
DEFAULT_OUTPUT_PREFIX = Path("report/generated/zhuzhou_sparse40_station_map")


def load_stations(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "station_id": int(row["station_id"]),
                    "station_name": row["station_name"],
                    "lon": float(row["station_lon"]),
                    "lat": float(row["station_lat"]),
                }
            )
    return rows


def make_plot(stations: list[dict[str, object]], output_prefix: Path) -> None:
    lons = [row["lon"] for row in stations]
    lats = [row["lat"] for row in stations]

    lon_pad = (max(lons) - min(lons)) * 0.08
    lat_pad = (max(lats) - min(lats)) * 0.10

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.scatter(
        lons,
        lats,
        s=46,
        c="#D95F02",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

    ax.set_xlim(min(lons) - lon_pad, max(lons) + lon_pad)
    ax.set_ylim(min(lats) - lat_pad, max(lats) + lat_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Zhuzhou 40-Station Optimization Instance")
    ax.grid(True, color="#e5e5e5", linewidth=0.6, zorder=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    station_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STATION_CSV
    output_prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_PREFIX
    stations = load_stations(station_csv)
    make_plot(stations, output_prefix)
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
