#!/usr/bin/env python3
"""
Filter zhuzhou_data to top-N most popular stations.

Popularity = total times a station appears as origin or destination in order.csv.
Orders are kept only if BOTH origin_station_id AND destination_station_id are in
the top-N set. Segments are kept only if BOTH endpoints are in the top-N set.
Vehicle and station files are filtered/copied accordingly.

Usage:
    python scripts/filter_stations.py <N> <output_dir>

Example:
    python scripts/filter_stations.py 40 Data/zhuzhou_data_40
    python scripts/filter_stations.py 20 Data/zhuzhou_data_20
"""

import sys
import csv
import os
from collections import Counter

DATA_DIR = "Data/zhuzhou_data"


def filter_to_top_n(n: int, out_dir: str) -> None:
    order_file   = os.path.join(DATA_DIR, "order.csv")
    station_file = os.path.join(DATA_DIR, "station.csv")
    segment_file = os.path.join(DATA_DIR, "segment.csv")
    vehicle_file = os.path.join(DATA_DIR, "vehicle.csv")

    # Count station popularity
    counts: Counter = Counter()
    with open(order_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[int(row["origin_station_id"])] += 1
            counts[int(row["destination_station_id"])] += 1

    top_n = {sid for sid, _ in counts.most_common(n)}
    print(f"Top-{n} stations: {sorted(top_n)}")

    os.makedirs(out_dir, exist_ok=True)

    # Filter orders — keep only rows where both endpoints are in top_n
    kept_orders = 0
    with open(order_file) as fin, open(os.path.join(out_dir, "order.csv"), "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            o = int(row["origin_station_id"])
            d = int(row["destination_station_id"])
            if o in top_n and d in top_n:
                writer.writerow(row)
                kept_orders += 1
    print(f"Orders kept: {kept_orders} / {sum(counts.values()) // 2}")

    # Filter stations
    kept_stations = 0
    with open(station_file) as fin, open(os.path.join(out_dir, "station.csv"), "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["station_id"]) in top_n:
                writer.writerow(row)
                kept_stations += 1
    print(f"Stations kept: {kept_stations}")

    # Filter segments — keep only rows where both endpoints are in top_n
    kept_segs = 0
    with open(segment_file) as fin, open(os.path.join(out_dir, "segment.csv"), "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["from_station"]) in top_n and int(row["to_station"]) in top_n:
                writer.writerow(row)
                kept_segs += 1
    print(f"Segments kept: {kept_segs}")

    # Copy vehicle file unchanged
    import shutil
    shutil.copy(vehicle_file, os.path.join(out_dir, "vehicle.csv"))
    print("Vehicle file copied.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    n = int(sys.argv[1])
    out_dir = sys.argv[2]
    filter_to_top_n(n, out_dir)
