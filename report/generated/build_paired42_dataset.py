#!/usr/bin/env python3

import csv
import json
import math
import shutil
from itertools import combinations
from pathlib import Path


SRC = Path("../Data/zhuzhou_data")
OUT = Path("generated/zhuzhou_data_42_paired")


def haversine_meters(lat1, lon1, lat2, lon2):
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


def load_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def greedy_closest_matching(stations):
    coords = {
        int(row["station_id"]): (float(row["station_lat"]), float(row["station_lon"]))
        for row in stations
    }
    remaining = set(coords)
    all_pairs = []
    for a, b in combinations(sorted(coords), 2):
        lat1, lon1 = coords[a]
        lat2, lon2 = coords[b]
        dist = haversine_meters(lat1, lon1, lat2, lon2)
        all_pairs.append((dist, a, b))
    all_pairs.sort()

    pairs = []
    for dist, a, b in all_pairs:
        if a in remaining and b in remaining:
            pairs.append((a, b, dist))
            remaining.remove(a)
            remaining.remove(b)
    if remaining:
        raise RuntimeError(f"Unmatched stations remain: {sorted(remaining)}")
    return pairs


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    station_rows = load_rows(SRC / "station.csv")
    order_rows = load_rows(SRC / "order.csv")
    segment_rows = load_rows(SRC / "segment.csv")

    pairs = greedy_closest_matching(station_rows)
    merged_id_of = {}
    pair_rows = []
    station_lookup = {int(r["station_id"]): r for r in station_rows}

    for merged_idx, (a, b, dist_m) in enumerate(pairs, start=1):
        merged_id_of[a] = merged_idx
        merged_id_of[b] = merged_idx
        ra = station_lookup[a]
        rb = station_lookup[b]
        pair_rows.append(
            {
                "merged_station_id": merged_idx,
                "member_station_ids": [a, b],
                "member_station_names": [ra["station_name"], rb["station_name"]],
                "pair_distance_m": dist_m,
            }
        )

    # Merged station file
    station_fields = ["station_id", "station_name", "station_lon", "station_lat"]
    with (OUT / "station.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=station_fields)
        writer.writeheader()
        for row in pair_rows:
            a, b = row["member_station_ids"]
            ra = station_lookup[a]
            rb = station_lookup[b]
            writer.writerow(
                {
                    "station_id": row["merged_station_id"],
                    "station_name": f"{ra['station_name']} / {rb['station_name']}",
                    "station_lon": (
                        float(ra["station_lon"]) + float(rb["station_lon"])
                    )
                    / 2.0,
                    "station_lat": (
                        float(ra["station_lat"]) + float(rb["station_lat"])
                    )
                    / 2.0,
                }
            )

    # Orders mapped to merged stations
    self_merged_orders = 0
    with (OUT / "order.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=order_rows[0].keys())
        writer.writeheader()
        for row in order_rows:
            mapped = dict(row)
            o = int(row["origin_station_id"])
            d = int(row["destination_station_id"])
            mapped["origin_station_id"] = str(merged_id_of[o])
            mapped["destination_station_id"] = str(merged_id_of[d])
            if mapped["origin_station_id"] == mapped["destination_station_id"]:
                self_merged_orders += 1
            writer.writerow(mapped)

    # Segments aggregated by minimum time/dist between merged nodes
    seg_best = {}
    for row in segment_rows:
        a = merged_id_of[int(row["from_station"])]
        b = merged_id_of[int(row["to_station"])]
        if a == b:
            continue
        key = (a, b)
        dist = float(row["seg_dist"])
        time = float(row["seg_time"])
        best = seg_best.get(key)
        if best is None or time < best["seg_time"]:
            seg_best[key] = {"seg_dist": dist, "seg_time": time}

    with (OUT / "segment.csv").open("w", newline="") as f:
        fields = ["id", "from_station", "to_station", "seg_dist", "seg_time"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, ((a, b), vals) in enumerate(sorted(seg_best.items()), start=1):
            writer.writerow(
                {
                    "id": idx,
                    "from_station": a,
                    "to_station": b,
                    "seg_dist": vals["seg_dist"],
                    "seg_time": vals["seg_time"],
                }
            )

    shutil.copy(SRC / "vehicle.csv", OUT / "vehicle.csv")

    metadata = {
        "source_dataset": "zhuzhou_data",
        "construction": "greedy closest-pair perfect matching over all 84 stations",
        "merged_station_count": len(pair_rows),
        "original_station_count": len(station_rows),
        "original_order_count": len(order_rows),
        "merged_order_count": len(order_rows),
        "self_merged_orders": self_merged_orders,
        "original_segment_count": len(segment_rows),
        "merged_segment_count": len(seg_best),
        "pairs": pair_rows,
    }
    (OUT / "merge_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(OUT)
    print(json.dumps(
        {
            "merged_station_count": metadata["merged_station_count"],
            "self_merged_orders": self_merged_orders,
            "merged_segment_count": metadata["merged_segment_count"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
