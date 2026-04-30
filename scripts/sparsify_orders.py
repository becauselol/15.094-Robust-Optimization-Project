#!/usr/bin/env python3
"""
Create a sparser, noisier copy of a dataset by degrading order observations.

This is intended to mimic real-world measurement imperfections where demand is
not fully observed: some trips are missing entirely, low-volume OD pairs are
under-recorded, and some time buckets drop out.

The script only modifies `order.csv`. It copies `station.csv`, `segment.csv`,
and `vehicle.csv` unchanged.

Usage:
    python scripts/sparsify_orders.py Data/zhuzhou_data_40 Data/zhuzhou_data_40_sparse

Example with stronger sparsity:
    python scripts/sparsify_orders.py \
        Data/zhuzhou_data_40 \
        Data/zhuzhou_data_40_sparse_v1 \
        --base-keep-prob 0.55 \
        --station-bias-gamma 1.2 \
        --min-station-keep 0.25 \
        --od-drop-frac 0.20 \
        --bucket-drop-frac 0.10 \
        --seed 42

Example with concentrated observed demand:
    python scripts/sparsify_orders.py \
        Data/zhuzhou_data_40 \
        Data/zhuzhou_data_40_sparse_v1 \
        --mode concentrate_head \
        --head-od-frac 0.03 \
        --head-weight-alpha 1.0 \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Directory containing station/order/segment/vehicle CSVs")
    parser.add_argument("output_dir", help="Directory for sparsified output")
    parser.add_argument("--mode", choices=["dropout", "concentrate_head"], default="dropout",
                        help="How to degrade observations: drop rows or reassign tail OD demand onto popular head ODs")
    parser.add_argument("--base-keep-prob", type=float, default=0.65,
                        help="Global order retention probability before other effects")
    parser.add_argument("--station-bias-gamma", type=float, default=1.0,
                        help="How strongly low-popularity stations are under-observed")
    parser.add_argument("--min-station-keep", type=float, default=0.35,
                        help="Minimum station-level reporting probability")
    parser.add_argument("--od-drop-frac", type=float, default=0.10,
                        help="Fraction of low-volume OD pairs to drop completely")
    parser.add_argument("--bucket-drop-frac", type=float, default=0.05,
                        help="Fraction of date-period buckets to drop completely")
    parser.add_argument("--bucket-mode", choices=["date_hour", "date_period4"], default="date_period4",
                        help="Granularity for time-bucket outages")
    parser.add_argument("--head-od-frac", type=float, default=0.03,
                        help="In concentrate_head mode, keep only this top fraction of OD pairs as directly observed")
    parser.add_argument("--head-weight-alpha", type=float, default=1.0,
                        help="In concentrate_head mode, sample replacement head ODs with weights count^alpha")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def load_orders(order_path: Path) -> list[dict[str, str]]:
    with order_path.open(newline="") as f:
        return list(csv.DictReader(f))


def save_orders(order_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with order_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def period4(hour: int) -> int | None:
    if 6 <= hour < 10:
        return 1
    if 10 <= hour < 15:
        return 2
    if 15 <= hour < 20:
        return 3
    if 20 <= hour < 24:
        return 4
    return None


def bucket_key(order_time: str, mode: str) -> tuple[str, str | int]:
    date = order_time[:10]
    hour = int(order_time[11:13])
    if mode == "date_hour":
        return (date, f"{hour:02d}")
    p = period4(hour)
    return (date, f"period_{p}" if p is not None else f"off_{hour:02d}")


def station_keep_probs(rows: list[dict[str, str]], min_keep: float, gamma: float) -> dict[str, float]:
    counts = Counter()
    for row in rows:
        counts[row["origin_station_id"]] += 1
        counts[row["destination_station_id"]] += 1

    max_count = max(counts.values()) if counts else 1
    probs: dict[str, float] = {}
    for sid, count in counts.items():
        rel = count / max_count
        probs[sid] = min_keep + (1.0 - min_keep) * (rel ** gamma)
    return probs


def choose_tail_ods_to_drop(rows: list[dict[str, str]], frac: float, rng: random.Random) -> set[tuple[str, str]]:
    od_counts = Counter((row["origin_station_id"], row["destination_station_id"]) for row in rows)
    if not od_counts or frac <= 0:
        return set()

    ordered = sorted(od_counts.items(), key=lambda item: (item[1], item[0]))
    n_drop = int(round(frac * len(ordered)))
    if n_drop <= 0:
        return set()

    tail_pool = ordered[: max(n_drop * 3, n_drop)]
    chosen = rng.sample(tail_pool, k=min(n_drop, len(tail_pool)))
    return {od for od, _ in chosen}


def choose_buckets_to_drop(rows: list[dict[str, str]], frac: float, mode: str, rng: random.Random) -> set[tuple[str, str | int]]:
    buckets = sorted({bucket_key(row["order_time"], mode) for row in rows})
    if not buckets or frac <= 0:
        return set()
    n_drop = int(round(frac * len(buckets)))
    if n_drop <= 0:
        return set()
    return set(rng.sample(buckets, k=min(n_drop, len(buckets))))


def sparsify_orders(
    rows: list[dict[str, str]],
    *,
    base_keep_prob: float,
    station_bias_gamma: float,
    min_station_keep: float,
    od_drop_frac: float,
    bucket_drop_frac: float,
    bucket_mode: str,
    rng: random.Random,
) -> tuple[list[dict[str, str]], dict]:
    st_keep = station_keep_probs(rows, min_station_keep, station_bias_gamma)
    dropped_ods = choose_tail_ods_to_drop(rows, od_drop_frac, rng)
    dropped_buckets = choose_buckets_to_drop(rows, bucket_drop_frac, bucket_mode, rng)

    kept: list[dict[str, str]] = []
    reasons = Counter()

    for row in rows:
        od = (row["origin_station_id"], row["destination_station_id"])
        if od in dropped_ods:
            reasons["od_drop"] += 1
            continue

        bucket = bucket_key(row["order_time"], bucket_mode)
        if bucket in dropped_buckets:
            reasons["bucket_drop"] += 1
            continue

        p_o = st_keep.get(row["origin_station_id"], min_station_keep)
        p_d = st_keep.get(row["destination_station_id"], min_station_keep)
        keep_prob = base_keep_prob * math.sqrt(p_o * p_d)
        keep_prob = max(0.0, min(1.0, keep_prob))
        if rng.random() <= keep_prob:
            kept.append(row)
        else:
            reasons["thinning_drop"] += 1

    metadata = {
        "input_orders": len(rows),
        "output_orders": len(kept),
        "retention_rate": len(kept) / len(rows) if rows else 0.0,
        "base_keep_prob": base_keep_prob,
        "station_bias_gamma": station_bias_gamma,
        "min_station_keep": min_station_keep,
        "od_drop_frac": od_drop_frac,
        "bucket_drop_frac": bucket_drop_frac,
        "bucket_mode": bucket_mode,
        "drop_reasons": dict(reasons),
        "dropped_od_pairs": [list(od) for od in sorted(dropped_ods)],
        "dropped_buckets": [list(bucket) for bucket in sorted(dropped_buckets)],
        "station_keep_prob_summary": {
            "min": min(st_keep.values()) if st_keep else 0.0,
            "median_like": sorted(st_keep.values())[len(st_keep) // 2] if st_keep else 0.0,
            "max": max(st_keep.values()) if st_keep else 0.0,
        },
    }
    return kept, metadata


def weighted_choice(items: list[tuple[str, str]], weights: list[float], rng: random.Random) -> tuple[str, str]:
    total = sum(weights)
    if total <= 0:
        return items[rng.randrange(len(items))]
    draw = rng.random() * total
    running = 0.0
    for item, weight in zip(items, weights):
        running += weight
        if draw <= running:
            return item
    return items[-1]


def concentrate_head_orders(
    rows: list[dict[str, str]],
    *,
    head_od_frac: float,
    head_weight_alpha: float,
    rng: random.Random,
) -> tuple[list[dict[str, str]], dict]:
    od_counts = Counter((row["origin_station_id"], row["destination_station_id"]) for row in rows)
    unique_ods = len(od_counts)
    if unique_ods == 0:
        return [], {
            "input_orders": 0,
            "output_orders": 0,
            "retention_rate": 0.0,
            "mode": "concentrate_head",
            "head_od_frac": head_od_frac,
            "head_weight_alpha": head_weight_alpha,
            "head_od_count": 0,
            "shifted_orders": 0,
            "shifted_fraction": 0.0,
        }

    head_count = max(1, int(math.ceil(unique_ods * head_od_frac)))
    ordered_ods = [od for od, _ in od_counts.most_common()]
    head_ods = ordered_ods[:head_count]
    head_set = set(head_ods)
    head_weights = [float(od_counts[od]) ** head_weight_alpha for od in head_ods]

    transformed: list[dict[str, str]] = []
    shifted_orders = 0
    shifted_from_counts: Counter[tuple[str, str]] = Counter()
    shifted_to_counts: Counter[tuple[str, str]] = Counter()

    for row in rows:
        od = (row["origin_station_id"], row["destination_station_id"])
        new_row = dict(row)
        if od not in head_set:
            new_od = weighted_choice(head_ods, head_weights, rng)
            new_row["origin_station_id"] = new_od[0]
            new_row["destination_station_id"] = new_od[1]
            shifted_orders += 1
            shifted_from_counts[od] += 1
            shifted_to_counts[new_od] += 1
        transformed.append(new_row)

    top_head_summary = [
        {
            "origin_station_id": od[0],
            "destination_station_id": od[1],
            "original_count": od_counts[od],
        }
        for od in head_ods[:20]
    ]
    metadata = {
        "input_orders": len(rows),
        "output_orders": len(transformed),
        "retention_rate": 1.0 if rows else 0.0,
        "mode": "concentrate_head",
        "head_od_frac": head_od_frac,
        "head_weight_alpha": head_weight_alpha,
        "unique_od_pairs_before": unique_ods,
        "unique_od_pairs_after": len({(row["origin_station_id"], row["destination_station_id"]) for row in transformed}),
        "head_od_count": head_count,
        "shifted_orders": shifted_orders,
        "shifted_fraction": shifted_orders / len(rows) if rows else 0.0,
        "top_head_od_pairs": top_head_summary,
        "largest_shifted_from_od_pairs": [
            {"origin_station_id": od[0], "destination_station_id": od[1], "count": count}
            for od, count in shifted_from_counts.most_common(20)
        ],
        "largest_shifted_to_od_pairs": [
            {"origin_station_id": od[0], "destination_station_id": od[1], "count": count}
            for od, count in shifted_to_counts.most_common(20)
        ],
    }
    return transformed, metadata


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    order_file = input_dir / "order.csv"
    station_file = input_dir / "station.csv"
    segment_file = input_dir / "segment.csv"
    vehicle_file = input_dir / "vehicle.csv"

    for path in [order_file, station_file, segment_file, vehicle_file]:
        if not path.exists():
            raise FileNotFoundError(f"missing required file: {path}")

    rows = load_orders(order_file)
    fieldnames = list(rows[0].keys()) if rows else []
    if args.mode == "dropout":
        kept_rows, metadata = sparsify_orders(
            rows,
            base_keep_prob=args.base_keep_prob,
            station_bias_gamma=args.station_bias_gamma,
            min_station_keep=args.min_station_keep,
            od_drop_frac=args.od_drop_frac,
            bucket_drop_frac=args.bucket_drop_frac,
            bucket_mode=args.bucket_mode,
            rng=rng,
        )
    else:
        kept_rows, metadata = concentrate_head_orders(
            rows,
            head_od_frac=args.head_od_frac,
            head_weight_alpha=args.head_weight_alpha,
            rng=rng,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_orders(output_dir / "order.csv", kept_rows, fieldnames)
    shutil.copy(station_file, output_dir / "station.csv")
    shutil.copy(segment_file, output_dir / "segment.csv")
    shutil.copy(vehicle_file, output_dir / "vehicle.csv")

    metadata["input_dir"] = str(input_dir)
    metadata["output_dir"] = str(output_dir)
    metadata["seed"] = args.seed
    with (output_dir / "sparsify_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created sparse dataset: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Orders kept: {metadata['output_orders']} / {metadata['input_orders']} ({metadata['retention_rate']:.1%})")
    if args.mode == "dropout":
        print(f"Dropped OD pairs: {len(metadata['dropped_od_pairs'])}")
        print(f"Dropped buckets: {len(metadata['dropped_buckets'])}")
        print(f"Drop reasons: {metadata['drop_reasons']}")
    else:
        print(f"Head OD pairs retained: {metadata['head_od_count']} / {metadata['unique_od_pairs_before']}")
        print(f"Orders shifted onto head ODs: {metadata['shifted_orders']} ({metadata['shifted_fraction']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
