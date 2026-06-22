#!/usr/bin/env python3
import argparse
import json
import sqlite3


def tables(conn):
    return {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}


def cuda_intervals(conn):
    names = tables(conn)
    intervals = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table not in names:
            continue
        cols = {row[1] for row in conn.execute(f"pragma table_info({table})")}
        if {"start", "end"} <= cols:
            intervals.extend((int(start), int(end), table) for start, end in conn.execute(f"select start,end from {table}"))
    return sorted((start, end, kind) for start, end, kind in intervals if end > start)


def merge(intervals):
    merged = []
    for start, end, _kind in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--min-gap-ms", type=float, default=10.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(args.sqlite)
    intervals = cuda_intervals(conn)
    if not intervals:
        raise SystemExit("no CUDA activity intervals found")
    merged = merge(intervals)
    span_start, span_end = merged[0][0], merged[-1][1]
    active_ns = sum(end - start for start, end in merged)
    min_gap_ns = int(args.min_gap_ms * 1e6)
    gaps = []
    for prev, cur in zip(merged, merged[1:]):
        gap = cur[0] - prev[1]
        if gap >= min_gap_ns:
            gaps.append({"start_s": prev[1] / 1e9, "end_s": cur[0] / 1e9, "duration_s": gap / 1e9})

    result = {
        "span_s": (span_end - span_start) / 1e9,
        "active_s": active_ns / 1e9,
        "inactive_s": ((span_end - span_start) - active_ns) / 1e9,
        "utilization_percent": 100.0 * active_ns / (span_end - span_start),
        "gap_count": len(gaps),
        "gaps": sorted(gaps, key=lambda item: item["duration_s"], reverse=True),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
