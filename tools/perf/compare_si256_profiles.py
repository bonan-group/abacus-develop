#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


TIMER_KEYS = (
    ("total_s", None, "total"),
    ("potxc_cal_veff_s", "PotXC", "cal_veff"),
    ("xc_functional_v_xc_s", "XC_Functional", "v_xc"),
    ("xc_resident_gpu_s", "XC_Functional", "v_xc_resident_gpu"),
    ("forces_cal_force_scc_s", "Forces", "cal_force_scc"),
    ("forces_cal_force_s", "Forces", "cal_force"),
)


def parse_timer_table(stdout_path):
    values = {key: None for key, _klass, _name in TIMER_KEYS}
    if not stdout_path.exists():
        return values

    timer_line = re.compile(
        r"^\s*(?:(?P<class>\S+)\s+)?(?P<name>\S+(?:\s+\S+)*)\s+"
        r"(?P<time>[0-9]+(?:\.[0-9]+)?)\s+"
        r"(?P<calls>[0-9]+)\s+"
        r"(?P<avg>[0-9]+(?:\.[0-9]+)?)\s+"
        r"(?P<percent>[0-9]+(?:\.[0-9]+)?)\s*$"
    )
    for raw_line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = timer_line.match(raw_line)
        if not match:
            continue
        klass = match.group("class")
        name = " ".join(match.group("name").split())
        for key, wanted_class, wanted_name in TIMER_KEYS:
            if wanted_class == klass and wanted_name == name:
                values[key] = float(match.group("time"))
    return values


def parse_gaps(profile_dir):
    gaps_path = profile_dir / "gaps.json"
    result = {
        "inactive_gap_s": None,
        "largest_gap_s": None,
        "cuda_utilization_percent": None,
        "gap_count": None,
    }
    if not gaps_path.exists():
        return result

    data = json.loads(gaps_path.read_text(encoding="utf-8"))
    gaps = data.get("gaps", [])
    result["inactive_gap_s"] = data.get("inactive_s")
    result["largest_gap_s"] = gaps[0]["duration_s"] if gaps else 0.0
    result["cuda_utilization_percent"] = data.get("utilization_percent")
    result["gap_count"] = data.get("gap_count")
    return result


def parse_gpu_usage(profile_dir):
    usage_path = profile_dir / "gpu_usage.csv"
    result = {"gpu_util_avg_percent": None, "gpu_util_max_percent": None}
    if not usage_path.exists():
        return result

    utils = []
    with usage_path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = {key.strip(): value for key, value in row.items() if key is not None}
            raw = normalized.get("utilization.gpu [%]")
            if raw is None:
                continue
            try:
                utils.append(float(raw.replace("%", "").strip()))
            except ValueError:
                pass

    if utils:
        result["gpu_util_avg_percent"] = sum(utils) / len(utils)
        result["gpu_util_max_percent"] = max(utils)
    return result


def parse_profile(label, directory):
    profile_dir = Path(directory)
    row = {"label": label, "path": str(profile_dir)}
    row.update(parse_timer_table(profile_dir / "abacus_stdout.log"))
    row.update(parse_gaps(profile_dir))
    row.update(parse_gpu_usage(profile_dir))
    return row


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_markdown(rows):
    columns = [
        ("label", "Profile"),
        ("total_s", "Total s"),
        ("potxc_cal_veff_s", "PotXC s"),
        ("xc_functional_v_xc_s", "v_xc s"),
        ("xc_resident_gpu_s", "resident s"),
        ("forces_cal_force_scc_s", "SCC force s"),
        ("inactive_gap_s", "Inactive s"),
        ("largest_gap_s", "Largest gap s"),
        ("cuda_utilization_percent", "CUDA util %"),
        ("gpu_util_avg_percent", "nvidia-smi avg %"),
        ("gpu_util_max_percent", "nvidia-smi max %"),
    ]
    print("| " + " | ".join(title for _key, title in columns) + " |")
    print("| " + " | ".join("---" for _key, _title in columns) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(key)) for key, _title in columns) + " |")


def main():
    parser = argparse.ArgumentParser(description="Compare Si256 ABACUS/Nsight profile directories.")
    parser.add_argument(
        "profiles",
        nargs="+",
        help="Profile entries in LABEL=DIR form.",
    )
    parser.add_argument("--csv", dest="csv_path", help="Optional CSV output path.")
    args = parser.parse_args()

    rows = []
    for entry in args.profiles:
        if "=" not in entry:
            raise SystemExit(f"profile entry must be LABEL=DIR: {entry}")
        label, directory = entry.split("=", 1)
        rows.append(parse_profile(label, directory))

    print_markdown(rows)

    if args.csv_path:
        fieldnames = list(rows[0].keys()) if rows else []
        with open(args.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
