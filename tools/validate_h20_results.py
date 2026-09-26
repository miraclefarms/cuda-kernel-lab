#!/usr/bin/env python3
"""Validate one H20 run before committing or citing its CSV.

Usage: python3 tools/validate_h20_results.py results/NN-slug/YYYY-MM-DD-h20.csv
This checks provenance, quiet-window verdict, sweep completeness, and sidecars.
It cannot detect a co-tenant that starts and stops wholly during the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

EXPECTED = {
    "01-execution-model": {"read", "copy", "write"},
    "02-measurement-discipline": {"wallclock", "no-flush", "batched", "disciplined"},
    "03-smem-staging": {"global", "sync-stage", "cp-async-wait", "cp-async"},
}


def validate(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"CSV missing: {path}"]
    if pathlib.Path(str(path) + ".invalid").exists():
        errors.append("CSV has an .invalid marker")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return errors + ["CSV has no data rows"]

    kernel = rows[0].get("kernel", "")
    if kernel not in EXPECTED or path.parent.name != kernel:
        errors.append(f"unexpected kernel or directory: {kernel!r}")
    variants = {row.get("variant", "") for row in rows}
    if kernel in EXPECTED and variants != EXPECTED[kernel]:
        errors.append(f"variant set {sorted(variants)} != {sorted(EXPECTED[kernel])}")

    commits = {row.get("git_commit", "") for row in rows}
    if len(commits) != 1 or any(len(commit) != 40 for commit in commits):
        errors.append("missing or inconsistent 40-character git commit")
    for index, row in enumerate(rows, 2):
        for key, wanted in (("machine", "h20"), ("git_dirty", "no"), ("mig", "Disabled")):
            if row.get(key) != wanted:
                errors.append(f"row {index}: {key}={row.get(key)!r}, expected {wanted!r}")
        if "H20" not in row.get("gpu_name", ""):
            errors.append(f"row {index}: GPU is not H20")
        for key in ("driver", "toolkit", "sm_clock_mhz", "mem_clock_mhz"):
            if not row.get(key):
                errors.append(f"row {index}: {key} is empty")
        expected_lock = "no" if "unlocked" in path.stem else "yes"
        if row.get("clocks_locked") != expected_lock:
            errors.append(f"row {index}: clocks_locked should be {expected_lock}")
        try:
            lo, p10, median, p90, hi = (
                float(row[key]) for key in ("min_ms", "p10_ms", "median_ms", "p90_ms", "max_ms")
            )
            if not (0 < lo <= p10 <= median <= p90 <= hi):
                errors.append(f"row {index}: timing quantiles are out of order")
        except (KeyError, ValueError):
            errors.append(f"row {index}: timing quantiles are missing or invalid")

    keys = [(row.get("variant"), row.get("mode"), row.get("shape")) for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate variant/mode/shape rows")
    if kernel == "01-execution-model":
        if len(rows) != 6 or {row["mode"] for row in rows} != {"cold", "hot"}:
            errors.append("01 should contain read/copy/write in both cold and hot modes (6 rows)")
        occupancy = path.parent / "occupancy" / path.name
        if not occupancy.is_file():
            errors.append(f"occupancy scan missing: {occupancy}")
        else:
            with occupancy.open(newline="") as stream:
                occ_rows = list(csv.DictReader(stream))
            regs = {r.get("actual_regs_per_thread") for r in occ_rows}
            smem = {r.get("dynamic_smem_kib") for r in occ_rows}
            if len(regs) < 2 or len(smem) < 4:
                errors.append("occupancy scan needs >=2 actual register counts and >=4 SMEM sizes")
    elif kernel == "02-measurement-discipline":
        shapes = {row["shape"] for row in rows}
        if len(shapes) != 7 or len(rows) != 35:
            errors.append("02 default sweep should contain 7 L2-relative sizes x 5 protocol rows")
        samples = path.with_name(path.stem + "-samples.csv")
        if not samples.is_file():
            errors.append(f"raw samples missing: {samples}")
    elif kernel == "03-smem-staging":
        shapes = {row["shape"] for row in rows}
        if len(shapes) != 7 or len(rows) != 56:
            errors.append("03 default sweep should contain 7 shapes x 4 variants x 2 modes")

    report = path.with_suffix(".quiet.json")
    if not report.is_file():
        errors.append(f"quiet-window report missing: {report}")
    else:
        try:
            verdict = json.loads(report.read_text())
            if verdict.get("verdict") != "pass" or verdict.get("exit_code") != 0:
                errors.append(f"quiet-window verdict is {verdict.get('verdict')!r}")
            for phase in ("pre", "post"):
                if not verdict.get(phase) or verdict[phase].get("max_util_pct") != 0:
                    errors.append(f"{phase.upper()} quiet-window check did not show 0% utilization")
        except (ValueError, OSError) as exc:
            errors.append(f"quiet-window report unreadable: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=pathlib.Path)
    args = parser.parse_args()
    errors = validate(args.csv)
    for error in errors:
        print(f"FAIL {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"PASS {args.csv}")
    print("Interpretation still requires inspecting the sweep and its failure boundaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
