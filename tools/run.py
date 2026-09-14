#!/usr/bin/env python3
"""Run a kernel benchmark and write a CSV that carries its own environment.

The binary reports only what it measured. Everything about the machine it ran on
is captured here, at run time, from nvidia-smi and git — never typed in by hand,
because a number whose environment was filled in from memory is not reproducible.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

ENV_FIELDS = [
    "timestamp",
    "machine",
    "gpu_name",
    "driver",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "compute_mode",
    "ecc",
    "mig",
    "clocks_locked",
    "toolkit",
    "git_commit",
    "git_dirty",
]

DERIVED_FIELDS = ["achieved_gbps", "achieved_tflops"]


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def capture_env(machine: str, clocks_locked: bool) -> dict[str, str]:
    query = "name,driver_version,clocks.sm,clocks.mem,compute_mode,ecc.mode.current,mig.mode.current"
    raw = sh(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    parts = [p.strip() for p in raw.split(",")] if raw else [""] * 7
    parts += [""] * (7 - len(parts))

    nvcc = sh(["nvcc", "--version"])
    toolkit = ""
    for line in nvcc.splitlines():
        if "release" in line:
            toolkit = line.split("release")[-1].strip().split(",")[0].strip()
    return {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": machine,
        "gpu_name": parts[0],
        "driver": parts[1],
        "sm_clock_mhz": parts[2],
        "mem_clock_mhz": parts[3],
        "compute_mode": parts[4],
        "ecc": parts[5],
        "mig": parts[6],
        "clocks_locked": "yes" if clocks_locked else "no",
        "toolkit": toolkit,
        "git_commit": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_dirty": "yes" if sh(["git", "-C", str(REPO), "status", "--porcelain"]) else "no",
    }


def derive(row: dict[str, str]) -> dict[str, str]:
    try:
        median_s = float(row["median_ms"]) / 1e3
        if median_s <= 0:
            raise ValueError
        gbps = float(row["bytes"]) / median_s / 1e9 if float(row["bytes"]) else 0.0
        tflops = float(row["flops"]) / median_s / 1e12 if float(row["flops"]) else 0.0
    except (KeyError, ValueError):
        gbps, tflops = 0.0, 0.0
    return {"achieved_gbps": f"{gbps:.3f}", "achieved_tflops": f"{tflops:.4f}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="kernel directory name, e.g. 00-template")
    ap.add_argument("--machine", required=True, help="machine codename: a100 | h20 | h200")
    ap.add_argument("--build", default="build", help="cmake build directory")
    ap.add_argument("--clocks-locked", action="store_true",
                    help="pass only if you actually ran nvidia-smi -lgc")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="args forwarded to the binary")
    args = ap.parse_args()

    binary = REPO / args.build / "kernels" / args.kernel / f"kernel-{args.kernel}"
    if not binary.exists():
        print(f"binary not found: {binary}\nbuild it first:\n"
              f"  cmake -B {args.build} -DCMAKE_CUDA_ARCHITECTURES=90a && "
              f"cmake --build {args.build} --target kernel-{args.kernel} -j", file=sys.stderr)
        return 1

    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    proc = subprocess.run([str(binary), *forwarded], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    env = capture_env(args.machine, args.clocks_locked)
    if not args.clocks_locked:
        print("[warn] clocks not locked — report relative ratios, not absolute TFLOPS",
              file=sys.stderr)

    rows = list(csv.DictReader(io.StringIO(proc.stdout)))
    if not rows:
        print("[error] binary produced no CSV rows", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else REPO / "results" / args.kernel
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dt.date.today().isoformat()}-{args.machine}.csv"

    fieldnames = ENV_FIELDS + list(rows[0].keys()) + DERIVED_FIELDS
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**env, **row, **derive(row)})

    print(f"wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
