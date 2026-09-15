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
import os
import pathlib
import subprocess
import sys
import threading

REPO = pathlib.Path(__file__).resolve().parent.parent

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    from cuda_compat import find_in_ld_library_path
except Exception:  # keep run.py usable even if the helper is absent
    def find_in_ld_library_path() -> tuple[str, str]:
        return "", ""

ENV_FIELDS = [
    "timestamp",
    "machine",
    "gpu_name",
    "driver",
    "cuda_compat",
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


def gpu_index() -> str:
    """Physical GPU index used by the benchmark's logical device 0.

    The binary calls cudaSetDevice(0); under CUDA_VISIBLE_DEVICES that maps to the
    first visible physical GPU, not necessarily physical 0. Querying with -i is what
    keeps a multi-GPU host from folding every other card into one CSV field.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return visible if visible.isdigit() else "0"


def nvidia_query(query: str) -> str:
    return sh(["nvidia-smi", "-i", gpu_index(), f"--query-gpu={query}",
               "--format=csv,noheader,nounits"])


class ClockSampler:
    """Poll the SM clock while the benchmark runs and keep its peak.

    A clock read after the process exits reports the idle value, which is not the
    clock the kernel ran at — the whole point of recording it. Sampling has to
    happen concurrently with the run.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []

    def __enter__(self) -> "ClockSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = nvidia_query("clocks.sm")
            if value.isdigit():
                self.samples.append(int(value))
            self._stop.wait(self._interval)

    def peak(self) -> str:
        return str(max(self.samples)) if self.samples else ""


def capture_env(machine: str, clocks_locked: bool, sm_clock: str = "") -> dict[str, str]:
    query = "name,driver_version,clocks.sm,clocks.mem,compute_mode,ecc.mode.current,mig.mode.current"
    raw = nvidia_query(query).splitlines()
    parts = [p.strip() for p in raw[0].split(",")] if raw else [""] * 7
    parts += [""] * (7 - len(parts))

    nvcc = sh(["nvcc", "--version"])
    toolkit = ""
    for line in nvcc.splitlines():
        if "release" in line:
            toolkit = line.split("release")[-1].strip().split(",")[0].strip()
    compat_dir, _ = find_in_ld_library_path()
    return {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": machine,
        "gpu_name": parts[0],
        "driver": parts[1],
        "cuda_compat": compat_dir,
        "sm_clock_mhz": sm_clock or parts[2],
        "mem_clock_mhz": parts[3],
        "compute_mode": parts[4],
        "ecc": parts[5],
        "mig": parts[6],
        "clocks_locked": "yes" if clocks_locked else "no",
        "toolkit": toolkit,
        "git_commit": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
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


def git_dirty() -> str:
    """Dirtiness of the source tree, ignoring benchmark outputs.

    The results/ and figures/ directories are written by this pipeline, so their
    presence must not mark a run dirty — otherwise no run could ever be clean.
    A modified or untracked source file still does.
    """
    out = sh(["git", "-C", str(REPO), "status", "--porcelain"])
    dirty = [line for line in out.splitlines()
             if not line.startswith("?? results/") and not line.startswith("?? figures/")]
    return "yes" if dirty else "no"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="kernel directory name, e.g. 00-template")
    ap.add_argument("--machine", required=True, help="machine codename: a100 | h20 | h200")
    ap.add_argument("--build", default="build", help="cmake build directory")
    ap.add_argument("--binary", default=None,
                    help="explicit binary path (repo-relative ok); overrides the "
                         "default build/kernels/<kernel>/kernel-<kernel>")
    ap.add_argument("--clocks-locked", action="store_true",
                    help="pass only if you actually ran nvidia-smi -lgc")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None,
                    help="suffix for a second run on the same day and machine, e.g. "
                         "'unlocked' -> {date}-{machine}-unlocked.csv")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="args forwarded to the binary")
    args = ap.parse_args()

    if args.binary:
        binary = pathlib.Path(args.binary)
        if not binary.is_absolute():
            binary = REPO / binary
    else:
        binary = REPO / args.build / "kernels" / args.kernel / f"kernel-{args.kernel}"
    if not binary.exists():
        print(f"binary not found: {binary}\nbuild it first:\n"
              f"  cmake -B {args.build} -DCMAKE_CUDA_ARCHITECTURES=90a && "
              f"cmake --build {args.build} --target kernel-{args.kernel} -j", file=sys.stderr)
        return 1

    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    with ClockSampler() as sampler:
        proc = subprocess.run([str(binary), *forwarded], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    env = capture_env(args.machine, args.clocks_locked, sampler.peak())
    if not args.clocks_locked:
        print("[warn] clocks not locked — report relative ratios, not absolute TFLOPS",
              file=sys.stderr)

    rows = list(csv.DictReader(io.StringIO(proc.stdout)))
    if not rows:
        print("[error] binary produced no CSV rows", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else REPO / "results" / args.kernel
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.date.today().isoformat()}-{args.machine}"
    out_path = out_dir / (f"{stem}-{args.tag}.csv" if args.tag else f"{stem}.csv")

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
