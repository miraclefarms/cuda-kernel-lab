#!/usr/bin/env python3
"""Turn result CSVs into the figures that go into an article.

Figures are generated from data, never drawn by hand. If a chart cannot be
produced by this script from a committed CSV, it does not belong in a post.

Two views:
  latency   — median ms with p10-p90 error bars. Always available.
  bandwidth — achieved GB/s against two reference lines: theoretical peak and
              the measured streaming ceiling from bench/machine-peaks.json.
              For a bandwidth-bound kernel this is the view that carries the
              argument; raw latency hides how much of the machine is left.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
PEAKS_PATH = REPO / "bench" / "machine-peaks.json"

# Matches the article visual system: slate blue, muted teal, charcoal on warm paper.
COLORS = {"baseline": "#6b7a99", "optimized": "#3f8f8a"}
FALLBACK = "#a8846b"
PAPER = "#faf7f2"
INK = "#3a3a3a"


def load(paths: list[pathlib.Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in paths:
        with p.open() as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def load_peaks() -> dict:
    try:
        return {k: v for k, v in json.loads(PEAKS_PATH.read_text()).items()
                if not k.startswith("_")}
    except (OSError, json.JSONDecodeError):
        print(f"[warn] could not read {PEAKS_PATH}; bandwidth view will have no reference lines")
        return {}


def group(rows: list[dict[str, str]], mode: str, field: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r.get("mode") != mode:
            continue
        try:
            value = float(r[field])
        except (KeyError, ValueError):
            continue
        out[r["machine"]][r["variant"]] = {
            "value": value,
            "median": float(r["median_ms"]),
            "p10": float(r["p10_ms"]),
            "p90": float(r["p90_ms"]),
        }
    return out


def new_axes(title: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def bar_positions(n_machines: int, n_variants: int, i: int, j: int) -> tuple[float, float]:
    width = 0.7 / max(n_variants, 1)
    return j + (i - (n_variants - 1) / 2) * width, width * 0.88


def plot_latency(rows, mode: str, out: pathlib.Path) -> None:
    data = group(rows, mode, "median_ms")
    if not data:
        return
    machines = sorted(data)
    variants = sorted({v for m in data.values() for v in m})

    fig, ax = new_axes(f"{rows[0].get('kernel', '')} — {mode} — latency")
    for i, variant in enumerate(variants):
        xs, ys, lo, hi, w = [], [], [], [], 0.0
        for j, machine in enumerate(machines):
            d = data[machine].get(variant)
            if not d:
                continue
            x, w = bar_positions(len(machines), len(variants), i, j)
            xs.append(x)
            ys.append(d["median"])
            lo.append(max(d["median"] - d["p10"], 0.0))
            hi.append(max(d["p90"] - d["median"], 0.0))
        ax.bar(xs, ys, width=w, label=variant, color=COLORS.get(variant, FALLBACK),
               yerr=[lo, hi], capsize=3, error_kw={"ecolor": INK, "elinewidth": 0.9})

    ax.set_xticks(range(len(machines)))
    ax.set_xticklabels(machines)
    ax.set_xlim(-0.6, len(machines) - 0.4)
    ax.set_ylabel("median latency (ms), error bars = p10–p90")
    ax.legend(frameon=False)
    save(fig, out)


def plot_bandwidth(rows, mode: str, out: pathlib.Path, peaks: dict) -> None:
    data = group(rows, mode, "achieved_gbps")
    data = {m: {v: d for v, d in vs.items() if d["value"] > 0} for m, vs in data.items()}
    data = {m: vs for m, vs in data.items() if vs}
    if not data:
        return
    machines = sorted(data)
    variants = sorted({v for m in data.values() for v in m})
    ceiling_key = f"bw_ceiling_{mode}_gbps"

    fig, ax = new_axes(f"{rows[0].get('kernel', '')} — {mode} — achieved bandwidth")
    for i, variant in enumerate(variants):
        for j, machine in enumerate(machines):
            d = data[machine].get(variant)
            if not d:
                continue
            x, w = bar_positions(len(machines), len(variants), i, j)
            ax.bar([x], [d["value"]], width=w,
                   label=variant if j == 0 else None,
                   color=COLORS.get(variant, FALLBACK))
            theo = (peaks.get(machine) or {}).get("bw_theoretical_gbps")
            if theo:
                ax.text(x, d["value"], f" {100 * d['value'] / theo:.0f}%", ha="center",
                        va="bottom", fontsize=8, color=INK)

    # Reference lines drawn per machine, spanning only that machine's group —
    # a single axis-wide line would be wrong the moment two machines differ.
    for j, machine in enumerate(machines):
        peak = peaks.get(machine) or {}
        for key, style, label in ((("bw_theoretical_gbps"), "--", "theoretical peak"),
                                  ((ceiling_key), ":", f"streaming ceiling ({mode})")):
            value = peak.get(key)
            if not value:
                continue
            ax.hlines(value, j - 0.45, j + 0.45, colors=INK, linestyles=style, linewidth=1.1,
                      label=label if j == 0 else None)

    ax.set_xticks(range(len(machines)))
    ax.set_xticklabels(machines)
    ax.set_xlim(-0.6, len(machines) - 0.4)
    ax.set_ylabel("achieved GB/s (label = % of theoretical peak)")
    # Headroom so the reference lines never sit under the legend or the frame.
    top = max([d["value"] for vs in data.values() for d in vs.values()]
              + [(peaks.get(m) or {}).get("bw_theoretical_gbps") or 0 for m in machines])
    ax.set_ylim(0, top * 1.18)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out)


def save(fig, out: pathlib.Path) -> None:
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=pathlib.Path)
    ap.add_argument("-o", "--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--view", choices=["latency", "bandwidth", "both"], default="both")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print("no rows")
        return 1
    peaks = load_peaks()
    kernel = rows[0].get("kernel", "kernel")

    missing = sorted({r["machine"] for r in rows} - set(peaks))
    if missing:
        print(f"[warn] no entry in machine-peaks.json for: {', '.join(missing)}")

    for mode in ("cold", "hot"):
        if args.view in ("latency", "both"):
            plot_latency(rows, mode, args.out_dir / f"fig-{kernel}-{mode}-latency.png")
        if args.view in ("bandwidth", "both"):
            plot_bandwidth(rows, mode, args.out_dir / f"fig-{kernel}-{mode}-bandwidth.png", peaks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
