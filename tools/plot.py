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

Style: Nature/Science journal conventions, applied uniformly so every figure
in the series reads as one visual system.

  * white background, no grid, no chartjunk
  * sans-serif (Nature requires it); thin 0.8 pt rules
  * inward major + minor ticks on all four sides, full box
  * Okabe-Ito colorblind-safe palette, with a distinct hatch per series so the
    encoding survives greyscale print and color-vision deficiency. Series are
    never distinguished by color alone (Color Universal Design).
  * value labels printed directly on the data; legend moved outside the axes
    so it can never overlap the bars or the reference lines.
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

INK = "#1a1a1a"

# Okabe-Ito colorblind-safe qualitative palette.
# blue, vermilion, bluish green, orange, sky blue, reddish purple.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]

# Redundant hatch per series index: the first series stays solid so the primary
# result reads cleanly, later ones get progressively different patterns.
HATCHES = ["", "///", "...", "xxx", "|||", "+++"]

# Stable series order, so a chart does not reshuffle when a row is added.
VARIANT_ORDER = ["read", "copy", "write", "baseline", "optimized"]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "hatch.linewidth": 0.6,
    })


def variant_style(variants: list[str]) -> dict[str, dict[str, str]]:
    return {
        v: {"color": OKABE_ITO[i % len(OKABE_ITO)], "hatch": HATCHES[i % len(HATCHES)]}
        for i, v in enumerate(variants)
    }


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


def ordered_variants(data: dict[str, dict[str, dict]]) -> list[str]:
    seen = {v for machine in data.values() for v in machine}
    ordered = [v for v in VARIANT_ORDER if v in seen]
    return ordered + sorted(seen - set(ordered))


def new_axes(title: str):
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=200)
    ax.set_title(title, pad=10)
    return fig, ax


def finish_legend(ax) -> None:
    # Outside the axes on the right: cannot collide with bars, value labels or
    # reference lines regardless of how tall the data gets.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0,
              handlelength=1.5, labelspacing=0.7)


def bar_positions(n_machines: int, n_variants: int, i: int, j: int) -> tuple[float, float]:
    width = 0.7 / max(n_variants, 1)
    return j + (i - (n_variants - 1) / 2) * width, width * 0.88


def plot_latency(rows, mode: str, out: pathlib.Path) -> None:
    data = group(rows, mode, "median_ms")
    if not data:
        return
    machines = sorted(data)
    variants = ordered_variants(data)
    styles = variant_style(variants)

    fig, ax = new_axes(f"{rows[0].get('kernel', '')} — {mode} — latency")
    for i, variant in enumerate(variants):
        xs, med, lo, hi, w = [], [], [], [], 0.0
        for j, machine in enumerate(machines):
            d = data[machine].get(variant)
            if not d:
                continue
            x, w = bar_positions(len(machines), len(variants), i, j)
            xs.append(x)
            med.append(d["median"])
            lo.append(max(d["median"] - d["p10"], 0.0))
            hi.append(max(d["p90"] - d["median"], 0.0))
        style = styles[variant]
        ax.bar(xs, med, width=w, label=variant, color=style["color"],
               hatch=style["hatch"], edgecolor=INK, linewidth=0.6, zorder=3,
               yerr=[lo, hi], capsize=2,
               error_kw={"ecolor": INK, "elinewidth": 0.8})
        for x, m, h in zip(xs, med, hi):
            ax.annotate(f"{m:.3f}", (x, m + h), textcoords="offset points",
                        xytext=(0, 2), ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(range(len(machines)))
    ax.set_xticklabels(machines)
    ax.set_xlim(-0.6, len(machines) - 0.4)
    ax.set_ylabel("Median latency (ms)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.text(0.99, -0.13, "error bars: p10–p90", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color="#666666")
    finish_legend(ax)
    save(fig, out)


def plot_bandwidth(rows, mode: str, out: pathlib.Path, peaks: dict) -> None:
    data = group(rows, mode, "achieved_gbps")
    data = {m: {v: d for v, d in vs.items() if d["value"] > 0} for m, vs in data.items()}
    data = {m: vs for m, vs in data.items() if vs}
    if not data:
        return
    machines = sorted(data)
    variants = ordered_variants(data)
    styles = variant_style(variants)
    ceiling_key = f"bw_ceiling_{mode}_gbps"

    fig, ax = new_axes(f"{rows[0].get('kernel', '')} — {mode} — achieved bandwidth")
    for i, variant in enumerate(variants):
        for j, machine in enumerate(machines):
            d = data[machine].get(variant)
            if not d:
                continue
            x, w = bar_positions(len(machines), len(variants), i, j)
            style = styles[variant]
            ax.bar([x], [d["value"]], width=w,
                   label=variant if j == 0 else None,
                   color=style["color"], hatch=style["hatch"],
                   edgecolor=INK, linewidth=0.6, zorder=3)
            theo = (peaks.get(machine) or {}).get("bw_theoretical_gbps")
            if theo:
                ax.annotate(f"{100 * d['value'] / theo:.0f}%", (x, d["value"]),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", va="bottom", fontsize=6.5)

    # Reference lines drawn per machine, spanning only that machine's group —
    # a single axis-wide line would be wrong the moment two machines differ.
    for j, machine in enumerate(machines):
        peak = peaks.get(machine) or {}
        for key, style, label in ((("bw_theoretical_gbps"), "--", "theoretical peak"),
                                  ((ceiling_key), ":", f"streaming ceiling ({mode})")):
            value = peak.get(key)
            if not value:
                continue
            ax.hlines(value, j - 0.45, j + 0.45, colors=INK, linestyles=style,
                      linewidth=1.0, zorder=4, label=label if j == 0 else None)

    ax.set_xticks(range(len(machines)))
    ax.set_xticklabels(machines)
    ax.set_xlim(-0.6, len(machines) - 0.4)
    ax.set_ylabel("Achieved bandwidth (GB/s)")
    # Headroom so the reference lines and the % labels never touch the frame.
    top = max([d["value"] for vs in data.values() for d in vs.values()]
              + [(peaks.get(m) or {}).get("bw_theoretical_gbps") or 0 for m in machines])
    ax.set_ylim(0, top * 1.12)
    finish_legend(ax)
    save(fig, out)


def save(fig, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=pathlib.Path)
    ap.add_argument("-o", "--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--view", choices=["latency", "bandwidth", "both"], default="both")
    args = ap.parse_args()

    apply_style()
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
