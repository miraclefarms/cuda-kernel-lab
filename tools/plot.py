#!/usr/bin/env python3
"""Turn result CSVs into the figures that go into an article.

Figures are generated from data, never drawn by hand. If a chart cannot be
produced by this script from a committed CSV, it does not belong in a post.

Three views:
  latency   — median ms with p10-p90 error bars. Always available.
  bandwidth — achieved GB/s against two reference lines: theoretical peak and
              the measured streaming ceiling from bench/machine-peaks.json.
              For a bandwidth-bound kernel this is the view that carries the
              argument; raw latency hides how much of the machine is left.
  sweep     — for kernels that emit several shapes per variant (a parameter
              sweep). One figure per mode, one panel per machine, x = shape,
              one line per variant. --metric gbps (with reference lines) or
              speedup (relative to --baseline, same machine/mode/shape — the
              form to use when clocks were not locked). The bar views cannot
              show a sweep: they would keep one arbitrary shape per variant, so
              the script switches to this view automatically when it sees one.

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

The palette, fonts and rcParams live in tools/plot_style.py — do not redeclare
them here. Workflow and rules: .agents/skills/figure-style/SKILL.md.
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

from plot_style import INK, VARIANT_ORDER, apply_style, variant_style  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
PEAKS_PATH = REPO / "bench" / "machine-peaks.json"


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


def parse_shape(shape: str) -> dict[str, str]:
    out = {}
    for tok in shape.split(";"):
        k, _, v = tok.partition("=")
        out[k] = v
    return out


def is_sweep(rows) -> bool:
    shapes: dict[tuple, set] = defaultdict(set)
    for r in rows:
        shapes[(r["machine"], r["variant"], r["mode"])].add(r["shape"])
    return any(len(v) > 1 for v in shapes.values())


def shape_labels(rows, keys: list[str] | None) -> dict[str, str]:
    """Short x tick labels: only the shape tokens that actually vary."""
    shapes = list(dict.fromkeys(r["shape"] for r in rows))
    parsed = {s: parse_shape(s) for s in shapes}
    if not keys:
        all_keys = list(dict.fromkeys(k for p in parsed.values() for k in p))
        keys = [k for k in all_keys if len({p.get(k) for p in parsed.values()}) > 1]
    return {s: "\n".join(f"{k}={parsed[s].get(k, '')}" for k in keys) or s for s in shapes}


def plot_sweep(rows, mode: str, out: pathlib.Path, peaks: dict, metric: str,
               baseline: str | None, keys: list[str] | None) -> None:
    rows = [r for r in rows if r.get("mode") == mode]
    if not rows:
        return
    machines = sorted({r["machine"] for r in rows})
    shapes = list(dict.fromkeys(r["shape"] for r in rows))
    labels = shape_labels(rows, keys)
    variants = ordered_variants({"_": {r["variant"]: None for r in rows}})
    if metric == "speedup":
        if not baseline or baseline not in variants:
            print(f"[warn] sweep speedup needs --baseline present in the data; skipping {mode}")
            return
    styles = variant_style(variants)
    by = {(r["machine"], r["variant"], r["shape"]): r for r in rows}

    fig, axes = plt.subplots(1, len(machines), figsize=(3.6 * len(machines) + 1.6, 3.8),
                             dpi=200, squeeze=False)
    for ax, machine in zip(axes[0], machines):
        for variant in variants:
            xs, ys, lo, hi = [], [], [], []
            for i, shape in enumerate(shapes):
                r = by.get((machine, variant, shape))
                if not r:
                    continue
                med, p10, p90 = (float(r[k]) for k in ("median_ms", "p10_ms", "p90_ms"))
                if med <= 0:
                    continue
                if metric == "gbps":
                    b = float(r["bytes"])
                    if b <= 0:
                        continue
                    # bytes / time: the p90 latency is the low end of the bandwidth band.
                    y, ylo, yhi = (b / (t / 1e3) / 1e9 for t in (med, p90, p10))
                else:
                    ref = by.get((machine, baseline, shape))
                    if not ref:
                        continue
                    base = float(ref["median_ms"])
                    y, ylo, yhi = base / med, base / p90, base / p10
                xs.append(i)
                ys.append(y)
                lo.append(max(y - ylo, 0.0))
                hi.append(max(yhi - y, 0.0))
            if not xs:
                continue
            st = styles[variant]
            ax.errorbar(xs, ys, yerr=[lo, hi], label=variant, color=st["color"],
                        marker=st["marker"], linestyle=st["linestyle"], markersize=3.5,
                        linewidth=1.0, capsize=1.5, elinewidth=0.6, zorder=3)
        if metric == "gbps":
            peak = peaks.get(machine) or {}
            for key, ls, label in (("bw_theoretical_gbps", "--", "theoretical peak"),
                                   (f"bw_ceiling_{mode}_gbps", ":", f"streaming ceiling ({mode})")):
                if peak.get(key):
                    ax.axhline(peak[key], color=INK, linestyle=ls, linewidth=0.9, label=label,
                               zorder=2)
            ax.set_ylim(bottom=0)
        else:
            ax.axhline(1.0, color=INK, linestyle="--", linewidth=0.9, zorder=2,
                       label=f"{baseline} = 1")
        ax.set_title(machine)
        ax.set_xticks(range(len(shapes)))
        ax.set_xticklabels([labels[s] for s in shapes], fontsize=6, rotation=0)
        ax.minorticks_off()
        ax.set_xlim(-0.5, len(shapes) - 0.5)
    axes[0][0].set_ylabel("Achieved bandwidth (GB/s)" if metric == "gbps"
                          else f"Speedup vs {baseline} (median)")
    kernel = rows[0].get("kernel", "")
    fig.suptitle(f"{kernel} — {mode} — sweep", fontsize=10, fontweight="semibold")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0,
                       handlelength=2.2, labelspacing=0.7)
    fig.text(0.99, 0.0, "error bars: p10–p90", ha="right", va="bottom", fontsize=6.5,
             color="#666666")
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
    ap.add_argument("--view", choices=["latency", "bandwidth", "both", "sweep"], default="both")
    ap.add_argument("--metric", choices=["gbps", "speedup"], default="gbps",
                    help="sweep view only")
    ap.add_argument("--baseline", default=None, help="sweep speedup: reference variant")
    ap.add_argument("--x-keys", default=None,
                    help="sweep view: comma-separated shape keys for x labels "
                         "(default: the keys that vary)")
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

    view = args.view
    if view != "sweep" and is_sweep(rows):
        print("[warn] several shapes per variant: bar views would drop all but one; "
              "using --view sweep")
        view = "sweep"
    keys = args.x_keys.split(",") if args.x_keys else None

    for mode in ("cold", "hot"):
        if view == "sweep":
            suffix = "sweep" if args.metric == "gbps" else f"speedup-vs-{args.baseline}"
            plot_sweep(rows, mode, args.out_dir / f"fig-{kernel}-{mode}-{suffix}.png", peaks,
                       args.metric, args.baseline, keys)
            continue
        if view in ("latency", "both"):
            plot_latency(rows, mode, args.out_dir / f"fig-{kernel}-{mode}-latency.png")
        if view in ("bandwidth", "both"):
            plot_bandwidth(rows, mode, args.out_dir / f"fig-{kernel}-{mode}-bandwidth.png", peaks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
