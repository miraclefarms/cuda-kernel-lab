#!/usr/bin/env python3
"""Turn result CSVs into the figures that go into an article.

Figures are generated from data, never drawn by hand. If a chart cannot be
produced by this script from a committed CSV, it does not belong in a post.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Matches the article visual system: slate blue, muted teal, charcoal on warm paper.
COLORS = {"baseline": "#6b7a99", "optimized": "#3f8f8a"}
PAPER = "#faf7f2"


def load(paths: list[pathlib.Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in paths:
        with p.open() as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def plot_mode(rows: list[dict[str, str]], mode: str, out: pathlib.Path) -> None:
    by_machine: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for r in rows:
        if r.get("mode") != mode:
            continue
        by_machine[r["machine"]][r["variant"]] = {
            "median": float(r["median_ms"]),
            "p10": float(r["p10_ms"]),
            "p90": float(r["p90_ms"]),
            "gbps": float(r.get("achieved_gbps") or 0.0),
        }
    if not by_machine:
        return

    machines = sorted(by_machine)
    variants = sorted({v for m in by_machine.values() for v in m})
    width = 0.8 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    for i, variant in enumerate(variants):
        xs, ys, lo, hi = [], [], [], []
        for j, machine in enumerate(machines):
            d = by_machine[machine].get(variant)
            if not d:
                continue
            xs.append(j + i * width - 0.4 + width / 2)
            ys.append(d["median"])
            lo.append(max(d["median"] - d["p10"], 0.0))
            hi.append(max(d["p90"] - d["median"], 0.0))
        ax.bar(xs, ys, width=width * 0.9, label=variant,
               color=COLORS.get(variant, "#a8846b"), yerr=[lo, hi], capsize=3,
               error_kw={"ecolor": "#3a3a3a", "elinewidth": 0.9})

    ax.set_xticks(range(len(machines)))
    ax.set_xticklabels(machines)
    ax.set_ylabel("median latency (ms), error bars = p10–p90")
    ax.set_title(f"{rows[0].get('kernel', '')} — {mode}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=pathlib.Path)
    ap.add_argument("-o", "--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print("no rows")
        return 1
    kernel = rows[0].get("kernel", "kernel")
    for mode in ("cold", "hot"):
        plot_mode(rows, mode, args.out_dir / f"fig-{kernel}-{mode}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
