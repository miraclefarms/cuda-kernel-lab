#!/usr/bin/env python3
"""The series visual system — single source of truth for every figure.

Do not re-declare colors, fonts, ticks or hatches anywhere else. A kernel
directory that draws its own chart must import this module, otherwise its
figure silently drifts away from the rest of the series.

Style: Nature/Science journal conventions.

  * white background, no grid, no chartjunk
  * sans-serif (Nature requires it); thin 0.8 pt rules
  * inward major + minor ticks on all four sides, full box
  * Okabe-Ito colorblind-safe palette, plus a distinct hatch per series so the
    encoding survives greyscale print and color-vision deficiency. Series are
    never distinguished by color alone (Color Universal Design).

Usage:

    from plot_style import apply_style, variant_style, VARIANT_ORDER
    apply_style()
    styles = variant_style(["read", "copy", "write"])
    ax.bar(x, y, color=styles["read"]["color"], hatch=styles["read"]["hatch"])

This module is plotting-only: no data loading, no CLI. The canonical renderer
is tools/plot.py; see .agents/skills/figure-style/SKILL.md for the workflow.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

INK = "#1a1a1a"

# Okabe-Ito colorblind-safe qualitative palette.
# blue, vermilion, bluish green, orange, sky blue, reddish purple.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]

# Redundant hatch per series index: the first series stays solid so the primary
# result reads cleanly, later ones get progressively different patterns.
HATCHES = ["", "///", "...", "xxx", "|||", "+++"]

# Redundant marker + dash per series index for line charts (sweep view), for the
# same reason bars get hatches: never distinguish series by color alone.
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

# Stable series order, so a chart does not reshuffle when a row is added.
# New names are appended, never inserted, so existing figures keep their colors.
VARIANT_ORDER = ["read", "copy", "write", "baseline", "optimized",
                 # 02-measurement-discipline: wrong protocols first, the right one last
                 "wallclock", "no-flush", "batched", "disciplined",
                 # 03-smem-staging
                 "global", "sync-stage", "cp-async-wait", "cp-async"]


def apply_style() -> None:
    """Install the rcParams for the series. Call once before drawing."""
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


def series_style(index: int) -> dict:
    """Color + hatch (bars) + marker/linestyle (lines) for the index-th series."""
    return {
        "color": OKABE_ITO[index % len(OKABE_ITO)],
        "hatch": HATCHES[index % len(HATCHES)],
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
    }


def variant_style(variants: list[str]) -> dict[str, dict]:
    """Map each variant name to its stable color + hatch by declared order."""
    return {v: series_style(i) for i, v in enumerate(variants)}
