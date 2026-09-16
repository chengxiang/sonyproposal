#!/usr/bin/env python3
"""Compact, print-ready versions of the proposal's four existing figures.

Run from any directory. All PDFs are exactly 7 inches wide, with labels of
at least 10 points. Schematics remain illustrative; quantitative values and
image crops are copied unchanged from the documented original figures.
Detailed experimental qualifications belong in the proposal captions.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
QA = ROOT.parent / "latex_figure_qa"
QA.mkdir(exist_ok=True)
INK, MUTED = "#203342", "#536575"
BLUE, TEAL, ORANGE = "#2865A6", "#167E83", "#C8782F"
GRAY, RULE = "#75879A", "#C9D3DC"
PB, PT, PO = "#EEF4FA", "#EDF7F5", "#FCF4EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": RULE})

def canvas(height):
    fig = plt.figure(figsize=(7, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax

def text(ax, x, y, s, size=10, weight="normal", color=INK,
         ha="left", va="center", **kwargs):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, linespacing=1.2, **kwargs)

def box(ax, x, y, w, h, color="white", edge=RULE):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=.012",
                 facecolor=color, edgecolor=edge, linewidth=.8))

def arrow(ax, start, end, color=MUTED, both=False):
    ax.add_patch(FancyArrowPatch(start, end,
                 arrowstyle="<->" if both else "-|>",
                 mutation_scale=9, linewidth=1.2, color=color,
                 shrinkA=0, shrinkB=0))

def vector(ax, x, y, w, h, vals, color):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    cell = w / len(vals)
    for k, val in enumerate(vals):
        ax.add_patch(Rectangle((x+k*cell, y), cell*.93, h,
                    facecolor=(1-val)*np.ones(3)+val*rgb,
                    edgecolor="white", linewidth=.3))

def save(fig, basename):
    # Do not use bbox_inches="tight": physical text sizes assume 7-inch width.
    fig.savefig(OUT / (basename+".pdf"), facecolor="white")
    fig.savefig(QA / (basename+".png"), dpi=180, facecolor="white")
    plt.close(fig)

def overview():
    from make_conceptual_figures import build_overview
    save(build_overview(), "fig1_mechanism_overview")


def recovery():
    from make_conceptual_figures import build_recovery_schedule
    save(build_recovery_schedule(), "fig2_recovery_schedule")


def reuse():
    from make_reuse_figure import build_reuse
    save(build_reuse(), "fig3_component_reuse")

def feasibility():
    from make_feasibility_figure import build_feasibility
    save(build_feasibility(), "fig4_existing_feasibility")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, choices=[1, 2, 3, 4],
                        default=[1, 2, 3, 4], help="Figure numbers to rebuild")
    selected = parser.parse_args().figures
    makers = {1: overview, 2: recovery, 3: reuse, 4: feasibility}
    for number in selected:
        makers[number]()
    print(f"Created 7-inch-wide PDF figures {selected}; all labels are at least 10 pt.")
