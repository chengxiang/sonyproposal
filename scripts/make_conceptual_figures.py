#!/usr/bin/env python3
"""Create proposal schematics. All curves are illustrative, not experimental data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "svg.fonttype": "none",
    "axes.titleweight": "bold",
    "savefig.facecolor": "white",
})

INK = "#203342"
MUTED = "#536575"
BLUE = "#2865A6"
TEAL = "#167E83"
ORANGE = "#C8782F"
GRAY = "#75879A"
RULE = "#C9D3DC"
PALE_BLUE = "#EEF4FA"
PALE_TEAL = "#EDF7F5"
PALE_ORANGE = "#FCF4EB"


def canvas(size):
    fig = plt.figure(figsize=size, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def box(ax, xywh, fc="white", ec=RULE, lw=1.0, r=0.014):
    x, y, w, h = xywh
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=11, weight="normal", color=INK,
        ha="left", va="center", **kwargs):
    return ax.text(x, y, s, fontsize=size, weight=weight, color=color,
                   ha=ha, va=va, linespacing=1.3, **kwargs)


def arrow(ax, start, end, color=MUTED, both=False, lw=1.6, ms=11):
    p = FancyArrowPatch(start, end, arrowstyle="<->" if both else "-|>",
                        mutation_scale=ms, linewidth=lw, color=color,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(p)
    return p


def vector(ax, x, y, w, h, values, color, border=True):
    n = len(values)
    cell = w / n
    rgb = np.array(matplotlib.colors.to_rgb(color))
    for k, value in enumerate(values):
        fc = tuple((1 - value) * np.ones(3) + value * rgb)
        ax.add_patch(Rectangle((x + k * cell, y), cell * 0.92, h,
                               facecolor=fc, edgecolor="white", lw=0.6))
    if border:
        ax.add_patch(Rectangle((x - 0.002, y - 0.004), w + 0.001, h + 0.008,
                               facecolor="none", edgecolor=color, lw=0.6))


def save(fig, basename):
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{basename}.{extension}", dpi=250)
    plt.close(fig)


def overview():
    fig, ax = canvas((10.2, 5.5))
    txt(ax, 0.035, 0.953, "Representations, transformer mechanisms, and denoising", 15, "bold")
    txt(ax, 0.035, 0.912, "Proposed mechanism (schematic)", 10.5, color=MUTED)

    # One creator request runs through all three views of the generative model.
    box(ax, (0.035, 0.748, 0.93, 0.123), PALE_ORANGE, "#E7CDB2")
    txt(ax, 0.053, 0.843, "REQUESTED REVISION", 9.5, "bold", ORANGE)
    txt(ax, 0.053, 0.805, "Woman gives cup to man", 11.4, "bold")
    arrow(ax, (0.381, 0.804), (0.428, 0.804), ORANGE)
    txt(ax, 0.448, 0.805, "Man gives cup to woman", 11.4, "bold")
    txt(ax, 0.053, 0.768, "Preserve the characters, their clothing, and the child reading.", 10.5, color=MUTED)

    px = [0.035, 0.365, 0.695]
    pw = 0.27
    for x, color in zip(px, [PALE_BLUE, PALE_TEAL, "#F3F5F8"]):
        box(ax, (x, 0.224, pw, 0.443), color)
    for x in [0.17, 0.83]:
        arrow(ax, (x, 0.735), (x, 0.682), color=ORANGE)

    txt(ax, 0.053, 0.631, "1  Learned components", 11.6, "bold", BLUE)
    txt(ax, 0.053, 0.589, "Labels below are illustrative", 10, color=MUTED)
    txt(ax, 0.053, 0.533, "Participants (example)", 10.5, "bold")
    vector(ax, 0.054, 0.477, 0.224, 0.031,
           [.8, .45, .2, .7, .5, .9, .4, .6], BLUE)
    txt(ax, 0.053, 0.426, "Giver–receiver roles (example)", 10.5, "bold", ORANGE)
    vector(ax, 0.054, 0.370, 0.224, 0.031,
           [.35, .9, .7, .2, .85, .4, .65, .3], ORANGE)
    txt(ax, 0.053, 0.319, "Other features (example)", 10.5, "bold")
    vector(ax, 0.054, 0.263, 0.224, 0.031,
           [.5, .25, .8, .35, .65, .5, .4, .75], GRAY)

    txt(ax, 0.383, 0.631, "2  Transformer modules", 11.6, "bold", TEAL)
    txt(ax, 0.383, 0.589, "Which information, where, when?", 9.5, color=MUTED)
    box(ax, (0.385, 0.446, 0.230, 0.105), "white", "#BEDCD8")
    txt(ax, 0.50, 0.517, "Attention + projections", 10.5, "bold", TEAL, ha="center")
    txt(ax, 0.50, 0.475, "Pass relevant information", 9.8, ha="center")
    arrow(ax, (0.50, 0.433), (0.50, 0.404), TEAL, both=True, ms=9)
    box(ax, (0.385, 0.287, 0.230, 0.105), "white", "#BEDCD8")
    txt(ax, 0.50, 0.358, "MLPs / shared bank", 10.5, "bold", TEAL, ha="center")
    txt(ax, 0.50, 0.316, "Store and use learned features", 9.4, ha="center")

    txt(ax, 0.713, 0.631, "3  Asynchronous updates", 11.6, "bold", BLUE)
    txt(ax, 0.713, 0.589, "Coordinate component progress", 9.6, color=MUTED)
    gax = fig.add_axes([0.731, 0.340, 0.202, 0.196])
    t = np.linspace(0, 1, 201)
    for a, color in [(0.105, BLUE), (-0.11, ORANGE), (0.03, TEAL)]:
        gax.plot(t, t + a * np.sin(2 * np.pi * t), color=color, lw=1.9)
    gax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 1], yticks=[0, 1])
    gax.tick_params(labelsize=8, length=2, pad=2, colors=MUTED)
    for s in ["top", "right"]:
        gax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        gax.spines[s].set_color(RULE)
    gax.set_facecolor("none")
    gax.set_xlabel("Generation progress", fontsize=9, color=MUTED, labelpad=2)
    txt(ax, 0.83, 0.251, "Schematic schedules", 9, color=MUTED, ha="center")

    # Reciprocal design: representation choices and mechanisms inform each other;
    # the mechanisms determine how component states evolve during generation.
    arrow(ax, (0.315, 0.490), (0.355, 0.490), BLUE, both=True)
    arrow(ax, (0.645, 0.490), (0.685, 0.490), TEAL, both=True)

    box(ax, (0.035, 0.060, 0.93, 0.104), PALE_TEAL, "#C3DDD9")
    arrow(ax, (0.83, 0.214), (0.83, 0.176), TEAL)
    txt(ax, 0.053, 0.135, "GOAL: LEARN WHICH STATES AND WEIGHTS REALIZE THE REQUEST", 9.5, "bold", TEAL)
    txt(ax, 0.053, 0.094,
        "Reverse who gives the cup; keep the characters, clothing, and the child reading.",
        10.1, "bold")
    save(fig, "fig1_mechanism_overview")


def recovery_schedule():
    fig, ax = canvas((10.2, 4.8))
    txt(ax, 0.035, 0.950, "Measure denoising dependencies by varying component noise", 15, "bold")
    txt(ax, 0.035, 0.901, "Proposed measurement and scheduling method (schematic)", 10.5, color=MUTED)
    ax.plot([0.548, 0.548], [0.225, 0.824], color=RULE, lw=1)
    txt(ax, 0.035, 0.826, "A. Controlled denoising", 12, "bold", BLUE)
    txt(ax, 0.035, 0.777, "Same target noise, other inputs, and transformer", 10.1, color=MUTED)

    # Noise indicators are pictograms, never claimed samples or measurements.
    txt(ax, 0.164, 0.718, "Target i", 10.5, "bold", ha="center")
    txt(ax, 0.300, 0.718, "Source j", 10.5, "bold", ha="center")
    target = [.35, .75, .45, .6, .25, .8]
    for label, y, source, shade in [
        ("Noisier j", 0.620, [.2, .7, .3, .75, .4, .55], GRAY),
        ("Cleaner j", 0.457, [.1, .2, .4, .6, .8, .95], TEAL),
    ]:
        txt(ax, 0.036, y + 0.016, label, 9.7, "bold")
        vector(ax, 0.118, y, 0.092, 0.037, target, BLUE)
        vector(ax, 0.254, y, 0.092, 0.037, source, shade)
        arrow(ax, (0.360, y + 0.018), (0.391, y + 0.018), GRAY, ms=9)
        box(ax, (0.399, y - 0.021, 0.051, 0.081), PALE_BLUE, "#C0D1E2", r=0.008)
        txt(ax, 0.4245, y + 0.019, r"$f_\theta$", 11, ha="center")
        arrow(ax, (0.458, y + 0.018), (0.478, y + 0.018), GRAY, ms=8)
        txt(ax, 0.488, y + 0.018, r"$L_i$", 12, ha="left")
    txt(ax, 0.164, 0.562, "fixed", 9, color=BLUE, ha="center")
    txt(ax, 0.300, 0.562, "vary only this", 9, color=TEAL, ha="center")

    txt(ax, 0.035, 0.365,
        r"$D_{j\to i}=L_i(\mathrm{noisier}\ j)-L_i(\mathrm{cleaner}\ j)$",
        12.5, "bold")
    txt(ax, 0.035, 0.310, "Positive D: cleaner j helps denoise i.", 10.5, "bold", TEAL)
    txt(ax, 0.035, 0.260, "Repeat across noise levels and contexts.", 10, color=MUTED)

    txt(ax, 0.588, 0.826, "B. Coordinate denoising", 12, "bold", TEAL)
    txt(ax, 0.588, 0.777, "Illustrative schedules — not measured", 10.1, color=MUTED)
    gax = fig.add_axes([0.658, 0.357, 0.295, 0.365])
    t = np.linspace(0, 1, 401)
    schedules = [t + 0.115 * np.sin(2*np.pi*t),
                 t - 0.115 * np.sin(2*np.pi*t),
                 t + 0.055 * np.sin(4*np.pi*t)]
    for k, (u, color) in enumerate(zip(schedules, [BLUE, ORANGE, TEAL]), 1):
        assert (np.diff(u) >= -1e-12).all()
        assert abs(u[0]) < 1e-12 and abs(u[-1] - 1) < 1e-12
        gax.plot(t, u, color=color, lw=2.1, label=f"Component {k}")
    gax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
    gax.set_xticklabels(["0", "0.5", "1"])
    gax.set_yticklabels(["0", "0.5", "1"])
    gax.set_xlabel("Generation time t", fontsize=10, color=INK, labelpad=3)
    gax.set_ylabel("Noise-to-data progress uₖ", fontsize=9.8, color=INK, labelpad=4)
    gax.tick_params(labelsize=9, colors=MUTED, length=3)
    for s in ["top", "right"]:
        gax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        gax.spines[s].set_color(RULE)
    gax.grid(color="#EDF0F3", lw=.7)
    gax.set_axisbelow(True)
    gax.legend(loc="upper left", fontsize=8.5, frameon=False, handlelength=1.5,
               borderaxespad=.2, labelspacing=.4)
    txt(ax, 0.588, 0.260, "Crossing curves allow changing relative rates.", 9.8, color=MUTED)

    box(ax, (0.035, 0.060, 0.93, 0.121), PALE_ORANGE, "#E7CDB2")
    txt(ax, 0.053, 0.147, "SCHEDULING TRADEOFF", 9.5, "bold", ORANGE)
    txt(ax, 0.053, 0.102,
        "Advancing j can help denoise i, but leaves less context for denoising j itself.",
        10.5)
    save(fig, "fig2_recovery_schedule")


if __name__ == "__main__":
    overview()
    recovery_schedule()
    print("Created fig1_mechanism_overview and fig2_recovery_schedule (PNG + SVG).")
