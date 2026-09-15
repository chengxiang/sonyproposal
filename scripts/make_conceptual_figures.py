#!/usr/bin/env python3
"""Create proposal schematics. All components and curves are illustrative.

The same figure builders produce the PNG/SVG and seven-inch-wide print PDFs,
so the program scope and physical label sizes remain consistent.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
INK, MUTED = "#203342", "#536575"
BLUE, TEAL, ORANGE = "#2865A6", "#167E83", "#C8782F"
GRAY, RULE = "#75879A", "#C9D3DC"
PB, PT, PO = "#EEF4FA", "#EDF7F5", "#FCF4EB"


def canvas(height):
    fig = plt.figure(figsize=(7, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def txt(ax, x, y, s, size=10, weight="normal", color=INK,
        ha="left", va="center", **kwargs):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, linespacing=1.2, **kwargs)


def box(ax, x, y, w, h, color="white", edge=RULE):
    return ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=.012", facecolor=color,
        edgecolor=edge, linewidth=.8))


def arrow(ax, start, end, color=MUTED, both=False, scale=9):
    return ax.add_patch(FancyArrowPatch(start, end,
        arrowstyle="<->" if both else "-|>", mutation_scale=scale,
        linewidth=1.2, color=color, shrinkA=0, shrinkB=0))


def vector(ax, x, y, w, h, vals, color):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    cell = w / len(vals)
    for k, val in enumerate(vals):
        ax.add_patch(Rectangle((x+k*cell, y), cell*.93, h,
            facecolor=(1-val)*np.ones(3)+val*rgb,
            edgecolor="white", linewidth=.3))


def build_overview():
    fig, ax = canvas(4.65)
    txt(ax, .018, .971, "Selective generation and revision", 12, "bold")
    box(ax, .018, .788, .964, .145, PO, "#E7CDB2")
    txt(ax, .035, .906, "REQUESTED REVISION", 10, "bold", ORANGE)
    txt(ax, .035, .858, "Woman gives cup to man", 10.5, "bold")
    arrow(ax, (.439, .858), (.490, .858), ORANGE)
    txt(ax, .512, .858, "Man gives cup to woman", 10.5, "bold")
    txt(ax, .035, .815, "Preserve the characters, their clothing, and the child reading.")

    # Three connected aspects; the middle aspect is an ongoing training loop.
    box(ax, .018, .622, .964, .126, PB)
    txt(ax, .035, .719, "2.1  Learn representations and component extractors", 10.5, "bold", BLUE)
    txt(ax, .035, .678, "Learn from fixed encoders or")
    txt(ax, .035, .643, "downstream tasks / rewards.")
    txt(ax, .512, .678, "Illustrative: participants / roles / other", 10, color=MUTED)
    for x, col, vals in [(.512, BLUE, [.8,.45,.2,.7,.5,.9]),
                         (.666, ORANGE, [.35,.9,.7,.2,.85,.4]),
                         (.820, GRAY, [.5,.25,.8,.35,.65,.5])]:
        vector(ax, x, .634, .142, .024, vals, col)
    arrow(ax, (.500, .617), (.500, .587), BLUE, both=True, scale=6)

    box(ax, .018, .278, .964, .304, PT)
    txt(ax, .035, .551, "2.2  Jointly learn denoising order, component dependencies,", 10.5, "bold", TEAL)
    txt(ax, .035, .515, "and the transformer", 10.5, "bold", TEAL)
    box(ax, .035, .341, .394, .139, "white", "#BEDCD8")
    box(ax, .553, .341, .412, .139, "white", "#BEDCD8")
    txt(ax, .052, .453, "Dependency diagnostics", 10, "bold", TEAL)
    txt(ax, .052, .409, "Compare noise conditions at")
    txt(ax, .052, .371, "the current model checkpoint.")
    txt(ax, .570, .453, "Schedule + transformer updates", 10, "bold", TEAL)
    txt(ax, .570, .409, "Dependencies and denoising order")
    txt(ax, .570, .371, "evolve together with training.")
    arrow(ax, (.442, .426), (.540, .426), TEAL)
    arrow(ax, (.540, .384), (.442, .384), TEAL)
    txt(ax, .500, .305, "Repeat diagnostics as training changes the model.", 10, "bold", TEAL, ha="center")
    arrow(ax, (.500, .273), (.500, .243), TEAL, both=True, scale=6)

    box(ax, .018, .063, .964, .175, PB)
    txt(ax, .035, .211, "2.3  Shared and specialized transformer computations", 10.5, "bold", BLUE)
    txt(ax, .035, .167, "Noise-dependent module gates", 10, "bold")
    vector(ax, .035, .102, .126, .025, [.85, .2, .6], BLUE)
    arrow(ax, (.174, .1145), (.209, .1145), BLUE)
    for x, active in [(.222, True), (.273, False), (.324, True)]:
        box(ax, x, .092, .038, .045, TEAL if active else "white", TEAL if active else RULE)
    box(ax, .412, .095, .238, .083, "white", "#BEDCD8")
    box(ax, .675, .095, .290, .083, "white", "#BEDCD8")
    txt(ax, .531, .1365, "Sparse attention", 10, "bold", TEAL, ha="center")
    txt(ax, .820, .1365, "Shared / switched MLPs", 10, "bold", TEAL, ha="center")
    txt(ax, .500, .027, "Goal: revise selected content while preserving the rest.",
        10.5, "bold", TEAL, ha="center")
    return fig


def build_recovery_schedule():
    fig, ax = canvas(3.7)
    txt(ax, .017, .960, "A. Diagnostic at a checkpoint", 10.5, "bold", BLUE)
    txt(ax, .017, .906, "Fix model only within each comparison.")
    txt(ax, .188, .822, "Target i", 10, "bold", ha="center")
    txt(ax, .310, .822, "Source j", 10, "bold", ha="center")
    target = [.35,.75,.45,.6,.25,.8]
    for label, y, vals, col in [("Noisier j", .711, [.2,.7,.3,.75,.4,.55], GRAY),
                               ("Cleaner j", .550, [.1,.2,.4,.6,.8,.95], TEAL)]:
        txt(ax, .017, y+.020, label, 10, "bold")
        vector(ax, .145, y, .087, .040, target, BLUE)
        vector(ax, .267, y, .087, .040, vals, col)
        arrow(ax, (.365,y+.020), (.389,y+.020), GRAY)
        box(ax, .398, y-.020, .049, .080, PB)
        txt(ax, .4225, y+.020, r"$f_\theta$", 11, ha="center")
        arrow(ax, (.455,y+.020), (.480,y+.020), GRAY)
        txt(ax, .490, y+.020, r"$L_i$", 11)
    txt(ax, .188, .657, "fixed", 10, color=BLUE, ha="center")
    txt(ax, .310, .657, "vary", 10, color=TEAL, ha="center")
    txt(ax, .017, .464, "Hold target noise and other inputs fixed.", 10, color=MUTED)
    txt(ax, .017, .402, r"$D_{j\to i}=L_i(\mathrm{noisier}\ j)-L_i(\mathrm{cleaner}\ j)$", 10.5)
    txt(ax, .017, .350, "Positive D: cleaner j helps denoise i.", 10, "bold", TEAL)
    txt(ax, .017, .301, "Across noise levels and contexts.", 10, color=MUTED)
    ax.plot([.556,.556], [.274,.976], color=RULE, linewidth=.8)

    txt(ax, .587, .960, "B. Joint schedule + model learning", 10.5, "bold", TEAL)
    txt(ax, .587, .906, "Illustrative schedules; not measured", 10, color=MUTED)
    gax = fig.add_axes([.650,.454,.323,.365])
    t = np.linspace(0,1,401)
    curves = [t+.115*np.sin(2*np.pi*t), t-.115*np.sin(2*np.pi*t),
              t+.055*np.sin(4*np.pi*t)]
    for i, (u, col) in enumerate(zip(curves, [BLUE,ORANGE,TEAL]), 1):
        assert np.all(np.diff(u) >= -1e-12)
        assert abs(u[0]) < 1e-12 and abs(u[-1]-1) < 1e-12
        gax.plot(t, u, color=col, lw=1.5, label=f"Component {i}")
    gax.set(xlim=(0,1), ylim=(0,1), xticks=[0,.5,1], yticks=[0,.5,1])
    gax.set_xticklabels(["0","0.5","1"])
    gax.set_yticklabels(["0","0.5","1"])
    gax.set_xlabel("Generation time t", fontsize=10, labelpad=1)
    gax.set_ylabel("Noise-to-data progress", fontsize=10, labelpad=1)
    gax.tick_params(labelsize=10, length=2, pad=2, colors=MUTED)
    for edge in ["top","right"]:
        gax.spines[edge].set_visible(False)
    for edge in ["left","bottom"]:
        gax.spines[edge].set_color(RULE)
    gax.grid(color="#edf0f3", lw=.6)
    gax.set_axisbelow(True)
    gax.legend(loc="upper left", fontsize=10, frameon=False, handlelength=1,
               borderaxespad=.1, labelspacing=.15, handletextpad=.4)
    txt(ax, .587, .322, "Denoising order and dependencies\nco-evolve with the transformer.", 10, "bold", TEAL)

    txt(ax, .017, .226, "REPEATED THROUGHOUT TRAINING", 10, "bold", ORANGE)
    box(ax, .017, .028, .396, .149, PO, "#E7CDB2")
    box(ax, .587, .028, .396, .149, PT, "#BEDCD8")
    txt(ax, .215, .1025, "Dependency diagnostics\nat each current checkpoint", 10, "bold", ORANGE, ha="center")
    txt(ax, .785, .1025, "Jointly update schedules\nand the transformer", 10, "bold", TEAL, ha="center")
    arrow(ax, (.423, .130), (.577, .130), TEAL)
    txt(ax, .500, .165, "inform", 10, color=TEAL, ha="center")
    arrow(ax, (.577, .066), (.423, .066), ORANGE)
    txt(ax, .500, .031, "remeasure", 10, color=ORANGE, ha="center")
    return fig


def save(fig, basename):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{basename}.{extension}", dpi=250, facecolor="white")
    plt.close(fig)


def overview():
    save(build_overview(), "fig1_mechanism_overview")


def recovery_schedule():
    save(build_recovery_schedule(), "fig2_recovery_schedule")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    overview()
    recovery_schedule()
    print("Created figures 1–2 (PNG + SVG); seven-inch width, labels ≥10 pt.")
