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
from matplotlib.path import Path as MplPath
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


def arrow(ax, start, end, color=MUTED, both=False):
    return ax.add_patch(FancyArrowPatch(start, end,
        arrowstyle="<->" if both else "-|>", mutation_scale=9,
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

    # Four connected steps. The turning arrow preserves the subsection order.
    box(ax, .018, .480, .451, .263, PB)
    box(ax, .531, .480, .451, .263, PT)
    box(ax, .018, .099, .451, .306, PB)
    box(ax, .531, .099, .451, .306, PT)
    arrow(ax, (.478, .609), (.522, .609), BLUE)
    path = MplPath([(.756, .473), (.756, .442), (.243, .442), (.243, .413)],
                   [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>",
        mutation_scale=9, linewidth=1.2, color=TEAL))
    arrow(ax, (.478, .248), (.522, .248), BLUE)

    txt(ax, .035, .710, "2.1  Learn representations", 10.5, "bold", BLUE)
    txt(ax, .035, .655, "Learn extractors from fixed encoders")
    txt(ax, .035, .615, "or downstream tasks / rewards")
    txt(ax, .035, .559, "Illustrative: participants / roles / other", 10, color=MUTED)
    for x, col, vals in [(.035, BLUE, [.8,.45,.2,.7,.5,.9]),
                         (.177, ORANGE, [.35,.9,.7,.2,.85,.4]),
                         (.319, GRAY, [.5,.25,.8,.35,.65,.5])]:
        vector(ax, x, .505, .131, .028, vals, col)

    txt(ax, .548, .710, "2.2  Measure dependencies", 10.5, "bold", TEAL)
    txt(ax, .548, .655, "Vary one component’s noise;")
    txt(ax, .548, .615, "measure other denoising losses.")
    txt(ax, .548, .559, "Does cleaner j help denoise i?", 10, "bold", TEAL)
    txt(ax, .548, .516, "Across noise levels and contexts.", 10, color=MUTED)

    txt(ax, .035, .371, "2.3  Jointly optimize generation", 10.5, "bold", BLUE)
    txt(ax, .035, .317, "Schedules, transformer, module activity")
    txt(ax, .035, .262, r"Component noise levels $\mathbf{u}$", 10, "bold")
    vector(ax, .035, .195, .126, .032, [.85, .2, .6], BLUE)
    arrow(ax, (.174, .211), (.217, .211), BLUE)
    for x, active in [(.232, True), (.295, False), (.358, True)]:
        box(ax, x, .184, .047, .052, TEAL if active else "white", TEAL if active else RULE)
    txt(ax, .035, .139, "Noise levels determine active modules.", 10, color=MUTED)

    txt(ax, .548, .371, "2.4  Share and specialize", 10.5, "bold", TEAL)
    txt(ax, .548, .317, "Across conditional denoising tasks")
    box(ax, .548, .190, .193, .089, "white", "#BEDCD8")
    box(ax, .766, .190, .199, .089, "white", "#BEDCD8")
    txt(ax, .6445, .2345, "Sparse\nattention", 10, "bold", TEAL, ha="center")
    txt(ax, .8655, .2345, "Shared /\nswitchable MLPs", 10, "bold", TEAL, ha="center")
    txt(ax, .548, .139, "Reuse and specialize computations.", 10, color=MUTED)

    txt(ax, .5, .044, "Goal: revise selected content while preserving the rest.",
        10.5, "bold", TEAL, ha="center")
    return fig


def build_recovery_schedule():
    fig, ax = canvas(3.7)
    txt(ax, .017, .960, "A. Controlled denoising", 10.5, "bold", BLUE)
    txt(ax, .017, .901, "Fix target noise, other inputs, and model.")
    txt(ax, .188, .820, "Target i", 10, "bold", ha="center")
    txt(ax, .310, .820, "Source j", 10, "bold", ha="center")
    target = [.35,.75,.45,.6,.25,.8]
    for label, y, vals, col in [("Noisier j", .699, [.2,.7,.3,.75,.4,.55], GRAY),
                               ("Cleaner j", .522, [.1,.2,.4,.6,.8,.95], TEAL)]:
        txt(ax, .017, y+.022, label, 10, "bold")
        vector(ax, .145, y, .087, .044, target, BLUE)
        vector(ax, .267, y, .087, .044, vals, col)
        arrow(ax, (.365,y+.022), (.389,y+.022), GRAY)
        box(ax, .398, y-.022, .049, .088, PB)
        txt(ax, .4225, y+.022, r"$f_\theta$", 11, ha="center")
        arrow(ax, (.455,y+.022), (.480,y+.022), GRAY)
        txt(ax, .490, y+.022, r"$L_i$", 11)
    txt(ax, .188, .640, "fixed", 10, color=BLUE, ha="center")
    txt(ax, .310, .640, "vary", 10, color=TEAL, ha="center")
    txt(ax, .017, .425, r"$D_{j\to i}=L_i(\mathrm{noisier}\ j)-L_i(\mathrm{cleaner}\ j)$", 10.5)
    txt(ax, .017, .365, "Positive D: cleaner j helps denoise i.", 10, "bold", TEAL)
    txt(ax, .017, .307, "Repeat over noise levels and contexts.", 10, color=MUTED)
    ax.plot([.556,.556], [.282,.976], color=RULE, linewidth=.8)

    txt(ax, .587, .960, "B. Schedules and module activity", 10.5, "bold", TEAL)
    txt(ax, .587, .901, "Illustrative; not measured", 10, color=MUTED)
    gax = fig.add_axes([.650,.443,.323,.390])
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
    txt(ax, .587, .307, "Noise vector → active modules", 10, "bold", TEAL)

    box(ax, .017, .017, .966, .217, PO, "#E7CDB2")
    txt(ax, .033, .192, "JOINT DESIGN", 10, "bold", ORANGE)
    txt(ax, .033, .129, "Optimize schedules and transformer jointly; condition module activity")
    txt(ax, .033, .073, "on all component noise levels, with sparse attention and shared or switched MLPs.")
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
