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
    """Three linked design choices, with concrete illustrative token components."""
    fig, ax = canvas(5.5)
    txt(ax, .018, .974, "Learn representation, generation, and computation together", 12, "bold")

    # The coordinate selections are an illustration, not an asserted learned basis.
    box(ax, .018, .613, .964, .323, PB)
    txt(ax, .037, .909, "REPRESENTATION: token vectors and component extractors", 10.5, "bold", BLUE)

    def selected_coordinates(y, selected, color):
        for j in range(12):
            x = .177 + j * .20 / 12
            ax.add_patch(Rectangle((x, y-.012), .20/12*.90, .024,
                facecolor=color if j in selected else "#DCE4EB",
                edgecolor="white", linewidth=.35))
        arrow(ax, (.390, y), (.423, y), color, scale=7)

    for label, y, ids, col, interpretation in [
        ("Token 1", .863, [1, 2, 3, 7, 8], BLUE, "woman; red coat"),
        ("Token 2", .811, [1, 2, 3, 7, 8], BLUE, "man; green sweater"),
        (r"Relation $R$", .757, [0, 1, 4, 5, 8, 9], ORANGE,
         "giver = token 1; receiver = token 2"),
        (r"Visual $V$", .681, [2, 3, 4, 7, 8, 9], TEAL,
         "hands and cup in image tokens"),
    ]:
        txt(ax, .037, y, label, 10, "bold", col)
        selected_coordinates(y, ids, col)
        txt(ax, .440, y, interpretation, 10)
    txt(ax, .440, .728, "object = blue cup", 10)
    txt(ax, .037, .635,
        "Colored cells illustrate selected coordinates; useful components are learned.",
        10, color=MUTED)

    # Each arrow states a concrete relationship; this is not a serial pipeline.
    arrow(ax, (.046, .606), (.046, .351), BLUE)
    txt(ax, .065, .476,
        "Make informative\ncomponents ready\nearlier to improve\ngeneration",
        10, color=BLUE)
    arrow(ax, (.480, .351), (.480, .606), TEAL)
    txt(ax, .462, .476,
        "Vary noise to learn\nwhich components\nhelp predict\nother components",
        10, color=TEAL, ha="right")
    arrow(ax, (.552, .606), (.552, .351), ORANGE)
    txt(ax, .575, .476,
        "Choose component inputs and targets;\nshare attention across tasks;\nspecialize prediction modules.",
        10, color=ORANGE)

    box(ax, .018, .072, .464, .273, PT)
    txt(ax, .036, .314, "DENOISING ALGORITHM", 10.5, "bold", TEAL)
    txt(ax, .038, .259, "One progress variable per component:", 10)
    txt(ax, .250, .208, r"$\tau(t)=(\tau_1(t),\ldots,\tau_K(t))$", 12,
        color=TEAL, ha="center")
    txt(ax, .038, .145,
        "Predict from current component states.\nLearn each component's rate of advance.", 10)

    box(ax, .518, .072, .464, .273, PO)
    txt(ax, .536, .314, "TRANSFORMER ARCHITECTURE", 10.5, "bold", ORANGE)
    txt(ax, .539, .279, "Conditions", 10, color=MUTED)
    box(ax, .656, .140, .110, .126, "white", "#E7CDB2")
    txt(ax, .711, .203, "Shared\nattention", 10, "bold", ORANGE, ha="center")
    for y, condition, target, color in [
        (.239, r"$1,2,R$", "V", TEAL),
        (.164, r"$1,2,V$", "R", ORANGE),
    ]:
        txt(ax, .539, y, condition, 10, color=color)
        arrow(ax, (.622, y), (.648, y), color, scale=7)
        arrow(ax, (.774, y), (.799, y), color, scale=7)
        box(ax, .805, y-.027, .098, .054, "white", "#E7CDB2")
        txt(ax, .854, y, r"$\mathrm{MLP}_{"+target+"}$", 10, ha="center", color=color)
        arrow(ax, (.911, y), (.943, y), color, scale=7)
        txt(ax, .961, y, "$"+target+"$", 10, "bold", color, ha="center")
    txt(ax, .537, .099, "Each task also receives its noisy target.", 10, color=MUTED)

    # Schedules and the transformer are fitted together, not in separate phases.
    from matplotlib.path import Path as MplPath
    connector = MplPath([(.739, .068), (.739, .030), (.261, .030), (.261, .068)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=connector, arrowstyle="<->",
        mutation_scale=8, linewidth=1.2, color=MUTED))
    txt(ax, .500, .030, "Jointly train schedules and transformer", 10, "bold", MUTED,
        ha="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    return fig


def build_recovery_schedule():
    """One concrete dependency factorization, schedule, and module allocation.

    These are hypothetical examples of learned objects. Colored groups contain
    several component vectors; the model need not learn these human labels.
    """
    fig, ax = canvas(5.2)
    txt(ax, .018, .975, "From component dependencies to denoising and computation", 12, "bold")
    txt(ax, .018, .931, "A. Dependencies given “woman gives the blue cup to the man”", 10.5, "bold", BLUE)

    # Information can pass through clothing bindings or directly from the cup
    # to the final interaction. Edges indicate useful conditioning information.
    node_specs = [
        (.040, .835, .245, .060, "Woman", PB, BLUE),
        (.378, .835, .245, .060, "Blue cup", PB, BLUE),
        (.715, .835, .245, .060, "Man", PB, BLUE),
        (.040, .738, .300, .064, "Woman in red coat", PO, ORANGE),
        (.660, .738, .300, .064, "Man in green sweater", PO, ORANGE),
        (.263, .639, .474, .064, "Woman hands cup to man", PT, TEAL),
    ]
    for x, y, w, h, label, fill, edge in node_specs:
        box(ax, x, y, w, h, fill, edge)
        txt(ax, x+w/2, y+h/2, label, 10, "bold", edge, ha="center")
    arrow(ax, (.163, .831), (.190, .806), ORANGE)
    arrow(ax, (.838, .831), (.810, .806), ORANGE)
    arrow(ax, (.192, .734), (.358, .706), TEAL)
    arrow(ax, (.808, .734), (.642, .706), TEAL)
    arrow(ax, (.500, .831), (.500, .707), TEAL)

    txt(ax, .018, .612, "B. Overlapping denoising schedules for these component groups", 10.5, "bold", TEAL)
    # The groups overlap in time. Chosen snapshots fall in the unshared parts
    # so that the activity diagram can be read without another gating legend.
    gax = fig.add_axes([.138, .390, .823, .163])
    t = np.linspace(0, 1, 501)
    schedules = [np.clip(t/.40, 0, 1),
                 np.clip((t-.20)/.45, 0, 1),
                 np.clip((t-.55)/.45, 0, 1)]
    colors = [BLUE, ORANGE, TEAL]
    labels = ["Participants + cup", "Clothing bindings", "Handover"]
    for u, col, label in zip(schedules, colors, labels):
        assert np.all(np.diff(u) >= -1e-12)
        assert abs(u[0]) < 1e-12 and abs(u[-1]-1) < 1e-12
        gax.plot(t, u, color=col, lw=2, label=label)
    for time in [.10, .50, .85]:
        gax.axvline(time, color=RULE, lw=.8, ls=(0,(2,2)), zorder=0)
    gax.set(xlim=(0,1), ylim=(-.025,1.025), xticks=[0,.1,.5,.85,1], yticks=[0,1])
    gax.set_xticklabels(["0", "0.1", "0.5", "0.85", "1"])
    gax.set_yticklabels(["0", "1"])
    gax.set_xlabel(r"Generation time $t$", fontsize=10, labelpad=0)
    gax.set_ylabel("Progress", fontsize=10, labelpad=0)
    gax.tick_params(labelsize=10, length=2, pad=2, colors=MUTED)
    for edge in ["top", "right"]:
        gax.spines[edge].set_visible(False)
    for edge in ["left", "bottom"]:
        gax.spines[edge].set_color(RULE)
    gax.legend(loc="lower center", bbox_to_anchor=(.50,1.015), ncol=3,
        fontsize=10, frameon=False, handlelength=1.2, columnspacing=1.4,
        borderaxespad=0, handletextpad=.4)

    txt(ax, .018, .304, "C. The same transformer at three generation times", 10.5, "bold", ORANGE)
    txt(ax, .018, .274, "Colored: active modules. Gray: gated off. Bind = clothing; Hand = handover.", 10, color=MUTED)

    def transformer(x, time, stage):
        w = .300
        col = colors[stage]
        box(ax, x, .050, w, .184, "white", RULE)
        txt(ax, x+w/2, .244, rf"$t={time}$", 10, "bold", col, ha="center")
        # Two attention-head groups, followed by three target-prediction slices.
        # Their layout and labels are identical at every time; only gates vary.
        txt(ax, x+.012, .188, "Attn", 10, color=MUTED)
        for j, label in enumerate(["Bind", "Hand"]):
            active = stage == j+1
            left = x+.078+j*.107
            box(ax, left, .169, .098, .041,
                [PO, PT][j] if active else "#F2F3F5",
                col if active else RULE)
            txt(ax, left+.049, .1895, label, 10,
                "bold" if active else "normal", col if active else "#89929B", ha="center")
        arrow(ax, (x+.176, .163), (x+.176, .149), MUTED, scale=6)
        txt(ax, x+.012, .124, "MLP", 10, color=MUTED)
        for j, label in enumerate(["Entity", "Bind", "Hand"]):
            active = stage == j
            left = x+.078+j*.071
            box(ax, left, .104, .066, .041,
                [PB, PO, PT][j] if active else "#F2F3F5",
                col if active else RULE)
            txt(ax, left+.033, .1245, label, 10,
                "normal", col if active else "#89929B", ha="center")
        txt(ax, x+w/2, .073, labels[stage], 10, "bold", col, ha="center")

    for x, time, stage in [(.018, "0.1", 0), (.350, "0.5", 1), (.682, "0.85", 2)]:
        transformer(x, time, stage)
    txt(ax, .018, .018,
        "Illustration only: component definitions, dependencies, schedules, and module gates are learned.",
        10, color=MUTED)
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
