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
    fig, ax = canvas(4.5)
    txt(ax, .018, .974, "Learn representation, generation, and computation together", 12, "bold")

    # The coordinate selections are an illustration, not an asserted learned basis.
    box(ax, .018, .600, .964, .336, PB)
    txt(ax, .037, .902, "REPRESENTATION: token vectors and component extractors", 10.5, "bold", BLUE)

    def selected_coordinates(y, selected, color):
        for j in range(12):
            x = .177 + j * .20 / 12
            ax.add_patch(Rectangle((x, y-.012), .20/12*.90, .024,
                facecolor=color if j in selected else "#DCE4EB",
                edgecolor="white", linewidth=.35))
        arrow(ax, (.390, y), (.423, y), color, scale=7)

    for label, y, ids, col, interpretation in [
        ("Token 1", .846, [1, 2, 3, 7, 8], BLUE, "woman; red coat"),
        ("Token 2", .790, [1, 2, 3, 7, 8], BLUE, "man; green sweater"),
        (r"Relation $R$", .734, [0, 1, 4, 5, 8, 9], ORANGE,
         "giver = token 1; receiver = token 2"),
        (r"Visual $V$", .651, [2, 3, 4, 7, 8, 9], TEAL,
         "hands and cup in image tokens"),
    ]:
        txt(ax, .037, y, label, 10, "bold", col)
        selected_coordinates(y, ids, col)
        txt(ax, .440, y, interpretation, 10)
    txt(ax, .440, .697, "object = blue cup", 10)

    # Each arrow states a concrete relationship; this is not a serial pipeline.
    arrow(ax, (.046, .589), (.046, .381), BLUE)
    txt(ax, .065, .484,
        "Make informative\ncomponents ready\nearlier to improve\ngeneration",
        10, color=BLUE)
    arrow(ax, (.480, .381), (.480, .589), TEAL)
    txt(ax, .462, .484,
        "Vary noise to learn\nwhich components\nhelp predict\nother components",
        10, color=TEAL, ha="right")
    arrow(ax, (.552, .589), (.552, .381), ORANGE)
    txt(ax, .575, .484,
        "Choose component inputs and targets;\nshare attention across tasks;\nspecialize prediction modules.",
        10, color=ORANGE)

    box(ax, .018, .078, .464, .295, PT)
    txt(ax, .036, .339, "DENOISING ALGORITHM", 10.5, "bold", TEAL)
    txt(ax, .038, .279, "One progress variable per component:", 10)
    txt(ax, .250, .220, r"$\tau(t)=(\tau_1(t),\ldots,\tau_K(t))$", 12,
        color=TEAL, ha="center")
    txt(ax, .038, .145,
        "Predict from current component states.\nLearn each component's rate of advance.", 10)

    box(ax, .518, .078, .464, .295, PO)
    txt(ax, .536, .339, "TRANSFORMER ARCHITECTURE", 10.5, "bold", ORANGE)
    txt(ax, .539, .297, "Conditions", 10, color=MUTED)
    box(ax, .656, .151, .110, .133, "white", "#E7CDB2")
    txt(ax, .711, .218, "Shared\nattention", 10, "bold", ORANGE, ha="center")
    for y, condition, target, color in [
        (.254, r"$1,2,R$", "V", TEAL),
        (.178, r"$1,2,V$", "R", ORANGE),
    ]:
        txt(ax, .539, y, condition, 10, color=color)
        arrow(ax, (.622, y), (.648, y), color, scale=7)
        arrow(ax, (.774, y), (.799, y), color, scale=7)
        box(ax, .805, y-.027, .098, .054, "white", "#E7CDB2")
        txt(ax, .854, y, r"$\mathrm{MLP}_{"+target+"}$", 10, ha="center", color=color)
        arrow(ax, (.911, y), (.943, y), color, scale=7)
        txt(ax, .961, y, "$"+target+"$", 10, "bold", color, ha="center")
    txt(ax, .537, .107, "Each task also receives its noisy target.", 10, color=MUTED)

    # Schedules and the transformer are fitted together, not in separate phases.
    from matplotlib.path import Path as MplPath
    connector = MplPath([(.739, .072), (.739, .032), (.261, .032), (.261, .072)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=connector, arrowstyle="<->",
        mutation_scale=8, linewidth=1.2, color=MUTED))
    txt(ax, .500, .032, "Jointly train schedules and transformer", 10, "bold", MUTED,
        ha="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    return fig


def build_recovery_schedule():
    """A conditional prediction example links dependence, scheduling, and modules.

    No images, losses, schedules, or module assignments are experimental data.
    The illustrative paired diagnostic withholds roles from prompt conditioning.
    """
    fig, ax = canvas(4.8)
    txt(ax, .018, .974, "How available information shapes denoising and computation", 12, "bold")

    # Meaningful vector labels are carried through all three panels.
    for x, w, label, detail, col, fill in [
        (.018, .286, r"$I$: participants", "woman, man, blue cup", BLUE, PB),
        (.322, .320, r"$R$: interaction roles", "giver, receiver, object", ORANGE, PO),
        (.660, .322, r"$V$: visual realization", "hand and cup positions", TEAL, PT),
    ]:
        box(ax, x, .873, w, .073, fill, col)
        txt(ax, x+w/2, .924, label, 10, "bold", col, ha="center")
        txt(ax, x+w/2, .891, detail, 10, ha="center")

    txt(ax, .018, .837, "A. What does clearer role information contribute to visual prediction?", 10.5, "bold", BLUE)
    txt(ax, .018, .801, "Hold participants, visual-target noise, and transformer weights fixed.", 10, color=MUTED)
    for x, clear in [(.018, False), (.516, True)]:
        w=.466
        box(ax, x, .640, w, .136, "white", RULE)
        txt(ax, x+.017, .751, "Clearer roles R" if clear else "Noisier roles R", 10, "bold", ORANGE)
        txt(ax, x+.085, .708, "woman", 10, ha="center")
        txt(ax, x+w-.079, .708, "man", 10, ha="center")
        arrow(ax, (x+.149,.708), (x+w-.141,.708), ORANGE if clear else GRAY,
              both=not clear, scale=8)
        txt(ax, x+w/2, .708, "gives cup" if clear else "who gives?", 10,
            color=ORANGE if clear else MUTED, ha="center",
            bbox={"facecolor":"white", "edgecolor":"none", "pad":1})
        txt(ax, x+.017, .662, r"Denoise the same hand/cup states $V$", 10, color=TEAL)
    txt(ax, .018, .612, "Compare visual denoising losses; role labels are withheld from the prompt.", 10)

    txt(ax, .018, .566, "B. Learn component schedules with the transformer", 10.5, "bold", TEAL)
    gax=fig.add_axes([.083, .357, .581, .142])
    t=np.linspace(0,1,501)
    schedules=[1-(1-t)**2.2, 1-(1-t)**1.35, t**1.5]
    for u,col,label in zip(schedules,[BLUE,ORANGE,TEAL],[r"$I$",r"$R$",r"$V$"]):
        gax.plot(t,u,color=col,lw=2,label=label)
    gax.set(xlim=(0,1),ylim=(0,1.04),xticks=[0,.5,1],yticks=[0,1])
    gax.set_xticklabels(["0","0.5","1"])
    gax.set_yticklabels(["0","1"])
    gax.set_xlabel(r"Generation time $t$",fontsize=10,labelpad=0)
    gax.set_ylabel("Progress",fontsize=10,labelpad=0)
    gax.tick_params(labelsize=10,length=2,pad=2,colors=MUTED)
    for edge in ["top","right"]: gax.spines[edge].set_visible(False)
    for edge in ["left","bottom"]: gax.spines[edge].set_color(RULE)
    gax.legend(loc="lower center",bbox_to_anchor=(.50,1.00),ncol=3,
        fontsize=10,frameon=False,handlelength=1.4,columnspacing=2,
        borderaxespad=0,handletextpad=.4)
    txt(ax,.703,.478,"Earlier R can help V,\nbut R must be\npredicted with less\nvisual information.",10,va="center")
    txt(ax,.703,.370,"All components\ncontinue developing.",10,color=TEAL)

    txt(ax,.018,.270,"C. Identify information paths and reusable computations",10.5,"bold",ORANGE)
    # One selected contribution, with explicit noisy input, read path, shared
    # predictor, target-specific projections, and noise-dependent output gate.
    for y,label,col,fill in [(.217,"I: participants",BLUE,PB),
                             (.161,"R: noisy roles",ORANGE,PO),
                             (.105,"V: noisy visual",TEAL,PT)]:
        box(ax,.018,y-.020,.183,.041,fill,col)
        txt(ax,.1095,y,label,10,color=col,ha="center")
        arrow(ax,(.207,y),(.282,.161),col,scale=7)
    box(ax,.291,.109,.127,.104,PO,ORANGE)
    txt(ax,.3545,.161,"Attention\nhead",10,"bold",ORANGE,ha="center")
    arrow(ax,(.426,.161),(.476,.161),ORANGE,scale=8)
    box(ax,.484,.115,.133,.092,"white",ORANGE)
    txt(ax,.5505,.161,"Shared\nMLP",10,"bold",ORANGE,ha="center")
    # A solid visual path and a dashed alternative semantic task show reuse
    # without duplicating the transformer or asserting that tasks are identical.
    arrow(ax,(.625,.177),(.670,.202),TEAL,scale=7)
    box(ax,.679,.180,.110,.044,PT,TEAL)
    txt(ax,.734,.202,"Map to V",10,color=TEAL,ha="center")
    arrow(ax,(.797,.202),(.866,.202),TEAL,scale=7)
    txt(ax,.833,.236,r"$g_m(\mathbf{u})$",10,color=TEAL,ha="center")
    txt(ax,.921,.202,"Predict V",10,"bold",TEAL,ha="center")
    ap=FancyArrowPatch((.625,.141),(.670,.104),arrowstyle="-|>",mutation_scale=7,
                      linewidth=1.1,linestyle=(0,(3,2)),color=ORANGE,shrinkA=0,shrinkB=0)
    ax.add_patch(ap)
    box(ax,.679,.081,.110,.044,PO,ORANGE)
    txt(ax,.734,.103,"Map to R",10,color=ORANGE,ha="center")
    ap=FancyArrowPatch((.797,.103),(.866,.103),arrowstyle="-|>",mutation_scale=7,
                      linewidth=1.1,linestyle=(0,(3,2)),color=ORANGE,shrinkA=0,shrinkB=0)
    ax.add_patch(ap)
    txt(ax,.921,.103,"Predict R",10,color=ORANGE,ha="center")
    txt(ax,.018,.034,r"Gate $g_m(\mathbf{u})$: noise-dependent contribution. Dashed: reuse for role prediction.",10,color=MUTED)
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
