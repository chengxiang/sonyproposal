#!/usr/bin/env python3
"""Build the proposal's compact method schematic (illustrative, not data).

Representations and component groups are fixed before dependency/schedule
learning. One common builder produces editable SVG, PNG, and print PDF.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
INK, MUTED = '#203342', '#536575'
BLUE, TEAL, ORANGE = '#2865A6', '#167E83', '#C8782F'
RULE = '#C9D3DC'
PB, PT, PO = '#EEF4FA', '#EDF7F5', '#FCF4EB'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})


def txt(ax, x, y, s, size=10, weight='normal', color=INK,
        ha='left', va='center', **kwargs):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, linespacing=1.15, **kwargs)


def box(ax, x, y, w, h, color='white', edge=RULE):
    return ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0,rounding_size=.012', facecolor=color,
        edgecolor=edge, linewidth=.8))


def arrow(ax, start, end, color=MUTED, scale=9, both=False):
    return ax.add_patch(FancyArrowPatch(start, end,
        arrowstyle='<->' if both else '-|>', mutation_scale=scale,
        linewidth=1.1, color=color, shrinkA=0, shrinkB=0))


def build_overview():
    """A fixed representation links conditional information, timing, and modules.

    Components refer to fixed coordinate groups, not asserted semantic factors.
    All schedules and the selected D1-to-Q1 path are hypothetical illustrations.
    """
    fig = plt.figure(figsize=(7, 4), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')

    txt(ax, .018, .968, 'A. Obtain representations; fix their component groups',
        10.5, 'bold', BLUE)
    box(ax, .018, .815, .337, .112, PB)
    txt(ax, .1865, .871, 'Pretrained or task-trained\nmodel activations',
        ha='center', color=BLUE)
    arrow(ax, (.365, .871), (.410, .871), BLUE)
    for x, label, fill, col in [
        (.427, r'$D_1$', PB, BLUE), (.539, r'$D_2$', PB, BLUE),
        (.651, r'$Q_1$', PO, ORANGE), (.763, r'$Q_2$', PO, ORANGE),
        (.875, r'$V$', PT, TEAL),
    ]:
        box(ax, x, .831, .092, .080, fill, col)
        txt(ax, x+.046, .871, label, 12, 'bold', col, ha='center')
    txt(ax, .018, .779,
        r'$D_1,D_2$: visual features; $Q_1,Q_2$: caption features; $V$: image latents.',
        color=MUTED)

    txt(ax, .018, .719, 'B. Measure which components help a conditional task',
        10.5, 'bold', TEAL)
    txt(ax, .018, .675,
        'Image + “The [MASK] holds the cup.” → predict the missing word.')
    for y, label, col in [(.611, r'$D_1$ mostly noise', MUTED),
                           (.551, r'$D_1$ clearer', BLUE)]:
        txt(ax, .038, y, label, color=col)
        arrow(ax, (.282, y), (.408, y), col)
    box(ax, .419, .526, .265, .111, PT, TEAL)
    txt(ax, .5515, .5815, 'Same conditional\ndenoiser', color=TEAL, ha='center')
    arrow(ax, (.696, .5815), (.746, .5815), TEAL)
    txt(ax, .761, .5815, 'Compare target\ndenoising loss', weight='bold')
    txt(ax, .018, .481,
        'Hold other inputs and target noise fixed; mask the caption before encoding.',
        color=MUTED)

    txt(ax, .018, .417, 'C. Dependencies guide timing', 10.5, 'bold', TEAL)
    txt(ax, .530, .417, 'D. Dependencies guide module access', 10.5, 'bold', ORANGE)
    # Five simultaneously evolving schedules, with no imposed semantic-first split.
    gax = fig.add_axes([.080, .180, .377, .152])
    t = np.linspace(0, 1, 401)
    specs = [(1-(1-t)**2.6, BLUE, '-', r'$D_1$'),
             (1-(1-t)**1.7, BLUE, '--', r'$D_2$'),
             (1-(1-t)**1.15, ORANGE, '-', r'$Q_1$'),
             (t**1.25, ORANGE, '--', r'$Q_2$'),
             (t**2.1, TEAL, '-', r'$V$')]
    for u, col, style, label in specs:
        gax.plot(t, u, color=col, ls=style, lw=1.6, label=label)
    gax.set(xlim=(0, 1), ylim=(0, 1.03), xticks=[0, 1], yticks=[0, 1])
    gax.set_xlabel(r'Generation time $t$', fontsize=10, labelpad=-4)
    gax.set_ylabel('Progress', fontsize=10, labelpad=-2)
    gax.tick_params(labelsize=10, length=2, pad=2, colors=MUTED)
    for edge in ['top', 'right']:
        gax.spines[edge].set_visible(False)
    for edge in ['left', 'bottom']:
        gax.spines[edge].set_color(RULE)
    gax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=5,
        fontsize=10, frameon=False, handlelength=1.2, columnspacing=.4,
        borderaxespad=0, handletextpad=.2)
    txt(ax, .018, .078, 'Curves illustrate overlapping schedules.', color=MUTED)

    # Follow a single contribution, rather than imply complete model explanation.
    txt(ax, .543, .324, r'$D_1$', 11, color=BLUE, ha='center')
    arrow(ax, (.571, .324), (.600, .324), BLUE, scale=8)
    box(ax, .610, .282, .127, .084, PO, ORANGE)
    txt(ax, .6735, .324, 'Head m', color=ORANGE, ha='center')
    arrow(ax, (.746, .324), (.770, .324), ORANGE, scale=8)
    box(ax, .780, .282, .120, .084, PO, ORANGE)
    txt(ax, .840, .324, 'MLP', color=ORANGE, ha='center')
    arrow(ax, (.909, .324), (.940, .324), ORANGE, scale=8)
    txt(ax, .969, .324, r'$Q_1$', 11, color=ORANGE, ha='center')
    txt(ax, .530, .215,
        'Turn the head off; restore its output.\nDoes the benefit of D₁ disappear\nand return?', color=INK)
    txt(ax, .530, .112,
        'Repeat across noise levels and tasks.', color=MUTED)

    ax.plot([.018, .982], [.046, .046], color=RULE, lw=.8)
    txt(ax, .5, .023,
        'Jointly learn schedules and transformer weights; representation groups remain fixed.',
        10, 'bold', MUTED, ha='center')
    return fig


def save(fig, basename):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'{basename}.{extension}', dpi=250, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    save(build_overview(), 'fig1_mechanism_overview')
    print('Created the integrated method figure at 7 x 4 inches; labels >= 10 pt.')
