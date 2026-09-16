#!/usr/bin/env python3
"""Rebuild the proposal's two active, compact print figures.

Each PDF is seven inches wide with labels of at least 10 points. The method
schematic is illustrative. Reuse images and measurements are unchanged from
previous supplied experiments. Historical figures are not rebuilt or deleted.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'latex' / 'figures'
QA = ROOT.parent / 'latex_figure_qa'


def save(fig, basename):
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(exist_ok=True)
    # Preserve physical size and text scale across PNG/SVG/PDF outputs.
    fig.savefig(OUT / f'{basename}.pdf', facecolor='white')
    fig.savefig(QA / f'{basename}.png', dpi=180, facecolor='white')
    plt.close(fig)


def method():
    from make_conceptual_figures import build_overview
    save(build_overview(), 'fig1_mechanism_overview')


def reuse():
    from make_reuse_figure import build_reuse
    save(build_reuse(), 'fig2_component_reuse')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', nargs='+', type=int, choices=[1, 2],
                        default=[1, 2], help='Active figure numbers to rebuild')
    selected = parser.parse_args().figures
    for number in selected:
        {1: method, 2: reuse}[number]()
    print(f'Created seven-inch-wide PDF figures {selected}; labels >= 10 pt.')
