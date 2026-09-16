"""Reproduce Figure 3 from unchanged image panels and reported measurements.

Sources: experiment3_summary.pdf, pp. 2, 5, and 10 (provided by Xiang Cheng).
The source PNGs are original embedded images extracted from pp. 2 and 5.
Crops use their native 2099 x 1174 pixel grids; no image enhancement is used.
AAHQ uses the first displayed row, adaptation seed 0 / latent 1. STL-10 uses
its third displayed row, adaptation seed 1 / latent 3, for all three columns.
Reported recovery averages come from a separately evaluated latent bank.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
INK, MUTED, RULE = "#203342", "#536575", "#C9D3DC"
TEAL, ORANGE = "#167E83", "#C8782F"

# Half-open native-pixel boxes: (left, top, right, bottom). Columns 0, 6, 7
# correspond to source, source QK with adapted VO/MLP, and jointly adapted.
# Only the generated-image pixels are included; report labels are excluded.
PANELS_AAHQ = [(148, 120, 384, 356), (1602, 120, 1839, 356),
               (1845, 120, 2081, 356)]
PANELS_STL10 = [(148, 653, 384, 889), (1602, 653, 1839, 889),
                (1845, 653, 2081, 889)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def build_reuse():
    """One shared 7 x 2.35 inch builder for the PNG, SVG, and print PDF."""
    fig = plt.figure(figsize=(7, 2.35), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def label(x, y, value, *, weight="normal", color=INK,
              ha="left", va="center", size=10):
        return ax.text(x, y, value, fontsize=size, fontweight=weight,
                       color=color, ha=ha, va=va, linespacing=1.12)

    xs = [.137, .308, .479]
    w = .137
    h = w * 7 / 2.35
    ys = [.455, .018]
    for x, heading in zip(xs, ["Source", "Source QK +\nadapted VO/MLP",
                              "Jointly\nadapted"]):
        label(x + w / 2, .931, heading, weight="bold", ha="center")

    rows = [
        ("source_transfer_aahq.png", "CelebA\n→ AAHQ", PANELS_AAHQ, TEAL),
        ("source_transfer_stl10.png", "CelebA\n→ STL-10", PANELS_STL10, ORANGE),
    ]
    for (filename, row_label, boxes, color), y in zip(rows, ys):
        label(.013, y + h / 2, row_label, color=color, weight="bold")
        source = mpimg.imread(FIGURES / filename)
        assert source.shape[:2] == (1174, 2099), (filename, source.shape)
        for x, (left, top, right, bottom) in zip(xs, boxes):
            image_ax = fig.add_axes([x, y, w, h])
            image_ax.imshow(source[top:bottom, left:right],
                            interpolation="nearest", aspect="auto")
            image_ax.axis("off")

    # Compact table replaces the two-bar plot. Values summarize the hybrid
    # center column against the jointly adapted endpoint, across matched
    # quantitative latents. P = 1 - E_hybrid / E_source; E is squared distance
    # in unit-normalized DINOv2 features. Thus source=0 and joint=1.
    left, right = .663, .985
    label(left, .924, "Recovery with source QK", weight="bold")
    label(left, .788, "Normalized DINOv2 recovery\nJoint endpoint = 1", color=MUTED)
    ax.plot([left, right], [.664, .664], color=RULE, lw=.8)
    label(left + .008, .592, "Target", weight="bold")
    label(right - .008, .592, "Recovery", weight="bold", ha="right")
    ax.plot([left, right], [.526, .526], color=RULE, lw=.8)
    label(left + .008, .431, "AAHQ", color=TEAL)
    label(right - .008, .431, "0.815", color=TEAL, weight="bold", ha="right")
    label(left + .008, .278, "STL-10", color=ORANGE)
    label(right - .008, .278, "0.143", color=ORANGE, weight="bold", ha="right")
    ax.plot([left, right], [.196, .196], color=RULE, lw=.8)
    return fig


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure = build_reuse()
    for extension in ["png", "svg"]:
        figure.savefig(FIGURES / f"fig3_component_reuse.{extension}", dpi=240)
    plt.close(figure)
    print("Created Figure 3 at 7 x 2.35 inches; all labels are at least 10 pt.")
