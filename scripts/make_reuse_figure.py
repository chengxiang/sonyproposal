"""Reproduce Figure 3 from unchanged image panels and reported measurements.

Sources: experiment3_summary.pdf, pp. 2, 5, and 10 (provided by Xiang Cheng).
The two source PNGs are the original embedded images extracted from pp. 2 and 5.
Image coordinates are measured in their native 2099 x 1174 pixel grids.
No denoising, enhancement, resampling to invent details, or model runs are used.
The report already selected illustrative latents; we take its first displayed
row, adaptation seed 0 / latent 1, for both transfers. The numbers at right
come from the separately evaluated quantitative latent bank, not these panels.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

# Half-open boxes in original embedded PNG pixel coordinates, (left, top,
# right, bottom). First illustrated row, original columns 0, 6, 7.
# These boxes contain only generated-image pixels, excluding report labels.
PANELS = [(148, 120, 384, 356), (1602, 120, 1839, 356),
          (1845, 120, 2081, 356)]
SOURCES = [FIGURES / "source_transfer_aahq.png",
           FIGURES / "source_transfer_stl10.png"]
ROW_LABELS = ["CelebA → AAHQ", "CelebA → STL-10"]
RECOVERY = [0.815, 0.143]
COLORS = ["#257E86", "#C57440"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})

fig = plt.figure(figsize=(10, 4.5), facecolor="white")
fig.text(.035, .963, "Existing component-reuse evidence",
         fontsize=17, fontweight="bold", color="#1D3043", va="top")
fig.text(.035, .894,
         "20M models • post-training component reversion",
         fontsize=10.5, color="#536271", va="top")

# Equal-size source panels, displayed with nearest-neighbor interpolation.
# Slicing isolates existing scientific-image panels without altering content.
xs = [.047, .235, .423]
w = .130
h = w * 10 / 4.5
ys = [.463, .123]
headings = ["Source", "Source QK +\nadapted VO/MLP", "Jointly adapted"]
for x, heading in zip(xs, headings):
    fig.text(x + w/2, .821, heading, ha="center", va="center",
             fontsize=10.5, fontweight="bold", color="#263B4B")

for row, (path, label, y, color) in enumerate(zip(SOURCES, ROW_LABELS, ys, COLORS)):
    source = mpimg.imread(path)
    assert source.shape[:2] == (1174, 2099), (path, source.shape)
    fig.text(xs[0], y + h + .014, label, fontsize=10,
             fontweight="bold", color=color, va="bottom")
    for x, box in zip(xs, PANELS):
        left, top, right, bottom = box
        panel = source[top:bottom, left:right]
        ax = fig.add_axes([x, y, w, h])
        ax.imshow(panel, interpolation="nearest", aspect="auto")
        ax.set_axis_off()

# Same hybrid as the center image column, compared with the joint endpoint.
# P=1-E_hybrid/E_source; E is squared unit-normalized DINOv2 distance to the
# joint output for the same latent. Source has P=0, joint has P=1.
fig.text(.676, .821, "Recovery with source QK",
         ha="left", va="center", fontsize=11.5, fontweight="bold",
         color="#263B4B")
fig.text(.676, .773, "VO/MLP adapted; joint endpoint = 1",
         fontsize=9.2, color="#536271")
ax = fig.add_axes([.676, .183, .275, .508])
positions = [1, 0]
ax.barh(positions, RECOVERY, color=COLORS, height=.30, zorder=3)
ax.set_xlim(0, 1.03)
ax.set_ylim(-.45, 1.45)
ax.set_yticks(positions, ["AAHQ", "STL-10"])
ax.tick_params(axis="y", length=0, labelsize=10)
ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
ax.tick_params(axis="x", length=0, labelsize=9.5, colors="#536271")
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#CBD3D9")
ax.grid(axis="x", color="#E3E8EC", linewidth=.7, zorder=0)
for y, value, color in zip(positions, RECOVERY, COLORS):
    ax.text(value + .025, y, f"{value:.3f}", va="center", ha="left",
            fontsize=12, fontweight="bold", color=color)
ax.set_xlabel("Normalized DINOv2 recovery", fontsize=9.5,
              color="#536271", labelpad=6)

fig.text(.035, .067,
         "Images: first illustrated noise input from each report. Scores: 1,024 matched latents per adaptation seed,",
         fontsize=8.3, color="#536271", va="top")
fig.text(.035, .032,
         "averaged over 2 seeds; scores are not computed from the displayed images alone.",
         fontsize=8.3, color="#536271", va="top")

FIGURES.mkdir(parents=True, exist_ok=True)
for extension in ["png", "svg"]:
    fig.savefig(FIGURES / f"fig3_component_reuse.{extension}", dpi=240)
plt.close(fig)
