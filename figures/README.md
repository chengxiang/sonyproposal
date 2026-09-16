# Proposal figure sources

The current proposal uses **two figures**: one integrated method schematic and one compact qualitative result. PNG versions are embedded in Markdown; matching SVGs retain editable text and vector elements. Print PDFs use the same builders at seven inches wide with labels of at least 10 points. No model was trained, sampled, or newly evaluated to make these figures.

| Active figure | Content | Evidence status |
|---|---|---|
| `fig1_mechanism_overview` | Fixed representation groups, a conditional-information diagnostic, overlapping schedules, and a selected transformer information path | Proposed method; curves and the selected component–module path are illustrative |
| `fig2_component_reuse` | Existing matched image panels and a two-row recovery table | Experiment 3 report, pages 2, 5, and 10; unchanged images and measurements |

## Integrated method figure

Figure 1 replaces the previous separate overview and schedule illustrations. It is **7 × 4 inches**. `scripts/make_conceptual_figures.py` supplies the common builder for PNG, SVG, and PDF outputs.

- **Panel A:** Initial semantic states consist of projected visual-model and caption-model activations. Two fixed coordinate groups of each give D1/D2 and Q1/Q2; V denotes the VAE image-latent state. The representation encoders and grouping are prepared before denoising/dependency learning and held fixed during that stage. Their *states* still evolve through generation. These groups are not assigned human-readable meanings such as “woman” or “giver.” The diagram also allows the subsequent planned alternative using activations from a model trained on a downstream task.
- **Panel B:** The incomplete caption is masked **before** its conditioning encoder runs. Denoising loss for the caption target is compared when visual component D1 is mostly noise versus clearer, holding the denoiser, other inputs, and target noise fixed. The full caption's clean representation provides a training target, not an unmasked conditioning path. The masked word illustrates the conditional task; word prediction can be a functional readout, while the plotted method's diagnostic uses target denoising loss. This establishes a component's conditional usefulness without claiming a unique semantic label or indispensability.
- **Panel C:** Five illustrative, overlapping component-progress schedules have common endpoints. No semantic-first split or strict discrete generation order is imposed. Conditional usefulness informs timing, and schedules are learned jointly with transformer weights. Earlier information can improve another component's prediction while making the early component's own prediction harder. The curves are sketches, not learned results.
- **Panel D:** A selected D1 → attention head → MLP → Q1 contribution illustrates the proposed mechanistic study. Other inputs, residual paths, and target projections are omitted for clarity. The contribution of D1 is compared with the head active, disabled, and restored, across noise configurations and conditional tasks. The diagram explains one information path; it does not claim to explain the entire pretrained backbone. This analysis motivates dependency-guided access and shared computation, subject to the proposed causal controls.

Suggested concise caption:

> **One dependency connects representation, denoising, and transformer computation.** Fixed groups of visual and caption features provide components whose meaning is measured through conditional prediction. Making one visual group clearer may help denoise a masked caption (B), motivating both component timing (C) and selective transformer access (D). Removing and restoring a selected head tests whether it carries that benefit. Component meanings, curves, and the illustrated path are hypothetical; the encoders and grouping remain fixed while schedules and transformer weights are learned.

## Component-reuse figure

The untouched embedded images from `experiment3_summary.pdf` are retained as `source_transfer_aahq.png` (page 2) and `source_transfer_stl10.png` (page 5). Both images are 2099 by 1174 pixels. The compact figure selects **seed 0 / latent 1 for AAHQ** and **seed 1 / latent 3 for STL-10**, the latter showing a recognizable airplane in the joint output. All three columns within each domain use the same report row. It extracts these panels without enhancement or regenerated content:

| Transfer / selected row | State | Half-open pixel box `(left, top, right, bottom)` |
|---|---|---|
| AAHQ, seed 0 / latent 1 | Source | `(148, 120, 384, 356)` |
| AAHQ, seed 0 / latent 1 | Source QK with adapted VO and MLP | `(1602, 120, 1839, 356)` |
| AAHQ, seed 0 / latent 1 | Jointly adapted | `(1845, 120, 2081, 356)` |
| STL-10, seed 1 / latent 3 | Source | `(148, 653, 384, 889)` |
| STL-10, seed 1 / latent 3 | Source QK with adapted VO and MLP | `(1602, 653, 1839, 889)` |
| STL-10, seed 1 / latent 3 | Jointly adapted | `(1845, 653, 2081, 889)` |

These rows are selected illustrations, not random or exhaustive samples. The complete original grids remain available above. Figure 2 is 7 inches wide and 2.35 inches high, with labels of at least 10 points. The PNG, SVG, and PDF share `build_reuse()` in `scripts/make_reuse_figure.py`.

The tabulated values 0.815 (CelebA to AAHQ) and 0.143 (CelebA to STL-10) are **separate quantitative results**, not measurements on the displayed images alone. For each hybrid H, the report defines recovery as `1 - E_H / E_source`, where E is mean squared distance to paired joint-model outputs in unit-normalized frozen DINOv2 features. Each state uses 1,024 matched latents per adaptation seed, and the figure shows the mean of two seed scores. Joint checkpoints were selected using existing test FID. These are 20M-model post-training component-reversion results; they do not establish success of training with QK fixed, nor do they measure target-distribution fidelity.

## Historical figures and evidence

The earlier `fig2_recovery_schedule`, `fig3_component_reuse`, and `fig4_existing_feasibility` files remain historical assets and are **not included in the current proposal**. Figure 4 and its language-model bar chart have been removed from the active build. Its old numbers must not be used for the updated language result: the PI supplied 62.4716% GSM8K accuracy with an approximately 800M-parameter denoiser and a 121M-parameter encoder. That updated result belongs in short proposal prose, not a new chart.

`feasibility_data.json` retains the prior figure's source record. LWD and parameter-sharing results are presented in proposal prose; their exact source settings remain documented in the proposal references and the supplied experiment reports.

## Reproduce

From the repository root, with Python, Matplotlib, NumPy, and Pillow installed:

```bash
python scripts/make_conceptual_figures.py
python - <<'PYCODE'
import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts')
from make_reuse_figure import build_reuse
fig = build_reuse()
for ext in ('png', 'svg'):
    fig.savefig(Path('figures') / f'fig2_component_reuse.{ext}', dpi=240)
plt.close(fig)
PYCODE
python scripts/make_latex_figures.py
```

The two active print outputs are `latex/figures/fig1_mechanism_overview.pdf` and `latex/figures/fig2_component_reuse.pdf`. The reuse builder preserves the original selected source panels and reported values; the older standalone PNG script keeps its historical filename. No GPU, model checkpoint, new sampling, or additional experiment is needed.
