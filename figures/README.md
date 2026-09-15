# Proposal figure sources

The proposal embeds PNG versions. Matching SVG files preserve editable text and vector elements for final layout. These figures contain schematic methods or existing results; no model was trained or evaluated to make them.

| Figure | Content | Evidence status |
|---|---|---|
| `fig1_mechanism_overview` | Semantic components, network computations, and asynchronous revision | Proposed mechanism; all vectors and schedules are schematic |
| `fig2_recovery_schedule` | Fixed-target-noise recovery comparison and crossing schedules | Proposed measurement; no loss values, empirical dependency graph, or learned schedules are fabricated |
| `fig3_component_reuse` | Existing image panels plus reported recovery scores | Experiment 3 report, pages 2, 5, and 10 |
| `fig4_existing_feasibility` | Language-state generation, schedule learning, and parameter sharing | ELF-L summary, LWD paper, and Experiment 1 report |

## Component-reuse figure

The untouched embedded images from `experiment3_summary.pdf` are retained as `source_transfer_aahq.png` (page 2) and `source_transfer_stl10.png` (page 5). Both images are 2099 by 1174 pixels. The compact figure selects the **first report-illustrated row, seed 0 / latent 1**, for both domains. It extracts these columns without enhancement or regenerated content:

| State | Half-open pixel box `(left, top, right, bottom)` |
|---|---|
| Source | `(148, 120, 384, 356)` |
| Source QK with adapted VO and MLP | `(1602, 120, 1839, 356)` |
| Jointly adapted | `(1845, 120, 2081, 356)` |

The first row was selected for a compact, matched illustration; it is not claimed to be a random or exhaustive visual sample. The complete original grids remain available above.

The plotted values 0.815 (CelebA to AAHQ) and 0.143 (CelebA to STL-10) are **separate quantitative results**, not measurements on the displayed images alone. For each hybrid H, the report defines recovery as `1 - E_H / E_source`, where E is mean squared distance to paired joint-model outputs in unit-normalized frozen DINOv2 features. Each state uses 1,024 matched latents per adaptation seed, and the figure shows the mean of two seed scores. Joint checkpoints were selected using existing test FID. These are 20M-model post-training component-reversion results; they do not establish success of training with QK fixed, nor do they measure target-distribution fidelity.

## Feasibility figure

The exact plotted values and settings are recorded in `feasibility_data.json`.

- **ELF-L:** `elf_l_grant_summary.md`, prepared 12 September 2026, matched epoch-12 comparison. Synchronous inference: 1,429/2,638 trials = 54.1698%; reversed asynchronous inference: 1,634/2,638 = 61.9409%. Both use the same synchronously trained checkpoint, EMA 0.9999, 32 ODE steps, and inference seeds 42/123 over the same 1,319 distinct GSM8K questions. The comparison uses no extra training. Qwen encodes the prompt once; ELF generates the answer states and decodes the tokens. This is not an AR-baseline comparison or an estimate of end-to-end latency.
- **LWD:** [Learning When to Denoise, arXiv:2606.19662v1](https://arxiv.org/html/2606.19662v1), abstract and Section 4.3. AutoGuidance FID 1.05 at 200 LWD epochs versus 800 SFD-XL epochs, with a matched 675M backbone. The figure shows documented epoch budgets, not a fabricated convergence curve or wall-clock benchmark.
- **Parameter sharing:** `experiment1.pdf`, page 6. DiT F1024: 10,215,472 parameters, FID-50k 17.574. Shared DiT-MLP H14/d32/W256: 10,239,536 parameters, FID-50k 14.304. CelebA64, one training seed, prespecified epoch-400 EMA, common 50-step sampler. The original page-5 plots omit the H14 model, so these bars are redrawn from the page-6 table.

## Reproduce

From the repository root, with Python, Matplotlib, NumPy, and Pillow installed:

```bash
python scripts/make_conceptual_figures.py
python scripts/make_reuse_figure.py
python scripts/make_feasibility_figure.py
```

The scripts resolve paths relative to the repository. No GPU, source-model checkpoint, new sampling, or additional experiment is needed. Original source reports remain the authority for interpretation; cropping coordinates and selected states are explicit above.
