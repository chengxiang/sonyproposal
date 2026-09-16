# Proposal figure sources

The proposal embeds PNG versions. Matching SVG files preserve editable text and vector elements for final layout. These figures contain schematic methods or existing results; no model was trained or evaluated to make them.

| Figure | Content | Evidence status |
|---|---|---|
| `fig1_mechanism_overview` | Concrete token-coordinate examples connect representation, denoising schedules, and conditional transformer computations through labeled relationships | Proposed program; coordinate meanings, shared attention, and task-specific MLPs are illustrative |
| `fig2_recovery_schedule` | A conditional dependency DAG, corresponding overlapping schedules, and three snapshots of gated transformer modules | Illustrative decomposition, factorization, schedules, and module roles; no learned graph or experimental result is claimed |
| `fig3_component_reuse` | Existing image panels plus reported recovery scores | Experiment 3 report, pages 2, 5, and 10 |
| `fig4_existing_feasibility` | Continuous latent diffusion language reasoning in published benchmark context | Language experiment summary; user-provided exact denoiser size; Nie et al., Table 1 |

## Conceptual figures

Figures 1–2 show the three connected research aspects. Section 2.1 learns representations and component extractors from fixed encoders or downstream tasks/rewards. Section 2.2 jointly learns denoising order, component dependencies, and the transformer: diagnostics compare noise conditions at selected current checkpoints, inform joint schedule/transformer updates, and repeat as the model changes. Holding the transformer fixed applies only within each diagnostic comparison; dependency measurement is not a frozen-model phase preceding schedule learning. Section 2.3 develops shared and specialized transformer computations, including noise-dependent module gates, sparse attention, and shared or switched MLPs across conditional denoising tasks. The main application is controllable generation satisfying complex prompt requirements: a woman in a red coat gives a blue cup to a man in a green sweater, while a child beside them reads a book. The example illustrates jointly realizing participants, attributes, and relationships; the representation decomposition is learned and need not follow these human labels. The figures do not introduce a separate model-merging, generated-weight-update, or meta-learning aim.

Both PNG/SVG versions and the print PDFs use the same builders in `scripts/make_conceptual_figures.py`. All versions are exactly 7 inches wide, with nominal text sizes of at least 10 points (ordinary mathematical subscripts scale normally). Figure 1 is 5.5 inches high and Figure 2 is 5.2 inches high. Figure 1 illustrates selected coordinates of participant tokens 1–2, a relation component R (giver, receiver, object), and a visual component V (the hand–cup interaction). These assignments are a toy interpretation; the actual component extractors are learned and may be more complex than coordinate selection. Representation-to-schedule arrows describe making informative components available for other predictions. The reverse arrow describes discovering useful dependencies by changing noise levels. A representation-to-architecture arrow assigns conditional inputs and targets to shared attention and specialized MLPs: denoise V given tokens 1–2 and R, or R given tokens 1–2 and V. Both tasks also receive their noisy target. The bottom connector specifies joint schedule/transformer training. Its vectors and conditional architecture are illustrative, not experimental measurements. Figure 1 summarizes the connections; Figure 2 works through one hypothetical ordering.

Figure 2 conditions its DAG on the prompt “woman gives the blue cup to the man.” Participant and cup components feed clothing bindings and the visual handover. Blue schedules advance participants/cup over [0, 0.4], orange schedules advance clothing bindings over [0.2, 0.65], and teal schedules advance handover states over [0.55, 1]. The intervals overlap. Snapshots at generation times 0.1, 0.5, and 0.85 repeat the same transformer layout and show selected attention heads and MLP slices active for the corresponding predictions. Gray modules are gated off; shared backbone computations are omitted. The DAG represents illustrative conditional information use, not a discovered causal graph or a required fixed ordering.

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

The exact plotted values and settings are recorded in `feasibility_data.json`. The PNG, SVG, and PDF share the builder in `scripts/make_feasibility_figure.py`; each is 7 inches wide and 2.65 inches high, with labels of at least 10 points.

- **Continuous diffusion language model (ours):** `elf_l_grant_summary.md`, prepared 12 September 2026, epoch-12 comparison. The displayed 61.9409% is reversed asynchronous inference: 1,634/2,638 trials. Synchronous inference on the same checkpoint gives 1,429/2,638 = 54.1698%. Both use EMA 0.9999, 32 ODE steps, and inference seeds 42/123 over the same 1,319 distinct GSM8K questions; each trial is scored separately, without majority voting or best-of-two selection. The 795,205,824-parameter denoiser (exact count supplied by the PI) is trained on mathematical question–solution data. A frozen Qwen3-4B-Instruct-2507 encodes the prompt once; the continuous diffusion model generates answer representations and decodes them to tokens. The 795M count therefore describes the denoiser, not the whole inference pipeline.
- **Published language-model context:** [Nie et al., *Large Language Diffusion Models*, arXiv v3, Table 1](https://arxiv.org/html/2502.09992v3) reports GSM8K 48.7% for Llama-3-8B Base (autoregressive) and 70.3% for LLaDA-8B Base (discrete diffusion), both with four-shot prompting. The chart labels these as published context because training data, conditioning, model size, and evaluation settings differ from our mathematical language experiment. It gives a familiar scale for the continuous diffusion result, not a controlled ranking of model families or a claim of superior parameter efficiency.
- **LWD (reported in prose, not plotted):** [*Learning When to Denoise*, arXiv v1, Table 2](https://arxiv.org/html/2606.19662v1), also reported in the supplied manuscript `19994_Learning_When_to_Denoise.pdf`, and [SFD Table 2](https://arxiv.org/html/2512.04926v1). The proposal reports AutoGuidance FID 1.05 at 200 LWD epochs versus 1.06 at 800 SFD-XL epochs, with matched 675M backbones and dopri5 sampling. These are documented epoch budgets, not a wall-clock benchmark. The public LWD paper additionally reports FID 1.02 at 600 epochs for its 675M model versus 1.04 at 800 epochs for 1B-parameter SFD-XXL; these final results are recorded in the JSON but are not additional bars. The proposal's REPA comparison comes from LWD Table 1: 120K main updates/FID 4.93 versus 4M/FID 5.84, plus a 10K-update LWD schedule-learning probe.
- **Parameter sharing (reported in prose, not plotted):** `experiment1.pdf`, page 6. DiT F1024: 10,215,472 parameters, FID-50k 17.574. Shared DiT-MLP H14/d32/W256: 10,239,536 parameters, FID-50k 14.304. CelebA64, one training seed, prespecified epoch-400 EMA, common 50-step sampler. The proposal uses the page-6 table; the original page-5 plots omit the H14 model.

## Reproduce

From the repository root, with Python, Matplotlib, NumPy, and Pillow installed:

```bash
python scripts/make_conceptual_figures.py
python scripts/make_latex_figures.py --figures 1 2
python scripts/make_reuse_figure.py
python scripts/make_feasibility_figure.py
python scripts/make_latex_figures.py --figures 3 4
```

The scripts resolve paths relative to the repository. No GPU, source-model checkpoint, new sampling, or additional experiment is needed. Original source reports remain the authority for interpretation; cropping coordinates and selected states are explicit above.
