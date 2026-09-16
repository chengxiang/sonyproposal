# Proposal review: discussion and revision record

Last updated: 2026-09-16.

## Purpose and working agreement

This document tracks our discussion of the criticisms in [the independent review](review_independent.md). We will work through them individually, then make a coordinated revision of the proposal. **Updating this record does not authorize or imply changes to the proposal itself.**

The proposal is currently held at commit `81e7d2e674e9484d4537baa2d0b1d4f093aee986`. The independent review examined the earlier draft at `ae9b6b50e746cec2665204b2abeb4fc839dbcc70`; some clarity and figure edits were made before this discussion began. We will check the current draft before applying any remaining review suggestions.

We distinguish an **agreed direction** from a **candidate method** and an **open decision**. Agreement that a discussion “looks reasonable” does not settle every implementation choice. Proposed studies below concern the award period; no additional preliminary experiments are required before submission.

## Criticism tracker

Numbering follows the review's eight ranked criticisms, rather than its shorter concluding list of revision priorities.

| No. | Criticism | Discussion status |
|---|---|---|
| 1 | Predictive dependence and module ablation do not, by themselves, establish the promised semantic mechanism. | Direction agreed; candidate methods recorded below; implementation choices remain open. |
| 2 | No decisive evaluation isolates the value of learning representations, denoising, and transformer computations together. | Working table accepted for now. Criticism 3 narrows the claim; corresponding H1/H4 revisions are recorded below for discussion. |
| 3 | The representation-refinement algorithm is underspecified. | Scope decision agreed: obtain representations from pretrained or task-trained models, then hold them fixed during dependency/schedule learning. Remove representation refinement from that stage. |
| 4 | Useful information in corrupted ground-truth states may not remain useful in generated states. | Treat as a bounded technical risk; add one short diagnostic paragraph and mention rollout-based training as a possible mitigation. |
| 5 | Integration, resources, and the minimum 12-month scope need firmer bounds. | User supports compute funding, lower PI salary support, bounded data/models/runs, and scaling gates. Proposed budget and execution plan recorded below. |
| 6 | The differentiation does not yet isolate one new scientific relationship. | Gap-first positioning agreed. User proposes a representation/dependency/module pipeline; refinements to the central connection and scope wording are recorded below. |
| 7 | Preliminary evidence takes more space than its direct support for the proposal warrants. | User accepts compression; supplies a new language result; retain the REPA comparison and omit new multi-seed experiments. Details below. |
| 8 | Mathematical detail is disproportionate to the unresolved method choices. | User agrees to remove irrelevant technical detail, especially stop-gradient notation; preserve self-contained definitions of the proposed method. |
| 9* | Figures occupy too much space. | Delete Figure 4, keep simple numerical comparisons in prose/tables, and consider merging Figures 1/2 and selective text wrapping. |

Organization, figures, and submission cleanup are cross-cutting review comments. They will be revisited during the final editing pass. *The independent review has eight ranked criticisms; “criticism 9” is the user's label for its figure/layout comments, and is tracked that way below for continuity.*

## 1. Give components and modules an operational meaning

### The concern

Showing that a cleaner component improves denoising establishes its usefulness to a particular fitted model under specified conditions. It does not automatically identify its semantic content. Likewise, removing a transformer module can damage many predictions without identifying a specific computation. Parallel information paths, including text conditioning, also limit claims about the entire generator.

### Agreed direction: functional meaning without mandatory human labels

We do **not** require every learned component to have a simple human-readable label. A useful decomposition may have no such interpretation. However, it must have an operational meaning: **which conditional predictions become possible, improve, or change when the component is available?**

The user's example is to remove “woman” from a sentence and reconstruct the missing text through image-to-text diffusion. We can investigate which image-derived representation components enable that prediction. Repeating this across different conditional tasks gives each component a functional characterization, without assigning it a universal semantic name.

This characterization is relative to the model, task, remaining context, and noise configuration. Components may carry redundant information or be useful only in combination.

### Candidate method: optimize information availability

The user proposed penalizing low noise levels while optimizing conditional prediction. This offers a continuous alternative to searching all component subsets.

Let $s_k$ be representation component $k$, with $k=1,\ldots,K$, and let $\varepsilon_k$ be independent Gaussian noise of the same dimension. Define the available version of that component by

```math
\widetilde{s}_k(u_k)=u_k s_k+(1-u_k)\varepsilon_k,
\qquad 0\leq u_k\leq 1.
```

Here $u_k=0$ exposes only noise and $u_k=1$ exposes the clean component. For a fixed conditional prediction task $\mathcal T$, a starting diagnostic is

```math
\min_{\mathbf u\in[0,1]^K}
\mathcal L_{\mathcal T}(\mathbf u)
+\lambda\sum_{k=1}^{K}u_k.
```

$\mathcal L_{\mathcal T}$ is the expected prediction loss over task examples and noise draws, $\mathbf u=(u_1,\ldots,u_K)$ controls the candidate conditioning components, and $\lambda>0$ penalizes exposing cleaner information. The prediction target's masking or noise level remains fixed. Components retained at lower noise despite the penalty provide useful information for the task.

Implementation suggestions from the discussion, still to be finalized:

- Start with a fixed representation and a fixed denoiser trained to handle independently varied component noise levels. Normalize component scales so the diagnostic does not reward arbitrary rescaling.
- Learn an allocation across many examples of a task, then assess it on unseen examples. An allocation optimized separately using each example's answer could itself leak answer information.
- Interpret the result as an economical sufficient allocation, rather than a unique list of indispensable components. Redundant alternatives and jointly useful groups may exist.
- The penalty $\sum_k u_k$ limits total exposure to clean information; it does not directly guarantee a small number of exposed components. Stochastic on/off gates are a possible alternative if component-count sparsity becomes the objective.

Following the decision under criticism 3, this diagnostic characterizes components of a fixed representation. The earlier suggestion that it might refine the representation is superseded. Representation learning occurs through the pretrained or downstream-task model, rather than through this diagnostic or the schedule-learning loss.

### Candidate method: learn module availability and connect it to component effects

The user suggested learnable dropout probabilities for transformer modules. Candidate modules include individual attention heads, attention or MLP branches, and specified slices of their computations; we have not chosen the initial granularity.

Task- and noise-dependent keep probabilities, together with a penalty on active computation, could identify a small set of useful modules. With weights fixed, this diagnoses existing computation. With weights trained jointly, it constructs a computation that works under the restriction. Neither result alone fully explains what a module computes.

The more informative proposed bridge is a **component–module interaction**:

1. Measure how much exposing component $X$ improves prediction of component $Y$.
2. Repeat with candidate module $m$ disabled.
3. Determine whether the benefit of $X$ is specifically reduced, while unrelated predictions remain usable.
4. Restore the module contribution and check whether the benefit returns.

This could support the bounded statement that a module helps transmit information from $X$ into predictions of $Y$ at particular noise levels. Generic damage from ablation is insufficient. Matched control interventions and restoration can help distinguish a specific information path from broad disruption.

### Open decision: text conditioning and T5

The user is uncertain about retaining the current T5 integration and is willing to reconsider it. **No decision to remove T5 has been made.**

The discussion distinguished two cases:

- For the masked-text diagnostic, remove the word **before contextual encoding**, so other text states cannot retain information from the complete sentence. Use the same incomplete text with different images, and ensure that relevant visual evidence enters through the components being studied.
- For generation from a complete prompt, the text encoder legitimately supplies semantic information. An added representation path can still improve generation, but its contribution does not establish a complete explanation of the generator's behavior.

The working recommendation is to make the core mechanistic study independent of a particular text encoder and settle the integration after defining the intended information paths. Removing T5 from a pretrained generator would itself require adaptation; it is not a cost-free simplification.

### Agreed addition: causal case studies

The user agrees that the proposal should include causal case studies. One candidate is:

1. Use matched scenes with the same participants and object but reversed giver and receiver roles.
2. Ask the same incomplete-caption question, such as “The [MASK] is giving the cup.”
3. Identify components useful for predicting the giver through the information-availability diagnostic.
4. Swap a candidate component between the matched scenes and examine whether the predicted giver changes while unrelated properties, such as clothing or object identity, are preserved.
5. Intervene on candidate transformer modules to identify which enables that component's effect, including restoration of the contribution.

These are proposed award-period studies, not new experiments to run before submission. The initial mechanistic goal is to explain task-specific information paths, their dependence on denoising stage, and their possible reuse. A human-readable account of every component or a complete interpretation of every weight is unnecessary.

### Changes to carry into the final revision

- Define a decomposition through its effects on conditional prediction and generation, without requiring a semantic label for every component.
- Give the information-availability objective as a concrete initial diagnostic, with its scope and symbols defined.
- Connect component usefulness to module interventions; avoid treating two separate importance rankings as a mechanistic explanation.
- Include one complete causal case study and state which part of the generator it explains.
- Resolve the conditioning paths and module granularity before committing to detailed T5 or dropout machinery.

### References considered in this discussion

These are feasibility precedents, not claims that the proposed synthesis is already established:

- [Restricting the Flow: Information Bottlenecks for Attribution](https://arxiv.org/abs/2001.00396): adding feature noise to study information useful for prediction.
- [Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312): learnable stochastic gates for sparse computation.
- [Analyzing Multi-Head Self-Attention](https://aclanthology.org/P19-1580/): head specialization and pruning.
- [How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255): interpreting interventions in the presence of redundancy and interacting computations.

## 2. Evaluate the contribution of learning the three elements together

### User's direction

A hypothesis-and-test table would be useful, with an indication of which hypotheses already have partial support from existing or preliminary work. The table should connect the proposed investigation to that evidence without suggesting that the full synthesis has already been demonstrated.

The user accepted the working table for now. The subsequent decision under criticism 3 removes representation refinement from the denoising stage. The table below reflects that narrower scope; the revised H1/H4 comparisons are proposed consequences of the decision, not newly finalized implementation commitments.

### Proposed central claim

Conditional dependencies within a fixed semantic representation provide a useful basis for designing denoising schedules and transformer computations. Learning to exploit those dependencies should improve generation satisfying interacting semantic requirements.

The project studies representations, denoising, and transformer computations together, but does not jointly optimize all three. The representation is obtained first and then held fixed. Schedules and the denoising transformer are learned together, with dependency measurements informing the proposed module organization. The earlier comparison involving repeated refinement of the representation is superseded by criticism 3.

### Proposed hypothesis-and-test table

| Hypothesis | Focused award-period comparison | Outcome that would support it | Existing support and remaining step |
|---|---|---|---|
| **H1. Activations from relevant task models provide useful representations for conditional generation.** | Compare pretrained features with features obtained after training or adapting a model for a relevant downstream task. Freeze each representation before dependency/schedule learning; use common evaluation targets and comparable representation dimensions and generation budgets. | Useful information for the specified conditional predictions, characterized through the availability diagnostic; improvements in held-out semantic prediction or generation. | **Partial feasibility:** the continuous diffusion language-model result supports generating useful Qwen contextual states. The additional benefit of reward/discriminator/task-trained representations for the proposed visual requirements remains to be established. |
| **H2. Denoising schedules should adapt to the representation's conditional prediction requirements.** | Hold the representation and architecture fixed; compare learned asynchronous schedules with fixed synchronous or asynchronous schedules, training a denoiser for each. | Better generation at comparable training and sampling cost; in the new multimodal setting, better simultaneous satisfaction of prompt requirements. | **Demonstrated in a narrower setting:** LWD jointly learns a schedule and denoiser for two fixed representation groups, improves generation, and obtains different schedules with different semantic encoders. Extending this to multiple fixed components and their measured conditional dependencies remains proposed. |
| **H3. Representation dependencies can guide useful and identifiable transformer computations.** | Compare dependency-guided access, sharing, or activity with comparably sized trainable modules whose organization does not use those dependencies. Keep representation and schedule fixed for this comparison. | Specific module interventions alter the benefit of exposing one component when predicting another; useful computations transfer across held-out conditional tasks or compositions. | **Partial evidence:** the attention/MLP analysis, sharing experiments, and selective weight replacement support differentiated and reusable computations. Their correspondence with semantic representation components and noise-dependent conditional tasks remains proposed. |
| **H4. Using a fixed representation's dependencies to organize denoising and transformer computation improves the resulting generator.** | Within one fixed representation, compare the integrated method with the focused variants from H2/H3 that omit schedule adaptation or dependency-guided module organization, under comparable supervision, capacity, and training budgets. | Better joint satisfaction of prompt requirements on held-out combinations, using generated intermediate states, with image quality and generation cost accounted for. | **Proposed synthesis:** existing work supports ingredients and some pairwise connections. No supplied result yet establishes the complete interaction. |

### Existing evidence worth emphasizing

The most directly relevant published evidence for H2 is in [LWD, Section 4.4 and Table 3](https://arxiv.org/html/2606.19662v1#S4.SS4). At 400,000 main-training iterations, unguided ImageNet-256 FID improves from 3.53 to 2.87 using SemVAE semantic latents, and from 4.06 to 2.97 using DINO-PCA latents. The learned schedule shapes also differ across semantic encoders. These results support adapting generation to the representation, without establishing that the resulting schedules identify a unique dependency structure. The schedule-learning phase is an additional cost to report when comparing total compute. The CLIP comparison changes the semantic compression method as well, so it is less clean for isolating schedule effects.

For H1, use the updated result reported by the user under criticism 7: **62.4716% GSM8K accuracy with an approximately 800M-parameter continuous diffusion model and the team's own 121M-parameter encoder**, using Qwen-3-derived latent states. This supersedes the earlier 61.94% result and its encoder description. It supports generating functionally useful contextual language states; it does not establish the additional benefit of task-trained visual representations or their visual grounding.

For H3, the preliminary CelebA64 sharing result (FID 17.574 to 14.304 at approximately 10.2M parameters) and source/adapted weight-replacement experiments support investigating reusable transformer computations. They do not yet identify a semantic component-to-module mechanism. These are the existing results summarized in [the current proposal](proposal_draft.md); no new preliminary results are claimed.

### Keep the evaluation focused

- Use one common creator-facing endpoint: the fraction of generated scenes satisfying **all** specified requirements on held-out combinations of participants, attributes, and relations. Report individual requirement scores, image quality, and cost to explain changes in that endpoint.
- Define held-out combinations so the relevant participants, attributes, and relations are individually encountered during training, while selected combinations are withheld.
- Train the transformer in every generative baseline. The third design choice concerns module access, sharing, and activity; it is not a comparison of a trained denoiser with an untrained or arbitrarily frozen one.
- Keep source features and semantic supervision fixed when testing schedules or module organization. When comparing representation sources in H1, explicitly account for their training signals and costs; where possible, compare activations before and after adapting the same source model. Count representation/interface preparation, schedule selection, and module refinement in the total training budget. Where resource matching is imperfect, report the actual costs.
- Compare conditional performance on common decoded or annotated targets. Raw latent denoising losses across different learned representations can change through scaling or easier targets and are not sufficient evidence of improvement.
- Use the causal case study from criticism 1 to explain a component–module relationship. A better final image alone does not establish that mechanism; a useful mechanism alone does not establish better full generation.
- Reuse a small set of variants across these comparisons. A full factorial sweep over every possible representation, schedule, and architecture is unnecessary. Detailed sample counts and thresholds remain open until the initial implementation and resources are settled.
- The integrated comparison evaluates the practical benefit of using dependencies to organize generation and computation within a fixed representation. It need not establish superadditivity or that every design choice is universally necessary. The individual comparisons help identify useful contributions.

### Changes to carry into the final revision, after agreement on the table

Add a compact table near the research approach or work plan, with short evidence labels such as “demonstrated in a simpler setting,” “partial feasibility,” and “proposed synthesis.” Let its first rows explain the links and its final row evaluate the integrated generator. Keep stronger headline performance results in the feasibility discussion, while using the closest matched comparisons to support specific hypotheses.

No proposal, figures, LaTeX, or compiled deliverables were changed during this discussion. All evaluation runs above are proposed award-period work.

## 3. Obtain representations first, then fix them during dependency and schedule learning

### Agreed scope decision

The user proposes a simpler and more concrete account of representation learning:

1. Use hidden activations of existing pretrained models as semantic representations.
2. Train or adapt reward models, discriminators, or models for other relevant downstream tasks, and use their hidden activations as semantic representations.
3. Once a representation has been obtained, hold it fixed while learning conditional dependencies, denoising schedules, and the denoising transformer.

These two representation sources suffice for the representation-learning part of the proposal. **The dependence/schedule-learning stage will not update the representation using denoising losses, regroup it based on dependency measurements, or alternate representation refinement with schedule optimization.** The difficult additional learning problem identified by the reviewer is removed from the committed approach.

Training a downstream-task model is still representation learning. Using an already pretrained model reuses its learned representation. Neither requires a separate claim that the generation objective discovers a new decomposition.

### Terminology and fixed quantities

Avoid “extractor” as the central terminology. Refer to the source network as a **representation encoder**, its activations as the **semantic representation**, and the units to which independent noise is applied as **representation components**.

“Fixed representation” means that the encoder and the definition of its components remain fixed in the dependency/schedule stage. It does not mean that noisy component values remain constant during generation or that the denoising transformer's parameters stop learning. If a projection, normalization, or compression map is needed to prepare the representation, it is settled before this stage and held fixed too.

The remaining bounded implementation choice is which units are called components. A recommended initial choice is existing token vectors or fixed groups of coordinates, selected before dependency learning. This is a proposal for implementation, not a newly agreed partition. Their functional meaning is established by the conditional-prediction diagnostics in criticism 1; no human-readable label for each group is required.

### What remains learned in the second stage

- Which fixed components provide useful information for predicting others, at different noise levels and for different conditional tasks.
- The asynchronous denoising schedules and the transformer weights used to predict those components.
- In the proposed module study, which component inputs, sharing patterns, or activity patterns support the relevant predictions.

Thus the components remain fixed while their useful dependencies and the computations exploiting them are learned. No explicit graph-learning formalism is required.

### Consequences for positioning and evaluation

The accurate overarching description becomes **studying the interaction of semantic representations, denoising processes, and transformer computations**. “Jointly learning all three” would overstate the revised method. The contribution to emphasize is discovering and exploiting conditional dependencies in learned semantic representations, with a mechanistic account of the transformer computations that use them.

This also changes the criticism-2 table accepted immediately before this decision. H1 should compare useful representation sources, not a decomposition learned from denoising against a fixed one. H4 should evaluate schedule and module organization within a fixed representation, not coupled refinement of all three elements. The current table above records these proposed adjustments explicitly. H2 and H3 remain applicable with the representation held fixed.

For criticism 1, the noise-availability objective remains a diagnostic for useful components; it is no longer a possible representation-refinement objective. Reusing that diagnostic does not require reintroducing the removed optimization loop.

### Changes to carry into the final proposal revision

- Recast Section 2.1 around the two sources of representations and state when those representations are fixed.
- Replace prominent “extractor” terminology with encoders and representation components, explaining any necessary fixed grouping or projection simply.
- Remove promises to refine representations from dependency measurements or denoising losses, including alternating refinement language, work-plan steps, and diagram arrows implying that feedback.
- Replace claims of jointly learning all three elements with the interaction-based account above; retain joint learning of schedules and the denoising transformer.
- Align the hypothesis table and evidence descriptions with that scope.

Only this discussion record has been updated. The proposal draft, figures, LaTeX, and compiled deliverables remain unchanged until the coordinated revision.

## 4. Check the gap between training states and generated states

### Agreed framing

The user considers this a useful question, but less damaging than the earlier criticisms: it concerns the practical discrepancy between states produced by sampling and states constructed by corrupting clean training examples. It should be handled as a bounded technical risk within the proposed method, with one short diagnostic paragraph. It does not require a new research aim or an elaborate experiment matrix.

For a correctly specified exact flow and exact integration, the generated and prescribed training states have matching time marginals. In practice, model approximation, numerical sampling, and errors in generated semantic components can create a gap. Matching nominal noise levels does not itself establish equal usefulness for downstream generation. This qualification belongs in the discussion record; the proposal need not expand it into a theoretical detour.

### Proposed paragraph for the final revision

We will compare conditional generation using components obtained by corrupting ground-truth examples with components produced by the model's own sampling procedure at the same noise levels. Keeping the downstream model and requested scene requirements fixed, we will assess prompt satisfaction and image quality to determine whether the useful dependencies persist during generation. If a substantial gap limits performance, we will investigate fine-tuning with model-generated conditioning states from short rollouts, drawing on approaches such as [Self Forcing](https://arxiv.org/abs/2506.08009).

### Feasibility and scope notes

[Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009) trains on self-generated context and supervises generated sequences using distribution-matching losses. It provides a relevant precedent for training under the conditions encountered at inference. Adapting that principle to our representation components is a candidate mitigation, not an already demonstrated solution for this setting.

The diagnostic compares aggregate task performance, not whether a generated state reproduces one particular reference sample. Generated states with different details can satisfy the same prompt. If rollout training is needed, its supervision must remain appropriate for those states; arbitrary rollout states cannot simply inherit a clean example's original flow-matching target. Task-level or distribution-level supervision is one possible route. These implementation details can remain outside the short proposal paragraph.

Representation encoders stay fixed, consistent with criticism 3. The conditioning-path distinction from criticism 1 also remains: the diagnostic must specify which information comes from the tested components and which is already supplied by the prompt. The user has not requested further preliminary experiments; both diagnosis and any mitigation are proposed award-period work.

### Changes to carry into the final revision

Insert the short diagnostic and contingency paragraph near the generation/evaluation discussion. Cite the rollout-training precedent, retain the emphasis on end-to-end generated outputs, and avoid presenting this issue as a separate major objective. No proposal or compiled deliverable changes have been made.

## 5. Budget computing and bound the implementation

### User's direction

The user agrees that the resource plan needs to be more realistic. Reduce requested PI salary support from 1.5 to 0.5 months, reduce the corresponding fringe benefits, and allocate the remaining room under the USD 150,000 award ceiling to computing. Specify datasets, architectures, approximate trainable parameter counts, and a run budget. Detailed tensor/component shapes are unnecessary for the proposal.

The user also supports a minimal demonstration, an early decision about scaling, and treating broader reference identity, multiple simultaneous activities, and video as extensions. The numerical run allocations and data targets below are proposed planning assumptions, not measured throughput or already curated datasets.

### Proposed budget using the supplied workbook's formulas

Source: `Budget_Sony_Cheng.xlsx`, worksheet `Budget`. Retain the supplied salary basis, student costs, fringe rates, tuition exclusion, 61.5% modified-total-direct-cost (MTDC) rate, and whole-dollar rounding. Budget rented cloud computing as an other direct expense included in MTDC. This follows the workbook's current formula treatment of other expenses; final institutional costing should use the applicable classification.

| Cost | Current budget (USD) | Proposed budget (USD) |
|---|---:|---:|
| PI salary | 23,833 (1.5 months) | 7,944 (0.5 months) |
| PI fringe benefits, 28.83% | 6,871 | 2,290 |
| Graduate stipend, 12 months | 44,213 | 44,213 |
| Graduate fringe benefits, 11.45% | 5,062 | 5,062 |
| Graduate tuition remission | 15,134 | 15,134 |
| Cloud computing and associated storage/service costs | 0 | **23,999** |
| Total direct costs | 95,113 | 98,642 |
| Indirect costs, 61.5% of MTDC | 49,187 | 51,357 |
| **Total requested** | **144,300** | **149,999** |

Calculation trace: workbook `B5` becomes 0.5; `C5=ROUND(143000/9*0.5,0)=7944`; `C8=ROUND(7944*0.2883,0)=2290`. Salaries and fringe total USD 59,509. With USD 23,999 in computing, MTDC is USD 83,508; `ROUND(83508*0.615,0)=51357` indirect costs. Tuition remains outside MTDC. Increasing computing to USD 24,000 would produce a rounded total of USD 150,001, so **USD 23,999 is the maximum whole-dollar computing line under these formulas**.

[Sony's terms](https://www.sony.com/en/SonyInfo/research-award-program/terms.html) make the award all-inclusive, including overhead. The calculation therefore includes the indirect cost on cloud services rather than allocating the entire apparent salary saving as direct computing money.

The original workbook and proposal budget have not been edited during this discussion. These are the figures to apply in the coordinated final revision.

### Architectures and trainable parameter targets

| Work | Proposed starting architecture | Approximate trainable parameters |
|---|---|---:|
| Mechanism development and representation-source screening | Existing small diffusion transformers, using cached semantic/image representations and existing experiment pipelines | 10–20M |
| One task-trained representation alternative | A modest multimodal prediction network using frozen visual/language features, trained to assess selected role or attribute requirements; use its internal activations, then freeze it | 10–20M as a design target during this separate task-training stage |
| Text-to-image demonstration | Pretrained PixArt-Σ, initially its 256px checkpoint with 512px as the planned larger demonstration; small semantic prediction branch and selective attention/MLP updates | Approximately 20–50M updated/new parameters, alongside the approximately 0.6B pretrained denoising backbone |

The [official PixArt-Σ repository](https://github.com/PixArt-alpha/PixArt-sigma) provides pretrained 256px/512px checkpoints, training code, feature-precomputation tools, and adaptation examples. DINOv2 and Qwen3-4B states remain initial representation sources. These source encoders stay fixed during dependency/schedule training; the existing 795M language denoiser need not be trained again for this project. Cache fixed features where possible and include encoding work in the compute budget.

The 20–50M estimate is a candidate implementation budget, not a measured count for an existing prototype. For scale, the [official PixArt configuration](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS/raw/main/transformer/config.json) has 28 layers of width 1,152. Rank-16–32 updates to self/cross-attention projections are approximately 8–17M parameters, or approximately 13–27M when feed-forward projections are also included. Allowing 10–20M for the semantic branch and interfaces gives the stated range. Low-rank updates are an implementation option; the scientific question concerns which computations should access, share, or specialize for particular representation components.

Frozen weights still incur forward computation and can require gradient propagation through their activations. GPU-hour estimates must therefore cover the full executed model, not scale linearly with the trainable parameter fraction. Retain the pretrained image-decoding path and establish a compatible denoising interface during the initial integration. The open conditioning-path questions in criticism 1 remain relevant; naming PixArt does not settle the role of its T5 conditioning.

### Bounded datasets and curation

- **Primary interaction annotations:** [SWiG](https://prior.allenai.org/projects/gsr), which supplies grounded semantic-role information.
- **Broader objects, attributes, and relationships:** [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html).
- **Training scale:** begin with approximately 20,000 selected distinct images and cap the main corpus at 50,000, including any controlled generated supplements. These are intended filtered-corpus sizes, not counts already verified after filtering.
- **Focused case:** two participants in one directional interaction family, with handover as the preferred example. Choose the final family after checking annotation coverage. Target roughly 100–300 manually checked matched cases for the mechanistic study; generated examples can supply controlled role reversals when natural pairs are unavailable.
- **Existing small-data pipelines:** retain CelebA64/AAHQ/STL-10 for inexpensive module-reuse implementation checks, while the annotated interaction corpus supplies the creator-facing and semantic case studies.

The [SWiG paper](https://arxiv.org/abs/2003.12058) describes 126,102 images across 504 verbs. This does not imply thousands of usable natural handover examples; role visibility and annotation coverage require filtering. Visual Genome supplies broad relation annotations but does not guarantee complete giver/receiver/object assignments or recurring character identities. Use the broad corpus for conditional training and the smaller checked set for the causal example.

Preserve image-level splits across captions/crops and keep paired synthetic variants in the same split. Withhold selected role/attribute combinations. These are implementation details for the planned data preparation, not additional proposal paragraphs or pre-submission experiments.

### Proposed computing allowance: 5,000 H100-80GB GPU-hours

Use one GPU type to make the estimate interpretable. A GPU-hour is one GPU used for one hour; four GPUs running for 72 hours consume 288 GPU-hours.

| Work | Planned allocation | GPU-hours |
|---|---|---:|
| Small-model experiments and repetitions | 12 runs × 1 GPU × 48 hours | 576 |
| Main image-generation comparisons | 8 runs × 4 GPUs × 72 hours | 2,304 |
| Feature encoding, controlled data generation, and one task-trained representation source | Combined allowance | 600 |
| Generation evaluation, component/module interventions, and the rollout-gap diagnostic | Combined allowance | 900 |
| Integration failures and repeated runs | Reserve | 620 |
| **Total** | | **5,000** |

The eight principal runs can implement four variants with two seeds: fixed versus learned schedules crossed with ordinary versus dependency-guided module organization, holding the chosen representation fixed. Screen representation sources primarily in the small-model stage. These shared variants cover the focused comparisons in criticism 2 without a large factorial sweep.

For a rough main-run assumption, plan 5,000–20,000 updates with effective batch 64. The upper end is 1.28M training presentations, or about 26 passes over a 50,000-image corpus. At an assumed average of 10 seconds per optimizer update on four H100 GPUs, 20,000 updates take about 56 wall-clock hours; the 72-hour allowance leaves time for validation and checkpointing. **This is a planning assumption, not benchmarked throughput.** Profile the actual modified model early in the award and reduce exploratory steps, resolution, or repetitions if necessary while preserving the core comparisons and evaluation allocation. No new profiling experiment is required before submission.

As checked on 2026-09-16, [Lambda's posted H100-SXM pricing](https://lambda.ai/pricing) is USD 4.29 per GPU-hour for a single-GPU instance and USD 4.09 for the four-GPU instance. Budgeting conservatively at **USD 4.30 per GPU-hour** gives USD 21,500 for 5,000 GPU-hours, leaving **USD 2,499** of the computing line for persistent storage, applicable cloud charges, and price variation. This is a planning rate, not a reserved-capacity quote or a claim of purchased access. It does not assume donated compute; existing institutional allocations would provide additional capacity if available.

### Minimum deliverable and scaling decisions

The proposed minimum is a conditional-generation demonstration with two participants and one interaction family, using one initial pretrained representation configuration and one task-trained alternative, each fixed during denoising training. Investigate one sharing hypothesis, such as reusing an attention computation across related conditional tasks, and complete one component–module intervention/restoration case study. Assess held-out role/attribute combinations alongside image quality.

The reviewer's phrase **“one representation refinement” is superseded by criticism 3**. Use “one task-trained representation alternative” instead; do not reintroduce refinement through the denoising objective.

| Decision point | Required information and scope decision |
|---|---|
| Month 3 | Usable role-annotated subset, a functioning conditional-generation baseline, prepared fixed representations, and measured memory/runtime. Select the initial interaction family and confirm the remaining run budget. |
| Month 6 | Usable conditional generation and a reproducible component–module effect in the bounded setup. Proceed to the larger 512px demonstration if data quality, integration, and runtime support it; otherwise retain a complete smaller-scale demonstration and mechanistic study. |

Broader reference identity across scenes, multiple simultaneous activities, and video are extensions beyond the minimum deliverable. The proposal should still explain their relevance to Sony without depending on all of them for success.

### Changes to carry into the final revision

Replace the “no computing costs requested” statement; apply the revised salary/fringe/compute/indirect figures; replace the broad execution section with a compact architecture/data/run table; state the minimum demonstration and scaling decisions. Preserve the fixed-representation decision throughout. The proposal, original budget workbook, figures, LaTeX, and compiled deliverables remain unchanged at this discussion stage.

## 6. Lead with the gap and connect the three ideas through conditional dependencies

### User's proposed positioning

The user agrees that the differentiation should begin with the gap. The proposed key ideas are:

1. Carefully learn a decomposable semantic representation, with more structure and task relevance than simply taking a pretrained model's final-layer activations.
2. Learn dependencies between its components through denoising and schedules, with controllability of generation as the objective rather than training speed alone.
3. Learn how transformer modules interact with those components, emphasizing the integrated study of these relationships.

The following refinements are the assistant's recommendations for discussion. They preserve the fixed-representation decision in criticism 3 and do not add a new representation-refinement objective.

### Recommended central insight

**Use the same conditional dependencies to organize both how semantic requirements are generated and which transformer computations realize them.**

The connection is more specific than combining useful representations, better schedules, and module analysis. A component that supplies useful information for a prediction should inform when that information becomes available and which transformer computations need access to it. Establishing that correspondence would connect semantic control with an internal account of generation.

One concrete example is information relevant to the giver–receiver assignment. First characterize which representation components help predict the visual interaction. Learn how their relative denoising progress should support that prediction. Then identify or train the attention/MLP computations that use the same information, checking their role through component–module interventions. The learned components need not each have a human-readable label.

### Sharpen the three points

**1. Task-relevant representations with functionally characterized components.** Retain the two concrete representation sources: pretrained activations, then activations of models trained on selected reward, discrimination, or other downstream tasks. The intended refinement is greater relevance to requirements such as participant-role and attribute binding, together with an explicit characterization of component usefulness. A useful decomposition means that changing component availability has distinguishable effects on conditional predictions; merely partitioning a vector does not establish that property.

Training a task model encourages relevant information but does not automatically guarantee a selective decomposition. The wording should therefore describe developing task-relevant representations and identifying useful components, unless the initial representation-training stage also specifies an objective that constructs a decomposition. Such an additional objective has not been agreed. Under criticism 3, the representation and component definitions are fixed before dependency/schedule learning, and are not revised by that stage.

**2. Learn conditional dependencies and schedules for controllable generation.** Component-wise noise variation reveals which information improves a prediction. Schedule optimization determines when making that information available is beneficial, accounting for the difficulty of predicting a component before its context is available. These two operations are closely connected, but a learned generation order alone does not identify a unique dependency structure. The accurate method is to learn dependencies through conditional denoising at varied noise levels while jointly learning schedules and the transformer.

Controllability is the goal: use the resulting information structure to satisfy interacting prompt requirements reliably. Training speed and FID remain useful supporting outcomes and feasibility evidence; the proposed advance concerns how specific requirements enter the generated content.

**3. Explain and organize the transformer computations that use those dependencies.** Establish which modules use the relevant component information, how this changes with noise levels, and which computations can be shared across conditional tasks. The component–module intervention from criticism 1 supplies a concrete mechanism: does disabling a module selectively remove the benefit of exposing one component when predicting another, and does restoration recover it? Use the same dependence measurements to inform access, activity, and sharing. This is stronger than attaching an attention visualization to an otherwise unrelated schedule-learning method.

### Proposed gap-first paragraph

Semantic representations and asynchronous denoising provide useful foundations for generation, but encoding a requirement does not by itself explain how the generator realizes it. We will connect three questions: which representation components supply the information needed for a conditional prediction, when that information should become available during denoising, and which transformer computations use it. Starting from pretrained and task-trained semantic representations, we will learn component dependencies through conditional denoising and use them to organize schedules and transformer computations for precise control of interacting semantic requirements.

This paragraph states the proposed gap and response; it does not assert that all prior work lacks mechanistic explanations or that each ingredient is individually new.

### Differentiation from close existing work

| Close work | Existing contribution relevant to this proposal | Proposed emphasis beyond that contribution |
|---|---|---|
| [LWD](https://arxiv.org/abs/2606.19662) and [Latent Forcing](https://arxiv.org/abs/2602.11401) | Asynchronous evolution across representations and the benefit of choosing or learning generation schedules | Multiple task-relevant components, measured conditional usefulness, and explicit connection to transformer computations for semantic control |
| [Local Mechanisms of Compositional Generalization in Conditional Diffusion](https://arxiv.org/abs/2509.16447) | Sparse conditional-score dependencies, including feature-space composition, with interventions supporting the locality/composition relationship | Connect component dependencies across noise configurations to learned schedules and identifiable, reusable transformer computations |
| [Vision-Language Binding in In-Context Image Generation](https://arxiv.org/abs/2605.24624) | Causal tracing of reference information through multimodal transformer computations | Use dependencies among semantic generation targets to organize both denoising and computation, alongside causal analysis |

The broader combination of representation learning, scheduling, and architecture work is not sufficient by itself to establish novelty. The proposed differentiation is the shared relationship among **component usefulness, denoising progress, and module function**, and its use for control. The present literature comparison motivates that positioning; it does not establish an exhaustive “first” claim.

### Scope and wording to carry into the final revision

- Lead the differentiation section with the missing connection, then explain the three linked steps.
- Describe an integrated pipeline and joint learning of schedules and transformer computations **within an obtained representation**. Avoid saying all three are simultaneously optimized.
- Preserve the user's emphasis on carefully obtained task-relevant semantic representations. Do not reduce Section 2.1 to an arbitrary frozen DINO feature choice, but do not silently restore denoising-driven representation refinement.
- Define decomposability through conditional usefulness and interventions, with the initial component grouping stated separately.
- Distinguish dependency measurement through varied noise levels from the subsequent choice of denoising progress; neither requires a unique graph or a strict semantic-first order.
- Emphasize the generation requirement, the information supporting it, and the module using that information. Keep implementation and evaluation detail subordinate to that story.

Only the discussion record has been updated. The proposal, budget workbook, figures, and compiled deliverables remain unchanged.

## 7. Compress preliminary evidence and replace the language result

### Updated result supplied by the user

The user reports a later experiment with **62.4716% GSM8K accuracy** and the team's own **121M-parameter encoder**, and asks that it replace the previous 61.94% result. The denoising model is described as approximately 800M parameters, using Qwen-3 hidden activations as latent states.

Treat this as a new PI-reported preliminary result; no additional experiment or independent re-evaluation is required before submission. Preserve the exact 62.4716% figure in this record and use **62.47%** in proposal prose. State the approximately 800M denoiser and 121M encoder separately. Do not reuse the old claim that this result uses a separate 4B question encoder, or infer a complete system parameter count or a more specific role for the new encoder than the user has supplied. Qwen-derived latent states and the encoder used by the final model are distinct descriptions.

The new result changes the feasibility evidence. It does not automatically select or replace the encoders in the proposed image-generation implementation; that integration choice can use the new model when appropriate.

### Keep the language evidence to one or two sentences

The result is several steps removed from multimodal binding. Its purpose is to show that continuous latent diffusion can generate reasoning-relevant language states, encouraging its use for semantic as well as visual components. Do not make it a separate performance centerpiece.

Proposed two-sentence version:

In preliminary experiments, our continuous diffusion language model, with an approximately 800M-parameter denoiser and a 121M-parameter encoder, uses Qwen-3-derived latent states and achieves **62.47% GSM8K accuracy**, compared with **66.6% reported for the 7B TESS 2 model after mathematics-specific fine-tuning**, under different training settings. This result supports using continuous diffusion to generate reasoning-relevant semantic states alongside visual components.

[TESS 2, Table 3](https://arxiv.org/html/2502.13917v1#S4.T3) supplies the 66.6% mathematics-fine-tuned reference point; its base comes from Mistral-7B. This is contextual evidence of strong continuous-diffusion reasoning, not a matched efficiency comparison or a claim of equal performance. One reference point is enough for the short feasibility paragraph.

The assistant recommends omitting the proposed historical statement that continuous diffusion had not previously succeeded at language generation. [Diffusion-LM](https://arxiv.org/abs/2205.14217) already studied continuous diffusion for controllable text generation in 2022, and TESS 2 demonstrated broader instruction-following. The useful claim here is substantial reasoning capability in continuous latent diffusion, rather than the absence of earlier language-generation results.

### Retain LWD versus REPA as evidence for the overall approach

The user disagrees with making the SFD-matched comparison the sole or dominant feasibility argument. SFD and LWD both motivate asynchronous multi-component diffusion; comparison with a more standard approach such as REPA helps demonstrate why that overall direction is useful.

Keep **class-conditional ImageNet-256** as the setting and retain the comparison of LWD's FID **4.93 versus REPA's 5.84**, at **120,000 versus 4 million main-training updates**. The 33-fold figure describes fewer main-training updates, not measured wall-clock speedup. The schedule-learning phase adds 10,000 updates; this can be acknowledged compactly where the training budget is stated. Do not attribute the entire system-level difference to schedule optimization alone.

The stronger final FID result can remain if space permits. The user considers the AutoGuidance setting unnecessary detail for the main proposal narrative. Omit that detail there; retain accurate experimental provenance in the source references and figure notes. SFD comparisons can still support the narrower schedule hypothesis in the hypothesis/evidence table without displacing the REPA comparison in the feasibility story.

### Explain the reviewer's “lead feasibility” recommendation

“Lead feasibility with the matched schedule result and the component-reversion finding” was a recommendation about ordering: first show that changing the denoising process helps image generation, then show that selected transformer weights can be reused, then briefly give the language-state result. “Component reversion” refers to restoring selected source weights after adaptation and examining which generated behavior remains.

It does not require new experiments or a rebuttal-style limitations paragraph. We will adapt the ordering to the user's preference: lead with the image-generation evidence, including REPA, connect the existing module results to the proposed mechanism, and keep language evidence short. Retain only the distinctions needed to avoid an inaccurate claim, such as not turning post-training weight restoration into evidence that those weights were frozen throughout training.

### Figure 3 and experimental evidence

No additional multi-seed runs are requested or required before submission. Use the existing qualitative results and available measurements; do not add a lengthy discussion of seed counts or uncertainty to the proposal.

The numerical panel of the current Figure 3 is already a compact two-row table, replacing earlier bars. Preserve that direction and consider putting the two numbers directly in the caption or adjacent prose if it saves space. Retain the qualitative source/hybrid/joint comparison only at a size where the image differences are readable. Do not reintroduce a bar chart for two values.

### Changes to carry into the final revision

Replace the old language score and encoder description everywhere in the submission text, restrict that result to one or two sentences, and remove its benchmark figure. Keep the main feasibility emphasis on the image-generation and transformer-mechanism results, using REPA as the broader baseline. Existing artifacts/provenance can retain historical results if clearly identified; the submitted proposal should use the updated result consistently.

## 8. Keep mathematical detail relevant and self-contained

The user agrees that the proposal should omit stop-gradient details because they do not help explain the proposed investigation. **Those details have already been removed from the current draft; preserve that omission in the final revision.**

Retain the definitions necessary to understand the representation components, noise levels, conditional prediction, joint schedule/transformer objective, and proposed interventions. Explain each symbol locally. Remove generic optimization tutorials and implementation notation that does not clarify a new idea, including the elementary parameter-update equation if it adds no useful content. Do not replace the removed detail with another elaborate formalism.

This accepts the reviewer's prioritization point without adopting every technical qualification as proposal text. Details of existing algorithms can remain in their citations and implementation materials.

## 9. Reduce figure space and integrate overlapping explanations

The user agrees that figures take too much space, supports deleting Figure 4, and is open to wrapping text around suitable figures and merging Figures 1/2 if that improves clarity. This section tracks the review's cross-cutting layout comments under the user's “criticism 9” label.

### Agreed changes

- **Delete Figure 4.** The updated language result belongs in its short prose description.
- Avoid charts for simple numerical comparisons; use a small table or prose for Figure 3's two values.
- Reduce duplication and unused space while preserving readable type and explanatory content.

### Recommended final layout, subject to visual review

Aim for **two figures in total**: one integrated method diagram and one compact qualitative preliminary-result figure.

For the method diagram, use the current Figure 2's concrete conditional-information comparison, overlapping schedules, and transformer path as the main structure. Incorporate only the necessary representation explanation from Figure 1, such as a small strip showing fixed representation components and a clear link to the conditional task. Remove the duplicated overview boxes and long text labels. The revised diagram must reflect the fixed-representation decision and present component meanings as illustrative, not prescribe human-labeled coordinates. Joint learning applies to schedules and transformer computations.

The method diagram will probably benefit from the full text width. Consider text wrapping for a compact version of the qualitative transfer figure if its images and labels remain legible. Wrapping a dense multi-panel diagram into a narrow column would likely reduce clarity. The exact placement, caption width, and figure dimensions should be decided from the rendered final PDF, rather than by shrinking all figures uniformly.

If merged, renumber the surviving figures and update every caption and cross-reference. Do not concatenate the existing Figures 1/2 unchanged; the goal is one clearer explanation with duplicated content removed.

## 10. Coverage audit against the original independent review

The user asked us to revisit the complete review, check whether every issue has been addressed, and identify useful suggestions that were missed. The assistant reread the original review and current discussion record, checked the held proposal, and obtained a separate coverage audit from another agent.

**Every major criticism has a response in this discussion, but several have only been resolved in principle. The coordinated proposal revision has not yet happened.** The table distinguishes an agreed response from a completed implementation choice.

| Review item | What our discussion resolves | What still needs attention in the coordinated revision |
|---|---|---|
| 1. Mechanistic claim | Functional meaning through conditional prediction; component–module interventions and restoration; no mandatory human-readable semantics | State the information paths and the scope of the mechanism; choose initial component/module units; specify matched control interventions |
| 2. Value of the synthesis | Hypothesis-and-test table, focused variants, common generation endpoint, held-out combinations, and partial-evidence labels | Give a compact evaluation size/scoring/uncertainty plan and a useful interpretation if sparse sharing does not help |
| 3. Representation algorithm | Obtain representations from pretrained or task-trained models, then fix them; remove denoising-driven refinement | Name one initial component grouping and one rule translating measured usefulness into module access |
| 4. Generated-state gap | One short diagnostic comparing corrupted target states with generated states; possible rollout-based mitigation | Keep missing-information diagnosis distinct from generation with complete prompts |
| 5. Feasibility | Proposed compute budget, bounded data and trainable model sizes, run allowance, minimum milestone, and scaling gates | Explain the native image-decoding path and the interface to the pretrained denoiser; apply the revised budget consistently |
| 6. Differentiation | Lead with the connection between component usefulness, generation progress, and transformer computation | Remove claims of simultaneous representation/schedule/network optimization; present the nearest-work comparison around the proposed relationship |
| 7. Evidence | Updated 62.4716% result and 121M encoder; short language paragraph; REPA comparison retained; no new preliminary runs | Replace stale scores/encoder descriptions and compress the evidence in the submission |
| 8. Mathematics | Keep necessary, defined notation; omit stop-gradient details and elementary optimization exposition | Preserve the existing final prediction-only fit and dependency remeasurement |
| 9. Figures and organization | Delete Figure 4; simple values in prose/tables; consider one integrated method diagram | Implement the layout changes and the additional organization suggestions below |

### Useful remaining method and evaluation details

1. **Scope the mechanistic result and its information paths.** The diagnostic can establish that a particular component helps a particular module make a conditional prediction. It should not imply that this explains every route through the pretrained image generator. Separately demonstrate the practical benefit with complete prompts. T5 removal remains undecided; a bounded claim about the identified path avoids requiring a complete explanation of the backbone. Masked words must be removed before contextual encoding in the diagnostic.

2. **Give one concrete starting configuration.** Fixing the representation resolves the largest algorithmic gap, but the reader still needs to know what receives an independent noise level, what counts as a transformer module, and how a measured dependency determines its inputs. Choose one initial component unit, one module granularity, and one access-selection rule when drafting. Token groups, coordinate groups, attention heads, and MLP branches should not all remain equally prominent alternatives. This does not require exact tensor shapes or restoring representation refinement.

3. **Finish the evaluation paragraph.** The current record already budgets four main variants with two seeds and proposes 100–300 checked mechanistic cases. It still leaves the main generation evaluation's prompt/sample count and scoring procedure open. State a bounded number of held-out requests and generations, how joint requirement satisfaction will be judged, and how paired differences and uncertainty will be summarized. Decision criteria can require improvement in joint satisfaction without a material quality loss, alongside a specific intervention/rescue effect; no numerical performance gain needs to be invented. These are award-period plans, not additional pre-submission experiments.

4. **Explain the pretrained-model integration in ordinary language.** State that image generation retains the pretrained image-latent/VAE decoding route, and explain where the semantic states enter its denoising computation. The implementation must also preserve or explicitly map the backbone's noise levels, input scaling, prediction target, and sampler. The present phrase “establish a compatible interface” leaves this bridge open. Settle one initial route before treating backbone freezing as a sufficient feasibility argument.

5. **Retain small controls with high explanatory value.** The original review specifically proposed equally sized random-module interventions and controls for perturbation/activation scale, in addition to restoration and collateral effects. Our record had only said “matched controls.” A short explicit phrase would make the causal study more convincing without a long ablation list. If a task requires broad access or little sharing, the useful outcome is a measured boundary of selective reuse; do not promise sparsity everywhere.

### Organization suggestions that had not yet been explicitly carried forward

- Introduce the concrete starting model, representation source, and bounded task near the beginning of the method, rather than delaying them until the execution section.
- State the central contribution once clearly, then let the approach develop it. Reduce repeated versions across the abstract, introduction, differentiation, applications, and work plan.
- Give differentiation, preliminary feasibility, Sony's use case, and evaluation distinct functions. Avoid making the performance evidence carry the novelty argument.
- Remove the closing list of distant future applications from the main narrative if it competes with the committed deliverable. Preserve deferred ideas in our discussion/history rather than using them to expand this proposal.
- After the coordinated edit, read the submission alone to check that its examples, equations, method, and figures use the same fixed-representation scope and initial implementation.

### Existing fixes and deliberately unadopted suggestions

The original review predates some edits. Figure 3 already uses a compact numerical table, and the held draft already omits stop-gradient detail. It also explicitly trains the final denoiser with the prediction loss alone after schedule selection and remeasures dependencies. Preserve that last safeguard; the review's technical caution does not require reopening a lengthy velocity-penalty discussion.

Do not revive additional preliminary experiments, new multi-seed runs, a SFD-first feasibility requirement, mandatory human-readable component names, denoising-driven representation refinement, or a full interpretation of every module. These were narrowed or rejected through the discussion. The remaining details can be settled within the existing approach rather than adding research directions.

### PI contact details and submission completion

The user supplied the PI email and phone in the conversation. The values are intentionally omitted from this GitHub record: automatic approval review blocked publishing them because it requires explicit authorization to disclose them in the repository. The information is available for the final revision once that destination is authorized.

Remaining production items are the separate PI CV, consistent revised budget amounts, confirmation of the workbook's calendar dates, complete citations, removal of editorial notes, and a final rendered-PDF check. The date question already appears in the draft's notes: the workbook's stated project dates and its blended academic-year rates are not aligned. Do not invent replacement dates.

## Status before the coordinated revision

All eight ranked criticisms and the user's figure/layout item have now been discussed. The directions and proposed implementations are recorded above, with remaining choices marked as recommendations rather than approved final designs. Continue to preserve the no-additional-preliminary-experiments constraint. The proposal, budget workbook, figures, LaTeX, and compiled deliverables remain unchanged until the coordinated revision.

## Record of updates

- **2026-09-16:** Created this discussion record at the user's request. Captured the agreed direction, candidate methods, and open decisions for criticism 1. Queued the remaining criticisms without treating the reviewer's suggestions as accepted commitments. No proposal, LaTeX, figure, budget, or compiled deliverable changes were made.
- **2026-09-16:** Added the discussion of criticism 2: a proposed four-row hypothesis-and-test table, distinctions among existing evidence and proposed extensions, and focused comparison principles. The user supports the table format and evidence annotations; the specific rows remain proposals for discussion. The proposal draft remains unchanged.
- **2026-09-16:** The user accepted the criticism-2 table for now, then narrowed criticism 3: obtain representations from pretrained or downstream-task models and fix them during dependency/schedule learning. Recorded this decision, superseded the proposed representation-refinement loop, and adjusted the working H1/H4 formulations to reflect the scope change. No proposal changes were made.
- **2026-09-16:** Recorded criticism 4 as a bounded technical risk, with a single proposed diagnostic paragraph and rollout-based fine-tuning as a possible mitigation. Added Self Forcing as a feasibility precedent. The proposal draft remains unchanged.
- **2026-09-16:** Recorded criticism 5, including a formula-traced proposed USD 149,999 budget with USD 23,999 for cloud services, a 5,000-H100-GPU-hour planning allowance, bounded datasets and trainable parameter targets, and month-3/month-6 scaling decisions. Corrected the reviewer's outdated “representation refinement” milestone to respect criticism 3. The original budget workbook and proposal remain unchanged.
- **2026-09-16:** Recorded the user's three-part positioning for criticism 6 and the recommendation to connect it through component usefulness, denoising progress, and module function. Clarified the meaning of decomposability, the distinction between dependency measurement and schedule order, and the fixed-representation scope. Added close-work comparisons without making an unsupported “first” claim. No proposal edits were made.
- **2026-09-16:** Recorded criticisms 7/8 and the user's figure/layout item 9. Updated the evidence record to the PI-reported 62.4716% GSM8K result with a 121M encoder, recommended a short TESS 2 comparison, retained the REPA framing, rejected additional multi-seed work, and recorded removal of Figure 4 plus a possible Figures 1/2 merger. Confirmed that Figure 3's table and omission of stop-gradient details are already present in the held draft. No submission artifact changes were made.
- **2026-09-16:** Audited all original review comments against the discussion, with a separate agent coverage check. Recorded the remaining implementation, evaluation, and organization details; distinguished responses in principle from completed draft changes; preserved existing fixes and decisions not to expand scope. The PI supplied contact information, but automatic approval review blocked publishing its values to GitHub without explicit disclosure authorization; the values are omitted from this record. Only this discussion record was changed.
