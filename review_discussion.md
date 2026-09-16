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
| 6 | The differentiation does not yet isolate one new scientific relationship. | Awaiting discussion. |
| 7 | Preliminary evidence takes more space than its direct support for the proposal warrants. | Awaiting discussion. |
| 8 | Mathematical detail is disproportionate to the unresolved method choices. | Awaiting discussion; some exposition was already simplified. |

Organization, figures, and submission cleanup are cross-cutting review comments. They will be revisited during the final editing pass.

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

For H1, the existing 61.94% GSM8K result is evidence that continuous latent diffusion can generate functionally useful contextual language states. It supports using pretrained contextual states as a representation source; it does not establish the additional benefit of task-trained visual representations or their visual grounding.

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

## Remaining discussions

The items below summarize reviewer questions to revisit. **They are not yet accepted changes or additional deliverables.**

| No. | Question to settle | Related to criticism 1 |
|---|---|---|
| 6 | Which relationship differentiates the proposal from existing schedule learning, sparse modules, and representation methods? | Functional dependencies that predict specific transformer information paths are a candidate, not a settled novelty claim. |
| 7 | Which existing results most directly establish feasibility, and how much space should each receive? | No new preliminary experiments are required; preserve the distinction between supporting ingredients and establishing their interaction. |
| 8 | Which equations make the new method understandable, and which details can be shortened? | Keep the new operational definition self-contained; avoid introducing unexplained gating or attribution notation. |

## Record of updates

- **2026-09-16:** Created this discussion record at the user's request. Captured the agreed direction, candidate methods, and open decisions for criticism 1. Queued the remaining criticisms without treating the reviewer's suggestions as accepted commitments. No proposal, LaTeX, figure, budget, or compiled deliverable changes were made.
- **2026-09-16:** Added the discussion of criticism 2: a proposed four-row hypothesis-and-test table, distinctions among existing evidence and proposed extensions, and focused comparison principles. The user supports the table format and evidence annotations; the specific rows remain proposals for discussion. The proposal draft remains unchanged.
- **2026-09-16:** The user accepted the criticism-2 table for now, then narrowed criticism 3: obtain representations from pretrained or downstream-task models and fix them during dependency/schedule learning. Recorded this decision, superseded the proposed representation-refinement loop, and adjusted the working H1/H4 formulations to reflect the scope change. No proposal changes were made.
- **2026-09-16:** Recorded criticism 4 as a bounded technical risk, with a single proposed diagnostic paragraph and rollout-based fine-tuning as a possible mitigation. Added Self Forcing as a feasibility precedent. The proposal draft remains unchanged.
- **2026-09-16:** Recorded criticism 5, including a formula-traced proposed USD 149,999 budget with USD 23,999 for cloud services, a 5,000-H100-GPU-hour planning allowance, bounded datasets and trainable parameter targets, and month-3/month-6 scaling decisions. Corrected the reviewer's outdated “representation refinement” milestone to respect criticism 3. The original budget workbook and proposal remain unchanged.
