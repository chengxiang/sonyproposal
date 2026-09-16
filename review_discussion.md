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
| 2 | No decisive evaluation isolates the value of learning representations, denoising, and transformer computations together. | Awaiting discussion. |
| 3 | The representation-refinement algorithm is underspecified. | Awaiting discussion. |
| 4 | Useful information in corrupted ground-truth states may not remain useful in generated states. | Awaiting discussion. |
| 5 | Integration, resources, and the minimum 12-month scope need firmer bounds. | Awaiting discussion. |
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

- Start with fixed extractors and a fixed denoiser trained to handle independently varied component noise levels. Normalize component scales so the diagnostic does not reward arbitrary rescaling.
- Learn an allocation across many examples of a task, then assess it on unseen examples. An allocation optimized separately using each example's answer could itself leak answer information.
- Interpret the result as an economical sufficient allocation, rather than a unique list of indispensable components. Redundant alternatives and jointly useful groups may exist.
- The penalty $\sum_k u_k$ limits total exposure to clean information; it does not directly guarantee a small number of exposed components. Stochastic on/off gates are a possible alternative if component-count sparsity becomes the objective.

This diagnostic could inform decomposition learning later. The objective above does not yet specify how to train or refine the extractors; that remains part of criticism 3.

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

## Remaining discussions

The items below summarize reviewer questions to revisit. **They are not yet accepted changes or additional deliverables.**

| No. | Question to settle | Related to criticism 1 |
|---|---|---|
| 2 | Which compact comparisons and generation outcomes will show the contribution of the joint approach? | Include the component–module interaction and causal case alongside generation performance. |
| 3 | What is the initial extractor family, and how will the learning objective refine the decomposition? | The proposed noise allocation characterizes an existing decomposition; learning it still needs a specified procedure. |
| 4 | How will we distinguish useful clean target information from useful model-generated information? | Separate masked-information diagnostics from complete-prompt generation and examine actual generated trajectories. |
| 5 | What is the smallest credible implementation and one-year deliverable, given available data and compute? | Decide the conditioning path and module granularity before committing to a large integration. |
| 6 | Which relationship differentiates the proposal from existing schedule learning, sparse modules, and representation methods? | Functional dependencies that predict specific transformer information paths are a candidate, not a settled novelty claim. |
| 7 | Which existing results most directly establish feasibility, and how much space should each receive? | No new preliminary experiments are required; preserve the distinction between supporting ingredients and establishing their interaction. |
| 8 | Which equations make the new method understandable, and which details can be shortened? | Keep the new operational definition self-contained; avoid introducing unexplained gating or attribution notation. |

## Record of updates

- **2026-09-16:** Created this discussion record at the user's request. Captured the agreed direction, candidate methods, and open decisions for criticism 1. Queued the remaining criticisms without treating the reviewer's suggestions as accepted commitments. No proposal, LaTeX, figure, budget, or compiled deliverable changes were made.
