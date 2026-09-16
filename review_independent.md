# Independent skeptical review of the Sony proposal

*Reviewed the draft at commit `ae9b6b50e746cec2665204b2abeb4fc839dbcc70`, before the accompanying clarity and figure edits. The review findings below are preserved independently of those edits.*

**Review basis.** I evaluated an immediate snapshot of `proposal_draft.md`, copied to `/tmp/sony_review_snapshot.md` before reading, and inspected the four proposal PNG figures. I did not consult discussion history, other drafts, memories, repository history, or other reviewers. I made a limited primary-source check of Sony's theme and the cited LWD method. The comments below concern the document as supplied, including its explicitly marked drafting material. They do not assume that the proposed experiments have already been performed.

**Overall assessment: promising and thematically relevant, but currently a borderline proposal with substantial scientific-design weaknesses.** The PI appears capable of executing diffusion research, and the proposed connection among representations, asynchronous generation, and reusable computations could be a worthwhile contribution. The weakness is that the document promises a mechanistic account and better compositional generation without specifying decisive tests that would establish either. Much of the mathematical detail formalizes an existing schedule-learning procedure, while the most original and consequential steps—learning a useful decomposition, interpreting its dependencies, and establishing transferable module function—remain verbal. I would request a sharper experimental argument before prioritizing this proposal for a focused industry award.

## Ranked major criticisms

### 1. The proposed mechanistic explanation is stronger than the evidence its experiments would establish

**Passages:** Section 2.2, “A positive $D_{j\to i}$ means cleaner information in $j$ improves denoising of $i$”; Section 2.3, “remove or replace a head or MLP contribution and measure which component predictions change”; Section 3, “The intended control is grounded in identifiable information paths and parameter roles.”

**Type: substantive scientific weakness; highest priority.** The dependency statistic is a sensible measure of predictive usefulness for the current model, representation, conditioning, and noise configuration. The paired examples and noise draws are good controls. Nevertheless, it does not establish what semantic information a component carries, what computation a module implements, or whether that computation causes a specific binding decision in final images. Cleaner role-associated vectors could help through identity, pose, scene context, or other correlated information. Module removal can establish that a contribution matters, but widespread output damage does not identify a role-binding function. Replacing a module from an unrelated task can fail because of interface incompatibility or activation scale, rather than because its semantic function differs.

The unrestricted backbone and persistent T5 conditioning create a further problem. A restricted added branch has an inspectable input mask by construction, but the final prediction can still receive the same information through the backbone. The proposal acknowledges that the backbone will be analyzed; it does not explain how that analysis will distinguish a mechanism of binding from an auxiliary contribution to image quality. Holding external conditioning fixed is useful, but does not resolve the existence of parallel paths.

**Likely reviewer interpretation:** “This may build interpretable-looking modules and measure their importance, but it has not specified how to explain the model's actual binding behavior.” This is particularly consequential because Sony explicitly requests internal, causal, and representation-level accounts, including cross-modal routing and interventions. The theme supports this project, but raises the required standard for its mechanistic claims. [Sony's focused research theme](https://www.sony.com/en/SonyInfo/research-award-program/)

**Specific change:** Define the intended finding as a model-relative predictive dependency, then add one complete causal case study. For example: form matched giver/receiver reversals; localize candidate internal information; interchange a component or a module contribution at a matched noise configuration; measure the intended role change and collateral changes in identity, clothing, and image quality; restore the contribution to test rescue; repeat on held-out participant combinations. Include matched random-module and activation-scale controls. State whether the result concerns the added mechanism, the pretrained backbone, or the entire model. A small number of convincing necessity, sufficiency, and rescue tests would be more valuable than many generic ablations.

### 2. There is no decisive evaluation of the central claim that learning all three elements together is useful

**Passages:** The abstract and Section 1 repeatedly promise joint learning; Section 4 says identity, roles, binding, activities, image quality, and joint satisfaction “will be assessed separately”; the work plan promises “baseline generation and denoising measurements.”

**Type: substantive experimental-design weakness.** There is no explicit baseline matrix, principal endpoint, success threshold, evaluation sample size, or plan for uncertainty. The proposal changes representations, training tasks, schedules, added capacity, access restrictions, and parameter sharing at once. Any improvement could result from extra semantic supervision or extra trainable modules. A comparison with the original pretrained generator alone would not test the proposed synthesis. Conversely, an unsuccessful image demonstration would not reveal which hypothesis failed.

The current central hypothesis is also permissive: some components will help predict others, and some computations may share while others specialize. Almost any result could satisfy that formulation. The fallback to expanding access and coarser sharing is sensible engineering, but it needs an explicit scientific interpretation when the sparse/reusable hypothesis fails.

**Likely reviewer interpretation:** “I can see the training agenda, but I cannot tell what result would convince me that the core idea worked.”

**Specific change:** Add a compact hypothesis-and-test table. Use an otherwise identical dense model with fixed groups and synchronous noise, a fixed semantic-first schedule, a learned schedule with fixed groups, and the proposed learned groups plus measured access/sharing. Include random groups or random access at equal sparsity where they answer a concrete question. Match semantic supervision, training examples, trainable capacity, and sampling evaluations; report the cost of learning the design. Predeclare a held-out composition metric and a mechanistic specificity/rescue metric. Give realistic planned sample sizes and decision criteria rather than inventing an expected improvement. State what negative finding would remain useful to Sony.

### 3. The representation-learning step is both the main novelty and the least specified algorithm

**Passages:** Section 2.1, “the denoising objective determines how that information should be divided”; Section 2.2, “Refine the extractors ... while making each component predictable from a limited set of others”; Section 5, “four to eight component groups.”

**Type: substantive method gap, with an exposition component.** The reader gets an exact loss for schedules and denoisers, but no equally concrete account of how a dependency measurement changes an extractor. Are projected coordinates reassigned to groups, rotated, pruned, or updated continuously? What objective trades semantic supervision, reconstruction, sparsity, and denoising? How are input sets selected when several correlated components carry the same information? What constitutes stabilization between phases?

Fixed dimension and normalized scale are useful controls, but do not establish a meaningful factorization. Information can be redundantly represented across correlated projected coordinates, concentrated into one group, or routed through residual visual features. Disjoint coordinates do not imply disjoint information. Keeping feature dimension fixed also does not match the capacity of the added semantic branch, interfaces, and task-specific projections. A decomposition may be statistically useful without aligning with a unique semantic story; the proposal correctly avoids requiring a human label per component, but still needs a criterion for the kind of interpretation it will support.

**Likely reviewer interpretation:** “The hardest new learning problem is described as a refinement step; the explicit mathematics largely belongs to the existing schedule method.”

**Specific change:** Commit to one initial extractor family and one update procedure. State the reconstruction and task objectives, the access-set selection rule, and the constraint preventing a single component from absorbing the useful information. Explain what stays fixed during each update and how the final model is selected. Frame nonlinear overlapping extractors as optional. Add diagnostics of retained information, redundancy, graph stability across seeds/checkpoints, and held-out predictive performance. A bounded algorithm is more persuasive than a menu of token groups, coordinate groups, projections, MLPs, and attention extractors.

### 4. The bridge from training-time information availability to useful generated states is untested

**Passages:** Section 1, “Making that information available earlier can help other components”; Section 2.2 defines $D_{j\to i}$ using Gaussian-corrupted clean targets; Figure 2 contrasts noisy and clearer role information; Section 5 describes generated Qwen scene-description states.

**Type: substantive scientific risk.** The cleaner component in the dependency experiment contains more information from the correct training target by construction. During generation, an advanced component can be confident and wrong. Making a role state progress earlier may reinforce a mistaken assignment, and its true usefulness can differ from its usefulness under teacher-forced corruption. The draft recognizes the tradeoff that earlier prediction has less context, and the flow formulation is coherent; however, a reduction in paired denoising loss is not itself evidence of better closed-loop compositional generation.

There is also a mismatch between diagnostic tasks that withhold role labels and the proposed creator request that supplies them. In a fully specified prompt, T5 already carries the role assignment, so the generated semantic branch may add little or may be ignored. In an incomplete prompt, “correct” unprovided roles may not be uniquely determined. Both regimes are legitimate, but they test different capabilities.

**Likely reviewer interpretation:** “The proposal shows why ground-truth semantic information helps, but has not shown how it will determine whether generated semantic information helps.”

**Specific change:** Make the oracle-to-generated gap an explicit experiment. Compare the same image prediction under clean target features, correctly corrupted targets, generated semantic states, and a disabled semantic branch. Measure semantic-state errors and their downstream effect through complete rollouts. Separate full-prompt instruction following from missing-information inference. Show that benefits survive with the full prompt, fixed guidance, and held-out combinations. These can all be proposed award-period tests; new preliminary experiments are not required to state them.

### 5. The implementation and personnel plan do not yet make the 12-month scope credible

**Passages:** Section 5 selects a roughly 0.6B PixArt-Σ backbone plus DINOv2, Qwen, T5, and VAE components; it proposes freezing the backbone, then releasing selected parameters; one graduate assistant has twelve months of support; “no separate computing costs are requested.”

**Type: substantive feasibility gap.** This is several coupled projects: constructing interaction data, learning feature interfaces, adapting a text-to-image model, optimizing schedules, learning component groups, implementing multiple sharing patterns, establishing causal evidence, and evaluating reference-conditioned composition. The existing small-model pipelines and schedule code are real strengths, but transferring a result on CelebA/AAHQ/STL-10 to reference-conditioned relational scenes is a large step.

The architecture is still underspecified at the integration point. The proposal's generative formulation predicts linear-interpolation flow displacements. It must explain how the selected pretrained image denoiser's existing input scaling, time embedding, prediction convention, and sampler are preserved or converted. I did not verify the checkpoint's exact parameterization and do not claim a demonstrated incompatibility. The gap is that the document offers no adaptation mapping while using backbone freezing as a central feasibility argument. It also remains unclear whether final image latents are generated directly throughout or reconstructed from a learned projected component interface.

No computing budget is required if resources already exist, but access and approximate run costs should be stated. Caching features reduces encoder work; it does not quantify the repeated diffusion adaptation and intervention cost. Reference identity across scenes is another underdeveloped resource requirement. Scene role annotations and object crops do not automatically supply training pairs that preserve a character's identity across new compositions.

**Likely reviewer interpretation:** “The plan may be feasible for this lab, but the application makes me infer the missing infrastructure and engineering effort.”

**Specific change:** Specify the initial component shapes, image-decoding path, denoiser parameterization bridge, trainable parameter count, and available compute. Give a bounded dataset size and a realistic run budget. Name a minimal committed milestone: two participants, one interaction family, one representation refinement, one sharing hypothesis, one rigorous mechanistic case study. Add a month-3 or month-6 decision gate for scaling. Treat broader reference identity, multiple simultaneous activities, and video as stretch directions if resources do not support all of them.

### 6. The novelty is plausible, but the document does not isolate the new scientific insight

**Passages:** Section 3, “The proposed advance is to learn representation dependencies, denoising order, and the transformer computations that implement them together”; “Generation order becomes a means of learning the representation's dependence structure.”

**Type: mostly differentiation/exposition, with a substantive test missing.** Joint synthesis can be novel and valuable; absence of a “first” claim is not a defect. The draft properly acknowledges semantic-first diffusion, independent noise levels, schedule learning, and local mechanisms. However, a reviewer can still read the proposal as extending the PI's LWD from two fixed representations to several learned groups, then adding familiar sparse attention and task-specific modules. The actual dependency identification step is the comparison across independently varied noise levels; scheduling subsequently uses that estimate. Saying that order discovers structure blurs this distinction.

The exact new insight should be a relationship the research will test: for example, whether predictive dependencies under controlled corruption forecast a computation's cross-task reusability and whether exploiting this relation improves held-out binding. That would unify the three parts scientifically, beyond saying they inform one another.

**Specific change:** Put the gap before the formalism. In a short comparison table, identify which closest methods fix representations, learn schedules, estimate component dependencies, or test module transfer. Then state the one additional relationship this project will establish or falsify. Include the corresponding experiment. The public LWD method already contains joint schedule/denoiser probing, stop-gradient weighting, and a kinetic penalty, so those details principally establish a foundation rather than the new contribution. [LWD method](https://arxiv.org/html/2606.19662v1)

### 7. Preliminary evidence receives too much space relative to its direct support for the hypothesis

**Passages:** Figure 3 and its normalized DINO recovery; Figure 4 and the 61.94% GSM8K result; Section 3's FID and parameter-count comparisons.

**Type: evidentiary interpretation and organization.** Figure 3 supports selective weight compatibility under domain adaptation. It does not yet support reuse of an identified conditional computation, or the attention/MLP functional split at the level promised here. The text is appropriately cautious about motivation, but the reader needs the exact recovery metric, uncertainty across the two seeds, and a clear statement of what a successful weight reversion establishes. The STL-10 failure may reflect substantial domain/interface shift, rather than a diagnostic test of conditional-task specialization.

Figure 4's bars visually invite model-efficiency comparisons even though the caption says the settings differ. The 795M number excludes a frozen 4B encoder, and the figure does not show that exclusion. The prose does. The benchmark is also several steps removed from multimodal role binding: generating useful language states is relevant feasibility evidence, but GSM8K accuracy does not validate controllable visual grounding. The final digit of the parameter count has little persuasive value.

I checked the cited LWD paper: the reported 1.02, 1.05, 4.93, and 5.84 results and the iteration counts are supported. The stronger matched comparison is with SFD; the large REPA update ratio compares different overall representation systems. Preserve the distinction between a reported system comparison and a gain attributable to schedule learning. The headline 1.02 result should explicitly identify class-conditional ImageNet-256 and AutoGuidance. [LWD results](https://arxiv.org/html/2606.19662v1)

**Specific change:** Lead feasibility with the matched schedule result and the component-reversion finding, including their limitations. Compress the language-model result to a short paragraph or a small table with total components and protocol. Use the recovered space for the experimental tests. No new experiments are necessary to improve the honesty and relevance of the existing evidence.

### 8. The mathematical detail is disproportionate to the remaining scientific uncertainty

**Passages:** Section 1's definitions of $N,d,D,P_k$, Section 2.2's lengthy stop-gradient explanation, and the generic update $\theta^+=\theta-\eta\nabla_\theta\mathcal J$.

**Type: exposition and prioritization, with a bounded technical caution.** The core symbols are defined, noise direction is clear, training targets are distinguished from conditions, and the final ODE includes the required schedule-rate factor. These are strengths. The problem is not a lack of notation definitions. It is that the reader learns the elementary mechanics of a gradient update before learning how the new grouping algorithm is implemented or how mechanistic success is judged.

The joint preliminary loss also merits a concise qualification: the velocity penalty updates the denoiser as well as the schedule. At a fixed progress configuration with rate $a>0$, minimizing $a\|f-v\|^2+\lambda a^2\|f\|^2$ gives $f^*=\mathbb E[v\mid\text{inputs}]/(1+\lambda a)$ in the unrestricted regression setting. Thus the preliminary predictor is deliberately biased by the penalty. Dropping the penalty for the final model, as proposed, is a reasonable resolution; dependency maps and schedules chosen in the preliminary phase should still be checked after final denoiser adaptation. This is a risk to validate, not evidence that the flow formulation is invalid.

**Specific change:** Retain the component interface, the dependency statistic, and the generation equation. Shorten the general gradient and stop-gradient tutorial. Spend those lines on extractor updates, schedule parameterization/selection, loss weighting among tasks, and final-model revalidation. Explain that dependency estimates are conditional on the fitted model and can change after adaptation.

## Organization and figure review

The current sequence delays the concrete implementation until Section 5. A reviewer must carry a broad abstract formulation through two detailed sections before learning the backbone, four-to-eight groups, training data, and frozen-module strategy. Move a compact implementation paragraph near the first formulation. Then organize around the scientific question, bounded method, decisive tests, and feasibility.

The contribution claim is repeated in the abstract, Section 1, the Section 2 opening, Section 3, Section 4, and Section 5. Several repetitions can be removed. Sections 3 and 4 mix novelty, prior performance, relevance, evaluation, and speculative video extension. Separate their functions clearly. The final list of future applications consumes space without strengthening the committed deliverable.

- **Figure 1:** Legible and conceptually helpful, but it visually assigns named meaning to selected coordinates more strongly than the prose warrants. Consider showing a learned projection and a separate semantic probe/readout. Label the noisy prediction target directly in each task path. The caption's illustration caveat is good and should remain.
- **Figure 2:** The clearest explanatory figure. It identifies controlled variables and connects noise availability to a module contribution. It partly duplicates Figure 1. Combining their most useful panels could free meaningful proposal space. The illustrated global order should remain explicitly labeled hypothetical.
- **Figure 3:** Useful preliminary evidence, with sample/seed counts helpfully disclosed. Define “normalized recovery” sufficiently to reproduce its meaning, distinguish it from quality or task accuracy, and show seed variation if existing records permit. The image panel demonstrates a striking domain difference but does not establish the proposed semantic mechanism.
- **Figure 4:** Lowest value per unit space. Its benchmark bars communicate an informal comparison more forcefully than their caveat. Compress or replace with a factual feasibility entry; disclose all encoder/decoder components alongside the denoiser count.

The drafting notes, missing PI contact details, and editorial notes must be removed or completed before submission. They are production issues rather than scientific criticisms. This review did not verify compiled-PDF pagination or administrative eligibility.

## What I would preserve

The handover example gives the research a concrete failure mode. The formulation correctly distinguishes training targets, noisy states, supplied conditions, and generated states. The proposed noise-level pairing and same-example comparisons are more rigorous than an informal attention visualization. The existing schedule-learning results give credible evidence of execution ability. The limited initial component count, frozen backbone strategy, and explicit fallback on sharing are sensible starting points. Sony fit is substantive through modality binding, generation dynamics, reusable internal patterns, and causal intervention; the proposal does not need to add every topic in the call.

## Revision priorities

1. Specify a falsifiable mechanism claim and a complete semantic intervention/rescue experiment.
2. Add matched ablations, held-out composition criteria, and a quantitative evaluation plan.
3. Make representation refinement and dependency-based access selection an explicit bounded algorithm.
4. Test the oracle-to-generated information gap and distinguish full-prompt control from missing-information inference.
5. State the backbone integration, compute access, dataset scale, and minimum one-year deliverable.
6. Reorganize around these decisions, reducing repeated claims, elementary update notation, and the large GSM8K figure.

With these changes, the proposal could become a focused investigation of a genuinely interesting relationship: how the usefulness of internal information during generation predicts the computations that can be shared and controlled. As written, it promises that relationship more clearly than it specifies how to establish it.
