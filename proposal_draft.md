# Mechanisms of Context Consistency in Multimodal Diffusion Models

**Principal investigator:** Xiang Cheng, Department of Electrical and Computer Engineering, Duke University  
**Focused Research Theme:** Internal Mechanisms of Multimodal Generative Models for Content Creation  
**Project duration:** 12 months  
**PI contact:** [Email to insert]; [Phone with country code to insert]

*Proposal draft, 13 September 2026. Figure placeholders and editorial notes are marked for removal or replacement before submission.*

## Abstract

A creator may ask a generative model to preserve an object's identity from one reference, adopt a material or style from another, and obey spatial relationships specified through language. Achieving this context consistency requires the model to bind different sources to different properties and maintain those bindings throughout generation. We propose to understand how the representation of these requirements relates to the internal computations and parameters that realize them. Our central hypothesis is that representations exposing sparse, recurring conditional dependencies can support reusable denoising computations, and that their alignment with attention and prediction components makes parameter effects more selective and predictable. Aim 1 will construct and test semantic–visual representations using masked prediction and controlled variations, including examples produced by existing generators. Aim 2 will identify how attention selects and transports the relevant information, determine which prediction components can be reused, and predict responses to both inference interventions and training. Controlled composition tasks and text-and-reference image generation will test predictions on unseen combinations. Our preliminary results and external studies establish feasibility for generative semantic states, coordinated denoising, and selective parameter interventions. The project will deliver a causally tested account of when these mechanisms align, together with a creator demonstration of selective control and reusable adaptation.

## 1. Motivation, hypothesis, and contribution

Content creation requires consistency with several sources of context at once. Consider a concept artist who supplies a reference object, a separate material reference, and the instruction to place the object to the left of a second asset. A useful generator must determine which reference supplies each property, preserve the requested object features, and realize the spatial relationship. If the material is changed later, the artist expects the object and arrangement to remain recognizable.

We use **context consistency** to describe this agreement between generated content and the requirements supplied by its context. The concept also motivates audiovisual synchronization and temporal coherence, but this project will focus on text-and-reference image generation. This setting provides a concrete multimodal binding problem, controlled interventions, and accessible measurements of composition and preservation.

The scientific question is: **Which representations expose the computations needed to satisfy contextual requirements, and how do the parameters of a diffusion Transformer implement and modify those computations?** During inference, parameter changes alter evolving representations and the final image. During training, changes in examples, representation targets, or losses induce parameter updates. Understanding both processes could explain when a desired change can be made selectively, when a learned component can be reused, and when coordinated adaptation is necessary.

**Central hypothesis.** Representations that expose sparse, recurring conditional dependencies can support a decomposition of denoising into reusable computations. Suitable representations and training conditions can align these computations with identifiable attention and prediction components, enabling predictions of their semantic effects under intervention and learning.

Sparsity provides a measurable starting point: recovering a state may require only a few other tokens or feature groups. Recurrence provides the reason to expect reusable parameters: many scenes require the same operations of selecting an object, retrieving its attributes, and translating those attributes into appearance. These relationships form an implicit dependency structure. We will study them through prediction and intervention, using graphs as explanatory abstractions. The hypothesis concerns functional organization, rather than recovery of a unique adjacency matrix.

This organization is not guaranteed by sparse dependencies alone. Shared parameters can entangle otherwise separable computations, and an early intervention can alter later attention. Establishing when the alignment exists, and predicting its failures, is the central research contribution.

The two aims address this hypothesis together:

1. **Learn representations that expose reusable conditional computations.**
2. **Explain and predict how attention and prediction parameters implement, acquire, and modify those computations.**

### Differentiation from existing work

Several parts of this program have strong precedents. Joint image–text diffusion is established by [UniDiffuser][unidiffuser]; [SFD][sfd] and [SeFi-Image][sefi] establish semantic-first visual generation. [Local Mechanisms of Compositional Generalization][local] connects sparse conditional-score dependencies to a specified compositional structure, including a feature-space formulation. [Vision-Language Binding in In-Context Image Generation][binding] identifies causal reference-to-text-to-image pathways. [Concept Sliders][sliders] learns selective low-rank parameter directions, including from controlled changes in another generator's representation.

Our contribution will be a **predictive connection between representation structure and parameter mechanisms**. We will test whether a representation's dependency structure predicts which components must change, which can be retained, and which output properties a finite update will affect. Predictions will be specified before evaluation on held-out compositions and adaptation tasks. This goes beyond locating useful features or finding a successful adapter: the proposed account must explain the internal route of an effect and predict preservation and interference.

> **Figure 1 placeholder — Creator problem and scientific hypothesis.** Use one running example with an object reference, a material/style reference, and a spatial instruction. Show semantic and visual state groups beside the attention-selection, feature-transport, and prediction operations proposed to implement their dependencies. Distinguish denoising time from optimizer steps. Include a small inset showing a material change with intended preservation of object and arrangement. All example outputs in this overview should be labeled illustrative unless taken from a documented experiment. The visual should communicate the representation–computation–parameter connection without suggesting a fixed graph or one head per concept.

## 2. Aim 1: Learn representations that expose reusable conditional computations

### 2.1 Semantic states as generative variables

Representations learned through different tasks retain different aspects of a scene. General self-supervised features such as [DINOv2][dino] provide a starting point for visual semantics; contextual language-model states can express object attributes and relationships. Multimodal alignment, reconstruction, and task-specific discrimination or reward learning provide additional sources when the initial representations omit a required distinction.

Our initial implementation will extend a semantic/texture DiT with text and reference tokens carrying explicit source identifiers. It will jointly denoise image latents and compact semantic states. Visual semantic targets will come from a frozen visual encoder; language-semantic targets will come from frozen Qwen contextual activations of scene descriptions. Controlled scenes will provide precise descriptions, and generated or curated image–description pairs will support natural-image experiments. At inference, the original prompt and references remain fixed conditioning, while semantic and visual states evolve from noise. Generated descriptions cannot redefine the requirements against which the image is evaluated.

We will begin with visual semantic states and then add language-semantic groups in the same architecture, using comparisons that hold the representation source fixed. The comparisons will distinguish ordinary conditioning, an autoregressively generated scene description, and explicitly generated semantic states. The central language question is whether precision available in the contextual states becomes effective visual binding, particularly for counts, attribute assignments, and spatial relations.

The image latent stream retains details that a semantic encoder may suppress. A representation will be useful only if it supports both faithful generation and interventions whose effects can be understood. Source-task performance alone will not determine selection.

### 2.2 Learning the decomposition through controlled variation

We will organize normalized features into token and coordinate groups using two complementary signals.

**Grouped masked prediction.** Independently mask or corrupt selected regions, semantic groups, or entire representation streams, then predict their clean targets from the remaining information. [MultiMAE][multimae] supplies a precedent for learning through prediction across masked modalities. We will vary semantic and visual noise levels separately to determine which dependencies change as information becomes reliable.

**Controlled-change supervision.** Construct pairs and small sets of examples that vary one designated factor while preserving others. Existing generators will provide material, style, and arrangement variations across multiple objects and backgrounds. Synthetic editing supervision is feasible at scale, as demonstrated by [InstructPix2Pix][instruct]. We will verify intended and preserved properties independently and use a small human audit, treating requested edits as imperfect labels.

The first construction will learn rotations and groupings of normalized frozen features, preserving their information content. Controlled pairs will encourage stability of preserved factors and concentration of the intended change in a small group. Swapping that group between compatible examples will test whether it produces the intended visual change. If a compact learned encoder is needed, reconstruction and prediction of fixed teacher features will constrain information loss. Original features and random groupings will provide controls.

This design allows a material-related feature group to recur across scenes while attention routes it to different objects. It also permits legitimate coupled effects, such as a new material changing highlights. The objective is selective semantic variation, not invariance of every pixel. For missing-input experiments, masking will occur before contextual encoding; completion of already encoded states will be reported separately.

### 2.3 Measuring useful dependence and generation order

At a specified representation interface, let Y_i be a target group and U the available states. Its own noisy observation, if present, is held fixed. We will estimate the extra prediction error incurred by restricting access to other states:

```text
R_i(S, t) = inf_f E[||Y_i - f(U_i, U_S, t)||^2]
k_i(epsilon, t) = min cost(S)
                 subject to R_i(S, t) - R_i(all, t) <= epsilon.
```

Here U_i is the target's noisy observation or mask, t denotes the semantic and visual noise levels, and cost counts accessible tokens or fixed-size feature groups other than i. In practice, predictors will have comparable capacity and training budgets. We will report loss-versus-access curves, absolute generation quality, and the benefit retained from additional context. Representation dimensions and feature scales will be controlled.

A shared semantic summary may legitimately support many visual predictions. Its dimension and construction cost will remain explicit. When subset selection itself examines the full context, we will distinguish sparse message use from sparse computation throughout the network.

Three tests will determine whether the decomposition is useful: restricted-information prediction, actual intervention on the trained generator, and generalization to unseen factor combinations. Random train/test splits will measure ordinary generalization; withholding material–object and object–relation combinations will test compositional reuse. Comparisons using identical generated examples will isolate the contribution of the representation objective.

We will also test whether making one semantic group reliable reduces the context required to predict another group or the visual stream. These measurements will propose semantic/visual denoising schedules, evaluated against synchronous and fixed semantic-leading schedules. Final generation must confirm that a locally useful dependency produces persistent consistency across the trajectory.

**Aim 1 outcome:** a representation and a validated description of its conditional computations, including the factors they preserve, the information they require, and how these properties change during denoising.

> **Figure 2 placeholder — Learning and testing a decomposition.** Show a small grid of matched scenes crossing two factors, such as material and arrangement, followed by grouped semantic/visual states and masked-prediction tasks. Reserve a panel for the proposed loss-versus-access curves at two noise regimes and a group-swap test. Show original/random/learned groupings as planned comparisons, without fabricated curves or scores. This figure should make clear how controlled variation supplies supervision beyond an ordinary collection of images.

## 3. Aim 2: Predict how parameter mechanisms realize and modify the decomposition

### 3.1 From conditional dependence to attention and prediction

We will investigate three cooperating operations: selecting relevant context through query–key interactions, transporting selected features through value/output projections, and predicting a representation component through MLPs and surrounding transformations. [Compositional Attention][compositional] demonstrates the utility of separating search from retrieval in attention. Our question is how these operations participate in denoising and whether their reuse follows the representation structure identified in Aim 1.

With normalization and residual paths omitted from the notation, the attention message has the form

```text
message_i = sum_h sum_j a_ij^h(U) M_h u_j,
M_h = W_O^h W_V^h.
```

The coefficients a select tokens, while M mixes feature coordinates. We will test which feature groups each head reads and writes, and whether several heads jointly realize a recurring dependency. This connects token and coordinate dependence to specific parameter operations.

A functional module may span several heads, feature subspaces, and layers. We will first analyze the dense trained model. Using the discovered organization to impose selective sharing or access restrictions will be a separate causal test of the proposed explanation.

Candidate pathways will be identified through grouped prediction and sensitivity measurements, then tested by removing messages, transplanting states between matched examples, and restoring a proposed mediator. Measurements will include immediate denoising effects and final semantic changes under paired initial noise. Restoring the predicted state should recover the corresponding behavior when the mechanism is correctly identified. We will explicitly test whether value/MLP changes alter later routing, since retaining query–key weights does not imply retaining the attention pattern.

### 3.2 Predicting the boundary of component reuse

The main experiment crosses changes in visual content with changes in compositional relationships:

| Adaptation task | Prediction to be tested |
|---|---|
| New material or appearance; familiar relationships | Context selection may transfer if its aggregate retains the information required by the new predictor. |
| Familiar content; changed binding or composition rules | Selection or feature transport may require modification while parts of the prediction mechanism remain reusable. |
| Both change | Coordinated updates may be needed because independently reusable components have incompatible interfaces. |
| Neither changes | A reference condition measures drift and unnecessary parameter changes. |

For example, rendering a new material on familiar objects may preserve the rule selecting the relevant geometry and illumination. Changing which reference supplies an object's material may instead require changing the binding operation.

We will develop diagnostics on source data and designated development shifts, then fix the prediction procedure before evaluating new adaptation tasks. Each new task will provide a limited support set; separate query examples will contain unseen compositions. We will predict the parameter groups requiring adaptation and compare direct training restricted to MLPs, value/output plus MLPs, query/key components, and broader updates. Diagnostic cost, trainable parameter count, and adaptation compute will be reported and controlled in the relevant comparisons. Coarse component tests will be followed by interventions on predicted heads and projection subspaces, compared with alternative allocations of the same update budget.

The disjoint-data experiment will retain an aggregation mechanism learned on one subset while fitting a predictor on another. It asks whether the aggregate remains sufficient across the change, rather than merely whether the complete network generalizes.

A theoretical starting point follows from conditional-mean prediction. If U is the full denoising input and A(U) is the retained aggregate, the irreducible excess squared error from using that aggregate is

```text
aggregation_gap = E[||E[Y | U] - E[Y | A(U)]||^2].
```

The target's local observation is included in A(U). If this gap is small under the target task, a sufficiently expressive and successfully trained predictor can reuse the aggregate. This does not assert that arbitrary MLP replacement succeeds: finite predictor capacity, optimization, and changing hidden-state interfaces remain separate sources of error.

We will first analyze controlled linear/Gaussian examples and explicit local predictors, then estimate the gap and its proxies in small DiTs. Our existing single-layer attention analysis provides bounds involving discarded attention mass and transported-feature magnitudes. We will study their propagation under stated stability assumptions and test where changing routing or amplification invalidates a local prediction. The same account will support one bounded sharing experiment: reuse a prediction bank across depth with layer-specific access, measuring quality against unique parameter count.

### 3.3 Connecting inference effects to training dynamics

At inference, we will predict how a finite parameter change affects selected semantic states, the relevant information paths, and the final image. During training, we will specify a change in examples, representation targets, or loss, and predict the resulting learning response from a common initialization. These are distinct processes connected by the network's computation.

For a fixed input and half-squared prediction loss, perturbing the target by delta_y gives the exact one-step SGD diagnostic

```text
change_in_parameter_update = learning_rate * J_theta^T * delta_y,
```

where J_theta is the prediction's parameter Jacobian at the common starting point. This supplies an initial connection between semantic target changes and parameter groups. A change in representation can also change the inputs and loss geometry; those effects will be measured separately.

We will extend beyond a single linearization through repeated linearization along short adaptation trajectories and structured approximations of interactions among selection, transport, and prediction components. Extensive flow-training runs on development tasks will provide reference trajectories and well-trained endpoints. They are operational references, not unique ideal weights. Predictions will be evaluated through denoising fields, semantic readouts, and final behavior, as well as the allocation of parameter changes.

The strongest test will identify a successful representation intervention and predict the parameter components capable of reproducing its effect across new scenes. One fitted update will be reused across multiple prompts and image-noise realizations. Generic low-rank adaptation and Concept Sliders will be comparisons using the same supervision and comparable budgets. We will measure intended change, preservation, and agreement with the predicted internal mechanism. Within this same experiment, combining two updates from a shared base will provide a limited test of predicted interference.

**Aim 2 outcome:** a causally validated procedure that predicts which components can be reused, which must change, and when their interactions prevent selective control. Improved adaptation performance will support the practical value of the account; predictive accuracy on unseen tasks will establish its explanatory value.

> **Figure 3 placeholder — Predicted versus observed mechanism.** Use the crossed content/composition experiment as the layout. Show how diagnostic observations lead to a prediction about reusable selection, transported features, and prediction parameters. Reserve side-by-side panels for predicted and observed semantic effects, including a failure requiring coordinated adaptation. A compact response matrix can use semantic factors as rows and parameter groups as columns. This is a planned result figure; do not populate it with invented observations.

## 4. Feasibility and preliminary evidence

Existing work supports several interactions needed for this project. SFD couples semantic and texture generation, while SeFi-Image extends semantic-first modeling to text-to-image generation. The locality study provides both a controlled architectural intervention supporting composition and suggestive feature-space evidence in SDXL; its SDXL heuristic does not by itself establish a causal parameter decomposition. Concept Sliders transfers controlled StyleGAN variations into diffusion-model parameter directions through paired-image supervision. Together with causal reference-binding studies, these results make the proposed measurements and interventions credible. [SFD][sfd], [SeFi-Image][sefi], [Local Mechanisms][local], [Concept Sliders][sliders], [Vision-Language Binding][binding].

Our preliminary work supplies complementary analytical and experimental capability:

| Evidence | Established result | Relevance to the proposed work |
|---|---|---|
| Generative language-semantic states: ELF-L, unpublished | A separately trained diffusion model generates projected frozen-Qwen contextual answer states, reaching 61.94% GSM8K accuracy. Qwen encodes the question but does not autoregressively generate the solution at inference. | Demonstrates that reasoning-relevant contextual states can themselves be generative variables. Visual grounding is the proposed next step. |
| Coupled representation trajectories | With a matched 675M backbone, learned asynchronous schedules reach AutoGuidance FID 1.05 after 200 epochs, matching an 800-epoch SFD-XL result. [Learning When to Denoise][schedule] | Establishes a working multi-representation training pipeline and control of semantic/texture timing. |
| Interpretable denoising computation | [From Softmax to Score][softmax] gives constructive attention–denoising connections. Our working mechanistic manuscript develops context aggregation, conditional-mean sufficiency, and single-layer masking analysis. | Supplies the initial theoretical tools for aggregation reuse and intervention analysis. |
| Parameter allocation and reuse, unpublished | On CelebA64, a shared prediction-bank variant improves FID from 17.574 to 14.304 at approximately 10.2M parameters. Restoring source QK in jointly adapted models yields paired DINO recovery of 0.815 for CelebA→AAHQ and 0.143 for CelebA→STL-10. | Motivates both selective sharing and the need to predict when component reuse fails. |

The parameter-allocation result uses one training seed. The transfer results are post-training component-reversion experiments relative to jointly adapted generators; they do not establish that freezing the same components throughout training would succeed. Direct restricted training is therefore part of Aim 2. ELF's reasoning performance establishes functional utility of generated contextual states, without establishing visual grounding or equivalence to the teacher's internal reasoning. Our work on [anisotropic trajectory optimization][trajectory] provides additional experience with representation-dependent diffusion dynamics.

The remaining uncertainty is the central hypothesis: whether a useful representation decomposition aligns with reusable parameter mechanisms. We will test that interaction early in the project using the common controlled task. A negative result will be analyzed through information loss, shared-parameter interference, and changing routing, allowing the account to predict when a broader update is necessary.

> **Figure 4 placeholder — Existing evidence supporting the proposed interaction.** Assemble three compact panels from existing material: (a) ELF's frozen-Qwen target extraction and diffusion generation, with reasoning accuracy as functional validation; (b) the published semantic/texture schedule result using matched-backbone training curves or a compact comparison; (c) one successful and one unsuccessful component-reversion case, paired with the recovery values above. Label unpublished results and distinguish post-training reversion from direct restricted training. Use documented example selection and identify illustrative images as such. If space is tight, omit the separate shared-bank FID panel and retain its table entry.

## 5. Creator demonstration and relevance to Sony

The demonstration will support **reference-guided asset variation**: preserve selected object features from one reference, transfer appearance from another, and obey spatial instructions. It will show both an image-specific representation edit and a reusable parameter update applied across new scenes. The interface will expose the selected source properties and requested changes; the research report will explain the internal pathways mediating them.

Evaluation will separate object/content preservation, appearance transfer, attribute binding, and spatial accuracy. Automated factor measurements on controlled scenes will be complemented by blinded human assessment on natural images. We will report FID and prompt–image alignment alongside the factor-specific tests, so aggregate quality cannot conceal failed binding or preservation. The same examples will be used across methods. Results will include unsuccessful edits and cases in which the predicted mechanism requires coordinated updates.

A bounded extension of this workflow will introduce a known appearance collection through a source-specific module. Enabling, replacing, or disabling that module will test its contribution through the predicted semantic states and final outputs. This provides a concrete route toward controllable source use and knowledge externalization. The claim will concern the newly introduced module's contribution; removing it does not establish erasure of related information already present in the base model.

These deliverables address Sony's interest in internal mechanisms, cross-modal binding, causal intervention, and controllable source contributions. They could inform asset development and visual iteration for games, animation, and film. Video, audio, and general-purpose model merging remain longer-term applications of the mechanism.

> **Figure 5 placeholder — Demonstration storyboard.** Show one reference object and one appearance reference above a row of new text-specified scenes. Reserve matched columns for the base model, a representation intervention, and one reused parameter update. A final pair should show the source-specific module enabled and disabled. Annotate requested and preserved properties separately. Use future demonstrator outputs when available; for submission, use a clearly labeled storyboard or existing documented examples, with no implied new experimental result.

## 6. Work plan, deliverables, and scope

| Period | Research milestone | Evidence of completion |
|---|---|---|
| Months 1–3 | Establish the common semantic/visual DiT, controlled-change data, and initial grounding tests. | Reproducible baselines; independent factor measurements; first integrated state intervention with documented success or failure. |
| Months 4–6 | Learn and test representation decompositions and identify candidate parameter mechanisms. | Held-out prediction-versus-access curves; message removal/restoration tests; prospective component-reuse predictions. |
| Months 7–9 | Evaluate finite parameter changes and learning responses; transfer the tests to natural-image contexts. | Direct restricted-training comparisons, reference adaptation trajectories, and tests of preservation on unseen compositions. |
| Months 10–12 | Complete the creator demonstration and consolidate the mechanistic account. | Reusable-update and source-module demonstrations; reproducible evaluation package; final analysis of valid regimes and failures. |

The core experiments will use small controlled DiTs and the existing multi-representation training pipeline. Natural-image validation will reuse a pretrained generator and limit adaptation to the interfaces under study. This concentrates computation on mechanistic comparisons and avoids foundation-model pretraining as a project dependency. Synthetic examples and feature targets can be prepared once and reused across controlled comparisons.

The project will prioritize the two aims and one creator workflow. Conditional generative models of weight updates, broad meta-learning, and long-video generation are follow-on directions. The principal technical risk is that semantic predictability does not translate into selective causal control. The staged design will reveal this through independent grounding tests, fixed-model interventions, and direct training, rather than inferring control from a probe alone.

We will provide three quarterly reports and a final research summary, with progress-questionnaire responses as required. Scientific deliverables will include representation and dependency diagnostics, parameter-intervention and adaptation protocols, benchmark configurations, and the creator demonstration. The intended outcome is an explanatory account whose predictions help determine how to change generated content while preserving the contextual requirements that matter to its creator.

## References

1. Oquab et al. [DINOv2: Learning Robust Visual Features without Supervision][dino]. arXiv:2304.07193, 2023.
2. Bachmann et al. [MultiMAE: Multi-modal Multi-task Masked Autoencoders][multimae]. arXiv:2204.01678, 2022.
3. Bao et al. [One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale][unidiffuser]. arXiv:2303.06555, 2023.
4. Pan et al. [Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion][sfd]. arXiv:2512.04926, 2025.
5. SeFi-Team. [SeFi-Image: A Text-to-Image Foundation Model with Semantic-First Diffusion][sefi]. arXiv:2606.22568, 2026.
6. Bradley. [Local Mechanisms of Compositional Generalization in Conditional Diffusion][local]. arXiv:2509.16447, revised 2026.
7. Mittal et al. [Compositional Attention: Disentangling Search and Retrieval][compositional]. ICLR, 2022.
8. Gandikota et al. [Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models][sliders]. arXiv:2311.12092, 2023.
9. Ge et al. [Vision-Language Binding in In-Context Image Generation][binding]. arXiv:2605.24624, 2026.
10. Brooks, Holynski, and Efros. [InstructPix2Pix: Learning to Follow Image Editing Instructions][instruct]. CVPR, 2023.
11. Rosu, Carin, and Cheng. [From Softmax to Score: Transformers Can Effectively Implement In-Context Denoising Steps][softmax]. NeurIPS, 2025.
12. Liu, Li, and Cheng. [Variational Trajectory Optimization of Anisotropic Diffusion Schedules][trajectory]. arXiv:2602.19512, 2026.
13. Qian and Cheng. [Learning When to Denoise: Optimizing Asynchronous Schedules for Latent Diffusion][schedule]. arXiv:2606.19662, 2026.

**Unpublished preliminary materials.** *A Mechanistic View of Diffusion Transformers as Adaptive Patchwise Denoisers*, working manuscript; associated Experiment 1 and Experiment 3 reports; ELF-L experimental summary and accuracy records, 2026. These materials support the explicitly labeled preliminary results in Section 4.

## Budget summary — separate page in the submission

*Editorial note: complete amounts and effort with the institutional budget. The [Focused Research Award limit][sony] is USD 150,000 inclusive of indirect costs; no award amount or staffing commitment is assumed in this draft.*

| Cost category | Research purpose | Amount (USD) |
|---|---|---|
| Personnel and associated benefits | Research effort for representation learning, mechanistic analysis, and experiments | [To complete] |
| Computing | Controlled training, reference adaptation runs, feature extraction, and generator evaluation | [To complete] |
| Other justified direct costs, if applicable | [Specify or remove] | [To complete] |
| Indirect costs | Institutional rate and applicable cost base | [To complete] |
| **Total requested** | **At or below USD 150,000** | **[To complete]** |

## Editorial notes for finalization — remove before submission

- Replace the five figure placeholders with compact diagrams, documented existing results, or explicitly labeled demonstration storyboards. No new pre-submission experiment is assumed.
- Target approximately one page for the abstract/problem and overview figure, one for positioning, four for the two aims, two for feasibility and demonstration, and two for the work plan and references. Adjust after figures are laid out; the narrative and references must fit within ten pages, with the budget on a separate eleventh page.
- Complete PI contact details and the institutional budget. The PI CV is a separate submission item.
- Specify the final model checkpoints, resolutions, and compute allocation in the methods once the implementation budget is fixed. The initial implementation is the semantic/texture DiT described in Section 2; the natural-image checkpoint should support the required text/reference and internal-intervention access.
- Review the final bibliography style and complete the authorship/citation form for unpublished preliminary materials. Confirm the exact existing panels and captions used for Figure 4.
- Typeset the four plain-text mathematical displays in the submission PDF. They use fenced text here to remain readable in GitHub without dollar-sign math delimiters.
- [Sony's submission requirements][sony]: ten proposal pages including references, one budget page, minimum 10-point font, PDF under 16 MB; deadline 15 September 2026 at 11:59 p.m. PDT.

[dino]: https://arxiv.org/abs/2304.07193
[multimae]: https://arxiv.org/abs/2204.01678
[unidiffuser]: https://arxiv.org/abs/2303.06555
[sfd]: https://arxiv.org/abs/2512.04926
[sefi]: https://arxiv.org/abs/2606.22568
[local]: https://arxiv.org/abs/2509.16447
[compositional]: https://arxiv.org/abs/2110.09419
[sliders]: https://arxiv.org/abs/2311.12092
[binding]: https://arxiv.org/abs/2605.24624
[instruct]: https://arxiv.org/abs/2211.09800
[softmax]: https://papers.nips.cc/paper_files/paper/2025/hash/d27af2bebeab8a3e6f3848fc71736235-Abstract-Conference.html
[trajectory]: https://arxiv.org/abs/2602.19512
[schedule]: https://arxiv.org/abs/2606.19662
[sony]: https://www.sony.com/en/SonyInfo/research-award-program/
