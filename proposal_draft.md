# How Representations and Weights Shape Multimodal Generation

**Principal investigator:** Xiang Cheng, Department of Electrical and Computer Engineering, Duke University  
**Focused Research Theme:** Internal Mechanisms of Multimodal Generative Models for Content Creation  
**Project duration:** 12 months  
**PI contact:** [Email to insert]; [Phone with country code to insert]

*Drafting note — remove before submission: This draft uses only existing results and published work as feasibility evidence. The research experiments proposed below are work for the award period. Submission figures will use existing results or explanatory drawings; no new training, sampling, ablation, or evaluation is required.*

## Abstract

We propose to understand how diffusion models turn complex prompt semantics into images by connecting their representations, generation steps, and weights. A prompt can specify who performs an action, how several objects relate, and which reference identifies each subject. We will study how such requirements are expressed in evolving language and visual states. Aim 1 will learn how this information can be divided into useful parts and which parts need to be resolved together or earlier during generation. Aim 2 will explain how attention and prediction components combine these parts, how weight changes affect the resulting content, and how changes in representation or training targets affect the weights learned. Our hypothesis is that recurring dependencies among semantic and visual states can reveal reusable network computations. The central test is whether these dependencies predict which components can be reused and which must change for a new generation task. Text-and-reference image generation will test whether a selected instruction can be revised while preserving the other requirements. Existing results on generating Qwen states, coordinating semantic and visual generation, and reusing network components support the ingredients of this study.

## 1. The proposed contribution

**We will learn how to divide a diffusion model's representation and generation process along conceptual distinctions in a complex prompt, and test whether this division explains the roles of its weights.** The proposed approach connects three questions: what semantic information must be resolved to satisfy an instruction, which network components combine that information with the evolving image, and which computations can be reused across new concepts and combinations.

This connection runs in both directions. During generation, we will study how a weight change alters semantic states and the image's agreement with the prompt. During training, we will study how changes in representation or prediction targets alter the weight updates required to learn them.

### Where the proposed advance lies

There are strong precedents for the individual tools. [UniDiffuser][unidiffuser] jointly generates image and text representations. [SFD][sfd] and [SeFi-Image][sefi] generate semantic information ahead of visual detail. [Local Mechanisms of Compositional Generalization][local] connects sparse dependencies to a specified form of composition. [Vision-Language Binding][binding] traces how reference information passes through text and image tokens. [Concept Sliders][sliders] learns weight changes that control selected visual properties.

Our proposed advance is to **connect a learned division of semantic and visual generation to predictions about the computations and weights that implement it**. We will test whether the dependencies among representation parts explain both the order in which constraints can be resolved and the parameter changes needed to learn new concepts or ways of combining them. Predictions will be tested on combinations withheld from training. Comparison with ordinary low-rank adaptation (LoRA), which learns small weight updates, will measure whether this account improves selectivity and explains interference between changes.

### The hypothesis and the two aims

Consider the prompt: “Show three characters. The character from reference A hands a key to one of the other two: the one nearest the door. The remaining character reads a map.” Satisfying it requires the model to connect several kinds of information: the number of characters, the identity specified by the reference, the relationship that identifies the recipient, and the separate action assigned to the remaining character.

We will ask whether learned representation groups can expose these dependencies. For example, representing the recipient requires connecting the instruction to hand over the key with information about which character is nearest the door. The same computation may apply across different characters and scenes. We will also ask when this information must become reliable during denoising, and whether revising the recipient can preserve the count and the other action. These distinctions will guide our tests; the representation groups and their correspondence to network components will be learned and evaluated.

We will seek *sparse dependence*: recovering a selected semantic or visual state should require only a small subset of the other states. The relevant parts may be groups of tokens or coordinates within tokens. They may connect a phrase, a reference identity, and distant image regions. Recurring dependencies could allow attention to reuse rules for combining information across different scenes.

**Sparse dependence in a representation does not guarantee separately controllable weights.** Shared weights may mix several computations, and later layers may spread or undo a change. The main scientific task is to establish when the proposed correspondence holds and when several components must change together. Graphs can illustrate the information relationships; the evidence will come from predicting and testing the effects of changing features, generation order, and weights.

The two aims are:

1. **Learn representations and generation schedules that expose how complex semantic requirements are resolved.**
2. **Explain how network components combine this information, and predict the effects of changing weights or training targets.**

### Why the connection matters for content creation

A creator may revise which character performs an action, which object a description refers to, or how several instructions should be combined. Our approach will ask which representation changes express that revision and how they affect the remaining generation steps. When a new concept or instruction pattern requires adaptation, we will ask which weight updates make the desired behavior reusable across scenes.

Text-and-reference image generation will provide the common test of context consistency, with separate measurements of reference identity, counts, action assignments, and relationships. Better understanding of component reuse could also guide parameter sharing and adaptation to new visual content. Consistency across video frames or between audio and video provides a longer-term motivation.

> **Figure 1 — Explanatory overview; no new results needed.** Use the running prompt to illustrate connections among a reference identity, an action, the relation identifying its recipient, and the corresponding visual states. Show which information might be resolved together or at different denoising steps. Connect these states to the attention and prediction components hypothesized to use them. Distinguish the effects of weight changes during generation from the weight updates caused by changed training targets. Label the correspondences as hypotheses. Use only text, schematic tokens, and simple drawings.

## 2. Aim 1: Representing and resolving complex prompt semantics

### 2.1 Generating language and visual states

Different training tasks produce representations with different strengths. A visual encoder such as [DINOv2][dino] can describe objects and spatial structure. A language model can represent relations, action roles, and conditions that determine which entity an instruction refers to. Other useful representations may come from image reconstruction, matching text to images, or learning to judge whether an image satisfies a request. We will begin with visual and language features and use targeted training when a required property is missing.

We will extend our existing diffusion Transformer for semantic features and image detail to accept text and reference images. It will generate both a compressed image representation and feature vectors describing the scene. During training, a visual encoder will provide image features, and Qwen will provide internal states for the corresponding scene description. These two feature-producing models will be kept fixed. During generation, the model will generate these feature vectors alongside the image. The original prompt and references will remain fixed, and evaluation will always use those original requirements.

This lets us ask whether generated language-model states help the image generator resolve interacting requirements: counts, assignments of actions and attributes, and references defined through relationships. We will compare three uses of the same representation source: supplying it as a fixed input, having the language model write a scene description, and generating the semantic states through diffusion. The comparison will test the value of allowing semantic states to evolve with the image. Our existing model for visual semantic states provides the starting point for adding language states.

The representations must retain enough information to generate the requested content. They must also let us identify what changes when a selected part is edited. Success on an encoder's original task does not establish either property.

### 2.2 Learning from missing information and controlled changes

We will learn groups of tokens or coordinates that help separate the information needed for different semantic requirements. A group may span several words or image regions. The question is whether these groups support reliable recovery and selective revision of a requirement. Two kinds of training examples will help organize them.

First, we will hide or add noise to selected groups and train the model to recover them from the remaining information. This builds on masked learning across different types of input, as in [MultiMAE][multimae]. We will vary which groups are available and how noisy they are.

Second, we will use examples in which one semantic requirement changes while others are preserved. Examples include changing which character receives an object, which action is assigned to a subject, or which reference a description identifies. Varying these changes across many scenes can help distinguish states that represent an entity from states that determine its role or relationships.

Existing generators can supply candidate examples during the project. [InstructPix2Pix][instruct] demonstrates the practicality of generated editing pairs. We will check that the intended semantic change occurred and that the other stated requirements were preserved. Reassigning an action can require changes in pose or position, so preservation will be evaluated against the instructions rather than exact pixel agreement.

Our initial method will learn rotations and groupings of normalized features from a fixed encoder. Rotations preserve the available information while changing how it is divided among coordinates. Training will encourage a designated edit to change a small group and encourage preserved properties to stay stable. Replacing that group with one from a compatible example will test whether it causes the expected image change. We will compare against the original features and random groupings. If a smaller learned representation is needed, image reconstruction and recovery of the original encoder features will guard against losing required information.

For experiments about genuinely missing inputs, information will be removed before the encoder can copy it elsewhere. Tests that remove already encoded features will answer a separate question about how the generator uses those features.

### 2.3 Testing whether the representation helps

We will measure how accurately a selected group can be recovered as we limit the other groups available to the predictor. Comparisons will use similar model capacity and training budgets. We will also check the generator itself: does removing the apparently unnecessary information preserve its behavior, and does changing a selected group produce the predicted effect?

A small number of groups is useful only if generation quality and fidelity to the request remain high. We will control group size and representation dimension. If one group summarizes the whole scene, we will count its size and the work required to compute it. Likewise, selecting a few tokens after examining every token will not by itself count as a faster computation.

Tests will include combinations withheld from training, such as familiar characters in new action roles or familiar relations combined to identify a subject in a new way. Methods will receive the same examples so that improvements can be attributed to how the representation is learned and used.

Finally, we will ask which semantic requirements need to be resolved together and which benefit from resolving others first. In the running example, does establishing which character is nearest the door help generate the interaction of handing over the key? Can that assignment be revised later without losing the character count or the other action? We will use these dependencies to guide the relative timing of representation groups, comparing with simultaneous generation and a fixed semantic-first schedule. Evaluation will check both the evolving states and agreement of the final image with the original prompt.

**Expected result:** a learned division of semantic and visual states, together with evidence about their dependencies, the order in which they can be resolved, and what remains intact when one requirement is revised.

> **Figure 2 — Explanatory training diagram; no new results needed.** Use schematic tokens to show the two learning signals: recover a hidden semantic or visual group, and identify what changes when one clause assigns an action to a different character. Show how a learned group might be revised at different denoising steps. Label the changed requirement and those to preserve. This is a diagram of the proposed method; include no new samples, measured curves, or accuracy comparisons.

## 3. Aim 2: Explain and predict the roles of network components and weights

### 3.1 How attention uses the representation

We will study three operations:

- **Select information:** query–key interactions in attention determine which tokens to use.
- **Extract features:** value and output projections determine which features are passed onward.
- **Predict the next denoising update:** MLPs and the surrounding network combine the available information to refine the generated state.

[Compositional Attention][compositional] shows that separating attention's search and retrieval operations can improve reuse in its evaluated tasks. We will investigate whether a similar separation explains the repeated computations identified in Aim 1.

The relevant computation may involve several heads or layers. We will test which feature groups they read and modify, then deliberately remove or replace those features. For example, if a state determines which character is the intended recipient, changing it should redirect the action while preserving the other requirements we predict to be independent. Restoring it should recover the corresponding behavior. We will measure the immediate prediction and the final image, since later denoising can preserve, amplify, or overwrite the change.

We will begin by examining the trained model. We will then use the findings to test selective weight sharing or changes to the allowed attention connections. This distinguishes evidence about what the model already does from evidence about how its design could be improved.

### 3.2 Which computations can be reused?

The main experiment will distinguish learning new concepts from learning new ways to combine them. We will first test which new combinations the model can generate without adaptation. Where training is needed, we will compare the following changes:

| What changes? | Question about the network |
|---|---|
| New entities or activities; familiar semantic relationships | Can existing attention still supply the information needed by an adapted predictor? |
| Familiar concepts; new combinations of roles, references, or relational instructions | Does failure arise from how attention combines information, or from how other components use it? |
| Both change | Which components must adapt together? |
| Neither changes | How much drift is introduced by unnecessary adaptation? |

A model may already represent the characters, the door, and the act of handing over a key, yet fail to combine “nearest the door” with the recipient of the action. We will test whether this failure can be corrected by changing how existing information is combined. By comparison, learning a new character may preserve that computation while requiring changes in how the character is represented and generated.

We will develop a prediction procedure on a set of source and adaptation tasks, then fix it before testing new tasks. A small support set will inform each prediction; separate test examples will contain new combinations. We will compare training different groups of weights, including MLPs, value/output projections, query/key projections, and broader combinations. Comparisons will account for the examples used, number of updated weights, and training cost. More detailed tests will target particular heads or feature subspaces.

An important test will learn attention on one data subset and retain it while fitting the remaining prediction components on another. This asks whether the same attention computation can support a new prediction task.

The theoretical starting point is simple: **a new predictor can reuse the attention output only if that output still contains the information it needs.** For squared prediction error, the cost of reuse can be separated into information lost by the fixed attention computation and error in learning the new prediction rule. We will study this separation first in linear models with Gaussian data, where the best predictions can be calculated, then test the resulting criteria in small diffusion Transformers.

Our existing analysis also bounds the effect of removing attention contributions using their weights and feature magnitudes. We will investigate how these errors accumulate across layers and denoising steps. Changing an MLP can change the states seen by later attention, even when attention's weights stay fixed; the prediction must account for this possibility.

One related experiment will test whether several layers can share learned prediction features while using different layer-specific projections. It will measure generation quality against the number of stored parameters. This addresses parameter efficiency within the same study of reusable computation.

### 3.3 How weight changes affect generation and learning

We will study two related questions. Given a weight change, can we predict which scene properties change during generation? Given new examples or a changed prediction target, can we predict how training changes the weights?

A first approximation will use the sensitivity of the network's prediction to each weight. For a fixed input and squared-error loss, the same sensitivity determines how a changed target alters one gradient update. This gives a starting point for identifying the weights associated with a desired change.

We will then examine larger changes, where a single approximation may fail. We will update the approximation along short training runs and model interactions between attention and prediction components. Longer training runs with the model's standard flow-matching objective will provide reference learning paths and well-trained models for comparison. These references specify the behavior we aim to approximate; they are not assumed to be unique optimal weights.

A key test will start with a representation edit that corrects how the model interprets or combines a semantic requirement. We will predict which weights can reproduce that correction, fit an update, and reuse it across prompts and scenes that require the same computation. Ordinary low-rank adaptation and Concept Sliders will provide comparisons using the same supervision and comparable budgets. We will assess the intended change, preservation of other properties, and whether the observed internal changes agree with the prediction. Combining two updates from the same base model will provide a limited test of when their effects interfere.

**Expected result:** an account that predicts which components can be reused, which need to change, and when several changes must be coordinated. The practical test is selective control; the scientific test is whether the account correctly predicts new cases.

> **Figure 3 — Existing component-reuse results only.** Use the already reported CelebA→AAHQ and CelebA→STL-10 results, with paired DINO recovery values 0.815 and 0.143. Draw which weights were restored to the source model and which remained adapted. Reuse existing illustrative images if suitable; the reported values alone are sufficient. Label the experiment as restoring components after joint adaptation. Do not add new runs, uncertainty estimates, direct restricted-training results, or predicted-versus-observed results from the proposed project.

## 4. Why the proposed work is feasible

Existing work supports several of the connections we will study. SFD and SeFi-Image show that generated semantic information can guide image generation. The locality study shows that restricting dependencies can improve composition in a controlled setting and finds suggestive structure in SDXL features. Concept Sliders uses controlled changes in StyleGAN to create paired examples and learn corresponding diffusion-model weight updates. Vision-language binding studies show that interventions can trace how references influence an image. These results support the proposed approach while leaving our central question—how representation structure relates to reusable weights—open. [SFD][sfd], [SeFi-Image][sefi], [Local Mechanisms][local], [Concept Sliders][sliders], [Vision-Language Binding][binding].

Our existing results provide the following evidence:

| Existing work | Result | What it establishes |
|---|---|---|
| ELF-L, unpublished | A separate diffusion model generates projected frozen-Qwen answer states and reaches 61.94% GSM8K accuracy. Qwen encodes the question but does not generate the solution autoregressively at inference. | Diffusion can generate language-model states that retain enough information for substantial reasoning performance. |
| [Learning When to Denoise][schedule] | With a matched 675M-parameter backbone, learned schedules reach AutoGuidance FID 1.05 after 200 epochs, matching an 800-epoch SFD-XL result. | We can train models that generate semantic and visual representations together and control their relative timing. |
| [From Softmax to Score][softmax] and our working mechanistic manuscript | Constructive connections between attention and denoising; analysis of the information needed for prediction and the effect of removing attention contributions. | We have analytical tools for studying what attention computes and when its output can be reused. |
| Parameter sharing and transfer, unpublished | A shared prediction-feature variant improves CelebA64 FID from 17.574 to 14.304 at approximately 10.2M parameters. Restoring source query/key weights after joint adaptation gives recovery scores of 0.815 for CelebA→AAHQ and 0.143 for CelebA→STL-10. | Sharing can improve parameter allocation, while reuse succeeds on some changes and fails on others. |

The sharing result uses one training seed. The transfer metric compares outputs in DINO feature space against the jointly adapted generator. Those experiments restored weights after training; they do not establish that keeping those weights fixed throughout training would work equally well. This distinction motivates the proposed reuse tests.

ELF is useful here because it demonstrates generation of language states that retain reasoning-relevant information. This motivates generating semantic states rich enough to carry a prompt's logical and relational requirements. Whether those states can be divided into useful parts and reliably connected to image generation is part of the proposed work. Our [trajectory optimization research][trajectory] adds experience with controlling diffusion across different representation directions.

Together, the existing evidence supports the ability to generate rich internal representations, coordinate their evolution, and study the effects of changing network components. The project will investigate their connection. If a representation predicts image content well but does not permit selective editing, we will determine whether information is missing, shared weights mix the relevant computations, or later layers undo the change.

> **Figure 4 — Existing representation-generation results only.** Use two panels: (a) a diagram of ELF's frozen-Qwen feature extraction and diffusion generation, with the already reported 61.94% reasoning accuracy; (b) the existing semantic/texture scheduling result, reusing a published plot or displaying the reported 200- versus 800-epoch comparison. Do not request new seeds, runs, decoded examples, or comparisons. Figure 3 already covers the transfer evidence, so it need not be repeated here.

## 5. Proposed creator demonstration and relevance to Sony

During the award period, we will build a demonstration in which a creator supplies reference characters and a prompt specifying their roles, actions, and relationships. The creator can revise one instruction while preserving the other requirements. We will show how that revision changes semantic states and subsequent generation. When the model repeatedly mishandles a class of instructions, we will compare a representation correction for one image with a weight update that applies the correction across new prompts.

Evaluation will measure reference identity, object counts, action and attribute assignments, and stated relationships separately. It will include instructions in which one relationship determines which subject another clause refers to. FID and prompt–image alignment will provide broader quality measures. Blinded human assessment will complement automated measurements on natural images. All methods will use the same examples, and we will report failures as well as successful edits.

A limited extension will add a collection of reference characters through a separate group of parameters. Enabling, replacing, or disabling those parameters will test which generated features depend on the collection and how their use in different roles passes through the model. This could support more explicit control over newly added sources. Removing those parameters would not imply that related information has been erased from the original model.

The demonstration addresses Sony's interests in internal mechanisms, the use of different sources, editing, and controllable contributions. It could inform visual asset development for games, animation, and film. Video and audio provide longer-term motivations.

> **Figure 5 — Conceptual workflow only.** Draw an annotated storyboard using reference labels, schematic characters, and prompt clauses. Show the creator changing an action assignment while preserving character identities, counts, and the other instructions. Illustrate a revision of semantic states and its effect on later generation, then show the proposed reuse of a corresponding weight update across new prompts. Label the figure “Proposed workflow.” Use explanatory drawings only; no new generated images or completed demonstration are needed for submission.

## 6. Work plan and deliverables

All experiments in this work plan are proposed for the 12-month award period.

| Period | Main task | Deliverable |
|---|---|---|
| Months 1–3 | Extend the existing semantic/visual diffusion model and establish controlled scene tests. | Baseline implementation, training examples, and independent measurements of requested and preserved properties. |
| Months 4–6 | Learn representation groups, test their generation order, and analyze how attention uses them. | Results on recovering semantic requirements, revising selected states, and predicting component reuse. |
| Months 7–9 | Test weight updates, learning behavior, and natural-image examples. | Comparisons of which weights are trained, predictions of their effects, and preservation tests on new combinations. |
| Months 10–12 | Complete the creator demonstration and consolidate the explanation. | Reusable editing examples, source-control demonstration, reproducible evaluation materials, and final report. |

The core studies will use small diffusion Transformers and our existing training pipeline. Natural-image validation will reuse a pretrained generator and adapt the components under study. This keeps foundation-model pretraining outside the project requirements and allows the same data and extracted features to support several comparisons.

We will prioritize the two aims and one creator workflow. Generating weight updates with another model, broad meta-learning, and long-video generation remain follow-on directions. The main technical risk is that a clear representation does not correspond to separately controllable weights. Tests at the representation, network, and output levels will distinguish these possibilities.

We will provide three quarterly reports, a final research summary, and the required progress-questionnaire responses. The research outputs will explain which information the model uses, which weights implement the relevant computations, and when a change can preserve the other requirements of a creator's request.

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
| Personnel and associated benefits | Research on representations and weights, model training, and evaluation | [To complete] |
| Computing | Training and adapting models, extracting features, and evaluating generation | [To complete] |
| Other justified direct costs, if applicable | [Specify or remove] | [To complete] |
| Indirect costs | Institutional rate and applicable cost base | [To complete] |
| **Total requested** | **At or below USD 150,000** | **[To complete]** |

## Editorial notes for finalization — remove before submission

- Prepare figures using only explanatory drawings and already reported results. Figures 1, 2, and 5 are diagrams; Figures 3 and 4 use existing evidence. Existing images and plots can be reused, and reported numbers can be redrawn. No additional training, sampling, ablations, evaluation, or completed demonstration is needed before submission.
- Fit the narrative and references within ten pages, with the budget on a separate eleventh page. Shorten prose as figures are laid out; page count has not yet been checked in a submission PDF.
- Complete PI contact details and the institutional budget. The PI CV is a separate submission item.
- Check the planned model choices and computing costs against the budget. This is a planning decision and does not require a new experiment.
- Review the bibliography and complete the authorship/citation form for unpublished preliminary materials. Preserve the qualifications about the existing sharing and transfer results in their figure captions.
- Remove all drafting notes and figure-production instructions from the submission.
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
