# How Representations and Weights Shape Multimodal Generation

**Principal investigator:** Xiang Cheng, Department of Electrical and Computer Engineering, Duke University  
**Focused Research Theme:** Internal Mechanisms of Multimodal Generative Models for Content Creation  
**Project duration:** 12 months  
**PI contact:** [Email to insert]; [Phone with country code to insert]

*Drafting note — remove before submission: This draft uses only existing results and published work as feasibility evidence. The research experiments proposed below are work for the award period. Submission figures will use existing results or explanatory drawings; no new training, sampling, ablation, or evaluation is required.*

## Abstract

We propose to understand how semantic representations, network weights, and the generation process jointly determine multimodal content. Aim 1 will obtain representations from discrimination, reward learning, and other tasks, and use selective adaptation to help define useful semantic structure. We will study which parts should become reliable earlier, develop together, or be revisited after an edit. Aim 2 will explain how attention and prediction components implement these dependencies, and how changing weights or the generation process alters their effects. Training on diverse compositions, meta-learning, and context-generated weight updates will help investigate and shape this organization. The central test is whether the mechanistic account predicts how to realize a requested semantic change while preserving other requirements across new prompts. Training speed and generation quality provide additional measures. Existing results on generating Qwen states, coordinating semantic and visual denoising, and sharing or replacing components support the ingredients. Two coupled aims will test whether a better representation and understanding of network computations allow more precise control over both generation trajectories and reusable parameter changes.

## 1. The proposed contribution

**We will study how representations, network weights, and the generation process interact, and use this understanding to predict and selectively change generated content.** A useful representation may divide a complex request into information that can be resolved separately or through a few recurring interactions. We will investigate how training creates such a division, which network computations implement it, and how it should guide the relative progress of different states during denoising.

This connection runs in both directions. During generation, we will study how weights and the chosen denoising process determine the evolution of semantic and visual states. During training, we will study how representation choices, prediction targets, and noise schedules alter the weights learned. A schedule changes which information is available to each network computation; changing that computation can in turn change which schedule is useful. Precise control and rapid adaptation will test the usefulness of this explanation.

### Where the proposed advance lies

The constituent tools have strong precedents. [UniDiffuser][unidiffuser] jointly generates image and text representations. [SFD][sfd], our [Learning When to Denoise][schedule], and [Latent Forcing][latentforcing] show the value of coordinating denoising across complementary representations. [Diffusion Forcing][diffusionforcing] trains with independent token noise levels and supports flexible sequence generation and guidance. [Local Mechanisms of Compositional Generalization][local] studies sparse dependencies, while [Vision-Language Binding][binding] traces reference influence. These works establish generation order as an important design choice. Our question is how learned conceptual dependencies and the network computations implementing them can predict a process suited to a specified semantic change.

Our proposed advance is to **connect learned representations and internal computations to predictions about both the generation process and parameter changes**. We will test whether task and adaptation objectives expose useful dependencies, whether diverse composition training produces reusable attention computations, and whether these findings predict when to update selected states or weights. The main outcome is faithful control of a requested change and preservation of other requirements; training efficiency and FID provide additional evidence. The predictions must hold on new contexts and compositions. Comparison with ordinary low-rank adaptation (LoRA), which learns compact parameter updates, and [Concept Sliders][sliders] will test the value of the mechanistic account.

### The hypothesis and the two aims

Consider the prompt: “Show three characters. The character from reference A hands a key to one of the other two: the one nearest the door. The remaining character reads a map.” The model must connect a reference identity, an action, the relationship identifying its recipient, and a separate action assigned to the remaining character.

We will learn and test representation groups that might expose these dependencies, including groups of tokens or coordinates within tokens. A model trained to detect incorrect action assignments could supply relevant features; a general language model could supply a different representation of the same requirements. We will ask which information each group needs, when it becomes reliable during denoising, and what remains intact when it is revised.

We will seek *sparse dependence*: recovery of a selected semantic or visual state should require only a small subset of the other states. These dependencies may connect language concepts, reference identities, and distant image regions. Their recurrence across scenes could allow the network to reuse computations while changing the knowledge those computations use.

**Sparse dependence in a representation does not guarantee separately controllable weights or a unique generation order.** Shared weights may mix computations, and later layers may spread or undo a change. Our hypothesis is that making a semantic requirement reliable before dependent states are resolved can help guide their generation; after an edit, revisiting the affected states may help the correction persist. Dependencies can vary with context and noise level, and mutually dependent states may need joint refinement. Graphs will illustrate these relationships; interventions will test the predicted computations, ordering, and parameter changes.

The two aims are:

1. **Learn representations that expose useful semantic structure.** Use task learning, recovery, and selective adaptation to study which states should be resolved together, earlier, or again after a revision.
2. **Understand and shape how network components store, compose, and modify knowledge.** Explain how internal computations interact with this generation process, train for reusable composition, and infer parameter changes from context.

### Why the connection matters for content creation

A creator may revise which entity an instruction refers to, how concepts are combined, or which reference supplies a requested feature. We will ask which representation changes express that revision and how they affect later generation. When learning is required, we will ask which parameter updates make the change reusable.

Text-and-reference generation will provide a common test of context consistency, including identity, counts, action assignments, and relationships. The character example illustrates the conceptual questions; controlled object compositions and reference-based creation provide additional instances. Parameter sharing, rapid adaptation, and combining separately learned changes will test consequences of the same account.

> **Figure 1 — Explanatory overview; no new results needed.** Show the interaction among representation groups, network components, and the generation process. Use the running prompt to illustrate a semantic requirement becoming reliable before dependent visual states, and a revision that requires those states to change. Connect task/meta-learning to the representation and composition training to attention and stored knowledge. Mark the correspondences as hypotheses. Use text, schematic tokens, and simple drawings only.

## 2. Aim 1: Learn representations that expose useful semantic structure

### 2.1 Obtain semantic spaces from tasks

**The task used to train a network can help determine which distinctions its representation makes accessible.** A model trained to detect whether the wrong character receives an object may develop useful features for action assignment. A general language or visual model could capture the same distinction through broader training. We will investigate both routes.

| Source | Training signal | Reason to investigate its activations |
|---|---|---|
| Adversarial discriminator | Distinguish training examples from generated examples, optionally conditioned on text and references | Its features may expose errors in realism or consistency encountered during generation |
| Reward or preference model | Predict which outputs satisfy a task or are preferred | Its features may expose the distinctions underlying a successful or unsuccessful generation |
| Other task-trained or self-supervised models | Language prediction, visual self-supervision, reconstruction, multimodal matching, or understanding tasks | General training may supply semantic information without a task-specific evaluator |

[Discriminator features][vaegan] have been used as learned reconstruction metrics, and [ImageReward][imagereward] demonstrates learning image-generation preferences and using the resulting score for training. We will investigate the intermediate activations as candidate semantic states. The source task is a design choice: reward prediction is one option alongside general visual representations such as [DINOv2][dino] and language-model states from Qwen.

We will compare selected pretrained features with features obtained through targeted training during the project. Controlled examples that satisfy or violate a chosen requirement can train an evaluator to distinguish the errors we wish to study. Comparisons between objectives will hold architecture, data, and compute comparable where practical. A strong source-task score motivates a representation; its usefulness for generation and selective intervention must be tested separately.

We will extend our existing semantic/visual diffusion model to generate selected activations alongside compressed image representations (image latents). A fixed source network will initially extract feature targets from training examples and their context. For evaluator-derived representations, examples satisfying the request, or explicit conditioning on task success, will define the desired distribution. At inference these states will be generated from context and noise. The original prompt and references remain fixed.

The same feature source will be compared as an auxiliary training target and as an explicit generated state. For evaluator sources we will also compare scalar reward supervision; for Qwen states, fixed prompt features and an autoregressively generated scene description provide additional comparisons. This asks what is gained by allowing semantic representations themselves to evolve with the image.

### 2.2 Learn structure through recovery and controlled changes

We will learn groups of tokens or coordinates that help separate the information needed for different semantic requirements. A group may span several words or image regions.

First, we will hide or add noise to groups and train the model to recover them from the remaining information, building on masked learning across modalities such as [MultiMAE][multimae]. We will vary which groups are available and their noise levels. For genuinely missing-input tests, removal occurs before encoding; removing an already encoded state tests how the generator uses it.

Second, we will use examples in which one requirement changes while others remain satisfied: a different entity receives an object, an action is reassigned, or a description identifies a different reference. Existing generators can supply candidate examples during the project, following the use of generated editing pairs in [InstructPix2Pix][instruct]. We will verify the intended change and preserved requirements. A changed action may require a changed pose, so preservation will be judged against the instructions.

Rotations and groupings of fixed features provide a controlled starting point. We will also learn projections or encoders using the source tasks and adaptation objective below. Reconstruction and retained source-task information will constrain loss of necessary content. Replacing a learned group with one from a compatible example will test whether the expected change appears in the generated image.

### 2.3 Use meta-learning to define and construct useful representations

**A good representation should help a limited change produce the intended effect while preserving other requirements.** We will use this criterion both to evaluate representations and to learn them.

In each training episode, a small support set will specify a new concept, contextual rule, or desired correction. A few update steps will change selected representation variables or a restricted set of generator parameters. Separate query examples will test whether the resulting model applies that change in new compositions and preserves the other requirements. Across episodes, we will update the representation or its projection to improve this behavior. [MAML][maml] and methods for adapting compact context variables such as [CAVIA][cavia] provide starting points.

This supplies a concrete objective for representation learning: quality after limited adaptation, preservation, and reuse across examples. We will control representation size and update budgets, and test entirely held-out tasks. In Aim 2, the same episode design will investigate model initialization and parameter-sharing structure.

Adaptation speed alone will not establish a mechanism. We will examine whether the learned representation makes dependencies easier to identify, whether successful changes use the predicted components, and when several components must change together. The learned and fixed representations will undergo the same tests.

### 2.4 Use semantic dependencies to organize generation and revision

**A useful decomposition should help us choose how generation proceeds.** We will measure recovery accuracy as we restrict the other groups available to a predictor, then test those restrictions in the generator. Useful sparsity must preserve semantic information and image quality. Comparisons will control group size, total dimension, and the cost of computing shared summaries or selecting inputs.

Building on our asynchronous diffusion work, we will assign separate noise levels to selected token or coordinate groups. Their schedules determine which states advance earlier, remain uncertain longer, or develop together. We will use recovery tests and the network interventions in Aim 2 to choose candidate schedules. For the running prompt, does making the recipient assignment reliable earlier help the model generate the interaction? If it does, which computations transmit that information, and when is it still possible to correct the assignment?

Selective revision provides the main test. After changing the recipient, we will allow the implicated semantic and visual states to update while retaining states predicted to support the other requirements. Initially, we will vary denoising rates and pause selected groups. As an extension for late edits, we will apply a specified forward-noising step to affected groups, then denoise them conditioned on the retained states. Preservation will be judged by identities, counts, and the separate action, allowing visual changes required by the revised instruction.

The model will receive each group's noise level and train on the combinations required by these processes. Pausing, conditioning on retained states, and restarting recovery must be supported by this training. Low noise alone does not establish that a semantic decision is correct; we will measure whether the intended requirement has actually been resolved.

First, we will hold the representation and trained network fixed and compare simultaneous denoising, a fixed semantic-first schedule, asynchronous schedules tuned directly for either quality or control, and schedules informed by measured dependencies. Comparisons will match computation and schedule-selection budgets. Next, separate training comparisons will test how scheduling changes convergence and the computations learned. This separates inference-process effects from changes due to training.

We will predict successful edit timing and the states requiring revision from limited diagnostic examples, then test new prompts and compositions. Edit fidelity, persistence, and preservation are the primary outcomes; training cost and generation quality are additional outcomes. Independent evaluation will use the original or explicitly revised request. A model supplying semantic features will not be the sole judge of success.
**Expected result:** an account of how task and adaptation objectives shape representations, and whether their dependencies predict a generation process that makes semantic revisions more selective and persistent.

> **Figure 2 — Explanatory learning diagram; no new results needed.** Show task-trained activations, masked recovery, and support/query feedback for learning the representation. Add a schematic schedule for several semantic and visual groups: one advances earlier, dependent groups follow, and selected groups resume after an edit. Label all schedules as proposed, with no measured curves or new samples.

## 3. Aim 2: Understand and shape how network components store, compose, and modify knowledge

### 3.1 Stored knowledge and compositional computation

By stored knowledge we mean information acquired through training and reused across prompts: concept features, associations, and prediction rules. We will investigate how this knowledge is represented in MLPs and shared prediction banks, and how attention accesses and combines it with the current semantic and visual states.

Our mechanistic analysis motivates three interacting operations:

- **Select information:** query–key interactions determine which tokens attention uses.
- **Pass features:** value and output projections determine which information is transmitted.
- **Predict a denoising update:** MLPs and surrounding components transform the available information into a refined state.

[Compositional Attention][compositional] provides a precedent for separating search and retrieval. We will test the roles of these operations through feature and parameter replacement, removal, and restoration, measuring immediate changes and the final image. For example, we will ask which components provide knowledge of an activity and which connect its participants to a relation specified in the prompt. Information may be distributed across components; the experiments will establish the roles of MLPs alongside attention and other network weights.

At different noise levels, we will replace a candidate semantic feature, interrupt its passage through selected attention or MLP components, and test restoration. This asks when a computation uses the feature and whether its influence persists through later denoising. The results will inform the schedules in Aim 1. Attention weights alone will not identify the mechanism: value projections, MLPs, residual paths, and subsequent steps can alter its effect.

We will also study **shared prediction banks**: learned features accessed by several layers through layer-specific projections. Which information can be shared, and which access rules must remain distinct? We will compare sharing and replacement with separate layer parameters, measuring quality against the number of stored parameters. Sharing across depth extends beyond the ordinary sharing of MLP weights across token positions and does not by itself imply faster computation.

### 3.2 Train attention on diverse compositions and test reuse

We will actively test whether exposure to diverse composition rules produces more reusable attention computations. Controlled training sets will vary how concepts and relations are combined while keeping the concept inventory, sample count, and compute comparable. We will then measure transfer to new concepts or requirements from limited examples.

The scientific question is whether broader composition exposure improves the information selected and passed by attention, allowing other learned knowledge to change with less disruption. Better adaptation alone will not establish this explanation; component interventions must identify the computations responsible.

We will first test new combinations without adaptation. Where training is required, the main comparison will separate changes in concepts from changes in their composition:

| What changes? | Question about the network |
|---|---|
| New entities or activities; familiar relationships | Can existing attention supply the information needed by adapted prediction components? |
| Familiar concepts; new combinations of roles, references, or instructions | Must information selection or combination change while other knowledge remains useful? |
| Both change | Which components must adapt together? |
| Neither changes | How much drift does unnecessary adaptation introduce? |

From limited diagnostic examples, we will predict which heads, feature groups, or parameter components can be retained. Separate tasks and compositions will test those predictions through direct restricted training and component replacement. Comparisons will match examples, updated parameter counts, and compute. One test will retain attention learned on one data subset while fitting prediction components on another.

For squared prediction error, our theoretical starting point separates the information lost by reusing a fixed attention output from the error in learning a new predictor from that output. We will study this first in linear models with Gaussian data, then test the resulting criteria in small diffusion Transformers. Existing bounds on removed attention contributions provide a starting point for examining how errors accumulate across layers and denoising steps.

### 3.3 Learn model structure and initialization for selective change

The meta-learning procedure in Aim 1 will also help construct the parameter organization under study. Across support/query episodes, we will learn initializations and feature-access projections, and compare alternative sharing structures or allowed update locations. The criterion is whether a small update achieves the requested change on new examples while preserving other requirements.

We will vary representation choice and parameter organization separately before testing their interaction. This asks whether a representation that is easy to edit also makes a particular architecture easy to adapt, and whether an architecture can make otherwise entangled features more usable.

Mechanistic tests will compare which information each component uses, which weights respond during adaptation, and whether the predicted effects persist. This can reveal how meta-learning changes the model's organization, including cases where fast adaptation remains distributed and difficult to interpret.

### 3.4 Predict parameter effects and training responses

Given a weight change and generation process, we will predict the effects on evolving states and the final image. Given a changed representation, target, dataset, or training noise schedule, we will predict the learning response. These are related processes, not inverse maps.

A first approximation will use the sensitivity of a prediction to each weight. For a fixed input and squared-error loss, this same sensitivity determines how a changed target alters one gradient update. We will then update the approximation along short training runs and account for interactions between components. Changing an MLP, for example, can change the states seen by later attention even if attention's weights remain fixed. We will test whether such component changes alter the useful generation order or the point at which an edit is preserved or overwritten. Predictions of final effects must account for the remaining denoising steps.

Extensive flow training on development tasks will provide reference trajectories and well-trained endpoints. They specify behaviors to approximate, without assuming unique optimal weights. We will test predictions on held-out tasks using limited diagnostic examples. Changing the source task or meta-learning objective from Aim 1 will test whether representation construction makes these parameter responses more localized or predictable.

### 3.5 Generate weight updates from context or demonstrations

**Can context specify a reusable parameter change whose effects we can explain?** We will train a model to generate compact weight deltas from contextual instructions or a few demonstrations. This asks whether the learning process studied above can be approximated with less data and computation.

The update will initially be expressed through coefficients in a shared basis or selected parameter groups. The attention and knowledge analyses will inform their locations and structure. Reference adaptations will supply initial training targets; training and evaluation will also compare denoising behavior, semantic effects, and preservation. Equivalent weights can implement the same function, so coefficient matching alone is insufficient.

[DiffLoRA][difflora] establishes a precedent for generating personalization updates from reference images. We will compare a direct context-to-update model with one using the representation and parameter organization developed here, alongside ordinary LoRA and Concept Sliders under comparable budgets. A conditional diffusion model over updates will be compared with deterministic prediction. When demonstrations admit several interpretations, we will test whether sampled updates express useful, distinct behaviors; one sampled update will be reused across multiple prompts with independently varied image noise.

The central test starts from a semantic correction that works through a representation intervention. Can the generated weight update reproduce that correction across new compositions, and do its internal effects agree with our prediction? Combining two updates from the same base will provide a bounded test of interference and model merging.

This study can begin with ordinary representations, parameter bases, and reference training. Its progress does not require every proposed representation or architecture to succeed first.

**Expected result:** an account of where reusable knowledge is stored, how attention composes it across denoising, and how training shapes that organization. It should predict how changes to the generation process or parameters affect a requested revision.

> **Figure 3 — Existing component-reuse results only.** Use the already reported CelebA→AAHQ and CelebA→STL-10 results, with paired DINO recovery values 0.815 and 0.143. Draw which weights were restored to the source model and which remained adapted. Reuse existing illustrative images if suitable; the reported values alone are sufficient. Label the experiment as restoring components after joint adaptation. Do not add new runs, uncertainty estimates, direct restricted-training results, or predicted-versus-observed results from the proposed project.

## 4. Why the proposed work is feasible

Existing work supports several of the connections we will study. SFD and SeFi-Image show that generated semantic information can guide image generation. The locality study shows that restricting dependencies can improve composition in a controlled setting and finds suggestive structure in SDXL features. Concept Sliders uses controlled changes in StyleGAN to create paired examples and learn corresponding diffusion-model weight updates. Vision-language binding studies show that interventions can trace how references influence an image. These results support the proposed approach while leaving our central question—how representation structure relates to reusable weights—open. [SFD][sfd], [SeFi-Image][sefi], [Local Mechanisms][local], [Concept Sliders][sliders], [Vision-Language Binding][binding].

[Latent Forcing][latentforcing] directly analyzes the ordering of latent and pixel generation, while [Diffusion Forcing][diffusionforcing] demonstrates flexible sampling and guidance using different noise levels across tokens. Together with our scheduling results, these support varying the generation process as a practical research method. They do not establish the proposed correspondence between complex semantic dependencies, internal computations, and selective revision.

Existing work also supports the constructive methods. [VAE/GAN][vaegan] uses discriminator features in a reconstruction objective, while [REPA][repa] aligns diffusion-network features with pretrained visual representations. [ImageReward][imagereward] provides a learned preference signal for generation. [MAML][maml] and [CAVIA][cavia] supply methods for learning from adaptation episodes, and [DiffLoRA][difflora] demonstrates conditional generation of parameter updates. These establish practical ingredients; the proposed connection between representation choice, parameter roles, and predictable learning remains to be tested.

Our existing results provide the following evidence:

| Existing work | Result | What it establishes |
|---|---|---|
| ELF-L, unpublished | A separate diffusion model generates projected frozen-Qwen answer states and reaches 61.94% GSM8K accuracy. Qwen encodes the question but does not generate the solution autoregressively at inference. | Diffusion can generate language-model states that retain enough information for substantial reasoning performance. |
| [Learning When to Denoise][schedule] | With a matched 675M-parameter backbone, learned schedules reach AutoGuidance FID 1.05 after 200 epochs, matching an 800-epoch SFD-XL result. | We can train models that generate semantic and visual representations together and control their relative timing. |
| [From Softmax to Score][softmax] and our working mechanistic manuscript | Constructive connections between attention and denoising; analysis of the information needed for prediction and the effect of removing attention contributions. | We have analytical tools for studying what attention computes and when its output can be reused. |
| Parameter sharing and transfer, unpublished | A shared prediction-feature variant improves CelebA64 FID from 17.574 to 14.304 at approximately 10.2M parameters. Restoring source query/key weights after joint adaptation gives recovery scores of 0.815 for CelebA→AAHQ and 0.143 for CelebA→STL-10. | Sharing can improve parameter allocation, while reuse succeeds on some changes and fails on others. |

The sharing result uses one training seed. The transfer metric compares outputs in DINO feature space against the jointly adapted generator. Those experiments restored weights after training; they do not establish that keeping those weights fixed throughout training would work equally well. This distinction motivates the proposed reuse tests.

ELF is useful here because it demonstrates generation of language states that retain reasoning-relevant information. This motivates generating semantic states rich enough to carry a prompt's logical and relational requirements. Whether those states can be divided into useful parts, generated in an effective order, and reliably connected to image generation is part of the proposed work. Our [trajectory optimization research][trajectory] adds experience with controlling diffusion across different representation directions.

Together, these results support generating rich internal states, coordinating their evolution, and studying reusable network components. They give the five approaches independent starting points. Their synthesis will test whether representations obtained through task learning or meta-learning make attention, stored knowledge, and parameter updates easier to explain. If a representation supports prediction but not selective change, we will examine missing information, shared computations, and the effects of later denoising.

> **Figure 4 — Existing representation-generation results only.** Use two panels: (a) a diagram of ELF's frozen-Qwen feature extraction and diffusion generation, with the already reported 61.94% reasoning accuracy; (b) the existing semantic/texture scheduling result, reusing a published plot or displaying the reported 200- versus 800-epoch comparison. Explain that ordering already improves training and quality; predicting an order for selective control is proposed work. Do not request new seeds, runs, decoded examples, or comparisons. Figure 3 already covers the transfer evidence, so it need not be repeated here.

## 5. Proposed creator demonstration and relevance to Sony

During the award period, we will build a demonstration in which a creator supplies reference characters or objects and a prompt specifying roles, actions, and relationships. The creator can revise one instruction while preserving the others. We will show how the revision changes semantic states and which subsequent updates realize it. The demonstration will compare advancing, retaining, or revisiting selected states under the proposed generation process. For a new concept or recurring instruction error, a few demonstrations will specify a reusable correction. We will compare a representation intervention, a fitted parameter update, and a context-generated update across new prompts.

Evaluation will measure reference identity, counts, action and attribute assignments, and stated relationships separately, including instructions where one relationship identifies the subject of another clause. FID and prompt–image alignment will provide broader quality measures. Independent checks and blinded human assessment will complement any learned evaluator used in training. Comparisons will account for training examples, denoising and adaptation steps, parameter count, and the offline cost of selecting schedules, meta-learning, or generating updates. We will report failures alongside successful edits.

A limited extension will add a collection of reference characters through a separate group of parameters. Enabling, replacing, or disabling those parameters will test which generated features depend on the collection and how their use in different roles passes through the model. This could support more explicit control over newly added sources. Removing those parameters would not imply that related information has been erased from the original model.

The demonstration addresses Sony's interests in internal mechanisms, the use of different sources, editing, and controllable contributions. It could inform visual asset development for games, animation, and film. Video and audio provide longer-term motivations.

> **Figure 5 — Conceptual workflow only.** Use reference labels, schematic entities, and prompt clauses to show a revised requirement and the requirements to preserve. Illustrate a representation correction and selective revision of dependent states, followed by contextual demonstrations producing a compact weight update reused across new prompts. Label this “Proposed workflow.” Use explanatory drawings only; no new generated images or completed demonstration is needed for submission.

## 6. Work plan and deliverables

All experiments in this work plan are proposed for the 12-month award period.

| Period | Main task | Deliverable |
|---|---|---|
| Months 1–3 | Establish shared semantic tasks, pretrained and task-trained representation candidates, and reference adaptation procedures. | Baselines for generation, controlled changes, and independent evaluation; initial feature and parameter analyses. |
| Months 4–6 | Learn representation groups; test generation and revision schedules, plus meta-learning of representations and model structure. | Predictions and tests of edit timing and preservation; comparisons of scheduling effects during inference and training. |
| Months 7–9 | Test composition-diverse training, knowledge sharing/replacement, and context-conditioned weight generation. | Tests of component reuse and its effect on generation order; update generators compared with reference learning and direct adaptation. |
| Months 10–12 | Test the interaction of the approaches on natural images and complete the creator demonstration. | Mechanistic findings and their limits, reusable editing examples, reproducible evaluation materials, and final report. |

The core studies will use small diffusion Transformers, our existing training pipeline, and pretrained generators. They will share tasks, extracted features, and adaptation episodes. Each approach also has an independent baseline: task-derived representations can be studied with an existing generator, attention and memory with fixed representations, and generated updates with ordinary parameter bases. Their combination will test whether a learned representation or model structure improves the predictions and interventions of the other aim.

The meta-learning studies will begin with compact representations, initializations, and feature-access or sharing choices; update generation will begin in a restricted parameter family. These provide tractable implementations of all five approaches within the two aims. Adaptation quality alone will not establish the proposed mechanism: improvements must be connected to changes in information use, stored knowledge, or parameter responses.

We will provide three quarterly reports, a final research summary, and the required progress-questionnaire responses. The outputs will explain what information the model uses, how training organizes it, and when a change can preserve the other requirements of a creator's request.

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

14. Larsen et al. [Autoencoding beyond pixels using a learned similarity metric][vaegan]. arXiv:1512.09300, 2015.
15. Xu et al. [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation][imagereward]. arXiv:2304.05977, 2023.
16. Yu et al. [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think][repa]. ICLR, 2025.
17. Finn, Abbeel, and Levine. [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks][maml]. ICML, 2017.
18. Zintgraf et al. [Fast Context Adaptation via Meta-Learning][cavia]. ICML, 2019.
19. Wu et al. [DiffLoRA: Generating Personalized Low-Rank Adaptation Weights with Diffusion][difflora]. arXiv:2408.06740, 2024.
20. Chen et al. [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion][diffusionforcing]. arXiv:2407.01392, 2024.
21. Baade et al. [Latent Forcing: Reordering the Diffusion Trajectory for Pixel-Space Image Generation][latentforcing]. arXiv:2602.11401, 2026.

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

[vaegan]: https://arxiv.org/abs/1512.09300
[imagereward]: https://arxiv.org/abs/2304.05977
[repa]: https://arxiv.org/abs/2410.06940
[maml]: https://arxiv.org/abs/1703.03400
[cavia]: https://arxiv.org/abs/1810.03642
[difflora]: https://arxiv.org/abs/2408.06740

[diffusionforcing]: https://arxiv.org/abs/2407.01392
[latentforcing]: https://arxiv.org/abs/2602.11401
