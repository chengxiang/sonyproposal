# How Representations and Weights Shape Multimodal Generation

**Principal investigator:** Xiang Cheng, Department of Electrical and Computer Engineering, Duke University  
**Focused Research Theme:** Internal Mechanisms of Multimodal Generative Models for Content Creation  
**Project duration:** 12 months  
**PI contact:** [Email to insert]; [Phone with country code to insert]

*Drafting note — remove before submission: This draft uses only existing results and published work as feasibility evidence. The research experiments proposed below are work for the award period. Submission figures will use existing results or explanatory drawings; no new training, sampling, ablation, or evaluation is required.*

## Abstract

We will develop controllable and adaptable multimodal generators through a joint study of representations, Transformer modules, and the denoising process. Images, videos, and associated language can be represented as collections of token vectors. Aim 1 will jointly learn semantic components within or across these vectors and design their denoising procedures, using features from discriminators, reward models, language models, and other tasks. Aim 2 will explain and shape how attention and MLPs store knowledge and compose these components. Its findings will guide representation and schedule design; the resulting semantic dependencies will guide network design and training. The goal is precise control of objects, attributes, actions, and relationships while preserving other requirements of a creator's request. Meta-learning and training on diverse compositions will support reusable representations and computations; context-generated weight updates will make desired changes transferable across prompts. Existing results on diffusion-generated Qwen states, asynchronous semantic/visual generation, and component sharing and replacement provide a foundation. The project will deliver coupled representation and denoising methods, network design principles, and selective parameter updates, with training efficiency and generation quality as additional objectives.

## 1. The proposed contribution

**Our approach jointly studies multimodal representations, the generative process, and the network modules that connect them, to enable precise control and efficient adaptation of image generation.** The representation determines which semantic information the model can express; attention and MLPs determine how that information is stored and combined; the denoising process determines when it becomes available. Designing these together gives us several ways to realize a creator's intent: change selected semantic vectors, change when they are generated, or change the weights that make their effects reusable.

### Three linked hypotheses

**1. Multimodal representations can be decomposed into meaningful semantic components.** Let x denote an image or video together with its available language description. An encoder E converts this example into N token vectors of dimension d, after any projections needed to give the tokens a common width:

```math
Z=E(x)=(z_1,\ldots,z_N)^\top\in\mathbb{R}^{N\times d},
\qquad z_i\in\mathbb{R}^{d}.
```

Here E is the representation function and Z is the represented example. Tokens may describe image regions, video frames, or language semantics. During training, encoders provide these target vectors; during generation, the diffusion model produces their counterparts from noise, conditioned on the creator's prompt and any supplied references.

A *semantic component* is a vector extracted from this representation. It can be a subset of coordinates within tokens, a group of tokens, or the output of a learned map spanning several tokens:

```math
s_k=P_k(Z)\in\mathbb{R}^{d_k},
\qquad k=1,\ldots,K.
```

The maps Pₖ may overlap: an object identity can be relevant to several relationships. We can also add semantic tokens to Z when a useful concept is not readily exposed by coordinate selection. A learned recombination map will combine the components, together with residual features where needed, to recover a representation suitable for image generation. This permits semantic decomposition while retaining visual information.

Compositional benchmarks such as [GenEval][geneval] motivate explicit treatment of objects, attributes, and relationships. For a creator-facing illustration, consider:

> “A woman in a red coat hands a blue cup to a man in a green sweater. A child beside them reads a book.”

Candidate components could represent the woman's identity, the association between the cup and its color, the assignment of giver and receiver, or the child's separate activity. These components can span both language and image tokens. The same principle extends to temporal semantics in video, such as “a child presses a switch, causing a toy train to start moving.”

We will learn these components from tasks that make the relevant distinctions useful: discrimination, reward or preference prediction, language modeling, and visual or multimodal understanding. Meta-learning will further organize the representation so that a limited change to selected vectors or weights produces a desired effect across new examples. The aim is a representation in which complex instructions can be expressed and revised through identifiable semantic components.

**2. Transformer modules specialize in computations that store and compose these components.** Write fθ for the denoising network, with θ its learned weights. A *network component* means an identifiable computation and its parameters: an attention head and its query/key/value/output projections, an MLP layer, or a selected set of rows or columns of a weight matrix. We use θⱼ to denote the parameters selected for such a component.

Our preliminary studies show how MLPs can store training samples or patches in the analyzed models, and how attention can combine locally relevant information for denoising and composition. This motivates a finer hypothesis: **module specialization can follow semantic components within and across tokens.** An attention head or part of an MLP may help maintain agreement between language and visual features for a relation such as “a fork resting on a spoon.” Its role can involve selecting relevant coordinates, transmitting features, or transforming them into a consistent visual prediction.

Locality here means dependence on a small set of relevant semantic or visual components; those components can occupy distant image regions or different modalities. We will identify these computations and use them to design reusable attention mechanisms and shared prediction banks. Training attention on diverse compositional rules will encourage reuse across new concepts, while meta-learning will help organize representations, parameter sharing, and initialization for selective adaptation.

**3. The denoising procedure should follow the chosen representation and its network interactions.** A conventional diffusion model typically applies a common noise schedule across its generated coordinates. Asynchronous denoising allows each selected component to follow its own schedule. For a simple flow-matching construction, a training state for clean component sₖ is:

```math
s_{k,t}=(1-\tau_k(t))\,\varepsilon_k+\tau_k(t)\,s_k,
\qquad \varepsilon_k\sim\mathcal{N}(0,I),
\qquad \tau_k(0)=0,\quad \tau_k(1)=1.
```

Here t runs from noise at t=0 to the clean target at t=1, and τₖ(t) specifies component k's progress. Using the same τₖ for every component gives a shared schedule; different choices let selected components advance earlier or be refined together. Shared noise levels need not produce equal semantic progress. When components overlap, the model jointly learns their compatible clean values; the noise variables act on their represented copies.

The representation choice therefore determines which generation schedules are available. [SFD][sfd], our [Learning When to Denoise][schedule], and [Latent Forcing][latentforcing] demonstrate benefits from coordinating complementary representations. [Diffusion Forcing][diffusionforcing] further shows how different token noise levels support flexible generation and guidance. We will extend this principle to learned semantic components and use the computations identified above to design their denoising procedures.

For the example, making the giver–receiver assignment available earlier can guide the network's generation of the exchange. Changing that assignment may require the relevant image states to remain revisable while the child's activity is retained. Knowledge of which modules use, transmit, or overwrite the assignment will guide when to advance these states and which to revise together. The network will be trained for the combinations of noise levels used by the chosen procedure.

This interaction also shapes learning. Changing the representation or its noise schedules changes the prediction problems presented to the network and hence its parameter updates. Conversely, changing an attention or MLP component changes how semantic information is used, which can call for a different denoising procedure. We will use this two-way relationship to develop generators whose semantic and parameter changes are easier to explain and control.

### What the joint approach enables

**The proposed advance is a coordinated design of semantic representations, generative procedures, and network modules.** Joint representation generation and asynchronous scheduling provide established starting points; our contribution is to connect them to the mechanisms that store, compose, and modify semantic content. This supports three practical outcomes:

- **Precise generation and editing:** express the requested change through selected semantic components and use the denoising procedure to carry it into the image while preserving other requirements.
- **Reusable adaptation:** identify parameter changes that implement the desired behavior across prompts, then infer compact updates from context or demonstrations.
- **Efficient model construction:** share reusable prediction knowledge and compositional computations, and allocate training and denoising effort according to their dependencies.

This organization makes control more interpretable: each intervention is tied to a semantic component, the computations that act on it, and its evolution during generation. These connections will guide both representation edits and compact parameter updates, including LoRA, and help manage interference when separately learned changes are combined.

### The two aims

1. **Learn semantic representations and design their denoising process.** Jointly learn token vectors, semantic component maps, and generation procedures that enable precise semantic control. Task learning and meta-learning will organize the representation around how it is generated and revised.
2. **Understand and design network mechanisms for knowledge storage and composition.** Establish how attention and prediction components implement semantic interactions, then use that understanding to design reusable computations, denoising procedures, and selective parameter updates.

**The aims will guide each other throughout the project.** If Aim 2 identifies a computation that carries the giver–receiver relationship into visual predictions, Aim 1 can use it to decide when to generate or revise the relevant semantic component. Conversely, a useful decomposition from Aim 1 can identify interactions that the network should support through its projections, attention, or training. Both aims will begin with existing representations and generators, then refine their designs through this feedback.

Text-and-reference image generation will provide the main application, with video motivating richer temporal and cross-frame semantics. The intended result is a creator who can change an object, action, or relationship with precise control over its consequences, and reuse learned concepts across new scenes. Existing results on rich semantic states, asynchronous denoising, and component reuse provide the foundation for this program.

> **Figure 1 — Explanatory overview; no new results needed.** Use the woman/man/child prompt to connect three elements: token vectors Z and semantic components sₖ; attention heads, MLP weight slices, and shared prediction features; and separate denoising schedules τₖ. Illustrate reversing who hands over the cup while preserving identities, clothing, and the child's activity. Show module findings guiding component maps and schedules, and semantic dependencies guiding network design. Distinguish established preliminary findings from the proposed finer semantic specialization. Use simple drawings, vector blocks, and schematic schedules only.

## 2. Aim 1: Learn semantic representations and design their denoising process

**This aim will deliver a semantic representation together with a denoising procedure for precise generation and editing.** We will learn the encoder E and component maps Pₖ, or build on pretrained features, while designing the schedules τₖ that generate and revise those components. Representation and schedule choices will inform each other, guided by the network mechanisms established in Aim 2.

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

**A good representation and denoising procedure should together make a limited change produce the intended effect while preserving other requirements.** We will use this criterion to learn component maps and schedules, as well as to evaluate them.

In each training episode, a small support set will specify a new concept, contextual rule, or desired correction. A few update steps will change selected representation variables or a restricted set of generator parameters, followed by denoising under the selected procedure. Separate query examples will assess the change in new compositions and the preservation of other requirements. Across episodes, we will update the representation or its projection and a compact family of schedules to improve this behavior. [MAML][maml] and methods for adapting compact context variables such as [CAVIA][cavia] provide starting points.

This supplies a concrete objective for joint representation and process design: quality after limited adaptation, preservation, and reuse across examples. We will control representation size, update and denoising budgets, and evaluate entirely held-out tasks. Aim 2 will use the same episodes to learn model initialization, feature-access projections, and parameter-sharing structure.

Adaptation speed alone will not establish a mechanism. We will examine whether the learned representation makes dependencies easier to identify, whether successful changes use the predicted components, and when several components must change together. The learned and fixed representations will undergo the same tests.

### 2.4 Use semantic dependencies to organize generation and revision

**The decomposition and the generation process will be designed together.** We will measure component recovery as we vary both the available information and its noise levels, and use these dependencies to revise the maps Pₖ and schedules τₖ. Aim 2 will identify the computations that support or limit these choices. Useful sparsity must preserve semantic information and image quality; comparisons will control group size, total dimension, and the cost of shared summaries or input selection.

Building on our asynchronous diffusion work, we will assign separate noise levels to selected token or coordinate groups. Their schedules determine which states advance earlier, remain uncertain longer, or develop together. We will use recovery tests and the network interventions in Aim 2 to choose candidate schedules. For the running prompt, does making the giver–receiver assignment reliable earlier help generate the exchange of the cup? If it does, which computations transmit that information, and when can the assignment still be reversed?

Selective revision provides the main test. After reversing who hands over the cup, we will allow the implicated semantic and visual states to update while retaining states predicted to support the other requirements. Initially, we will vary denoising rates and pause selected groups. As an extension for late edits, we will apply a specified forward-noising step to affected groups, then denoise them conditioned on the retained states. Preservation will be judged by identities, counts, and the separate action, allowing visual changes required by the revised instruction.

The model will receive each group's noise level and train on the combinations required by these processes. Pausing, conditioning on retained states, and restarting recovery must be supported by this training. Low noise alone does not establish that a semantic decision is correct; we will measure whether the intended requirement has actually been resolved.

Controlled comparisons will first hold the representation and trained network fixed: simultaneous denoising, a fixed semantic-first schedule, asynchronous schedules tuned directly for either quality or control, and schedules informed by measured dependencies. We will then learn the decomposition and schedule together, initially by alternating their updates, with network findings from Aim 2 guiding each revision. Comparisons will match computation and schedule-selection budgets. Separate training comparisons will establish how scheduling changes convergence and the computations learned, distinguishing inference-process effects from changes due to training.

We will predict successful edit timing and the states requiring revision from limited diagnostic examples, then test new prompts and compositions. Edit fidelity, persistence, and preservation are the primary outcomes; training cost and generation quality are additional outcomes. Independent evaluation will use the original or explicitly revised request. A model supplying semantic features will not be the sole judge of success.

**Expected result:** semantic token representations and component maps paired with denoising procedures for selective, persistent revisions, together with an explanation of the dependencies that make them effective.

> **Figure 2 — Explanatory learning diagram; no new results needed.** Show task-trained activations, masked recovery, and support/query feedback updating both component maps and denoising schedules. Include feedback from Aim 2's module analysis. Illustrate several semantic and visual groups advancing at different rates and selected groups resuming after an edit. Label all schedules as proposed, with no measured curves or new samples.

## 3. Aim 2: Understand and design network mechanisms for knowledge storage and composition

**This aim will establish how network components implement semantic interactions and use that knowledge to guide architecture, denoising, and parameter updates.** We will begin with pretrained models and fixed representations, and incorporate Aim 1's learned components and schedules as they develop. The resulting account will guide both network construction and revisions to the representation and generation process.

### 3.1 Stored knowledge and compositional computation

By stored knowledge we mean information acquired through training and reused across prompts: concept features, associations, and prediction rules. We will investigate how this knowledge is represented in MLPs and shared prediction banks, and how attention accesses and combines it with the current semantic and visual states.

Our mechanistic analysis motivates three interacting operations:

- **Select information:** query–key interactions determine which tokens attention uses.
- **Pass features:** value and output projections determine which information is transmitted.
- **Predict a denoising update:** MLPs and surrounding components transform the available information into a refined state.

[Compositional Attention][compositional] provides a precedent for separating search and retrieval. We will test the roles of these operations through feature and parameter replacement, removal, and restoration, measuring immediate changes and the final image. For example, we will ask which components provide knowledge of an activity and which connect its participants to a relation specified in the prompt. Information may be distributed across components; the experiments will establish the roles of MLPs alongside attention and other network weights.

At different noise levels, we will replace a candidate semantic feature, interrupt its passage through selected attention or MLP components, and test restoration. This will identify when a computation uses the feature and how its influence persists through denoising. The results will guide Aim 1's component maps and schedules. When a useful semantic interaction is poorly supported, we will use these findings to guide projections and sharing choices in Section 3.3. Attention weights alone will not identify the mechanism: value projections, MLPs, residual paths, and subsequent steps can alter its effect.

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

Aim 1's support/query episodes will also help construct the parameter organization under study. Here the learned objects will be model initializations and feature-access projections, with comparisons of sharing structures and allowed update locations. The objective is a small update that realizes the requested change on new examples while preserving other requirements under the chosen denoising procedure.

We will use separate changes to representation, schedule, and parameter organization to identify their contributions, then refine them together. A network design that makes a semantic component easier to use will feed back into Aim 1; the dependencies exposed there will guide which projections or sharing structures to learn here.

Mechanistic tests will compare which information each component uses, which weights respond during adaptation, and whether the predicted effects persist. This can reveal how meta-learning changes the model's organization, including cases where fast adaptation remains distributed and difficult to interpret.

### 3.4 Predict parameter effects and training responses

Given a weight change and generation process, we will predict the effects on evolving states and the final image. Given a changed representation, target, dataset, or training noise schedule, we will predict the learning response. These are related processes, not inverse maps.

A first approximation will use the sensitivity of a prediction to each weight. For a fixed input and squared-error loss, this same sensitivity determines how a changed target alters one gradient update. We will then update the approximation along short training runs and account for interactions between components. Changing an MLP, for example, can change the states seen by later attention even if attention's weights remain fixed. We will test whether such component changes alter the useful generation order or the point at which an edit is preserved or overwritten. Predictions of final effects must account for the remaining denoising steps.

Extensive flow training on development tasks will provide reference trajectories and well-trained endpoints. They specify behaviors to approximate, without assuming unique optimal weights. We will test predictions on held-out tasks using limited diagnostic examples. Varying Aim 1's source tasks, component maps, and noise schedules will reveal how information available during training shapes parameter responses. These findings will guide representation and schedule choices that support selective learning.

### 3.5 Generate weight updates from context or demonstrations

**Can context specify a reusable parameter change whose effects we can explain?** We will train a model to generate compact weight deltas from contextual instructions or a few demonstrations. This asks whether the learning process studied above can be approximated with less data and computation.

The update will initially be expressed through coefficients in a shared basis or selected parameter groups. The attention and knowledge analyses will inform their locations and structure. Reference adaptations will supply initial training targets; training and evaluation will also compare denoising behavior, semantic effects, and preservation. Equivalent weights can implement the same function, so coefficient matching alone is insufficient.

[DiffLoRA][difflora] establishes a precedent for generating personalization updates from reference images. We will compare a direct context-to-update model with one using the representation and parameter organization developed here, alongside ordinary LoRA and Concept Sliders under comparable budgets. A conditional diffusion model over updates will be compared with deterministic prediction. When demonstrations admit several interpretations, we will test whether sampled updates express useful, distinct behaviors; one sampled update will be reused across multiple prompts with independently varied image noise.

The goal is to turn a semantic correction realized through Aim 1's representation and denoising procedure into a reusable parameter update. We will evaluate its effects across new compositions and establish how the update acts through the identified network computations. Combining two updates from the same base will provide a bounded test of interference and model merging.

This study can begin with ordinary representations, parameter bases, and reference training. Its progress does not require every proposed representation or architecture to succeed first.

**Expected result:** a mechanistic account of knowledge storage, composition, and learning that guides reusable network designs and selective parameter updates, and supplies Aim 1 with concrete guidance on which semantic components to generate or revise together.

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
| Months 1–3 | Begin both aims with shared semantic tasks, existing representations and generators, and reference adaptation procedures. | Baseline representations and schedules; initial analyses of module roles and independent evaluation. |
| Months 4–6 | Jointly refine component maps and schedules using module analysis; apply meta-learning to representations, processes, and model structure. | Semantic control methods and mechanistic guidance connecting representation, denoising, and network design. |
| Months 7–9 | Train for diverse compositions; develop knowledge sharing/replacement and context-generated updates, feeding findings back into representations and schedules. | Reusable network designs, selective updates, and comparisons of their effects on generation and learning. |
| Months 10–12 | Consolidate the coupled methods on natural images and complete the creator demonstration. | Mechanistic findings and their limits, reusable editing examples, reproducible evaluation materials, and final report. |

The core studies will use small diffusion Transformers, our existing training pipeline, and pretrained generators, sharing tasks, extracted features, and adaptation episodes. Both aims can start immediately: task-derived representations and schedules can be developed with an existing generator, module roles with fixed representations, and generated updates with ordinary parameter bases. Throughout the project, module findings will guide representation and process design, while semantic dependencies and adaptation needs will guide network design and training.

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
22. Ghosh, Hajishirzi, and Schmidt. [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment][geneval]. arXiv:2310.11513, 2023.

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

[geneval]: https://arxiv.org/abs/2310.11513
