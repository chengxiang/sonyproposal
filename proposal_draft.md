# How Representations and Weights Shape Multimodal Generation

**Principal investigator:** Xiang Cheng, Department of Electrical and Computer Engineering, Duke University  
**Focused Research Theme:** Internal Mechanisms of Multimodal Generative Models for Content Creation  
**Project duration:** 12 months  
**PI contact:** [Email to insert]; [Phone with country code to insert]

*Drafting note — remove before submission: This draft uses only existing results and published work as feasibility evidence. The research experiments proposed below are work for the award period. Submission figures will use existing results or explanatory drawings; no new training, sampling, ablation, or evaluation is required.*

## Abstract

We will connect meaningful components of multimodal representations to the network computations that generate them and the denoising stages in which those computations matter. This correspondence will enable creators to revise an action or relationship while preserving other requirements, and to incorporate learned concepts into new scenes. Aim 1 will obtain representations from pretrained models and downstream tasks, learn their semantic divisions, and jointly optimize their asynchronous denoising schedules. Aim 2 will identify and construct network mechanisms that store and combine these components, then use the resulting parameter organization for model merging and updates generated from demonstrations. Meta-learning and compositional training will make these representations and computations reusable. Existing results on diffusion-generated Qwen states, asynchronous generation, and component sharing and replacement provide the foundation. The one-year project will deliver a coupled representation, denoising procedure, and mechanistic account, demonstrated through selective scene revision and the integration of additional character or interaction knowledge.

## 1. The proposed contribution

**We will learn and construct a correspondence between semantic representation components, the network modules that act on them, and their evolution during generation.** Our approach jointly studies these three elements to explain how a generative model realizes a complex instruction and to design more precise control over its consequences.

Consider a creator's request:

> “A woman in a red coat hands a blue cup to a man in a green sweater. A child beside them reads a book.”

The creator may reverse who hands over the cup while retaining the characters, clothing, and child's activity. Our approach will identify the semantic states expressing that change, the computations translating it into the image, and how the affected states should develop together. A second use case will incorporate characters or interactions learned from additional collections and reuse them in new combinations. These connect mechanistic understanding to scene revision and reusable creative knowledge.

### Three linked hypotheses

**1. Multimodal representations can be decomposed into meaningful semantic components.** Let x denote an image or video together with its available language description. An encoder E maps it to N token vectors of dimension d, with projections to a common width where needed:

```math
Z=E(x)=(z_1,\ldots,z_N)^\top\in\mathbb{R}^{N\times d},
\qquad s_k=P_k(Z)\in\mathbb{R}^{d_k},\quad k=1,\ldots,K.
```

Z is the representation; sₖ is a component extracted by a map Pₖ. A component may select token coordinates, group tokens, or combine features across tokens. Components can overlap, and additional semantic tokens can expose information absent from a coordinate selection. A recombination map, together with residual visual features, retains the information needed for image generation. Encoders supply training targets; the diffusion model generates their counterparts from noise, conditioned on the prompt and references.

For the example, components could describe participants, the cup's attributes, the giver–receiver assignment, and the child's activity. We will begin with a small set of participant and interaction components, using downstream tasks to give them semantic meaning and recovery objectives to learn their dependencies. Discrimination, reward or preference prediction, and visual or language understanding provide useful training signals. Meta-learning will further organize components for selective change across examples. The construction can later extend to temporal relationships, such as a switch press causing a toy train to move.

**2. Network specialization can follow these components and their changing information needs.** Let fθ denote the denoising network. A network component is an identifiable computation and its parameters θₘ: an attention head with its query/key/value/output projections, an MLP, or selected rows or columns of a weight matrix.

Our preliminary analyses show how MLPs can store training samples or patches in the studied models, and how attention combines locally relevant information for denoising. We will investigate which semantic components a module reads, which states it influences, and under which noise configurations. Locality means dependence on a small set of relevant components, potentially spanning distant image regions or different modalities.

Sparse representation dependencies need not be implemented by isolated weights. An explicit contribution will therefore be to identify the computations carrying those dependencies and construct feature-access projections and shared prediction components that make their roles reusable. These will support selective changes even when several modules must act together. Training across varied semantic relationships will develop compositional computations; meta-learning will help organize their access and initialization.

**3. Denoising should be designed for the representation and its network interactions.** A conventional diffusion model typically uses a common noise schedule across generated coordinates. For asynchronous flow matching, component k has its own progress schedule τₖ:

```math
s_{k,t}=(1-\tau_k(t))\,\varepsilon_k+\tau_k(t)\,s_k,
\qquad \varepsilon_k\sim\mathcal{N}(0,I),
\qquad \tau_k(0)=0,\quad \tau_k(1)=1.
```

Time t runs from noise to data. Each τₖ increases from zero to one; the curves may cross, and several components can develop together without a fixed ranking. For overlapping components, noise acts on their represented copies, whose clean values must be compatible.

[SFD][sfd], our [Learning When to Denoise][schedule], and [Latent Forcing][latentforcing] demonstrate benefits from coordinating representations; [Diffusion Forcing][diffusionforcing] supports generation under different token noise levels. We will extend these ideas to learned semantic divisions. In the handover example, coordinating the reliability of the role assignment and the interacting visual states can help preserve their agreement during revision.

This relationship also shapes training: changing representations or schedules changes the prediction problems and parameter updates. Changing network computations, in turn, changes which denoising procedures are effective. Our contribution will connect these effects to semantic dependencies and use them to guide generation and knowledge integration.

### Differentiation from existing approaches

**Our advance is to connect a semantic division to the computations and denoising stages that realize it, then use that correspondence to design control and knowledge integration.** Existing methods provide important ingredients; the proposed correspondence determines how those ingredients should work together.

| Existing capability | What this proposal adds |
|---|---|
| Generate semantic features and coordinate representation schedules ([SFD][sfd], [LWD][schedule], [anisotropic trajectory optimization][trajectory]). | Task-grounded divisions with measured conditional recovery dependencies, jointly informing component maps and asynchronous generation. |
| Edit concept directions, trace modality binding, or specialize across timesteps ([Concept Sliders][sliders], [Vision-Language Binding][binding], [DeMe][deme]). | Identify which components a module reads and changes at each noise configuration, then construct selective access to reusable prediction knowledge. |
| Generate personalization weights or merge adapters ([DiffLoRA][difflora], [ZipLoRA][ziplora]). | Use the shared semantic and parameter organization to retain compatible contributions, localize reconciliation, and generate updates for specified sub-representations. |

### The two aims and their practical outcomes

1. **Learn semantic representations and design their denoising process.** Obtain useful token vectors, learn semantic component maps and recovery dependencies, and coordinate their asynchronous generation for precise scene revision.
2. **Understand and design network mechanisms for knowledge storage and composition.** Identify the computations implementing those dependencies and organize them for reuse, model merging, and targeted updates from demonstrations.

The aims will guide each other throughout the project. A module carrying the giver–receiver assignment into visual predictions can guide when Aim 1 generates or revises that component. A useful decomposition from Aim 1 can guide which interactions Aim 2 supports through learned projections, shared knowledge, and training. Both begin with existing representations and generators.

**The distinctive outcome is interpretable control through the correspondence between representations, computation, and denoising.** It will guide both immediate representation edits and parameter changes that make a desired effect reusable. Text-and-reference image generation is the main application; video motivates future temporal and cross-frame extensions. The practical goals are reliable scene revision and the integration of additional creative knowledge without repeatedly training the entire generator.

![Figure 1: fig1 mechanism overview](figures/fig1_mechanism_overview.png)

**Figure 1. Proposed correspondence between representations, network computations, and denoising.** A role reversal is expressed through semantic components; the network's information access and the components' asynchronous evolution determine how it reaches the image. The intended outcome is selective revision with preservation and reusable knowledge. All vectors and schedules are schematic.

## 2. Aim 1: Learn semantic representations and design their denoising process

**This aim will learn meaningful components and coordinate their generation for precise semantic control.** Representation learning determines what information is available; structure learning determines how it is divided and used. We will obtain token vectors Z=E(x), learn component maps sₖ=Pₖ(Z), and jointly train the denoiser and asynchronous schedules. Aim 2 will identify the computations implementing these dependencies and guide their refinement.

### 2.1 Obtain representations and give their components semantic meaning

We will proceed through three approaches of increasing training complexity.

**Start with pretrained activations.** Frozen [DINOv2][dino] features and Qwen hidden states provide immediate visual and language representations. We will generate selected features alongside image latents, beginning with token groups and coordinate projections. Our existing results support generating both kinds of states through diffusion; they provide a starting point for learning finer semantic divisions.

**Learn representations through downstream tasks.** Adversarial discriminators, reward/preference models, and visual or language understanding tasks can expose distinctions needed for generation. [Discriminator feature losses][vaegan] and [ImageReward][imagereward] support using such learned signals. Intermediate activations will provide representation targets; task losses applied to selected components will actively give those components meaning.

A concrete initial construction will use a small set of participant vectors and an interaction vector, extracted by learned projections from image and language tokens, together with residual visual features. For the handover example, image-derived components will predict which participants give and receive the cup; language-derived components will predict the corresponding roles in the description. Each task predictor will initially read only its own modality's components; agreement on correctly paired examples will then connect the two modalities. Varying participants and role assignments will encourage separation of identity from the particular action role. These tasks anchor the semantic division, while Section 2.2 learns how the components depend on one another.

This construction supports revision: changing the interaction component should alter the giver–receiver assignment, while participant and independent-activity information remains available. Other tasks can extend the component vocabulary. For discriminator or reward-derived representations, task-successful examples or explicit success conditioning will specify the desired generation targets. Predictive task losses and reconstruction will remain part of learning; semantic meaning will not be inferred from sparse recovery dependencies alone.

**Use meta-learning to make a semantic contribution reusable.** Begin with a learned projection of frozen features and a small context vector a whose influence is restricted to selected representation components. Each episode supplies a few images of one character and separate examples of that character in other scenes:

| Stage | Operation |
|---|---|
| Adapt to support examples | Update only a through denoising loss, keeping shared encoder and generator weights fixed during these steps. |
| Learn from query examples | Update the shared representation projection through the support-update computation, using query denoising/reconstruction loss to make the adapted vector useful in new scenes. |
| Retain generative information | Train the generator on its shared objective and retain encoder-feature reconstruction and fixed image latents. |

This instantiates [MAML][maml] and context adaptation ideas such as [CAVIA][cavia] with a concrete learning target: features that let a small change capture a character across different compositions. Independent identity and instruction assessment will establish the resulting control. Aim 2 will extend these episodes to selected parameter updates and their shared initialization.

### 2.2 Learn dependencies through recovery at different noise levels

**Structure consists of the component maps and the information one component needs from the others.** We will begin with token or coordinate groups and information-preserving rotations, then refine learned projections while retaining semantic task losses and reconstruction.

Let u collect the K components' denoising progress, with larger values indicating less noise. For component i, define average flow-matching error per coordinate:

```math
L_i(\mathbf{u})=
\frac{1}{d_i}\,
\mathbb{E}\!\left[
\left\|f_{\theta,i}(\widetilde{\mathbf{s}}(\mathbf{u}),\mathbf{u},c)
-(s_i-\varepsilon_i)\right\|^2
\right].
```

Here c is the supplied context, and fθ,i predicts component i's velocity from the noisy collection. The target is velocity per unit of that component's own progress, so slowing its schedule does not trivially reduce the diagnostic.

For distinct components i and j, change only j's noise level:

```math
D_{j\to i}(\mathbf{u})=
L_i(\mathbf{u}_{j\leftarrow u_j^{\mathrm{noisy}}})
-
L_i(\mathbf{u}_{j\leftarrow u_j^{\mathrm{cleaner}}}),
\qquad
u_j^{\mathrm{noisy}}<u_j^{\mathrm{cleaner}}.
```

A positive value means cleaner information in j improves recovery of i. The network, examples, noise draws, context, target noise level, and all other noise levels stay fixed. Repeating this comparison across configurations reveals dependencies that change during generation; varying small groups together captures interactions missed by pairwise comparisons. [Masked recovery][multimae] supplies a related training principle.

We will learn component maps that maintain semantic prediction and reconstruction while recovering each component from a limited subset of others. The measured benefits guide which inputs to retain; omitted components are replaced by independent noise. Normalized feature scales and a fixed total dimension prevent gains through rescaling or adding redundant copies. Overlapping components and residual features count toward that budget.

These dependencies describe information useful to the current generator. Aim 2 will establish which attention heads, projections, and MLP computations carry the measured benefits and construct more selective access where needed. This connects a learned semantic division to the network mechanisms required for its generation.

### 2.3 Jointly learn asynchronous denoising and use it for control

**All K components have asynchronous schedules without an imposed ranking.** Each τₖ progresses from noise to data, but curves may cross and components may develop together. The semantic/texture split in our preliminary work is a special case.

Our [anisotropic trajectory optimization work][trajectory] jointly learns a matrix-valued noise path and score network through a trajectory-level objective. [Learning When to Denoise][schedule] supplies a complementary flow-matching formulation with corrected weighting as schedules change. We will extend these methods to the learned components through three alternating updates:

1. Train the denoiser on scheduled and additional mixed-noise configurations.
2. Update schedules using the trajectory or appropriately weighted flow-matching objective, retaining each component's noise-to-data endpoints.
3. Refine component maps through semantic prediction, recovery, and reconstruction, then update their measured dependencies.

When cleaner information in j reduces i's prediction error, schedule optimization can exploit that interaction while balancing all components' recovery. The key tradeoff is that denoising one component earlier can reduce other components' prediction errors while increasing its own because it has less informative context, linking schedule optimization to the conditional dependencies that make different factorizations of the joint distribution easier or harder to learn.

We will adapt our prior schedule weighting and gradient estimators, including the learned denoiser's dependence on the noise process. Diagnostics at fixed noise configurations and corrected local-noise weighting will distinguish improved information use from merely downweighting difficult prediction problems.

For the handover revision, the learned dependencies and Aim 2's module analysis will guide which semantic and visual components to update together and how their denoising rates should change. A late edit can revisit selected components through forward noising followed by conditional denoising, with training covering the required mixed-noise configurations. This makes asynchronous generation a means of preserving the creator's other requirements while realizing a specific change.

**Expected result:** a task-grounded semantic decomposition, measurable recovery dependencies, and a jointly trained denoising procedure for selective scene revision.

![Figure 2: fig2 recovery schedule](figures/fig2_recovery_schedule.png)

**Figure 2. Recovery dependencies guide asynchronous generation.** Make component j cleaner while holding target i's noise, other inputs, and the network fixed. The change in target recovery loss measures the benefit of information from j. Schedules can cross as relative information needs change. Advancing j can help i while making j's own recovery harder because less context is available. The schedules are illustrative, not measured.

## 3. Aim 2: Understand and design network mechanisms for knowledge storage and composition

**This aim will connect semantic dependencies to network computations that can be reused, combined, and changed selectively.** We will identify which computations connect representation components during denoising, construct more selective access to those computations, and use the resulting organization to incorporate knowledge from collections or demonstrations. The initial setting uses a small number of components, a common base generator, and a shared representation interface.

### 3.1 Identify and organize computations across components and denoising stages

For an attention head, feature projection, or selected MLP weights, we will ask: **which representation components does it read, which does it influence, and under which noise configurations?** With asynchronous generation, a stage is described by the vector u of component noise levels. The same weights may perform different functions as available information changes.

Aim 1's recovery measurements make these questions concrete. If cleaner giver–receiver features improve recovery of the visual exchange, we will identify the computations carrying that benefit. Removing and restoring selected contributions, with controls for general disruption, will establish their role. Value projections, MLPs, residual paths, and later denoising steps can all affect how an attention interaction reaches the output.

Our preliminary analysis provides candidate mechanisms: attention combines information across tokens, while MLPs can contribute reusable predictions and stored knowledge. We will investigate these roles across semantic components without assuming an exclusive division of functions. Timestep specialization has precedents such as [Decouple-Then-Merge][deme]; our objective is to explain and design specialization through changing semantic dependencies.

**We will actively construct parameter organization that supports the identified roles.** Learned feature-access projections will select relevant components and connect them to shared prediction banks: reusable prediction features, potentially stored in MLP weights, accessed across layers through layer- or component-specific projections. This extends sharing beyond the usual reuse of an MLP at different token positions. Access can depend on component reliability, activating a computation when its inputs become informative. Residual computation will retain interactions that require coordinated treatment. [Compositional Attention][compositional] provides a precedent for separating information selection from retrieval.

This organization supports learning from partial representations. If demonstrations supply character-identity features, we can supervise those targets and adapt the associated computations while the base generator supplies new scenes and actions. Different demonstrations can provide different semantic components without depicting every complete combination. The intended benefit is less learning of incidental scene details and fewer required combinations. Module roles will also guide Aim 1's schedules: related computations can indicate which states should develop or be revised together.

### 3.2 Learn reusable computations through diverse semantic dependencies

**Compositional training will vary both the relationships in examples and the information available during recovery.** We will change participants in the handover, reverse their roles, and combine the exchange with independent activities. This varies which concepts interact while providing repeated opportunities to reuse familiar relationships.

Aim 1's components and independently varied noise levels turn these examples into conditional recovery problems. Clear role information can guide a noisy visual exchange; clearer visual information can help recover participant roles. Other configurations require components to develop together. Training shared access projections and prediction components across these problems will encourage computations that remain useful when participants or surrounding activities change.

New character knowledge could then be incorporated while retaining computations for familiar interactions. New relationships may require changes to both knowledge and its combination. For squared prediction error, a tractable theoretical starting point will separate information lost by reusing a fixed attention output from the remaining error in predicting with that output. Linear models with Gaussian data will provide initial conditions for reuse, informing the network construction.

We will extend Aim 1's meta-learning episodes to this parameter organization. A few support examples will update selected parameters; query examples with different compositions will guide the shared initialization and access projections through those updates. This makes adaptability a concrete objective for constructing reusable computations, alongside ordinary compositional training.

### 3.3 Incorporate knowledge through model merging and generated updates

The preceding organization connects parameter changes to their effects on represented concepts and recovery dependencies. Two applications will use the same representation interface and restricted parameter family.

**Integrate knowledge from separately trained collections.** Begin with models or adapters trained from a common base on different datasets, using fixed encoders and component maps and the same allowed parameter groups. One collection might contribute characters and another interactions. The goal is to let those characters participate in the learned interactions while retaining existing generation capabilities.

For each source model, we will identify the representation components and recovery dependencies changed by training. We will combine source updates within these parameter groups, retaining contributions that serve compatible components and fitting mixing coefficients or connecting projections where their effects overlap. New interactions will require focused learning in the computations connecting the contributions. Retained examples or samples from the source generators will provide targets for this focused training.

This reuses learning already performed on individual datasets. For an unprocessed collection, we will train selected knowledge components before integration. The opportunity is to avoid repeatedly fine-tuning the complete generator on the growing union of collections. [ZipLoRA][ziplora] establishes the usefulness of adapter merging; our proposed advantage is a semantic and computational account of which contributions can be combined and where additional learning is needed.

**Generate targeted updates from demonstrations.** Demonstrations will specify desired semantic contributions, and a conditional diffusion model will generate coefficients in a shared update basis associated with the relevant parameter groups. Successful adaptations within this organization will provide initial training targets. Training will also match their effects on represented components and denoising behavior, since equivalent weights can implement the same function. Partial representations can specify the contribution to learn without requiring demonstrations of complete scenes.

[DiffLoRA][difflora] already generates personalization weights, and [Concept Sliders][sliders] provides selective parameter directions. Our approach adds an explicit connection between the requested change, the computations implementing it, and their roles during denoising. This can reduce the update space to predict and help preserve behavior outside the intended change. A generated update will be reused across prompts and image-noise realizations; when demonstrations admit alternatives, sampled updates can represent different reusable interpretations. Generated updates and merged contributions will share the same interface, supporting their combined use.

**Expected result:** an interpretable organization of semantic access and shared prediction computations, with bounded methods for adapting, combining, and generating parameter updates that incorporate new knowledge.

![Figure 3: fig3 component reuse](figures/fig3_component_reuse.png)

**Figure 3. Existing component-reuse evidence.** Images are extracted from the first report-selected noise input (seed 0, latent 1) in Experiment 3, pages 2 and 5. Restoring source query/key weights while retaining adapted value/output projections and MLPs preserves much of the AAHQ change but does not reproduce the STL-10 joint result. Bars report 1 − E_hybrid/E_source, where E is mean squared distance to the paired joint output in normalized DINOv2 features: 1,024 matched latents per adaptation seed, averaged over two seeds. The score measures recovery of joint-model outputs, not target fidelity. These are 20M-model post-training reversion experiments; joint endpoints were selected using the existing test-FID results.

## 4. Why the proposed work is feasible

Existing work supports the connections underlying the proposal. [SFD][sfd] and [SeFi-Image][sefi] show that generated semantic information can guide image generation; [Latent Forcing][latentforcing] and [Diffusion Forcing][diffusionforcing] support varying generation across representations or token noise levels. [Local Mechanisms][local] demonstrates benefits of restricted dependencies for composition in a controlled setting, with suggestive structure in SDXL features. [Vision-Language Binding][binding] shows that interventions can trace how references influence an image.

The constructive methods also have practical foundations. [VAE/GAN][vaegan] uses discriminator features for reconstruction, [REPA][repa] aligns diffusion features with pretrained representations, and [ImageReward][imagereward] supplies learned preference supervision. [MAML][maml] and [CAVIA][cavia] support learning from adaptation episodes. [DiffLoRA][difflora], [Concept Sliders][sliders], and [ZipLoRA][ziplora] establish useful parameter generation, editing, and merging operations. The proposed advance connects these operations to learned semantic divisions, conditional recovery, and the computations carrying their effects through denoising.

Our existing results provide the following evidence:

| Existing work | Result | What it establishes |
|---|---|---|
| ELF-L, unpublished | A separate diffusion model generates projected frozen-Qwen answer states and reaches 61.94% GSM8K accuracy. Qwen encodes the question but does not generate the solution autoregressively at inference. | Diffusion can generate language-model states that retain enough information for substantial reasoning performance. |
| [Learning When to Denoise][schedule] | With a matched 675M-parameter backbone, learned schedules reach AutoGuidance FID 1.05 after 200 epochs, matching an 800-epoch SFD-XL result. | We can train models that generate semantic and visual representations together and control their relative timing. |
| [From Softmax to Score][softmax] and our working mechanistic manuscript | Constructive connections between attention and denoising; analysis of the information needed for prediction and the effect of removing attention contributions. | We have analytical tools for studying what attention computes and when its output can be reused. |
| Parameter sharing and transfer, unpublished | A shared prediction-feature variant improves CelebA64 FID from 17.574 to 14.304 at approximately 10.2M parameters. Restoring source query/key weights after joint adaptation gives recovery scores of 0.815 for CelebA→AAHQ and 0.143 for CelebA→STL-10. | Sharing can improve parameter allocation, while reuse succeeds on some changes and fails on others. |

The sharing result uses one training seed. The transfer metric compares outputs in DINO feature space against the jointly adapted generator. Those experiments restored weights after training; they do not establish that keeping those weights fixed throughout training would work equally well. This distinction motivates the proposed reuse tests.

ELF's reasoning accuracy is a functional demonstration that diffusion-generated Qwen states retain precise language information. This motivates semantic targets capable of carrying complex prompt requirements. Our [anisotropic trajectory optimization research][trajectory] provides joint network/schedule learning and estimators accounting for a changing noise process; our asynchronous flow work provides another direct algorithmic starting point.

These results establish separate foundations for representation generation, schedule design, and component reuse. The proposed research connects them through task-grounded component maps and module-specific information access. Existing encoders and generators let both aims begin immediately, while improvements to their organization can be incorporated progressively.

![Figure 4: fig4 existing feasibility](figures/fig4_existing_feasibility.png)

**Figure 4. Existing results support representation generation, schedule design, and parameter sharing.** (A) ELF generates projected frozen-Qwen contextual states; decoded answers reach 54.17% GSM8K accuracy with synchronous inference and 61.94% with asynchronous inference from the same checkpoint (EMA 0.9999, 32 ODE steps, two inference seeds over the same 1,319 questions). This supports reasoning-relevant information in generated language states; no matched AR advantage is claimed. (B) [LWD][schedule] reaches AutoGuidance FID 1.05 at 200 epochs, matching SFD-XL at 800 epochs with a 675M backbone. (C) Experiment 1, page 6, reports FID-50k 17.574 versus 14.304 for 10,215,472 versus 10,239,536 parameters on CelebA64, using one training seed, epoch-400 EMA, and a 50-step sampler. Values are redrawn from documented results; no new evaluations are included.

## 5. Creator demonstrations and relevance to Sony

**Reliable scene revision.** A creator supplies character references and a description of the handover scene, then reverses who gives the cup while retaining identities, clothing, and the child's activity. The demonstration will expose the relevant semantic change and show how identified network computations and asynchronous denoising carry it into the image. Learning from selected representation components will also let demonstrations supply identity or interaction information without specifying every complete scene.

**Expand a reusable collection.** Starting with two source models or adapters trained from the same base, the creator will incorporate character knowledge from one collection and interaction knowledge from another, then request a new scene using both. Focused reconciliation will address the computations connecting the contributions. A compact update generated from demonstrations will provide a complementary way to add or correct a contribution and reuse it across prompts.

These workflows target fewer unintended changes during revision and less repeated training as a collection grows. They could support visual asset development for games, animation, and film. They also address [Sony's stated interests][sony] in modality binding, generation dynamics, causal interventions, and externalized knowledge. We will retain the identity of newly added source modules so their contributions can be enabled, replaced, and traced. Disabling an added module does not imply removal of related knowledge from the pretrained base.

Validation will connect internal effects to the creator's requirements: reference identity, participant roles, attributes, and independent activities will be assessed separately, alongside generation quality. Independent evaluators and human assessment will complement training signals. Source integration will track preservation and the compute used for learning, merging, and inference. Comparisons will isolate the benefit of the proposed correspondence at comparable budgets. Prompt-grounded assessment such as [GenEval][geneval] provides a starting point, extended to action roles.

Existing generators can supply original/revised prompt pairs, following the practical precedent of [InstructPix2Pix][instruct]. Shared initial noise can aid comparison, but pairs will be checked for the intended change and preserved requirements before use. These are proposed award-period resources and demonstrations.


## 6. Work plan and deliverables

**The core deliverable is a working correspondence between semantic components, their denoising process, and the network computations enabling selective control.** Task-derived features, meta-learning, and compositional training are methods for constructing it. Partial-representation adaptation, model merging, and generated updates will use the same representation interface and restricted parameter family.

### Concrete execution

We will begin with our existing 10–20M-parameter diffusion Transformers and semantic/visual diffusion pipeline, reusing checkpoints, component-replacement code, and schedule-learning implementations. For the text-to-image prototype, we will use the released **PixArt-Σ 512-pixel model**, with an approximately 0.6B-parameter denoising Transformer and available training/adaptation code. Its native T5 text conditioning and VAE will remain in place. [PixArt-Σ][pixart]

Image latents will be augmented with frozen DINOv2 features and Qwen3-4B-Instruct-2507 states from the representation pipeline used in ELF. We will initially learn four to eight component groups for participants, relations, and remaining visual information. Small projections, a semantic denoising branch, and trainable attention interfaces will allow these states to evolve with image latents; compact parameter updates will adapt the pretrained image generator. Caching encoder outputs and retaining frozen text/image encoders will concentrate training on the component maps, denoising branch, and selected network parameters.

| Resource | Role in the initial implementation |
|---|---|
| Existing CelebA64, AAHQ, and STL-10 experiment pipelines | Establish module reuse and shared prediction behavior in small models before applying the same analyses to the text-to-image prototype. |
| [SWiG grounded situation recognition][swig] | Activity labels, participant-role labels, and entity locations for training participant and interaction components. |
| [Visual Genome][visualgenome] | Objects, attributes, and pairwise relationships for compositional training and semantic prediction. |
| [CelebA identity annotations][celeba] | Support/query episodes for facial identity across photographs, with identities separated between training and evaluation. |

We will select a bounded vocabulary of annotated activities and relations and form descriptions from their labels. Some participant–relation combinations will be held out to assess composition. SWiG and Visual Genome provide scene supervision; CelebA supplies the separate identity task. We will obtain the identity annotations for the existing image collection as needed. Independently checked generated scene variations can extend demonstrations during the award; success on facial identity alone will not stand in for control of full scenes.

Knowledge integration will start with two adaptations trained during the award from the same generator and fixed representation interface, using different annotated subsets. Their shared allowed parameter groups will also define the output of the context-to-update model. Training will fit source updates first, then learn mixing coefficients or connecting projections for integration. This shared implementation makes merging and generated updates applications of the same mechanisms. The small-model work builds on our existing mechanistic experiments; funded staffing and computing allocations will be specified in the budget.

All experiments below are proposed for the 12-month award period.

| Period | Main work | Concrete output |
|---|---|---|
| Months 1–3 | Begin both aims with existing image/language features; learn participant and interaction projections using semantic tasks; measure recovery dependencies and identify candidate network computations. | Initial component maps, recovery measurements, and module assignments for the shared scene workflow. |
| Months 4–6 | Jointly refine component maps and schedules; learn selective feature access and shared prediction components, using compositional training and compact meta-learning episodes. | An asynchronous denoising procedure and network organization supporting selective scene revision. |
| Months 7–9 | Use the shared organization for partial-representation adaptation, integration of two source models/adapters, and generation of compact updates from demonstrations. | Bounded merging and update-generation methods that add character or interaction knowledge through identified parameter groups. |
| Months 10–12 | Consolidate both creator workflows on natural images and document how the identified computations support preservation and reuse. | Demonstrations, implementation, mechanistic findings, reproducible evaluation materials, and final report. |

The core studies will use small diffusion Transformers, our existing training pipeline, and pretrained generators. Learned projections, a small number of component groups, and restricted updates provide tractable initial constructions; the generator, extracted features, adaptation episodes, and parameter basis will be shared across applications. Initial module analysis uses fixed representations, so it can guide Aim 1 before a new decomposition is learned.

The first merging setting uses a common base and shared representation interface, avoiding a requirement to align unrelated architectures. Generated updates begin in the same restricted parameter family, using successful adaptations as training targets. Where useful dependencies span several computations, the methods will operate on those groups and focus reconciliation there. This retains a concrete route to control and integration without requiring a perfectly separated network.

Text-and-reference image generation defines the award-period application. Larger collections, broader architectural changes, and temporal/audio extensions provide future directions. We will provide three quarterly reports, a final research summary, and the required progress-questionnaire responses.

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
23. Ma et al. [Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning][deme]. CVPR, 2025.
24. Shah et al. [ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs][ziplora]. ECCV, 2024.
25. Chen et al. [PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation][pixart]. Implementation and released checkpoints.
26. Pratt et al. [Grounded Situation Recognition (SWiG)][swig]. ECCV, 2020; project and annotations.
27. Krishna et al. [Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations][visualgenome]. IJCV, 2017.
28. Liu et al. [Deep Learning Face Attributes in the Wild (CelebA)][celeba]. ICCV, 2015; dataset and annotations.

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

- Figures 1 and 2 are explanatory diagrams; Figures 3 and 4 use existing results. Figure sources, selected image panels, and reproduction instructions are documented in [figures/README.md](figures/README.md). No additional experiments are required before submission.
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

[deme]: https://arxiv.org/abs/2410.06664
[ziplora]: https://arxiv.org/abs/2311.13600

[pixart]: https://github.com/PixArt-alpha/PixArt-sigma
[swig]: https://prior.allenai.org/projects/gsr
[visualgenome]: https://arxiv.org/abs/1602.07332
[celeba]: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
