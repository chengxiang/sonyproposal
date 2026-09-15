# How Representations and Weights Shape Multimodal Generation

**Principal investigator:** Xiang Cheng, Department of Electrical and Computer Engineering, Duke University  
**Focused Research Theme:** Internal Mechanisms of Multimodal Generative Models for Content Creation  
**Project duration:** 12 months  
**PI contact:** [Email to insert]; [Phone with country code to insert]

*Drafting note — remove before submission: This draft uses only existing results and published work as feasibility evidence. The research experiments proposed below are work for the award period. Submission figures will use existing results or explanatory drawings; no new training, sampling, ablation, or evaluation is required.*

## Abstract

We will jointly design multimodal representations, denoising procedures, and the transformer computations that connect them. The goal is to generate complex scenes and revise a specified action or relationship while preserving other requirements. First, we will obtain representations from pretrained models and downstream tasks and learn component extractors. Second, component-wise denoising losses will reveal which information helps predict each component. Third, we will jointly optimize asynchronous schedules and the transformer, including which modules operate at different component noise levels. Fourth, we will train shared and specialized computations across conditional denoising tasks and varied semantic compositions. Module interventions will explain which computations carry a change and when they remain reusable. Existing results on diffusion-generated Qwen states, learned schedules, and transformer component sharing and replacement provide the foundation. The one-year project will deliver a working generation and revision method together with a mechanistic account of its representation dependencies and transformer computations.

## 1. The proposed contribution

**We will jointly learn representation components, their denoising process, and the transformer computations used to predict them.** Understanding which information a computation uses, when it is needed, and when it can be shared will provide interpretable ways to control complex generation and revision.

Consider a creator's request: “A woman in a red coat hands a blue cup to a man in a green sweater. A child beside them reads a book.” Reversing who gives the cup requires changing the interaction while retaining the clothing and the child's activity. We will learn which representation changes express such an edit, identify the transformer computations that transmit it, and design the denoising procedure that realizes it. Participants and relationships are illustrative meanings; the learned decomposition need not assign one component to each human-named concept.

### Three linked hypotheses

**1. Useful representations admit learnable component extractors.** An encoder E maps an image or video, together with its available description, to N token vectors of dimension d. Flattening these vectors gives z; an extraction function P_k selects or computes component k:

```math
Z=E(x)\in\mathbb R^{N\times d},\qquad z=\operatorname{vec}(Z)\in\mathbb R^D,
\qquad s_k=P_k(z)\in\mathbb R^{d_k},\quad D=Nd.
```

For one token, an extractor simply maps d coordinates to d_k coordinates. A low-dimensional linear projection is the simplest example; an MLP or attention-based extractor can learn more complex features and combine information across tokens. We use P_k for this general extraction function. The initial construction uses disjoint groups of projected coordinates or tokens, together with a reconstruction map and residual visual features. This retains information needed for generation while allowing the divisions to be learned. Overlapping nonlinear components provide a later extension.

We will obtain representations from pretrained models and downstream tasks, then learn their component extractors. Encoders supply clean training targets; the diffusion model generates corresponding states from noise. Our aim is a decomposition useful for generation and editing, including complex prompt semantics, without prescribing an exhaustive list of semantic factors.

**2. Transformer modules can specialize in how they access and change these components.** A *module* is an identifiable transformer computation together with its parameters: an attention head, including its query/key/value/output projections; an MLP block; or a specified group of rows or columns in these projections. Denote its parameters by θₘ. This definition supports interventions on existing modules and the shared or specialized constructions in Sections 2.3–2.4.

Our preliminary studies show how MLPs can store training samples or patches in the studied models, and how attention combines information during denoising. We will identify which representation components supply useful inputs to a module, which predictions change when its contribution is removed or replaced, and how these roles vary with noise levels. **A particular focus is attention's use of a local subset of information: locality means dependence on a small set of representation components, which may span distant tokens, image regions, or modalities.** Sparse dependence alone does not imply isolated weights; we will construct and analyze the corresponding transformer computations.

**3. Denoising schedules should be learned with the representation and transformer.** In asynchronous flow matching, component k follows its own progress schedule:

```math
s_{k,t}=(1-\tau_k(t;\phi))\varepsilon_k+\tau_k(t;\phi)s_k,
\qquad \varepsilon_k\sim\mathcal N(0,I),\qquad
\tau_k(0;\phi)=0,\quad\tau_k(1;\phi)=1.
```

The parameters φ determine monotone schedules from noise to data. Their curves may cross; there is no required first/second ordering. The transformer can likewise vary which modules are active as different components become informative.

**Our work provides a practical basis for learning these schedules.** [Learning When to Denoise (LWD)][schedule] reaches unguided FID 4.93 after 120,000 main-training updates, versus the reported [REPA][repa] result of 5.84 after 4 million updates, using 675M-parameter models: 33 times fewer main-training updates, plus LWD's 10,000-update schedule-learning probe. These update counts compare the complete methods; they are not measured wall-clock speedups. Our [variational trajectory optimization][trajectory] and LWD jointly learn the noise process and score/flow network. Benefits from coordinating generation across representations or tokens have also been reported in [SFD][sfd], [Latent Forcing][latentforcing], and [Diffusion Forcing][diffusionforcing].

The interaction with training is explicit. With extractor parameters ψ and schedule parameters φ, a training step is

```math
\theta^+=\theta-\eta\nabla_\theta\mathcal J(\theta,\phi;\psi),
\qquad s_k=P_{k,\psi}(\operatorname{vec}(E(x))).
```

Changing ψ changes the prediction targets; changing φ changes the noisy inputs and their weighting. Both therefore change the parameter update. Sections 2.1–2.3 learn the targets, dependencies, and schedules; Section 2.4 organizes transformer computations around the resulting conditional denoising tasks.

### Differentiation from existing approaches

| Existing capability | What this proposal adds |
|---|---|
| Semantic feature generation and coordinated schedules ([SFD][sfd], our [LWD][schedule] and [trajectory optimization][trajectory]). | Extend our joint schedule/denoiser learning to learned component divisions, with transformer module activity governed by the components' noise levels. |
| Sparse score dependencies and feature-space composition ([Local Mechanisms][local]). | Learn component divisions through task supervision and conditional denoising, then use their measured dependencies to organize both generation schedules and transformer information access. |
| Modality-binding interventions, timestep specialization, and separated attention operations ([Vision-Language Binding][binding], [DeMe][deme], [Compositional Attention][compositional]). | Identify which computations can be shared across conditional denoising tasks and which must specialize as conditions, targets, and noise configurations change. Use those findings to support selective revision. |

### One research program in four connected steps

1. **Learn representations and component extractors** from pretrained activations and downstream tasks.
2. **Learn denoising dependencies** by measuring which components help predict one another.
3. **Jointly optimize schedules and transformer activity** according to the information available at each noise configuration.
4. **Train shared and specialized computations** across conditional denoising tasks and diverse compositions.

The same component vectors, losses, and transformer implement all four steps. Measured dependencies guide schedules and attention access; restricting or sharing a computation changes its prediction errors and therefore informs the next schedule update. Selective generation and revision of text-and-reference images provide the common application.

![Figure 1: fig1 mechanism overview](figures/fig1_mechanism_overview.png)

**Figure 1. Learning a representation, transformer, and denoising procedure together.** The creator requests a reversal of who gives the cup. We learn an edit to the component vectors that realizes this change, while preserving the clothing and the child's activity. Participants and relations illustrate possible information in the vectors; the actual division is learned and can be more complex. The diagram describes the proposed method, not an already demonstrated semantic assignment to modules.

## 2. Research approach: representations, denoising, and transformer computations

**Representation learning determines what the vectors encode; dependency learning determines what information is useful for denoising each component.** Downstream tasks supply the first signal. Component-wise flow-matching errors supply the second. We will alternate these steps with schedule learning while retaining reconstruction of the generation target. The resulting conditional prediction tasks determine the module activity and parameter sharing studied in Sections 2.3–2.4.

### 2.1 Obtain representations and learn component extractors

**Begin with pretrained activations.** Frozen [DINOv2][dino] features and Qwen states provide visual and language targets. We will generate selected features alongside image latents, initially using token groups and disjoint groups of learned projected coordinates. A small MLP or attention extractor is a later refinement. Keeping the total feature dimension fixed and retaining residual visual information prevents apparent gains from simply adding capacity or discarding difficult content.

**Use downstream tasks to improve the targets.** Discriminators, reward/preference models, and visual or language understanding tasks can reveal distinctions that a generation model needs. [Discriminator feature losses][vaegan], [ImageReward][imagereward], and [representation alignment][repa] provide precedents. We will first train extractors to predict annotated participants and interaction roles, while also reconstructing the original features. Task losses give components useful information; the denoising objective determines how that information should be divided. These tasks guide learning without requiring a one-to-one correspondence between a component and a named concept.



### 2.2 Measure and learn dependencies through component denoising

Here *denoising* means predicting a clean component, or equivalently its flow target, from partially noisy states. *Generation* is the complete trajectory from noise to a sample. To measure dependence, hold the network and the target component's noise fixed and vary the information available in another component.

Let u collect the K components' noise-to-data progress, and let fθ,i predict the local velocity for component i. Define

```math
L_i(\mathbf u)=\frac{1}{d_i}\mathbb E
\left\|f_{\theta,i}(\widetilde{\mathbf s}(\mathbf u),\mathbf u,c)
-(s_i-\varepsilon_i)\right\|^2,
\qquad
D_{j\to i}(\mathbf u)=
L_i(\mathbf u_{j\leftarrow u_j^{\rm noisy}})
-L_i(\mathbf u_{j\leftarrow u_j^{\rm cleaner}}).
```

The context c contains the supplied prompt and reference features; larger u_j means less noise. A positive D means cleaner information in j improves denoising of i conditional on that context. For visual-to-semantic tasks, we omit the target role labels from the supplied prompt so that the task requires visual information. Examples, noise draws, other noise levels, and parameters remain fixed. These are local-velocity errors, so merely slowing a component's schedule cannot trivially reduce the diagnostic. Varying groups of components captures dependencies missed by pairwise measurements.

We will learn extractors that retain semantic prediction and reconstruction while making each component predictable from a limited subset of the others. Starting with the largest measured benefits, retain a small set of inputs and replace omitted inputs with independent noise during training, drawing on [masked prediction][multimae]. Normalized feature scales and a fixed total dimension limit rescaling and redundant-copy solutions. The result is an operational dependence structure defined by denoising performance, rather than a separately imposed graph. Section 2.4 will implement and analyze the corresponding attention restrictions. These measurements identify information useful to the current denoiser; for editing, they propose candidate groups to revise, whose effects must also be checked in the generated image.

### 2.3 Jointly optimize denoising schedules and the transformer

We will extend LWD's joint probe to K components, with monotone schedules that can cross. For fixed extractors, let v_k=s_k−ε_k be the local flow target and let f_k abbreviate the transformer's prediction at the current noisy states and progress vector. A concrete starting objective is

```math
\mathcal J_{\rm probe}(\theta,\phi)=
\mathbb E_{t,x,\varepsilon}\sum_{k=1}^K\frac{1}{d_k}
\left[
\operatorname{sg}(\dot\tau_k)\,\|f_k-v_k\|^2
+\lambda\|\dot\tau_k f_k\|^2
\right],
\qquad f_k=f_{\theta,k}(\widetilde{\mathbf s}_t,\boldsymbol\tau(t;\phi),c).
```

The first term fits the denoising network with a change-of-variable weight for each component's own progress. The operator sg holds that weight fixed during differentiation; schedule gradients still pass through the noisy states and progress inputs. The second term penalizes large predicted velocities in generation time and serves as a schedule-selection regularizer during the probe. This is LWD's prescribed surrogate-gradient construction, generalized from two representations to K components; it is not an unweighted flow loss whose value can be reduced just by changing the sampling of noise levels.

We will jointly update a temporary denoiser and φ during a short probe, then fix the schedules and train the main denoiser, as in our [LWD implementation][schedule]. Main training uses the weighted flow-fitting term without the kinetic penalty, together with auxiliary flow losses at independently sampled component noise levels, including components held clean. This trains the mixed-noise configurations needed for dependency measurements and editing. Extractors remain fixed within a probe and are refined between probes. Our [variational anisotropic method][trajectory] supplies a complementary score-based formulation.

Generation integrates the predicted velocity in generation time:

```math
\frac{d\widehat s_k}{dt}
=\dot\tau_k(t;\phi)\,
f_{\theta,k}(\widehat{\mathbf s}_t,\boldsymbol\tau(t;\phi),c).
```

**The schedule must balance information gained and information missing.** Denoising one component early can improve other components' predictions, but raises its own error if useful context is still noisy. Optimizing the joint trajectory balances these effects across conditional prediction problems; a fixed semantic-first rule cannot express all such tradeoffs.

**Localize computations as well as generation stages.** An illustrative procedure generates A over time [0, 0.2], then B and C conditioned on A over [0.2, 0.6], and finally D conditioned on B and C over [0.6, 1]. Different transformer modules may serve these conditional prediction problems, with some computations shared across stages. In the learned procedure, schedules can overlap and cross; module activity depends on the vector of component noise levels rather than a fixed global time interval. For a module selected by the current condition–target task, we write its gated contribution as

```math
g_m(\mathbf u)\,h_{\theta_m}
(\widetilde{\mathbf s}_{I_m},\mathbf u,c),
\qquad \mathbf u=\boldsymbol\tau(t;\phi).
```

Here I_m is the set of representation components accessible to module m; h is its computation and g_m controls its activity. Initially, a few gates select coarse configurations such as a relatively clean condition and a noisy target. We will then learn smooth gates, with their parameters included in θ, under the same denoising objective while updating the schedules during the probe. Section 2.4 specifies the module inputs, outputs, and sharing patterns. Shared backbone computations can remain active across all configurations.

**Explain when a computation is needed.** At fixed noisy inputs, remove or replace a head or MLP contribution and measure which component predictions change. Repeat across noise configurations and trace selected interventions through the remaining generation trajectory. This reveals which computations become useful when a condition is clearer and which remain reusable when the prediction target changes. The measured roles guide which modules to activate; the analysis includes the shared backbone as well as the gated additions.

For revision, use the semantic task loss to optimize selected component vectors toward the requested change, with preservation measured after reconstruction into fixed encoder coordinates. The initial task is changing participant roles. Add noise to the changed components and candidate dependent groups, then denoise conditionally while retaining the others. Task predictions on the re-encoded output image will assess whether the semantic edit is realized, and fixed image features will assess the requirements to preserve. A useful denoising dependency proposes a resampling group; the final image determines whether the group was sufficient or needs to expand. The corresponding module interventions explain how the revision propagates.

![Figure 2: fig2 recovery schedule](figures/fig2_recovery_schedule.png)

**Figure 2. Denoising dependencies guide schedules and module activity.** Make component j cleaner while holding target i's noise, the other inputs, and the transformer fixed. The change in i's flow-prediction error measures the benefit of information from j. Learned schedules balance these benefits against the difficulty of denoising a component before its useful context is available; module activity can depend on the same noise configuration. The curves and module assignments are illustrative.

### 2.4 Train shared and specialized transformer computations for compositional denoising

**Use diverse conditional prediction tasks to learn composition.** Section 2.2 supplies tasks such as predicting B from A, C from A, or D from B and C, with varying noise on the conditions and targets. Across training examples, we will vary the participants, relationships, and their combinations. Thus the model encounters both different semantic compositions and different uses of the same representation. These tasks encourage it to reuse a useful computation whenever the relevant information recurs.

The initial construction will use a shared transformer backbone and a small family of attention and MLP modules. A task specifies which components supply conditions and which are prediction targets; these masks are supplied to the module selector, and the noise vector specifies how reliable each input is. The flow loss is applied to the target components. The first restricted branches read selected component vectors before unrestricted backbone mixing and add only to the chosen targets' velocity predictions. Their input/output projections therefore specify access for the added computation. We will keep the learned representation interface fixed while comparing three concrete designs:

- **Sparse attention across representations.** Use the dependencies measured in Section 2.2 to restrict the component groups accessible to selected attention heads. Token masks or input projections implement these restrictions. This lets a head learn a cross-representation rule from a limited set of relevant inputs.
- **Shared prediction computations.** Share an MLP prediction bank across conditional tasks, with task-specific input and output projections. This asks whether a common learned predictor can be used through different information-selection and output mappings.
- **Specialized prediction computations.** Switch selected MLPs or projections according to which components are conditions and targets. For example, predicting visual features from interaction features may require a different predictor from inferring an interaction from visual features, while retaining common computations elsewhere.

These are alternative sharing patterns within one implementation. We will begin by freezing the pretrained backbone and training the small added modules, then release selected backbone parameters where conditional prediction requires it. In the small-model studies, we can also replace or share existing MLP blocks directly. The preliminary attention/MLP interpretation motivates these choices; the experiments will identify the useful division of computation. [Compositional Attention][compositional] provides a related separation of selection and retrieval.

**Identify the computations that transfer across compositions.** Hold the representation and noise configuration fixed, then remove or swap a module between conditional tasks. Component-wise denoising errors show where the computation remains useful; completed generations show whether it preserves the intended meaning in a new participant–relation combination. Replacing a candidate module with one serving an unrelated task provides a direct check of its assigned role. Restrictions establish which inputs an added module can access; these interventions determine what it contributes in the full transformer. Effects transmitted through the shared backbone are included in the analysis.

A tractable analysis begins with squared-error denoising: separate the prediction error caused by removing useful inputs from the error of the learned predictor using retained inputs. Linear models with Gaussian features provide an initial setting for identifying when conditional tasks can share a predictor or require different projections. The analysis and module interventions will guide which restrictions to retain or relax. The resulting architecture will support selective generation and revision through reusable computations, without requiring each human-named concept to occupy a unique module.

![Figure 3: fig3 component reuse](figures/fig3_component_reuse.png)

**Figure 3. Existing component-reuse evidence.** Images are extracted from the first report-selected noise input (seed 0, latent 1) in Experiment 3, pages 2 and 5. Restoring source query/key weights while retaining adapted value/output projections and MLPs preserves much of the AAHQ change but does not reproduce the STL-10 joint result. Bars report 1 − E_hybrid/E_source, where E is mean squared distance to the paired joint output in normalized DINOv2 features: 1,024 matched latents per adaptation seed, averaged over two seeds. The score measures recovery of joint-model outputs, not target fidelity. These are 20M-model post-training reversion experiments; joint endpoints were selected using the existing test-FID results.

## 3. Why the proposed work is feasible

Existing work supports the connections underlying the proposal. [SFD][sfd] and [SeFi-Image][sefi] show that generated semantic information can guide image generation; [Latent Forcing][latentforcing] and [Diffusion Forcing][diffusionforcing] support varying generation across representations or token noise levels. [Local Mechanisms][local] demonstrates benefits of restricted dependencies for composition in a controlled setting, with suggestive structure in SDXL features. [Vision-Language Binding][binding] shows that interventions can trace how references influence an image.

The representation-learning methods also have practical foundations. [VAE/GAN][vaegan] uses discriminator features for reconstruction, [REPA][repa] aligns diffusion features with pretrained representations, and [ImageReward][imagereward] supplies learned preference supervision. [UniDiffuser][unidiffuser] demonstrates training a shared transformer for different multimodal conditional generation tasks. Our proposed advance connects learned component divisions and schedules to shared or specialized transformer computations for those tasks.

Our existing results provide the following evidence:

| Existing work | Result | What it establishes |
|---|---|---|
| ELF-L, unpublished | A separate diffusion model generates projected frozen-Qwen answer states and reaches 61.94% GSM8K accuracy. Qwen encodes the question but does not generate the solution autoregressively at inference. | Diffusion can generate language-model states that retain enough information for substantial reasoning performance. |
| [Learning When to Denoise][schedule] | With a matched 675M-parameter backbone, AutoGuidance, and dopri5 sampler, LWD reaches FID 1.05 after 200 epochs versus SFD-XL's 1.06 after 800 epochs. | We can train models that generate semantic and visual representations together and control their relative timing. |
| [From Softmax to Score][softmax] and our working mechanistic manuscript | Constructive connections between attention and denoising; analysis of the information needed for prediction and the effect of removing attention contributions. | We have analytical tools for studying what attention computes and when its output can be reused. |
| Parameter sharing and transfer, unpublished | A shared prediction-feature variant improves CelebA64 FID from 17.574 to 14.304 at approximately 10.2M parameters. Restoring source query/key weights after joint adaptation gives recovery scores of 0.815 for CelebA→AAHQ and 0.143 for CelebA→STL-10. | Sharing can improve parameter allocation, while reuse succeeds on some changes and fails on others. |

The sharing result uses one training seed. The transfer metric compares outputs in DINO feature space against the jointly adapted generator. Those experiments restored weights after training; they do not establish that keeping those weights fixed throughout training would work equally well. This distinction motivates the proposed reuse tests.

ELF's reasoning accuracy is a functional demonstration that diffusion-generated Qwen states retain precise language information. This motivates semantic targets capable of carrying complex prompt requirements. Our [anisotropic trajectory optimization research][trajectory] provides joint network/schedule learning and estimators accounting for a changing noise process; our asynchronous flow work provides another direct algorithmic starting point.

These results establish starting points for representation generation, schedule design, and component reuse. The proposed research connects them within the same conditional denoising tasks. Existing encoders, checkpoints, and schedule-learning code let the four steps begin from a common implementation; learned projections and small modules can then be incorporated progressively.

![Figure 4: fig4 existing feasibility](figures/fig4_existing_feasibility.png)

**Figure 4. Existing results support representation generation, schedule design, and parameter sharing.** (A) ELF generates projected frozen-Qwen contextual states; decoded answers reach 54.17% GSM8K accuracy with synchronous inference and 61.94% with asynchronous inference from the same checkpoint (EMA 0.9999, 32 ODE steps, two inference seeds over the same 1,319 questions). This supports reasoning-relevant information in generated language states; no matched AR advantage is claimed. (B) [LWD][schedule] reaches FID 1.05 at 200 epochs versus SFD-XL's 1.06 at 800 epochs, with matched 675M backbones, AutoGuidance, and dopri5 sampling. (C) Experiment 1, page 6, reports FID-50k 17.574 versus 14.304 for 10,215,472 versus 10,239,536 parameters on CelebA64, using one training seed, epoch-400 EMA, and a 50-step sampler. Values are redrawn from documented results; no new evaluations are included.

## 4. Creator demonstration and relevance to Sony

**Generate a scene, then revise one interaction.** A creator supplies character references and a description of the handover scene. The model first generates an image satisfying the participants, clothing, interaction, and independent activity. The creator then reverses who gives the cup while retaining the other requirements. This single workflow brings together component learning, conditional dependencies, asynchronous denoising, and shared or specialized transformer computations.

The demonstration will show the requested semantic change, the components revised, and the attention or MLP computations that carry the change into the image. It will also show whether those computations remain useful with different participants or a new arrangement. Reference identity, participant roles, attributes, and independent activities will be assessed separately, alongside image quality. Preservation will be evaluated at comparable success in making the requested edit. Independent evaluators and human assessment will complement the training signals; [GenEval][geneval] provides a starting point for prompt-grounded assessment, extended to action roles.

The practical goal is fewer unintended changes when a creator revises a scene, supporting visual asset development for games, animation, and film. The accompanying account of what the modules compute and when they matter addresses [Sony's stated interests][sony] in modality binding, generation dynamics, and causal interventions. Keeping conditional prediction modules explicit also makes their contributions easier to inspect through removal and replacement.

The initial study will use a bounded vocabulary of annotated interactions and participant roles. Existing generators can supply original/revised prompt pairs, following [InstructPix2Pix][instruct]. Shared initial noise can aid comparison, but proposed pairs will be checked for the intended change and preserved requirements. Some participant–relation combinations will be held out so that successful revision requires the computations to transfer. These are award-period resources; the proposal uses only existing results as preliminary evidence.

## 5. Work plan and deliverables

**The core deliverable is a generation and revision method whose representation dependencies, denoising schedules, and transformer computations are understood together.** All four research steps use the same component interface and conditional prediction tasks.

### Concrete execution

We will begin with our existing 10–20M-parameter diffusion Transformers and semantic/visual diffusion pipeline, reusing checkpoints, component-replacement code, and schedule-learning implementations. For the text-to-image prototype, we will use the released **PixArt-Σ 512-pixel model**, with an approximately 0.6B-parameter denoising Transformer and available training/adaptation code. Its native T5 text conditioning and VAE will remain in place. [PixArt-Σ][pixart]

The initial representation combines image latents with frozen DINOv2 features and Qwen3-4B-Instruct-2507 states from the representation pipeline used in ELF. We will learn four to eight component groups using semantic tasks, reconstruction, and denoising losses. The groups begin as disjoint projected coordinates or token groups. Small projections, a semantic denoising branch, and trainable attention interfaces connect the generated features to image latents. Reference-image features enter as clean conditioning; generated component states carry the features of the requested scene. Qwen targets are extracted from annotated training scene descriptions containing participant and role information. T5 receives the supplied prompt, which can specify only a subset of that information; conditional tasks withhold the attributes or roles being inferred. At inference, the description-derived target states are generated; reference features and prompt conditioning remain available as inputs.

The first prototype will freeze the pretrained image backbone and train the projections, semantic branch, and added conditional modules. Caching encoder outputs limits repeated feature extraction. We will release selected backbone parameters only where the restricted additions cannot support the required denoising tasks. The same fixed conditioning will be used when comparing module interventions, so changes can be traced through the proposed representation path.

| Resource | Role in the initial implementation |
|---|---|
| Existing CelebA64, AAHQ, and STL-10 experiment pipelines | Study module reuse, sharing, and noise-dependent activity in small models before applying the same construction to text-to-image generation. |
| [SWiG grounded situation recognition][swig] | Activity labels, participant-role labels, and entity locations for task-supervised component learning and conditional denoising. |
| [Visual Genome][visualgenome] | Objects, attributes, and relationships for training varied semantic compositions and checking preservation. |

We will select a bounded set of activities and relations and form scene descriptions from their annotations. The first conditional tasks will predict selected semantic components from visual components and vice versa, followed by joint generation at varying noise levels. The first revision cases will involve two interacting participants and clearly specified attributes; an independent activity, such as the child reading, provides a harder preservation case. Reference crops and original/revised pairs will be checked for identity and role consistency. Success on the small-model studies will guide the sharing patterns used in the text-to-image prototype.

The budget requests 1.5 months of PI salary and 12 months of graduate research assistant support. The research uses existing pipelines and pretrained models; no separate computing costs are requested. All experiments below are proposed for the 12-month award period.

| Period | Main work | Concrete output |
|---|---|---|
| Months 1–3 | Obtain pretrained targets, learn initial projections using semantic tasks and reconstruction, and train mixed-noise conditional denoisers. | A fixed initial component interface and conditional prediction tasks; baseline generation and denoising measurements. |
| Months 4–6 | Measure component dependencies and jointly optimize schedules and denoiser parameters; analyze module activity across noise configurations. | Learned schedules and an initial account of which computations are needed as conditions become informative. |
| Months 7–9 | Train sparse attention and shared or specialized MLP computations across conditional tasks and participant–relation combinations. | A transformer construction with measured module roles and a working selective-revision procedure. |
| Months 10–12 | Consolidate the text-and-reference image workflow, assess preservation and transfer to new combinations, and document the internal computations. | Creator demonstration, implementation, mechanistic findings, reproducible evaluation materials, and final report. |

We will retain the simplest sharing pattern that supports the conditional tasks. Where useful dependencies span several components, attention access can be expanded and computations shared at a coarser level. The common implementation allows useful representation or schedule improvements to be retained even when a particular module-sharing restriction is ineffective.

Text-and-reference image generation and selective revision define the award-period application. Meta-learning, model merging, generation of parameter updates from demonstrations, and temporal/audio generation are possible future uses of the learned organization. They are outside the committed deliverables. We will provide three quarterly reports, a final research summary, and the required progress-questionnaire responses.

## References

1. Oquab et al. [DINOv2: Learning Robust Visual Features without Supervision][dino]. arXiv:2304.07193, 2023.
2. Bachmann et al. [MultiMAE: Multi-modal Multi-task Masked Autoencoders][multimae]. arXiv:2204.01678, 2022.
3. Bao et al. [One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale][unidiffuser]. arXiv:2303.06555, 2023.
4. Pan et al. [Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion][sfd]. arXiv:2512.04926, 2025.
5. SeFi-Team. [SeFi-Image: A Text-to-Image Foundation Model with Semantic-First Diffusion][sefi]. arXiv:2606.22568, 2026.
6. Bradley. [Local Mechanisms of Compositional Generalization in Conditional Diffusion][local]. arXiv:2509.16447, revised 2026.
7. Mittal et al. [Compositional Attention: Disentangling Search and Retrieval][compositional]. ICLR, 2022.
8. Ge et al. [Vision-Language Binding in In-Context Image Generation][binding]. arXiv:2605.24624, 2026.
9. Brooks, Holynski, and Efros. [InstructPix2Pix: Learning to Follow Image Editing Instructions][instruct]. CVPR, 2023.
10. Rosu, Carin, and Cheng. [From Softmax to Score: Transformers Can Effectively Implement In-Context Denoising Steps][softmax]. NeurIPS, 2025.
11. Liu, Li, and Cheng. [Variational Trajectory Optimization of Anisotropic Diffusion Schedules][trajectory]. arXiv:2602.19512, 2026.
12. Qian and Cheng. [Learning When to Denoise: Optimizing Asynchronous Schedules for Latent Diffusion][schedule]. arXiv:2606.19662, 2026.
13. Larsen et al. [Autoencoding beyond pixels using a learned similarity metric][vaegan]. arXiv:1512.09300, 2015.
14. Xu et al. [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation][imagereward]. arXiv:2304.05977, 2023.
15. Yu et al. [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think][repa]. ICLR, 2025.
16. Chen et al. [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion][diffusionforcing]. arXiv:2407.01392, 2024.
17. Baade et al. [Latent Forcing: Reordering the Diffusion Trajectory for Pixel-Space Image Generation][latentforcing]. arXiv:2602.11401, 2026.
18. Ghosh, Hajishirzi, and Schmidt. [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment][geneval]. arXiv:2310.11513, 2023.
19. Ma et al. [Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning][deme]. CVPR, 2025.
20. Chen et al. [PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation][pixart]. Implementation and released checkpoints.
21. Pratt et al. [Grounded Situation Recognition (SWiG)][swig]. ECCV, 2020; project and annotations.
22. Krishna et al. [Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations][visualgenome]. IJCV, 2017.

**Unpublished preliminary materials.** *A Mechanistic View of Diffusion Transformers as Adaptive Patchwise Denoisers*, working manuscript; associated Experiment 1 and Experiment 3 reports; ELF-L experimental summary and accuracy records, 2026. These materials support the explicitly labeled preliminary results in Section 3.

## Budget summary

**Total requested: USD 144,300 for 12 months.** The budget supports 1.5 months of PI salary and 12 months of support for one graduate research assistant, together with associated fringe benefits and graduate tuition remission. Personnel will carry out the proposed representation learning, mechanistic analysis, model development, and creator demonstrations.

| Budget category | Amount (USD) |
|---|---:|
| PI salary: Xiang Cheng, 1.5 months | 23,833 |
| Graduate research assistant stipend: 12 months | 44,213 |
| PI fringe benefits: 28.83% of PI salary | 6,871 |
| Graduate research assistant fringe benefits: 11.45% of stipend | 5,062 |
| Graduate tuition remission: 34.23% of stipend | 15,134 |
| **Total direct costs** | **95,113** |
| Indirect costs: 61.5% of modified total direct costs | 49,187 |
| **Total requested** | **144,300** |

**Budget basis.** Amounts follow the supplied institutional budget, rounded to whole dollars. PI salary uses a nine-month salary basis of USD 143,000. The modified total direct cost base is USD 79,979, consisting of salaries/stipend and fringe benefits; tuition remission is excluded. Applying the 61.5% indirect cost rate gives USD 49,187. No funding is requested for equipment, supplies, travel, other expenses, or subcontracts. The total includes indirect costs and is within the [Focused Research Award limit][sony] of USD 150,000.

## Editorial notes for finalization — remove before submission

- Figures 1 and 2 are explanatory diagrams; Figures 3 and 4 use existing results. Figure sources, selected image panels, and reproduction instructions are documented in [figures/README.md](figures/README.md). No additional experiments are required before submission.
- The LaTeX version in `latex/proposal.tex` has been compiled and checked: ten pages of narrative, figures, and references, followed by a separate budget page, with 10-point body text. Recheck pagination after further edits.
- Complete PI contact details. The PI CV is a separate submission item.
- Confirm the budget calendar dates with the institutional administrator. The supplied workbook lists July 1, 2026–June 30, 2027, but its student stipend and tuition calculations blend eight months of 2026–27 rates and four months of 2027–28 rates. The proposal preserves the supplied amounts and 12-month duration without assigning replacement dates.
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
[binding]: https://arxiv.org/abs/2605.24624
[instruct]: https://arxiv.org/abs/2211.09800
[softmax]: https://papers.nips.cc/paper_files/paper/2025/hash/d27af2bebeab8a3e6f3848fc71736235-Abstract-Conference.html
[trajectory]: https://arxiv.org/abs/2602.19512
[schedule]: https://arxiv.org/abs/2606.19662
[sony]: https://www.sony.com/en/SonyInfo/research-award-program/

[vaegan]: https://arxiv.org/abs/1512.09300
[imagereward]: https://arxiv.org/abs/2304.05977
[repa]: https://arxiv.org/abs/2410.06940

[diffusionforcing]: https://arxiv.org/abs/2407.01392
[latentforcing]: https://arxiv.org/abs/2602.11401

[geneval]: https://arxiv.org/abs/2310.11513

[deme]: https://arxiv.org/abs/2410.06664

[pixart]: https://github.com/PixArt-alpha/PixArt-sigma
[swig]: https://prior.allenai.org/projects/gsr
[visualgenome]: https://arxiv.org/abs/1602.07332
