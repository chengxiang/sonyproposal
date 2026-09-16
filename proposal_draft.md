# How Representations and Weights Shape Multimodal Generation

**Principal investigator:** Xiang Cheng, Department of Electrical and Computer Engineering, Duke University  
**Focused Research Theme:** Internal Mechanisms of Multimodal Generative Models for Content Creation  
**Project duration:** 12 months  
**PI contact:** xiang.cheng@duke.edu; +1 484-630-3381

## Abstract

Multimodal diffusion models can generate rich visual content, yet we have limited understanding of how particular semantic requirements are carried through denoising and implemented by transformer computations. We propose to use **conditional dependencies among representation components to organize both generation and the computations that realize it**. We will obtain semantic representations from pretrained models and models trained on relevant downstream tasks, then hold these representations fixed while jointly learning denoising schedules and the transformer. By varying the information available in each component and intervening on attention and multilayer perceptron (MLP) contributions, we will identify which computations use that information, when they are needed, and when they can be shared. The PI's existing and ongoing work on continuous diffusion language models, learned asynchronous schedules, and mechanistic analysis of transformer components provides the foundation. The one-year project will deliver an integrated representation-to-generation pipeline, a mechanistic account of selected conditional predictions, and a text-to-image demonstration that satisfies interacting requirements for participants, attributes, and action roles.

## 1. The proposed contribution

**Which information does a diffusion model need for a particular prediction, when should that information become available, and which transformer computations use it?** Answering these questions together connects representation design to the internal mechanisms of generation. Existing methods demonstrate the value of semantic representations, asynchronous diffusion, and modular computations. Our proposed advance is to connect them through the same measurable conditional dependencies, providing a basis for precise generation and reusable computation.

Consider the request: “A woman in a red coat hands a blue cup to a man in a green sweater.” A plausible-looking image can still reverse the action or attach clothing to the wrong person. Information that helps predict the giver–receiver assignment should also help determine when relevant states are denoised and which attention or MLP computations carry that information into visual predictions. We will identify this relationship through controlled changes in component availability and module contributions.

Our central hypothesis is that **dependencies useful for conditional generation also identify opportunities to specialize and share transformer computations**. A component need not have a human-readable semantic label: its meaning can be characterized by which conditional predictions improve when it is available. Here, *locality* means dependence on a limited subset of representation components, which can span distant tokens or modalities. We seek selective information use where the task supports it, while identifying cases that require broader interactions.

The approach has three connected steps. First, obtain task-relevant semantic activations and define fixed components. Second, jointly learn their denoising schedules and the transformer, measuring how one component helps predict another. Third, identify and train computations that exploit these dependencies across conditional tasks. Representation learning occurs in the first step; dependency and schedule learning do not subsequently change the representation or regroup its coordinates.

**A concrete starting point.** We will begin with existing 10–20M-parameter diffusion transformers, then build a text-to-image prototype around pretrained [PixArt-Σ][pixart]. The initial representation combines fixed DINOv2 and Qwen activations with the generator's existing image latents. Four semantic components and one image-latent component provide a bounded starting configuration. A small semantic denoiser and conditional modules connect these states while retaining the pretrained image-decoding path. Section 2 specifies their construction; Section 6 bounds the data and compute.

Continuous diffusion is important for both image and video generation. Text-to-image examples make the proposal concrete, while the same questions arise when video frames and semantic states progress at different noise levels, as motivated by [Diffusion Forcing][diffusionforcing]. The committed demonstration focuses on images; video is a subsequent extension.

![Figure 1: representation components, conditional information, denoising schedules, and transformer computations](figures/fig1_mechanism_overview.png)

**Figure 1. Conditional information connects representation, generation, and computation.** Fixed representation components are characterized by their contribution to conditional predictions. Varying their noise levels reveals useful dependencies; learned schedules determine when information becomes available. Component and module interventions identify the transformer paths that use it. The illustrated component meanings, schedules, and module assignments are examples, not prescribed semantic labels or a fixed generation order.

## 2. Research approach: dependencies that organize denoising and transformer computation

We will obtain semantic representations first, then hold them fixed while learning their conditional dependencies, denoising schedules, and transformer computations. The initial prototype combines a small semantic denoising branch with the released 0.6B-parameter [PixArt-Σ][pixart] image transformer, retaining its text conditioning and image decoder. Approximately 20–50M parameters in the added branch, interfaces, and selected adaptations will be trainable. Our existing 10–20M-parameter diffusion transformers provide a smaller setting for developing the conditional tasks and module interventions.

### 2.1 Obtain task-relevant representations and fix their components

**Start with pretrained activations, then use activations learned for the desired task.** Frozen [DINOv2][dino] image-token activations and Qwen caption-token activations provide the first semantic representations. The second representation source will be a prompt-image compatibility model trained to distinguish correct descriptions from descriptions with reversed action roles or exchanged attributes. We will use its visual branch's hidden activations as generation targets: the training task encourages useful distinctions, while the resulting image representation does not require the correct caption as an input. [Discriminator features][vaegan] and [reward models][imagereward] motivate obtaining representations from such task signals. The committed comparison uses one such task-trained alternative to the pretrained visual features.

**A component is a specified group of representation coordinates.** For an image and its training description, let the encoder output token vectors with $d$ coordinates. Fit a principal-component projection on the training features, normalize their coordinate scales, and divide each projected token into two fixed coordinate groups. Collecting the first group across tokens gives one component vector; collecting the second gives another. Applying this construction separately to the visual and language representations produces four semantic components. The complete VAE image-latent array forms a fifth component. Write these vectors as $s_k\in\mathbb R^{d_k}$, where $k=1,\ldots,K$, $K=5$ initially, and $d_k$ is the number of coordinates in component $k$. Each component retains its token positions internally. Encoders, projections, normalization, and group assignments remain fixed during the following stages.

The groups need not correspond to human-named concepts. Their meaning will come from which conditional predictions improve when their information is available. For example, which visual groups help recover the missing word in “The [MASK] gives the cup”? This functional characterization distinguishes representation learning through the source task from dependency learning through denoising. Full-caption states and image features supply training targets; the generator must produce these states at inference.

### 2.2 Jointly learn denoising order, conditional dependencies, and the transformer

**Vary how much information each component supplies.** Let $u_k\in[0,1]$ measure component $k$'s progress from noise toward data, and let $\varepsilon_k$ be independent standard Gaussian noise with $d_k$ coordinates. Form a noisy training state and its target rate of change as

```math
\widetilde s_k(u_k)=\alpha_k(u_k)s_k+\sigma_k(u_k)\varepsilon_k,
\qquad
r_k(u_k)=\alpha'_k(u_k)s_k+\sigma'_k(u_k)\varepsilon_k.
```

Primes denote derivatives with respect to $u_k$. For semantic components, use $\alpha_k(u_k)=u_k$ and $\sigma_k(u_k)=1-u_k$, so $r_k=s_k-\varepsilon_k$. For image latents, retain PixArt's signal and noise coefficients, traversed from high noise toward the image. The transformer receives all noisy components, their progress vector $\mathbf u=(u_1,\ldots,u_K)$, and conditioning $c$, which contains the supplied prompt or incomplete caption. Its output $f_{\theta,k}$ predicts $r_k$; $\theta$ denotes the trainable transformer parameters. Squared prediction error, averaged over a component's coordinates, gives its denoising loss. Generation integrates these predictions to produce complete states.

**Learn when to generate information.** Assign each component a monotone schedule $u_k=\tau_k(t;\phi)$, with generation time $t\in[0,1]$, schedule parameters $\phi$, and endpoints $\tau_k(0)=0$, $\tau_k(1)=1$. The schedules may overlap or cross. We begin with piecewise-linear schedules whose positive increments are learned. Write $\dot\tau_k$ for their rates, and abbreviate the prediction and target evaluated at these progress levels by $f_k$ and $r_k$. A concrete extension of our variational [schedule-learning][schedule] and [trajectory-optimization][trajectory] approach is

```math
\mathcal J(\theta,\phi)
=\mathbb E\!\left[\sum_{k=1}^{K}\frac{1}{d_k}
\left\{\dot\tau_k\|f_k-r_k\|_2^2
+\lambda\|\dot\tau_k f_k\|_2^2\right\}\right]
+\beta\mathcal L_{\mathrm{ind}}(\theta).
```

The expectation averages over training examples, Gaussian noise, and uniformly sampled $t$; $\|\cdot\|_2$ is the Euclidean norm. The first term trains predictions along the schedules, and the second discourages large generation velocities. The nonnegative coefficients $\lambda$ and $\beta$ set the penalties' relative weights. Here $\mathcal L_{\mathrm{ind}}$ is the same coordinate-averaged prediction loss at independently varied component progress levels, including clean conditions and fully masked targets. These configurations teach the model to use combinations of available information beyond those encountered on a single schedule.

Changing $\phi$ changes both the information available to the transformer and the prediction problems that train its weights. If cleaner component $X$ improves prediction of component $Y$, advancing $X$ can help generate $Y$, but may make $X$ itself harder to predict from its noisier surroundings. The joint objective balances these conditional prediction requirements without imposing a semantic-first order. After selecting schedules, we fit the final denoiser using prediction losses alone and remeasure dependencies, so the velocity penalty does not determine the final predictor's answer.

**Measure which information helps each conditional task.** Freeze a trained checkpoint and compare target denoising error when one conditioning component is clearer or noisier, keeping the target noise, other inputs, and examples fixed. A reduction in error measures a useful dependency for this model and context; it does not identify a unique underlying graph. To characterize several conditioning components together, optimize

```math
\min_{\mathbf u_C\in[0,1]^{|C|}}
\mathcal L_{\mathcal T}(\mathbf u_C)
+\rho\sum_{j\in C}u_j.
```

Here $\mathcal T$ is a conditional task, $C$ is its set of candidate conditioning components, $\mathbf u_C$ specifies their progress levels, $\mathcal L_{\mathcal T}$ is the expected target prediction loss, and $\rho>0$ penalizes exposing cleaner information. Target components remain at fixed noise levels. Fit one allocation across task examples and evaluate it on held-out examples. The result identifies an economical allocation of useful information; redundant components may substitute for one another, and the penalty limits total availability rather than guaranteeing a particular number of active components.

For the masked-caption diagnostic, remove the word from the conditioning caption before contextual encoding and keep all full-caption target states at pure noise. Disable the image-latent/backbone route so visual evidence enters only through the candidate visual groups. The task loss focuses on the missing caption position; a readout trained on clean training-caption states and then frozen reports missing-word accuracy from the predicted states. The same incomplete caption accompanies matched images with different giver–receiver assignments. Full-prompt image generation is a separate task: the native text encoder legitimately supplies the stated relationship, and the question is whether the additional semantic path helps realize it visually.

**Integrate the components with the pretrained image model.** Retain PixArt's native image noising, T5 conditioning, and VAE decoding. Convert its noise prediction into a rate of change along that same image path: if $y=\alpha s+\sigma\varepsilon$ is the noisy image latent and $\widehat\varepsilon$ its predicted noise, then $\widehat s=(y-\sigma\widehat\varepsilon)/\alpha$ and the image-rate prediction is $\alpha'\widehat s+\sigma'\widehat\varepsilon$. This algebraic conversion uses the checkpoint's finite noise endpoints. Added conditional modules modify the image prediction using semantic states and jointly predict those states; they do not require retraining the image model under a different corruption process. Sampling uses the learned rates $\dot\tau_k f_k$ to advance all components, then decodes the final image latents through the existing VAE.

We will compare downstream generation conditioned on corrupted ground-truth semantic states with generation conditioned on model-generated states at corresponding noise levels. Prompt satisfaction and image quality will reveal whether accumulated semantic errors reduce the measured benefit. If necessary, short-rollout training will expose the conditional modules to generated states, following the motivation of [Self Forcing][selfforcing], using supervision appropriate to the conditional task.

### 2.3 Build and explain reusable conditional computations

**Use measured dependencies to define module access.** A module initially means one attention head or one MLP residual branch. Keep the target's own noisy state available, rank other components by their measured denoising benefit, and retain the smallest ranked input set whose masked-input validation error is within 5% of the full-input predictor. Omitted components are replaced by independent noise at progress zero during this selection; retain all inputs if needed. Train restricted attention heads to read only the selected component vectors before unrestricted mixing, with output projections writing only to designated targets. We will implement a shared attention–MLP core with target-specific input/output projections, training it across different condition/target assignments and participant–attribute combinations. This gives “diverse composition” a concrete meaning: the same computation must support recurring conditional relationships in different scenes. If a shared predictor cannot serve both directions, specialize its target projection or MLP while retaining useful shared computations.

A learned scalar gate for each added branch takes the component progress levels as input. It controls the branch's contribution under the same denoising loss, allowing a computation to become active when its inputs are sufficiently informative. The first study therefore localizes computation in both representation components and denoising stages. Restrictions apply to the added branches; the pretrained backbone remains an additional information path. Dense access remains available where conditional prediction requires it.

**Connect a component's effect to a particular computation.** In a matched role-swap case, first measure the benefit of exposing a candidate visual component for predicting the giver. Exchange that component between the two images, then disable a candidate attention head or MLP and repeat. A specific mechanism is supported when removing the module suppresses the component's effect, restoring its contribution restores that effect, and unrelated predictions remain usable. Equally sized random-module interventions and perturbation-magnitude controls distinguish selective effects from general damage, following [activation-patching methodology][patching]. Repeat at several noise levels and on unseen compositions to identify when the path is needed and when it is reusable.

These case studies explain specified information paths in the conditional denoiser and added semantic branches. Carrying selected interventions through complete image generation connects that account to useful control, with T5 conditioning held constant. The result will be an interpretable account of how particular components influence predictions through attention and MLP computations, rather than a requirement that every component or weight have a human-readable semantic label.

## 3. Differentiation from the current state of the art

**The proposed research connects the usefulness of representation components to both denoising progress and transformer function.** Asynchronous generation alone does not identify a reusable computation, and an important attention head alone does not explain which conditional information it uses. Our contribution is to establish this correspondence and use it to organize generation.

| Closest approach | Existing contribution | Proposed advance |
|---|---|---|
| [SFD][sfd], [Latent Forcing][latentforcing], and our [LWD][schedule] | Generate complementary representations asynchronously; LWD learns schedules jointly with a denoiser. | Measure dependencies among multiple task-relevant components and use them to organize semantic control as well as denoising progress. |
| [Local Mechanisms of Compositional Generalization][local] | Connect sparse conditional-score dependencies, including feature-space composition, to generalization and interventions. | Relate noise-dependent component usefulness to identifiable attention/MLP computations and their reuse across conditional tasks. |
| [Vision-Language Binding][binding] | Trace how reference information influences multimodal transformer predictions. | Combine causal analysis with learning schedules and conditional modules that exploit the identified information paths. |

The schedule is an operational use of dependence, rather than a unique graph recovered from data: varying noise levels measures predictive usefulness, while schedule optimization determines when producing that information is worthwhile. Our variational schedule-learning work makes this possible beyond a manually selected semantic-first order. Compared with a general low-rank adaptation, the proposed module construction explicitly specifies the information a computation can access and the predictions it can change. This gives interventions and parameter sharing a more interpretable basis, whose practical value is assessed in the integrated demonstration.

## 4. Existing results and feasibility

**Learned generation schedules.** On class-conditional ImageNet-256, our [LWD][schedule] achieves FID **4.93 versus REPA's 5.84** with **120,000 versus 4 million main-training updates**, plus a 10,000-update schedule-learning phase. This is approximately 33 times fewer main-training updates for the overall method. Its strongest reported FID is **1.02 with a 675M-parameter model**, below the **1.04** reported for the 1B-parameter SFD-XXL in that comparison. These results demonstrate the potential of asynchronous multi-representation generation. At a matched 400,000-update budget, learning the schedule also improves unguided FID from **3.53 to 2.87** with the same SemVAE representation, directly supporting schedule adaptation within a fixed representation.

**Interpretable and reusable transformer computations.** Our [attention-as-denoising analysis][softmax] and ongoing *Adaptive Patchwise Denoisers* study connect attention-based aggregation with prediction knowledge that can be stored in MLPs. Preliminary sharing experiments improve CelebA64 FID from **17.574 to 14.304** at approximately **10.2M parameters**. Selective restoration of source weights after adaptation produces different outcomes across dataset shifts (Figure 2), motivating an account of which computations remain compatible and reusable. These experiments provide implementations and evidence for the ingredients; the proposed semantic component–module correspondence is the next step.

![Figure 2: existing selective transformer reuse results](figures/fig2_component_reuse.png)

**Figure 2. Existing evidence motivates selective reuse.** Source query/key weights are restored after adaptation, retaining adapted value/output projections and MLPs. Recovery is $1-E_{\mathrm{hybrid}}/E_{\mathrm{source}}$, where $E$ is mean squared distance to paired jointly adapted outputs in unit-normalized DINOv2 features. Scores average two adaptation seeds with 1,024 paired latents per seed; the displayed images are illustrative matched examples. The AAHQ result motivates selective reuse, while the STL-10 result motivates understanding when coordinated changes are needed.

**Continuous diffusion of language states.** In preliminary experiments based on [Embedded Language Flows][elf], our approximately **800M-parameter continuous diffusion denoiser**, with a **121M-parameter encoder**, uses Qwen-3-derived latent states and achieves **62.47% GSM8K accuracy**, compared with **66.6%** reported for the 7B [TESS 2][tess] model after mathematics-specific fine-tuning, under different training settings. This provides an encouraging foundation for generating reasoning-relevant contextual states via continuous latent diffusion alongside visual components.

## 5. Demonstration, evaluation, and relevance to Sony

**The core demonstration generates two participants in a specified interaction, with their attributes and action roles correctly bound.** A creator enters a complete description; the system generates an image and provides a focused account of the component–module path supporting one requirement. Held-out combinations of participants, attributes, and roles demonstrate whether the computation transfers beyond familiar scenes. The operational goal is more reliable translation of instructions into assets and scene concepts for games, animation, and film. The accompanying mechanistic account addresses [Sony's interests][sony] in modality binding, generation dynamics, and causal interventions.

The masked-caption task diagnoses what an internal path contributes when particular information is absent. The creator demonstration separately establishes whether using that path helps when the complete prompt is available. The mechanistic claim concerns the identified conditional computation and its contribution to the full generator; it does not require explaining every parallel path in the pretrained backbone.

### Hypotheses, comparisons, and existing support

| Hypothesis | Focused comparison and outcome | Existing support |
|---|---|---|
| Task-trained activations carry useful information for generation. | Compare pretrained and task-trained representations on common conditional tasks; freeze each before denoising training. Measure semantic correctness on held-out examples. | Language-state diffusion supports feasibility; the task-trained visual extension is proposed. |
| Schedules should reflect conditional prediction requirements. | Fixed versus learned schedules, with the same representation and architecture and a trained denoiser in both. Measure joint prompt satisfaction and image quality. | LWD demonstrates benefits for two fixed representation groups. |
| Dependencies identify useful, reusable computations. | Ordinary versus dependency-guided module access/sharing at matched capacity. Measure component–module effects, restoration, and reuse across tasks. | Attention/MLP analysis and weight-sharing/replacement results provide partial support. |
| The integrated generator benefits from both choices. | Cross fixed/learned schedules with ordinary/dependency-guided modules within one fixed representation. Assess all prompt requirements together using generated states. | The full connection is the proposed contribution. |

**A bounded evaluation.** The main endpoint is the fraction of images satisfying all specified participant, attribute, and role requirements. Plan **200 held-out prompts with four noise realizations each per trained model**. The four main variants use two training seeds, with matched data, semantic supervision, trainable capacity, guidance, and sampling budget. An inference-only comparison with unmodified PixArt under the same prompts and sampling budget shows the overall benefit relative to the starting generator. Report paired differences with prompt-level bootstrap confidence intervals and variation across training seeds, alongside individual requirement scores and image quality. Count feature preparation and schedule/module selection in training cost. Representation-source comparisons run primarily in the smaller-model stage.

[GenEval][geneval] supplies a starting point for object and attribute assessment. Role correctness requires an independent role-aware evaluator, checked against two raters blinded to the method on a common 200-image subset per variant; human judgments take precedence where the automated role score is unreliable. The separate **100–300 matched mechanistic cases** use component swaps, module removal/restoration, equally sized control interventions, and checks for collateral changes. Useful progress requires an improvement in joint satisfaction supported by the paired comparison, without a material loss of visual quality, together with a specific, reproducible component–module effect. If a task needs broad access or separate predictors, identifying that boundary still informs reliable architecture choices.

Reference-based identity across scenes, multiple simultaneous activities, and video extend the same motivation. In video, an interaction's roles must remain consistent while object positions evolve; asynchronous diffusion already provides a relevant foundation. These extensions follow the bounded image demonstration when data and compute permit.

## 6. Execution, resources, and deliverables

### Models, data, and compute

Existing 10–20M-parameter diffusion-transformer pipelines provide the first setting for schedule and module studies, including the existing CelebA64, AAHQ, and STL-10 experiments. The text-to-image prototype uses the approximately **0.6B-parameter PixArt-Σ backbone**, beginning at 256 pixels and scaling to 512 pixels after the month-6 decision. Its VAE, native text encoder, and most backbone weights remain fixed. A semantic branch, interfaces, and selected conditional computations target **20–50M trainable parameters**. A separate **10–20M-parameter task model** on cached visual/language features supplies one task-trained representation alternative. Frozen backbone computation is included in the compute estimates.

We will select approximately **20,000 training images**, with a **50,000-image ceiling**, from [SWiG][swig] role annotations and [Visual Genome][visualgenome] objects, attributes, and relations. Handover is the preferred directional interaction, subject to annotation coverage. The broad corpus supports conditional training; 100–300 manually checked matched cases support the focused mechanistic study. Existing generators can supply controlled role reversals where natural pairs are unavailable, with generated examples checked for the intended requirements. Image-level splits keep captions, crops, and related synthetic variants together; selected attribute/role combinations are withheld while their individual elements remain represented in training.

We estimate a total workload of approximately **5,000 H100-80GB GPU-hours**, distributed across the studies below. One GPU-hour denotes one GPU used for one hour.

| Work | Planned allocation | GPU-hours |
|---|---|---:|
| Small-model studies | 12 runs × 1 GPU × 48 hours | 576 |
| Main generation comparisons | 8 runs × 4 GPUs × 72 hours | 2,304 |
| Feature preparation and task-model training | Combined allowance | 600 |
| Generation evaluation and mechanistic studies | Combined allowance | 900 |
| Integration and repeat-run reserve | Reserve | 620 |
| **Total** | | **5,000** |

Main runs target 5,000–20,000 updates at effective batch 64. At an assumed 10 seconds per update on four H100 GPUs, 20,000 updates take about 56 hours, leaving time within the 72-hour allowance for validation and checkpointing. These are planning estimates. Month-3 profiling will set the final update count and resolution while protecting the core comparisons and evaluation allowance.

### Milestones and minimum deliverable

| Period | Main work | Concrete output or decision |
|---|---|---|
| Months 1–3 | Curate the interaction subset; obtain fixed representations and one task-trained alternative; build conditional denoisers and the image interface. | Working baseline, component definitions, and measured runtime. Select the interaction family and confirm the planned runs. |
| Months 4–6 | Jointly learn schedules and the transformer; characterize dependencies; conduct initial component–module interventions. | A usable conditional generator and reproducible information-path effect. Scale to 512 pixels if data, integration, and runtime support it. |
| Months 7–9 | Train dependency-guided attention and shared prediction modules across conditional tasks; complete the focused comparisons. | Integrated generation pipeline and a measured account of when computation can be reused. |
| Months 10–12 | Complete held-out generation and intervention/restoration studies; consolidate the demonstration and documentation. | Creator demonstration, implementation, mechanistic case study, evaluation materials, and final report. |

The minimum deliverable combines **two participants, one interaction family, one pretrained representation configuration and one task-trained alternative, one sharing hypothesis, and one rigorous component–module case study**. Each representation is fixed before denoising training. If larger-scale integration is slower than anticipated, the complete study remains at the smaller resolution. Where sparse access or sharing is ineffective, retain broader access or separate predictors and document the conditions requiring them.

The PI will direct the mechanistic analysis and research design, with one graduate research assistant supported for 12 months. The project will provide three quarterly reports, a final research summary, and progress-questionnaire responses.

## References

1. Oquab et al. [DINOv2: Learning Robust Visual Features without Supervision][dino]. arXiv:2304.07193, 2023.
2. Pan et al. [Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion][sfd]. arXiv:2512.04926, 2025.
3. Bradley. [Local Mechanisms of Compositional Generalization in Conditional Diffusion][local]. arXiv:2509.16447, revised 2026.
4. Ge et al. [Vision-Language Binding in In-Context Image Generation][binding]. arXiv:2605.24624, 2026.
5. Rosu, Carin, and Cheng. [From Softmax to Score: Transformers Can Effectively Implement In-Context Denoising Steps][softmax]. NeurIPS, 2025.
6. Liu, Li, and Cheng. [Variational Trajectory Optimization of Anisotropic Diffusion Schedules][trajectory]. arXiv:2602.19512, 2026.
7. Qian and Cheng. [Learning When to Denoise: Optimizing Asynchronous Schedules for Latent Diffusion][schedule]. arXiv:2606.19662, 2026.
8. Larsen et al. [Autoencoding beyond pixels using a learned similarity metric][vaegan]. arXiv:1512.09300, 2015.
9. Xu et al. [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation][imagereward]. arXiv:2304.05977, 2023.
10. Chen et al. [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion][diffusionforcing]. arXiv:2407.01392, 2024.
11. Baade et al. [Latent Forcing: Reordering the Diffusion Trajectory for Pixel-Space Image Generation][latentforcing]. arXiv:2602.11401, 2026.
12. Ghosh, Hajishirzi, and Schmidt. [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment][geneval]. arXiv:2310.11513, 2023.
13. Chen et al. [PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation][pixart]. Implementation and released checkpoints.
14. Pratt et al. [Grounded Situation Recognition (SWiG)][swig]. ECCV, 2020; project and annotations.
15. Krishna et al. [Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations][visualgenome]. IJCV, 2017.
16. Hu et al. [ELF: Embedded Language Flows][elf]. arXiv:2605.10938, 2026.
17. Tae et al. [TESS 2: A Large-Scale Generalist Diffusion Language Model][tess]. arXiv:2502.13917, 2025.
18. Huang et al. [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion][selfforcing]. NeurIPS, 2025.
19. Heimersheim and Nanda. [How to Use and Interpret Activation Patching][patching]. arXiv:2404.15255, 2024.

**Unpublished preliminary materials.** The PI's working manuscript, *A Mechanistic View of Diffusion Transformers as Adaptive Patchwise Denoisers*, and associated sharing/transfer experiment reports; continuous diffusion language-model results supplied by the PI. These support the preliminary findings in Section 4.

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

**Budget basis.** Amounts follow the supplied institutional budget, rounded to whole dollars. PI salary uses a nine-month salary basis of USD 143,000. The modified total direct cost base is USD 79,979, consisting of salaries/stipend and fringe benefits; tuition remission is excluded. Applying the 61.5% indirect cost rate gives USD 49,187. No funding is requested for equipment, supplies, travel, other expenses, or subcontracts. The total includes indirect costs and is within the Focused Research Award limit of USD 150,000.


[dino]: https://arxiv.org/abs/2304.07193
[sfd]: https://arxiv.org/abs/2512.04926
[local]: https://arxiv.org/abs/2509.16447
[binding]: https://arxiv.org/abs/2605.24624
[softmax]: https://papers.nips.cc/paper_files/paper/2025/hash/d27af2bebeab8a3e6f3848fc71736235-Abstract-Conference.html
[trajectory]: https://arxiv.org/abs/2602.19512
[schedule]: https://arxiv.org/abs/2606.19662
[sony]: https://www.sony.com/en/SonyInfo/research-award-program/
[vaegan]: https://arxiv.org/abs/1512.09300
[imagereward]: https://arxiv.org/abs/2304.05977
[diffusionforcing]: https://arxiv.org/abs/2407.01392
[latentforcing]: https://arxiv.org/abs/2602.11401
[geneval]: https://arxiv.org/abs/2310.11513
[pixart]: https://github.com/PixArt-alpha/PixArt-sigma
[swig]: https://prior.allenai.org/projects/gsr
[visualgenome]: https://arxiv.org/abs/1602.07332
[elf]: https://arxiv.org/abs/2605.10938
[tess]: https://arxiv.org/html/2502.13917v1#S4.T3
[selfforcing]: https://arxiv.org/abs/2506.08009
[patching]: https://arxiv.org/abs/2404.15255
