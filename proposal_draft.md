# How Representations and Weights Shape Multimodal Generation

**Principal investigator:** Xiang Cheng, Department of Electrical and Computer Engineering, Duke University  
**Focused Research Theme:** Internal Mechanisms of Multimodal Generative Models for Content Creation  
**Project duration:** 12 months  
**PI contact:** [Email to insert]; [Phone with country code to insert]

*Drafting note — remove before submission: This draft uses only existing results and published work as feasibility evidence. The research experiments proposed below are work for the award period. Submission figures will use existing results or explanatory drawings; no new training, sampling, ablation, or evaluation is required.*

## Abstract

We will jointly learn multimodal representations, denoising procedures, and the transformer computations that connect them. Representations will come from hidden activations of pretrained models and models trained on task or reward signals relevant to generation. We will divide these representations into components and jointly learn their denoising schedules with the transformer. Conditional denoising losses will reveal which components help predict others; module interventions will explain which computations implement these interactions and can be shared across tasks. The PI's existing and ongoing work on diffusion language models, learned asynchronous denoising schedules, and mechanistic analysis of transformer components provides the foundation. The one-year project will deliver an end-to-end pipeline for learning semantically meaningful, decomposable representations and their conditional dependencies, together with a mechanistic account of how the diffusion transformer uses them. The core demonstration will generate scenes that jointly satisfy a creator's requirements for participants, attributes, and interactions from text and reference images.

## 1. The proposed contribution

**We will jointly learn representation components, their denoising process, and the transformer computations used to predict them.** Understanding which information a computation uses, when it is needed, and when it can be shared will provide interpretable ways to control complex multimodal generation. Continuous diffusion is a central framework for [image][sfd] and [video generation][diffusionforcing]. Both text-to-image and text-to-video generation motivate the proposed method: representation components can span semantic states, image tokens, and video frames. We use text-to-image examples throughout to explain the ideas more simply.

Consider a creator's request: “A woman in a red coat hands a blue cup to a man in a green sweater. A child beside them reads a book.” Successful generation must bind clothing to the correct people, distinguish the giver from the receiver, and realize the child's separate activity. We will learn representations, denoising schedules, and transformer computations that jointly support these requirements. Participants and relationships are illustrative meanings; the learned decomposition need not assign one component to each human-named concept.

### Central hypothesis: learned dependencies can organize generation and transformer computation

**What information does the model need to generate each component accurately?** We hypothesize that a useful multimodal representation can be divided into components whose conditional dependencies guide both their denoising schedules and the transformer computations used to predict them. Learning these elements together will provide more precise control over how complex semantic requirements are realized in an image.

In the handover scene, generating the visual interaction requires information about the participants and their respective roles; other predictions may depend on different subsets of the scene description. These requirements need not correspond to separate, human-named coordinates. We will learn component vectors that capture information useful for these conditional predictions, starting from pretrained representations and downstream tasks.

**Denoising provides a way to discover these dependencies.** Suppose component $X$ describes who gives the cup to whom, while component $Y$ describes the positions of the hands and cup. If reducing noise in $X$ improves prediction of $Y$, then $X$ supplies information useful for generating $Y$. Generating $X$ earlier can therefore help $Y$, but it may make $X$ harder to predict because the visual information in $Y$ is still noisy. Jointly learning the transformer and component schedules balances this tradeoff. Repeating the comparison as training progresses shows which dependencies the model learns to use.

**Transformer modules implement these conditional predictions.** Attention selects and combines information across components, while multilayer perceptrons (MLPs) transform that information into predictions. Our preliminary mechanistic results motivate learning computations that can be shared across conditional tasks, alongside computations that specialize to particular inputs, targets, or noise levels. Module interventions will explain which computations implement the learned dependencies and when they remain reusable. This connects the representation's organization to concrete choices about attention access, parameter sharing, and module activity during generation.

### Representation components, noise schedules, and transformer modules

**Representation components.** Let $x$ be a training target: an image or video together with its available description. An encoder $E$ represents it as a matrix $Z$ of $N$ token vectors, each with $d$ coordinates. Flattening $Z$ gives $z\in\mathbb R^D$, where $D=Nd$. We divide this representation into $K$ components. For each component index $k\in\{1,\ldots,K\}$, an extractor $P_k$ produces a vector $s_k$ with $d_k$ coordinates:

```math
Z=E(x)\in\mathbb R^{N\times d},\qquad z=\operatorname{vec}(Z),\qquad
P_k:\mathbb R^D\longrightarrow\mathbb R^{d_k},\qquad s_k=P_k(z).
```

An extractor can select tokens or coordinates, apply a low-dimensional projection, or use an MLP or attention to combine information across tokens. If it acts on one token, its input has $d$ coordinates; an extractor acting across the full representation has $D$ input coordinates. Encoders and extractors supply clean training targets. The diffusion model learns to generate the corresponding component vectors from noise.

**Component schedules.** Let $t\in[0,1]$ denote generation time, and let $\tau_k(t;\phi)$ describe component $k$'s progress from noise to data. Here $\phi$ contains the learnable schedule parameters. Draw independent standard Gaussian noise $\varepsilon_k\in\mathbb R^{d_k}$. During training, form a partially noisy component by interpolation:

```math
s_{k,t}=(1-\tau_k(t;\phi))\varepsilon_k+\tau_k(t;\phi)s_k,
\qquad \tau_k(0;\phi)=0,\quad\tau_k(1;\phi)=1.
```

Each schedule increases monotonically: progress $0$ gives pure noise and progress $1$ gives the clean target. Different components can progress at different rates, with overlapping or crossing schedules. This determines how much information is available in each component when the transformer predicts the others. Section 2.2 explains how we learn these schedules with the model.

**Transformer modules.** Denote all transformer parameters by $\theta$. A module, indexed by $m$, is an identifiable computation with parameters $\theta_m$: an attention head with its query/key/value/output projections, an MLP block, or a specified group of rows or columns in those projections. We will identify which components it uses and predicts, and how its role varies with noise levels. Here *locality* means dependence on a small set of representation components, potentially spanning distant tokens, image regions, or modalities. Module access and sharing patterns connect this dependence to parameter organization.

Section 2.1 learns the representation and extractors; Section 2.2 jointly learns component dependencies, schedules, and the transformer; Section 2.3 develops shared or specialized computations across the resulting conditional tasks. Figure 1 illustrates these connections.

![Figure 1: fig1 mechanism overview](figures/fig1_mechanism_overview.png)

**Figure 1. Representation, denoising algorithm, and transformer architecture inform one another.** Colored coordinates illustrate components extracted from participant tokens, a relation token $R$, and visual tokens $V$. The example assigns exact participants and giver–receiver roles; these assignments illustrate the idea, rather than prescribe the learned representation. Component schedules control when useful information becomes available; noise-dependent prediction errors reveal dependencies that refine the components. The architecture uses selected components as inputs and targets, sharing attention across tasks while specializing prediction modules. The two illustrated tasks predict visual states $V$ given roles $R$, or roles $R$ given visual states $V$, with participant information in both. Schedules, coordinate assignments, and module roles are illustrative.

## 2. Research approach: representations, denoising, and transformer computations

**Representation learning determines what the vectors encode; joint denoising training determines how those components help predict one another and when to generate them.** Downstream tasks and reconstruction guide the representation. Component-wise flow losses jointly train the transformer and schedules and expose useful dependencies. Section 2.3 studies the computations that implement these conditional predictions.

### 2.1 Obtain representations and learn component extractors

**First, use hidden activations from pretrained models as latent representations.** Frozen [DINOv2][dino] activations and Qwen contextual states provide visual and language representations. We will train diffusion to generate selected activations alongside image latents. This approach uses semantic information already learned by powerful visual and language models, and connects directly to our preliminary continuous diffusion language-model results.

**Second, use activations from models trained for the desired task.** We will train or adapt discriminators, reward/preference models, and visual-language understanding models, then use their hidden activations as latent representations for generation. Because these models learn from signals closer to our goals, their activations may encode more relevant semantic distinctions. For example, a prompt-image compatibility model trained to distinguish correct giver–receiver assignments from reversed ones should learn features useful for representing that relationship. The diffusion model would generate these features together with visual states. [Discriminator features][vaegan] and [ImageReward][imagereward] motivate obtaining representations from such task-trained models; [representation alignment][repa] demonstrates the value of external features for diffusion training.

**Learn how to divide the chosen representation.** Given the resulting token vectors $Z$, we will learn extractors $P_k$ that form component vectors. We will begin with token groups and disjoint groups of learned projected coordinates. Reconstruction preserves the information needed to generate an image, while semantic prediction tasks encourage the components to retain useful distinctions. The denoising losses in Section 2.2 identify which groups of information help predict one another and guide further refinement of the division. Residual image features preserve details that the semantic components do not capture. Small MLP or attention extractors are a later extension beyond linear projections.

### 2.2 Jointly learn denoising order, component dependencies, and the transformer

**Denoising order and dependence structure are connected through conditional prediction.** A dependency describes which components supply useful information for predicting another; a schedule determines when that information becomes available. Both depend on what the transformer learns. We will refine the component divisions and schedules together with the model. Here *denoising* means predicting a clean component or its displacement from noise; *generation* integrates such predictions from noise to a complete sample.

**Set up the prediction task.** For each training example, $x$ contains the target scene and $c$ contains the supplied prompt and reference features. The clean components $s_k=P_k(\operatorname{vec}(E(x)))$ are training targets; they are not supplied at generation time. For an arbitrary progress vector $\mathbf u=(u_1,\ldots,u_K)\in[0,1]^K$, define the noisy inputs
```math
\widetilde s_k(u_k)=(1-u_k)\varepsilon_k+u_k s_k,\qquad
\widetilde{\mathbf s}(\mathbf u)
=\bigl(\widetilde s_1(u_1),\ldots,\widetilde s_K(u_K)\bigr).
```
The transformer receives these inputs, their progress levels, and $c$. Its output $f_{\theta,k}(\widetilde{\mathbf s}(\mathbf u),\mathbf u,c)\in\mathbb R^{d_k}$ predicts $v_k=s_k-\varepsilon_k$, the displacement along component $k$'s noise-to-data interpolation. Training this displacement predictor by squared error is the flow-matching objective used here.

**Learn component schedules and the transformer together.** Begin with the component vectors from Section 2.1 and simple monotone schedules. At each update, sample a training pair $(x,c)$, a time $t$ uniformly from $[0,1]$, and independent Gaussian noises $\varepsilon_1,\ldots,\varepsilon_K$. Set $\mathbf u=\boldsymbol\tau(t;\phi)=(\tau_1(t;\phi),\ldots,\tau_K(t;\phi))$. Write $f_k$ for the corresponding transformer output and $\dot\tau_k=\partial\tau_k(t;\phi)/\partial t$ for component $k$'s progress rate.

The training criterion combines denoising error with a penalty on large velocities along the generation trajectory. The coefficient $\lambda\geq0$ controls the penalty; $\|\cdot\|_2$ is the Euclidean norm, and division by $d_k$ averages over the component's coordinates. The resulting training criterion is

```math
\begin{aligned}
\mathcal J_{\mathrm{learn}}(\theta,\phi)
&=\mathbb E\!\left[
\sum_{k=1}^{K}\frac{1}{d_k}
\left\{
\dot\tau_k\,\|f_k-v_k\|_2^2
+\lambda\|\dot\tau_k f_k\|_2^2
\right\}\right],\\
f_k&=f_{\theta,k}
\bigl(\widetilde{\mathbf s}(\boldsymbol\tau(t;\phi)),
\boldsymbol\tau(t;\phi),c\bigr).
\end{aligned}
```

The expectation averages over training pairs, sampled times, and Gaussian noises. The first term trains the transformer to denoise each component, weighted by its progress rate. The second term penalizes large generation-time velocities $\dot\tau_k f_k$. The schedule parameters affect both terms: they change the information available to each prediction and the rate at which that component develops. This variational approach extends our [Learning When to Denoise (LWD)][schedule] and [trajectory optimization][trajectory] methods to multiple learned components.

**Use the same noise configurations to learn dependencies.** At selected checkpoints, choose a target component $i$ and another component $j\ne i$. For fixed progress levels $\mathbf u$, let $L_i(\mathbf u)$ be the expected squared error per coordinate in predicting $s_i-\varepsilon_i$. Compare two values of $u_j$: a noisier value $a$ and a cleaner value $b$, where $0\leq a<b\leq1$. The notation $\mathbf u_{j\leftarrow a}$ means replace only the $j$th entry of $\mathbf u$ by $a$. Define
```math
\begin{aligned}
L_i(\mathbf u)
&=\frac{1}{d_i}\mathbb E
\left\|f_{\theta,i}(\widetilde{\mathbf s}(\mathbf u),\mathbf u,c)
-(s_i-\varepsilon_i)\right\|_2^2,\\
D_{j\to i}(\mathbf u;a,b)
&=L_i(\mathbf u_{j\leftarrow a})
-L_i(\mathbf u_{j\leftarrow b}).
\end{aligned}
```
These expectations average over training pairs and Gaussian noise at the current transformer checkpoint. A positive $D_{j\to i}$ means cleaner information in $j$ improves denoising of $i$. Each paired comparison uses the same examples, noise draws, other progress levels, and transformer weights. For visual-to-semantic tasks, the prompt omits the target role labels so prediction requires visual information. Varying groups of components captures dependencies missed by pairwise measurements.

**Train the conditional predictions needed to measure dependencies.** A single schedule exposes the transformer to only some combinations of noise levels. We will also train on other combinations: for example, relatively clear role information with noisy visual states, and relatively clear visual states with noisy roles. For each combination, the same flow-matching loss trains prediction of the noisy target components. This makes the comparisons above meaningful for a model that has learned to use different patterns of available information.

**Use the measured dependencies to organize the representation and its computations.** For each target component, identify the other components that most improve its denoising. For example, predicting hand–cup positions may benefit from participant and role information but require little information about the child reading nearby. We will use these measurements to select inputs for the conditional modules in Section 2.3. If a component mixes information needed by different tasks, we will refine its projection or token grouping while preserving semantic prediction and reconstruction. We then continue joint transformer and schedule training, because changing the component division changes the prediction problems. This connects representation learning, dependency measurement, and generation through the same denoising objective.

Once the component division and schedules are chosen, we train the final denoiser with the prediction loss alone. The velocity penalty helps select the generation path; removing it lets the final model fit the denoising targets directly. We remeasure dependencies for this final model.

Representation and schedule choices also shape what the transformer learns. If $\psi$ denotes extractor parameters and $\eta>0$ the learning rate, an update has the form
```math
\theta^+=\theta-\eta\nabla_\theta\mathcal J(\theta,\phi;\psi),
\qquad s_k=P_{k,\psi}(\operatorname{vec}(E(x))).
```
Here $\theta^+$ denotes the updated weights and $\mathcal J$ the training criterion. Changing $\psi$ changes the targets; changing $\phi$ changes the noisy inputs and their weighting.

**Generate with the learned schedules.** Let $\widehat{\mathbf s}(t)=(\widehat s_1(t),\ldots,\widehat s_K(t))$ denote generated component states, as distinct from noisy training targets. Starting from Gaussian noise, integrate
```math
\frac{d\widehat s_k(t)}{dt}
=\dot\tau_k(t;\phi)\,
f_{\theta,k}\bigl(\widehat{\mathbf s}(t),\boldsymbol\tau(t;\phi),c\bigr),
\qquad \widehat s_k(0)=\varepsilon_k.
```
The progress-rate factor converts the predicted displacement per unit component progress into a velocity per unit generation time. The final component states are reconstructed into image latents and decoded. Supplied prompt and reference features remain available as conditions throughout.

![Figure 2: conditional information, overlapping denoising schedules, and a shared transformer computation](figures/fig2_recovery_schedule.png)

**Figure 2. Information availability connects denoising order and transformer computation.** (A) The same noisy visual target is predicted with uncertain or clearer interaction roles, keeping participant information and target noise fixed. The conditioning prompt omits these roles. (B) Schedules learned jointly with the transformer balance supplying information early against predicting that component from less-developed context. Progress runs from noise (0) to clean states (1). (C) An attention–MLP path combines component inputs, with its contribution gated by their progress levels; input/output projections permit reuse across conditional tasks. Component meanings, schedules, and module roles illustrate proposed mechanisms; the shared backbone is omitted.

### 2.3 Learn shared and specialized transformer computations

The jointly learned order and dependencies define conditional prediction problems for the transformer. We will study which computations are needed at different component noise levels, which can be shared across tasks, and which must specialize.

**Localize computations as well as generation stages.** Figure 2 illustrates visual denoising that can use participant and interaction information. We will identify attention and MLP computations that carry this information into selected predictions, then control their contributions according to which inputs are available and how noisy they are. Reversing the conditional task—for example, inferring roles from visual information—may reuse prediction computations with different input and output maps. Schedules may overlap or cross; module activity therefore depends on the full progress vector $\mathbf u$, rather than a fixed global time interval.

For a module indexed by $m$, let $I_m\subseteq\{1,\ldots,K\}$ list the input components it can access. The collection $\widetilde{\mathbf s}_{I_m}(\mathbf u)$ contains just those noisy component vectors. Let $h_{\theta_m}$ be a computation that produces a velocity contribution for the selected target components, and let $g_m(\mathbf u)\in[0,1]$ be a scalar gate controlling how strongly that contribution is used. The gated contribution is

```math
g_m(\mathbf u)\,
h_{\theta_m}\bigl(\widetilde{\mathbf s}_{I_m}(\mathbf u),\mathbf u,c\bigr),
\qquad \mathbf u=\boldsymbol\tau(t;\phi).
```

Initially, a few gates select coarse configurations such as a relatively clean condition and a noisy target. We will then learn smooth gates, with their parameters included in $\theta$, under the same denoising objective used to learn the schedules. Shared backbone computations can remain active across all configurations. The constructions below specify the restricted modules' inputs, target outputs, and parameter-sharing patterns.


**Use diverse conditional prediction tasks to learn composition.** Tasks include predicting visual interaction features given participant and role components, or predicting roles given participant and visual components, with varying noise on the conditions and targets. Across training examples, we will vary the participants, relationships, and their combinations. Thus the model encounters both different semantic compositions and different uses of the same representation. These tasks encourage it to reuse a useful computation whenever the relevant information recurs.

The initial construction will use a shared transformer backbone and a small family of attention and MLP modules. A task specifies which components supply conditions and which are prediction targets; these masks are supplied to the module selector, and the noise vector specifies how reliable each input is. The flow loss is applied to the target components. The first restricted branches read selected component vectors before unrestricted backbone mixing and add only to the chosen targets' velocity predictions. Their input/output projections therefore specify access for the added computation. We will keep the learned representation interface fixed while comparing three concrete designs:

- **Sparse attention across representations.** Use the dependencies measured in Section 2.2 to restrict the component groups accessible to selected attention heads. Token masks or input projections implement these restrictions. This lets a head learn a cross-representation rule from a limited set of relevant inputs.
- **Shared prediction computations.** Share an MLP prediction bank across conditional tasks, with task-specific input and output projections. This asks whether a common learned predictor can be used through different information-selection and output mappings.
- **Specialized prediction computations.** Switch selected MLPs or projections according to which components are conditions and targets. For example, predicting visual features from interaction features may require a different predictor from inferring an interaction from visual features, while retaining common computations elsewhere.

These are alternative sharing patterns within one implementation. We will begin by freezing the pretrained backbone and training the small added modules, then release selected backbone parameters where conditional prediction requires it. In the small-model studies, we can also replace or share existing MLP blocks directly. The preliminary attention/MLP interpretation motivates these choices; the experiments will identify the useful division of computation. [Compositional Attention][compositional] provides a related separation of selection and retrieval.

**Explain when computations are needed and reusable.** At fixed noisy inputs, remove or replace a head or MLP contribution and measure which component predictions change. Repeat across noise configurations and conditional tasks, then trace selected interventions through the remaining generation trajectory. This reveals which computations become useful when a condition is clearer and which transfer to a new target or participant–relation combination. Replacing a candidate module with one serving an unrelated task checks its assigned role. The analysis includes the shared backbone and gated additions, and its findings guide the next refinement of schedules and module access.

A tractable analysis begins with squared-error denoising: separate the prediction error caused by removing useful inputs from the error of the learned predictor using retained inputs. Linear models with Gaussian features provide an initial setting for identifying when conditional tasks can share a predictor or require different projections. The analysis and module interventions will guide which restrictions to retain or relax. The resulting architecture will support controllable generation through reusable computations, without requiring each human-named concept to occupy a unique module.

![Figure 3: fig3 component reuse](figures/fig3_component_reuse.png)

**Figure 3. Existing evidence of selective component reuse.** Restoring source query/key weights while retaining adapted value/output projections and MLPs preserves much of the AAHQ adaptation; the larger shift to STL-10 requires more coordinated changes. The table reports the fraction of the joint model's adaptation recovered, relative to the source model in DINO feature space. Each image row uses the same noise input across its three columns. These results motivate identifying which transformer computations remain reusable as conditional prediction tasks change.

## 3. Differentiation from the current state of the art

**The proposed advance is to learn representation dependencies, denoising order, and the transformer computations that implement them together.** This will connect semantic requirements to concrete choices about when information becomes available and which parameters combine it. Two features distinguish our research from existing approaches.

**Use ordered denoising to discover and refine dependency structure.** [SFD][sfd] and [SeFi-Image][sefi] demonstrate semantic-first generation; [Diffusion Forcing][diffusionforcing] supports independent token noise levels; [Latent Forcing][latentforcing] analyzes how cleaner states in one representation improve prediction in another. Our [LWD][schedule] learns schedules for a given pair of representations. We will extend these foundations to a learned division into multiple components. Joint transformer and schedule training determines useful denoising orders; comparisons across noise levels identify which components help predict one another; these measurements refine the division and permitted inputs before joint training resumes. **Generation order becomes a means of learning the representation's dependence structure**, alongside learning a model that uses it. Our variational schedule-learning approach makes this practical without prescribing a semantic-first split or a fixed sequence of components.

**Translate those dependencies into interpretable transformer computations.** [Local Mechanisms][local] relates sparse score dependencies to composition, while [Vision-Language Binding][binding] traces the influence of reference information. We will connect dependencies among learned semantic and visual components to the functions of attention and MLP modules at different noise levels. This connection leads directly to an architecture: sparse attention selects relevant components, shared or specialized MLPs predict chosen targets, and noise-dependent gates activate computations when their inputs become useful. Training across varied conditional tasks will develop reusable composition rules. Module removal, replacement, and sharing will explain which computations implement them. **The intended control is grounded in identifiable information paths and parameter roles**, making it possible to understand how a requested relation is realized and which computations can be reused for a new combination.

**Existing results provide the foundation for this proposed synthesis.** Our preliminary **continuous diffusion language model**, based on [Embedded Language Flows][elf], has **795,205,824 parameters** and achieves **61.94% GSM8K accuracy** by generating projected Qwen answer states. A separate frozen Qwen3-4B encoder supplies question states and is excluded from that count. Figure 4 provides published Llama-3 and LLaDA reference points; [TESS 2][tess] reports 66.6% with continuous simplex diffusion after mathematics-specific fine-tuning. Training and evaluation settings differ. **Our preliminary result establishes a practical foundation for generating reasoning-relevant contextual states via continuous latent diffusion.** We will extend this capability to joint semantic and visual generation.

Our [LWD][schedule] achieves **state-of-the-art FID 1.02** within its comparison of 675M-parameter image generators, improving on **1.04** for the 1B-parameter SFD-XXL. It reaches FID **1.05 in 200 epochs**, versus **1.06 in 800 epochs** for SFD-XL. Against REPA, LWD achieves better unguided FID (**4.93 versus 5.84**) with **33 times fewer main-training updates** (120,000 versus 4 million), plus a 10,000-update schedule-learning phase. These results establish the practical value of learning the generation process.

Our [attention-as-denoising analysis][softmax] and working manuscript, *A Mechanistic View of Diffusion Transformers as Adaptive Patchwise Denoisers*, support attention-based aggregation and reusable prediction knowledge in MLPs. Preliminary sharing experiments improve CelebA64 FID from **17.574 to 14.304** at approximately **10.2M parameters**; Figure 3 illustrates selective component reuse. These results and implementations support the proposed joint study.

![Figure 4: continuous latent diffusion language reasoning](figures/fig4_existing_feasibility.png)

**Figure 4. Continuous latent diffusion retains substantial reasoning capability.** Our 795M-parameter diffusion model generates Qwen-derived answer states; its separate frozen Qwen3-4B question encoder is excluded from that count. The chart gives published autoregressive Llama-3 and discrete-diffusion LLaDA Base results [from the LLaDA paper][llada] as context. Training and evaluation settings differ.

## 4. Creator demonstration and relevance to Sony

**Generate a scene that satisfies interacting semantic requirements.** A creator supplies character references and a description of the handover scene. The model generates an image with the requested participants, clothing, giver–receiver relationship, and independent activity. This workflow brings together component learning, conditional dependencies, asynchronous denoising, and shared or specialized transformer computations.

The demonstration will pair generated images with an explanation of how the representation components and attention or MLP computations realize the requested information. Across prompts, we will vary the participants, attributes, and relationships to show which computations remain reusable in new combinations. Reference identity, participant roles, attribute binding, and independent activities will be assessed separately, alongside image quality and joint satisfaction of the prompt. Independent evaluators and human assessment will complement the training signals; [GenEval][geneval] provides a starting point for prompt-grounded assessment, extended to action roles.

The practical goal is more reliable translation of a creator's instructions and references into visual content, supporting asset development for games, animation, and film. Understanding where and when the model combines semantic information will help address failures such as assigning an attribute to the wrong character or reversing an action's roles. The accompanying account of what the modules compute addresses [Sony's stated interests][sony] in modality binding, generation dynamics, and causal interventions. Explicit conditional prediction modules make these contributions easier to inspect through removal and replacement.

**The same formulation extends naturally to video.** Continuous diffusion generates both images and video, making control over its internal process relevant to Sony's animation and film workflows. [Diffusion Forcing][diffusionforcing] demonstrates that varying noise levels across video frames can support stable generation beyond the training horizon. Our proposed representation goes beyond frame groups to include semantic and visual components across time. In a video of the handover, participant identity and giver–receiver roles must remain consistent while hand positions and cup motion evolve. Learning which components need information from others can guide their relative denoising progress and the transformer computations that connect them. Image generation provides the initial implementation of this general approach.

The initial study will use a bounded vocabulary of annotated interactions and participant roles. Existing generators can supplement annotated data with scenes from systematically varied prompts; generated examples will be checked against the requested identities, roles, and attributes. Some participant–relation combinations will be held out so that successful generation requires the computations to transfer. These are award-period resources; the proposal uses only existing results as preliminary evidence.

## 5. Work plan and deliverables

**The core deliverable is an end-to-end pipeline that learns semantic representations and their component divisions, identifies conditional dependencies through denoising, and explains the transformer computations that use them.** Learned schedules and conditional modules will turn this organization into a generation procedure. The core demonstration will generate scenes satisfying interacting requirements from text and reference images, with an account of how the relevant information enters the visual predictions.

### Concrete execution

We will begin with our existing 10–20M-parameter diffusion Transformers and semantic/visual diffusion pipeline, reusing checkpoints, component-replacement code, and schedule-learning implementations. For the text-to-image prototype, we will use the released **PixArt-Σ 512-pixel model**, with an approximately 0.6B-parameter denoising Transformer and available training/adaptation code. Its native T5 text conditioning and VAE will remain in place. [PixArt-Σ][pixart]

The initial representation combines image latents with frozen DINOv2 features and Qwen3-4B-Instruct-2507 states from our continuous diffusion language-model pipeline. We will learn four to eight component groups using semantic tasks, reconstruction, and denoising losses. A second representation source will be hidden activations from a model trained to assess prompt-image compatibility, particularly participant roles and attribute binding. The groups begin as disjoint projected coordinates or token groups. Small projections, a semantic denoising branch, and trainable attention interfaces connect the generated features to image latents. Reference-image features enter as clean conditioning; generated component states carry the features of the requested scene. Qwen targets are extracted from annotated training scene descriptions containing participant and role information. T5 receives the supplied prompt, which can specify only a subset of that information; conditional tasks withhold the attributes or roles being inferred. At inference, the description-derived target states are generated; reference features and prompt conditioning remain available as inputs.

The first prototype will freeze the pretrained image backbone and train the projections, semantic branch, and added conditional modules. Caching encoder outputs limits repeated feature extraction. We will release selected backbone parameters only where the restricted additions cannot support the required denoising tasks. The same fixed conditioning will be used when comparing module interventions, so changes can be traced through the proposed representation path.

| Resource | Role in the initial implementation |
|---|---|
| Existing CelebA64, AAHQ, and STL-10 experiment pipelines | Study module reuse, sharing, and noise-dependent activity in small models before applying the same construction to text-to-image generation. |
| [SWiG grounded situation recognition][swig] | Activity labels, participant-role labels, and entity locations for task-supervised component learning and conditional denoising. |
| [Visual Genome][visualgenome] | Objects, attributes, and relationships for training varied semantic compositions and checking attribute and relation binding. |

We will select a bounded set of activities and relations and form scene descriptions from their annotations. The first conditional tasks will predict selected semantic components from visual components and vice versa, followed by joint generation at varying noise levels. The first generation cases will involve two interacting participants and clearly specified attributes; an independent activity, such as the child reading, introduces additional requirements to satisfy simultaneously. Reference crops and scene annotations will be checked for identity and role consistency. Success on the small-model studies will guide the sharing patterns used in the text-to-image prototype.

The budget requests 1.5 months of PI salary and 12 months of graduate research assistant support. The research uses existing pipelines and pretrained models; no separate computing costs are requested. All experiments below are proposed for the 12-month award period.

| Period | Main work | Concrete output |
|---|---|---|
| Months 1–3 | Obtain pretrained activations; train or adapt a task model and extract its activations; learn initial component projections and conditional denoisers. | A fixed initial component interface and conditional prediction tasks; baseline generation and denoising measurements. |
| Months 4–6 | Jointly train transformer and schedules; measure dependencies at selected checkpoints and refine component divisions and access. Analyze module activity across noise configurations. | Learned denoising order and an evolving dependence structure, with an initial account of the computations used. |
| Months 7–9 | Train sparse attention and shared or specialized MLP computations across conditional tasks; revisit dependencies and schedules after module changes. | A pipeline combining learned components, schedules, and conditional transformer modules, with an account of their roles. |
| Months 10–12 | Consolidate the text-and-reference image workflow, assess prompt fidelity and transfer to new combinations, and document the internal computations. | Creator demonstration, implementation, mechanistic findings, reproducible evaluation materials, and final report. |

We will retain the simplest sharing pattern that supports the conditional tasks. Where useful dependencies span several components, attention access can be expanded and computations shared at a coarser level. The common implementation allows useful representation or schedule improvements to be retained even when a particular module-sharing restriction is ineffective.

Controllable generation from text and reference images defines the award-period application. Selective revision of existing images, meta-learning, model merging, generation of parameter updates from demonstrations, and audio generation are possible future uses of the learned organization. They are outside the committed deliverables. We will provide three quarterly reports, a final research summary, and the required progress-questionnaire responses.

## References

1. Oquab et al. [DINOv2: Learning Robust Visual Features without Supervision][dino]. arXiv:2304.07193, 2023.
2. Bachmann et al. [MultiMAE: Multi-modal Multi-task Masked Autoencoders][multimae]. arXiv:2204.01678, 2022.
3. Bao et al. [One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale][unidiffuser]. arXiv:2303.06555, 2023.
4. Pan et al. [Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion][sfd]. arXiv:2512.04926, 2025.
5. SeFi-Team. [SeFi-Image: A Text-to-Image Foundation Model with Semantic-First Diffusion][sefi]. arXiv:2606.22568, 2026.
6. Bradley. [Local Mechanisms of Compositional Generalization in Conditional Diffusion][local]. arXiv:2509.16447, revised 2026.
7. Mittal et al. [Compositional Attention: Disentangling Search and Retrieval][compositional]. ICLR, 2022.
8. Ge et al. [Vision-Language Binding in In-Context Image Generation][binding]. arXiv:2605.24624, 2026.
9. Rosu, Carin, and Cheng. [From Softmax to Score: Transformers Can Effectively Implement In-Context Denoising Steps][softmax]. NeurIPS, 2025.
10. Liu, Li, and Cheng. [Variational Trajectory Optimization of Anisotropic Diffusion Schedules][trajectory]. arXiv:2602.19512, 2026.
11. Qian and Cheng. [Learning When to Denoise: Optimizing Asynchronous Schedules for Latent Diffusion][schedule]. arXiv:2606.19662, 2026.
12. Larsen et al. [Autoencoding beyond pixels using a learned similarity metric][vaegan]. arXiv:1512.09300, 2015.
13. Xu et al. [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation][imagereward]. arXiv:2304.05977, 2023.
14. Yu et al. [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think][repa]. ICLR, 2025.
15. Chen et al. [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion][diffusionforcing]. arXiv:2407.01392, 2024.
16. Baade et al. [Latent Forcing: Reordering the Diffusion Trajectory for Pixel-Space Image Generation][latentforcing]. arXiv:2602.11401, 2026.
17. Ghosh, Hajishirzi, and Schmidt. [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment][geneval]. arXiv:2310.11513, 2023.
18. Ma et al. [Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning][deme]. CVPR, 2025.
19. Chen et al. [PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation][pixart]. Implementation and released checkpoints.
20. Pratt et al. [Grounded Situation Recognition (SWiG)][swig]. ECCV, 2020; project and annotations.
21. Krishna et al. [Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations][visualgenome]. IJCV, 2017.
22. Hu et al. [ELF: Embedded Language Flows][elf]. arXiv:2605.10938, 2026.
23. Nie et al. [Large Language Diffusion Models][llada]. arXiv:2502.09992v3, 2025.
24. Tae et al. [TESS 2: A Large-Scale Generalist Diffusion Language Model][tess]. arXiv:2502.13917, 2025.

**Unpublished preliminary materials.** *A Mechanistic View of Diffusion Transformers as Adaptive Patchwise Denoisers*, working manuscript; associated Experiment 1 and Experiment 3 reports; continuous diffusion language-model summary and accuracy records, 2026. These materials support the explicitly labeled preliminary results in Section 3.

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
- Review the bibliography and complete the authorship/citation form for unpublished preliminary materials. Detailed experimental settings and figure provenance are retained in figures/README.md.
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

[elf]: https://arxiv.org/abs/2605.10938
[llada]: https://arxiv.org/html/2502.09992v3#S3.T1
[tess]: https://arxiv.org/html/2502.13917v1#S4.T3
