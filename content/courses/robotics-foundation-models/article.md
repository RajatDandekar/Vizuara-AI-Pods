# Developments in Robotics Foundation Models

*From task-specific controllers to general-purpose robot brains — how foundation models are transforming robotics*

Vizuara AI

---

## One Model to Rule Them All

Let us start with a thought experiment. Imagine a robot in your kitchen that has never seen your favorite coffee mug before. It has never been in your kitchen. It has never been asked to "make coffee." Yet it picks up the mug, places it under the coffee machine, and presses the right button.

How? Because it has a **foundation model** — a single, general-purpose brain trained on massive amounts of data that gives it the ability to generalize to new situations, new objects, and even new robots.


![A robot in a kitchen confidently performing a coffee-making task it has never been explicitly trained on. Thought bubbles show knowledge transfer: understanding of mugs from web images, understanding of coffee machines from text descriptions, manipulation skills from diverse robot training. Clean illustration style.](figures/figure_1.png)
*A robot in a kitchen confidently performing a coffee-making task it has never been explicitly trained on. Thought bubbles show knowledge transfer: understanding of mugs from web images, understanding of coffee machines from text descriptions, manipulation skills from diverse robot training. Clean illustration style.*


In the world of NLP, foundation models like GPT transformed everything. A single model, pre-trained on internet-scale text, could be adapted to translation, summarization, coding, and thousands of other tasks. In computer vision, models like CLIP and DALL-E did the same.

But robotics has been the stubborn holdout. Until now.

Over the past three years (2022-2025), we have witnessed an explosion of robotics foundation models that are finally bringing the foundation model paradigm to the physical world. Let us trace this remarkable journey.

---

## Why Robotics Needed Foundation Models

The traditional approach to robot learning was painfully narrow. You wanted a robot to pick up cups? Train a cup-picking policy. You wanted it to open drawers? Train a separate drawer-opening policy. Each task, each robot, each environment required its own model trained from scratch.

This approach had three fundamental problems:

**1. Data scarcity** — Collecting robot demonstrations is expensive and slow. While NLP has trillions of tokens from the web, a typical robot learning project might have only 1,000-10,000 demonstrations.

**2. No knowledge transfer** — A model trained to pick up cups knows nothing about opening drawers. Knowledge does not transfer between tasks.

**3. The embodiment problem** — A policy trained on a Franka Panda robot arm cannot be directly used on a UR5 arm, even for the same task. Different robots have different kinematics, sensors, and action spaces.


![Three-panel diagram showing the three problems. Panel 1 Data Scarcity: small pile of robot demos vs massive mountain of web text. Panel 2 No Transfer: two separate isolated brain icons for cup picking and drawer opening. Panel 3 Embodiment Gap: same task same object but different robots cannot share policies. Clean infographic.](figures/figure_2.png)
*Three-panel diagram showing the three problems. Panel 1 Data Scarcity: small pile of robot demos vs massive mountain of web text. Panel 2 No Transfer: two separate isolated brain icons for cup picking and drawer opening. Panel 3 Embodiment Gap: same task same object but different robots cannot share policies. Clean infographic.*


Foundation models promise to solve all three problems simultaneously. By pre-training on massive, diverse data — including internet-scale vision and language data — a single model can acquire general knowledge that transfers across tasks, objects, and even robot bodies.

---

## The Timeline: A Revolution in Three Years

### 2022: The Beginning — BC-Z and RT-1

The story begins with two models from Google that demonstrated the power of scale.

**BC-Z** (Behavioral Cloning Zero-shot) showed that a model trained on diverse demonstrations from many tasks could generalize to new tasks zero-shot — without any task-specific fine-tuning. This was one of the first proofs that scale and diversity in robot data could yield generalization.

**RT-1** (Robotics Transformer 1) took this further. Google collected 130,000+ demonstrations across 700+ tasks using 13 robots over 17 months. The resulting model achieved 97% success on seen tasks and 76% on unseen instructions. RT-1 proved that **data scale matters** in robotics just as it does in NLP.


![Timeline starting at 2022 showing BC-Z and RT-1 as the first major milestones. RT-1 architecture shown: EfficientNet plus TokenLearner plus Transformer. Key stat: 130K demos, 700 tasks, 97 percent success. Arrow pointing forward to 2023.](figures/figure_3.png)
*Timeline starting at 2022 showing BC-Z and RT-1 as the first major milestones. RT-1 architecture shown: EfficientNet plus TokenLearner plus Transformer. Key stat: 130K demos, 700 tasks, 97 percent success. Arrow pointing forward to 2023.*


### 2023: The VLA Revolution — RT-2 and Octo

**RT-2** (Google, 2023) was the game-changer. Instead of training a robot policy from scratch, RT-2 took a massive Vision-Language Model (PaLM-E / PaLI-X) and fine-tuned it to output robot actions as text tokens. The result: a robot that could reason about concepts it had never seen in robot training data — understanding "unhealthy food," "Taylor Swift," and spatial concepts like "top-left corner."

RT-2 proved the most important insight of this era: **internet knowledge transfers to robotics.**

**Octo** (UC Berkeley, 2024) brought this revolution to the open-source community. Trained on the Open X-Embodiment dataset with data from 22 robot types, Octo was the first **open-source generalist robot policy**. Its modular design with a diffusion action head made it easy to fine-tune for new robots and tasks.


![2023 timeline segment showing RT-2 and Octo. RT-2 shown with VLM backbone producing action tokens. Octo shown with open-source badge and cross-embodiment design. Key innovation highlighted: Actions as language tokens. Arrow pointing to 2024.](figures/figure_4.png)
*2023 timeline segment showing RT-2 and Octo. RT-2 shown with VLM backbone producing action tokens. Octo shown with open-source badge and cross-embodiment design. Key innovation highlighted: Actions as language tokens. Arrow pointing to 2024.*


### 2024: The Explosion — OpenVLA, Pi0, GR-2, and More

2024 saw an explosion of robotics foundation models:

**OpenVLA** (Stanford/Berkeley) — A fully open-source 7B parameter VLA built on Llama 2. Trained on Open X-Embodiment data and fine-tunable with LoRA on as few as 10-20 demonstrations. OpenVLA democratized VLA research by making weights, code, and training recipes fully available.

**Pi0** (Physical Intelligence) — Used flow matching instead of action tokenization, enabling smooth continuous actions. Combined PaliGemma VLM with a dedicated action expert transformer. Achieved state-of-the-art on dexterous tasks like laundry folding and box assembly.

**GR-2** (ByteDance) — Took a completely different approach: use video prediction as a world model. GR-2 predicts future video frames of the robot performing the task, then extracts actions from the predicted video. The key idea: internet video teaches physics and object interactions.

**Genie 2** (Google DeepMind) — A learned world model that generates interactive 3D environments from a single image. While not a robot policy itself, Genie 2 creates realistic simulated worlds where policies can be trained — a potential solution to the data scarcity problem.


![2024 explosion diagram showing four models radiating from a central point. OpenVLA with open-source badge and 7B parameter label. Pi0 with flow matching and dexterous manipulation labels. GR-2 with video prediction label. Genie 2 with world generation label. Each model has its key innovation highlighted.](figures/figure_5.png)
*2024 explosion diagram showing four models radiating from a central point. OpenVLA with open-source badge and 7B parameter label. Pi0 with flow matching and dexterous manipulation labels. GR-2 with video prediction label. Genie 2 with world generation label. Each model has its key innovation highlighted.*


---

## The Open X-Embodiment Dataset: ImageNet for Robots

One of the most important developments has been the creation of large-scale, cross-robot datasets. The **Open X-Embodiment (OXE)** dataset, created through a collaboration of 21 institutions, is the closest thing robotics has to ImageNet.

OXE contains:
- **1 million+ robot episodes**
- Data from **22 different robot embodiments**
- **527 different skills** spanning manipulation, navigation, and more
- Contributions from Google, Stanford, Berkeley, CMU, and many others

The key finding from OXE was profound: **cross-embodiment training helps everyone.** A model trained on data from all 22 robots performs better on any single robot than a model trained on that robot's data alone. The shared visual and semantic representations transfer across morphologies.


![Open X-Embodiment dataset visualization. Central database icon surrounded by diverse robot types: Franka Panda, UR5, Google Robot, ALOHA, Stretch, Kuka, Sawyer, etc. Statistics: 1M episodes, 22 embodiments, 527 skills, 21 institutions. Arrows showing data flowing in and improved policies flowing out.](figures/figure_6.png)
*Open X-Embodiment dataset visualization. Central database icon surrounded by diverse robot types: Franka Panda, UR5, Google Robot, ALOHA, Stretch, Kuka, Sawyer, etc. Statistics: 1M episodes, 22 embodiments, 527 skills, 21 institutions. Arrows showing data flowing in and improved policies flowing out.*


---

## The Architecture Blueprint: How They All Work

Despite their differences, most robotics foundation models follow a remarkably similar architecture pattern:

**Vision Encoder** → **Language Model Backbone** → **Action Head**

The **vision encoder** (typically a ViT variant like SigLIP or DINOv2) converts camera images into visual tokens. The **language model backbone** (Llama, PaLM, Gemma) processes visual and text tokens together, creating a rich multimodal representation. The **action head** converts this representation into robot motor commands.

Where models differ is primarily in the action head:
- **Discrete tokenization** (RT-2, OpenVLA) — actions as classification over bins
- **Diffusion** (Octo) — iterative denoising to continuous actions
- **Flow matching** (Pi0) — straighter denoising with fewer steps


![Generic VLA architecture blueprint. Three rows showing options for each component. Vision Encoder row: SigLIP, DINOv2, EfficientNet. Language Backbone row: Llama 7B, PaLM-E 12B, PaLI-X 55B, Gemma 2B. Action Head row: Discrete Tokens, Diffusion, Flow Matching. Arrows show how any combination can be assembled into a VLA.](figures/figure_7.png)
*Generic VLA architecture blueprint. Three rows showing options for each component. Vision Encoder row: SigLIP, DINOv2, EfficientNet. Language Backbone row: Llama 7B, PaLM-E 12B, PaLI-X 55B, Gemma 2B. Action Head row: Discrete Tokens, Diffusion, Flow Matching. Arrows show how any combination can be assembled into a VLA.*


---

## The Training Recipe: Three Stages

The training pipeline has also converged on a standard recipe:

**Stage 1: Internet pre-training** — Train the VLM backbone on billions of image-text pairs from the web. This gives the model rich understanding of objects, scenes, language, and world knowledge. This stage is typically done once and shared.

**Stage 2: Robot data pre-training** — Fine-tune on large-scale robot demonstration data (like OXE). The VLM learns to map its visual-language understanding to motor commands. This creates a generalist robot policy.

**Stage 3: Task-specific fine-tuning** — Adapt the generalist policy to specific tasks, robots, or environments using a small number of demonstrations. Techniques like LoRA make this efficient — sometimes requiring only 10-20 demos.


![Three-stage funnel diagram. Top wide section: Internet Pre-training with billions of image-text pairs. Middle section: Robot Pre-training with millions of robot episodes from OXE. Bottom narrow section: Task Fine-tuning with 10-100 task-specific demos. Each stage shows data volume decreasing but specificity increasing.](figures/figure_8.png)
*Three-stage funnel diagram. Top wide section: Internet Pre-training with billions of image-text pairs. Middle section: Robot Pre-training with millions of robot episodes from OXE. Bottom narrow section: Task Fine-tuning with 10-100 task-specific demos. Each stage shows data volume decreasing but specificity increasing.*


This recipe mirrors exactly what happened in NLP: GPT was pre-trained on web text, then fine-tuned for specific tasks. The same pattern now works for robots.

---

## The Role of Simulation

Simulation has become increasingly important as a complement to real-world data:

**MuJoCo** — The classic physics simulator, now open-source, widely used for locomotion and manipulation research.

**NVIDIA Isaac Sim / Isaac Lab** — GPU-accelerated simulation that can run thousands of environments in parallel. This has been transformative for reinforcement learning in robotics.

**Genie 2** (DeepMind) — A learned simulator that generates realistic interactive environments from single images. Rather than hand-designing simulation environments, Genie 2 generates them from data.

The frontier is **learned simulators** — neural networks that simulate physics directly. These promise to bridge the gap between the scalability of simulation and the realism of the physical world.


![Simulation landscape diagram. Three tiers: Traditional Simulators at bottom with MuJoCo PyBullet SAPIEN. GPU-Accelerated middle tier with Isaac Sim Isaac Lab. Learned Simulators top tier with Genie 2 UniSim. Arrow on right showing increasing realism from bottom to top.](figures/figure_9.png)
*Simulation landscape diagram. Three tiers: Traditional Simulators at bottom with MuJoCo PyBullet SAPIEN. GPU-Accelerated middle tier with Isaac Sim Isaac Lab. Learned Simulators top tier with Genie 2 UniSim. Arrow on right showing increasing realism from bottom to top.*


---

## Current Limitations: Honest Assessment

Despite the excitement, robotics foundation models face real limitations:

**Data is still scarce** — Even 1 million episodes is orders of magnitude less than what language models use. The physical world is expensive to sample.

**The sim-to-real gap persists** — Simulation cannot perfectly replicate real-world physics, especially for contact-rich manipulation and deformable objects.

**Safety is unsolved** — A model that generalizes to new situations might also fail in unpredictable ways. There are no formal safety guarantees.

**Real-time inference is hard** — Large VLMs (55B parameters) cannot run at the control frequencies robots need (50-100Hz). Smaller models sacrifice capability for speed.

**Long-horizon planning is weak** — Current models predict actions one step (or one chunk) at a time. Tasks requiring planning over minutes or hours remain out of reach.

---

## Where the Field Is Heading

Several trends are converging:

**Scaling laws for robotics** — Just as language models showed predictable improvement with scale, early evidence suggests the same holds for robot foundation models. More data, more parameters, and more compute lead to better generalization.

**Foundation models plus RL** — Combining the generalization of foundation models with the optimization power of reinforcement learning. Pre-train a general policy, then refine it with RL for specific objectives.

**Multi-task generalization** — The holy grail: a single model that can perform any manipulation task given a natural language instruction. We are not there yet, but models like Pi0 are getting closer.

**Learned world models** — Instead of predicting actions directly, learn to predict the consequences of actions. This enables planning and imagination — hallmarks of intelligent behavior.

The trajectory is clear. Just as NLP went from task-specific models (2015) to GPT-4 (2023) in eight years, robotics is on a similar path. We are currently in the "GPT-2 era" of robot learning — the models are impressive but limited. The "GPT-4 era" of robotics, where a single model can handle any physical task, may be closer than we think.

In the next article, we will dive into one of the most critical challenges in this journey: the **Sim-to-Real Problem** — why robots trained in virtual worlds struggle in reality, and how we are closing the gap. See you next time!

---

*References:*
- *Brohan et al., "RT-1" (2022); "RT-2" (2023)*
- *Octo Model Team, "Octo: An Open-Source Generalist Robot Policy" (2024)*
- *Kim et al., "OpenVLA" (2024)*
- *Black et al., "Pi0" (2024)*
- *Open X-Embodiment Collaboration (2023)*
- *Wu et al., "GR-2: Video Generation for Robot Policy Learning" (2024)*