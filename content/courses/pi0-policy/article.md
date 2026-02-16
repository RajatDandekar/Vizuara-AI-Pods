# How Does the Pi0 Policy from Physical Intelligence Exactly Work?

*Inside the architecture that taught a single AI model to fold laundry, assemble boxes, and bus tables*

Vizuara AI

---

## A Robot That Folds Your Laundry

Let us start with a deceptively simple task. You pull a crumpled t-shirt out of the dryer, lay it on a table, and fold it neatly. You do this without thinking — your hands feel the fabric, adjust to wrinkles, and adapt on the fly.

Now imagine asking a robot to do the same thing. This is one of the hardest challenges in all of robotics. The fabric is **deformable** — it changes shape as you touch it. There is no fixed trajectory to follow. Every shirt is different. Every wrinkle creates a unique scenario.

In late 2024, a company called **Physical Intelligence** (or "pi" for short) released a model called **pi0** that could do exactly this. A single neural network — the same set of weights — could fold laundry, assemble cardboard boxes, bus tables, and pack groceries.


![A grid of four panels showing pi0 performing different tasks: top-left folding a t-shirt on a table, top-right assembling a cardboard box, bottom-left clearing dishes from a table, bottom-right packing items into a grocery bag. Each panel labeled with the task name and success rate.](figures/figure_1.png)
*A grid of four panels showing pi0 performing different tasks: top-left folding a t-shirt on a table, top-right assembling a cardboard box, bottom-left clearing dishes from a table, bottom-right packing items into a grocery bag. Each panel labeled with the task name and success rate.*


How did they do it? The answer involves a clever combination of two powerful ideas:

1. A **pre-trained Vision-Language Model (VLM)** that understands the world from internet-scale data
2. **Flow matching** — a technique from generative modeling that produces smooth, continuous robot actions

Let us break this down piece by piece.

---

## Why Not Just Use a Regular VLA?

In the previous article, we discussed Vision-Language-Action Models (VLAs) like RT-2 and OpenVLA. These models output robot actions as **discrete tokens** — they discretize each action dimension into bins and treat it like a classification problem.

But here is the problem. Have a look at the following scenario.

Imagine a robot approaching an obstacle. It can go left or it can go right — both are valid. If we train a model to predict the "average" of these two demonstrations, the robot will try to go **straight through the obstacle**.


![The multimodal action distribution problem. Left panel: two valid trajectories around an obstacle, one going left and one going right. Right panel: a naive mean regression predicts the average trajectory which goes straight into the obstacle. Labels: Valid Left Path, Valid Right Path, Mean Prediction (fails), Obstacle.](figures/figure_2.png)
*The multimodal action distribution problem. Left panel: two valid trajectories around an obstacle, one going left and one going right. Right panel: a naive mean regression predicts the average trajectory which goes straight into the obstacle. Labels: Valid Left Path, Valid Right Path, Mean Prediction (fails), Obstacle.*


This is the **multimodal action distribution problem**. Real robot demonstrations are often multimodal — there are multiple valid ways to accomplish a task. Action tokenization (discretization) handles this to some extent, but it fundamentally limits the precision of actions and struggles with continuous, flowing motions.

Now the question is: can we do better?

This brings us to the core innovation of pi0 — **flow matching**.

---

## Flow Matching: Learning to Transform Noise into Actions

Let us build an intuition for flow matching with an analogy.

Imagine you are an artist starting with a canvas of pure random noise — TV static. Now imagine there is a vector field — a set of arrows at every point on the canvas — that tells you exactly how to smoothly transform each point of noise into a beautiful painting. If you follow these arrows, the noise gradually morphs into art.


![Flow matching intuition diagram. Left: a cloud of random noise points. Center: arrows (vector field) showing the direction each point should move. Right: the points have followed the arrows and formed a structured pattern representing a clean robot action trajectory. Time axis labeled t=0 (noise) to t=1 (action).](figures/figure_3.png)
*Flow matching intuition diagram. Left: a cloud of random noise points. Center: arrows (vector field) showing the direction each point should move. Right: the points have followed the arrows and formed a structured pattern representing a clean robot action trajectory. Time axis labeled t=0 (noise) to t=1 (action).*


This is exactly what flow matching does for robot actions:

1. Start with random noise ($$t = 0$$)
2. Learn a velocity field that transforms noise into valid robot actions
3. At inference time, start from noise and follow the learned velocity field to generate an action ($$t = 1$$)

Let us now look at the mathematics. The forward process defines an interpolation between noise and a real action:

$$
x_t = (1-t) \varepsilon + t \cdot x_1, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

Here, $$x_1$$ is a real robot action from our demonstrations, $$\varepsilon$$ is random noise, and $$x_t$$ is the interpolated point at time $$t$$. At $$t = 0$$, we have pure noise. At $$t = 1$$, we have the real action.

The velocity field that transforms noise into actions is simply:

$$
u_t(x_t) = x_1 - \varepsilon
$$

And the neural network learns to predict this velocity field. The training loss is:

$$
\mathcal{L}_{\mathrm{flow}} = \mathbb{E}_{t, \varepsilon, x_1} \left[ \| v_{\theta}(x_t, t) - (x_1 - \varepsilon) \|^2 \right]
$$

This is exactly what we want. The network learns to predict the direction from noise to data, and during inference, it follows this learned direction to generate actions.

**Why flow matching over diffusion?** In diffusion models, the denoising path is curved and requires many steps (sometimes 100+). Flow matching produces **straighter paths** from noise to data, so it needs far fewer denoising steps — as few as 5-10 at inference time. This makes it practical for real-time robot control.


![Comparison between diffusion and flow matching trajectories. Left: Diffusion model with curved paths from noise to data requiring many denoising steps (showing 50 plus steps). Right: Flow matching with much straighter paths requiring only 5 to 10 steps. Both start from same noise distribution and reach same data distribution.](figures/figure_4.png)
*Comparison between diffusion and flow matching trajectories. Left: Diffusion model with curved paths from noise to data requiring many denoising steps (showing 50 plus steps). Right: Flow matching with much straighter paths requiring only 5 to 10 steps. Both start from same noise distribution and reach same data distribution.*


---

## The Pi0 Architecture: Putting It All Together

Now let us look at the full pi0 architecture. It has three main components:

### 1. Vision-Language Model Backbone: PaliGemma

Pi0 uses **PaliGemma** as its VLM backbone — a 3 billion parameter model that combines:
- **SigLIP** vision encoder (400M parameters) — processes camera images into visual tokens
- **Gemma** language model (2.6B parameters) — processes text instructions and visual tokens

This VLM has been pre-trained on internet-scale image-text data. It already understands objects, spatial relationships, and language instructions before seeing a single robot demonstration.

### 2. Action Expert: A Separate Transformer

Here is the clever part. Instead of forcing the VLM to directly output actions (like RT-2 does), pi0 adds a **separate action expert** — a smaller transformer that is dedicated to generating robot actions.

The action expert communicates with the VLM through **cross-attention**. It attends to the VLM's internal representations to understand the visual scene and language instruction, but it has its own dedicated parameters for action generation.


![Pi0 full architecture diagram. Left side: camera images feed into SigLIP vision encoder, text instruction feeds into Gemma tokenizer. Both feed into PaliGemma VLM which produces rich representations. Right side: Action Expert transformer receives noise plus timestep t, cross-attends to VLM representations, and outputs denoised robot actions. The flow matching process is shown with an arrow from noise at t=0 to clean action at t=1.](figures/figure_5.png)
*Pi0 full architecture diagram. Left side: camera images feed into SigLIP vision encoder, text instruction feeds into Gemma tokenizer. Both feed into PaliGemma VLM which produces rich representations. Right side: Action Expert transformer receives noise plus timestep t, cross-attends to VLM representations, and outputs denoised robot actions. The flow matching process is shown with an arrow from noise at t=0 to clean action at t=1.*


Think of it this way: the VLM is like the **brain** that understands the world, and the action expert is like the **spinal cord** that translates understanding into precise motor commands. The brain does not need to know the details of muscle control — it provides high-level understanding, and the spinal cord handles the execution.

### 3. Action Chunking: Predicting the Future

Pi0 does not predict a single action at each timestep. Instead, it predicts an **action chunk** — a sequence of 50 future actions at 50Hz, covering 1 full second of robot motion.


![Action chunking diagram. A single forward pass of pi0 producing a chunk of 50 sequential action steps shown as a time series of joint angles. The x-axis shows time from 0 to 1 second with 50 steps, the y-axis shows joint angles for multiple joints. Smooth, continuous curves demonstrate temporal coherence.](figures/figure_6.png)
*Action chunking diagram. A single forward pass of pi0 producing a chunk of 50 sequential action steps shown as a time series of joint angles. The x-axis shows time from 0 to 1 second with 50 steps, the y-axis shows joint angles for multiple joints. Smooth, continuous curves demonstrate temporal coherence.*


Why action chunks? Three reasons:
1. **Temporal coherence** — actions within a chunk are smooth and consistent
2. **Handles latency** — even if inference takes 100ms, the robot has 50 pre-planned actions
3. **Better for flowing motions** — folding fabric requires smooth, continuous trajectories

---

## The Training Recipe: Three Stages

Pi0's training follows a three-stage pipeline that mirrors how foundation models are trained in NLP:

**Stage 1: Start with PaliGemma (pre-trained on internet data)**

The VLM backbone already understands images and language. Pi0 inherits all of this knowledge for free.

**Stage 2: Pre-train on diverse robot data**

The model is pre-trained on a mixture of:
- **Open X-Embodiment** dataset (diverse robot demonstrations from many labs)
- **Pi's own proprietary data** (10,000+ hours of robot demonstrations)
- Data from **7+ different robot embodiments** (single arm, bimanual, different grippers)

During this stage, the VLM backbone is fine-tuned and the action expert is trained from scratch. The model learns a general understanding of robot manipulation.

**Stage 3: Task-specific fine-tuning**

For each specific task (like laundry folding), the model is further fine-tuned on high-quality demonstrations of that particular task.


![Three-stage training pipeline diagram. Stage 1 on the left: Internet Pre-training with icons of web images and text, labeled PaliGemma 3B. Arrow to Stage 2 in the middle: Robot Pre-training with icons of diverse robots and demonstrations, labeled Mixed Robot Data 10K plus hours. Arrow to Stage 3 on the right: Task Fine-tuning with icons of specific tasks like laundry folding, labeled High-Quality Demos.](figures/figure_7.png)
*Three-stage training pipeline diagram. Stage 1 on the left: Internet Pre-training with icons of web images and text, labeled PaliGemma 3B. Arrow to Stage 2 in the middle: Robot Pre-training with icons of diverse robots and demonstrations, labeled Mixed Robot Data 10K plus hours. Arrow to Stage 3 on the right: Task Fine-tuning with icons of specific tasks like laundry folding, labeled High-Quality Demos.*


---

## Cross-Embodiment: One Model, Many Robots

One of pi0's remarkable capabilities is **cross-embodiment transfer**. The same model, with the same weights, can control different robot platforms:

- **ALOHA bimanual** — two robot arms working together
- **Franka Panda** — single arm with parallel-jaw gripper
- **UR5** — industrial robot arm
- Various custom platforms from Pi's lab

How does this work? Different robots have different numbers of joints (degrees of freedom). Pi0 handles this by **padding the action space** — all robots share a maximum-dimensional action vector, and unused dimensions are masked out for robots with fewer joints.


![Cross-embodiment diagram. A single pi0 model icon in the center with arrows pointing to four different robot types around it: ALOHA bimanual system with two arms, Franka Panda single arm, UR5 industrial arm, and a custom dexterous hand. Each robot shows its degrees of freedom. The shared VLM representations are highlighted in the center.](figures/figure_8.png)
*Cross-embodiment diagram. A single pi0 model icon in the center with arrows pointing to four different robot types around it: ALOHA bimanual system with two arms, Franka Panda single arm, UR5 industrial arm, and a custom dexterous hand. Each robot shows its degrees of freedom. The shared VLM representations are highlighted in the center.*


The shared VLM backbone means that visual understanding transfers across all platforms. A concept learned on one robot (like "pick up the cup") transfers to another robot even though the motor commands are completely different.

---

## Results: What Pi0 Can Actually Do

Pi0's results are genuinely impressive. Let us look at some key tasks:

**Laundry folding** — arguably the hardest dexterous manipulation task ever demonstrated by a learned policy. The fabric is deformable, every configuration is unique, and success requires multi-step reasoning (flatten → fold sides → fold in half).

**Box assembly** — taking a flat cardboard box and folding it into a 3D box. Requires understanding 3D geometry, sequential manipulation, and applying appropriate forces.

**Table bussing** — clearing a messy table of plates, cups, and utensils. Requires scene understanding, prioritization, and careful manipulation of fragile objects.

In direct comparisons:
- Pi0 dramatically outperforms **Octo** (the open-source generalist) on dexterous tasks
- Pi0 outperforms **RT-2-X** on cross-task generalization
- Pi0 is **competitive with or better than task-specific models** while being a single generalist

---

## Pi0-FAST: Making It Even Faster

One limitation of flow matching is that it still requires multiple denoising steps at inference time (typically 5-10 forward passes). For time-critical applications, even this can be too slow.

Pi0-FAST addresses this by replacing flow matching with **discrete action tokenization** — similar to RT-2's approach but using a learned **VQ-VAE** (Vector Quantized Variational Autoencoder) to tokenize actions.

The key difference: instead of learning to denoise over multiple steps, pi0-FAST generates all action tokens in a **single forward pass**. This is significantly faster while retaining most of pi0's performance.


![Comparison diagram of pi0 vs pi0-FAST. Left: pi0 with flow matching showing multiple denoising steps from noise to action with 5 to 10 iterations. Right: pi0-FAST with VQ-VAE tokenization showing a single forward pass producing discrete action tokens that are decoded to continuous actions. Speed comparison shown below.](figures/figure_9.png)
*Comparison diagram of pi0 vs pi0-FAST. Left: pi0 with flow matching showing multiple denoising steps from noise to action with 5 to 10 iterations. Right: pi0-FAST with VQ-VAE tokenization showing a single forward pass producing discrete action tokens that are decoded to continuous actions. Speed comparison shown below.*


---

## How Does Pi0 Compare to Other Approaches?

Let us put pi0 in context with other robot foundation models:

| **Model** | **VLM Size** | **Action Representation** | **Key Innovation** |
|-----------|-------------|--------------------------|-------------------|
| RT-2 | 55B (PaLI-X) | Discrete tokens | Actions as text tokens |
| Octo | ~93M | Diffusion head | Open-source, cross-embodiment |
| OpenVLA | 7B (Llama 2) | Discrete tokens | Open-source VLA |
| **Pi0** | **3B (PaliGemma)** | **Flow matching** | **VLM + flow matching + action chunking** |
| Pi0-FAST | 3B (PaliGemma) | VQ-VAE tokens | Single-pass discrete tokenization |

Pi0's strength lies in the combination: a pre-trained VLM for world understanding, flow matching for expressive continuous actions, and action chunking for smooth temporal execution.

---

## A Simplified Flow Matching Implementation

Let us look at a simplified implementation of the flow matching training loop:

```python
import torch
import torch.nn as nn

class FlowMatchingPolicy(nn.Module):
    """Simplified flow matching for robot actions."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Network predicts the velocity field v(x_t, t, obs)
        self.net = nn.Sequential(
            nn.Linear(action_dim + 1 + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_t, t, obs):
        """Predict velocity field given noisy action, timestep, observation."""
        return self.net(torch.cat([x_t, t, obs], dim=-1))

def train_step(model, obs, actions):
    """One training step of flow matching."""
    batch_size = actions.shape[0]

    # Sample random timestep t in [0, 1]
    t = torch.rand(batch_size, 1)

    # Sample noise
    eps = torch.randn_like(actions)

    # Interpolate: x_t = (1-t) * eps + t * actions
    x_t = (1 - t) * eps + t * actions

    # Target velocity: u_t = actions - eps
    target = actions - eps

    # Predict velocity
    pred = model(x_t, t, obs)

    # MSE loss
    loss = ((pred - target) ** 2).mean()
    return loss

def sample_action(model, obs, num_steps=10):
    """Generate an action via flow matching inference."""
    # Start from pure noise
    x = torch.randn(1, action_dim)

    # Integrate from t=0 to t=1 in num_steps
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.tensor([[i * dt]])
        v = model(x, t, obs)
        x = x + v * dt   # Euler integration

    return x
```

Let us understand this code. The `FlowMatchingPolicy` is a simple neural network that predicts the velocity field — given a noisy action, a timestep, and an observation, it predicts which direction the action should move. During training, we sample random timesteps, create interpolated noisy actions, and train the network to predict the correct velocity. During inference, we start from noise and take small steps following the predicted velocity until we reach a clean action.

This is the core idea behind pi0, wrapped in a much larger architecture with a VLM backbone and cross-attention.

---

## Limitations and What Comes Next

Pi0 is a remarkable achievement, but it has clear limitations:

1. **Data requirements** — Pi0 needs high-quality demonstrations, especially for Stage 3 fine-tuning. Data collection is expensive.

2. **Inference cost** — Even with flow matching's efficiency, running a 3B parameter model at 50Hz on robot hardware is challenging.

3. **Manipulation only** — Pi0 focuses on manipulation. Navigation, locomotion, and full-body humanoid control are not addressed.

4. **Safety** — There are no formal guarantees about what the model will do in novel situations.

But the trajectory is clear. Pi0 demonstrates that the recipe of **pre-trained VLM + expressive action generation + large-scale robot data** produces models that can handle tasks previously thought to be far out of reach for learning-based approaches.

In the next article, we will look at **RT-1 and RT-2 from Google** — the models that first showed the world that transformers and large language models could be used to control robots. See you next time!

---

*References:*
- *Black et al., "pi0: A Vision-Language-Action Flow Model for General Robot Control" (2024)*
- *Pertsch et al., "pi0-FAST: Fast Action Tokenization for Vision-Language-Action Models" (2025)*
- *Lipman et al., "Flow Matching for Generative Modeling" (2023)*