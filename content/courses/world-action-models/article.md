# World Action Models: How AI Learns to Imagine Before It Acts

From mental simulations to robot control — a complete guide to world models, imagination-based learning, and Vision-Language-Action architectures

Vizuara AI

---

Let us start with a simple thought experiment.

Close your eyes and imagine picking up a glass of water from the table in front of you. Before your hand even moves, your brain has already run a **mental simulation**. You can feel the cold surface of the glass, you can predict its weight, and you know exactly how much you need to tilt it to take a sip.

You have not touched the glass yet, but you already know what will happen.


![Mental simulation vs trial and error](figures/figure_01.png)
*Humans simulate before acting. Robots without world models must learn by trial and error.*


Now imagine a robot trying to do the same thing — but it has no mental model of the world. It does not know what a glass is, what happens when you tilt it, or how gravity works. It has to try every possible action blindly and learn from the consequences.

This is the fundamental challenge: **How do we give AI an internal model of the world so it can imagine the consequences of its actions before taking them?**

This is where world models come in.

In this article, we will go on a journey through one of the most exciting areas in modern AI — from the first papers that taught agents to "dream," all the way to modern robots that can see, understand language, and act in the real world.

---

## What is a World Model?

Let us understand this concept more formally.

A world model is a learned internal representation of the environment that allows an agent to **predict what will happen next**, given its current state and a chosen action.

In other words, the world model answers the question:

**"If I am in state s and I take action a, what will the next state be?"**

The mathematical representation for this can be written as:


$$
s_{t+1} = f(s_t, a_t)
$$


Here, $$s_t$$ is the current state, $$a_t$$ is the action taken by the agent, and $$s_{t+1}$$ is the predicted next state. The function $$f$$ is our world model — it has learned the dynamics of the environment.

Now, you might be thinking: "Why do we need a world model? Can the agent not just learn by interacting with the environment?"

Yes, it can. This is what we call **model-free** reinforcement learning. Remember from our lectures on reinforcement learning, the agent interacts with the environment, receives rewards, and updates its policy. No model of the environment is needed.

But there is a problem with this approach.

Let us take the example from our lecture on Imitation Learning. We wanted to train a robot to pour a heart shape in a coffee latte. With model-free RL, the robot has to explore randomly at first — it will spill hot milk on the floor, smash the mug, and pour milk onto its own circuits.

What if, instead, the robot could **first build a mental model of the environment** and then **plan actions using that model**?

This is what we call **model-based** reinforcement learning.


![Model-free vs model-based reinforcement learning](figures/figure_02.png)
*Model-free RL learns by direct interaction, while model-based RL imagines multiple trajectories before selecting the best action.*


Think of it this way:

Model-free RL is like learning to drive by actually crashing into things and learning from each crash.

Model-based RL is like learning to drive in a **driving simulator** first, and then driving on the real road.

Which one sounds safer? Which one sounds more efficient?

This is exactly the motivation behind world models. Now let us look at the paper that started it all.

---

## The Birth of World Models — Ha & Schmidhuber (2018)

In 2018, David Ha and Jürgen Schmidhuber published a paper simply titled **"World Models."** The key insight of this paper was remarkably simple yet powerful:

**What if the agent could dream?**

What if the agent learned a model of its environment, and then trained its policy *entirely inside that dream* — without ever interacting with the real environment?

Let us understand how this works.

The architecture consists of three components: **V** (Vision), **M** (Memory), and **C** (Controller).

### V — The Vision Component (VAE)

The first problem is that real-world observations are very high-dimensional. A game screen might be 96×96 pixels with 3 color channels — that is 27,648 numbers!

We need to compress this into a smaller representation. Does this remind you of something?

Yes — this is exactly what a **Variational Autoencoder** does. (Refer to our article on VAEs: the encoder compresses high-dimensional inputs into a latent vector.)

The Vision component is a VAE that compresses the observation $$x_t$$ into a small latent vector $$z_t$$:


$$
z_t = \text{Encoder}(x_t)
$$


For example, a 96×96 pixel game frame gets compressed into a vector of just 32 numbers. This latent vector captures the essential information about the current state of the game.

### M — The Memory Component (MDN-RNN)

This is the **world model** itself.

The Memory component is a recurrent neural network that takes the current latent state $$z_t$$ and the action $$a_t$$, and predicts a probability distribution over possible next latent states.


$$
P(z_{t+1} | a_t, z_t, h_t)
$$


Here, $$h_t$$ is the hidden state of the RNN, which captures the history of past observations and actions. The model does not predict a single next state — it predicts a **distribution** over possible next states. This is important because the real world is stochastic — the same action in the same state can lead to different outcomes.

### C — The Controller

The Controller is surprisingly simple — just a single linear layer that takes the latent state $$z_t$$ and the hidden state $$h_t$$ of the Memory component, and outputs an action:

$$a_t = W \cdot [z_t, h_t] + b$$

That is it. The entire intelligence of the system comes from the quality of the world model, not from the complexity of the controller.

### The "Dream" Training Procedure

Now here is the most exciting part.

The training happens in three steps:

**Step 1:** Collect random experiences from the real environment and train the VAE (V) to compress observations into latent vectors.

**Step 2:** Train the RNN world model (M) on sequences of latent vectors and actions.

**Step 3:** Train the Controller (C) entirely inside the world model's "dream." The agent imagines trajectories by rolling forward through M, and the Controller learns to take actions that maximize rewards in this imagined world.


![The V-M-C world model architecture](figures/figure_03.png)
*The Vision-Memory-Controller pipeline: observations are encoded into latent vectors, the world model predicts future states, and the controller is trained entirely inside the "dream."*


The results were remarkable. The agent learned to play a car racing game *inside its own dream* and then successfully transferred that skill to the real game. The dream was not perfect — it was blurry and sometimes hallucinated — but it was good enough for the agent to learn useful behaviors.


![Real environment vs agent's dream](figures/figure_04.png)
*The real game frame compared to the agent's VAE-reconstructed "dream" -- blurry but recognizable enough to learn useful behaviors.*


Let us look at a simplified code implementation for the V-M pipeline:

```python
import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    """V: Compresses observations into latent vectors"""
    def __init__(self, obs_dim=27648, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

class WorldModel(nn.Module):
    """M: Predicts next latent state given current state and action"""
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256):
        super().__init__()
        self.rnn = nn.LSTMCell(latent_dim + action_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, action, hidden):
        # Concatenate latent state and action
        rnn_input = torch.cat([z, action], dim=-1)
        h, c = self.rnn(rnn_input, hidden)
        # Predict distribution over next latent state
        next_mu = self.fc_mu(h)
        next_logvar = self.fc_logvar(h)
        return next_mu, next_logvar, (h, c)
```

Let us understand this code in detail.

The `VAEEncoder` class is our Vision component (V). It takes a raw observation (flattened game frame) and compresses it through two linear layers into a latent vector of just 32 dimensions. Notice how it outputs both a mean (`mu`) and log-variance (`logvar`) — this is the same probabilistic encoding we saw in our article on Variational Autoencoders.

The `WorldModel` class is our Memory component (M). It takes the current latent state `z` and the action, concatenates them, and feeds them into an LSTM cell. The output is a prediction of the next latent state — again as a probability distribution (mean and log-variance).

This is exactly the V-M pipeline: compress observations, then predict future states in the compressed space.

Not bad, right? But can we push this idea further?

---

## Learning in Imagination — DreamerV3

The dream idea from Ha & Schmidhuber was powerful, but it was limited to simple environments. The question naturally arises: **Can we scale imagination-based learning to complex, diverse tasks?**

This brings us to the Dreamer series of algorithms, developed by Danijar Hafner and colleagues. The latest version, **DreamerV3**, was published in Nature in 2023 and achieved something remarkable — it mastered over **150 diverse tasks** with a single algorithm and a single set of hyperparameters.

Let us understand the key innovation: the **Recurrent State-Space Model (RSSM)**.

### The RSSM Architecture

The RSSM improves upon the simple RNN world model from Ha & Schmidhuber by combining two types of information:

1. A **deterministic** path that captures predictable dynamics (like a recurrent hidden state)
2. A **stochastic** path that captures uncertainty and randomness

The dynamics can be written as follows:


$$
h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1}) \quad \text{(deterministic path)} \\ z_t \sim q_\theta(z_t | h_t, x_t) \quad \text{(stochastic path -- encoder)} \\ \hat{z}_t \sim p_\theta(\hat{z}_t | h_t) \quad \text{(stochastic path -- prior/imagination)}
$$

Let us break this down.

The first equation is the deterministic path. It takes the previous hidden state $$h_{t-1}$$, the previous latent state $$z_{t-1}$$, and the previous action $$a_{t-1}$$, and produces a new deterministic hidden state $$h_t$$. This captures the predictable, consistent parts of the dynamics.

The second equation is the encoder. When we have access to the real observation $$x_t$$, we use it to compute a stochastic latent state $$z_t$$. This is used during training when real data is available.

The third equation is the prior or imagination path. When we do **not** have access to real observations (i.e., when we are imagining), we predict the stochastic state from the deterministic state alone. This is what makes "dreaming" possible.


![RSSM deterministic and stochastic paths](figures/figure_05.png)
*The Recurrent State-Space Model combines a deterministic recurrent path with a stochastic latent path, enabling both training from real data and imagination without observations.*


Why is this important? Because the world is both predictable and unpredictable. If you push a ball, you can predict its general trajectory (deterministic), but small variations in friction and spin add randomness (stochastic). The RSSM captures both.

### Imagination-Based Training

Now, here is where DreamerV3 truly shines.

Once the world model is trained, the actor (policy) and critic (value function) are trained **entirely on imagined trajectories.** The world model "rolls forward" from a real state, imagining a sequence of future states, rewards, and episode termination signals for a fixed number of steps.

The agent does not interact with the real environment during this phase. It practices entirely inside its imagination.

This is massively more sample-efficient than model-free methods. Instead of collecting thousands of real-world experiences, the agent can generate millions of imagined experiences at almost no cost.

### Results: Diamond Collecting in Minecraft

DreamerV3 was tested on a notoriously difficult task: **collecting diamonds in Minecraft from scratch.** This requires a long sequence of steps — chopping trees, crafting planks, making a crafting table, building a wooden pickaxe, mining stone, making a stone pickaxe, mining iron, smelting iron, making an iron pickaxe, and finally mining diamonds.

No hand-crafted rewards. No curriculum. Just a single sparse reward for collecting a diamond.

DreamerV3 solved this task, making it the first algorithm to collect diamonds in Minecraft without human demonstrations or pre-programmed knowledge.


![DreamerV3 diamond collection in Minecraft](figures/figure_06.png)
*DreamerV3 learns the full Minecraft crafting progression -- from chopping trees to collecting diamonds -- entirely through imagination-based training.*


This is truly amazing. The agent learned an entire crafting progression by imagining future outcomes inside its learned world model.

But now let us take a step back and ask a deeper question.

---

## Predicting in Abstract Space — JEPA

The world models we have seen so far — VAEs, RNNs, RSSMs — all try to predict future observations at the pixel level or in a latent space that can be decoded back to pixels.

But do we really need to predict every detail?

Yann LeCun, one of the pioneers of deep learning, raised this important critique. His argument is simple yet profound:

**When you imagine catching a ball, you do not simulate every blade of grass in the field. You predict the trajectory of the ball in abstract terms — direction, speed, where it will land.**

Your brain predicts at the level of **what matters**, ignoring irrelevant details.

This is the core idea behind the **Joint Embedding Predictive Architecture**, or JEPA.

### How JEPA Works

Instead of predicting future observations in pixel space, JEPA predicts in an **abstract representation space.** The architecture works as follows:

1. A **context encoder** transforms the current observation into an embedding
2. A **target encoder** transforms the target observation (what we want to predict) into an embedding
3. A **predictor module** predicts the target embedding from the context embedding


$$
\hat{s}_y = f_\theta(s_x, c)
$$


Here, $$s_x$$ is the embedding of the context (current observation), $$c$$ is a conditioning variable (like a mask indicating which part to predict), and $$\hat{s}_y$$ is the predicted embedding of the target.

The key difference from generative models is this: JEPA **never reconstructs pixels.** It operates entirely in the space of abstract representations.


![Generative prediction vs JEPA prediction](figures/figure_07.png)
*Generative models reconstruct full pixel details, while JEPA predicts only in abstract embedding space, ignoring irrelevant visual information.*


### I-JEPA and V-JEPA

This idea has been implemented in two major systems:

**I-JEPA (Image JEPA):** Given an image with masked regions, I-JEPA predicts the abstract representation of the masked regions — not the pixels themselves. This allows it to learn rich visual understanding without generating any images.

**V-JEPA (Video JEPA):** Extends this to the temporal domain. Given a video with masked space-time regions, V-JEPA predicts the abstract representations of the missing parts. This allows it to learn about physical dynamics — how objects move, how gravity works, how things interact — all from unlabeled video.


![I-JEPA masked patch prediction](figures/figure_08.png)
*I-JEPA predicts abstract embeddings of masked image patches using a context encoder and predictor, with a stop-gradient target encoder providing ground truth.*


The key advantage of predicting in abstract space is that the model can **ignore irrelevant details** and focus on what actually matters for understanding the world. Pixel-level prediction wastes capacity on modeling textures, lighting variations, and other details that do not matter for decision-making.

This makes sense because if you are a robot trying to pick up a mug, you do not need to predict the exact reflection pattern on the mug's surface — you just need to know where the mug is and where the handle is.

Now, the story gets even more exciting.

---

## Interactive Worlds — Genie

So far, the world models we have discussed take an action and predict **what happens next.** But what if the world model could generate entire interactive worlds that you can step into and explore?

This is exactly what Google DeepMind's **Genie** does.

### Genie 1 (2024)

Genie 1 was trained on hundreds of thousands of hours of internet platformer game videos. Given a single image — say, a sketch of a game level — Genie can generate a **fully playable game** from it.

The model learns three things from unlabeled video:
1. A **latent action model** — it infers what actions were taken between frames, even though no action labels were provided
2. A **video tokenizer** — it compresses video frames into discrete tokens
3. A **dynamics model** — it predicts the next frame given the current frame and an action

The remarkable thing is that Genie learned all of this without ever being told what an "action" is. It figured out from the video data that there are consistent controls (move left, move right, jump) and learned to respond to them.

### Genie 2 and Genie 3 (2024–2025)

Genie 2 scaled this idea dramatically. Trained on a large-scale video dataset, it can generate action-conditioned video for **up to a minute** at a time. It generates new content on the fly and maintains a consistent world. It even **remembers parts of the world that go out of view** and renders them accurately when they become visible again.

Genie 3 pushed even further — the first real-time interactive world model, generating navigable 3D worlds at 24 frames per second.


![Genie interactive world generation pipeline](figures/figure_09.png)
*Genie takes a text prompt, generates a starting frame, and then produces a fully playable interactive world driven by user actions.*


Why does this matter? Three reasons:

**Unlimited training environments:** Instead of hand-crafting simulation environments for training RL agents, we can now generate them on the fly. Describe a world, generate it, and train an agent inside it.

**Creative applications:** Describe a world in natural language, generate an image, and then step into it and interact with it.

**Evaluation:** We can test how well agents generalize by generating novel environments they have never seen before.


![Diverse worlds generated by Genie 2/3](figures/figure_10.png)
*Genie 2 and 3 can generate diverse interactive environments spanning 2D platformers, 3D driving scenes, first-person interiors, and more.*


Genie represents a paradigm shift: the world model is no longer just a tool for planning — it is a **generator of entire realities.**

This brings us to the most exciting development in this space.

---

## The Convergence — Vision-Language-Action Models

All the pieces are now coming together.

World models can imagine future states. Language models can understand goals and instructions. Vision models can perceive the world. The natural question is: **Can we build a single system that sees, understands, and acts?**

This is exactly what **Vision-Language-Action (VLA) models** do.

### Physical Intelligence's π0

In late 2024, Physical Intelligence released **π0** (pi-zero) — a single model that takes camera images and language instructions as input, and directly outputs continuous robot actions at 50 Hz.

Let us understand what this means.

Imagine you have a Franka robot arm in front of you, and you say: **"Pick up the red mug and place it on the shelf."**

π0 does the following:
1. **Sees** the scene through the robot's camera
2. **Understands** the instruction through its language component
3. **Generates** a smooth sequence of joint angles and gripper positions — 50 times per second — to execute the task

A single model. No separate perception module, no separate planner, no separate controller. One model that maps vision and language to action.

π0 was trained on data from **7 different robotic platforms** across **68 unique tasks**, including laundry folding, table bussing, grocery bagging, and object retrieval.

The model uses **flow matching** to generate smooth action trajectories. The simplified idea is:


$$
v_\theta(x_t, t) : \text{noise} \to \text{action trajectory}
$$

The model learns a vector field $$v_\theta$$ that transforms random noise into a smooth action trajectory. During inference, the model starts from noise and iteratively refines it into a precise sequence of robot actions. This is similar in spirit to how diffusion models generate images — but here, we are generating **actions** instead of pixels.


![Vision-Language-Action model architecture](figures/figure_11.png)
*A VLA model takes camera images, language instructions, and robot state as input, and outputs continuous robot actions through a shared vision-language backbone and action decoder.*


### Meta's V-JEPA 2

Remember JEPA from the previous section? Meta took this architecture and pushed it into the world of robotics.

**V-JEPA 2** was released in January 2026 and achieved something remarkable:

- Trained on over **1 million hours** of internet video using self-supervised learning
- Achieved **65–80% success rates** on pick-and-place tasks with **unfamiliar objects** in **novel environments**
- Required only **62 hours** of unlabeled robot video for action conditioning
- Deployed **zero-shot** on Franka robot arms in two different labs — no task-specific training, no reward engineering

Let that sink in. The model watched internet videos, learned how the physical world works, saw a tiny amount of robot video, and then successfully controlled a real robot in an environment it had never seen before.

This is the power of world models combined with action generation. The model has built such a rich internal representation of physical dynamics that it can transfer that understanding to robot control with minimal additional training.

---

## The Big Picture

Now let us step back and see the complete picture of how this field has evolved.


![Evolution of world models from 2018 to 2026](figures/figure_12.png)
*The progression from dreaming agents (2018) through imagination-based RL, abstract prediction, interactive world generation, and finally Vision-Language-Action models for real robot control.*


The evolution follows a clear arc:

1. **2018 — World Models:** The agent learns a model of the environment and trains inside its own dream.
2. **2020–2023 — Dreamer:** Imagination-based RL scales to 150+ diverse tasks, including long-horizon challenges like Minecraft.
3. **2022–2025 — JEPA:** Prediction moves from pixel space to abstract representation space, enabling more efficient and robust world understanding.
4. **2024–2025 — Genie:** World models become **generators of interactive realities**, creating entire environments from a single image.
5. **2025–2026 — VLAs:** Vision, language, and action converge into single models that can control real robots in the physical world.

The vision is becoming clear: AI systems that can **imagine, plan, and act** — the three pillars of intelligence.

We started this article with the simple observation that humans can imagine the consequences of their actions before taking them. We have now seen how this idea has been formalized into world models, scaled from simple games to complex environments, moved from pixel-level prediction to abstract reasoning, and ultimately connected to real-world robot control.

The next frontier? Building world models that can reason about long time horizons, understand causality, and generalize across entirely new environments — bringing us closer to truly autonomous agents that learn the way humans do.

In the next article, we will dive deep into one of these architectures and build a world model from scratch. We will cover this in detail in the next lecture.

See you next time!

---

**References:**

1. Ha, D. & Schmidhuber, J. (2018). *World Models.* arXiv:1803.10122
2. Hafner, D. et al. (2023). *Mastering Diverse Domains through World Models.* Nature.
3. LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* Technical Report.
4. Assran, M. et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR.
5. Bruce, J. et al. (2024). *Genie: Generative Interactive Environments.* arXiv:2402.15391
6. Black, K. et al. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164
7. Bardes, A. et al. (2026). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985
