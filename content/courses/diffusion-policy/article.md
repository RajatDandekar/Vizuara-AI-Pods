# Diffusion Policy: Teaching Robots to Act Using Diffusion Models

## How the denoising process from image generation became the secret weapon for robot learning

Vizuara AI

---

In the previous lecture, we had looked at the problem of imitation learning for robotics. We discussed how a robot can learn from human demonstrations instead of learning through trial and error.

Refer to the previous article here: [Robot Imitation Learning](https://vizuara.substack.com/p/modern-robot-learning-lecture-1-robot)

We encountered a very important problem towards the end of that lecture. Let us revisit it.

Remember our tree obstacle example? A robot is navigating a path, and there is a large tree in the way. A human expert records demonstrations — sometimes steering left around the tree, sometimes steering right.


![Tree obstacle with multimodal paths](figures/figure_1.png)


Both "steer left" and "steer right" are perfectly valid actions. But when we train a simple supervised learning model on this data, what happens?

The model sees the same state ("tree ahead") with two different actions ("left" and "right"). To minimize its total error, it takes the average: "go straight."

And the robot crashes right into the tree.

We called this the **multi-modality problem**. As we discussed:

> "Point-estimate policies typically fail to learn multimodal targets, which are very common in human demonstrations solving real-world robotics problems, as multiple trajectories can be equally as good towards the accomplishment of a goal."

We ended that lecture by saying that we need policies which can capture the **distribution** of all possible actions, not just the average.

So, how do we learn such a distribution? This brings us to today's topic.

What if we used **diffusion models** — the same technique we studied for image generation — to generate robot actions?

This is exactly the idea behind the **Diffusion Policy** paper by Chi et al. (RSS 2023), and it turns out to work remarkably well.

---

## From Images to Actions — The Key Insight

Let us start by recalling what we learned about diffusion models.

Refer to this article on Diffusion Models: [What exactly are Diffusion Models?](https://vizuara.substack.com/p/what-exactly-are-diffusion-models)

In that article, we took the Batman image and gradually added Gaussian noise to it, step by step, until it became pure noise. Then we trained a neural network to reverse this process — to denoise the image step by step and recover the original Batman image.

Now here is the key insight:

**Actions are just numbers too.**

Think about it. When a robot moves its arm, the action at each time step is just a set of numbers — joint angles, velocities, or end-effector positions. For example, for our SO-101 robot with 6 degrees of freedom, each action is simply a vector of 6 numbers.

And if we record a full trajectory of actions over time, say 16 time steps, we get a sequence of 16 such vectors. This is just a matrix of numbers — not too different from a grid of pixel values in an image!


![Image denoising vs action trajectory denoising](figures/figure_2.png)


So instead of:
- **Image generation:** Start from noise → denoise → get a clean image

We do:
- **Action generation:** Start from noise → denoise → get a clean action trajectory

The neural network learns to gradually transform random Gaussian noise into a smooth, meaningful action sequence that the robot can execute.

But there is one crucial difference. When we generate an image, we just generate it — there is no context needed. But when a robot generates actions, it needs to know what it is currently **observing**. The actions should depend on the current state of the world.

This means our denoising process needs to be **conditioned on observations**. The robot looks at its cameras, sees the current state of the environment, and then generates actions accordingly.

Now let us understand how this works mathematically.

---

## The Diffusion Policy Formulation

### The Forward Process: Adding Noise to Expert Actions

Let us say we have collected expert demonstrations. At each time step $t$, the expert performed a sequence of actions:


![Equation](equations/eq_1.png)


Here, $T_p$ is the **prediction horizon** — the number of future action steps in our sequence. We will talk more about this later.

Now, just like we did for images, we add Gaussian noise to this action sequence over $K$ steps:


![Equation](equations/eq_2.png)


At step $k = 0$, we have the clean expert actions. At step $k = K$, the actions are completely drowned in noise — they look like random Gaussian samples.

This should look very familiar from our Diffusion Models article! The only difference is that instead of noising pixel values, we are noising action values.

### The Reverse Process: Denoising to Get Actions

Now comes the interesting part. During inference — when the robot actually needs to act — we reverse this process.

We start from pure Gaussian noise:


![Equation](equations/eq_3.png)


And iteratively denoise it:


![Equation](equations/eq_4.png)


Let us understand what each term means:

- $A_t^k$ is the noisy action sequence at denoising step $k$
- $\epsilon_\theta$ is our **noise prediction network** — this is the neural network we train
- $O_t$ is the **current observation** — this is how the robot knows what it is looking at
- $k$ is the current denoising step
- $\alpha$, $\gamma$, and $\sigma$ are coefficients from the noise schedule

The key difference from standard image diffusion is the **observation conditioning** $O_t$. The noise prediction network takes three inputs: the current observation, the noisy actions, and the denoising step. This allows the model to generate actions that are appropriate for the current situation.

Have a look at the full pipeline below:


![Diffusion Policy pipeline](figures/figure_3.png)


### The Training Objective

How do we train the noise prediction network? This is exactly the same as DDPM for images.

During training, we:
1. Take a clean expert action sequence $A_t^0$
2. Sample a random denoising step $k$
3. Add the appropriate amount of noise to get $A_t^k$
4. Train the network to predict the noise that was added

The loss function is:


![Equation](equations/eq_5.png)


Here, $\epsilon^k$ is the actual noise we added, and $\epsilon_\theta(\cdot)$ is the noise predicted by our network.

This is the same MSE noise prediction loss we saw in the Diffusion Models article. The network simply learns to predict what noise was added to the clean actions.

This is exactly what we want — a simple, stable training objective that we already know works well for diffusion models.

---

## Action Chunking — Why Predict a Sequence?

Now the question is: why predict a whole sequence of actions? Why not just predict one action at a time, like most policies do?

There are three important reasons.

**Reason 1: Temporal Consistency**

If you predict one action per step, each prediction is made independently. This can lead to jerky, inconsistent behavior — the robot might twitch back and forth because consecutive predictions don't agree with each other.

By predicting a whole sequence, the actions are generated together and are naturally smooth and consistent.

**Reason 2: Multimodal Commitment**

This is perhaps the most important reason. Remember our tree obstacle? If you predict one action at a time, the model might waver — go a little left, then a little right, then a little left again. It never commits.

But if you predict an entire trajectory in one shot, the model must commit to a coherent plan. If the initial noise sends it toward the left, the whole trajectory goes left. No averaging, no wavering.

**Reason 3: Efficiency**

Generating actions through diffusion requires multiple denoising steps, which takes time. If you had to run the full denoising process for every single action, the robot would be too slow.

Instead, the robot generates a sequence of, say, 16 future actions, executes a few of them, and then replans. This is much more efficient.

### The Three Horizons

The Diffusion Policy paper defines three important parameters:

- **$T_o$ (Observation Horizon):** How many past observation steps we feed to the model. This gives the robot context about recent history.

- **$T_p$ (Prediction Horizon):** How many future action steps we predict. This is the total length of the generated action sequence.

- **$T_a$ (Action Horizon):** How many of those predicted steps we actually execute before replanning.

Have a look at the diagram below:


![The three horizons: observation, prediction, and action](figures/figure_4.png)


This is called **receding horizon control**. The robot predicts far into the future ($T_p$ steps), but only commits to a short window ($T_a$ steps). Then it replans with fresh observations. This gives a nice balance between long-horizon planning and responsiveness to new information.

The paper found that **an action horizon of 8 steps worked best** for most tasks. The policy can also tolerate latency of up to 4 steps without losing performance — which is important for real-world deployment.

---

## Two Network Architectures

The Diffusion Policy paper proposes two different neural network architectures for the noise prediction network $\epsilon_\theta$. Let us look at both.

### CNN-Based: 1D Temporal U-Net

The first architecture uses a **1D convolutional network** that operates along the time dimension of the action sequence.

The key idea is that we treat the action sequence as a 1D signal (similar to audio) and apply convolutional layers to it. The observation is injected into every convolutional layer using a technique called **FiLM (Feature-wise Linear Modulation)** — which essentially scales and shifts the convolutional features based on the observation embedding.

This architecture works well when actions are smooth and continuous, such as position control tasks.

### Transformer-Based: Diffusion Transformer

The second architecture uses a **Transformer decoder**. Here, each action in the sequence is treated as a token. The noisy action tokens are fed into transformer decoder blocks, and the observation embeddings enter through **cross-attention**.

The diffusion step $k$ is encoded using a sinusoidal positional embedding and prepended as the first token. Causal masking ensures that each action token only attends to itself and previous actions.

This architecture is better suited for tasks where actions change rapidly, such as velocity control.


![CNN-based vs Transformer-based architectures](figures/figure_5.png)


Both architectures were tested extensively, and the results show that the CNN-based approach works better for most tasks, while the Transformer shines in velocity-control settings.

---

## How Diffusion Policy Handles Multimodality

Now let us come to the main character in our story — how does Diffusion Policy solve the multimodality problem that we started with?

The answer is beautifully simple.

Remember that during inference, we start from **random Gaussian noise** $A_t^K$. This randomness is the key.

Different random noise initializations lead the denoising process to converge to **different modes** of the action distribution. If the noise happens to push the trajectory toward the "go left" basin, the entire denoised sequence will be a "go left" trajectory. If the noise pushes it toward "go right," the whole sequence commits to going right.

There is no averaging. Each rollout samples one coherent mode.

Let us see this in the **Push-T task** — the signature benchmark of this paper.

The task is simple to describe: you have a T-shaped block (shown in gray) and a target location (shown in red). A circular end-effector (shown in blue) must push the T-block into the target.


![Push-T: Two equally valid approaches](figures/figure_6.png)


Now, the end-effector can approach the T-block from the left or the right — both are valid strategies that appear in the demonstration data.

When we run Diffusion Policy on this task:
- In some rollouts, the random initial noise causes the policy to approach from the left
- In other rollouts, the random noise causes it to approach from the right
- In each individual rollout, the policy fully commits to one approach

This is exactly what we want!

Compare this with baseline methods:
- **LSTM-GMM** (a popular baseline): Only achieves 20% success in the real world — it often fails to commit to a single mode
- **IBC** (Implicit Behavior Cloning): Achieves 0% success in the real world — it struggles with the high-dimensional action space
- **Diffusion Policy:** Achieves **95% success rate** in the real world

Not bad right? :)

---

## Results — 46.9% Average Improvement

The Diffusion Policy paper tested their method across **12 tasks** spanning **4 different robot manipulation benchmarks**. The results are impressive.

### Simulation Results


![Diffusion Policy vs Baselines — success rates across tasks](figures/figure_7.png)


Look at the ToolHang task — one of the hardest manipulation tasks, requiring the robot to hang a tool onto a hook. Diffusion Policy achieves 1.0 success rate, while LSTM-GMM scores 0.0. This is truly amazing.

### Real-World Results

The paper also demonstrates Diffusion Policy on real robots:

- **Push-T:** 95% success (vs 20% for LSTM-GMM)
- **Mug Flipping:** 90% success — flipping a mug near the robot's kinematic limits
- **Sauce Pouring:** 79% IoU — pouring sauce with periodic actions
- **Sauce Spreading:** 100% success — spreading sauce with a spatula

### Fast Inference with DDIM

One concern with diffusion models is the inference speed — running $K$ denoising steps takes time. The paper addresses this using **DDIM** (Denoising Diffusion Implicit Models), which decouples the number of training and inference steps.

During training, the model uses 100 denoising steps. During inference, it only needs **10 steps** — achieving 0.1 second latency on an Nvidia 3080 GPU. This is fast enough for real-time robot control at 10 Hz.

---

## Summary and What's Next

Let us summarize what we have learned.

**Diffusion Policy** takes the core idea of diffusion models — iterative denoising — and applies it to robot action generation. The key ingredients are:

1. **Action sequence denoising:** Instead of denoising images, denoise sequences of robot actions
2. **Observation conditioning:** The denoising process is conditioned on what the robot currently sees
3. **Action chunking:** Predict a sequence of actions, execute a few, then replan
4. **Natural multimodality:** Different random noise leads to different valid action modes

The result is a robot learning framework that achieves state-of-the-art performance with a 46.9% average improvement across diverse manipulation benchmarks.

But can we make this even faster? What if, instead of taking $K$ denoising steps, we could generate actions in a **single step**? Does this remind you of something? This brings us to the concept of **Flow Matching** for robot policies, which we will explore in the next lecture.

See you next time!

---

**References:**

- Chi, C. et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023 / IJRR 2024. [Paper](https://arxiv.org/abs/2303.04137) | [Code](https://github.com/real-stanford/diffusion_policy) | [Project Page](https://diffusion-policy.cs.columbia.edu/)

- Previous Vizuara lectures: [Imitation Learning](https://vizuara.substack.com/p/modern-robot-learning-lecture-1-robot) | [Diffusion Models](https://vizuara.substack.com/p/what-exactly-are-diffusion-models) | [Denoising Score Matching](https://vizuara.substack.com/p/what-exactly-is-denoising-score-matching)
