# Understanding World Models from Scratch

## How AI agents learn to dream about their environment — and use those dreams to make better decisions

Let us start with a simple thought experiment. Imagine you are standing at the edge of a busy road, waiting to cross. Cars are zooming past you in both directions. Before you step off the curb, what does your brain do?

It runs a simulation.

Without even thinking about it, your brain predicts the trajectories of the cars. It estimates when a gap will appear. It imagines what will happen if you step forward *right now* — and decides to wait two more seconds.

You are running a **world model** inside your head.

This ability is not unique to crossing roads. Think about a chess grandmaster: before making a move, they play out entire sequences of moves in their imagination — "If I move my bishop here, my opponent will likely move their queen there, and then I can..." They are simulating future board states without physically touching the pieces.


![Before acting, our brain simulates possible futures.](figures/figure_1.png)
*Before acting, our brain simulates possible futures.*


Now the question is: **Can we give an AI agent this same ability — to imagine what will happen before it acts?**

This is exactly what a **World Model** does. It is an internal representation of the environment that the agent builds inside itself. The agent uses this internal model to simulate experiences, predict outcomes, and plan its actions — all without interacting with the real world.

Let us understand how this works, step by step.

## Model-Free vs Model-Based: Two Ways to Learn

Before we dive into world models, let us quickly understand the two broad approaches to reinforcement learning.

In reinforcement learning, an agent interacts with an environment. At each time step, the agent observes a **state**, takes an **action**, and receives a **reward**. The goal is to learn a policy — a strategy for choosing actions — that maximizes the total reward over time.

Now, there are two fundamentally different philosophies for learning this policy.

**Approach 1: Model-Free RL (Learn by Doing)**

In model-free RL, the agent learns purely through trial and error. It interacts with the real environment, observes the results, and updates its policy directly.

Think of it like a baby learning that a hot stove is dangerous. The baby touches the stove, feels pain (negative reward), and learns not to do it again. The baby never builds an internal model of "heat transfer" — it simply learns the mapping: hot stove → do not touch.

Model-free methods like Q-Learning, REINFORCE, and PPO all fall into this category.

**Approach 2: Model-Based RL (Learn by Imagining)**

In model-based RL, the agent first builds an **internal model** of how the world works. Once it has this model, it can simulate experiences inside its own imagination and plan its actions before executing them in the real world.

Think of it like an engineer designing a bridge. The engineer does not build 50 bridges and see which one collapses. Instead, they build a simulation of the bridge on a computer and test thousands of configurations virtually. Only the best design gets built in the real world.


![Model-free agents learn by doing; model-based agents learn by imagining.](figures/figure_2.png)
*Model-free agents learn by doing; model-based agents learn by imagining.*


The key advantage of model-based RL is **sample efficiency**. Since the agent can practice in its imagination, it needs far fewer interactions with the real environment.

The environment can be described mathematically as a transition function:


$$
s_{t+1} = f(s_t, a_t)
$$


This equation says: the next state $s_{t+1}$ depends on the current state $s_t$ and the action $a_t$ taken by the agent.

Let us plug in some simple numbers. Imagine a 1D grid world where the state is your position (an integer from 0 to 9), and you have two actions: move left (-1) or move right (+1).

If you are at position $s_t = 3$ and take action $a_t = +1$ (move right):


$$
s_{t+1} = f(3, +1) = 3 + 1 = 4
$$


Simple! But in the real world, the transition function $f$ is unknown — images change in complex ways when a robot moves its arm. The question is: **can we learn this function from data?**

This brings us to the main topic of our article.

## The World Model Architecture: V, M, and C

In 2018, Ha and Schmidhuber published a landmark paper called "World Models." They proposed a beautifully simple architecture made up of three components:

1. **V (Vision)** — a Variational Autoencoder (VAE) that compresses raw images into a compact code
2. **M (Memory)** — a Recurrent Neural Network (RNN) that learns to predict how the world evolves over time
3. **C (Controller)** — a simple linear policy that decides what actions to take

Here is the key insight: **the controller never sees raw pixels.** It lives entirely in a compressed "dream space" created by V and M. All the complexity of the visual world is handled by the vision and memory components, so the controller can be incredibly simple.


![The World Model: Vision compresses, Memory predicts, Controller decides.](figures/figure_3.png)
*The World Model: Vision compresses, Memory predicts, Controller decides.*


Let us now understand each of these three components in detail.

## The Vision Component (V): Seeing in Compressed Form

Imagine you are a talented sketch artist. Someone shows you a detailed 4K photograph of a race track. In just a few seconds, you capture the essence of the scene in a quick pencil sketch — the road curve, the grass on either side, the position of the car. That sketch is much simpler than the original photograph, but it contains all the information needed to understand the scene.

This is exactly what the VAE does. It takes a high-dimensional image (64×64 pixels with 3 color channels = 12,288 numbers) and compresses it into a tiny **latent code** $z$ — a vector of just 32 numbers!

How does it work? The VAE has two parts:

- **Encoder**: Takes the image $x$ as input and outputs the parameters of a probability distribution in latent space — specifically, a mean $\mu$ and standard deviation $\sigma$
- **Decoder**: Takes a sample $z$ from this distribution and reconstructs the original image

The encoder does not output a single point — it outputs a distribution. Why? Because there is inherent uncertainty in compression. Multiple slightly different images could map to the same latent code.

To sample from this distribution during training (while still allowing gradients to flow), we use the **reparameterization trick**:


$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$


Let us plug in some simple numbers. Suppose for a particular car racing frame, the encoder outputs $\mu = 0.5$ and $\sigma = 0.2$. We sample $\epsilon = 0.3$ from a standard normal distribution:


$$
z = 0.5 + 0.2 \times 0.3 = 0.56
$$


So this particular dimension of our latent code is 0.56. We do this for all 32 dimensions to get our full latent vector $z$.


![The VAE compresses 12,288 pixel values into just 32 latent numbers.](figures/figure_4.png)
*The VAE compresses 12,288 pixel values into just 32 latent numbers.*


The VAE is trained by optimizing two objectives simultaneously:

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q(z|x)}[-\log p(x|z)]}_{\text{Reconstruction Loss}} + \underbrace{D_{\text{KL}}(q(z|x) \| p(z))}_{\text{Regularization Loss}}
$$

The first term says: **reconstruct well** — the decoded image should look like the original. The second term says: **keep the latent space organized** — the encoder's distribution should stay close to a standard Gaussian.

Let us plug in some simple numbers. Suppose for a given image, the reconstruction loss is 150 (the decoded image differs from the original) and the KL divergence is 8 (the encoder's distribution has drifted from a standard Gaussian):

$$
\mathcal{L}_{\text{VAE}} = 150 + 8 = 158
$$

The optimizer will try to reduce both terms. Over training, the reconstruction gets sharper (loss drops from 150 to, say, 20) and the latent space stays well-organized (KL stays around 5–10). This is exactly what we want.

After training, the VAE can compress any 64×64 car racing frame into a 32-dimensional vector $z$. This is a compression ratio of over 384:1! The agent can now "see" the world through this compact representation.

## The Memory Component (M): Learning to Predict the Future

Now our agent can see in compressed form. But **can it predict what comes next?**

Think about watching a movie. If you pause the movie at any point, your brain can make a reasonable guess about what the next frame will look like. If a ball is flying through the air, you predict it will continue on its trajectory. If a character is reaching for a door handle, you predict the door will open.

The Memory component does exactly this. Given the current latent state $z_t$ and an action $a_t$, it predicts the distribution of the **next** latent state $z_{t+1}$.

The Memory is an LSTM (a type of recurrent neural network) combined with a **Mixture Density Network (MDN)**.

### Why a Mixture Density Network?

Here is an important insight: the future is not deterministic. When you are driving down a road and you see an intersection ahead, the road could curve left **or** right. A single Gaussian prediction would average these two outcomes and predict "straight ahead" — which is wrong in both cases!

An MDN solves this by predicting a **mixture of Gaussians** — multiple possible futures, each with its own probability.


![The MDN-RNN predicts multiple possible futures as a mixture of Gaussians.](figures/figure_5.png)
*The MDN-RNN predicts multiple possible futures as a mixture of Gaussians.*


The MDN output is a weighted sum of $K$ Gaussian components:


$$
P(z_{t+1}) = \sum_{i=1}^{K} \pi_i \cdot \mathcal{N}(z_{t+1} \mid \mu_i, \sigma_i)
$$


Here, $\pi_i$ is the mixing coefficient (how likely future $i$ is), and $\mu_i$, $\sigma_i$ are the mean and standard deviation of each Gaussian component.

Let us plug in some simple numbers with $K = 3$ components (three possible futures):

- Future 1: $\pi_1 = 0.6$, $\mu_1 = 2.0$, $\sigma_1 = 0.3$ (road curves left — most likely)
- Future 2: $\pi_2 = 0.3$, $\mu_2 = -1.5$, $\sigma_2 = 0.4$ (road curves right)
- Future 3: $\pi_3 = 0.1$, $\mu_3 = 0.0$, $\sigma_3 = 0.8$ (road goes straight — least likely)

Note that the mixing coefficients sum to 1: $0.6 + 0.3 + 0.1 = 1.0$. This is exactly what we want — they represent a valid probability distribution.

To predict the next latent state, we first pick which Gaussian to sample from (with probability $\pi_i$), then sample from that Gaussian. This naturally captures the multimodal nature of the future.


![The MDN captures multiple possible futures — the road could curve left, right, or go straight.](figures/figure_6.png)
*The MDN captures multiple possible futures — the road could curve left, right, or go straight.*


The LSTM hidden state $h_t$ is crucial here. It acts as the agent's **memory** — it accumulates information from all previous time steps. This means the prediction is not based on just the current frame, but on the entire history of observations and actions.

## The Controller (C): Keeping It Simple

We have a vision system (V) and a memory system (M). Now we need a decision-maker.

Here is the surprise: **the Controller is just a single linear layer.**


$$
a_t = W_c \cdot [z_t, h_t] + b_c
$$


That is it. The latent code $z_t$ (32 dimensions) and the RNN hidden state $h_t$ (256 dimensions) are concatenated into a single vector of 288 dimensions. This vector is multiplied by a weight matrix $W_c$ and a bias $b_c$ is added to produce the action.

Let us plug in some simple numbers. Suppose we have a tiny example where $z_t = [0.5, -0.3]$ and $h_t = [0.8, 0.1]$, and the weights and bias are:


$$
a_t = \begin{bmatrix} 0.2 & -0.1 & 0.5 & 0.3 \end{bmatrix} \cdot \begin{bmatrix} 0.5 \\ -0.3 \\ 0.8 \\ 0.1 \end{bmatrix} + 0.05
$$



$$
a_t = (0.1 + 0.03 + 0.4 + 0.03) + 0.05 = 0.61
$$


So the controller outputs 0.61, which could represent a steering angle. Not bad for a single matrix multiplication!

Why is the controller so simple? Because all the complexity is already handled by V and M. The VAE has compressed the visual world into a meaningful latent code, and the RNN has learned the dynamics of the environment. The controller just needs to make a simple decision based on this already-processed information.


![The Controller is a single linear layer — all complexity lives in V and M.](figures/figure_7.png)
*The Controller is a single linear layer — all complexity lives in V and M.*


In the original paper on CarRacing, the controller has only about **867 parameters**. Compare this to the millions of parameters in the VAE and RNN! The entire decision-making logic fits in fewer parameters than a single layer of a typical neural network.

But how do we train these 867 parameters? The authors use **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy), a derivative-free optimization method. The idea is simple and elegant:

1. Generate a population of controllers with random parameters
2. Test each controller (in the dream, as we will see next)
3. Keep the best performers
4. Generate a new population centered around the best parameters
5. Repeat

Think of it like natural selection — the fittest controllers survive and pass on their "genes" to the next generation.

## Training Inside the Dream: The Key Insight

Here is where things get truly fascinating.

Remember the three components: V (Vision), M (Memory), and C (Controller). The training happens in three stages:

**Stage 1: Collect real data.**
The agent interacts with the real environment using a random policy and collects a dataset of observations.

**Stage 2: Train V and M.**
The VAE is trained on the collected observations to learn the compressed representation. Then the MDN-RNN is trained on sequences of $(z_t, a_t, z_{t+1})$ to learn the dynamics of the world.

**Stage 3: Train C inside the dream.**
This is the key insight. The controller is trained **entirely inside the learned world model** — never touching the real environment!

Here is how it works: the Memory model (M) can generate imagined sequences. Given a starting latent state $z_0$ and a sequence of actions, the MDN-RNN predicts what will happen at each step — producing a "dream." The controller is trained on these dreams, not on real experience.


![First learn how the world works, then train by dreaming.](figures/figure_8.png)
*First learn how the world works, then train by dreaming.*


The analogy here is a flight simulator. Real flight data was used to build the simulator, but a pilot can train for thousands of hours in the simulator without ever flying a real airplane. The risks, the fuel costs, the time — all are eliminated.

The advantages of this approach are profound:

1. **Speed**: Simulating the world inside a neural network is orders of magnitude faster than running the actual environment
2. **Safety**: The agent cannot break anything in a dream
3. **Scalability**: You can run thousands of dream rollouts in parallel

But there is one important limitation: **if the world model is inaccurate, the controller may learn to exploit "bugs" in the dream.** Imagine a pilot who learns to fly in a simulator where gravity is slightly wrong — their skills may not transfer to the real world. This is called the "model exploitation" problem, and it is an active area of research.

## Practical Implementation: World Models on Car Racing

Enough theory, let us look at some practical implementation now.

We will implement a simplified World Model for the CarRacing environment from OpenAI Gymnasium. This is a top-down driving game where the agent sees a 64×64 pixel view of a race track and must steer, accelerate, and brake.


![The CarRacing environment: the agent sees a 64×64 top-down view and must navigate the track.](figures/figure_9.png)
*The CarRacing environment: the agent sees a 64×64 top-down view and must navigate the track.*


First, let us define the VAE:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder: 64x64x3 -> latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

        # Decoder: latent_dim -> 64x64x3
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),  nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),   nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),    nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps       # The reparameterization trick!

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 256, 2, 2)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

Let us understand this code in detail. The encoder is a stack of convolutional layers that progressively reduce the spatial dimensions: 64→30→14→6→2. The output is flattened and passed through two linear layers to produce $\mu$ and $\log \sigma^2$. The `reparameterize` function implements the reparameterization trick we discussed earlier. The decoder reverses this process using transposed convolutions.

Next, the MDN-RNN:

```python
class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.fc_pi = nn.Linear(hidden_dim, latent_dim * n_gaussians)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim * n_gaussians)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim * n_gaussians)
        self.latent_dim = latent_dim
        self.n_gaussians = n_gaussians

    def forward(self, z, a, hidden=None):
        # z: (batch, seq_len, latent_dim), a: (batch, seq_len, action_dim)
        x = torch.cat([z, a], dim=-1)
        h, hidden = self.lstm(x, hidden)

        K = self.n_gaussians
        pi = self.fc_pi(h).view(-1, self.latent_dim, K)
        pi = F.softmax(pi, dim=-1)        # Mixing coefficients sum to 1
        mu = self.fc_mu(h).view(-1, self.latent_dim, K)
        sigma = torch.exp(self.fc_sigma(h).view(-1, self.latent_dim, K))

        return pi, mu, sigma, hidden
```

Here you can see the LSTM takes the concatenation of $z_t$ and $a_t$ as input. The MDN head produces three outputs for each of the $K=5$ Gaussian components: the mixing coefficients $\pi$ (passed through softmax so they sum to 1), the means $\mu$, and the standard deviations $\sigma$ (exponentiated to ensure they are positive). This is exactly the mixture of Gaussians formula we saw earlier.

Finally, the Controller:

```python
class Controller(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return torch.tanh(self.fc(x))    # Actions in [-1, 1]

# Count parameters: (32 + 256) * 3 + 3 = 867 parameters!
controller = Controller()
print(f"Controller parameters: {sum(p.numel() for p in controller.parameters())}")
```

Notice how simple the Controller is — just a single linear layer with a tanh activation. The total number of parameters is $(32 + 256) \times 3 + 3 = 867$. Just 867 parameters to drive a car!


![The Controller improves steadily as it trains inside the learned dream, approaching human-level performance.](figures/figure_10.png)
*The Controller improves steadily as it trains inside the learned dream, approaching human-level performance.*



![Real observations (top) vs. what the agent imagines (bottom) — the dreams capture the essence of the track.](figures/figure_11.png)
*Real observations (top) vs. what the agent imagines (bottom) — the dreams capture the essence of the track.*


The trained agent achieves a score of around 900 on CarRacing — competitive with model-free methods that require orders of magnitude more environment interactions. Not bad for an agent that learned to drive mostly in its dreams!

## Modern World Models: Where the Field Went Next

The 2018 World Models paper was just the beginning. Let us look at how the field evolved over the following years.

**Dreamer (Hafner et al., 2020)** replaced CMA-ES with backpropagation through the dream. Instead of treating the controller training as a black-box optimization problem, Dreamer directly computes gradients through the learned dynamics model. This makes training much more efficient.

**DreamerV2 (2021)** introduced discrete latent representations instead of continuous ones — using categorical variables rather than Gaussians. This achieved human-level performance on the Atari benchmark for the first time with a world model approach.

**DreamerV3 (2023)** was perhaps the most impressive: a single algorithm with a single set of hyperparameters that works across radically different domains — Atari games, continuous control tasks, and even **Minecraft**. An agent that can mine diamonds in Minecraft using a learned world model is truly remarkable.

**JEPA (LeCun, 2022)** took the idea even further at a philosophical level. Yann LeCun proposed that world models are the path to general AI — but with a twist: instead of predicting in pixel space (which is extremely detailed and wasteful), we should predict in **latent space**. This is exactly what the original World Models paper does, and LeCun argued this principle should be the foundation of next-generation AI systems.


![The rapid evolution of world models from 2018 to 2023.](figures/figure_12.png)
*The rapid evolution of world models from 2018 to 2023.*


The common thread across all of these works is a simple but powerful idea: **agents that can imagine and plan outperform agents that only react.**

## Conclusion

Let us step back and appreciate what we have built. A World Model is an AI agent that learns to:

1. **See** — compress the visual world into a compact representation (V)
2. **Remember and Predict** — learn how the world evolves over time (M)
3. **Decide** — choose actions based on the compressed state and predictions (C)

The most remarkable insight is that once V and M are trained, the controller can be trained **entirely inside the dream** — no real-world interaction needed. This mirrors how humans rehearse scenarios mentally before acting.

World models represent a fundamentally different approach to AI — one that is closer to how biological brains work. Instead of learning reactive mappings from inputs to outputs, we first build an internal model of reality and then use that model to plan.

That's it!

## References

- Ha and Schmidhuber, "World Models" (2018)
- Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet, 2019)
- Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer, 2020)
- Hafner et al., "Mastering Atari with Discrete World Models" (DreamerV2, 2021)
- Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023)
- LeCun, "A Path Towards Autonomous Machine Intelligence" (JEPA, 2022)
- Kingma and Welling, "Auto-Encoding Variational Bayes" (2014)
