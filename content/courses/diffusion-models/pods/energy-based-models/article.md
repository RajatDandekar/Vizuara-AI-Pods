# Energy Based Models and the Score Function

*How an energy landscape, a gradient, and a drunk hiker unlock modern generative modeling*

---

## The Energy Landscape

Let us start with a simple example. Imagine you are holding a ball at the top of a hilly landscape. What happens when you let go? The ball rolls downhill and settles at the lowest point. This is a fundamental principle from physics: systems naturally settle into configurations that minimize their energy.

Now, here is a powerful idea: **what if we could use this same principle to describe data?**

Think about it this way. Suppose we have a collection of images — say, photographs of cats. These images are not random pixel noise. They live in very specific regions of "pixel space." What if we assigned an energy value to every possible image, such that real cat images have low energy and random noise has high energy?

This is the core idea behind **Energy-Based Models (EBMs)**. We define an energy function, denoted as:


$$
E_\theta(x)
$$


Here, $x$ is a data point (an image, a sentence, a molecular configuration — anything), $\theta$ represents the learnable parameters, and $E_\theta(x)$ outputs a single scalar: the energy of that configuration.

Let us plug in some simple numbers to see how this works. Suppose we have three states: $x_1$, $x_2$, and $x_3$ with energies $E(x_1) = 5$, $E(x_2) = 1$, and $E(x_3) = 3$. Since lower energy means more likely, we would expect $x_2$ to be the most probable configuration, followed by $x_3$, and then $x_1$. This makes sense because $x_2$ has the lowest energy.


![Energy and probability are inversely related.](figures/figure_1.png)
*Energy and probability are inversely related.*


But how do we convert this energy function into an actual probability distribution? Some things we know intuitively:

- Points with **low energy** should have a **higher probability**.
- Points with **high energy** should have a **lower probability**.
- The probability should always be positive.
- The probabilities should sum (or integrate) to 1.

People use an exponential function to relate energy to probability, since it satisfies all these properties. The relationship is given by the **Boltzmann distribution**:


$$
p(x) = \frac{\exp(-E_\theta(x))}{Z}
$$

where $Z = \int \exp(-E_\theta(x)) \, dx$ is called the **partition function** — it is simply the normalizing constant that ensures the probabilities integrate to 1.

Let us plug in some simple numbers. Take three discrete states with energies $E(x_1) = 3$, $E(x_2) = 1$, $E(x_3) = 2$:

- $\exp(-E(x_1)) = \exp(-3) = 0.050$
- $\exp(-E(x_2)) = \exp(-1) = 0.368$
- $\exp(-E(x_3)) = \exp(-2) = 0.135$

The partition function is $Z = 0.050 + 0.368 + 0.135 = 0.553$.

Now we normalize: $p(x_1) = 0.050 / 0.553 = 0.090$, $p(x_2) = 0.368 / 0.553 = 0.665$, $p(x_3) = 0.135 / 0.553 = 0.244$.

The probabilities sum to 1, and the state with the lowest energy ($x_2$) gets the highest probability. This is exactly what we want.


![Lower energy states get higher probability after normalization.](figures/figure_2.png)
*Lower energy states get higher probability after normalization.*


---

## The Partition Function Problem

This looks great, until it is not.

In our toy example with three discrete states, computing $Z$ was easy — we just summed three numbers. But now imagine $x$ is a 256x256 color image. Each pixel can take 256 values across 3 channels. The number of possible configurations is $256^{256 \times 256 \times 3}$ — a number so large it dwarfs the number of atoms in the observable universe.

Computing $Z$ means summing $\exp(-E(x))$ over every single one of these configurations. This is **completely intractable**.

Think of it this way: computing the partition function is like trying to measure the total volume of every valley, hill, and plateau across an entire mountain range that you have never fully explored. You would have to visit every single point in the landscape — an impossible task.


![The partition function Z is intractable in high dimensions.](figures/figure_3.png)
*The partition function Z is intractable in high dimensions.*


This means that we cannot directly use maximum likelihood to train the energy function. The standard approach would be to maximize $\log p(x) = -E_\theta(x) - \log Z$, but $\log Z$ is impossible to compute.

So, how do we train a model if we cannot compute the normalizing constant? This brings us to one of the most elegant ideas in modern generative modeling: **the score function**.

---

## The Score Function — A Compass Toward High Probability

Instead of trying to model the probability density directly (which requires the intractable partition function), we model its **gradient**. The score function is defined as the gradient of the log probability density:


$$
s(x) = \nabla_x \log p(x)
$$


What does this mean intuitively? The score function is a **vector field** — at every point in space, it gives you a vector that points in the direction where the probability increases the fastest. It is like a compass that always points toward the nearest region of high probability.


![The score field points toward regions of high probability.](figures/figure_4.png)
*The score field points toward regions of high probability.*


Let us build our intuition with a concrete example. Suppose the data follows a 1D standard Gaussian distribution: $p(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)$.

First, we take the logarithm:


$$
\log p(x) = -\frac{1}{2}\log(2\pi) - \frac{x^2}{2}
$$

Now, we take the gradient (derivative with respect to $x$). The constant term vanishes, and we get:


$$
s(x) = \frac{d}{dx}\log p(x) = -x
$$


Let us plug in some simple numbers:

- At $x = 2$: the score is $s(2) = -2$. The negative sign means "move left" — back toward the center where the density is highest.
- At $x = -3$: the score is $s(-3) = 3$. The positive sign means "move right" — again, toward the center.
- At $x = 0$: the score is $s(0) = 0$. At the peak of the distribution, there is no preferred direction — you are already at the most likely point.

This tells us that the score function acts like a restoring force, always pulling you toward high-density regions. The farther you are from the center, the stronger the pull. This is exactly what we want.


![The score points toward the Gaussian peak from every direction.](figures/figure_5.png)
*The score points toward the Gaussian peak from every direction.*


Now here is the key insight that makes the score function so powerful for energy-based models.

Let us write down the score function and substitute the Boltzmann distribution:


$$
s(x) = \nabla_x \log p(x) = \nabla_x \left[ -E_\theta(x) - \log Z \right] = -\nabla_x E_\theta(x)
$$


The $\log Z$ term disappears because $Z$ is a constant that does not depend on $x$, so its gradient is zero!

This means that **the score function completely bypasses the partition function**. We can compute the score simply as the negative gradient of the energy function, without ever needing to compute $Z$.

Let us verify this with a simple example. Suppose $E(x) = x^2$. Then the score is $s(x) = -\nabla_x E(x) = -2x$. At $x = 3$, the score is $-6$ (push strongly to the left, toward lower energy). At $x = -1$, the score is $2$ (push to the right, toward the minimum at $x=0$). This makes perfect sense — the score always pushes us toward the energy minimum, which corresponds to the probability maximum.

---

## Sampling with Langevin Dynamics

Now that we understand the score function, a natural question arises: **if we know the score, how do we actually generate samples from the distribution?**

Let us start with an analogy. Imagine that you are dropped into a thick fog on a vast landscape. Your goal is to find the deepest valley because that is where the treasure is hidden. You cannot see the full landscape — all you have is the ability to feel the slope of the ground beneath your feet.


![Langevin dynamics combines gradient steps with random noise.](figures/figure_6.png)
*Langevin dynamics combines gradient steps with random noise.*


What is the strategy that you will use? You will take a step in the direction where the downward slope is the steepest. Mathematically, this is gradient descent:


$$
x_{t+1} = x_t + \eta \nabla_x \log p(x_t)
$$


Here, $\eta$ is the step size. This update rule says: move in the direction the score points (toward higher probability).

Let us plug in some simple numbers. Suppose $p(x)$ is a standard Gaussian, so $s(x) = -x$. Starting at $x_0 = 4$ with $\eta = 0.1$: $x_1 = 4 + 0.1 \times (-4) = 3.6$. Then $x_2 = 3.6 + 0.1 \times (-3.6) = 3.24$. We are slowly moving toward $x = 0$, the peak. This makes sense.

But there is a problem. Consider a landscape with two valleys — one shallow and one deep:


![Gradient descent can get stuck in local minima.](figures/figure_7.png)
*Gradient descent can get stuck in local minima.*


If the ball starts near the shallow valley, gradient descent will trap it there. The gradient is zero at the local minimum, so the ball stops moving — even though there is a much better valley nearby.

To solve this, we add a **random shake** — a small amount of noise at each step. This gives the ball enough energy to escape shallow traps and eventually find the global minimum. The update rule becomes:


$$
x_{t+1} = x_t + \eta \nabla_x \log p(x_t) + \sqrt{2\eta} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$


This is called **Langevin dynamics**. The first term follows the score (gradient), and the second term adds random noise. Together, they produce a zig-zag path — like a hiker who is drunk and trying to navigate their way through the terrain.

Let us plug in some simple numbers. Start at $x_0 = 3$, with score $s(x) = -x$, step size $\eta = 0.1$, and suppose we sample $\epsilon = 0.5$:

$x_1 = 3 + 0.1 \times (-3) + \sqrt{0.2} \times 0.5 = 3 - 0.3 + 0.224 = 2.924$

We moved toward the center (good!) but with some randomness (the noise). Over many steps, the trajectory will converge to the distribution while exploring broadly enough to avoid getting stuck.


![Langevin trajectories zig-zag toward high-density regions.](figures/figure_8.png)
*Langevin trajectories zig-zag toward high-density regions.*


Here is a simple implementation of Langevin sampling when the score function is known:

```python
import torch
import matplotlib.pyplot as plt

def langevin_sampling(score_fn, x_init, n_steps=1000, step_size=0.01):
    """Sample using Langevin dynamics given a known score function."""
    x = x_init.clone()
    trajectory = [x.clone()]

    for _ in range(n_steps):
        noise = torch.randn_like(x)
        score = score_fn(x)
        x = x + step_size * score + (2 * step_size) ** 0.5 * noise
        trajectory.append(x.clone())

    return x, trajectory

# Example: Sample from a 1D Gaussian (score = -x)
score_fn = lambda x: -x
x_init = torch.tensor([4.0])
sample, traj = langevin_sampling(score_fn, x_init, n_steps=500, step_size=0.01)
print(f"Started at {x_init.item():.2f}, ended at {sample.item():.2f}")
```

Let us understand this code in detail. We start at an initial point `x_init` and iteratively update it using the Langevin dynamics formula. At each step, we compute the score, take a gradient step, and add scaled Gaussian noise. After enough steps, the sample will be drawn (approximately) from the target distribution.

---

## Score Matching — Learning the Score

So far, we have assumed that we know the score function. But in practice, we do not know the true data distribution $p(x)$, so we cannot compute $\nabla_x \log p(x)$ directly.

The idea is to train a neural network $s_\theta(x)$ to approximate the true score. We want to minimize the difference between our predicted score and the true score:


$$
J(\theta) = \frac{1}{2} \mathbb{E}_{p(x)} \left[ \| s_\theta(x) - \nabla_x \log p(x) \|^2 \right]
$$


This is called the **Fisher divergence** — it measures the mean squared error between the predicted and true scores.


![s_theta(x) - true_score(x)||^2]". || Training a neural network to match the true score field.](figures/figure_9.png)
*s_theta(x) - true_score(x)||^2]". || Training a neural network to match the true score field.*


But here is the problem: this loss function requires the **true** score $\nabla_x \log p(x)$, which we do not have! If we knew the true distribution, we would not need to learn it.

This is where a remarkable result from Hyvarinen (2005) comes in. Through a technique called integration by parts, Hyvarinen showed that the Fisher divergence can be rewritten into an equivalent objective that depends only on the model and the data samples:


$$
J(\theta) = \mathbb{E}_{p(x)} \left[ \text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2} \| s_\theta(x) \|^2 \right]
$$


This is the **tractable score matching loss**. It has two terms, each with a clear intuition:

**Term 1: The Jacobian Trace** — $\text{tr}(\nabla_x s_\theta(x))$

This measures the **divergence** of the score field. A negative trace means the arrows are converging inward — like water flowing into a drain. By minimizing this term, we force the score arrows to point inward at data points, creating "sinks" that attract samples.

**Term 2: The Score Magnitude** — $\frac{1}{2} \| s_\theta(x) \|^2$

This penalizes large score vectors. In regions where the probability is high (near the data), the score should be small — you are already where you want to be. This term drives the score toward zero at high-probability locations, making these points stationary.


![The two terms create sinks at data points and make them stationary.](figures/figure_10.png)
*The two terms create sinks at data points and make them stationary.*


Let us plug in some simple numbers. Suppose our score model is $s_\theta(x) = ax + b$ with $a = -1$ and $b = 0$, so $s_\theta(x) = -x$. The Jacobian trace is simply $a = -1$ (the derivative of $-x$ with respect to $x$). At $x = 0$ (a data point), $\|s_\theta(0)\|^2 = 0$. The loss contribution at this point is $-1 + 0 = -1$. The negative trace means the arrows converge inward, and the zero magnitude means the data point is stationary. This is exactly what we want for a standard Gaussian.

Here is a simple implementation of score matching on 2D data:

```python
import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    """Simple MLP to estimate the score function."""
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)

def score_matching_loss(model, x):
    """Compute the tractable score matching loss (Hyvarinen 2005)."""
    x.requires_grad_(True)
    score = model(x)

    # Term 2: score magnitude
    score_norm = 0.5 * (score ** 2).sum(dim=-1)

    # Term 1: trace of the Jacobian
    trace = torch.zeros(x.shape[0], device=x.device)
    for i in range(x.shape[1]):
        grad = torch.autograd.grad(
            score[:, i].sum(), x, create_graph=True
        )[0][:, i]
        trace += grad

    loss = (trace + score_norm).mean()
    return loss
```

---

## Denoising Score Matching — A Simpler Path

The tractable score matching loss from the previous section is elegant, but it has a computational problem. Computing the Jacobian trace requires calculating all $D$ diagonal entries of the Jacobian matrix, where $D$ is the dimensionality of the data. For a 256x256 image, that is $D = 196{,}608$ — making the computation extremely expensive.

Pascal Vincent (2010) introduced a beautifully simple alternative called **Denoising Score Matching**. To understand his idea, let us use an analogy.

Imagine you have a tabletop. There are invisible magnets hidden at specific spots on this table — these represent your real data points. Your goal is to draw a map of the magnetic field that tells you, for any point on the table, which direction the nearest magnet is pulling.

If you just look at the empty table, you cannot calculate the field. But here is the trick: place a metal ball on top of a hidden magnet, then flick the ball in a random direction. It rolls away and stops at a new, random spot — this is your **noisy data point**.

Now, the direction from the noisy position back to the original magnet position — that is exactly the score of the noisy distribution! And crucially, you know both positions, so you can compute this target analytically.


![Denoising score matching: learn arrows from noisy to clean data.](figures/figure_11.png)
*Denoising score matching: learn arrows from noisy to clean data.*


The Denoising Score Matching loss is:


$$
J_{DSM}(\theta) = \mathbb{E}_{q_\sigma(\tilde{x}|x)\, p(x)} \left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) \right\|^2 \right]
$$

When we add Gaussian noise with variance $\sigma^2$ (i.e., $\tilde{x} = x + \sigma\epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$), the target score simplifies beautifully:


$$
\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) = -\frac{\tilde{x} - x}{\sigma^2}
$$

So the loss becomes: the neural network should predict the direction from the noisy point back to the clean point, scaled by $1/\sigma^2$.

Let us plug in some simple numbers. Suppose the clean data point is $x = 2$, the noise is $\epsilon = 0.6$, and $\sigma = 0.5$. Then $\tilde{x} = 2 + 0.5 \times 0.6 = 2.3$. The target score at $\tilde{x} = 2.3$ is:

$-\frac{2.3 - 2}{0.5^2} = -\frac{0.3}{0.25} = -1.2$

So the neural network should predict $-1.2$ at the point $\tilde{x} = 2.3$. This value is negative, meaning "move to the left" — back toward the clean data point at $x = 2$. This makes perfect sense.

Does this remind you of something? In **Denoising Diffusion Probabilistic Models (DDPM)**, we arrived at a very similar conclusion — the model learns to predict the noise that was added to create the noisy image. The connection between denoising score matching and diffusion models is deep and fundamental.

---

## From Score Matching to Modern Diffusion Models

The ideas we have covered — energy functions, score functions, and denoising score matching — are not just abstract concepts. They form the intellectual backbone of the most powerful generative models in use today.

Here is the key insight that bridges everything together: **both score-based models and diffusion models are learning to denoise data, but diffusion models do it across multiple noise scales.**

In a single noise level, denoising score matching works well for simple distributions. But real-world data (like images) has complex, multi-modal structure. Using a single noise level leads to inaccurate score estimates in low-density regions where data is sparse.

The solution, proposed by Song and Ermon (2019), is to use **multiple noise scales** — from very small noise (preserving fine details) to very large noise (covering broad structure). At each scale, the model learns a separate score function. During sampling, you start from pure noise and progressively reduce the noise level, running Langevin dynamics at each scale. This is called **annealed Langevin dynamics**.


![The intellectual path from EBMs to modern diffusion models.](figures/figure_12.png)
*The intellectual path from EBMs to modern diffusion models.*


Ho et al. (2020) showed that DDPM is equivalent to learning the score function at each noise level in a forward diffusion process. Song et al. (2021) unified everything further by showing that both diffusion models and score-based models are instances of the same underlying framework — stochastic differential equations (SDEs) — where the score function plays the central role.

This is truly amazing. A paper released by Hyvarinen in 2005 on score matching serves as the backbone of the generative AI revolution happening today. The score function — a simple gradient of the log density — turned out to be the key that unlocks tractable training of energy-based models and, by extension, all of modern diffusion-based generation.

---

## Practical Implementation — Putting It All Together

Enough theory, let us look at a practical implementation. We will train a score network on a 2D mixture of Gaussians and then sample from it using Langevin dynamics.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Data: mixture of two 2D Gaussians ---
def sample_data(n=2000):
    """Generate samples from a mixture of two Gaussians."""
    mix = torch.rand(n) < 0.5
    x = torch.randn(n, 2) * 0.5
    x[mix, 0] += 2.0   # shift first component right
    x[~mix, 0] -= 2.0  # shift second component left
    return x

# --- Score Network ---
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.net(x)

# --- Denoising Score Matching Training ---
model = ScoreNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sigma = 0.5  # noise level

for epoch in range(2000):
    x = sample_data(512)
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    target_score = -noise / (sigma ** 2)    # DSM target
    pred_score = model(x_noisy)
    loss = ((pred_score - target_score) ** 2).sum(dim=-1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Langevin Sampling ---
x_samples = torch.randn(500, 2) * 3  # start from noise
for step in range(1000):
    eps = torch.randn_like(x_samples)
    score = model(x_samples).detach()
    x_samples = x_samples + 0.01 * score + (0.02) ** 0.5 * eps
```

Let us understand what this code does. First, we create training data from a mixture of two Gaussians centered at $(-2, 0)$ and $(2, 0)$. Then we train a simple neural network using denoising score matching — we corrupt data with Gaussian noise and train the network to predict the direction back to the clean data. Finally, we use Langevin dynamics to generate new samples, starting from random noise and iteratively following the learned score field.


![Training a score network and sampling with Langevin dynamics.](figures/figure_13.png)
*Training a score network and sampling with Langevin dynamics.*


We can see from the results that the generated samples closely match the true data distribution — two distinct clusters centered at $(-2, 0)$ and $(2, 0)$. The learned score field correctly points toward both modes, and the Langevin trajectories show the characteristic zig-zag paths converging to high-density regions. Not bad right?

---

## Summary

Let us recap the journey we have taken:

1. **Energy-Based Models** assign low energy to likely data and high energy to unlikely data, converting energy to probability via the Boltzmann distribution.
2. **The partition function** makes direct training intractable in high dimensions.
3. **The score function** — the gradient of the log density — completely bypasses the partition function, providing a tractable way to characterize the distribution.
4. **Langevin dynamics** uses the score function to generate samples through iterative gradient steps plus noise.
5. **Score matching** (Hyvarinen, 2005) provides a way to learn the score from data without knowing the true distribution.
6. **Denoising score matching** (Vincent, 2010) simplifies this further — just add noise and learn to undo it.
7. **Modern diffusion models** extend these ideas to multiple noise scales, forming the foundation of today's generative AI.

The score function started as an elegant mathematical trick to train energy-based models. It evolved into the central component of the most powerful generative models ever built. This is truly amazing.

That's it!

---

**References:**

- Hyvarinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching.* Journal of Machine Learning Research.
- Vincent, P. (2010). *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation.
- Song, Y. and Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS.
- Ho, J., Jain, A., and Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, Y., et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR.

For further reading, please refer to the book: *The Principles of Diffusion Models From Origins to Advances* (https://arxiv.org/abs/2510.21890) [Pages 56-79]
