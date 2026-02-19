# What exactly is Denoising Score Matching?

How a simple trick of adding noise solves the hardest problem in score-based generative models.

---

## The Invisible Magnets

Let us start with an analogy. Imagine you have a tabletop. There are invisible magnets hidden at specific spots on this table. These magnets represent your real data.

Your goal is to draw a map of the magnetic field — a map that tells you, for any point on the table, which direction the nearest magnet is pulling.


![Magnetic field arrows point toward hidden data locations.](figures/figure_1.png)
*Magnetic field arrows point toward hidden data locations.*


If you just look at the empty table, you cannot calculate the magnetic field. You do not know where the magnets are or how strong they are. There might be more magnets than you expect, and the field at any arbitrary point is unknown.

This is exactly the challenge in score matching — we want to learn a function that tells us the direction toward high-probability data regions, but computing it directly is too expensive. So, what is the practical solution? This brings us to Denoising Score Matching.

But before we dive into DSM, let us quickly recap why we care about the score function in the first place.

## A Quick Recap: The Score Function

The **score function** is the gradient of the log probability density with respect to the data:


$$
s(x) = \nabla_x \log p(x)
$$


Let us plug in some simple numbers to see how this works. Suppose our probability density is a simple Gaussian: $p(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$. The log density is $\log p(x) = -\frac{1}{2}x^2 + \text{const}$. Taking the gradient, we get $s(x) = -x$. So at $x = 3$, the score is $s(3) = -3$, pointing back toward the origin where the density is highest. This is exactly what we want.

Intuitively, the score acts as a **compass** — at any point in space, it tells you the direction where the probability density is increasing the fastest.


![Score vectors point toward the peak of the distribution.](figures/figure_2.png)
*Score vectors point toward the peak of the distribution.*


Now, why do we care about scores instead of the density itself? The key reason is the **partition function problem**. If we model the density using an energy function $E_\theta(x)$, the probability is:


$$
p_\theta(x) = \frac{1}{Z(\theta)} \exp(-E_\theta(x))
$$


The normalization constant $Z(\theta) = \int \exp(-E_\theta(x)) \, dx$ is intractable for high-dimensional data. But here is the beautiful part — if we take the gradient of the log:


$$
\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)
$$


The partition function vanishes! This is because $Z(\theta)$ does not depend on $x$, so its gradient with respect to $x$ is zero.

Let us verify this with a simple numerical example. Suppose $E_\theta(x) = (x-2)^2$ and $Z = 10$ (some constant). Then $\log p_\theta(x) = -(x-2)^2 - \log(10)$. The gradient is $\nabla_x \log p_\theta(x) = -2(x-2)$. Notice that $\log(10)$ disappeared — the partition function does not affect the score.

So, the strategy is clear: instead of learning the density $p(x)$ (which requires computing $Z$), we learn the score $s_\theta(x)$ directly with a neural network and train it using **score matching**.

## The Tractable Score Matching Objective

Hyvarinen (2005) showed that we can train a score model by minimizing:


$$
\mathcal{L}_{\text{SM}} = \mathbb{E}_{p(x)} \left[ \text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2} \| s_\theta(x) \|^2 \right]
$$

Let us unpack this. The first term is the **trace of the Jacobian** of the score model — it measures how the score vectors are diverging or converging at each point. The second term is the **squared magnitude** of the score vectors.

Let us do a quick numerical example to build intuition. Suppose we have a 1D case where $s_\theta(x) = -2x$. The Jacobian (just a derivative in 1D) is $\frac{d}{dx}(-2x) = -2$. The trace is just $-2$. The squared magnitude at $x = 1$ is $(-2)^2 = 4$. So the loss at $x = 1$ would be $-2 + \frac{1}{2}(4) = 0$. If the true score is also $-2x$ (standard Gaussian with $\sigma^2 = 0.5$), this makes sense — the loss is minimized at zero.

This objective is wonderful because it does not require knowing the true score. However, there is a serious computational problem.

## The Computational Bottleneck

The trace of the Jacobian $\text{tr}(\nabla_x s_\theta(x))$ requires computing the full Jacobian matrix. If $x$ is a $D$-dimensional vector, the Jacobian is a $D \times D$ matrix, and computing it costs $O(D^2)$.


$$
\nabla_x s_\theta(x) \in \mathbb{R}^{D \times D}
$$

Let us put this in perspective with concrete numbers. For a small 28x28 grayscale image (like MNIST), $D = 784$. The Jacobian is a $784 \times 784$ matrix with 614,656 entries. For a 256x256 RGB image, $D = 196,608$, and the Jacobian has over **38 billion** entries.


![Jacobian computation scales quadratically with data dimension.](figures/figure_3.png)
*Jacobian computation scales quadratically with data dimension.*


This makes the tractable score matching loss computationally infeasible for anything beyond toy problems.

So, what is the practical solution? This brings us to Pascal Vincent's breakthrough in 2010.

## Pascal Vincent's Key Insight

What Pascal Vincent proposed was very elegant. Let us understand it through our magnet analogy.

Recall our tabletop with invisible magnets. We could not map the magnetic field because we could not see the magnets.

Now, we do a small trick:

**Step 1.** We place a metal ball exactly on top of a hidden magnet.

**Step 2.** We flick the ball in a random direction. It rolls away and stops at a new, random spot. This is the **noisy data**.

**Step 3.** We bring in a student (our neural network). We show the student the ball's new location. We hide the original magnet location. We ask the student: "Draw an arrow representing the force needed to pull this ball back to where it started."


![Adding noise creates a known target for the score network.](figures/figure_4.png)
*Adding noise creates a known target for the score network.*


The student has no idea where the noisy data came from. But **we** know — because we placed the ball on the magnet and we flicked it.

So if we can give the student feedback based on the difference between the student's prediction and the true direction, we can teach the student how to draw the correct arrow for every possible noisy data point.

Through this process, the student learns to approximate the entire magnetic field — the score function.

The key insight is this: **instead of trying to match the score of the original (clean) data distribution, we match the score of a noisy version of the data distribution.** Since we control the noise, we know the target score in closed form.

## The Mathematics of Denoising Score Matching

Let us now formalize this idea. We start with our clean data $x$ drawn from the true distribution $p(x)$.

We corrupt each data point by adding Gaussian noise to create a noisy version $\tilde{x}$:


$$
q_\sigma(\tilde{x} | x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)
$$


This simply says: $\tilde{x}$ is drawn from a Gaussian centered at $x$ with standard deviation $\sigma$.

Let us plug in a concrete example. Suppose $x = 5$ (a 1D data point) and $\sigma = 0.5$. Then $\tilde{x} \sim \mathcal{N}(5, 0.25)$. A sample might be $\tilde{x} = 5.3$ — the data point shifted slightly by noise.

The marginal noisy distribution is:


$$
q_\sigma(\tilde{x}) = \int q_\sigma(\tilde{x} | x) \, p(x) \, dx
$$


This is the distribution of all noisy samples across all possible clean data points.

The **Denoising Score Matching** objective says: train a score model $s_\theta(\tilde{x})$ to match the score of the conditional noisy distribution $q_\sigma(\tilde{x} | x)$:


$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{p(x)} \mathbb{E}_{q_\sigma(\tilde{x}|x)} \left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x} | x) \right\|^2 \right]
$$

Now, here is the beautiful part. Since $q_\sigma(\tilde{x}|x)$ is a Gaussian, we can compute its score in closed form.

Starting from the Gaussian:


$$
q_\sigma(\tilde{x}|x) = \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{\|\tilde{x} - x\|^2}{2\sigma^2}\right)
$$

Taking the log:


$$
\log q_\sigma(\tilde{x}|x) = -\frac{\|\tilde{x} - x\|^2}{2\sigma^2} + \text{const}
$$

Taking the gradient with respect to $\tilde{x}$:


$$
\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) = -\frac{\tilde{x} - x}{\sigma^2} = \frac{x - \tilde{x}}{\sigma^2}
$$

Let us verify this with our earlier numbers. We had $x = 5$, $\tilde{x} = 5.3$, $\sigma = 0.5$. The target score is:

$$\frac{x - \tilde{x}}{\sigma^2} = \frac{5 - 5.3}{0.25} = \frac{-0.3}{0.25} = -1.2$$

This says: "the score at $\tilde{x} = 5.3$ points in the negative direction (back toward $x = 5$) with magnitude 1.2." This makes sense — the score is telling us to move back toward the clean data point.

Substituting back into the DSM loss, we get the final, beautiful objective:


$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{p(x)} \mathbb{E}_{q_\sigma(\tilde{x}|x)} \left[ \left\| s_\theta(\tilde{x}) - \frac{x - \tilde{x}}{\sigma^2} \right\|^2 \right]
$$


![The DSM target is the direction from noisy to clean data.](figures/figure_5.png)
*The DSM target is the direction from noisy to clean data.*


This is truly elegant. We have replaced the intractable Jacobian trace computation with a simple regression target: **predict the direction from the noisy data back to the clean data, scaled by the noise level.**

Let us do one more numerical example to make this concrete. Suppose we have a 2D data point $x = [3, 4]$ and $\sigma = 1.0$. We sample noise $\epsilon = [0.5, -0.3]$, so $\tilde{x} = [3.5, 3.7]$. The target score is:

$$\frac{x - \tilde{x}}{\sigma^2} = \frac{[3, 4] - [3.5, 3.7]}{1.0} = [-0.5, 0.3]$$

If our neural network predicts $s_\theta(\tilde{x}) = [-0.4, 0.2]$, the DSM loss for this sample is:

$$\|[-0.4, 0.2] - [-0.5, 0.3]\|^2 = \|[0.1, -0.1]\|^2 = 0.01 + 0.01 = 0.02$$

The network is close to the target, so the loss is small. This is exactly what we want.

## Connection to Diffusion Models (DDPM)

Does this remind you of something? In Denoising Diffusion Probabilistic Models (DDPM), we came to a very similar conclusion.

In DDPM, the training objective simplifies to predicting the noise that was added:


$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x, \epsilon, t} \left[ \| \epsilon_\theta(\tilde{x}_t, t) - \epsilon \|^2 \right]
$$

Now, let us see how this connects to DSM. The noise added was $\epsilon = \frac{\tilde{x} - x}{\sigma}$, which means:

$$\frac{x - \tilde{x}}{\sigma^2} = \frac{-\sigma \epsilon}{\sigma^2} = -\frac{\epsilon}{\sigma}$$

So the DSM target is just the **negative noise scaled by the noise level**. Predicting the score is equivalent to predicting the noise!


![Score prediction and noise prediction are two sides of the same coin.](figures/figure_6.png)
*Score prediction and noise prediction are two sides of the same coin.*


Let us verify with numbers. If $\epsilon = [0.5, -0.3]$ and $\sigma = 1.0$, then the DSM target is $-\epsilon/\sigma = [-0.5, 0.3]$, which matches what we calculated before. Not bad, right?

This connection is deep and important — it tells us that the entire modern diffusion model framework is built on top of the denoising score matching principle.

## Practical Implementation

Enough theory, let us look at some practical implementation now.

We will train a simple neural network to learn the score function of a 2D data distribution using the DSM loss.

Let us start with a mixture of Gaussians — a distribution with two clusters of data points:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate 2D data from a mixture of Gaussians
def generate_data(n_samples=2000):
    # Two clusters
    cluster1 = torch.randn(n_samples // 2, 2) * 0.5 + torch.tensor([2.0, 2.0])
    cluster2 = torch.randn(n_samples // 2, 2) * 0.5 + torch.tensor([-2.0, -2.0])
    return torch.cat([cluster1, cluster2], dim=0)

# Simple MLP score network
class ScoreNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)  # Output: 2D score vector
        )
    def forward(self, x):
        return self.net(x)

# DSM Training
data = generate_data()
model = ScoreNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sigma = 0.5  # Noise level

for epoch in range(2000):
    noise = torch.randn_like(data) * sigma
    noisy_data = data + noise
    target_score = (data - noisy_data) / (sigma ** 2)  # DSM target
    predicted_score = model(noisy_data)
    loss = ((predicted_score - target_score) ** 2).sum(dim=-1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Let us understand this code in detail. We generate data from two Gaussian clusters centered at $(2, 2)$ and $(-2, -2)$. The score network is a simple MLP that takes a 2D point and outputs a 2D score vector. During training, we add Gaussian noise with $\sigma = 0.5$, compute the DSM target as $(x - \tilde{x})/\sigma^2$, and minimize the mean squared error between the predicted and target scores.

Now, let us visualize the learned score field:

```python
# Visualize the learned score field
x_range = torch.linspace(-5, 5, 20)
y_range = torch.linspace(-5, 5, 20)
xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

with torch.no_grad():
    scores = model(grid).numpy()

plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], alpha=0.1, s=5, c='blue')
plt.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1],
           color='red', alpha=0.7, scale=50)
plt.title("Learned Score Field (DSM)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```


![Learned score arrows converge toward both data clusters.](figures/figure_7.png)
*Learned score arrows converge toward both data clusters.*


From the above figure, we can see that the score field behaves exactly like a compass — all the arrows point toward the regions where the data density is highest. The arrows between the two clusters point toward the nearest cluster, which makes sense.

## Sampling with Langevin Dynamics

Now that we have a trained score function, let us use it to generate new data points. We will use **Langevin Dynamics**, which is a sampling algorithm that uses the score function to navigate toward high-probability regions.

The update rule is:


$$
x_{t+1} = x_t + \eta \cdot s_\theta(x_t) + \sqrt{2\eta} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$


Here, $\eta$ is the step size, $s_\theta(x_t)$ is the learned score, and $\epsilon_t$ is random noise that prevents us from getting stuck in local minima.

Let us plug in some numbers. Suppose we are at $x_t = [0, 0]$ (halfway between the two clusters), $\eta = 0.01$, and $s_\theta([0, 0]) = [0.5, 0.5]$ (pointing slightly toward the cluster at $(2,2)$), and we sample $\epsilon_t = [0.3, -0.1]$:

$$x_{t+1} = [0, 0] + 0.01 \cdot [0.5, 0.5] + \sqrt{0.02} \cdot [0.3, -0.1]$$
$$= [0, 0] + [0.005, 0.005] + [0.042, -0.014]$$
$$= [0.047, -0.009]$$

The point moves slightly in the direction of the score, with a small random perturbation. Over many steps, this process will carry us toward one of the data clusters.

```python
# Langevin Dynamics Sampling
def langevin_sample(model, n_steps=1000, step_size=0.01, n_samples=500):
    x = torch.randn(n_samples, 2) * 3  # Start from random noise
    trajectory = [x.clone()]
    for t in range(n_steps):
        with torch.no_grad():
            score = model(x)
        noise = torch.randn_like(x)
        x = x + step_size * score + np.sqrt(2 * step_size) * noise
        if (t + 1) % 100 == 0:
            trajectory.append(x.clone())
    return x, trajectory

samples, traj = langevin_sample(model)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.2, s=5, label='True Data')
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5, c='red', label='Generated')
plt.legend()
plt.title("Generated vs True Data")
plt.axis('equal')
plt.subplot(1, 2, 2)
# Show a single trajectory
single = torch.stack([t[0] for t in traj]).numpy()
plt.plot(single[:, 0], single[:, 1], 'g.-', alpha=0.7, markersize=3)
plt.scatter(data[:, 0], data[:, 1], alpha=0.1, s=5, c='blue')
plt.title("Single Langevin Trajectory")
plt.axis('equal')
plt.tight_layout()
plt.show()
```


![Langevin dynamics generates samples matching the true distribution.](figures/figure_8.png)
*Langevin dynamics generates samples matching the true distribution.*


Our "drunk hiker" navigates from a random starting point and ends up close to the true data clusters. The generated samples match the true data distribution well — the two peaks are located at the same positions. This is exactly what we want.


![Generated samples closely match the true data distribution.](figures/figure_9.png)
*Generated samples closely match the true data distribution.*


## Summary

Let us summarize what we have learned:

1. **Score matching** lets us train models without computing the partition function, but the tractable loss requires an expensive Jacobian trace computation.

2. **Denoising Score Matching (DSM)** bypasses this by adding noise to the data. The target score becomes a simple direction: from the noisy point back to the clean point, scaled by the noise level.

3. The DSM loss is equivalent to the **noise prediction** objective used in DDPM — predicting the score is the same as predicting the added noise (up to a scaling factor).

4. Once trained, we can generate new samples using **Langevin Dynamics** — a random walk guided by the learned score function.

That's it!

Here is the link to the original paper by Pascal Vincent which introduced Denoising Score Matching: [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)

For further reading, please refer to the book: [The Principles of Diffusion Models: From Origins to Advances](https://arxiv.org/abs/2510.21890) [Pages 68-79]
