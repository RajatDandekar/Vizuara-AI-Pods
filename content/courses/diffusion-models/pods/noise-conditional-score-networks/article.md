# Noise Conditioned Score Networks: Teaching a Neural Network to Denoise at Every Scale

How Song & Ermon solved the low-density problem in score matching by conditioning on multiple noise levels

---

## The Problem with Score Matching in Empty Space

Let us start with a practical problem. In the previous articles on score matching, we learned that the score function acts like a compass — it points you toward regions where the data lives. We also learned how to train a neural network to predict this compass using either the tractable score matching loss or the denoising score matching loss.

But there is a major issue that we have not addressed yet.

Think about it this way: imagine you have a compass that works perfectly when you are near a city, but the moment you step into the desert — far from any civilization — the compass starts spinning wildly. It gives you no useful direction at all.

This is exactly what happens with score matching in practice. The score function is well-defined and useful where the data density is high. But in regions where there is almost no data, the score becomes unreliable or even meaningless.


![Score estimation fails in low-density regions far from the data.](figures/figure_1.png)
*Score estimation fails in low-density regions far from the data.*


Why does this happen? This brings us to something called the **manifold hypothesis**.

## The Manifold Hypothesis and Where Score Estimation Breaks Down

Real-world data — images, audio, text — does not fill up the entire high-dimensional space uniformly. Instead, it lives on a thin, low-dimensional surface (called a **manifold**) embedded within that high-dimensional space.

Consider a simple example. Suppose our data space is 2D, but the actual data points lie along a thin curve. Most of the 2D space is completely empty — no data exists there.


![Data lives on a thin manifold; most of the ambient space is empty.](figures/figure_2.png)
*Data lives on a thin manifold; most of the ambient space is empty.*


Mathematically, the score function is defined as:


$$
s(x) = \nabla_x \log p(x)
$$


When $p(x) \approx 0$, the logarithm goes to negative infinity, and its gradient becomes undefined or extremely noisy.

Let us plug in some simple numbers. Suppose our data follows a 1D Gaussian with mean $\mu = 0$ and standard deviation $\sigma = 1$. The score at any point is $s(x) = -x/\sigma^2 = -x$. At $x = 0$ (the center), $s(0) = 0$ — the score is zero because we are already at the peak. At $x = 2$, $s(2) = -2$ — a moderate pull back toward the center. But what happens at $x = 100$? The score says $s(100) = -100$ — a huge pull. The problem is that $p(100)$ is essentially zero for a standard Gaussian. The score is defined, but there is almost no data there to learn it from. Any training signal at $x = 100$ is pure noise.

So, how do we fix a compass that only works near civilization?

## The Key Insight — Fill the Empty Space with Noise

This brings us to a remarkably elegant idea proposed by Yang Song and Stefano Ermon in 2019.

The idea is simple: if the problem is that most of the space is empty, then **fill it with noise**.

By adding Gaussian noise to the data, we "spread" the data points outward into the previously empty space. The noisy version of the data has support everywhere — its density is positive at every point in the space.

But here is the critical insight: we do not use just one noise level. We use **many noise levels**, from very small to very large.

Think of it like fogging up a window. A very thin layer of fog lets you see the shapes clearly — the data structure is preserved. A very thick layer of fog covers everything, but you know something is behind the glass. Each fog thickness gives you different information about what lies behind.

Mathematically, for a given noise level $\sigma$, the perturbed data distribution is:


$$
q_\sigma(\tilde{x}) = \int p(x) \cdot \mathcal{N}(\tilde{x} \mid x, \sigma^2 I) \, dx
$$


This is the convolution of the data distribution with a Gaussian kernel. For any positive $\sigma$, this distribution has positive density everywhere.

Let us plug in some simple numbers. Suppose our data consists of a single point at $x = 3$ in 1D, and we use $\sigma = 0.5$. Then $q_\sigma(\tilde{x}) = \mathcal{N}(\tilde{x} \mid 3, 0.25)$. At $\tilde{x} = 4$, the density is:

$$q_{0.5}(4) = \frac{1}{\sqrt{2\pi \cdot 0.25}} \exp\left(-\frac{(4-3)^2}{2 \cdot 0.25}\right) = \frac{1}{0.7979} \exp(-2) = 1.2533 \times 0.1353 = 0.1695$$

This is a small but positive number. The score at $\tilde{x} = 4$ is now well-defined: $s(4) = -(4-3)/0.25 = -4$, pointing back toward the data at $x = 3$. This is exactly what we want.


![Multiple noise levels progressively fill the empty space with signal.](figures/figure_3.png)
*Multiple noise levels progressively fill the empty space with signal.*


Now the question is, how do we learn the score at all these noise levels simultaneously?

## Noise Conditioned Score Network — One Model, Many Noise Levels

The answer is beautifully simple. Instead of training a separate score network for each noise level, we train **one single neural network** that takes both the data point $\tilde{x}$ and the noise level $\sigma$ as inputs.

The network outputs the estimated score for that particular noise level:

$$s_\theta(\tilde{x}, \sigma) \approx \nabla_{\tilde{x}} \log q_\sigma(\tilde{x})$$

The training objective combines the denoising score matching losses across all noise levels:


$$
\mathcal{L}(\theta) = \frac{1}{L} \sum_{i=1}^{L} \lambda(\sigma_i) \, \mathbb{E}_{p(x)} \, \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma_i^2 I)} \left[ \left\| s_\theta(\tilde{x}, \sigma_i) + \frac{\tilde{x} - x}{\sigma_i^2} \right\|^2 \right]
$$


Let us break this down. For each noise level $\sigma_i$, we:

1. Sample a clean data point $x$ from the training data
2. Add noise: $\tilde{x} = x + \sigma_i \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$
3. The target score is $-(\tilde{x} - x)/\sigma_i^2 = -\epsilon / \sigma_i$
4. Train the network to match this target

The weighting factor $\lambda(\sigma_i) = \sigma_i^2$ ensures that the losses at different noise levels are comparable in magnitude.

Let us plug in some simple numbers. Suppose we have $L = 3$ noise levels: $\sigma_1 = 2.0$, $\sigma_2 = 1.0$, $\sigma_3 = 0.5$. Consider a single data point $x = [1, 2]$ and a noise sample $\epsilon = [0.3, -0.1]$.

For $\sigma_2 = 1.0$:
- Noisy sample: $\tilde{x} = [1, 2] + 1.0 \cdot [0.3, -0.1] = [1.3, 1.9]$
- Target score: $-\epsilon / \sigma_2 = -[0.3, -0.1] / 1.0 = [-0.3, 0.1]$
- Suppose the network predicts $s_\theta = [-0.25, 0.15]$
- Loss for this sample: $\|[-0.25, 0.15] - [-0.3, 0.1]\|^2 = (0.05)^2 + (0.05)^2 = 0.005$
- Weighted loss: $\sigma_2^2 \cdot 0.005 = 1.0 \cdot 0.005 = 0.005$

The total loss averages this across all noise levels and all data samples.


![The network takes both noisy data and noise level as input.](figures/figure_4.png)
*The network takes both noisy data and noise level as input.*


Here is a simple PyTorch implementation of a noise-conditioned score network:

```python
import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    """Noise Conditioned Score Network for 2D data."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Input: 2D data point + 1D noise level = 3
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # Output: 2D score
        )

    def forward(self, x, sigma):
        # x: (batch, 2), sigma: (batch, 1)
        sigma_input = sigma.view(-1, 1)
        net_input = torch.cat([x, sigma_input], dim=1)
        return self.net(net_input)
```

This is the simplest possible architecture. In the original paper, Song and Ermon used a deeper network with residual connections. For images, they used a U-Net architecture. But the core idea is the same: concatenate the noise level with the input and let the network learn to produce the right score for each noise level.

## Choosing the Noise Levels — The Geometric Sequence

Now let us look at how to choose the noise levels $\sigma_1 > \sigma_2 > \dots > \sigma_L$.

Song and Ermon proposed a specific strategy:

- **$\sigma_1$ (largest):** Should be large enough to cover the entire data distribution. A good rule of thumb is to set it to the maximum pairwise distance between data points.
- **$\sigma_L$ (smallest):** Should be small enough to preserve the fine structure of the data, without destroying details.
- **The sequence:** Use a geometric progression between $\sigma_1$ and $\sigma_L$.


$$
\sigma_i = \sigma_1 \cdot \left(\frac{\sigma_L}{\sigma_1}\right)^{\frac{i-1}{L-1}}, \quad i = 1, 2, \dots, L
$$

Let us plug in some simple numbers. Suppose $\sigma_1 = 10$, $\sigma_L = 0.01$, and $L = 10$. What is $\sigma_5$?

$$\sigma_5 = 10 \cdot \left(\frac{0.01}{10}\right)^{\frac{4}{9}} = 10 \cdot (0.001)^{0.4444} = 10 \cdot 0.01778 = 0.1778$$

So the noise levels are roughly: 10, 4.64, 2.15, 1.0, 0.464, 0.215, 0.1, 0.0464, 0.0215, 0.01. Notice how they span three orders of magnitude. The geometric spacing ensures that each step reduces the noise by a consistent ratio, which is important for the sampling procedure we will discuss next.


![The geometric noise schedule spans from coarse to fine resolution.](figures/figure_5.png)
*The geometric noise schedule spans from coarse to fine resolution.*


## Annealed Langevin Dynamics — Sampling from Coarse to Fine

We have a trained noise-conditioned score network. But how do we actually generate new samples?

We already know about Langevin dynamics from the score matching article — the "drunk hiker" who takes steps proportional to the score plus some random noise. But standard Langevin dynamics uses a single noise level, and we have learned that this fails in low-density regions.

The solution is **Annealed Langevin Dynamics**. The idea is to run Langevin dynamics in stages, starting with the largest noise level and progressively decreasing to the smallest.

Think of it like sculpting a statue. First, you rough out the general shape with a large chisel — this is like sampling with large $\sigma$, where the large-scale structure of the data emerges. Then you switch to a medium chisel for the intermediate details. Finally, you use fine sandpaper for the finishing touches — this is like sampling with the smallest $\sigma$, where the sharp details appear.

At each noise level $\sigma_i$, we run $T$ steps of Langevin dynamics:


$$
x_{t+1} = x_t + \alpha_i \, s_\theta(x_t, \sigma_i) + \sqrt{2\alpha_i} \, z_t, \quad z_t \sim \mathcal{N}(0, I)
$$


The step size $\alpha_i$ depends on the noise level:


$$
\alpha_i = \epsilon \cdot \frac{\sigma_i^2}{\sigma_L^2}
$$

This makes the step size proportional to $\sigma_i^2$. At large noise levels, we take large steps (exploring broadly). At small noise levels, we take tiny steps (refining carefully).

Let us plug in some simple numbers. Suppose $\epsilon = 0.00005$, $\sigma_i = 1.0$, and $\sigma_L = 0.01$. Then:

$$\alpha_i = 0.00005 \times \frac{1.0^2}{0.01^2} = 0.00005 \times 10000 = 0.5$$

Now suppose our current sample is $x_t = [0.5, 0.5]$, the predicted score is $s_\theta = [1.2, -0.8]$, and $z_t = [0.3, -0.2]$. Then:

$$x_{t+1} = [0.5, 0.5] + 0.5 \cdot [1.2, -0.8] + \sqrt{2 \times 0.5} \cdot [0.3, -0.2]$$
$$= [0.5, 0.5] + [0.6, -0.4] + 1.0 \cdot [0.3, -0.2]$$
$$= [0.5 + 0.6 + 0.3, \; 0.5 - 0.4 - 0.2] = [1.4, -0.1]$$

The sample moved from $[0.5, 0.5]$ to $[1.4, -0.1]$ — pulled by the score and jostled by the noise. Over many steps and decreasing noise levels, the sample converges to a point drawn from the data distribution.


![Annealed Langevin dynamics refines samples from coarse to fine.](figures/figure_6.png)
*Annealed Langevin dynamics refines samples from coarse to fine.*


Here is the implementation of annealed Langevin dynamics:

```python
def annealed_langevin_dynamics(score_net, sigmas, n_steps=100,
                                eps=0.00005, dim=2):
    """Generate samples using Annealed Langevin Dynamics."""
    # Start from random noise
    x = torch.randn(1, dim)

    for sigma in sigmas:
        # Step size for this noise level
        alpha = eps * (sigma / sigmas[-1]) ** 2

        for t in range(n_steps):
            # Predict score at current position and noise level
            sigma_tensor = torch.full((1,), sigma)
            score = score_net(x, sigma_tensor)

            # Langevin update
            noise = torch.randn_like(x)
            x = x + alpha * score + (2 * alpha) ** 0.5 * noise

    return x
```

## Putting It All Together — NCSN on 2D Data

Enough theory. Let us see how NCSN works in practice.

We will train a noise-conditioned score network on a simple 2D mixture of Gaussians and then sample from it using annealed Langevin dynamics.

Here is the training loop:

```python
import torch
import torch.nn as nn
import numpy as np

# Generate training data: mixture of 2 Gaussians
def generate_data(n=5000):
    mix = torch.rand(n, 1)
    centers = torch.tensor([[-3.0, 0.0], [3.0, 0.0]])
    idx = (mix > 0.5).long().squeeze()
    data = centers[idx] + 0.5 * torch.randn(n, 2)
    return data

# Noise levels (geometric sequence)
L = 10
sigma_1, sigma_L = 10.0, 0.01
sigmas = torch.tensor([
    sigma_1 * (sigma_L / sigma_1) ** (i / (L - 1))
    for i in range(L)
])

# Training
model = ScoreNet(hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
data = generate_data(5000)

for epoch in range(2000):
    # Sample a random noise level
    idx = torch.randint(0, L, (data.shape[0],))
    sigma = sigmas[idx].unsqueeze(1)

    # Add noise
    noise = torch.randn_like(data)
    noisy_data = data + sigma * noise

    # Target: -noise / sigma
    target = -noise / sigma

    # Predict score
    pred = model(noisy_data, sigma.squeeze(1))

    # Weighted loss: sigma^2 * ||pred - target||^2
    loss = (sigma.squeeze(1) ** 2 * (pred - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

After training, we can visualize the learned score fields at different noise levels:


![The score field transitions from broad global structure to precise local detail.](figures/figure_7.png)
*The score field transitions from broad global structure to precise local detail.*


And here are the generated samples using annealed Langevin dynamics:

```python
# Generate 500 samples
samples = []
for _ in range(500):
    s = annealed_langevin_dynamics(model, sigmas.numpy(),
                                   n_steps=100, eps=5e-5)
    samples.append(s.detach().numpy())

samples = np.concatenate(samples, axis=0)
```


![Generated samples closely match the true data distribution.](figures/figure_8.png)
*Generated samples closely match the true data distribution.*


Not bad, right? The generated samples closely match the true data distribution. This is exactly what we want.

## From NCSN to Modern Diffusion Models

Noise Conditioned Score Networks were introduced by Yang Song and Stefano Ermon in their 2019 paper "Generative Modeling by Estimating Gradients of the Data Distribution" at NeurIPS 2019. This paper was a breakthrough because it showed that score-based generative models could produce high-quality samples on par with GANs — without adversarial training.

What is truly amazing is how NCSN connects to diffusion models. Remember how in DDPM, we add noise progressively and then learn to reverse the process? The NCSN approach does something very similar:

- DDPM trains a network to predict the noise $\epsilon$ added at each step
- NCSN trains a network to predict the score $\nabla \log q_\sigma(x)$ at each noise level
- These are essentially the same thing. The score of the noisy distribution is proportional to the noise direction: $\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}) = -(\tilde{x} - x)/\sigma^2 = -\epsilon / \sigma$

In 2021, Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole published "Score-Based Generative Modeling through Stochastic Differential Equations," which unified the discrete noise-level perspective (NCSN) and the discrete time-step perspective (DDPM) into a single continuous-time framework using stochastic differential equations.


![NCSN and DDPM were unified through stochastic differential equations.](figures/figure_9.png)
*NCSN and DDPM were unified through stochastic differential equations.*


The score function — this simple compass pointing toward the data — has become one of the most important concepts in modern generative modeling. And it all started with the insight that adding noise at multiple scales can make score estimation reliable everywhere.

That's it!

---

## References

- Yang Song and Stefano Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution," NeurIPS 2019. [https://arxiv.org/abs/1907.05600](https://arxiv.org/abs/1907.05600)
- Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole, "Score-Based Generative Modeling through Stochastic Differential Equations," ICLR 2021. [https://arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)
- For further reading, please refer to the book: The Principles of Diffusion Models From Origins to Advances. [https://arxiv.org/abs/2510.21890](https://arxiv.org/abs/2510.21890)
