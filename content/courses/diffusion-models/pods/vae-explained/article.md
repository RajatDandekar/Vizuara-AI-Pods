# Variational Autoencoders From Scratch

*Building VAEs from first principles — intuition, math, and a full PyTorch implementation*

---

Let us start with a simple thought experiment.

Imagine that you walk into an art class. The teacher asks 20 students to draw a cat. Every student picks up their pencil and gets to work.

When you collect the drawings, something interesting happens: every cat looks different. Some cats are fat and round. Some are skinny and long. Some have pointy ears, others have floppy ears. Some look angry, some look cute.

Now, here is the question: **what are the hidden factors that cause these differences?**

If you think about it, each drawing is determined by a set of invisible characteristics — stroke thickness, ear shape, eye size, body proportions, tail curliness. These factors are not directly visible in the final drawing, but they *determine* everything about it.

Every drawing has a **secret recipe** of hidden factors that produced it.

Now, someone comes to you and says:

*"Build me a machine that can learn these hidden factors from the drawings, and then use them to generate brand new cat drawings that look realistic."*

How will you solve this problem?

This is exactly what a **Variational Autoencoder (VAE)** does. It learns the hidden factors behind your data and uses them to generate new, realistic samples.


![A VAE learns hidden factors from data and generates new, realistic samples.](figures/figure_1.png)
*A VAE learns hidden factors from data and generates new, realistic samples.*


---

## Latent Variables — The Hidden Recipe

Let us give a proper name to these hidden factors. In machine learning, we call them **latent variables**, and we denote them by the symbol **z**.

The word "latent" simply means hidden. These are the variables that you cannot directly observe, but they control everything about the data you *can* observe.

Let us go back to our cat drawing example. Suppose we simplify things and say there are only two latent variables:

1. **Body roundness** — from very skinny to very fat
2. **Ear pointiness** — from round ears to sharp pointy ears

Now, we can represent every cat drawing as a point on a 2D plane, where the x-axis is body roundness and the y-axis is ear pointiness.


![Every cat drawing maps to a point in the 2D latent space.](figures/figure_2.png)
*Every cat drawing maps to a point in the 2D latent space.*


From this figure, you can see that every single point on this plane corresponds to a specific style of cat. If we pick a point in the top-right corner, we get a fat cat with pointy ears. If we pick a point in the bottom-left, we get a skinny cat with round ears.

This 2D plane is called the **latent space**.

Here is the key insight: if we can learn the mapping from this latent space to actual images, we can generate new images by simply picking new points in the latent space.

But how do we build this mapping? This brings us to the encoder-decoder framework.

---

## The Encoder-Decoder Framework

We need two machines to make this work.

### The Decoder: From Recipe to Drawing

The first machine takes a point from the latent space and generates an image. We call this the **decoder**.

You can think of it as a "drawing machine" — you feed in a recipe (the latent variables), and it produces a drawing.

For example, if you feed in z = [fat, pointy ears], the decoder should output an image of a fat cat with pointy ears.

### The Encoder: From Drawing to Recipe

But wait — there is a problem. We said that we want to pick points in the latent space to generate images. But how do we know which regions of the latent space correspond to which types of images?

We need a second machine that does the reverse: it takes an image and tells us the latent variables that produced it. We call this the **encoder**.

The encoder looks at a drawing and says: "This cat has body roundness of 2.1 and ear pointiness of -0.5."

### Putting Them Together

When we connect the encoder and decoder, we get the full pipeline:

1. Take a real image
2. Pass it through the encoder to get the latent variables
3. Pass the latent variables through the decoder to reconstruct the image


![The encoder compresses images into latent variables; the decoder reconstructs them.](figures/figure_3.png)
*The encoder compresses images into latent variables; the decoder reconstructs them.*


This is the basic autoencoder architecture. But we are not done yet.

The question is: how do we make this *probabilistic*? This is where the "Variational" in Variational Autoencoder comes in.

---

## Making It Probabilistic

### Why Not Just Use a Regular Autoencoder?

A regular autoencoder maps each image to a **single point** in the latent space. This seems fine at first, but it creates a serious problem.

Imagine that we train a regular autoencoder on images of digits 0 through 9. Each image gets mapped to a specific dot in the latent space. Now, what happens if we pick a point in the latent space that is *between* two dots? A point that the encoder has never produced?

The decoder has no idea what to do with this point. It was never trained on it. The result? Garbage.

The latent space is full of "dead zones" — regions that don't correspond to any meaningful image. This makes it useless for generation.


![Regular autoencoders have dead zones in latent space; VAEs ensure smooth, complete coverage.](figures/figure_4.png)
*Regular autoencoders have dead zones in latent space; VAEs ensure smooth, complete coverage.*


### The VAE Solution: Distributions, Not Points

The VAE solves this elegantly. Instead of mapping each image to a single point, the encoder outputs a **probability distribution** over the latent space.

Specifically, the encoder outputs two things for each image:

1. A **mean** μ — the center of the distribution
2. A **standard deviation** σ — how spread out the distribution is

The latent code z is then **sampled** from this distribution:

$$q_\phi(z|x) = \mathcal{N}(z;\, \mu_\phi(x),\, \sigma_\phi^2(x) \cdot I)$$

This reads: "The encoder's estimate of z, given input x, is a Gaussian distribution with mean μ and variance σ²."

Let us plug in some simple numbers to see how this works. Suppose our encoder processes an image of a fat, round-eared cat. It outputs:

- μ = [2.1, -0.5]
- σ = [0.3, 0.2]

This means: "This cat is *around* body_roundness = 2.1 and ear_pointiness = -0.5, but with some uncertainty."

We then sample z from this distribution. One sample might give us z = [2.25, -0.38]. Another sample might give z = [1.94, -0.61]. Both are valid representations of this cat, and both will produce slightly different but plausible reconstructions.

Because the distributions overlap between different images, the latent space has no dead zones. Every point in the latent space is covered by *someone's* distribution. This is exactly what we want.

### The Decoder Distribution

Similarly, the decoder does not output a single fixed image. It outputs a **distribution** over pixel values. For binary images like MNIST handwritten digits, this is a Bernoulli distribution for each pixel:

$$p_\theta(x|z) = \prod_{i=1}^{D} \text{Bernoulli}(x_i;\, p_i(z))$$

Here, D is the total number of pixels, and $p_i(z)$ is the probability that pixel $i$ is "on" (white), given the latent code z. The decoder neural network computes these probabilities.

---

## The Training Objective — Evidence Lower Bound (ELBO)

Now that we have the probabilistic framework set up, the critical question is: **what objective function do we optimize during training?**

### What Do We Want to Maximize?

Let us think from first principles. We want our model to be good at generating realistic data. Mathematically, this means we want to maximize the **probability of the real data under our model**:

$$p_\theta(x) = \int p_\theta(x|z) \, p(z) \, dz$$

This integral says: "To compute the probability of image x, consider every possible latent code z, generate an image from each z, and sum up the probabilities."

But here is the problem. This integral requires us to try *every possible* z in the entire latent space. For a latent space of even modest dimension, this is completely **intractable**. We cannot compute this integral.

So we need a clever workaround.

### Deriving the ELBO — Step by Step

We will now derive a quantity called the **Evidence Lower Bound (ELBO)**, which is always less than or equal to the true log-likelihood. If we maximize the ELBO, we are pushing up the true objective as well.

Let us do the derivation in steps.

**Step 1: Start with the log-likelihood**

We want to maximize $\log p_\theta(x)$. Let us introduce our encoder $q_\phi(z|x)$ into the picture by multiplying and dividing:

$$\log p_\theta(x) = \log \int p_\theta(x, z) \, dz = \log \int \frac{p_\theta(x, z)}{q_\phi(z|x)} \, q_\phi(z|x) \, dz$$

This is just a mathematical identity — we have not changed anything.

**Step 2: Apply Jensen's Inequality**

Jensen's inequality tells us that the log of an expectation is greater than or equal to the expectation of the log. In simple words: if you take the average first and then the logarithm, the result is at least as large as taking the logarithm first and then the average.

Applying this:

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

**Step 3: Expand the joint probability**

We can write $p_\theta(x, z) = p_\theta(x|z) \cdot p(z)$. Substituting this in:

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x)\right]$$

**Step 4: Rearrange into two terms**

Grouping the terms:

$$\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \,\|\, p(z))}_{\text{Regularization}}$$

This is the **Evidence Lower Bound (ELBO)**. This is the objective function that we maximize during training.

### Interpreting the Two Terms

Let us understand what each term is asking us to do.

**Term 1: Reconstruction Loss**

$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$$

This says: "Sample a latent code z from the encoder, pass it through the decoder, and check how well the decoded image matches the original." In simple terms: **the reconstructed image should look like the original image.**

If the reconstruction is perfect, this term is maximized.

**Term 2: KL Divergence (Regularization)**

$$D_{KL}(q_\phi(z|x) \,\|\, p(z))$$

This says: "The encoder's output distribution should stay close to a simple prior distribution p(z), which we choose to be a standard Gaussian $\mathcal{N}(0, I)$." In simple terms: **the latent space should be well-organized.**

Without this term, the encoder could cheat — it could map each image to a wildly different, isolated region of the latent space, making generation impossible.


![Training a VAE is a balancing act between faithful reconstruction and a well-organized latent space.](figures/figure_5.png)
*Training a VAE is a balancing act between faithful reconstruction and a well-organized latent space.*


Let us plug in some concrete numbers to build intuition for the ELBO.

Suppose we have a tiny 4-pixel image with true pixel values x = [1, 0, 1, 1]. Our decoder predicts probabilities p = [0.9, 0.1, 0.8, 0.95].

**Reconstruction loss** (negative binary cross-entropy):

$$\mathcal{L}_{\text{recon}} = -[1 \cdot \log(0.9) + 0 \cdot \log(0.9) + 1 \cdot \log(0.8) + 1 \cdot \log(0.95)]$$

$$= -[-0.105 + (-0.105) + (-0.223) + (-0.051)]$$

Wait — let us be more careful. The binary cross-entropy for each pixel is:

$$-[x_i \log(p_i) + (1 - x_i)\log(1 - p_i)]$$

So:

- Pixel 1: $-[1 \cdot \log(0.9) + 0 \cdot \log(0.1)] = -(-0.105) = 0.105$
- Pixel 2: $-[0 \cdot \log(0.1) + 1 \cdot \log(0.9)] = -(-0.105) = 0.105$
- Pixel 3: $-[1 \cdot \log(0.8) + 0 \cdot \log(0.2)] = -(-0.223) = 0.223$
- Pixel 4: $-[1 \cdot \log(0.95) + 0 \cdot \log(0.05)] = -(-0.051) = 0.051$

Total reconstruction loss = 0.105 + 0.105 + 0.223 + 0.051 = **0.484**

This is a relatively low loss, which means the decoder is doing a good job at reconstruction. If all predictions were perfect (p = x exactly), this loss would be 0.

---

## The KL Divergence Term — Closed Form

Now let us look at the KL Divergence term in more detail. This is the term that regularizes the latent space.

We want to compute $D_{KL}(q_\phi(z|x) \,\|\, p(z))$, where:

- $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$ is the encoder's output
- $p(z) = \mathcal{N}(0, 1)$ is the prior (standard Gaussian)

The beautiful thing about KL divergence between two Gaussians is that it has a **closed-form solution**. We do not need to estimate it with sampling.

Let us derive it step by step.

**Step 1:** Write the KL divergence definition:

$$D_{KL}(q \,\|\, p) = \mathbb{E}_q\left[\log q(z) - \log p(z)\right]$$

**Step 2:** Substitute the Gaussian densities and simplify. After the algebra (which involves expanding the log of the Gaussian PDF and taking expectations), we get:

$$D_{KL} = -\frac{1}{2}\left(1 + \log(\sigma^2) - \mu^2 - \sigma^2\right)$$

For a latent space of dimension J, we sum over all dimensions:

$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{J}\left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$

Let us plug in some numbers. Suppose our encoder outputs μ = 0.5 and σ = 1.2 for a single latent dimension.

$$D_{KL} = -\frac{1}{2}(1 + \log(1.44) - 0.25 - 1.44)$$

$$= -\frac{1}{2}(1 + 0.3646 - 0.25 - 1.44)$$

$$= -\frac{1}{2}(-0.3254)$$

$$= 0.1627$$

A KL divergence of 0.1627 means the encoder's distribution is fairly close to the standard normal, but not perfectly aligned. The model is gently penalized for this deviation.

What if μ = 0 and σ = 1 (a perfect standard normal)?

$$D_{KL} = -\frac{1}{2}(1 + \log(1) - 0 - 1) = -\frac{1}{2}(1 + 0 - 0 - 1) = 0$$

The KL divergence is exactly 0. This makes sense — if the encoder's distribution is already a standard normal, there is no divergence to penalize. This is exactly what we want.

---

## The Reparameterization Trick

We have the ELBO, we know how to compute both terms, and we are ready to train. But there is one more technical hurdle.

### The Problem

During the forward pass, we need to **sample** z from the encoder's distribution $\mathcal{N}(\mu, \sigma^2)$. But sampling is a **random** operation — and random operations are not differentiable.

If we cannot compute gradients through the sampling step, we cannot backpropagate through the encoder. The training breaks down.


![The reparameterization trick moves randomness outside the computational graph, enabling backpropagation.](figures/figure_6.png)
*The reparameterization trick moves randomness outside the computational graph, enabling backpropagation.*


### The Solution

The reparameterization trick solves this with a simple but clever idea. Instead of sampling z directly from $\mathcal{N}(\mu, \sigma^2)$, we:

1. Sample ε from a standard normal: $\epsilon \sim \mathcal{N}(0, I)$
2. Compute z as a deterministic function of μ, σ, and ε:

$$z = \mu + \sigma \odot \epsilon$$

Here, ⊙ denotes element-wise multiplication.

The key insight is this: the randomness is now entirely in ε, which does **not** depend on the encoder's parameters. The quantities μ and σ are deterministic outputs of the encoder, and gradients can flow through them freely.

Let us plug in numbers. Suppose:

- μ = [2.1, -0.5]
- σ = [0.3, 0.2]
- We sample ε = [0.7, -1.1]

Then:

$$z = [2.1 + 0.3 \times 0.7,\; -0.5 + 0.2 \times (-1.1)]$$

$$z = [2.1 + 0.21,\; -0.5 + (-0.22)]$$

$$z = [2.31, -0.72]$$

If we had sampled a different ε = [-0.4, 0.8], we would get:

$$z = [2.1 + 0.3 \times (-0.4),\; -0.5 + 0.2 \times 0.8] = [1.98, -0.34]$$

Both are valid latent codes for the same image — just slightly different samples from the same distribution. And in both cases, the gradient with respect to μ and σ is well-defined.

---

## Full PyTorch Implementation

Enough theory — let us build a Variational Autoencoder from scratch in PyTorch.

We will train our VAE on the MNIST dataset of handwritten digits. Each image is 28×28 = 784 pixels, and we will use a 2-dimensional latent space so that we can visualize it easily.

### The Encoder

The encoder takes a 784-dimensional input (a flattened MNIST image) and outputs the mean μ and log-variance log(σ²) of the latent distribution.

We output log(σ²) instead of σ directly because log-variance can take any real value (positive or negative), which makes optimization more stable. We can always recover σ later with $\sigma = \exp(0.5 \cdot \log(\sigma^2))$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

Notice that the last layer splits into two heads: one for μ and one for log(σ²). This is because these are two independent quantities that the encoder needs to predict.

### The Decoder

The decoder takes a 2-dimensional latent code and outputs a 784-dimensional vector of pixel probabilities.

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=512, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))  # pixel probabilities between 0 and 1
```

The final sigmoid activation ensures that every output value is between 0 and 1 — since these represent pixel probabilities.

### The VAE: Putting It Together

Now we connect the encoder, reparameterization trick, and decoder into a single model.

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)        # convert log-variance to std deviation
        eps = torch.randn_like(std)           # sample epsilon from N(0, I)
        return mu + std * eps                 # the reparameterization trick

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
```

You can see here that we are using the exact same reparameterization formula which we derived earlier: $z = \mu + \sigma \cdot \epsilon$. We compute σ from log(σ²) using `torch.exp(0.5 * logvar)`.

### The Loss Function

The ELBO loss consists of two terms: reconstruction loss (binary cross-entropy) and KL divergence.

```python
def vae_loss(x_recon, x, mu, logvar):
    # Reconstruction loss: binary cross-entropy summed over pixels
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence: closed-form formula for Gaussian vs standard normal
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
```

This is the exact ELBO formula translated into code. The reconstruction loss uses PyTorch's built-in binary cross-entropy function. The KL divergence uses the closed-form expression that we derived earlier: $-\frac{1}{2}\sum(1 + \log(\sigma^2) - \mu^2 - \sigma^2)$.

### Training Loop

Finally, let us put it all together with a training loop on MNIST.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(latent_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Data ---
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# --- Training ---
model.train()
for epoch in range(20):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)  # flatten 28x28 to 784

        optimizer.zero_grad()
        x_recon, mu, logvar = model(data)
        loss = vae_loss(x_recon, data, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/20, Loss: {avg_loss:.2f}')
```

Let us understand this code. We load the MNIST dataset, flatten each 28×28 image into a 784-dimensional vector, pass it through our VAE, compute the ELBO loss, and backpropagate. After 20 epochs, our VAE should have learned a meaningful latent representation of handwritten digits.

---

## Training Results on MNIST

Let us see what our VAE has learned after training.

### Reconstructions Over Epochs

First, let us look at how the reconstructions improve over the course of training.


![After 20 epochs, the VAE reconstructs digits that are recognizable but slightly softer than the originals.](figures/figure_7.png)
*After 20 epochs, the VAE reconstructs digits that are recognizable but slightly softer than the originals.*


The reconstructions are not pixel-perfect — they are slightly blurry compared to the originals. But the digit identity is clearly preserved. A 7 looks like a 7, a 4 looks like a 4. The model has captured the essential structure.

### Latent Space Visualization

Because we chose a 2-dimensional latent space, we can directly visualize it. Let us pass every image in the test set through the encoder and plot the mean μ, colored by the digit class.


![Each digit class forms a distinct cluster in the 2D latent space, with smooth transitions between classes.](figures/figure_8.png)
*Each digit class forms a distinct cluster in the 2D latent space, with smooth transitions between classes.*


We can clearly see that the VAE has learned to separate different digits into different regions of the latent space. Digits that look similar (like 4 and 9, or 3 and 8) are placed close together, which makes intuitive sense.

This is exactly what we want — a structured, meaningful latent space where proximity implies similarity.

### Sampling New Digits

Now for the exciting part. Let us sample random points from the latent space and pass them through the decoder to generate brand new digits that the model has never seen before.


![Sampling random points from the standard normal prior produces diverse, recognizable handwritten digits.](figures/figure_9.png)
*Sampling random points from the standard normal prior produces diverse, recognizable handwritten digits.*


Not bad, right? The VAE generates diverse digits by simply sampling from a standard Gaussian and passing through the decoder. Some digits are cleaner than others, but they are all recognizable.

---

## Latent Space Interpolation

One of the most beautiful properties of a VAE is that the latent space is **continuous and smooth**. This means we can smoothly interpolate between two images by moving along a straight line in the latent space.

Let us take two images — say, a **3** and an **8** — encode them to get their latent representations z₃ and z₈, and then generate images at evenly spaced points along the line from z₃ to z₈.


![Moving linearly through latent space produces a smooth, gradual transformation from one digit to another.](figures/figure_10.png)
*Moving linearly through latent space produces a smooth, gradual transformation from one digit to another.*


Notice how the transition is completely smooth — there are no sudden jumps or glitches. The digit gradually morphs from a 3 to an 8. This tells us that the VAE has learned a *continuous* and *meaningful* latent space where nearby points produce similar images.

This would not be possible with a regular autoencoder, where the gaps between encoded points would produce garbage.

---

## Why VAEs Produce Blurry Images

You might have noticed that the generated images are a bit blurry. This is not a bug — it is a fundamental limitation of the standard VAE framework.

The reason is the decoder distribution. When we use a Gaussian (or Bernoulli) distribution for each pixel independently, the model is forced to hedge its bets. If a pixel could be either black or white depending on the sample, the model will predict a value in the middle (gray). This averaging over modes causes the characteristic blurriness.

Think of it this way: if you ask someone to draw a digit that is "somewhere between a 3 and an 8," they will draw something vague and blurry rather than committing to one or the other.

This limitation motivated the development of **diffusion models**, which sidestep the encoder-decoder joint training issue and can generate much sharper images.

Despite this limitation, VAEs remain one of the most elegant frameworks in generative modeling. They teach us the fundamental insight that **generation = learning a latent space + sampling from it.** This idea carries forward into virtually every modern generative model, from diffusion models to large language models.

---

## Closing

That's it! We have built a Variational Autoencoder completely from scratch — starting from the intuition of latent variables and secret recipes, through the full mathematical derivation of the ELBO with Jensen's inequality, the closed-form KL divergence, and the reparameterization trick, all the way to a working PyTorch implementation trained on MNIST.

Here is a summary of the key ideas:

1. **Latent variables** capture the hidden factors that determine the data
2. The **encoder** maps data to distributions in latent space (not points)
3. The **decoder** maps latent codes back to data
4. The **ELBO** balances reconstruction quality against latent space regularity
5. The **reparameterization trick** enables gradient-based training through sampling
6. The result is a **smooth, continuous latent space** that supports generation and interpolation

Thanks!

---

**References:**

- Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
- Doersch, "Tutorial on Variational Autoencoders" (2016)
- Blei, Kucukelbir & McAuliffe, "Variational Inference: A Review for Statisticians" (2017)
