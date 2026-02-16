# Diffusion LLMs from Scratch

*What if language models could generate all tokens at once — like image diffusion, but for text?*

---

Let us start with a simple analogy. Imagine you are writing an essay on a typewriter. You type one letter at a time, left to right. Once a letter is pressed onto the page, it is permanent — you cannot go back and change it. If you realize halfway through a sentence that the beginning was wrong, too bad. You are stuck with it.

This is exactly how modern large language models like GPT-4 and LLaMA work. They generate text one token at a time, from left to right. Each token depends on all the previous tokens, but it has absolutely no knowledge of what will come after it. The model commits to each word before it knows how the sentence will end.


![Autoregressive models generate token-by-token; diffusion models refine the entire sequence simultaneously.](figures/figure_1.png)
*Autoregressive models generate token-by-token; diffusion models refine the entire sequence simultaneously.*


Now think of how an artist works. An artist does not paint the top-left pixel first, then the next pixel, then the next. Instead, they start with a rough sketch of the whole canvas, then progressively refine the details — adding color here, sharpening edges there, going back to fix proportions. The whole image comes into focus *at the same time.*

This is how diffusion models generate images. They start from pure noise and refine the entire image simultaneously, step by step. And recently, researchers have asked a bold question:

**Can we do the same thing for text?**

The answer is yes — and the resulting models are called **Diffusion LLMs**. In this article, we will understand how they work completely from scratch.

## A Quick Recap: How Diffusion Works for Images

Before we dive into text, let us make sure we understand how diffusion works for images. If you already know this, feel free to skim through — but the intuition here will be crucial for understanding the text version.

The idea behind image diffusion is beautifully simple. It has two phases:

**Phase 1 — The Forward Process (Destroying the image):**

Take a clean image. Gradually add random Gaussian noise to it, step by step. After enough steps, the image becomes pure random noise — no trace of the original remains.

**Phase 2 — The Reverse Process (Reconstructing the image):**

Train a neural network that learns to reverse each noise step. Given a noisy image, predict what it looked like one step earlier (slightly less noisy). Chain these predictions together, and you can go from pure noise all the way back to a clean image.


![The forward process adds noise until the image is destroyed; the reverse process learns to undo it.](figures/figure_2.png)
*The forward process adds noise until the image is destroyed; the reverse process learns to undo it.*


The forward process can be written mathematically. At any timestep $t$, the noisy image is:


$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$


Let us plug in some simple numbers to see how this works. Suppose we have a single pixel with value $x_0 = 0.8$ (a bright pixel), and at timestep $t$, the noise schedule gives us $\bar{\alpha}_t = 0.5$. A random noise sample gives $\epsilon = 0.3$.

$$x_t = \sqrt{0.5} \times 0.8 + \sqrt{0.5} \times 0.3 = 0.707 \times 0.8 + 0.707 \times 0.3 = 0.566 + 0.212 = 0.778$$

The pixel value has shifted from 0.8 towards the noise. As $\bar{\alpha}_t$ decreases towards 0 (more noise), the first term shrinks and the noise term dominates. At $\bar{\alpha}_t = 0$, the original image is completely gone — this is exactly what we want.

The key insight is this: **the neural network refines all pixels simultaneously at each step.** It does not generate the top-left pixel first, then the next one. It sees the entire noisy image and predicts improvements for every pixel at once.

Now here is where things get interesting.

## "But Wait — Diffusion Is for Images, Not Text!"

If you have been following along, you might be thinking: *"Okay, diffusion is great for images. But images are made of pixels, which are continuous numbers between 0 and 1. Text is made of discrete tokens — words like 'cat', 'the', 'sat'. You cannot add half a unit of Gaussian noise to the word 'cat'. That does not even make sense."*

And you would be absolutely right. This is the central challenge.

Let us make this concrete. In image diffusion, you can take a pixel with value 0.8 and add noise to get 0.73 or 0.85. That is a perfectly valid pixel value. But in text, your tokens are integers — 'cat' might be token 3421, 'dog' might be token 7856. What does token 5638.5 mean? Nothing. It is not a real word.


![Gaussian noise works for continuous pixels but breaks for discrete tokens.](figures/figure_3.png)
*Gaussian noise works for continuous pixels but breaks for discrete tokens.*


So people have tried three approaches to solve this:

1. **Continuous embedding diffusion** (Diffusion-LM, Li et al., 2022) — Embed each token into a continuous vector space, run standard Gaussian diffusion on the embeddings, then round back to the nearest token. Clever, but the rounding step introduces errors.

2. **Masked diffusion** (MDLM, LLaDA) — Instead of adding Gaussian noise, *replace tokens with a [MASK] token.* Masking is a natural form of "noise" for discrete data.

3. **Score-based discrete diffusion** (SEDD) — Define a score function directly in discrete space using probability ratios. Mathematically elegant, but harder to build intuition for.

In this article, we will focus on **approach number 2 — masked diffusion.** It is the simplest, the most intuitive, and the most commercially successful. It powers models like Mercury (the first commercial diffusion LLM, generating 1000+ tokens per second) and LLaDA (competitive with LLaMA 3).

And the beautiful thing? If you understand BERT's masked language modeling, you are already 80% of the way there.

## The Big Intuition: Masking Is the "Noise" for Text

Let us build the intuition step by step.

In image diffusion, the forward process *destroys* the image by adding Gaussian noise. At the end, you have pure random static — no information from the original image remains.

Now think about text. What is the equivalent of "destroying" a sentence? **Masking out the words.** If you mask every single word in a sentence, no information remains — just like pure noise for an image.

Let us see this with a real example. Take the sentence: **"The cat sat on the mat"**

| Timestep $t$ | Masking probability | What the model sees |
|---|---|---|
| $t = 0.0$ | 0% masked | The cat sat on the mat |
| $t = 0.2$ | 20% masked | The cat sat [M] the mat |
| $t = 0.5$ | 50% masked | [M] cat [M] on [M] mat |
| $t = 0.8$ | 80% masked | [M] [M] [M] on [M] [M] |
| $t = 1.0$ | 100% masked | [M] [M] [M] [M] [M] [M] |

At $t = 0$, we have the clean sentence. At $t = 1$, we have pure "noise" — all information is destroyed. This is the exact analogy of going from a clean image to pure static.


![The forward process gradually masks tokens until the entire sequence is destroyed.](figures/figure_4.png)
*The forward process gradually masks tokens until the entire sequence is destroyed.*


Mathematically, the forward process is elegantly simple. At timestep $t$, each token is independently masked with probability $t$:


$$
q(x_t^{(i)} \mid x_0^{(i)}) = \begin{cases} x_0^{(i)} & \text{with probability } 1 - t \\ [\text{MASK}] & \text{with probability } t \end{cases}
$$

Let us plug in some numbers. Suppose we have a sentence with 10 tokens and the timestep is $t = 0.4$. Each token has a 40% chance of being masked. The expected number of masked tokens is $10 \times 0.4 = 4$. So on average, 4 out of 10 tokens will be replaced with [MASK], and the other 6 remain visible.

Now here is the crucial connection: **Does this remind you of something?**

If you have studied BERT, you know that BERT is trained by masking 15% of tokens and predicting what they should be. This is called **Masked Language Modeling (MLM)**.

Diffusion LLMs do exactly the same thing — but instead of always masking 15%, they train with *every possible masking ratio from 0% to 100%.* This single change transforms BERT from a language understanding model into a full generative model.

This is the key insight: **Masked diffusion for text is just BERT training, generalized across all masking ratios.**

## The Reverse Process: Learning to Unmask

Now let us understand how the model learns to reverse this process.

We train a neural network — specifically, a bidirectional Transformer — that takes a partially masked sequence as input and predicts the original tokens for all masked positions *simultaneously*.


![The bidirectional Transformer predicts all masked tokens simultaneously using full context.](figures/figure_5.png)
*The bidirectional Transformer predicts all masked tokens simultaneously using full context.*


There are two key things to notice:

**First, the attention is bidirectional.** Unlike GPT, which can only look at tokens to the left, this model sees *all* unmasked tokens — both to the left and to the right. When predicting what the second [MASK] should be in "The [M] sat [M] the [M]", the model can use both "The" and "sat" on the left, and "the" on the right. It has full context.

**Second, all masked tokens are predicted at once.** The model does not predict the first mask, then the second, then the third. It predicts all of them in a single forward pass. This is what makes diffusion LLMs fast.

The training objective is cross-entropy loss on the masked positions — exactly the same as BERT:


$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim U(0,1)} \, \mathbb{E}_{x_t \sim q(x_t \mid x_0)} \left[ -\sum_{i : x_t^{(i)} = [\text{MASK}]} \log p_\theta\!\left(x_0^{(i)} \mid x_t\right) \right]
$$


Let us break this down term by term:

- $t \sim U(0,1)$: We randomly sample a masking ratio between 0% and 100%
- $x_t \sim q(x_t \mid x_0)$: We create a corrupted version of the clean text by masking tokens with probability $t$
- The sum runs over all masked positions $i$
- $\log p_\theta(x_0^{(i)} \mid x_t)$: We compute the log probability that the model assigns to the correct token at each masked position

Let us work through a concrete example. Suppose we have the sentence "The cat sat" (3 tokens), and we sample $t = 0.67$, which masks 2 out of 3 tokens: "[M] cat [M]". The model predicts:

- Position 1: P("The") = 0.6, P("A") = 0.3, P("This") = 0.1
- Position 3: P("sat") = 0.8, P("ran") = 0.15, P("ate") = 0.05

The loss for this example is:

$$\mathcal{L} = -\log(0.6) - \log(0.8) = 0.511 + 0.223 = 0.734$$

As training progresses, the model gets better at predicting masked tokens, and this loss decreases. This is exactly what we want.

## Generation: How to Actually Create Text

Now comes the exciting part. We have trained a model that can predict masked tokens. How do we use it to *generate* brand new text?

The process is called **iterative unmasking**, and it mirrors the reverse process of image diffusion:

1. Start with a completely masked sequence: [M] [M] [M] [M] [M] [M]
2. Run the model — it predicts tokens for all positions
3. Keep the predictions where the model is most confident, re-mask the rest
4. Repeat for several steps until all tokens are unmasked

Let us walk through this step by step with a real example. Suppose we want to generate a 6-token sentence in 4 steps:

**Step 1:** Input: [M] [M] [M] [M] [M] [M]

The model predicts all 6 tokens simultaneously. It is most confident about tokens 1 and 6 (perhaps common words like "The" and "mat" that fit many contexts). We unmask those and keep the rest masked.

Result: **The** [M] [M] [M] [M] **mat**

**Step 2:** Input: The [M] [M] [M] [M] mat

Now the model has context — it sees "The" at the start and "mat" at the end. It confidently predicts positions 3 and 5.

Result: The [M] **sat** [M] **the** mat

**Step 3:** Input: The [M] sat [M] the mat

With even more context, the model fills in position 2.

Result: The **cat** sat [M] the mat

**Step 4:** Input: The cat sat [M] the mat

Only one mask remains. With full surrounding context, the model confidently predicts "on."

Result: **The cat sat on the mat**


![Generation works by iteratively unmasking tokens, starting with the most confident predictions.](figures/figure_6.png)
*Generation works by iteratively unmasking tokens, starting with the most confident predictions.*


Notice something remarkable: **the model filled in "mat" at the end before it filled in "cat" in the middle.** It did not go left to right. It generated the tokens in order of confidence, not position. And at every step, it had access to both the left and right context.

This is fundamentally different from autoregressive models, which *must* generate left to right and can never "look ahead."

## The Math: Why This Is a Valid Diffusion Process

You might be wondering: *"This all makes intuitive sense, but is it mathematically rigorous? Is it really a proper diffusion model?"*

The answer is yes. Let us connect this to the formal framework.

In any diffusion model, we want to maximize the log-likelihood of our data, $\log p_\theta(x_0)$. Directly computing this is intractable, so we optimize a lower bound called the **Evidence Lower Bound (ELBO)**:


$$
\log p_\theta(x_0) \geq \text{ELBO} = \mathbb{E}_{q} \left[ \log p_\theta(x_0 \mid x_1) + \sum_{t=2}^{T} \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1}, x_0)} \right]
$$


This looks complicated, but the intuition is straightforward. The ELBO measures how well our reverse process (unmasking) matches the true reverse of the forward process (masking). If our model is perfect at predicting masked tokens, the ELBO is tight and equals the true log-likelihood.

Let us plug in some numbers for a small example. Suppose we have a 2-token vocabulary {A, B}, a 3-token sequence, and T = 2 steps.

At step $t=2$, the model starts from a fully masked sequence and unmasks some tokens. At step $t=1$, it unmasks the rest. If the model assigns probability 0.9 to the correct token at each step:

$$\text{ELBO} \approx \log(0.9) + \log(0.9) + \log(0.9) = 3 \times (-0.105) = -0.315$$

A perfect model would give $\text{ELBO} = 0$ (probability 1 for every token). As our model improves, the ELBO increases towards 0.

Here is the beautiful result from the MDLM paper (NeurIPS 2024): **when you work through the math for masked diffusion, this ELBO simplifies to a weighted mixture of masked language modeling losses at different masking ratios.**

In other words, the theoretically rigorous diffusion training objective turns out to be almost identical to "train BERT at all masking ratios." The theory confirms what the intuition suggested.

## Building It from Scratch: A Minimal Implementation

Now let us implement a minimal diffusion LLM in PyTorch. We will build four components: the forward process, the model, the training loop, and the generation procedure.

**The Forward Process:**

This is the masking function. Given clean tokens and a timestep $t$, it randomly masks tokens with probability $t$.

```python
import torch
import torch.nn.functional as F

MASK_TOKEN_ID = 0  # We reserve token 0 as [MASK]

def forward_process(x_0, t):
    """
    x_0: clean token IDs, shape (batch_size, seq_len)
    t: masking probability, shape (batch_size, 1)
    Returns: masked tokens x_t and the boolean mask
    """
    # Each token is independently masked with probability t
    mask = torch.rand_like(x_0.float()) < t
    x_t = x_0.clone()
    x_t[mask] = MASK_TOKEN_ID
    return x_t, mask
```

Let us understand this code in detail. We generate a random number between 0 and 1 for each token. If that number is less than $t$, the token gets masked. So when $t = 0.3$, roughly 30% of tokens will be replaced with [MASK]. When $t = 1.0$, every token gets masked.

**The Model:**

We use a standard bidirectional Transformer encoder. The key difference from GPT: **no causal mask.** Every token can attend to every other token.

```python
import torch.nn as nn
import math

class DiffusionLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        # Timestep embedding — the model needs to know the noise level
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x_t, t):
        """
        x_t: masked token IDs (batch_size, seq_len)
        t: timestep (batch_size, 1)
        Returns: logits (batch_size, seq_len, vocab_size)
        """
        seq_len = x_t.shape[1]
        positions = torch.arange(seq_len, device=x_t.device)
        h = self.token_embed(x_t) + self.pos_embed(positions)
        # Add timestep information
        h = h + self.time_embed(t).unsqueeze(1)
        # Bidirectional Transformer — no causal mask!
        h = self.transformer(h)
        return self.output_head(h)
```

Notice that we pass no attention mask to the Transformer. In GPT, a causal mask prevents tokens from attending to future positions. Here, we intentionally allow full bidirectional attention — this is the superpower of diffusion LLMs.

**The Training Loop:**

The training loop is remarkably simple. Sample a random masking ratio, mask the tokens, predict, and compute cross-entropy loss on the masked positions.

```python
model = DiffusionLM(vocab_size=10000)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:  # batch shape: (batch_size, seq_len)
    # Sample a random masking ratio for each example
    t = torch.rand(batch.shape[0], 1, device=batch.device)

    # Forward process: mask tokens
    x_t, mask = forward_process(batch, t)

    # Model predicts logits for every position
    logits = model(x_t, t)

    # Loss only on masked positions
    loss = F.cross_entropy(
        logits[mask],     # predictions at masked positions
        batch[mask],      # true tokens at masked positions
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This is essentially BERT training with a random masking ratio. That is the beauty of this approach — the implementation is simple, but the theory guarantees it is a valid diffusion model.

**Generation (Iterative Unmasking):**

Finally, the generation procedure. We start fully masked and iteratively unmask.

```python
@torch.no_grad()
def generate(model, seq_len, num_steps=10):
    # Start with all [MASK] tokens
    x = torch.full((1, seq_len), MASK_TOKEN_ID)

    for step in range(num_steps):
        # Current noise level (decreasing from 1 to 0)
        t = torch.tensor([[1.0 - step / num_steps]])

        # Model predicts all tokens
        logits = model(x, t)
        probs = F.softmax(logits, dim=-1)

        # Sample tokens from the predicted distribution
        predicted = torch.multinomial(
            probs.view(-1, probs.shape[-1]), 1
        ).view(x.shape)

        # Compute confidence for each prediction
        confidence = probs.max(dim=-1).values

        # Number of tokens to unmask at this step
        is_masked = (x == MASK_TOKEN_ID)
        n_to_unmask = max(1, int(is_masked.sum() / (num_steps - step)))

        # Unmask the most confident predictions
        masked_confidence = confidence * is_masked.float()
        _, top_indices = masked_confidence.view(-1).topk(n_to_unmask)
        x.view(-1)[top_indices] = predicted.view(-1)[top_indices]

    return x
```

Let us understand the generation strategy. At each step, the model predicts tokens for all masked positions. We look at the confidence (the highest probability assigned by the model) and unmask the positions where the model is most sure. The less confident positions stay masked for later steps, when the model will have more context to work with.

## How Do Diffusion LLMs Compare?

Now that we understand how diffusion LLMs work, let us look at how well they actually perform. The progress in the last two years has been remarkable.

| Model | Year | Approach | Key Result |
|---|---|---|---|
| MDLM | 2024 | Masked diffusion | Within 14% of GPT-2 perplexity (NeurIPS 2024) |
| SEDD | 2024 | Score-based discrete diffusion | 6-8× better perplexity than un-annealed GPT-2 (ICML Best Paper) |
| LLaDA | 2025 | Masked diffusion (8B params) | Competitive with LLaMA 3; beats GPT-4o on reversal tasks |
| Mercury | 2025 | Diffusion (commercial) | 1,000+ tokens/sec on H100 — 10× faster than autoregressive |
| Gemini Diffusion | 2025 | Diffusion (commercial) | 1,479 tokens/sec — 5× faster than comparable models |


![Diffusion LLMs achieve 5-10x higher throughput than autoregressive models.](figures/figure_7.png)
*Diffusion LLMs achieve 5-10x higher throughput than autoregressive models.*


The standout result is **LLaDA** — a diffusion LLM trained from scratch at 8 billion parameters that is competitive with LLaMA 3. Even more remarkably, LLaDA solves the famous **reversal curse.** Autoregressive models like GPT-4o, when trained that "A is B," often cannot infer that "B is A" — because they only learn left-to-right patterns. LLaDA, with its bidirectional context, handles this naturally.

## Why This Matters

Let us step back and appreciate what diffusion LLMs bring to the table:

**1. Speed.** Because diffusion LLMs predict multiple tokens in parallel, they can be dramatically faster. Mercury generates over 1,000 tokens per second — fast enough for real-time applications that were previously impossible.

**2. Bidirectional context.** Every token sees both past and future context at every generation step. This means the model can use the ending of a sentence to inform the beginning. Autoregressive models can never do this.

**3. Controllable generation.** Want to fill in the middle of a sentence while keeping the beginning and end fixed? Diffusion LLMs do this naturally — just mask the middle and unmask. For autoregressive models, this kind of "infilling" requires special tricks.

**4. Error correction.** During iterative refinement, the model can effectively "change its mind." A token that was unmasked in step 3 informs the predictions in step 4. If early predictions create a coherent context, later predictions will be better. This is like the artist stepping back to look at the whole painting before refining details.


![Diffusion LLMs offer bidirectional context, parallel generation, and natural editing capabilities.](figures/figure_8.png)
*Diffusion LLMs offer bidirectional context, parallel generation, and natural editing capabilities.*


Of course, challenges remain. Diffusion LLMs currently require more training compute than autoregressive models of similar size. They can struggle with tasks that require strict sequential reasoning (like counting or arithmetic). And generating variable-length text — where you do not know the output length in advance — requires additional techniques.

But the progress has been breathtaking. In just two years, we have gone from "diffusion cannot possibly work for text" to commercial models that outperform autoregressive baselines on speed by 10×.

## Conclusion

Let us recap our journey. We started with the intuition that autoregressive LLMs are like typewriters — sequential, one-directional, and permanent. We then asked: what if we could generate text like diffusion generates images?

The core insight was surprisingly simple: **replace Gaussian noise with token masking.** The forward process randomly masks tokens. The reverse process — a bidirectional Transformer — learns to predict masked tokens. Generation works by iteratively unmasking from a fully masked sequence, starting with the most confident predictions.

The math confirms that this is a rigorous diffusion process, with the training objective simplifying to masked language modeling across all masking ratios. And the results speak for themselves: LLaDA matches LLaMA 3, Mercury generates 1,000+ tokens per second, and the reversal curse is solved.

We are witnessing the emergence of a fundamentally new paradigm for language generation. The era of one-token-at-a-time may be coming to an end.

**References:**

- Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024)
- Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (ICML 2024, Best Paper)
- Nie et al., "Large Language Diffusion Models (LLaDA)" (2025)
- Inception Labs, "Mercury: Ultra-Fast Language Models Based on Diffusion" (2025)
- Li et al., "Diffusion-LM Improves Controllable Text Generation" (NeurIPS 2022)
