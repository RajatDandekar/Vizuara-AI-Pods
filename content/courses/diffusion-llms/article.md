# Diffusion LLMs: What If Language Models Could Think Before They Speak?

*From masking to unmasking — how diffusion models are challenging the autoregressive paradigm in text generation.*

---

## The Typewriter vs. The Painter

Let us start with a simple analogy. Imagine two people writing an essay.

The first person uses a **typewriter**. They type one letter at a time, left to right, line by line. Once a letter is stamped onto the page, there is no going back. Every word they write commits them to a direction — they cannot revise what came before. If they realize halfway through a sentence that the opening was wrong, too bad. The typewriter only moves forward.

The second person is a **painter**. They start with a blank canvas and sketch a rough outline of the entire composition. Then, they progressively refine every part simultaneously — adjusting proportions here, adding detail there, fixing colors everywhere. At no point are they locked into a single direction. The painting emerges holistically, from coarse to fine.


![figures/figure_1.png](figures/figure_1.png)
*figures/figure_1.png*


**Caption:** Autoregressive models generate text one token at a time from left to right, while diffusion models refine all tokens simultaneously from noise to clarity.

Now here is the thing: every large language model you have used — GPT-4, Claude, LLaMA, Gemini — they are all typewriters. They generate text **autoregressively**, predicting one token at a time, always left to right. The probability of the next token depends only on the tokens that came before it.

This approach has been extraordinarily successful. But it comes with some fundamental limitations:

1. **Speed bottleneck:** Since each token must wait for the previous one, generation is inherently sequential. If your response is 1,000 tokens long, you need 1,000 forward passes through the model. There is no shortcut.

2. **The reversal curse:** Autoregressive models can complete "Line 1 → Line 2" of a poem, but they struggle terribly with "Line 2 → Line 1." Since they only model left-to-right dependencies, they literally cannot think backwards.

3. **No global planning:** The model commits to each token without ever seeing the full picture. It is like writing a mystery novel one word at a time without knowing who the murderer is.

So, what if we could build a language model that works like the painter — one that sees the entire output at once and refines it from coarse to fine?

This is exactly the idea behind **Diffusion LLMs**.

---

## From Pixels to Words — Can We Diffuse Text?

This brings us to a natural question. We have seen in previous articles how diffusion models work beautifully for images. In the forward process, we gradually add Gaussian noise to an image until it becomes pure static. In the reverse process, a neural network learns to denoise, step by step, recovering the original image from randomness.

Can we apply the same idea to text?

At first glance, this seems straightforward. But there is a fundamental problem: **images are continuous, text is discrete.**

A pixel value can be 127.3 or 127.8 — you can add a small amount of Gaussian noise and get 128.1, and that is still a perfectly valid pixel value. But the word "cat" is a discrete token. You cannot add "half a noise" to "cat" and get something between "cat" and "dog." It does not make sense.


![figures/figure_2.png](figures/figure_2.png)
*figures/figure_2.png*


**Caption:** Gaussian noise works naturally for continuous pixel values but breaks down for discrete text tokens.

Researchers explored three main approaches to solve this:

1. **Embedding-space diffusion** (Diffusion-LM, Li et al. 2022): Add Gaussian noise in the continuous embedding space, then round back to the nearest token. This works, but the rounding step introduces errors and the approach struggles to scale.

2. **Discrete transition matrices** (D3PM, Austin et al. 2021): Define Markov transition matrices directly over the vocabulary. At each step, a token can randomly transition to any other token. Elegant in theory, but the transition matrices are enormous and optimization is unstable.

3. **Masked diffusion** (MDLM, LLaDA): Instead of adding noise, simply **replace tokens with a [MASK] token**. This is the absorbing state approach — once a token is masked, it stays masked until the reverse process recovers it.

It turns out that the third approach — masked diffusion — is the one that works best in practice. It is simple, stable, and scales beautifully. Let us understand it in detail.

---

## Masked Diffusion — The Winning Recipe

Let us understand the masked diffusion approach, which has emerged as the most practical and powerful method for building diffusion language models.

### The Forward Process: Gradually Masking the Text

Consider a clean sentence:

> **"The cat sat on the mat"**

In the forward process, we gradually mask tokens. At each timestep $t$ (where $t$ goes from 0 to 1), we independently decide for each token: should it stay or should it be replaced with [MASK]?

The probability that any given token gets masked is simply $t$. So at $t = 0$, nothing is masked. At $t = 0.25$, roughly 25% of tokens are masked. At $t = 1$, everything is masked.


![figures/figure_3.png](figures/figure_3.png)
*figures/figure_3.png*


**Caption:** The forward process gradually masks tokens. At $t = 0$, the sentence is clean. At $t = 1$, everything is masked.

Let us write this mathematically. For a sequence of $L$ tokens, the forward process at time $t$ is:

$$q(x_t \mid x_0) = \prod_{i=1}^{L} q(x_t^i \mid x_0^i)$$

where each token is independently masked with probability $t$:

$$q(x_t^i \mid x_0^i) = (1 - t) \cdot \mathbb{1}[x_t^i = x_0^i] + t \cdot \mathbb{1}[x_t^i = \texttt{[MASK]}]$$

In plain English: at time $t$, each token either stays as itself (with probability $1 - t$) or gets replaced with [MASK] (with probability $t$).

Let us plug in some simple numbers to see how this works. Suppose we have a 4-token sentence: ["The", "cat", "sat", "here"] and $t = 0.3$.

For each token, there is a 70% chance it stays and a 30% chance it becomes [MASK]. The probability of getting the configuration ["The", [MASK], "sat", [MASK]] would be:

$$P = 0.7 \times 0.3 \times 0.7 \times 0.3 = 0.0441$$

This tells us there is about a 4.4% chance of this specific masking pattern at $t = 0.3$. There are many possible configurations, but the expected number of masked tokens is $0.3 \times 4 = 1.2$ — roughly 1 to 2 tokens masked on average. This is exactly what we want.

### The Reverse Process: Unmasking to Generate Text

Now comes the interesting part. The reverse process starts from a **fully masked sequence** and progressively unmasks tokens to generate text.

Here is how it works:

1. Start with all [MASK] tokens
2. Feed the entire masked sequence into a Transformer
3. The model predicts what each [MASK] token should be, along with a **confidence score**
4. Unmask the tokens with the **highest confidence** (these are the predictions the model is most sure about)
5. **Remask** the low-confidence predictions — the model will try again next step
6. Repeat until all tokens are unmasked


![figures/figure_4.png](figures/figure_4.png)
*figures/figure_4.png*


**Caption:** During generation, the model unmasks high-confidence tokens first and remasks uncertain ones for re-evaluation.

This is a powerful idea. Notice how the model gets to **reconsider** its uncertain predictions. In an autoregressive model, once a token is generated, it is set in stone. Here, the model can think: "I am not sure about this word yet, let me see what the other words look like first, and then I will decide."

This is also why diffusion LLMs can handle the **reversal curse.** Since the model sees all positions simultaneously — not just the left context — it can fill in tokens in any order. There is no inherent "direction" to the generation.

---

## The Training Objective — Simpler Than You Think

Now the question is: how do we actually train this model?

You might expect a complex loss function involving KL divergences, ELBO bounds, and noise schedules — just like we saw for image diffusion models. And indeed, the theoretical derivation does go through those steps. But the beautiful thing about masked diffusion is that the training objective simplifies to something remarkably clean.

The loss function is:

$$\mathcal{L} = -\mathbb{E}_{t \sim U(0,1)} \left[ \frac{1}{t \cdot L} \sum_{i:\, x_t^i = \texttt{[MASK]}} \log p_\theta(x_0^i \mid x_t) \right]$$

Let us break this down:
- Sample a random masking ratio $t$ uniformly from $(0, 1)$
- Mask $t$ fraction of the tokens in the input
- Ask the model to predict the original token at each masked position
- Compute the negative log-likelihood of the correct predictions
- Average over the masked positions (dividing by $t \cdot L$, the expected number of masks)

Let us plug in some simple numbers. Suppose we have a 4-token sentence $x_0$ = ["The", "cat", "sat", "here"], and we sample $t = 0.5$. This means 2 out of 4 tokens get masked. Say the masked sequence is: ["The", [MASK], [MASK], "here"].

The model looks at this and predicts:
- Position 2: $p_\theta(\text{"cat"} \mid x_t) = 0.8$ (pretty confident)
- Position 3: $p_\theta(\text{"sat"} \mid x_t) = 0.6$ (less confident)

The loss for this sample would be:

$$\mathcal{L} = -\frac{1}{0.5 \times 4} \left[ \log(0.8) + \log(0.6) \right] = -\frac{1}{2} \left[ -0.223 + (-0.511) \right] = \frac{0.734}{2} = 0.367$$

This tells us that the model's predictions were decent but not perfect — the loss would be 0 only if the model predicted both tokens with probability 1.0. The model will be trained to minimize this loss, pushing it to make more confident and accurate predictions.

Now, you might be thinking: *"Wait a moment. This looks a lot like BERT's masked language modeling!"*

And you would be absolutely right. But there are two crucial differences:


![figures/figure_5.png](figures/figure_5.png)
*figures/figure_5.png*


**Caption:** BERT uses a fixed 15% masking rate for understanding tasks, while Diffusion LLMs use a variable masking rate that provides a principled generative objective.

1. **BERT** uses a fixed masking rate of 15%. Diffusion LLMs sample $t$ uniformly from $(0, 1)$, so the model sees every possible masking ratio during training — from nearly clean to nearly fully masked. This is critical for generation, because the model needs to handle all stages of the denoising process.

2. **BERT** is trained for understanding (classification, question answering). The Diffusion LLM objective is a **variational lower bound on the log-likelihood**, which makes it a principled generative model. This subtle difference in the weighting ($1/t$) is what transforms a simple MLM loss into a proper generative training objective.

The Transformer architecture itself is a **vanilla Transformer** — very similar to LLaMA — but with one key change: it uses **bidirectional attention** instead of causal (left-to-right) masking. This makes sense because during generation, the model needs to see all positions — both the unmasked tokens and the [MASK] tokens — to make its predictions.

---

## Generation — How Diffusion LLMs Write

We have trained our model. Now let us see how it actually generates text.

Suppose we have a prompt: "Explain quantum computing in simple terms." The model needs to generate a response. Here is the algorithm:

**Step 1:** Create a response of length $L$ filled entirely with [MASK] tokens. (The length $L$ can be fixed or predicted.)

**Step 2:** Choose the number of sampling steps $S$ (typically 10 to 50).

**Step 3:** For each step $s$ from $S$ down to 1:
- Feed the prompt + current response (with masks) into the Transformer
- The model predicts a probability distribution over the vocabulary at every masked position
- Sample a token from each predicted distribution
- Compute a **confidence score** for each prediction
- Keep the top $(1 - s/S)$ fraction of predictions (the most confident ones)
- **Remask** the rest — they will be re-predicted in the next step

**Step 4:** At the final step, unmask everything.

Let us trace through a small example. Say we want to generate a 6-token response with $S = 3$ steps.

| Step | Sequence | Confidence | Action |
|------|----------|------------|--------|
| Start | [M] [M] [M] [M] [M] [M] | — | All masked |
| s=3 | "It" [M] "a" [M] [M] [M] | 0.92, 0.35, 0.88, 0.41, 0.29, 0.33 | Keep top-2 (highest confidence), remask rest |
| s=2 | "It" [M] "a" "way" "to" [M] | 0.92, 0.45, 0.88, 0.78, 0.81, 0.52 | Keep top-4, remask rest |
| s=1 | "It" "is" "a" "way" "to" "compute" | all unmasked | Done |

**Final output:** "It is a way to compute"

Notice something interesting: the function words ("It", "a", "to") appeared first — they are easy to predict and have high confidence. The content words ("way", "compute") appeared later, after the model had more context to work with. And the word "is" was the last to be committed, because the model kept reconsidering it until it had the full picture.

This is fundamentally different from autoregressive generation. There is no left-to-right constraint. The model fills in whatever it is most confident about first, regardless of position. This is exactly what we want.

There are two main **remasking strategies** that researchers have explored:

1. **Low-confidence remasking:** At each step, keep the most confident predictions and remask everything else. This is the default and works well for most tasks.

2. **Semi-autoregressive remasking:** Generate text block by block from left to right. Within each block, use diffusion. This is a hybrid approach that combines the global planning of diffusion with the sequential coherence of autoregressive models.

---

## LLaDA — Scaling Diffusion to a Real LLM

This brings us to the main character in our story: **LLaDA** (Large Language Diffusion with mAsking).

Published in February 2025 by researchers from Renmin University and the Chinese Academy of Sciences, LLaDA showed for the first time that masked diffusion can be scaled to a full-sized language model and **compete with autoregressive models on standard benchmarks.**

Here is what makes LLaDA remarkable:

- **Architecture:** Vanilla Transformer with bidirectional attention — nothing fancy, just like LLaMA but without the causal mask
- **Training:** Standard pre-training + supervised fine-tuning (SFT) pipeline — the exact same recipe used for GPT and LLaMA
- **Scale:** 8B parameters, trained on 2.3 trillion tokens
- **Results:** Competitive with LLaMA3 8B on benchmarks like MMLU, HumanEval, and GSM8K


![figures/figure_6.png](figures/figure_6.png)
*figures/figure_6.png*


**Caption:** LLaDA 8B achieves competitive performance with LLaMA3 8B across standard NLP benchmarks, demonstrating that diffusion-based language models can match autoregressive models in quality.

But the most fascinating result from LLaDA is not the benchmark scores — it is the **reversal curse** experiment.

### The Reversal Curse: Where Autoregressive Models Fail

Here is a fascinating limitation of autoregressive models. Suppose you train a model on the fact: "The capital of France is Paris." Now ask it: "Paris is the capital of which country?"

You would expect any intelligent system to answer this easily. But autoregressive models struggle with it, because they have only ever modeled the left-to-right conditional probability $P(\text{Paris} \mid \text{The capital of France is})$. They have never been trained on the reverse direction.

The LLaDA authors tested this using Chinese poetry. Given 496 famous poem couplets, they asked models to:
- **Forward task:** Given Line 1, generate Line 2
- **Reversal task:** Given Line 2, generate Line 1


![figures/figure_7.png](figures/figure_7.png)
*figures/figure_7.png*


**Caption:** Autoregressive models suffer from the reversal curse — they can only reason left-to-right. Diffusion models, with their bidirectional attention, handle both directions naturally.

The results were striking. LLaDA achieved **42% accuracy** on the reversal task, while GPT-4o managed only **32%**. On the forward task, both performed comparably.

Why does LLaDA succeed here? Because during training, the masking is applied **uniformly at random** — the model learns to predict any token from any combination of context. There is no inherent "direction." The model naturally builds bidirectional representations, which means it can fill in Line 1 given Line 2 just as easily as the other way around.

---

## The Speed Revolution — Mercury and Beyond

But there is one more advantage that makes diffusion LLMs truly exciting: **speed.**

Let us think about why autoregressive models are slow. To generate a response of $L$ tokens, you need $L$ sequential forward passes through the model. Each pass produces just one token, and the next pass cannot start until the previous one finishes. For a 1,000-token response, that is 1,000 forward passes — one after another.

Diffusion models break this bottleneck. Instead of $L$ sequential steps, you need only $S$ denoising steps, where $S$ is typically between 10 and 50. And here is the key: **each step processes all $L$ tokens in parallel.** So generating a 1,000-token response takes the same 10-50 steps regardless of length.

In February 2025, **Inception Labs** released **Mercury** — the first commercial-scale diffusion LLM. The speed numbers are remarkable:


![figures/figure_8.png](figures/figure_8.png)
*figures/figure_8.png*


**Caption:** Mercury achieves up to 10x faster inference than speed-optimized autoregressive models by generating tokens in parallel.

Mercury Coder Mini achieves **1,109 tokens per second** on an NVIDIA H100 GPU — roughly **10x faster** than speed-optimized autoregressive models of comparable quality.

Why is it so much faster? Let us do the math:

Suppose you are generating a response of $L = 500$ tokens.
- **Autoregressive model:** 500 sequential forward passes. If each takes 5ms, total time = 2.5 seconds.
- **Diffusion model (S = 25 steps):** 25 forward passes, each processing all 500 tokens in parallel. If each takes 20ms (larger because of bidirectional attention), total time = 0.5 seconds.

That is a **5x speedup**, and the advantage grows with longer outputs. For a 2,000-token response, the autoregressive model needs 4x more steps, but the diffusion model still needs only 25. This is exactly what we want.

Google's **Gemini Diffusion** (May 2025) also demonstrated thousands of tokens per second, confirming that this speed advantage is not a fluke but a fundamental property of the diffusion approach.

---

## Practical Implementation — A Toy Diffusion Language Model

Enough theory, let us look at some practical implementation now.

We will build a simple masked diffusion language model in PyTorch. Our toy model will learn to generate short sequences from a simple vocabulary.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration ---
VOCAB_SIZE = 32     # Small vocabulary
SEQ_LEN = 16        # Short sequences
D_MODEL = 128       # Embedding dimension
N_HEADS = 4         # Attention heads
N_LAYERS = 4        # Transformer layers
MASK_TOKEN = 0       # Token ID for [MASK]

# --- Bidirectional Transformer ---
class DiffusionLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.time_embed = nn.Linear(1, D_MODEL)  # Time conditioning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, N_LAYERS)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x_t, t):
        # x_t: (B, L) token IDs with masks, t: (B, 1) masking ratio
        h = self.embed(x_t) + self.time_embed(t).unsqueeze(1)
        h = self.transformer(h)
        return self.head(h)  # (B, L, VOCAB_SIZE)
```

Let us understand this code. The model is a standard Transformer — but with **no causal mask** (bidirectional attention), and with a **time embedding** that tells the model how much of the input is masked. The output is a probability distribution over the vocabulary at every position.

Now, the forward masking and training loop:

```python
def mask_tokens(x_0, t):
    """Apply forward masking process: mask each token with probability t."""
    mask = torch.rand_like(x_0.float()) < t.unsqueeze(1)  # (B, L)
    x_t = x_0.clone()
    x_t[mask] = MASK_TOKEN
    return x_t, mask

# --- Training Loop ---
model = DiffusionLM()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    x_0 = sample_training_batch()  # (B, L) clean token sequences

    # Sample random masking ratio t ~ U(0, 1)
    t = torch.rand(x_0.size(0), 1)  # (B, 1)

    # Apply forward process: mask tokens
    x_t, mask = mask_tokens(x_0, t)

    # Model predicts original tokens at masked positions
    logits = model(x_t, t)  # (B, L, VOCAB_SIZE)

    # Compute loss only at masked positions
    loss = F.cross_entropy(
        logits[mask], x_0[mask], reduction='mean'
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The training is straightforward: sample a random masking ratio, mask the tokens, ask the model to predict the originals, and compute the cross-entropy loss at masked positions. This is the weighted MLM objective we discussed earlier.

Finally, the generation (sampling) process:

```python
@torch.no_grad()
def generate(model, prompt_len=0, total_len=SEQ_LEN, steps=10):
    """Generate a sequence using iterative unmasking."""
    # Start fully masked (except prompt)
    x = torch.full((1, total_len), MASK_TOKEN)

    for s in range(steps, 0, -1):
        t = torch.tensor([[s / steps]])
        logits = model(x, t)
        probs = F.softmax(logits, dim=-1)

        # Sample tokens and compute confidence
        sampled = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(1, -1)
        confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        # Determine which positions are still masked
        is_masked = (x == MASK_TOKEN)

        # Number of tokens to unmask this step
        n_unmask = max(1, int(is_masked.sum() * (1 / s)))

        # Unmask the most confident predictions
        confidence[~is_masked] = float('inf')  # Don't touch already unmasked
        _, top_idx = confidence.topk(n_unmask, dim=-1)
        x.scatter_(1, top_idx, sampled.gather(1, top_idx))

    return x
```

In this generation function, we start with all [MASK] tokens and iteratively unmask the most confident predictions at each step. After $S$ steps, the entire sequence is revealed. Not bad right?

---

## The Road Ahead

We have seen how diffusion LLMs work — from the masked forward process to the iterative unmasking generation, from the elegant MLM-like training objective to real systems like LLaDA and Mercury that rival or outperform autoregressive models.

But we are still in the early days. There are some important challenges:

1. **Compute cost:** Diffusion LLMs currently require roughly **16x more compute** than autoregressive models to match the same perplexity. They are catching up fast, but the gap is real.

2. **Long-form coherence:** While diffusion models excel at parallel generation and bidirectional reasoning, maintaining coherent structure over very long outputs is still an active area of research.

3. **Scaling laws:** We have well-established scaling laws for autoregressive models (Chinchilla, etc.). The scaling behavior of diffusion LLMs is still being mapped out.

But the progress is rapid. In just the past year:
- **LLaDA** showed that diffusion can match autoregressive quality at 8B scale
- **Mercury** demonstrated 10x speed improvements in commercial settings
- **Gemini Diffusion** brought diffusion LLMs to production at Google
- **Dream 7B** pushed the quality frontier further
- **MMaDA** extended the approach to multimodal (text + images) generation

We are at the beginning of a new chapter in language modeling. Remember when image diffusion models first appeared around 2020 — they were interesting but could not match GANs in quality? Within just two years, they completely overtook GANs and became the foundation of Stable Diffusion, DALL-E, and Midjourney.

Does this remind you of something? We might just be at the same inflection point for text generation. The autoregressive typewriter has served us well, but the diffusion painter is warming up.

We will cover the mathematical connections between diffusion LLMs and score-based models in the next article — where the continuous and discrete worlds meet in a beautiful unified framework.

See you next time!

---

## References

- Nie et al., "Large Language Diffusion Models" (2025) — LLaDA
- Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (2024) — MDLM
- Inception Labs, "Mercury: Ultra-Fast Language Models Based on Diffusion" (2025)
- Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces" (2021) — D3PM
- Li et al., "Diffusion-LM Improves Controllable Text Generation" (2022)
- Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (2024) — SEDD
- Ye et al., "Dream 7B: Diffusion Large Language Models" (2025)
