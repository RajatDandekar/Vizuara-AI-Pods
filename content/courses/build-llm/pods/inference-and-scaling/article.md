# Inference & Scaling: KV Cache, Sampling, Fine-Tuning & Alignment

You have built the engine. The architecture is designed, the weights are trained, the loss has converged. Now you need to make it drive.

Let us start with a simple analogy. Imagine you have spent months building a Formula 1 car in a garage. The chassis is aerodynamic, the engine is tuned, and every bolt is tightened. But building the car is not the same as racing it. To actually race, you need to figure out how to start the engine efficiently, how to shift gears at the right moment, how to tune the suspension for different tracks, and how to keep the driver safe at 300 km/h. This is the difference between **training** a language model and **running inference** on it. Training gives you the weights. Inference is where the model actually generates text, token by token, in the real world.

In this article, we will cover the full journey from raw autoregressive generation to aligned, efficient, and scaled language models. We will understand the KV cache (how to avoid redundant computation), sampling strategies (how to control the randomness of outputs), fine-tuning and alignment (how to make models actually useful and safe), and scaling laws (how performance improves predictably with more resources).

Let us begin.

---

## 1. Autoregressive Generation: The Sequential Bottleneck

To understand why inference is hard, we first need to understand how a language model generates text.

A model like GPT does not produce an entire paragraph at once. It generates text **one token at a time**. Each token depends on all the tokens that came before it. This is called **autoregressive generation**.

Here is how it works:

1. You give the model a prompt: "The capital of France is"
2. The model processes all the tokens in the prompt and predicts the next token: "Paris"
3. Now the input becomes: "The capital of France is Paris"
4. The model processes this entire sequence again and predicts the next token: "."
5. This continues until the model produces a stop token or reaches a maximum length.


![Autoregressive generation: one token at a time](figures/figure_1.png)
*Autoregressive generation: one token at a time*


Do you see the problem? At step 4, the model is re-processing the tokens "The", "capital", "of", "France", "is" all over again, even though it already processed them in step 2. The computation for those earlier tokens has not changed at all, because the model uses **causal (masked) attention** where each token can only attend to tokens before it.

For a sequence of length $n$, generating $T$ new tokens requires processing roughly $n + 1$, then $n + 2$, then $n + 3$, and so on up to $n + T$ tokens. The total computation scales as:


$$
\text{Total tokens processed} = \sum_{t=1}^{T} (n + t) = nT + \frac{T(T+1)}{2}
$$

Let us plug in some simple numbers. Suppose our prompt has $n = 100$ tokens and we want to generate $T = 50$ new tokens:


$$
\text{Total} = 100 \times 50 + \frac{50 \times 51}{2} = 5000 + 1275 = 6275 \text{ tokens processed}
$$

But we only generated 50 new tokens. We processed 6,275 tokens to get 50 outputs. That is enormously wasteful. The vast majority of computation is redundant, re-processing tokens we have already seen.

This is exactly the problem that the **KV cache** solves.

---

## 2. The KV Cache: Never Compute the Same Thing Twice

Let us think about what happens inside the transformer's attention mechanism during generation.

In self-attention, each token produces three vectors: a **Query (Q)**, a **Key (K)**, and a **Value (V)**. The attention output for a token is computed by comparing its Query against the Keys of all previous tokens, and then using those comparisons to take a weighted sum of the Values.


![KV cache: store Keys and Values, only compute Query for new tokens](figures/figure_2.png)
*KV cache: store Keys and Values, only compute Query for new tokens*


Here is the key insight: when we generate token $t+1$, the Keys and Values for tokens $1$ through $t$ have not changed. Under causal attention, each token's Key and Value depend only on that token and the tokens before it, not on any future tokens. So there is no reason to recompute them.

The **KV cache** stores the Key and Value vectors from all previously processed tokens. When generating the next token, we only need to:

1. Compute the Query, Key, and Value for the **new token only**
2. Append the new Key and Value to the cache
3. Compute attention between the new Query and **all cached Keys**
4. Take the weighted sum of **all cached Values**

This reduces the computation from processing the entire sequence to processing just one token at each step.

### The Math Behind the Savings

Without the KV cache, at generation step $t$, we compute attention over all $t$ tokens. The cost of attention scales as:


$$
\text{Cost without cache (step } t\text{)} = O(t \cdot d)
$$


where $d$ is the model dimension. Over $T$ generation steps, the total attention cost is:


$$
\text{Total cost without cache} = \sum_{t=1}^{T} O(t \cdot d) = O\left(\frac{T^2 \cdot d}{2}\right)
$$


With the KV cache, at each step we only compute one new Q, K, V triplet and perform attention with the cached Keys. The cost per step is:


$$
\text{Cost with cache (step } t\text{)} = O(d) + O(t \cdot d_{\text{head}})
$$

The first term is for computing Q, K, V for the new token. The second term is for the attention dot products with all $t$ cached Keys. The crucial saving is that we no longer recompute the K and V projections for all previous tokens.

Let us work through a concrete numerical example. Suppose we have a model with:
- Model dimension $d = 4096$
- Number of attention heads $h = 32$
- Head dimension $d_{\text{head}} = 128$
- Number of layers $L = 32$
- Sequence length so far: $t = 500$ tokens

**Without KV cache**, at this step we recompute K, V projections for all 500 tokens across all 32 layers. The projection cost alone is:


$$
500 \times 3 \times d^2 \times L = 500 \times 3 \times 4096^2 \times 32 \approx 805 \text{ billion operations}
$$

**With KV cache**, we only compute K, V projections for 1 new token:


$$
1 \times 3 \times d^2 \times L = 1 \times 3 \times 4096^2 \times 32 \approx 1.6 \text{ billion operations}
$$

That is a **500x reduction** in projection computation. This is exactly what we want.

### The Memory Tradeoff

There is no free lunch. The KV cache trades **compute** for **memory**. We need to store all those Key and Value vectors somewhere.

For each layer, we store K and V vectors for every token. The memory cost is:


$$
\text{KV cache memory} = 2 \times L \times t \times d \times \text{bytes per element}
$$

The factor of 2 is for K and V. Let us calculate this for our example model with $t = 2048$ tokens, using 16-bit (2 bytes) floating point:


$$
2 \times 32 \times 2048 \times 4096 \times 2 = 1{,}073{,}741{,}824 \text{ bytes} \approx 1 \text{ GB}
$$

For a single sequence, the KV cache needs about 1 GB of GPU memory. Now think about what happens in production. If you are serving 32 users simultaneously, that is 32 GB just for the cache alone. And this is for a sequence length of only 2,048 tokens. Modern models like GPT-4 and Claude support context windows of 100,000 tokens or more. Let us recalculate for a 100K context window:

$$
2 \times 32 \times 100{,}000 \times 4096 \times 2 = 52{,}428{,}800{,}000 \text{ bytes} \approx 52 \text{ GB}
$$

A single user with a 100K context window would consume 52 GB of GPU memory just for the KV cache. This is why techniques like **GQA (Grouped Query Attention)**, which reduces the number of Key-Value heads, and **KV cache quantization**, which stores cached values in lower precision (say 8-bit instead of 16-bit, halving the memory), have become essential. With 8-bit quantization, our 100K example drops from 52 GB to about 26 GB, which is far more manageable.

This is why **KV cache memory management** is one of the most important problems in large-scale LLM serving.


![KV cache: trading massive compute savings for moderate memory cost](figures/figure_3.png)
*KV cache: trading massive compute savings for moderate memory cost*


---

## 3. Sampling Strategies: Controlling the Output

Once we have efficient generation with the KV cache, the next question is: **how do we actually pick the next token?**

At each step, the model outputs a vector of **logits**, one value for each token in the vocabulary. These logits are converted into a probability distribution using the softmax function. But having a probability distribution over 50,000+ tokens does not tell us which single token to pick. That is where **sampling strategies** come in.

### Greedy Decoding

The simplest approach is **greedy decoding**: always pick the token with the highest probability.


$$
y_t = \arg\max_{v \in V} P(v \mid y_{<t})
$$


For example, suppose at a certain step our model gives the following probabilities for the top 5 tokens:

| Token | Probability |
|-------|-------------|
| "the" | 0.35 |
| "a" | 0.25 |
| "this" | 0.15 |
| "my" | 0.10 |
| "that" | 0.08 |

Greedy decoding always picks "the" (probability 0.35).

The problem? Greedy decoding is **deterministic** and often **repetitive**. It tends to produce safe, boring text. The model gets stuck in loops, repeating the same phrases over and over. You might get: "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."

We need some randomness.

### Temperature Scaling

**Temperature** is a simple but powerful knob that controls how "sharp" or "flat" the probability distribution is. Before applying softmax, we divide the logits by a temperature parameter $\tau$:


$$
P(v \mid y_{<t}) = \frac{\exp(z_v / \tau)}{\sum_{v' \in V} \exp(z_{v'} / \tau)}
$$

where $z_v$ is the logit for token $v$.

Let us see how temperature changes the distribution. Suppose our raw logits for three tokens are $z_A = 2.0$, $z_B = 1.0$, $z_C = 0.5$.

**At temperature $\tau = 1.0$ (default):**


$$
P(A) = \frac{e^{2.0}}{e^{2.0} + e^{1.0} + e^{0.5}} = \frac{7.39}{7.39 + 2.72 + 1.65} = \frac{7.39}{11.76} = 0.628
$$


$$
P(B) = \frac{2.72}{11.76} = 0.231, \quad P(C) = \frac{1.65}{11.76} = 0.140
$$


**At temperature $\tau = 0.5$ (sharper, more confident):**

We divide logits by 0.5, so they become $4.0, 2.0, 1.0$:


$$
P(A) = \frac{e^{4.0}}{e^{4.0} + e^{2.0} + e^{1.0}} = \frac{54.60}{54.60 + 7.39 + 2.72} = \frac{54.60}{64.71} = 0.844
$$


$$
P(B) = \frac{7.39}{64.71} = 0.114, \quad P(C) = \frac{2.72}{64.71} = 0.042
$$


**At temperature $\tau = 2.0$ (flatter, more random):**

We divide logits by 2.0, so they become $1.0, 0.5, 0.25$:


$$
P(A) = \frac{e^{1.0}}{e^{1.0} + e^{0.5} + e^{0.25}} = \frac{2.72}{2.72 + 1.65 + 1.28} = \frac{2.72}{5.65} = 0.481
$$


$$
P(B) = \frac{1.65}{5.65} = 0.292, \quad P(C) = \frac{1.28}{5.65} = 0.227
$$


Notice the pattern: low temperature concentrates probability on the top token (0.844 for token A at $\tau = 0.5$), while high temperature spreads probability more evenly (0.481 for token A at $\tau = 2.0$). At $\tau \to 0$, temperature sampling becomes greedy decoding. At $\tau \to \infty$, it becomes uniform random sampling.

Think of temperature like a volume knob on a stereo. At low volume (low temperature), only the loudest instrument (the top token) is really audible. At high volume (high temperature), you hear all the instruments roughly equally, including the quiet background ones. For code generation, you typically want $\tau \approx 0.2$ because correctness matters and there is usually one right answer. For creative writing, $\tau \approx 0.8$ to $1.0$ gives you more variety and surprise. For brainstorming, you might push it even higher to $\tau \approx 1.2$.


![How temperature reshapes the probability distribution](figures/figure_4.png)
*How temperature reshapes the probability distribution*


### Top-k Sampling

Even with temperature, sampling from the full vocabulary can produce nonsensical tokens. If the vocabulary has 50,000 tokens, even a very low probability token might occasionally get selected.

**Top-k sampling** restricts the selection to the $k$ most probable tokens, then renormalizes the probabilities.

For example, with $k = 3$ and our original distribution:

| Token | Original Prob | After Top-3 (renormalized) |
|-------|--------------|---------------------------|
| "the" | 0.35 | 0.35 / 0.75 = 0.467 |
| "a" | 0.25 | 0.25 / 0.75 = 0.333 |
| "this" | 0.15 | 0.15 / 0.75 = 0.200 |
| "my" | 0.10 | 0 |
| "that" | 0.08 | 0 |

The sum of the top 3 probabilities is $0.35 + 0.25 + 0.15 = 0.75$, so we divide each by 0.75 to renormalize.

The problem with top-k is that $k$ is fixed. Sometimes the model is very confident and the top 2 tokens hold 95% of the probability, so we should sample from only 2 tokens. Other times the model is uncertain and 20 tokens each have about 5% probability, so we should sample from 20. A fixed $k$ cannot adapt to this.

### Top-p (Nucleus) Sampling

**Top-p sampling** (also called nucleus sampling) solves this adaptivity problem. Instead of fixing the number of tokens, we fix the cumulative probability mass $p$ and include the smallest set of tokens whose probabilities add up to at least $p$.

For example, with $p = 0.80$:

| Token | Probability | Cumulative | Included? |
|-------|-------------|------------|-----------|
| "the" | 0.35 | 0.35 | Yes |
| "a" | 0.25 | 0.60 | Yes |
| "this" | 0.15 | 0.75 | Yes |
| "my" | 0.10 | 0.85 | Yes (crosses 0.80 threshold) |
| "that" | 0.08 | 0.93 | No |

We include tokens until we first exceed $p = 0.80$. This gives us 4 tokens. After renormalization (dividing each by $0.35 + 0.25 + 0.15 + 0.10 = 0.85$):


$$
P(\text{"the"}) = 0.412, \quad P(\text{"a"}) = 0.294, \quad P(\text{"this"}) = 0.176, \quad P(\text{"my"}) = 0.118
$$


The beauty of top-p is that it adapts automatically. When the model is confident, the nucleus is small (maybe 2-3 tokens). When the model is uncertain, the nucleus expands to include many more options. This is exactly what we want.

### When to Use What: A Practical Guide

In practice, you almost never use just one of these strategies. Most inference APIs let you combine temperature with either top-k or top-p (or both). Let us walk through a concrete scenario to see how they interact.

Suppose we have five tokens with raw logits $z = [3.0, 2.0, 1.5, 0.5, 0.1]$. We apply temperature $\tau = 0.7$ first, giving us scaled logits $z / 0.7 = [4.29, 2.86, 2.14, 0.71, 0.14]$. After softmax, the probabilities become approximately:

$$
P = [0.52, 0.13, 0.06, 0.015, 0.009]
$$

Now we apply top-p with $p = 0.90$. Token A alone has 0.52, tokens A+B give 0.65, tokens A+B+C give 0.71, tokens A+B+C+D give 0.725... we keep going until we cross 0.90. In this case, the temperature has already sharpened the distribution so much that we only need the top 2-3 tokens. The low temperature did most of the heavy lifting.

Now consider the same logits but with $\tau = 1.5$: scaled logits become $[2.0, 1.33, 1.0, 0.33, 0.07]$, and after softmax the probabilities flatten to roughly $P = [0.35, 0.18, 0.13, 0.07, 0.05]$. With $p = 0.90$, we now include all five tokens. The high temperature spread things out, and top-p adapts by widening the nucleus.

This is why the combination works so well: temperature controls the overall sharpness, and top-p provides a safety net that adapts to whatever distribution the temperature produces.


![Top-k uses fixed count; Top-p adapts to model confidence](figures/figure_5.png)
*Top-k uses fixed count; Top-p adapts to model confidence*


---

## 4. Putting It Together: A Generation Loop

Now that we understand the KV cache and sampling strategies, let us put them together into a complete generation loop. The following pseudocode shows how modern LLM inference works:

```python
import torch
import torch.nn.functional as F

def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """Generate text with KV cache and top-p sampling."""
    tokens = prompt_tokens.clone()
    kv_cache = None  # Will be populated on first forward pass

    for _ in range(max_new_tokens):
        # Only feed the last token if cache exists, else feed full sequence
        if kv_cache is not None:
            input_tokens = tokens[:, -1:]  # Only the newest token
        else:
            input_tokens = tokens  # Full prompt on first pass

        # Forward pass: get logits and updated cache
        logits, kv_cache = model(input_tokens, kv_cache=kv_cache)
        logits = logits[:, -1, :]  # Logits for the last position

        # Apply temperature scaling
        logits = logits / temperature

        # Apply top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float('-inf')

        # Sample from the filtered distribution
        probs = F.softmax(sorted_logits, dim=-1)
        next_token_index = torch.multinomial(probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_token_index)

        tokens = torch.cat([tokens, next_token], dim=-1)

        # Stop if we hit the end-of-sequence token
        if next_token.item() == model.eos_token_id:
            break

    return tokens
```

Let us understand this code in detail:

- **Lines 9-12**: On the first pass, we feed the entire prompt into the model. On every subsequent pass, we feed only the single newest token, since the KV cache already stores the representations for all previous tokens.
- **Lines 15-16**: The model returns logits for the next token and an updated KV cache.
- **Line 19**: We divide logits by the temperature to control randomness.
- **Lines 22-25**: We sort tokens by probability, compute cumulative probabilities, and mask out everything beyond the top-p threshold.
- **Lines 28-29**: We sample one token from the remaining distribution.

This is the core loop behind every chatbot, code assistant, and text generator you have used.

---

## 5. Fine-Tuning: Making a General Model Specific

A pretrained language model is trained on a broad corpus of internet text. It learns grammar, facts, reasoning patterns, and even some code. But it is trained with a single objective: **predict the next token**. This does not directly translate to following instructions, answering questions accurately, or writing in a particular style.

Think of it this way. A medical student who has read every textbook knows a lot of medicine, but they still need residency training to actually treat patients. The knowledge is there, but the skill of applying it in the right context needs to be developed. This is what **fine-tuning** does for language models.

### Full Fine-Tuning

The most straightforward approach is **full fine-tuning**: take the pretrained model and continue training all of its parameters on a task-specific dataset. For example, you might fine-tune on a dataset of (instruction, response) pairs to create a chatbot.

The problem? Modern LLMs have billions of parameters. GPT-3 has 175 billion. LLaMA 70B has 70 billion. Fine-tuning all of these parameters requires:
- Storing a full copy of the model gradients (same size as the model)
- Storing optimizer states (2x model size for Adam)
- A large amount of high-quality task-specific data to avoid overfitting

For a 70B parameter model in 16-bit precision, the model alone takes about 140 GB. With gradients and optimizer states, you need roughly **560 GB** of GPU memory for full fine-tuning. That is 7 A100 GPUs (80 GB each) just for the memory, before you even start thinking about batch size.

This brings us to a more practical approach.

### LoRA: Low-Rank Adaptation

**LoRA** (Low-Rank Adaptation of Large Language Models) is based on a beautiful insight: when you fine-tune a large model, the weight updates tend to be **low-rank**. This means the updates can be well-approximated by much smaller matrices.

Instead of updating the full weight matrix $W \in \mathbb{R}^{d \times d}$, LoRA freezes the original weights and adds a low-rank decomposition:


$$
W' = W + \Delta W = W + BA
$$


where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$, and $r \ll d$ is the rank.


![LoRA: freeze the big matrix, train two small ones](figures/figure_6.png)
*LoRA: freeze the big matrix, train two small ones*


Let us see why this saves so much. For $d = 4096$ and rank $r = 16$:

**Full weight matrix** parameters: $d \times d = 4096 \times 4096 = 16,777,216$ parameters

**LoRA adapter** parameters: $d \times r + r \times d = 4096 \times 16 + 16 \times 4096 = 131,072$ parameters


$$
\text{Reduction factor} = \frac{16{,}777{,}216}{131{,}072} = 128\times
$$


We only need to train **0.78%** of the parameters. This is exactly what we want. A 70B parameter model might only need 50-100 million trainable LoRA parameters, bringing fine-tuning within reach of a single GPU.

Let us plug in some numbers for a full model. A 7B parameter model like LLaMA-7B has about 32 attention layers, each with Q, K, V, and output projections of dimension 4096. If we apply LoRA with rank $r = 16$ to just the Q and V projections:


$$
\text{LoRA parameters} = 32 \text{ layers} \times 2 \text{ projections} \times 2 \times 4096 \times 16 = 8{,}388{,}608 \approx 8.4\text{M parameters}
$$

That is 8.4 million trainable parameters out of 7 billion total, which is just **0.12%** of the model. And remarkably, this is often enough to match or come close to full fine-tuning performance.

Now the question is: how do you choose the rank $r$? Think of it like choosing the resolution of a photograph. A very low rank ($r = 4$) is like a thumbnail image: it captures the broad strokes but misses fine details. A higher rank ($r = 64$ or $r = 128$) captures more nuance but requires more parameters and memory. In practice, $r = 8$ to $r = 32$ works well for most tasks. Let us compare the parameter counts for a single weight matrix with $d = 4096$:

| Rank $r$ | LoRA Parameters | % of Full Matrix | Typical Use Case |
|-----------|----------------|-------------------|------------------|
| 4 | 32,768 | 0.20% | Simple style transfer |
| 16 | 131,072 | 0.78% | General instruction tuning |
| 64 | 524,288 | 3.13% | Complex domain adaptation |
| 128 | 1,048,576 | 6.25% | Approaching full fine-tuning quality |

The sweet spot for most practitioners is $r = 16$: it gives enough capacity to learn new behaviors while keeping memory usage tiny. A full fine-tune of LLaMA-7B requires roughly 140 GB of GPU memory (model + gradients + optimizer states). With LoRA at $r = 16$, you need only about 16 GB total: 14 GB for the frozen model weights (loaded in inference mode) plus roughly 2 GB for the LoRA adapters and their optimizer states. That is the difference between needing a cluster of GPUs and needing a single consumer GPU.

---

## 6. Alignment: From Next-Token Prediction to Being Helpful

Fine-tuning gets us a model that can follow instructions. But there is a deeper problem. A model trained purely on next-token prediction is optimizing for a different objective than what we actually want.

We want the model to be **helpful**, **harmless**, and **honest**. The pretraining objective says nothing about these qualities. A model that perfectly predicts the next token on internet text will happily generate toxic content, confidently state falsehoods, or comply with dangerous requests, because all of those patterns exist in its training data.

This gap between "predicting text well" and "being a good assistant" is the **alignment problem**. Let us look at the two most important approaches to solving it.

### RLHF: Reinforcement Learning from Human Feedback

**RLHF** is the technique that made ChatGPT feel so different from plain GPT-3. It works in three stages:

**Stage 1: Supervised Fine-Tuning (SFT)**

Take the pretrained model and fine-tune it on high-quality (prompt, response) pairs written by humans. This gives us a model that can follow instructions, but it may not know which of several possible responses is best.

**Stage 2: Reward Model Training**

Collect pairs of model responses to the same prompt and have humans rank them. "Response A is better than Response B." Use these comparisons to train a separate **reward model** $r_\theta(x, y)$ that takes a prompt $x$ and response $y$ and outputs a scalar score.

The reward model is trained with the following loss function:


$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]
$$

where $y_w$ is the preferred (winning) response and $y_l$ is the rejected (losing) response.

Let us work through a numerical example to see why this loss makes sense. Suppose for a given prompt, the reward model assigns:
- Score for preferred response: $r_\theta(x, y_w) = 3.0$
- Score for rejected response: $r_\theta(x, y_l) = 1.0$

The difference is $3.0 - 1.0 = 2.0$. Then:


$$
\sigma(2.0) = \frac{1}{1 + e^{-2.0}} = \frac{1}{1 + 0.135} = 0.881
$$


$$
\mathcal{L} = -\log(0.881) = 0.127
$$


The loss is small (0.127) because the reward model correctly ranks the preferred response higher. This is exactly what we want.

Now suppose the reward model gets it wrong: $r_\theta(x, y_w) = 1.0$ and $r_\theta(x, y_l) = 3.0$. The difference is $1.0 - 3.0 = -2.0$:


$$
\sigma(-2.0) = \frac{1}{1 + e^{2.0}} = \frac{1}{1 + 7.39} = 0.119
$$


$$
\mathcal{L} = -\log(0.119) = 2.128
$$


The loss is much higher (2.128) because the reward model ranked incorrectly. The gradient will push the model to fix this.

**Stage 3: RL Fine-Tuning with PPO**

Use the reward model to provide rewards, and optimize the language model's policy using **Proximal Policy Optimization (PPO)**. The objective is:


$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_{y \sim \pi_\theta} \left[ r_\theta(x, y) - \beta \cdot D_{\text{KL}}\left(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)\right) \right]
$$

The KL divergence term $D_{\text{KL}}$ prevents the model from deviating too far from the pretrained model. Think of it as a leash: the model can explore new behaviors to maximize the reward, but the leash keeps it from wandering so far that it forgets how to write coherent English. The $\beta$ parameter controls the length of this leash. A small $\beta$ gives the model more freedom; a large $\beta$ keeps it close to the original.

Without this constraint, the model might find degenerate ways to maximize the reward (reward hacking), like repeating "thank you" endlessly to get high scores from a sentiment-based reward model.


![The three stages of RLHF: SFT, Reward Model, PPO](figures/figure_7.png)
*The three stages of RLHF: SFT, Reward Model, PPO*


### DPO: A Simpler Alternative

RLHF works, but it is complex. You need to train a separate reward model, run PPO (which is notoriously unstable), and manage multiple models simultaneously. In 2023, **Direct Preference Optimization (DPO)** showed that you can skip the reward model and RL entirely.

DPO starts from the same preference data (pairs of responses ranked by humans), but it directly optimizes the language model with a simple classification-style loss:


$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right) \right]
$$

The idea is elegant: we want the model to increase the probability of the preferred response $y_w$ relative to the reference model, and decrease the probability of the rejected response $y_l$. The $\beta$ parameter controls how strongly we enforce these preferences.

Let us work through a simple numerical example. Suppose $\beta = 0.1$ and:
- $\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} = 2.0$ (model increased probability of preferred response)
- $\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} = -1.0$ (model decreased probability of rejected response)


$$
\beta \left( 2.0 - (-1.0) \right) = 0.1 \times 3.0 = 0.3
$$



$$
\sigma(0.3) = \frac{1}{1 + e^{-0.3}} = \frac{1}{1 + 0.741} = 0.574
$$


$$
\mathcal{L}_{\text{DPO}} = -\log(0.574) = 0.555
$$

The loss is moderate, pushing the model to further separate the preferred and rejected responses. As training progresses and the model gets better at distinguishing good from bad responses, this loss will decrease.

DPO has become extremely popular because it is simple to implement (just a single supervised training loop), stable to train (no RL instability), and requires only one model (no separate reward model).

### Fine-Tuning vs. RLHF vs. DPO: When to Use What

Now that we have seen all three approaches, let us compare them directly. This is a question that comes up constantly in practice: when should you use supervised fine-tuning alone, when do you need RLHF, and when is DPO the right choice?

**Supervised fine-tuning (SFT)** is the right starting point when you have high-quality demonstration data and a well-defined task. If you want a model that writes SQL queries from natural language, and you have 10,000 (question, SQL query) pairs, SFT is straightforward and effective. The limitation is that SFT only teaches the model to imitate. It does not teach it to distinguish between good and bad responses when there are multiple plausible outputs.

**RLHF** is the heavy machinery. It shines when output quality is subjective and hard to capture in a simple loss. "Is this response helpful?" is not a question you can answer with cross-entropy. RLHF excels because the reward model learns subjective preferences from human judgments, and PPO optimizes for them. The downside is complexity: a separate reward model, notoriously unstable PPO training, and 3-4x more GPU memory since you run multiple models simultaneously.

**DPO** occupies the pragmatic middle ground. It captures preference information without the complexity of a reward model or RL. In practice, DPO achieves results comparable to RLHF on most benchmarks while being significantly simpler. If you have preference data and want alignment without the engineering overhead, DPO is often the best choice.

A practical rule of thumb: start with SFT to get a solid baseline, then apply DPO with preference data to refine quality. Reserve full RLHF for cases where you need maximum control over the reward signal or when your preference structure is too complex for DPO to capture.

---

## 7. Scaling Laws: The Predictable Science of Bigger Models

So far we have discussed how to generate text efficiently, how to sample well, and how to align models. But there is a more fundamental question: **how good will the model be in the first place?**

It turns out that the performance of language models follows remarkably predictable **scaling laws**. The loss decreases as a power law with respect to three factors: the number of parameters $N$, the amount of training data $D$, and the total compute budget $C$.

### The Kaplan Scaling Laws

The original scaling laws paper (Kaplan et al., 2020) found:


$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

where $L$ is the test loss, and $N_c, D_c, C_c, \alpha_N, \alpha_D, \alpha_C$ are empirically determined constants. The exponents are roughly $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, and $\alpha_C \approx 0.050$.

Let us plug in some numbers to see what this means. Suppose the current loss with $N = 1\text{B}$ parameters is $L = 3.0$. If we scale to $N = 10\text{B}$ parameters (10x increase):


$$
\frac{L(10B)}{L(1B)} = \left(\frac{1B}{10B}\right)^{0.076} = (0.1)^{0.076} = 10^{-0.076} = 0.839
$$



$$
L(10B) = 3.0 \times 0.839 = 2.52
$$


A 10x increase in model size reduces the loss from 3.0 to 2.52, a 16% reduction. This might not sound like much, but on language modeling benchmarks, this can translate to dramatically improved capabilities in reasoning, instruction following, and knowledge recall.


![Scaling laws: loss decreases predictably as a power law](figures/figure_8.png)
*Scaling laws: loss decreases predictably as a power law*


### Chinchilla: Compute-Optimal Training

The Kaplan scaling laws had a surprising implication: it seemed like you should always make models bigger, even if it means training on less data. Many labs followed this, creating very large models trained on relatively little data (GPT-3 with 175B parameters was trained on only 300B tokens).

In 2022, the **Chinchilla paper** (Hoffmann et al., "Training Compute-Optimal Large Language Models") challenged this. They found that for a fixed compute budget, the optimal allocation between parameters and data is roughly **equal scaling**: if you double your compute, you should double both the model size and the training data.

The Chinchilla rule of thumb is:


$$
D_{\text{optimal}} \approx 20 \times N
$$

That is, the number of training tokens should be roughly 20 times the number of parameters.

Let us see what this means for some well-known models:

| Model | Parameters $N$ | Chinchilla-optimal tokens $20N$ | Actual training tokens | Status |
|-------|---------------|-------------------------------|----------------------|--------|
| GPT-3 | 175B | 3.5T | 300B | Severely undertrained |
| Chinchilla | 70B | 1.4T | 1.4T | Compute-optimal |
| LLaMA | 65B | 1.3T | 1.4T | Roughly optimal |
| LLaMA-2 | 70B | 1.4T | 2.0T | Slightly overtrained |

The stunning result was that Chinchilla (70B parameters, 1.4T tokens) outperformed the much larger Gopher (280B parameters, 300B tokens) on nearly every benchmark. A model with 4x fewer parameters beat a model 4x its size, simply by training on more data. This completely changed how the industry thought about scaling.


![Chinchilla optimal: balance parameters and data for best efficiency](figures/figure_9.png)
*Chinchilla optimal: balance parameters and data for best efficiency*


### Beyond Chinchilla

More recent work has pushed beyond the Chinchilla ratio. Models like LLaMA-3 are trained on significantly more tokens than the 20N rule suggests (LLaMA-3 8B was trained on 15T tokens, nearly 2000x the parameter count). The reasoning is practical: once you have spent the compute to train the model, a smaller but more thoroughly trained model is cheaper to serve at inference time. Since inference costs dominate in production, "overtrained" small models can be more economical overall.

This highlights an important point: scaling laws tell you how to minimize training loss for a given compute budget, but the real-world optimization includes inference costs, deployment constraints, and latency requirements.

---

## 8. Conclusion

Let us step back and see how far we have come.

We started with the basic question of how a language model generates text: one token at a time, in a sequential loop that can be painfully slow. The **KV cache** eliminated the redundant computation by storing and reusing the Key and Value vectors from previous tokens, trading a modest amount of memory for massive compute savings.

We then explored **sampling strategies** to control the quality and diversity of generated text. Greedy decoding is deterministic and repetitive. Temperature scaling gives us a knob to control randomness. Top-k sampling cuts off the long tail, and top-p (nucleus) sampling adapts dynamically to the model's confidence. In practice, top-p sampling with a moderate temperature is the standard approach.

With efficient generation solved, we turned to **fine-tuning** to make general models useful for specific tasks. Full fine-tuning is effective but expensive. LoRA provides a brilliant shortcut by decomposing weight updates into low-rank matrices, reducing trainable parameters by 100x or more while preserving most of the performance.

**Alignment** tackles the fundamental gap between "predicting text" and "being helpful." RLHF uses human preferences to train a reward model, then optimizes the language model with reinforcement learning. DPO simplifies this into a single supervised training step that directly optimizes the preference objective.

Finally, **scaling laws** revealed that language model performance improves predictably with more parameters, data, and compute. The Chinchilla paper taught us that the key is balance: allocating compute equally between model size and training data yields the best results.

From raw autoregressive generation to aligned, efficient, scaled systems, these techniques together have transformed language models from research curiosities into the most widely-deployed AI systems in history. Every time you interact with ChatGPT, Claude, or any other AI assistant, all of these mechanisms are working together behind the scenes: KV caches keeping inference fast, sampling strategies keeping outputs diverse, alignment keeping responses helpful, and scaling laws ensuring the model is powerful enough to be useful in the first place.

That's it!

---

**References:**

1. Vaswani et al., "Attention Is All You Need" (2017) - The original transformer paper
2. Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020) - Nucleus (top-p) sampling
3. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) - LoRA
4. Ouyang et al., "Training language models to follow instructions with human feedback" (2022) - InstructGPT/RLHF
5. Rafailov et al., "Direct Preference Optimization" (2023) - DPO
6. Kaplan et al., "Scaling Laws for Neural Language Models" (2020) - Scaling laws
7. Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) - Chinchilla scaling laws
