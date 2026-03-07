# Build Qwen3.5 from Scratch: Hybrid Attention, Mixture-of-Experts, and the Future of Local AI

*How three architectural innovations — linear attention, gated memory, and sparse experts — let a 9B model outperform one 13× its size*

---

Let us start with a thought experiment. You download a 5 GB file onto your laptop. You type a question. In under a second, your laptop — with no internet connection, no API key, no cloud server — starts generating a fluent, intelligent response. No data leaves your machine. The model running locally on your hardware just scored higher on a graduate-level science exam than a model 13 times its size running on a billion-dollar data center.

This is not science fiction. This is Qwen3.5.

In February 2026, Alibaba's Qwen team released a family of language models that fundamentally changed what is possible with local, on-device AI. The smallest model fits on a smartphone. The 9-billion parameter version runs comfortably on a laptop and outperforms OpenAI's GPT-OSS-120B on GPQA Diamond — a benchmark designed to stump experts. The flagship 397-billion parameter model activates only 17 billion parameters per token, yet competes with the most powerful proprietary models in the world.

How is this possible? The answer lies in three architectural innovations, each building on decades of research, fused together for the first time in a single model: **hybrid attention** (mixing linear and full attention), **fine-grained Mixture-of-Experts**, and **native multimodal fusion**. In this article, we will build up each of these ideas from first principles, understand the mathematics behind them, and see exactly why they make Qwen3.5 so remarkably efficient.


![The Qwen3.5 efficiency breakthrough — small models outperforming much larger ones](figures/figure_1.png)
*The Qwen3.5 efficiency breakthrough — small models outperforming much larger ones*


---

## The Attention Bottleneck

Before we can understand what Qwen3.5 does differently, we need to understand the problem it solves. Let us start with the mechanism that powers every modern language model: self-attention.

In a standard Transformer, every token in a sequence looks at every other token. If you have a sequence of $N$ tokens, the model computes an $N \times N$ attention matrix where each entry tells you how much token $i$ should attend to token $j$. The formula you have likely seen before is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

Here, $Q$, $K$, and $V$ are matrices of queries, keys, and values, each of shape $N \times d_k$. The matrix $QK^T$ is $N \times N$ — and this is where the problem lives.

Let us plug in some numbers. Suppose you are processing a document with $N = 100{,}000$ tokens (roughly a 300-page book). The attention matrix has $100{,}000 \times 100{,}000 = 10^{10}$ entries. If each entry is a 16-bit float, that is 20 gigabytes of memory — just for one attention layer in one head. A model with 32 layers and 32 heads would need over 20 terabytes of memory for attention alone. This is clearly impractical.

Even with modern tricks like FlashAttention that reduce memory usage, the fundamental computation is still $O(N^2)$ — the number of operations grows quadratically with sequence length. Double the context window, and you quadruple the compute. This quadratic scaling is the single biggest bottleneck preventing language models from efficiently processing very long sequences.

Now the question is: can we do better?


![Quadratic scaling of standard attention — why long contexts are expensive](figures/figure_2.png)
*Quadratic scaling of standard attention — compute and memory grow as N² with sequence length*


---

## Linear Attention: Flipping the Multiplication Order

Here is where the story gets interesting. Let us look at the attention formula again, but this time, think carefully about the order of matrix multiplications.

In standard attention, we first compute $QK^T$ (an $N \times N$ matrix), then multiply by $V$ (an $N \times d_v$ matrix). The bottleneck is that intermediate $N \times N$ matrix. But what if we could avoid creating it altogether?

The key insight behind linear attention is deceptively simple: **change the order of multiplication**.

In standard softmax attention, the softmax function couples all the tokens together — you cannot factor the computation. But if we replace softmax with a simpler kernel function $\phi$, something remarkable happens. Let us denote:

$$
\tilde{Q} = \phi(Q), \quad \tilde{K} = \phi(K)
$$

Then the attention output for a single query $\tilde{q}_i$ becomes:

$$
\text{output}_i = \frac{\sum_{j=1}^{N} \tilde{q}_i^T \tilde{k}_j \cdot v_j}{\sum_{j=1}^{N} \tilde{q}_i^T \tilde{k}_j}
$$

Now here is the trick. Instead of computing $\tilde{q}_i^T \tilde{k}_j$ for every pair $(i, j)$ — which is $O(N^2)$ — we can rearrange:

$$
\text{output}_i = \frac{\tilde{q}_i^T \left(\sum_{j=1}^{N} \tilde{k}_j v_j^T\right)}{\tilde{q}_i^T \left(\sum_{j=1}^{N} \tilde{k}_j\right)}
$$

The term $S = \sum_{j=1}^{N} \tilde{k}_j v_j^T$ is a $d_k \times d_v$ matrix — crucially, it does not depend on $i$ and its size does not grow with $N$. We compute it once by accumulating over all tokens, and then each query simply multiplies against this fixed-size summary. The cost drops from $O(N^2 d_k)$ to $O(N d_k d_v)$ — linear in $N$.

Let us plug in numbers to feel the difference. With $N = 100{,}000$ and $d_k = d_v = 128$:

- **Standard attention:** $100{,}000^2 \times 128 = 1.28 \times 10^{12}$ operations
- **Linear attention:** $100{,}000 \times 128 \times 128 = 1.64 \times 10^{9}$ operations

That is roughly a **780x reduction** in computation. This is exactly what we want.

But there is a catch. We can also view this linear attention as a recurrent model. At each time step $t$, we maintain a hidden state:

$$
S_t = S_{t-1} + \tilde{k}_t v_t^T
$$

This state $S_t$ acts as the model's "memory" — it is a compressed summary of everything the model has seen so far. Each new token adds its information to this memory by writing $\tilde{k}_t v_t^T$ into the state. The output for token $t$ is then simply:

$$
\text{output}_t = \tilde{q}_t^T S_t
$$

This recurrent view reveals both the power and the limitation of linear attention. The power: inference costs $O(1)$ per token, since you just update the state and query it. The limitation: every token writes to the same memory with a simple additive update. Over time, old information gets buried under new writes. The memory has no mechanism to selectively erase or update specific entries — it just keeps accumulating.

You might be thinking: that sounds like a recipe for forgetting. And you would be right.


![Standard attention vs linear attention — the key reordering trick](figures/figure_3.png)
*Standard attention computes the full N×N matrix; linear attention avoids it by reordering multiplications*


---

## The Delta Rule: Error-Correcting Memory

Let us understand the forgetting problem with a concrete example. Imagine a simple key-value memory where you want to store facts:

- Key: "capital of France" → Value: "Paris"
- Key: "capital of Germany" → Value: "Berlin"

In naive linear attention, writing a new fact is purely additive: $S \leftarrow S + k \cdot v^T$. Now suppose you later learn that the capital of France has been updated (hypothetically) to "Lyon." You write the new fact: $S \leftarrow S + k_{\text{France}} \cdot v_{\text{Lyon}}^T$. But the old value "Paris" is still in there! The memory now contains a messy superposition of "Paris" and "Lyon" for the same key. Over many updates, these superpositions accumulate and the memory becomes increasingly noisy. This is the fundamental weakness of naive linear attention — it can only add, never correct.

This brings us to the **delta rule**, an idea from classical machine learning that dates back to the 1960s. The insight is elegant: before writing new information, first compute what the memory *currently* predicts for that key, then write only the *error* — the difference between what you want to store and what is already there.

Mathematically, the update becomes:

$$
S_t = S_{t-1} + k_t \left(v_t - S_{t-1}^T k_t\right)^T
$$

Let us break this apart:

- $S_{t-1}^T k_t$ is what the current memory predicts when queried with key $k_t$
- $v_t - S_{t-1}^T k_t$ is the **error** — the difference between the desired value and the current prediction
- We write this error, scaled by the key, into the memory

Let us plug in some simple numbers. Suppose our memory state is a $2 \times 2$ matrix, and we have already stored one fact:

$$
S_0 = \begin{bmatrix} 0.8 & 0.2 \\ 0.1 & 0.9 \end{bmatrix}
$$

Now a new token arrives with $k_1 = [1, 0]^T$ and $v_1 = [0.5, 0.3]^T$. The memory currently predicts:

$$
S_0^T k_1 = \begin{bmatrix} 0.8 & 0.1 \\ 0.2 & 0.9 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.8 \\ 0.2 \end{bmatrix}
$$

The error is:

$$
v_1 - S_0^T k_1 = \begin{bmatrix} 0.5 \\ 0.3 \end{bmatrix} - \begin{bmatrix} 0.8 \\ 0.2 \end{bmatrix} = \begin{bmatrix} -0.3 \\ 0.1 \end{bmatrix}
$$

The update:

$$
S_1 = S_0 + k_1 (v_1 - S_0^T k_1)^T = \begin{bmatrix} 0.8 & 0.2 \\ 0.1 & 0.9 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} \begin{bmatrix} -0.3 & 0.1 \end{bmatrix}
$$

$$
S_1 = \begin{bmatrix} 0.8 & 0.2 \\ 0.1 & 0.9 \end{bmatrix} + \begin{bmatrix} -0.3 & 0.1 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.3 \\ 0.1 & 0.9 \end{bmatrix}
$$

Now if we query the memory with $k_1 = [1, 0]^T$:

$$
S_1^T k_1 = \begin{bmatrix} 0.5 & 0.1 \\ 0.3 & 0.9 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.3 \end{bmatrix} = v_1
$$

The memory now retrieves exactly the desired value. The old information was surgically corrected, not blindly overwritten. This is the power of the delta rule — it turns a dumb accumulator into an error-correcting associative memory.

The difference is profound. Naive linear attention is like writing on a whiteboard with no eraser — you can only keep adding ink. The delta rule gives you an eraser. You can correct mistakes, update facts, and keep the memory clean.


![The delta rule — error-correcting memory updates](figures/figure_4.png)
*Naive additive updates vs delta rule updates — the delta rule corrects errors instead of accumulating noise*


---

## Gated DeltaNet: The Core Innovation

Now we arrive at the heart of Qwen3.5. The **Gated DeltaNet** is what powers 75% of the layers in the model. It combines the delta rule with an idea borrowed from Mamba — **exponential gating** for selective memory decay.

The question the delta rule alone does not answer is: what if you want to *forget* something entirely? The delta rule can correct values, but the information still persists in the state. Sometimes the model needs to aggressively clear old context — for example, when processing a new paragraph that has nothing to do with the previous one.

This is where gating comes in. The Gated DeltaNet introduces a learnable gate $\alpha_t \in (0, 1)$ that controls how much of the old memory to retain:

$$
S_t = \alpha_t \cdot S_{t-1} + \beta_t \cdot k_t \left(v_t - S_{t-1}^T k_t\right)^T
$$

Here:
- $\alpha_t$ is the **decay gate** — when $\alpha_t$ is close to 0, the model aggressively forgets old memory. When close to 1, it retains everything.
- $\beta_t$ is the **write strength** — how strongly to write the new correction.

Both $\alpha_t$ and $\beta_t$ are computed from the input through learned projections and sigmoid activations:

$$
\alpha_t = \sigma(W_\alpha x_t + b_\alpha), \quad \beta_t = \sigma(W_\beta x_t + b_\beta)
$$

The gating and the delta rule are **complementary**. Gating enables rapid, wholesale erasure ("forget this entire context"), while the delta rule enables precise, surgical correction ("update this specific fact"). Together, they give the model fine-grained control over its memory.

Let us plug in numbers to see how gating works. Suppose $\alpha_t = 0.3$ (the model wants to mostly forget) and $\beta_t = 0.9$ (and write the new information strongly). Starting from:

$$
S_0 = \begin{bmatrix} 0.5 & 0.3 \\ 0.1 & 0.9 \end{bmatrix}, \quad k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_1 = \begin{bmatrix} 0.7 \\ 0.4 \end{bmatrix}
$$

First, decay the old memory: $\alpha_t \cdot S_0 = 0.3 \times S_0 = \begin{bmatrix} 0.15 & 0.09 \\ 0.03 & 0.27 \end{bmatrix}$

Then compute the error: $v_1 - S_0^T k_1 = [0.7, 0.4]^T - [0.5, 0.3]^T = [0.2, 0.1]^T$

The delta update scaled by $\beta_t$: $0.9 \times k_1 [0.2, 0.1]^T = \begin{bmatrix} 0.18 & 0.09 \\ 0 & 0 \end{bmatrix}$

Final state:

$$
S_1 = \begin{bmatrix} 0.15 & 0.09 \\ 0.03 & 0.27 \end{bmatrix} + \begin{bmatrix} 0.18 & 0.09 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0.33 & 0.18 \\ 0.03 & 0.27 \end{bmatrix}
$$

Notice how the old memory has been largely cleared (multiplied by 0.3) and the new information has been written in. The model dynamically decided, based on the input, to perform a near-complete memory reset. In a different context, $\alpha_t$ might be 0.95, preserving most of the memory. This adaptability is what makes Gated DeltaNet so powerful.

The full Gated DeltaNet layer also includes:
- A **short causal convolution** (Conv1D) over recent tokens for local context
- **SiLU activation** (Sigmoid Linear Unit) for smooth gating
- **L2 normalization** on queries and keys for stable training
- **Multi-head structure** — each head maintains its own independent memory state

The key property that makes this efficient is that inference costs $O(1)$ per token. Unlike standard attention, which needs the full KV cache of all previous tokens, Gated DeltaNet only needs its fixed-size state matrix $S_t$. For Qwen3.5 with $d_k = d_v = 128$ and 64 heads, each layer's state is just $64 \times 128 \times 128 \times 2 = 4$ MB — regardless of whether you have processed 1,000 tokens or 1,000,000 tokens.


![The Gated DeltaNet architecture — exponential gating meets error-correcting memory](figures/figure_5.png)
*The Gated DeltaNet layer: Conv1D → SiLU → L2 norm → exponential gating → delta rule update*


---

## Gated Attention: The Global Refresh

You might be thinking: if Gated DeltaNet is so good, why not use it for every layer? Why does Qwen3.5 still use softmax attention at all?

The answer has to do with a fundamental trade-off between **efficiency** and **retrieval precision**.

Gated DeltaNet processes each token in constant time by maintaining a compressed state. But this compression is lossy — the fixed-size state matrix cannot perfectly represent every detail of a very long sequence. Think of it like this: Gated DeltaNet is like reading a book and keeping running notes. Your notes capture the key ideas, but if someone asks you about a specific sentence on page 47, you might not remember it exactly. Your notes are a summary, not a photographic memory.

Standard softmax attention, on the other hand, is like having the entire book open in front of you. You can look up any specific passage instantly. It is perfect for precise retrieval — but it is expensive, because you are literally maintaining the full text.

Qwen3.5 solves this by keeping 25% of its layers as **Gated Attention** layers — standard softmax attention enhanced with a learnable sigmoid gate:

$$
\text{GatedAttn}(Q, K, V) = \sigma(g) \odot \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Here, $\sigma(g)$ is a sigmoid gate that dynamically scales each feature of the attention output. This gate allows the model to selectively suppress or amplify parts of the attention output on a per-feature basis.

The sigmoid gate $g$ is computed as a learned linear projection of the input:

$$
g = W_g x + b_g, \quad \sigma(g) = \frac{1}{1 + e^{-g}}
$$

Let us understand why this gating is useful. In standard attention, every feature of the output is treated equally — whatever the attention weights say, it goes through. But sometimes, certain features of the attended information are irrelevant to the current context. The sigmoid gate gives the model a feature-level "volume knob" — it can turn down features that are noisy and turn up features that are informative.

These Gated Attention layers serve as periodic "global refresh" points. The Gated DeltaNet layers process most of the sequence efficiently with their compressed state, but every fourth layer, a Gated Attention layer performs exact, full-context retrieval. This combination is like a student who takes efficient running notes for most of the lecture but periodically looks back at the textbook to make sure the notes are accurate.


![Gated Attention — softmax attention with a learned sigmoid gate](figures/figure_6.png)
*Gated Attention adds a per-feature sigmoid gate to standard softmax attention*


---

## The 3:1 Hybrid Pattern

Now let us put the two pieces together. Qwen3.5 arranges its layers in a repeating pattern:

$$
\underbrace{\text{GatedDeltaNet}}_{\text{Layer 1}} \to \underbrace{\text{GatedDeltaNet}}_{\text{Layer 2}} \to \underbrace{\text{GatedDeltaNet}}_{\text{Layer 3}} \to \underbrace{\text{GatedAttention}}_{\text{Layer 4}} \to \text{repeat}
$$

Three Gated DeltaNet layers (linear, $O(1)$ per token) followed by one Gated Attention layer (quadratic, but only $\frac{1}{4}$ of the layers). For the flagship 397B model with 60 layers, this means 45 layers of Gated DeltaNet and 15 layers of Gated Attention — arranged as 15 repeating blocks of [DeltaNet, DeltaNet, DeltaNet, Attention].

Why 3:1? The Qwen team experimented with different ratios and found that 3:1 hits a sweet spot. Let us understand the trade-offs:

- **4:0 (all linear):** Maximum speed, but retrieval accuracy degrades on long contexts. The model struggles with tasks like finding a specific number buried deep in a document.
- **1:1 (half and half):** Good retrieval, but gives up too much of the speed advantage.
- **3:1 (three linear, one full):** The linear layers handle most of the computation cheaply. The full attention layers periodically "check the original source" to maintain retrieval precision. The result is near-linear scaling with minimal loss in accuracy.

Let us quantify the speedup. During autoregressive decoding (generating one token at a time), each Gated DeltaNet layer costs $O(d^2)$ regardless of sequence length — it just updates its state and queries it. Each Gated Attention layer costs $O(N \cdot d)$ because it must attend to all $N$ previous tokens. With the 3:1 ratio:

$$
\text{Cost per token} = 3 \times O(d^2) + 1 \times O(N \cdot d) = O(d^2 + N \cdot d)
$$

Compare this to a pure Transformer:

$$
\text{Cost per token} = 4 \times O(N \cdot d) = O(N \cdot d)
$$

For long sequences where $N \gg d$, the hybrid model's cost is dominated by the single attention layer — roughly $4\times$ faster than a pure Transformer with the same number of layers. For the flagship Qwen3.5 model, the Qwen team reported **8.6 to 19 times faster decoding** compared to the previous generation, with the speedup increasing for longer sequences.

This is exactly what we want — efficiency that improves as sequences get longer, which is precisely the regime where it matters most.


![The 3:1 hybrid attention pattern — efficiency meets retrieval precision](figures/figure_7.png)
*The 3:1 hybrid pattern: three cheap linear layers followed by one precise full-attention layer*


---

## Fine-Grained Mixture-of-Experts

So far, we have discussed the attention mechanism — how the model processes relationships between tokens. But after attention, each token passes through a feed-forward network (FFN) that transforms its representation. In a standard Transformer, this FFN is a dense network applied identically to every token. For a large model, this FFN can have billions of parameters, and every single parameter is activated for every single token.

The question is: does every token really need all those parameters?

Think about it this way. When processing the sentence "The Schrödinger equation governs quantum mechanics," the word "Schrödinger" probably needs parameters that understand physics and mathematics. The word "The" probably does not. Yet in a dense model, both words activate exactly the same parameters. This is wasteful.

**Mixture-of-Experts (MoE)** solves this by replacing the single dense FFN with a collection of smaller, specialized "expert" networks. For each token, a router network selects only a few experts to activate. The token passes through just those experts, and the outputs are combined.

Qwen3.5 takes this to an extreme. Instead of 8 large experts (as in Mixtral) or 256 medium experts, Qwen3.5 uses **512 tiny experts** with an intermediate dimension of just 1,024 each. For every token, a router selects **10 experts** to activate, plus **1 shared expert** that is always active:

$$
\text{FFN}(x) = g_s \cdot E_{\text{shared}}(x) + \sum_{i \in \text{TopK}(r(x), 10)} g_i \cdot E_i(x)
$$

where:
- $r(x)$ is the router, a learned linear layer that produces a score for each of the 512 experts
- $\text{TopK}$ selects the 10 experts with the highest scores
- $g_i = \text{softmax}(\text{TopK scores})_i$ are the gating weights
- $E_i(x)$ is the output of expert $i$ applied to token $x$
- $E_{\text{shared}}(x)$ is the shared expert that always fires

Let us plug in numbers. The flagship Qwen3.5-397B-A17B has:

- **Total parameters:** 397 billion
- **Active parameters per token:** ~17 billion (only the 10 routed + 1 shared expert fire)
- **Expert intermediate dimension:** 1,024
- **Number of experts per layer:** 512

This means each forward pass uses only $17/397 \approx 4.3\%$ of the total parameters. The model has the *knowledge capacity* of 397B parameters but the *inference cost* of a 17B model. This is the key reason Qwen3.5 can be so capable while running on practical hardware.

But why 512 tiny experts instead of, say, 16 large ones? The answer is **routing precision**. With 512 experts, each expert can specialize on a very narrow domain — one might handle mathematical notation, another might handle French grammar, another might handle Python syntax. With only 16 experts, each must be a generalist covering a wider range of topics. More experts means sharper specialization and less wasted computation. The small intermediate dimension (1,024) keeps each individual expert lightweight, so the overhead of having 512 of them is manageable.

The **shared expert** serves a crucial role. Some computations — like basic syntactic processing, common word representations, positional adjustments — are needed for virtually every token. Rather than forcing the router to waste one of its 10 expert slots on this common functionality, the shared expert handles it as a guaranteed baseline. The 10 routed experts can then focus entirely on specialized knowledge.

Training a MoE model introduces an additional challenge: **load balancing**. If the router learns to always send tokens to the same few experts, most experts go unused and the model degenerates. Qwen3.5 uses an auxiliary load-balancing loss during training that penalizes uneven expert utilization, ensuring all 512 experts get roughly equal traffic.


![Fine-grained MoE routing — 512 experts, 10 active per token](figures/figure_8.png)
*Fine-grained Mixture-of-Experts: a router selects 10 of 512 tiny specialists plus 1 shared expert*


---

## Putting It All Together

Now let us assemble the complete Qwen3.5 block. Every layer in the model follows this structure:

**Step 1: Hybrid Attention.** The input passes through either a Gated DeltaNet layer or a Gated Attention layer, depending on the position in the 3:1 pattern. This step handles token-to-token interactions — figuring out which parts of the context are relevant to each token.

**Step 2: Add & Norm.** A residual connection adds the attention output back to the input, followed by RMSNorm (Root Mean Square Layer Normalization):

$$
h = \text{RMSNorm}(x + \text{HybridAttn}(x))
$$

**Step 3: Mixture-of-Experts FFN.** The normalized representation passes through the MoE layer, where 10 of 512 experts plus the shared expert process the token. This step transforms the token's representation using specialized knowledge.

**Step 4: Add & Norm.** Another residual connection and RMSNorm:

$$
\text{output} = \text{RMSNorm}(h + \text{MoE}(h))
$$

For the dense models (0.8B through 27B), the MoE layer is replaced with a standard SwiGLU feed-forward network:

$$
\text{SwiGLU}(x) = (\text{SiLU}(x W_1) \odot x W_3) W_2
$$

where $\odot$ is element-wise multiplication, and $W_1$, $W_2$, $W_3$ are learned weight matrices.

The flagship 397B model stacks 60 of these blocks: 15 repeating units of [GatedDeltaNet + MoE, GatedDeltaNet + MoE, GatedDeltaNet + MoE, GatedAttention + MoE]. With a hidden dimension of 4,096 and a vocabulary of 248,320 tokens supporting 201 languages, the full model is a carefully balanced system where every component earns its place.

The context length deserves special mention. Qwen3.5 supports a native context of 262,144 tokens (roughly 800 pages of text), extendable to over 1 million tokens using **YaRN** (Yet Another RoPE Normalization) — a technique that adjusts Rotary Position Embeddings to maintain stable attention patterns at extreme lengths. The hybrid attention design is particularly well-suited for long contexts because the Gated DeltaNet layers handle long-range dependencies in constant memory, while the Gated Attention layers provide periodic high-fidelity snapshots.


![Complete Qwen3.5 block architecture — hybrid attention meets MoE](figures/figure_9.png)
*The complete Qwen3.5 block: Hybrid Attention → Add & Norm → MoE FFN → Add & Norm*


---

## Running It Locally

All of the architectural innovations we have discussed serve a single practical purpose: making powerful AI models run on hardware that ordinary people own.

Here is what the Qwen3.5 model family looks like in practice:

| Model | Total Params | Active Params | Memory (Q4) | Runs On |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | 0.8B (dense) | ~1.6 GB | Smartphones, Raspberry Pi |
| Qwen3.5-2B | 2B | 2B (dense) | ~2.5 GB | Mobile devices |
| Qwen3.5-4B | 4B | 4B (dense) | ~3.5 GB | Laptops |
| Qwen3.5-9B | 9B | 9B (dense) | ~5 GB | Standard laptops |
| Qwen3.5-35B-A3B | 35B | 3B (MoE) | ~20 GB | Single consumer GPU |
| Qwen3.5-122B-A10B | 122B | 10B (MoE) | ~70 GB | Multi-GPU server |
| Qwen3.5-397B-A17B | 397B | 17B (MoE) | ~230 GB | GPU cluster |

**Quantization** is what makes local deployment practical. The models are trained in full precision but can be compressed to 4-bit integers (GGUF Q4_K_M format) with minimal quality loss. A 4-bit quantized Qwen3.5-9B is roughly 5 GB — it fits comfortably in the memory of a laptop with 8 GB of RAM, leaving room for the operating system and other applications.

Frameworks like **Ollama**, **llama.cpp**, and **LM Studio** make local deployment as simple as a single command. The hybrid attention design provides an additional advantage here: the Gated DeltaNet layers do not need a traditional KV cache (they maintain their fixed-size state), so only the Gated Attention layers (25% of the model) contribute to the KV cache. This reduces memory usage during inference and allows longer conversations on limited hardware.

The MoE models are particularly interesting for local deployment. The Qwen3.5-35B-A3B has 35 billion total parameters (requiring more storage) but activates only 3 billion per token (requiring less compute). You need enough memory to hold the full model, but the computation per token is equivalent to running a much smaller model. On a consumer GPU like the RTX 4090 with 24 GB of VRAM, you can run the 35B MoE model in 4-bit quantization with performance that far exceeds what a dense 3B model could deliver.

One more feature that matters enormously for local deployment: **thinking modes**. Qwen3.5 supports both a "thinking" mode (step-by-step chain-of-thought reasoning for complex tasks) and a "non-thinking" mode (fast, direct responses for simple queries). Users can even set **thinking budgets** — specifying how many tokens the model is allowed to spend on reasoning before it must produce an answer. On constrained hardware, this lets you trade reasoning depth for speed exactly when you need to.


---

## Why It Matters

Let us step back and consider what Qwen3.5 represents in the broader arc of AI development.

For years, the narrative in AI has been simple: bigger models are better, and running the best models requires massive cloud infrastructure. Qwen3.5 disrupts this narrative in a fundamental way. It demonstrates that architectural innovation — not just scale — can deliver frontier-class performance.

The numbers tell the story:

- **Qwen3.5-9B** (runs on a laptop) scores **81.7 on GPQA Diamond**, beating GPT-OSS-120B (71.5) — a model with 13.5 times more parameters
- **Qwen3.5-122B-A10B** scores **72.2 on BFCL-V4** (a tool-use benchmark), beating GPT-5 mini (55.5) by 30%
- **Qwen3.5-397B-A17B** achieves **52.5 on Terminal-Bench 2.0** (agentic tasks), up from 22.5 for the previous generation — a 133% improvement

These are not incremental gains. They represent a step change in what is possible with open-weight, locally deployable models.

The implications are profound. A researcher in a developing country with just a laptop can now run models that rival what was only available through expensive API calls. A healthcare startup that cannot send patient data to the cloud for compliance reasons can run state-of-the-art AI entirely on-premises. A student learning about language models can inspect, modify, and experiment with every weight in the model — something impossible with proprietary black boxes.

The three architectural innovations we explored — Gated DeltaNet for efficient linear attention, the 3:1 hybrid pattern for balancing speed and precision, and fine-grained MoE for parameter efficiency — are not isolated tricks. They represent a convergence of research threads spanning decades: from the delta rule of the 1960s, to the attention mechanism of 2017, to the state-space models of 2023, to the mixture-of-experts scaling laws of 2024. Qwen3.5 weaves them together into a single, coherent architecture that pushes the boundaries of what efficient AI looks like.


![Benchmark performance — Qwen3.5 vs frontier models](figures/figure_10.png)
*Qwen3.5 benchmark performance compared to other frontier models*


---

## Key Takeaways

1. **Standard attention scales quadratically** ($O(N^2)$) with sequence length, creating a bottleneck for long contexts.

2. **Linear attention** reformulates the computation to $O(N)$ by reordering matrix multiplications, but suffers from memory degradation.

3. **The delta rule** fixes the memory problem by writing error corrections instead of raw values, enabling precise associative recall.

4. **Gated DeltaNet** combines exponential gating (for forgetting) with the delta rule (for correcting), giving the model fine-grained memory control at $O(1)$ per-token cost.

5. **The 3:1 hybrid pattern** interleaves 75% Gated DeltaNet layers with 25% Gated Attention layers, achieving near-linear scaling while preserving full-attention retrieval precision.

6. **Fine-grained MoE** with 512 tiny experts activates only 10+1 per token, giving the model 397B parameters of knowledge at the inference cost of 17B.

7. **All of this serves local deployment** — the 9B dense model fits on a laptop in 5 GB and outperforms models 13 times its size.

The age of AI being locked behind API calls and cloud servers is ending. Qwen3.5 is what the future of local AI looks like — and now you understand exactly how it works.

That's it!

For the complete hands-on implementation, check out the accompanying Colab notebooks where we build every component from scratch — from linear attention to a full mini-Qwen3.5 you can train and run on your own machine.
