# GPU Memory — The Four Major Components

*Understanding where your GPU memory actually goes: weights, gradients, optimizer states, and activations — the four components that determine whether your model fits.*

---

## The OOM Problem

Let us start with a frustrating but familiar scenario. You just downloaded a 7-billion-parameter model — say LLaMA 2 7B. You check the model size: about 14 GB in half precision. Your GPU has 24 GB of memory. Simple arithmetic says 14 < 24, so it should fit for training, right?

You fire up your training script, set a modest batch size of 4, hit enter — and within seconds:

```
CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 23.65 GiB.
18.32 GiB is already allocated.
```

What happened? The model weights are "only" 14 GB, but your GPU ran out of 24 GB of memory. Where did the other 10+ GB go? And in practice, training a 7B model comfortably requires **60+ GB** of memory — more than four times the weight size.

The answer lies in four components that consume GPU memory during training. Understanding these four components is the single most important thing you need to know before studying any parallelism strategy. Every technique we will cover in this course — data parallelism, tensor parallelism, ZeRO, pipeline parallelism — exists to address the memory footprint of one or more of these four components.

![GPU memory breakdown: weights, gradients, optimizer states, and activations](figures/figure_1.png)
*The four components of GPU memory during training. Model weights are just the tip of the iceberg — gradients, optimizer states, and activations consume the majority of memory.*

Let us understand each one, starting with the most obvious.

---

## Component 1: Model Weights

The weights are the learnable parameters of your neural network — all the matrices in your attention layers, feed-forward layers, embeddings, and layer norms. This is the component most people think about when they estimate memory.

The memory required for weights is straightforward:

$$M_{\text{weights}} = P \times b$$

where $P$ is the number of parameters and $b$ is the number of bytes per parameter.

But what is $b$? That depends on the **numerical precision** you use. This brings us to an important topic: floating-point formats.

### Floating-Point Formats: fp32, fp16, and bf16

A 32-bit floating point number (fp32) uses 4 bytes and provides about 7 decimal digits of precision. For decades, this was the default for all neural network training.

A 16-bit floating point number (fp16) uses 2 bytes — half the memory. But it has only about 3.3 decimal digits of precision and, critically, a much smaller range of representable numbers. Very large or very small values can overflow or underflow.

Brain floating point (bf16), developed by Google Brain, also uses 2 bytes but allocates its bits differently. It has the **same exponent range as fp32** (so it can represent the same magnitude of numbers) but with less precision. For deep learning, where the range of values matters more than exact precision, bf16 is often the better choice.

![Floating point formats: fp32, fp16, and bf16 bit layouts](figures/figure_2.png)
*Bit layout comparison of fp32, fp16, and bf16. Notice that bf16 has the same 8-bit exponent as fp32, giving it the same dynamic range but with reduced precision.*

Let us plug in some numbers. For a 7-billion-parameter model:

| Format | Bytes per Param | Weight Memory |
|--------|----------------|---------------|
| fp32   | 4              | $7 \times 10^9 \times 4 = 28$ GB |
| fp16   | 2              | $7 \times 10^9 \times 2 = 14$ GB |
| bf16   | 2              | $7 \times 10^9 \times 2 = 14$ GB |

So in fp16 or bf16, a 7B model's weights take 14 GB. In fp32, they take 28 GB. This is the most visible component, and it is often the smallest fraction of total training memory.

Let us write a quick helper function:

```python
def weight_memory_gb(num_params, bytes_per_param=2):
    """Calculate weight memory in GB.

    Args:
        num_params: Number of model parameters (e.g., 7e9 for 7B)
        bytes_per_param: 4 for fp32, 2 for fp16/bf16
    """
    return num_params * bytes_per_param / (1024**3)

# Examples
print(f"GPT-2 (124M) in fp16:  {weight_memory_gb(124e6, 2):.2f} GB")
print(f"LLaMA-7B in fp16:      {weight_memory_gb(7e9, 2):.2f} GB")
print(f"LLaMA-7B in fp32:      {weight_memory_gb(7e9, 4):.2f} GB")
print(f"LLaMA-70B in fp16:     {weight_memory_gb(70e9, 2):.2f} GB")
```

Output:
```
GPT-2 (124M) in fp16:  0.23 GB
LLaMA-7B in fp16:      13.04 GB
LLaMA-7B in fp32:      26.08 GB
LLaMA-70B in fp16:     130.39 GB
```

Now the question is: if the weights are 14 GB for a 7B model, where does the rest of the memory go?

---

## Component 2: Gradients

During backpropagation, PyTorch computes a gradient for every trainable parameter. The gradient has **exactly the same shape** as the weight it corresponds to — if a weight matrix is $[4096 \times 4096]$, its gradient is also $[4096 \times 4096]$.

This means:

$$M_{\text{gradients}} = P \times b_g$$

where $b_g$ is the bytes per gradient element. In pure fp32 training, gradients are fp32 (4 bytes). In mixed precision training (which we will discuss shortly), gradients are typically computed in fp16 (2 bytes) during the backward pass.

Let us compute this for our 7B model:

- **fp32 gradients**: $7 \times 10^9 \times 4 = 28$ GB
- **fp16 gradients**: $7 \times 10^9 \times 2 = 14$ GB

So gradients alone double the memory compared to just storing the weights. But we are not done yet — the biggest memory hog is still ahead.

---

## Component 3: Optimizer States

This is where most people are surprised. The optimizer — specifically **Adam**, which is used in nearly all modern LLM training — maintains its own set of internal states that consume a **massive** amount of memory.

Adam tracks two quantities for every parameter:

1. **First moment (m)** — the exponential moving average of the gradient (momentum)
2. **Second moment (v)** — the exponential moving average of the squared gradient (variance)

The update rule for Adam is:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $g_t$ is the gradient at step $t$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\hat{m}_t$, $\hat{v}_t$ are bias-corrected versions.

The critical point is that both $m$ and $v$ have the **same shape as the weights**, and they are **always stored in fp32** for numerical stability — even when the weights and gradients are in fp16. If you stored them in fp16, the small values in $v$ (which involve squaring already-small gradients) would underflow to zero, destroying your training.

![Adam optimizer memory: weights, gradients, first moment, and second moment](figures/figure_3.png)
*The Adam optimizer stores two additional tensors (m and v) for every parameter, both in fp32. This triples or quadruples the per-parameter memory cost compared to storing weights alone.*

So for Adam with fp32 optimizer states:

$$M_{\text{optimizer}} = P \times 4 \times 2 = 8P \text{ bytes}$$

That is 8 bytes per parameter just for the optimizer. For our 7B model:

$$M_{\text{optimizer}} = 7 \times 10^9 \times 8 = 56 \text{ GB}$$

Let us put that in perspective. For a 7B model in fp16:

| Component | Memory |
|-----------|--------|
| Weights (fp16) | 14 GB |
| Gradients (fp16) | 14 GB |
| Optimizer states (fp32) | **56 GB** |
| **Total (so far)** | **84 GB** |

The optimizer states alone are **four times** the size of the weights! And we have not even counted activations yet.

Let us plug in numbers for GPT-2 (124M parameters) to see the same pattern at a smaller scale:

| Component | Memory |
|-----------|--------|
| Weights (fp16) | 0.23 GB |
| Gradients (fp16) | 0.23 GB |
| Optimizer states (fp32) | **0.93 GB** |
| **Total (so far)** | **1.39 GB** |

Even for this tiny model, the optimizer states dominate. This is exactly why techniques like **ZeRO** (which we will study in Pod 5) focus on partitioning optimizer states across GPUs — they are the single largest memory consumer.

---

## Component 4: Activations

Activations are the **intermediate outputs** of each layer during the forward pass. When you run a forward pass through a transformer, every layer produces an output tensor that must be **saved in memory** so that it can be used during backpropagation to compute gradients.

This is a fundamental requirement of the chain rule. To compute the gradient of layer $\ell$, you need the activation from layer $\ell - 1$ (the input to that layer). If you throw away the activation after the forward pass, you cannot compute the gradient.

The memory for activations depends on:
- **Batch size** ($B$)
- **Sequence length** ($S$)
- **Hidden dimension** ($H$)
- **Number of layers** ($L$)

For a transformer model, the activation memory per layer includes the outputs of the attention computation, the intermediate feed-forward expansion (typically $4H$), layer norms, and attention scores. A widely-used approximation from the Megatron-LM paper is:

$$M_{\text{activations}} \approx L \times B \times S \times H \times (34 + 5 \cdot \frac{A \cdot S}{H}) \text{ bytes}$$

where $A$ is the number of attention heads. But for a simpler, more intuitive estimate, we can approximate:

$$M_{\text{activations}} \approx L \times B \times S \times H \times 34 \text{ bytes}$$

when the attention-score term $\frac{A \cdot S}{H}$ is small (which it is for typical hidden sizes).

![Activation memory: intermediate tensors saved during the forward pass](figures/figure_4.png)
*Activations are the intermediate outputs saved at each layer during the forward pass. They must be retained in memory for the backward pass. Memory grows linearly with batch size, sequence length, and number of layers.*

Let us plug in some real numbers for LLaMA-7B ($L = 32$, $H = 4096$, $A = 32$):

With batch size $B = 1$ and sequence length $S = 2048$:

$$M_{\text{activations}} \approx 32 \times 1 \times 2048 \times 4096 \times 34 = 9.15 \text{ GB}$$

With batch size $B = 4$:

$$M_{\text{activations}} \approx 32 \times 4 \times 2048 \times 4096 \times 34 = 36.6 \text{ GB}$$

Now you can see why you got that OOM error! With batch size 4, the activations alone need 36.6 GB. Combined with weights, gradients, and optimizer states (84 GB), that is over 120 GB — well beyond your 24 GB GPU.

The critical observation here is that activation memory **scales linearly with batch size**. Double the batch size, double the activation memory. This is why batch size is the first knob practitioners turn when they hit OOM — reducing batch size is the fastest way to free up memory.

Let us also compute activation memory for GPT-2 (124M) with $L = 12$, $H = 768$, $A = 12$, $B = 8$, $S = 1024$:

$$M_{\text{activations}} \approx 12 \times 8 \times 1024 \times 768 \times 34 = 2.45 \text{ GB}$$

This is manageable on a modern GPU, which is why GPT-2 is a great model for learning and experimentation.

---

## The Complete Memory Formula

Now we have all four pieces. The total GPU memory required for training is:

$$M_{\text{total}} = M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activations}}$$

For mixed precision training with Adam (the most common setup), using $\Psi$ for the number of parameters:

$$M_{\text{total}} = 2\Psi + 2\Psi + 12\Psi + M_{\text{activations}}$$

Wait — why $12\Psi$ for the optimizer instead of $8\Psi$? In mixed precision training, the optimizer also keeps a **master copy of the weights in fp32** (we will explain why shortly). So the optimizer stores:

- fp32 master weights: $4\Psi$ bytes
- fp32 first moment $m$: $4\Psi$ bytes
- fp32 second moment $v$: $4\Psi$ bytes

That gives us $12\Psi$ bytes for the optimizer component. So:

$$M_{\text{total}} = 2\Psi + 2\Psi + 12\Psi + M_{\text{activations}} = 16\Psi + M_{\text{activations}}$$

Let us compute the complete memory budget for two models.

### Worked Example 1: GPT-2 (124M)

- $\Psi = 124 \times 10^6$, $L = 12$, $H = 768$, $A = 12$
- Batch size $B = 8$, Sequence length $S = 1024$

| Component | Formula | Memory |
|-----------|---------|--------|
| Weights (fp16) | $2 \times 124\text{M}$ | 0.23 GB |
| Gradients (fp16) | $2 \times 124\text{M}$ | 0.23 GB |
| Optimizer (fp32 master + m + v) | $12 \times 124\text{M}$ | 1.38 GB |
| Activations | $12 \times 8 \times 1024 \times 768 \times 34$ | 2.45 GB |
| **Total** | | **4.29 GB** |

This fits easily on a single consumer GPU. This is exactly what we want for a learning-oriented setup.

### Worked Example 2: LLaMA-7B

- $\Psi = 7 \times 10^9$, $L = 32$, $H = 4096$, $A = 32$
- Batch size $B = 1$, Sequence length $S = 2048$

| Component | Formula | Memory |
|-----------|---------|--------|
| Weights (fp16) | $2 \times 7\text{B}$ | 13.04 GB |
| Gradients (fp16) | $2 \times 7\text{B}$ | 13.04 GB |
| Optimizer (fp32 master + m + v) | $12 \times 7\text{B}$ | 78.2 GB |
| Activations | $32 \times 1 \times 2048 \times 4096 \times 34$ | 9.15 GB |
| **Total** | | **113.4 GB** |

Even with batch size 1, training LLaMA-7B needs 113 GB — well beyond a single A100 (80 GB). And with a realistic batch size for training? The activations would push this far higher.

Now you understand the OOM error from the beginning. The weights are only a fraction of total memory.

---

## Mixed Precision Training

Now the question is: if fp16 saves half the memory for weights and gradients, why not do everything in fp16?

The problem is that fp16 has very limited precision. When you compute the optimizer update — which involves dividing a small gradient by an even smaller number (the square root of $v$) — the result can lose significant digits or even become zero in fp16. This destroys training.

**Mixed precision training** is the elegant solution. The idea, introduced by Micikevicius et al. (2018), is to use the best of both worlds:

1. Keep a **master copy** of the weights in fp32 (full precision)
2. Before the forward pass, **cast** the weights to fp16 (half precision)
3. Run the forward and backward pass in fp16 (fast, memory-efficient)
4. **Accumulate** the fp16 gradients into the fp32 master weights
5. The optimizer (Adam) updates the fp32 master weights using fp32 arithmetic

![Mixed precision training flow](figures/figure_5.png)
*Mixed precision training: the forward and backward passes use fp16 for speed, while the optimizer maintains fp32 master weights for numerical stability. Gradients are computed in fp16, then cast to fp32 for the optimizer update.*

Why does this work? The forward and backward passes involve large matrix multiplications where fp16 is perfectly adequate — modern GPUs have dedicated fp16/bf16 tensor cores that are 2-8x faster than fp32. But the optimizer update involves tiny increments to large values, where fp32 precision is essential.

The trade-off is clear: we need extra memory for the fp32 master weights (an additional $4\Psi$ bytes), but we save memory on the activations and get a significant speedup from fp16 computation.

Here is a simplified mixed precision training loop in PyTorch:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  # handles fp16 gradient scaling

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in fp16 (autocast handles the casting)
    with autocast(dtype=torch.float16):
        loss = model(batch)

    # Backward pass: scaler scales the loss to prevent
    # fp16 gradient underflow, then computes gradients
    scaler.scale(loss).backward()

    # Optimizer step: scaler unscales gradients back,
    # checks for infs/NaNs, and updates fp32 master weights
    scaler.step(optimizer)
    scaler.update()
```

The `GradScaler` handles an important detail: **loss scaling**. Because fp16 has a limited range, small gradients can underflow to zero. The scaler multiplies the loss by a large factor before the backward pass (so gradients are larger and do not underflow), then divides the gradients back before the optimizer step.

---

## Building a Memory Calculator

Let us put everything together into a practical tool. The following code calculates the full training memory for any model configuration:

```python
def training_memory_gb(
    num_params,        # Total parameters (e.g., 7e9)
    num_layers,        # Number of transformer layers
    hidden_dim,        # Hidden dimension
    num_heads,         # Number of attention heads
    batch_size,        # Training batch size
    seq_len,           # Sequence length
    precision="mixed", # "fp32", "fp16", or "mixed"
):
    """Estimate total GPU memory for training in GB."""

    # --- Weights ---
    if precision == "fp32":
        weight_bytes = num_params * 4
    else:  # fp16, bf16, or mixed
        weight_bytes = num_params * 2

    # --- Gradients ---
    if precision == "fp32":
        grad_bytes = num_params * 4
    else:
        grad_bytes = num_params * 2

    # --- Optimizer (Adam) ---
    # Always fp32: master weights (if mixed) + m + v
    if precision == "mixed":
        opt_bytes = num_params * 12  # fp32 master + m + v
    elif precision == "fp32":
        opt_bytes = num_params * 8   # just m + v (weights already fp32)
    else:
        opt_bytes = num_params * 8   # m + v in fp32

    # --- Activations (simplified Megatron-LM estimate) ---
    act_bytes = num_layers * batch_size * seq_len * hidden_dim * 34

    # Convert to GB
    to_gb = lambda x: x / (1024**3)

    w = to_gb(weight_bytes)
    g = to_gb(grad_bytes)
    o = to_gb(opt_bytes)
    a = to_gb(act_bytes)

    print(f"{'Component':<25} {'Memory':>10}")
    print("-" * 37)
    print(f"{'Weights':<25} {w:>9.2f} GB")
    print(f"{'Gradients':<25} {g:>9.2f} GB")
    print(f"{'Optimizer States':<25} {o:>9.2f} GB")
    print(f"{'Activations':<25} {a:>9.2f} GB")
    print("-" * 37)
    print(f"{'TOTAL':<25} {w+g+o+a:>9.2f} GB")

    return w + g + o + a

# GPT-2 (124M)
print("=== GPT-2 (124M) ===")
training_memory_gb(124e6, 12, 768, 12, 8, 1024, "mixed")
print()

# LLaMA-7B
print("=== LLaMA-7B ===")
training_memory_gb(7e9, 32, 4096, 32, 1, 2048, "mixed")
```

---

## Key Takeaways

Let us summarize the four components and their key properties:

| Component | Scales With | Precision | Typical Share |
|-----------|-------------|-----------|---------------|
| **Weights** | Parameters | fp16/bf16 | Small |
| **Gradients** | Parameters | fp16/bf16 | Small |
| **Optimizer States** | Parameters | fp32 (always) | **Largest** (for small batch) |
| **Activations** | Params $\times$ Batch $\times$ Seq Len | fp16/bf16 | **Largest** (for large batch) |

Two important patterns emerge:

1. **For small batch sizes**, optimizer states dominate. This is why **ZeRO** (partitioning optimizer states across GPUs) is so powerful.

2. **For large batch sizes**, activations dominate. This is why **activation recomputation** (trading compute for memory by recomputing activations during the backward pass instead of storing them) is so powerful.

Understanding these four components gives you a mental framework for every memory optimization technique in deep learning. When someone says "ZeRO-1 partitions optimizer states," you now know exactly which of the four components that addresses and why it has such a large impact. When someone says "gradient checkpointing saves memory," you know it is reducing the activation component by recomputing instead of storing.

---

## What Comes Next

We now know exactly where GPU memory goes during training. The natural follow-up question is: **how do we reduce the memory footprint so the model actually fits?**

In the next pod, we will study three foundational techniques that directly attack the memory and scaling problem:

- **Activation Recomputation** — instead of storing all activations, recompute them during the backward pass (trading compute for memory)
- **Gradient Accumulation** — simulate large batch sizes without the memory cost by accumulating gradients over multiple small forward passes
- **Data Parallelism** — the simplest multi-GPU strategy, which replicates the model on every GPU and splits the data

These three techniques form the foundation upon which all the more advanced parallelism strategies are built.

---

*That's it for the four components of GPU memory! In the notebooks for this pod, you will calculate memory budgets for various models, measure activation memory empirically, and build a "Will It Fit?" calculator that predicts whether a given training configuration will fit on your GPU.*
