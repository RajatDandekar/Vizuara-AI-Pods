# Activation Recomputation, Gradient Accumulation, and Data Parallelism

*Three foundational techniques that reduce memory and scale training: recomputing activations, accumulating gradients, and replicating your model across GPUs.*

---

## (1) The Memory Wall

Let us start with a concrete scenario. You have a 1.3 billion parameter transformer model that you want to train. You fire up your NVIDIA A100 with 40 GB of memory, load the model weights in mixed precision, and begin training. Within seconds, you hit an `OutOfMemoryError`.

Where did all the memory go? If you recall, GPU memory during training is consumed by four things: **model weights**, **gradients**, **optimizer states**, and **activations**. For a 1.3B parameter model in mixed precision, the weights alone need about 2.6 GB. Gradients take another 2.6 GB. Adam optimizer states add roughly 10.4 GB. That totals about 15.6 GB — well within your 40 GB budget.

So what fills the remaining 24.4 GB? **Activations.** The intermediate outputs of every layer that we must store during the forward pass so that we can compute gradients during the backward pass. For large batch sizes or long sequences, activation memory can easily reach 20-30 GB or more, pushing us over the limit.

This is what practitioners call the **memory wall**. Your model fits, your optimizer fits, but the activations during training blow up the memory budget.

In this article, we will study three foundational techniques that form the first line of defense against this memory wall:

1. **Activation Recomputation (Gradient Checkpointing)** — trade compute for memory by discarding and recomputing activations
2. **Gradient Accumulation** — simulate large batch sizes without the memory cost
3. **Data Parallelism** — replicate the model across GPUs and split the data

These three techniques are the tools you reach for *before* you consider any form of model parallelism. Let us start with the biggest memory hog: activations.

## (2) Activation Recomputation (Gradient Checkpointing)

### Why Activations Consume So Much Memory

During the forward pass of a neural network, every layer produces an output — an **activation**. For a transformer with $L$ layers, the forward pass produces activations $a_1, a_2, \ldots, a_L$. During the backward pass, to compute the gradient at layer $l$, we need the activation $a_l$ from the forward pass (this is how the chain rule works — we multiply the incoming gradient by the local Jacobian, which depends on $a_l$).

The standard approach is simple: store every activation during the forward pass, then use them during the backward pass. This means we need memory proportional to $L$ — every layer's output is kept alive simultaneously.

For a concrete example, consider a transformer with 24 layers, a hidden dimension of 2048, a sequence length of 2048, and a micro-batch size of 8. Each activation tensor has shape $(8, 2048, 2048)$, and in FP16 each element takes 2 bytes:

$$\text{Memory per activation} = 8 \times 2048 \times 2048 \times 2 \text{ bytes} = 64 \text{ MB}$$

With 24 layers, this is $24 \times 64 = 1536$ MB, or about 1.5 GB — just from activations. And this is a relatively small model! For larger models with longer sequences, this grows to tens of gigabytes.

![Standard forward pass: all activations are stored in memory simultaneously.](figures/figure_1.png)
*Activation checkpointing: standard approach stores all activations (top), while checkpointing stores only selected checkpoints and recomputes the rest (bottom).*

### The Key Idea: Discard and Recompute

Here is the insight behind **activation recomputation** (also called **gradient checkpointing**): instead of storing all activations, we discard most of them after the forward pass and **recompute them on-the-fly** during the backward pass.

Think of it this way. Suppose you are reading a long textbook and taking notes. The standard approach is to write down every detail — every calculation, every intermediate step. This uses a lot of paper (memory). The checkpointing approach is to write down only the key results at the end of each chapter. When you need an intermediate result for a later chapter, you go back to the nearest chapter summary and re-derive it from there.

We are trading **compute** for **memory**. We do more computation (we repeat parts of the forward pass), but we use much less memory.

### How It Works: Full Checkpointing

In **full checkpointing**, we divide the $L$ layers into $\sqrt{L}$ segments. We only store the activations at the **boundary** of each segment — these are the "checkpoints". All other activations are discarded.

During the backward pass, when we need the activation for layer $l$, we:

1. Find the nearest checkpoint before layer $l$ (say it is at layer $k$)
2. Re-run the forward pass from layer $k$ to layer $l$ to recompute $a_l$
3. Use $a_l$ to compute the gradient at layer $l$
4. Discard $a_l$ again

Let us work through a numerical example. Suppose we have $L = 16$ layers. We place checkpoints every $\sqrt{16} = 4$ layers, at layers 0, 4, 8, and 12.

**Memory usage:**
- Standard: store all 16 activations simultaneously → $16 \times 64 = 1024$ MB
- Checkpointed: store 4 checkpoint activations + at most 4 recomputed activations within a segment → $(4 + 4) \times 64 = 512$ MB

That is a **2x reduction** in activation memory. In general, for $L$ layers with $\sqrt{L}$ checkpoints:

$$\text{Activation memory} = O(\sqrt{L}) \quad \text{(down from } O(L)\text{)}$$

The price we pay is in compute. In the worst case, every activation must be recomputed once, which means the forward pass is essentially run twice. The total compute overhead is approximately:

$$\text{Compute overhead} \approx 33\%$$

This 33% number comes from the fact that the forward pass is roughly one-third of the total compute (forward + backward), so repeating the forward pass adds about one-third on top. In practice, it is often less because not every activation needs recomputation at the same time.

### Selective Checkpointing

Full checkpointing treats all layers equally, but not all layers produce equally large activations. In a transformer, the **attention matrices** (shape: batch $\times$ heads $\times$ seq_len $\times$ seq_len) are enormous compared to the **linear layer outputs**. Selective checkpointing exploits this:

- **Checkpoint** the layers with the largest activations (attention, layer norms)
- **Keep** the layers with smaller activations (linear projections)

This gives nearly the same memory savings as full checkpointing but with less recomputation overhead — often below 20%.

### Checkpointing in PyTorch

PyTorch provides `torch.utils.checkpoint.checkpoint` to implement this. Here is a simplified example:

```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 4 * hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Self-attention with residual
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        return x

class CheckpointedModel(torch.nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            # Wrap each layer with checkpointing
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

The `checkpoint` wrapper tells PyTorch: "Do not store the intermediate activations for this layer. During the backward pass, rerun the forward pass to recompute them." From the user's perspective, the model behaves identically — same outputs, same gradients — but uses far less memory.

## (3) Gradient Accumulation: Big Batches Without Big Memory

### The Batch Size Dilemma

Now let us address a second memory challenge. Suppose activation checkpointing brought your memory usage down enough to train with a micro-batch size of 2. But research has shown that your model converges much better with an effective batch size of 32. Loading 32 samples simultaneously would require $16\times$ more activation memory — and you simply do not have the GPU memory.

This is where **gradient accumulation** comes in. The idea is beautifully simple: instead of computing the gradient on a single large batch, we compute gradients on multiple small **micro-batches** and **accumulate** (sum) them before updating the weights.

![Gradient accumulation: multiple micro-batches contribute to one optimizer step.](figures/figure_2.png)
*Gradient accumulation: multiple micro-batches contribute to a single optimizer step.*

### Why It Works: The Mathematical Equivalence

Let us see why this is mathematically equivalent to using a large batch. Suppose our loss function for a batch of $B$ samples is the average:

$$\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \ell(x_i, \theta)$$

The gradient with respect to parameters $\theta$ is:

$$\nabla_\theta \mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(x_i, \theta)$$

Now suppose we split the $B$ samples into $K$ micro-batches, each of size $b = B / K$. The gradient for micro-batch $k$ is:

$$g_k = \frac{1}{b} \sum_{i \in \text{micro-batch } k} \nabla_\theta \ell(x_i, \theta)$$

If we accumulate (average) these $K$ micro-batch gradients:

$$\frac{1}{K} \sum_{k=1}^{K} g_k = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{b} \sum_{i \in \text{micro-batch } k} \nabla_\theta \ell(x_i, \theta) = \frac{1}{Kb} \sum_{i=1}^{B} \nabla_\theta \ell(x_i, \theta) = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(x_i, \theta)$$

This is exactly the same as the full-batch gradient. This is exactly what we want.

Let us plug in concrete numbers. Suppose $B = 32$ and we use $K = 16$ accumulation steps with $b = 2$ samples per micro-batch. At each step, we compute the gradient over 2 samples, divide by 16, and add it to the running total. After 16 steps, we have the exact same gradient as if we had processed all 32 samples in one go — but we never needed more than 2 samples in memory at a time.

### Implementation

Here is a clean implementation of gradient accumulation:

```python
import torch

model = ...        # your model
optimizer = ...    # e.g., AdamW
dataloader = ...   # yields micro-batches of size b

ACCUMULATION_STEPS = 16  # K = 16, effective batch = b * K

optimizer.zero_grad()

for step, (inputs, labels) in enumerate(dataloader):
    # Forward pass on micro-batch
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, labels)

    # Scale loss by accumulation steps (so gradients average correctly)
    scaled_loss = loss / ACCUMULATION_STEPS
    scaled_loss.backward()  # Gradients ACCUMULATE in .grad buffers

    # Only step the optimizer every K micro-batches
    if (step + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

The key insight is that PyTorch **accumulates gradients by default** — calling `loss.backward()` multiple times adds to the `.grad` buffers rather than overwriting them. We only call `optimizer.zero_grad()` after every $K$ micro-batches, so the gradients accumulate naturally. We divide the loss by $K$ to ensure the accumulated gradient is the mean, not the sum.

### The Trade-Off

Gradient accumulation is almost free in terms of extra computation. The only cost is **time**: instead of one forward+backward pass per optimizer step, you do $K$ sequential forward+backward passes. Since these micro-batches are processed sequentially (not in parallel), training takes about $K$ times longer per optimizer step.

This is fine when you have a single GPU. But what if you have multiple GPUs? This brings us to our third technique.

## (4) Data Parallelism: Replicate the Model, Split the Data

### The Core Idea

**Data parallelism** is the simplest and most widely used multi-GPU training strategy. The idea is:

1. **Copy** the entire model to every GPU
2. **Split** each mini-batch into equal shards — one shard per GPU
3. Each GPU computes the forward and backward pass **independently** on its shard
4. **Synchronize** the gradients across all GPUs (so every copy sees the same averaged gradient)
5. Each GPU updates its model weights identically

Because each GPU processes a different subset of the data, we effectively multiply the batch size by the number of GPUs. If each GPU processes a micro-batch of 8 samples and we have 4 GPUs, the effective batch size is 32 — without any single GPU ever holding 32 samples in memory.

![Data parallelism: N GPUs each hold a full model copy and process different data shards.](figures/figure_3.png)
*Data parallelism: N GPUs each hold a full model copy, process different data shards, and synchronize gradients.*

### The Gradient Synchronization Step

After each GPU computes its local gradients, we need to average them across all GPUs. This is done using an operation called **AllReduce**.

Suppose we have $N$ GPUs. GPU $i$ has computed local gradient $g_i$. The AllReduce operation computes:

$$\bar{g} = \frac{1}{N} \sum_{i=1}^{N} g_i$$

and places the result $\bar{g}$ on **every** GPU. After this, every GPU has the same averaged gradient and performs the same optimizer update, keeping all model copies perfectly in sync.

Let us plug in numbers. Suppose we have $N = 4$ GPUs and a single parameter whose gradients on each GPU are: $g_1 = 0.8$, $g_2 = 1.2$, $g_3 = 0.6$, $g_4 = 1.0$. The AllReduce computes:

$$\bar{g} = \frac{0.8 + 1.2 + 0.6 + 1.0}{4} = \frac{3.6}{4} = 0.9$$

Every GPU now has $\bar{g} = 0.9$ and applies the same update. The models stay synchronized.

### Why Data Parallelism Scales (Almost) Linearly

With $N$ GPUs, each processing a micro-batch of size $b$, the effective batch size is $B = N \times b$. The total compute for one training step is the same (each GPU does $1/N$ of the work), so in theory, training is $N$ times faster.

In practice, the speedup is slightly less than $N$ because of the **communication overhead** — the AllReduce operation requires sending gradient tensors across GPUs. For a model with $P$ parameters in FP16, the AllReduce must transfer $2P$ bytes in each direction. For a 1.3B parameter model, that is about 2.6 GB of data per AllReduce.

The typical scaling efficiency for data parallelism is:

| GPUs | Theoretical Speedup | Typical Actual Speedup |
|------|--------------------:|----------------------:|
| 1    | 1.0x               | 1.0x                  |
| 2    | 2.0x               | 1.9x                  |
| 4    | 4.0x               | 3.7x                  |
| 8    | 8.0x               | 7.2x                  |
| 16   | 16.0x              | 13.5x                 |

The gap grows with more GPUs because communication overhead increases. We will study the AllReduce algorithm in detail in the next pod — for now, just know that it is the bottleneck in data parallelism.

### Data Parallelism in PyTorch

PyTorch provides `DistributedDataParallel` (DDP) for data parallelism. Here is a simplified conceptual example:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize the process group (one process per GPU)
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

# Create model and wrap with DDP
model = MyTransformer().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Each GPU gets a different subset of the data
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for inputs, labels in dataloader:
    inputs, labels = inputs.cuda(local_rank), labels.cuda(local_rank)
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss.backward()       # DDP automatically synchronizes gradients
    optimizer.step()
    optimizer.zero_grad()
```

The beauty of DDP is that the AllReduce is handled automatically inside `loss.backward()`. From the user's perspective, the code looks almost identical to single-GPU training — you just wrap the model in `DDP` and use a `DistributedSampler` to ensure each GPU sees different data.

## (5) Putting It All Together: A Concrete Example

Now let us see how these three techniques combine. Suppose we want to train a **1.3B parameter transformer** on a dataset with the following setup:

- **GPU**: 4x NVIDIA A100 (40 GB each)
- **Target effective batch size**: 256 samples
- **Sequence length**: 2048

**Step 1: Memory budget.** The model weights, gradients, and optimizer states take about 15.6 GB in mixed precision. That leaves $40 - 15.6 = 24.4$ GB for activations on each GPU.

**Step 2: Activation checkpointing.** Without checkpointing, a micro-batch of even 4 samples would require ~24 GB of activation memory — dangerously close to the limit. With full checkpointing, we reduce activation memory by roughly $\sqrt{L}$ factor. For 24 layers, this drops activation memory to about 8 GB for a micro-batch of 4. Now we fit comfortably.

**Step 3: Data parallelism.** Each of our 4 GPUs processes a micro-batch of 4 samples, giving us an effective batch of $4 \times 4 = 16$ samples per step.

**Step 4: Gradient accumulation.** We need an effective batch size of 256, but data parallelism gives us only 16 per step. So we set accumulation steps $K = 256 / 16 = 16$. Every 16 micro-batch steps, we perform one optimizer update.

![Memory savings comparison: baseline vs. checkpointing vs. checkpointing + accumulation.](figures/figure_4.png)
*Memory savings comparison: baseline vs. checkpointing vs. checkpointing + accumulation.*

The result: a model that **would not train at all** on a single GPU now trains effectively across 4 GPUs with a batch size of 256.

Here is the combined workflow:

![Combined workflow: checkpoint activations, accumulate gradients, synchronize across GPUs.](figures/figure_5.png)
*The combined workflow: activation checkpointing reduces per-layer memory, gradient accumulation simulates large batches, and data parallelism distributes across GPUs with AllReduce synchronization.*

## (6) When These Techniques Are Not Enough

These three techniques — checkpointing, accumulation, and data parallelism — are powerful, but they have a fundamental limitation: **every GPU must hold a complete copy of the model.** For a 1.3B model, that is fine. But what about a 70B model? The weights alone need 140 GB in FP16 — far more than a single A100 can hold.

When the model itself does not fit on a single GPU, we need a completely different strategy. We need to **split the model** across multiple GPUs. This is where **model parallelism** comes in — tensor parallelism, pipeline parallelism, ZeRO optimizer, and the other techniques that form the rest of this course.

But before we get to model parallelism, there is one more thing we have glossed over: the AllReduce operation. We said that data parallelism requires "synchronizing gradients across GPUs", but how exactly does this work? How do you efficiently average tensors across 8, 64, or even 1,024 GPUs? This brings us to the **Ring-AllReduce** algorithm — the subject of our next pod.
