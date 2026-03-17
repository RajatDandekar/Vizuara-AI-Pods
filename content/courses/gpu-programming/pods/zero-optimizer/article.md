# ZeRO-1, 2, 3 — Zero Redundancy Optimizer

*The elegant idea behind ZeRO: instead of replicating everything on every GPU, partition optimizer states, gradients, and parameters across GPUs.*

---

Let us start with a concrete observation. You are training a 7-billion-parameter model using data parallelism across 8 GPUs. Each GPU holds a complete copy of the model weights, a complete copy of the gradients, and a complete copy of the Adam optimizer states. In mixed precision training, that means each GPU stores:

- **Weights (fp16)**: $2\Psi = 14$ GB
- **Gradients (fp16)**: $2\Psi = 14$ GB
- **Optimizer states (fp32 master weights + m + v)**: $12\Psi = 84$ GB

That is roughly **112 GB per GPU** just for model states (ignoring activations). Across 8 GPUs, the system is using $8 \times 112 = 896$ GB of total memory. But here is the thing that should bother you: all 8 GPUs are storing **exactly the same weights**, **exactly the same optimizer states**, and after the allreduce, **exactly the same gradients**. There are 8 identical copies of everything. The total unique information across the entire cluster is still just 112 GB — the remaining 784 GB is pure redundancy.

This is an enormous waste. What if, instead of replicating everything on every GPU, we **partitioned** these states across GPUs so that each GPU only stores $1/N$-th of the data? That is the core idea behind **ZeRO — the Zero Redundancy Optimizer**, introduced by Rajbhandari et al. (2020) at Microsoft Research. It is one of the most elegant and impactful ideas in distributed training.

![Data Parallelism redundancy: 4 GPUs each storing full copies of weights, gradients, and optimizer states.](figures/figure_1.png)
*Data Parallelism redundancy: every GPU stores a complete copy of weights, gradients, and optimizer states. With 4 GPUs, the system uses 4x the memory of a single GPU, despite all copies being identical.*

## The Redundancy Problem in Numbers

Let us formalize the waste. In standard data parallelism with mixed precision training and the Adam optimizer, each GPU stores:

$$M_{\text{per GPU}} = 2\Psi + 2\Psi + 12\Psi = 16\Psi \text{ bytes}$$

where $\Psi$ is the number of parameters. The three terms are:

1. **fp16 weights**: $2\Psi$ bytes
2. **fp16 gradients**: $2\Psi$ bytes
3. **fp32 optimizer states** (master weights + first moment $m$ + second moment $v$): $4\Psi + 4\Psi + 4\Psi = 12\Psi$ bytes

With $N$ GPUs, the total memory used across the cluster is $N \times 16\Psi$, but the total unique data is only $16\Psi$. The redundancy factor is $N$ — with 64 GPUs, you are wasting 98.4% of your aggregate memory.

Let us plug in numbers for our 7B model on 8 GPUs:

| Component | Per GPU | 8 GPUs (Total) | Unique Data |
|-----------|---------|----------------|-------------|
| Weights (fp16) | 14 GB | 112 GB | 14 GB |
| Gradients (fp16) | 14 GB | 112 GB | 14 GB |
| Optimizer states (fp32) | 84 GB | 672 GB | 84 GB |
| **Total** | **112 GB** | **896 GB** | **112 GB** |

That is 784 GB of redundant memory. ZeRO eliminates this redundancy in three progressive stages, each one more aggressive than the last.

## ZeRO Stage 1: Partition the Optimizer States ($P_{os}$)

The first observation is that the optimizer states are by far the largest component — $12\Psi$ bytes out of $16\Psi$ total, or **75% of the memory**. And here is a key insight about how Adam works: during the optimizer step, each parameter's update depends **only on that parameter's own gradient, first moment, and second moment**. There is no cross-parameter dependency. This means GPU 0 does not need the optimizer states for parameter 5,000 if GPU 3 is responsible for updating that parameter.

ZeRO Stage 1 partitions the optimizer states across $N$ GPUs. Each GPU stores only $1/N$-th of the optimizer states — the portion corresponding to the parameters it is responsible for updating.

![ZeRO Stage 1: optimizer states partitioned across GPUs, weights and gradients still replicated.](figures/figure_2.png)
*ZeRO Stage 1 ($P_{os}$): optimizer states are partitioned across GPUs. Each GPU stores only 1/N of Adam's master weights, first moment, and second moment. Weights and gradients remain fully replicated.*

Here is how the training step works under ZeRO Stage 1:

1. **Forward pass**: Each GPU uses its full local copy of the fp16 weights (still replicated) to compute the forward pass on its data shard.
2. **Backward pass**: Each GPU computes a full set of fp16 gradients (still replicated in shape).
3. **AllReduce gradients**: Same as standard data parallelism — all GPUs average their gradients via allreduce.
4. **Optimizer step (partitioned)**: Each GPU only updates the $1/N$ fraction of parameters it owns. It uses its local partition of the optimizer states ($m$, $v$, and the fp32 master weights) to compute the update.
5. **AllGather weights**: After the optimizer step, each GPU has updated only its $1/N$ partition of the fp16 weights. An allgather operation collects all partitions so that every GPU ends up with the full, updated fp16 weights.

Now the question is: how much memory does this save?

### Memory Formula for ZeRO Stage 1

Each GPU still stores:
- Full fp16 weights: $2\Psi$ bytes
- Full fp16 gradients: $2\Psi$ bytes
- **Partitioned** optimizer states: $12\Psi / N$ bytes

$$M_{\text{ZeRO-1}} = 2\Psi + 2\Psi + \frac{12\Psi}{N} = 4\Psi + \frac{12\Psi}{N}$$

Let us plug in our 7B model with $N = 8$ GPUs:

$$M_{\text{ZeRO-1}} = 4 \times 7\text{B} + \frac{12 \times 7\text{B}}{8} = 28 \text{ GB} + 10.5 \text{ GB} = 38.5 \text{ GB}$$

Compare this to standard data parallelism at 112 GB per GPU. That is a **2.9x reduction** — from 112 GB to 38.5 GB — just by partitioning the optimizer states. With more GPUs, the savings are even more dramatic. At $N = 64$:

$$M_{\text{ZeRO-1}} = 28 + \frac{84}{64} = 28 + 1.3 = 29.3 \text{ GB}$$

The optimizer memory becomes almost negligible, and the bottleneck shifts to the weights and gradients.

### Communication Cost of ZeRO Stage 1

Here is the best part: ZeRO Stage 1 has **the same communication volume as standard data parallelism**. The allreduce for gradients is identical. The additional allgather for updated weights transfers $2\Psi$ bytes total, but if you look carefully, the standard allreduce already transfers $2 \times 2\Psi$ bytes (recall from the ring-allreduce discussion: the total data per GPU approaches $2D$ for large $N$). The allgather can be overlapped with the allreduce, and in practice, the extra communication is negligible.

This is exactly what we want — a massive memory reduction with essentially zero communication overhead.


## ZeRO Stage 2: Partition Gradients Too ($P_{os+g}$)

ZeRO Stage 1 saved us 75% of the memory waste by partitioning optimizer states. But the gradients ($2\Psi$ bytes) are still fully replicated on every GPU. Do we really need that?

Think about it: after the backward pass, each GPU has a full gradient vector. But each GPU only needs the gradients for the parameters it owns (the ones whose optimizer states it stores). GPU 0 does not need the gradient for parameter 5,000 if GPU 3 is responsible for updating it. Why store gradients you will never use?

ZeRO Stage 2 partitions the gradients in addition to the optimizer states. Instead of an allreduce (which gives every GPU the full averaged gradient), ZeRO Stage 2 uses a **reduce-scatter** operation. In a reduce-scatter, the gradient vector is divided into $N$ chunks, and each GPU ends up with only the fully reduced chunk corresponding to its partition — exactly the gradients it needs for its optimizer step.

![ZeRO Stage 2: optimizer states and gradients partitioned, weights still replicated.](figures/figure_3.png)
*ZeRO Stage 2 ($P_{os+g}$): both optimizer states and gradients are partitioned. Each GPU stores only 1/N of the gradients via reduce-scatter, keeping just the gradients it needs for its own optimizer partition.*

The training step becomes:

1. **Forward pass**: Same as before — full fp16 weights are replicated.
2. **Backward pass**: Each GPU computes a full set of gradients.
3. **Reduce-scatter gradients**: Instead of allreduce, each GPU receives only its $1/N$ partition of the averaged gradients. The remaining gradient data is **immediately discarded**, freeing memory.
4. **Optimizer step**: Each GPU updates its $1/N$ partition using its local optimizer states and the reduced gradient partition.
5. **AllGather weights**: Same as ZeRO Stage 1 — collect the updated fp16 weights.

### Memory Formula for ZeRO Stage 2

Each GPU stores:
- Full fp16 weights: $2\Psi$ bytes
- **Partitioned** gradients: $2\Psi / N$ bytes
- **Partitioned** optimizer states: $12\Psi / N$ bytes

$$M_{\text{ZeRO-2}} = 2\Psi + \frac{2\Psi}{N} + \frac{12\Psi}{N} = 2\Psi + \frac{14\Psi}{N}$$

For our 7B model on 8 GPUs:

$$M_{\text{ZeRO-2}} = 14 + \frac{14 \times 7\text{B}}{8} = 14 + 12.25 = 26.25 \text{ GB}$$

That is down from 38.5 GB with ZeRO-1 and 112 GB with standard DP. And the communication cost? A reduce-scatter transfers exactly half the data of an allreduce, and the allgather transfers the other half. Together, reduce-scatter + allgather = allreduce. So **ZeRO Stage 2 has the same total communication volume as standard data parallelism**. We are still not paying any communication penalty.


## ZeRO Stage 3: Partition Everything ($P_{os+g+p}$)

At this point, the only component that is still fully replicated is the model weights themselves ($2\Psi$ bytes per GPU). For a 7B model, that is 14 GB on every GPU — and for a 70B model, it would be 140 GB, which does not fit on a single GPU regardless of what we do with the optimizer states.

ZeRO Stage 3 takes the partitioning to its logical conclusion: it also partitions the model weights. Each GPU stores only $1/N$-th of the parameters.

But wait — every GPU needs the full weights to compute the forward and backward pass. If each GPU only stores $1/N$ of the weights, how does it run the forward pass?

The answer is: **allgather on demand**. Before each layer's forward computation, the GPUs perform an allgather to temporarily reconstruct the full weights for that layer. After the layer is done, the gathered weights are discarded. The same happens during the backward pass — allgather the weights for each layer, compute the gradients, then discard.

![ZeRO Stage 3: everything partitioned, with allgather arrows for the forward pass.](figures/figure_4.png)
*ZeRO Stage 3 ($P_{os+g+p}$): weights, gradients, and optimizer states are all partitioned. Each GPU stores only 1/N of everything. During the forward and backward pass, an allgather reconstructs the full weights for each layer on demand.*

The training step for ZeRO Stage 3:

1. **Forward pass (layer by layer)**:
   - For each layer: allgather the full weights for that layer from all GPUs
   - Compute the layer's output
   - Discard the non-local weight partitions (keep only the $1/N$ you own)
2. **Backward pass (layer by layer)**:
   - For each layer: allgather the full weights again (needed for gradient computation)
   - Compute the gradients for that layer
   - Reduce-scatter the gradients so each GPU keeps only its $1/N$ partition
   - Discard the non-local weights
3. **Optimizer step**: Each GPU updates its $1/N$ partition of weights using its local optimizer states and gradient partition.

### Memory Formula for ZeRO Stage 3

Each GPU stores:
- **Partitioned** fp16 weights: $2\Psi / N$ bytes
- **Partitioned** gradients: $2\Psi / N$ bytes
- **Partitioned** optimizer states: $12\Psi / N$ bytes

$$M_{\text{ZeRO-3}} = \frac{2\Psi + 2\Psi + 12\Psi}{N} = \frac{16\Psi}{N}$$

For our 7B model on 8 GPUs:

$$M_{\text{ZeRO-3}} = \frac{16 \times 7\text{B}}{8} = \frac{112}{8} = 14 \text{ GB}$$

From 112 GB down to 14 GB per GPU. That is an **8x reduction** — exactly the number of GPUs, which makes sense because we have eliminated all redundancy. Every byte of model state is stored on exactly one GPU.

And on 64 GPUs:

$$M_{\text{ZeRO-3}} = \frac{112}{64} = 1.75 \text{ GB}$$

This means you could theoretically fit a 7B model's states in under 2 GB per GPU, leaving almost the entire GPU memory for activations and larger batch sizes. This is truly amazing.

### Communication Cost of ZeRO Stage 3

Here is where ZeRO Stage 3 pays a price. In the forward pass, each layer requires an allgather of the full weights ($\Psi$ elements total across all layers). In the backward pass, each layer requires another allgather (for the weights) plus a reduce-scatter (for the gradients).

Let us count the total communication per GPU:

- **Forward allgather**: $\Psi \times 2$ bytes (fp16 weights, once per forward pass)
- **Backward allgather**: $\Psi \times 2$ bytes (fp16 weights, once per backward pass)
- **Backward reduce-scatter**: $\Psi \times 2$ bytes (fp16 gradients)

Total: $6\Psi$ bytes per GPU.

Compare this to ZeRO Stage 1 or 2, where the total communication is approximately $4\Psi$ bytes per GPU (the allreduce, which is equivalent to a reduce-scatter + allgather of gradients). ZeRO Stage 3 has **1.5x the communication volume** of standard data parallelism, because of the extra allgather of weights during the forward pass.

$$\frac{\text{ZeRO-3 comm}}{\text{DP comm}} = \frac{6\Psi}{4\Psi} = 1.5\times$$

This 50% communication increase is the price you pay for the maximum memory savings. Whether this trade-off is worth it depends on your situation: if your model does not fit with ZeRO-2, the extra communication is a small price to pay for actually being able to train at all.


## Memory Comparison: The Complete Picture

Let us put all four approaches side by side for our 7B model on 8 GPUs. We use $\Psi = 7 \times 10^9$ parameters in mixed precision (fp16 weights/gradients, fp32 optimizer states):

| Approach | Weights | Gradients | Optimizer | Total per GPU | Comm vs DP |
|----------|---------|-----------|-----------|--------------|------------|
| **Data Parallelism** | 14 GB | 14 GB | 84 GB | **112 GB** | 1.0x |
| **ZeRO Stage 1** | 14 GB | 14 GB | 10.5 GB | **38.5 GB** | 1.0x |
| **ZeRO Stage 2** | 14 GB | 1.75 GB | 10.5 GB | **26.25 GB** | 1.0x |
| **ZeRO Stage 3** | 1.75 GB | 1.75 GB | 10.5 GB | **14 GB** | 1.5x |

![Memory comparison bar chart: standard DP vs ZeRO-1 vs ZeRO-2 vs ZeRO-3.](figures/figure_5.png)
*Memory per GPU for a 7B model on 8 GPUs: standard Data Parallelism vs. ZeRO-1, ZeRO-2, and ZeRO-3. The optimizer states (purple) dominate in standard DP, and each ZeRO stage progressively eliminates more redundancy.*

Let us also compute a second example with a larger model. For a **30B model on 64 GPUs** ($\Psi = 30 \times 10^9$):

| Approach | Formula | Memory per GPU |
|----------|---------|---------------|
| **DP** | $16\Psi$ | 480 GB |
| **ZeRO-1** | $4\Psi + 12\Psi/N$ | 125.6 GB |
| **ZeRO-2** | $2\Psi + 14\Psi/N$ | 66.6 GB |
| **ZeRO-3** | $16\Psi/N$ | 7.5 GB |

Standard data parallelism would require 480 GB per GPU — impossible. Even ZeRO-1 requires 125 GB — still larger than an 80 GB A100. ZeRO-2 at 66.6 GB fits on an A100. ZeRO-3 at 7.5 GB leaves enormous headroom for activations. This illustrates why ZeRO-3 is essential for training very large models.


## Practical Implementation with DeepSpeed

ZeRO was developed as part of Microsoft's **DeepSpeed** library, which makes it remarkably easy to use. The beauty of DeepSpeed's design is that you barely need to change your training code — most of the configuration is done through a JSON config file.

Here is a minimal example:

```python
import deepspeed
import torch

# Your standard model definition
model = MyTransformer()
optimizer = None  # DeepSpeed will create the optimizer

# DeepSpeed configuration (can also be a JSON file)
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 2,                  # ZeRO Stage 2
        "allgather_partitions": True,
        "reduce_scatter": True,
        "overlap_comm": True,        # Overlap communication with computation
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
        }
    }
}

# Initialize DeepSpeed — it wraps the model and optimizer
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)

# Training loop looks almost identical to standard PyTorch
for batch in dataloader:
    inputs, labels = batch
    outputs = model_engine(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, labels)

    model_engine.backward(loss)    # Replaces loss.backward()
    model_engine.step()            # Replaces optimizer.step() + zero_grad()
```

Let us understand this code in detail. The `deepspeed.initialize` call wraps your model in a DeepSpeed engine that automatically handles gradient partitioning, reduce-scatter instead of allreduce (for Stage 2+), and the partitioned optimizer step. The `model_engine.backward(loss)` call triggers the backward pass with ZeRO's gradient partitioning, and `model_engine.step()` runs the partitioned optimizer update followed by the allgather of updated weights.

To switch between ZeRO stages, you simply change the `"stage"` field:

```python
# ZeRO Stage 1: Partition optimizer states only
"zero_optimization": {"stage": 1}

# ZeRO Stage 2: Partition optimizer states + gradients
"zero_optimization": {"stage": 2}

# ZeRO Stage 3: Partition everything
"zero_optimization": {
    "stage": 3,
    "param_persistence_threshold": 1e5,  # Keep small params replicated
    "max_live_parameters": 1e9,          # Max params in memory at once
}
```

ZeRO Stage 3 has a few extra configuration options. The `param_persistence_threshold` tells DeepSpeed to keep very small parameter tensors (below this size) replicated on all GPUs, because the allgather overhead for tiny tensors would outweigh the memory savings. The `max_live_parameters` controls how many parameters can be gathered simultaneously — this caps the temporary memory spike during the forward pass.


## ZeRO-Offload and ZeRO-Infinity: Going Beyond GPU Memory

What if even ZeRO Stage 3 is not enough? For truly massive models — 100B+ parameters — or when you are training on a limited number of GPUs, you might still run out of GPU memory. ZeRO-Offload and ZeRO-Infinity extend the partitioning idea beyond GPU memory by offloading states to **CPU memory** or even **NVMe storage**.

**ZeRO-Offload** (Ren et al., 2021) keeps the model weights on the GPU for fast forward and backward passes, but offloads the optimizer states and optionally the gradients to CPU memory. Since optimizer states are only accessed during the optimizer step (not during the forward/backward pass), moving them to CPU has a relatively small impact on throughput. The optimizer step itself runs on the CPU, which is slower, but this only happens once per training step.

**ZeRO-Infinity** (Rajbhandari et al., 2021) goes one step further: it can offload states all the way to NVMe SSDs. A modern NVMe drive offers several TB of storage at a fraction of the cost of GPU memory. ZeRO-Infinity uses sophisticated prefetching — it overlaps the NVMe reads for the next layer's parameters with the computation on the current layer — so that the storage latency is partially hidden.

The configuration in DeepSpeed is straightforward:

```python
"zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
        "device": "cpu",             # Offload optimizer states to CPU
        "pin_memory": True,          # Use pinned memory for faster transfers
    },
    "offload_param": {
        "device": "cpu",             # Offload parameters to CPU (ZeRO-Infinity)
        "pin_memory": True,
    },
}
```

With ZeRO-Infinity, a single DGX-2 node (16 V100 GPUs with 32 GB each = 512 GB total GPU memory) was shown to train models with over **1 trillion parameters**. The model states are distributed across CPU memory and NVMe storage, while the GPUs handle the actual computation layer by layer. The throughput is lower than keeping everything on-GPU, but it makes the previously impossible, possible.


## When to Use Which Stage

Here is a practical decision guide:

**ZeRO Stage 1** is the easiest upgrade from standard data parallelism. Use it when your model fits in GPU memory with standard DP but you want to free up memory for larger batch sizes or longer sequences. There is no communication penalty and minimal code changes.

**ZeRO Stage 2** is the sweet spot for most training runs. It provides significant memory savings (partitioning both optimizer states and gradients) with the same communication volume as standard DP. If your model fits with ZeRO-2, there is rarely a reason to go to ZeRO-3.

**ZeRO Stage 3** is necessary when the model weights themselves do not fit on a single GPU, or when you need maximum memory efficiency. The 1.5x communication overhead is the trade-off, but in practice, with good communication-computation overlap, the actual slowdown is often less than 50%.

**ZeRO-Offload / Infinity** is for when you are truly GPU-memory-constrained — either training very large models on a small number of GPUs, or pushing the limits of what your hardware can handle. The throughput trade-off is more significant here, but it extends the reach of your hardware dramatically.


## What Comes Next

ZeRO solves the memory redundancy problem in data parallelism beautifully. By partitioning optimizer states, gradients, and parameters across GPUs, it lets us train models that are $N$ times larger than what a single GPU could handle — with minimal or modest communication overhead.

But notice what ZeRO does not do: it still treats each layer's computation as a single, indivisible unit. Every GPU executes the full forward and backward pass for every layer (even if it only stores $1/N$ of the parameters at rest). The computation itself is not split.

ZeRO partitions the model's states but still treats each layer as a whole. What if we want to split the computation within a single layer? This brings us to **Tensor Parallelism** — where we slice individual weight matrices across GPUs so that each GPU computes only a portion of a layer's output, reducing both memory and computation per GPU simultaneously.
