# Intro to GPUs and GPU Parallelism for LLMs

*What GPUs are, why they matter for deep learning, and a bird's-eye view of the five parallelism strategies that make training large language models possible.*

---

Let us start with a simple thought experiment. Imagine you are a teacher who needs to grade 10,000 exam papers by tomorrow morning. You have two options. Option A: hire one brilliant professor who reads each paper carefully, one at a time, spending 30 seconds per paper. Option B: hire 10,000 teaching assistants who are each a bit slower individually -- maybe 60 seconds per paper -- but they all work at the same time, each grading one paper simultaneously.

Option A finishes in about 83 hours. Option B finishes in 60 seconds.

This, in essence, is the difference between a CPU and a GPU. And understanding this difference is the key to understanding why modern AI -- especially large language models like GPT-4, LLaMA, and DeepSeek -- would be completely impossible without GPUs.

In this article, we will build a solid understanding of what GPUs are, why deep learning needs them so desperately, why even a single powerful GPU is not enough for today's largest models, and finally, we will take a bird's-eye view of the five parallelism strategies that make training these massive models possible.


## What Is a GPU?

A **GPU** (Graphics Processing Unit) was originally designed for one thing: rendering pixels on a screen. When you play a video game, your screen might display two million pixels, and each pixel needs its color calculated independently, 60 times per second. This is a *massively parallel* workload -- millions of tiny, identical calculations happening simultaneously.

To handle this, GPU designers made a fundamentally different architectural choice from CPUs. Instead of building a few very powerful processing cores (like a CPU), they packed thousands of simpler cores onto a single chip.

Let us look at the architecture of a modern NVIDIA GPU like the A100. At the highest level, the chip is organized into **Streaming Multiprocessors (SMs)**. The A100 has 108 SMs. Each SM contains:

- **64 CUDA cores** (also called FP32 cores) for general arithmetic
- **4 Tensor Cores** for specialized matrix multiplication
- **Shared memory** (164 KB) that threads within the SM can use to communicate
- **A register file** -- the fastest storage on the chip

In total, the A100 has 6,912 CUDA cores and 432 Tensor Cores. Compare this with a high-end CPU like the Intel Xeon, which might have 32 to 64 cores. The GPU has roughly 100 times more processing units.

But here is the crucial point: each individual CUDA core is much simpler and slower than a CPU core. A CPU core can handle complex branching logic, out-of-order execution, branch prediction, and sophisticated caching. A CUDA core does one thing well: arithmetic on numbers.

This is exactly what we want for deep learning.


![CPU vs GPU architecture comparison showing few powerful cores versus thousands of simple cores.](figures/figure_1.png)
*CPU vs GPU architecture comparison: a CPU has a few powerful cores optimized for sequential tasks, while a GPU has thousands of simpler cores optimized for parallel throughput.*


## CPUs vs GPUs: Latency vs Throughput

Let us formalize this distinction. CPUs and GPUs represent two fundamentally different design philosophies:

**CPUs are latency-oriented.** They are designed to complete a *single* task as fast as possible. To achieve this, CPUs dedicate enormous chip area to:
- Large caches (L1, L2, L3 -- often 30+ MB total)
- Branch predictors that guess which instruction comes next
- Out-of-order execution engines that rearrange instructions for speed
- Prefetching logic that anticipates which data you will need

**GPUs are throughput-oriented.** They are designed to complete *many* tasks simultaneously, even if each individual task takes longer. GPUs dedicate their chip area to:
- Thousands of arithmetic units
- Hardware thread schedulers that can switch between thousands of threads instantly
- Wide memory buses (the A100 has a 5120-bit memory bus vs a CPU's 128-bit bus)

Here is a simple Python benchmark that illustrates this difference:

```python
import numpy as np
import time

# Create two large matrices
size = 4096
A = np.random.randn(size, size).astype(np.float32)
B = np.random.randn(size, size).astype(np.float32)

# CPU matrix multiplication
start = time.time()
C_cpu = A @ B
cpu_time = time.time() - start
print(f"CPU matrix multiply ({size}x{size}): {cpu_time:.3f} seconds")
```

On a typical machine, this takes around 2-5 seconds. Now let us try the same on a GPU:

```python
import torch

A_gpu = torch.randn(size, size, device='cuda')
B_gpu = torch.randn(size, size, device='cuda')

# Warm up the GPU
_ = A_gpu @ B_gpu
torch.cuda.synchronize()

# GPU matrix multiplication
start = time.time()
C_gpu = A_gpu @ B_gpu
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"GPU matrix multiply ({size}x{size}): {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time / gpu_time:.1f}x")
```

On an A100, this completes in roughly 5-15 milliseconds -- a speedup of 100x to 500x. This is not a small improvement. This is the difference between a training run taking one day versus one year.


## Why Deep Learning Needs GPUs

Now the question is: why does deep learning benefit so enormously from GPUs? The answer comes down to one operation: **matrix multiplication**.

Let us think about what happens when data passes through a single layer of a neural network. If we have an input vector $\mathbf{x}$ of size $d_{\text{in}}$ and a weight matrix $\mathbf{W}$ of shape $d_{\text{out}} \times d_{\text{in}}$, the output is:

$$\mathbf{y} = \mathbf{W} \mathbf{x}$$

Each element of the output vector $\mathbf{y}$ is a dot product:

$$y_i = \sum_{j=1}^{d_{\text{in}}} W_{ij} \cdot x_j$$

Here is the key insight: **every element of the output can be computed independently**. If $d_{\text{out}} = 4096$, we have 4,096 dot products that share no dependencies. We can compute all of them at the same time.

Let us plug in some numbers. For a single transformer layer in a 7-billion parameter model like LLaMA-2 7B:
- The hidden dimension is $d = 4096$
- The attention QKV projection computes $\mathbf{W}_{QKV} \mathbf{x}$, where $\mathbf{W}_{QKV}$ is of shape $12288 \times 4096$
- That is roughly 50 million multiply-add operations -- just for one projection, for one token, in one layer

A modern LLM might process 2,048 tokens through 32 layers, each containing multiple such projections. The total operation count for a single forward pass reaches into the **trillions** of floating-point operations (FLOPs).

A CPU can perform roughly 1 teraFLOP/s (one trillion operations per second). An A100 GPU can perform 312 teraFLOPs/s in FP16. That is a 300x advantage -- and this is precisely why training a large model on CPUs alone would take years instead of days.


![Matrix multiplication showing why GPUs excel at parallel dot products.](figures/figure_3.png)
*Matrix multiplication is the core operation of deep learning. Each output element is an independent dot product -- perfectly suited for the GPU's thousands of parallel cores.*


## The GPU Memory Hierarchy

Before we move further, let us understand how GPU memory is organized, because memory -- not compute -- is often the real bottleneck in training large models.

A GPU has a layered memory hierarchy, from fastest (but smallest) to slowest (but largest):

1. **Registers** -- Each thread has its own registers. This is the fastest storage (zero latency), but extremely limited (255 registers per thread on modern GPUs).

2. **Shared Memory / L1 Cache** -- Shared among all threads within an SM. On the A100, this is 164 KB per SM and has a bandwidth of roughly 19 TB/s. Threads within the same SM can use shared memory to communicate and share intermediate results.

3. **L2 Cache** -- Shared across all SMs. The A100 has 40 MB of L2 cache. This sits between shared memory and main memory.

4. **HBM (High Bandwidth Memory)** -- This is the GPU's main memory, what we commonly refer to as "GPU memory" or "VRAM." The A100 has 80 GB of HBM2e with a bandwidth of 2 TB/s.

For comparison, a CPU's DRAM bandwidth is typically 50-100 GB/s. The A100's memory bandwidth is 20-40 times higher. But even 2 TB/s is not always enough -- this is why kernel optimization (which we will cover in later pods) focuses heavily on keeping data in shared memory and registers rather than constantly fetching from HBM.


![GPU memory hierarchy from registers to shared memory to L2 cache to HBM.](figures/figure_2.png)
*The GPU memory hierarchy. Data moves from HBM (large, slow) through L2 cache and shared memory to registers (tiny, fast). Effective GPU programming means minimizing trips to HBM.*


## Why a Single GPU Is Never Enough

Now that we understand why GPUs are so powerful for deep learning, you might ask: if a single A100 has 80 GB of memory and 312 teraFLOPs of compute, is that not enough?

Let us do some concrete arithmetic. We will calculate exactly how much memory is needed to train a 7-billion parameter model like LLaMA-2 7B using the standard AdamW optimizer.

**Component 1: Model Weights.**
Each parameter is a floating-point number. In FP32 (full precision), each parameter requires 4 bytes. In FP16 (half precision, commonly used in mixed-precision training), each requires 2 bytes.

$$\text{Weight memory (FP16)} = 7 \times 10^9 \times 2 \text{ bytes} = 14 \text{ GB}$$

**Component 2: Gradients.**
During backpropagation, we compute a gradient for every parameter. These are the same size as the weights.

$$\text{Gradient memory (FP16)} = 7 \times 10^9 \times 2 \text{ bytes} = 14 \text{ GB}$$

**Component 3: Optimizer States.**
AdamW maintains two additional values per parameter: a running mean (first moment) and a running variance (second moment) of the gradients. These are stored in FP32 for numerical stability, plus a master copy of the weights in FP32.

$$\text{Optimizer memory} = 7 \times 10^9 \times (4 + 4 + 4) \text{ bytes} = 84 \text{ GB}$$

**Component 4: Activations.**
During the forward pass, intermediate values must be stored so they can be used during backpropagation. The activation memory depends on the batch size and sequence length, but for a typical training configuration (batch size 32, sequence length 2048), activations can consume 30-60+ GB.

Let us add it up:

| Component | Memory |
|---|---|
| Weights (FP16) | 14 GB |
| Gradients (FP16) | 14 GB |
| Optimizer states (FP32) | 84 GB |
| Activations (estimate) | ~40 GB |
| **Total** | **~152 GB** |

A single A100 has 80 GB of memory. We need roughly **twice** that just for a 7B model. And remember -- 7B is considered a *small* model by today's standards. GPT-3 has 175B parameters. LLaMA-3 405B has 405 billion. The largest models require thousands of GPUs working together.

This is exactly why we need parallelism strategies.


![Memory breakdown of a 7B parameter model showing weights, gradients, optimizer states, and activations.](figures/figure_4.png)
*Memory breakdown for training a 7B parameter model with AdamW. The optimizer states alone exceed the capacity of a single 80 GB GPU.*


## The Five Parallelism Strategies: A Bird's-Eye View

So how do we train a model that does not fit on a single GPU? The answer is that we split the work across multiple GPUs -- but there are different ways to split it. Over the years, researchers have developed five major parallelism strategies, and modern large-scale training systems typically combine several of them simultaneously.

Let us take a quick tour of each one. We will cover each strategy in depth in its own dedicated pod later in this course, but it is important to see the full landscape first.

### 1. Data Parallelism

**The idea:** Every GPU holds a complete copy of the model. We split the training *data* across GPUs.

Imagine you are training with a global batch of 64 samples and you have 8 GPUs. Each GPU gets a mini-batch of 8 samples, runs a full forward and backward pass, computes gradients, and then all GPUs *communicate* to average their gradients. Every GPU then updates its model with the same averaged gradient, keeping all copies in sync.

```python
# Pseudocode for data parallelism
# Each GPU runs this same code, but on different data
local_batch = get_my_shard(global_batch, rank=my_gpu_id)
loss = model(local_batch)
loss.backward()  # compute local gradients

# All-reduce: average gradients across all GPUs
all_reduce(model.gradients, op='mean')

optimizer.step()  # every GPU applies the same update
```

**Advantage:** Simple to implement. PyTorch's `DistributedDataParallel` handles it almost automatically.

**Limitation:** Every GPU must hold the entire model. If the model does not fit in one GPU's memory, data parallelism alone is not enough.

### 2. Tensor Parallelism

**The idea:** Split individual weight matrices (tensors) across GPUs.

Consider a linear layer with weight matrix $\mathbf{W}$ of shape $4096 \times 4096$. Instead of placing the entire matrix on one GPU, we can slice it column-wise across 4 GPUs, giving each GPU a $4096 \times 1024$ slice. Each GPU computes its portion of the matrix multiply, and then the partial results are combined.

**Advantage:** Allows models whose individual layers are too large for one GPU.

**Limitation:** Requires very fast inter-GPU communication (NVLink), because the GPUs must communicate within each layer's computation, not just at the end of a step.

### 3. Pipeline Parallelism

**The idea:** Assign different *layers* of the model to different GPUs.

If a model has 32 transformer layers and we have 4 GPUs, we place layers 1-8 on GPU 0, layers 9-16 on GPU 1, layers 17-24 on GPU 2, and layers 25-32 on GPU 3. Data flows through the pipeline: GPU 0 processes a micro-batch and passes its output to GPU 1, then starts on the next micro-batch while GPU 1 works on the first.

**Advantage:** Allows very deep models. Communication only happens between adjacent stages.

**Limitation:** The naive approach leaves GPUs idle while waiting for data (the "pipeline bubble"). Clever micro-batching schedules (like GPipe and 1F1B) reduce but do not eliminate this idle time.

### 4. Sequence and Context Parallelism

**The idea:** Split the *sequence dimension* across GPUs.

Modern LLMs process very long sequences (4K, 32K, even 128K tokens). The self-attention mechanism has memory that grows quadratically with sequence length: $O(n^2)$ where $n$ is the sequence length. For very long sequences, even the activations for a single layer do not fit on one GPU.

Sequence parallelism splits the sequence across GPUs, with each GPU handling a portion of the tokens. Context parallelism is a closely related strategy specifically for the attention computation, which requires careful handling since every token must attend to every other token.

**Advantage:** Enables training with very long context windows.

**Limitation:** The attention computation requires all tokens to see all other tokens, so careful communication patterns are needed.

### 5. Expert Parallelism

**The idea:** Used with Mixture-of-Experts (MoE) models, where only a subset of "expert" sub-networks are activated for each token.

Models like Mixtral and DeepSeek-V3 have dozens or hundreds of expert feed-forward networks, but each token is routed to only 2-8 of them. Expert parallelism places different experts on different GPUs. Since each token only visits a few experts, each GPU only needs to process a fraction of the tokens.

**Advantage:** Scales model capacity (total parameters) without proportionally scaling compute per token.

**Limitation:** Requires a routing mechanism and careful load balancing to ensure all GPUs stay equally busy.


![Overview diagram of the five parallelism strategies.](figures/figure_5.png)
*The five parallelism strategies at a glance. Modern training systems like Megatron-LM and DeepSpeed combine multiple strategies simultaneously -- this is sometimes called 3D or 5D parallelism.*


## Putting It All Together

Let us summarize what we have covered:

1. **GPUs are massively parallel processors** with thousands of cores designed for throughput, not latency. Their architecture is perfectly matched to the matrix multiplications that dominate deep learning.

2. **The GPU memory hierarchy** (registers, shared memory, L2 cache, HBM) determines how efficiently we can feed those cores. Memory bandwidth is often the bottleneck, not raw compute.

3. **Training a large model requires far more memory than a single GPU provides.** Even a "small" 7B model needs ~152 GB for weights, gradients, optimizer states, and activations -- nearly double the 80 GB available on an A100.

4. **Five parallelism strategies** allow us to distribute the work across many GPUs: Data Parallelism (split data), Tensor Parallelism (split weight matrices), Pipeline Parallelism (split layers), Sequence/Context Parallelism (split the sequence), and Expert Parallelism (split experts in MoE models).

In practice, state-of-the-art training systems combine several of these strategies. For example, Megatron-LM uses tensor parallelism within a node (8 GPUs connected by fast NVLink), pipeline parallelism across nodes, and data parallelism across groups of nodes. Understanding how and why each strategy works -- and when to use which -- is the central goal of this course.

In the next pod, we will look at exactly where GPU memory goes during training. We will break down the four major components -- weights, gradients, optimizer states, and activations -- and learn how to profile and measure each one using PyTorch. This foundational understanding of GPU memory is essential before we can appreciate how the parallelism strategies we just previewed actually save memory.

Let us get started.
