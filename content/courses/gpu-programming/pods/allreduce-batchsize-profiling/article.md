# Ring-AllReduce, Choosing Batch Size, and TensorBoard GPU Profiling

*How GPUs talk to each other, how to pick the right batch size for distributed training, and how to find the bottlenecks that are slowing everything down.*

---

Let us start with a familiar situation. You are training a model using data parallelism across four GPUs. Each GPU computes gradients on its own mini-batch, and now all four GPUs need to end up with the same averaged gradient so they can take the same optimizer step. The naive approach is obvious: send all gradients to one "master" GPU, average them there, and broadcast the result back. But think about what happens as you scale to 64 GPUs, or 256, or 1,024. That single master GPU becomes a catastrophic bottleneck — every other GPU is sitting idle, waiting for one machine to finish averaging and broadcasting.

This is not a hypothetical problem. It is one of the most important engineering challenges in distributed deep learning. The solution — ring-allreduce — is an elegant algorithm that lets every GPU contribute equally to the communication, with no single bottleneck. It is the backbone of how PyTorch's DistributedDataParallel (DDP) actually works under the hood.

In this article, we will understand ring-allreduce from first principles, learn how to choose the right batch size for distributed training, and then use PyTorch's built-in profiler to find and fix bottlenecks in real training loops.


![Naive allreduce creates a single-GPU bottleneck as all gradients flow to one master.](figures/figure_1.png)
*Ring-AllReduce step-by-step: 4 GPUs in a ring, showing the scatter-reduce phase where each GPU sends and receives partial sums.*


## The Communication Problem in Data Parallelism

Recall from the previous pod that data parallelism replicates the entire model on every GPU. Each GPU processes a different mini-batch, computes local gradients, and then all GPUs must synchronize their gradients before the optimizer step. The operation that accomplishes this synchronization is called **allreduce** — it takes a vector that exists on every GPU, reduces it (typically by summing or averaging), and ensures every GPU ends up with the identical result.

Let us formalize this. Suppose we have $N$ GPUs, and each GPU $i$ holds a gradient vector $\mathbf{g}_i$ of size $D$ (the total number of model parameters). The allreduce operation must compute:

$$\bar{\mathbf{g}} = \frac{1}{N} \sum_{i=0}^{N-1} \mathbf{g}_i$$

and place $\bar{\mathbf{g}}$ on every GPU.

**Naive allreduce.** The simplest approach is to designate GPU 0 as the master. Every other GPU sends its gradient to GPU 0, which sums them, divides by $N$, and broadcasts the result back. The total data transferred through GPU 0 is:

$$\text{Naive traffic at master} = (N-1) \cdot D \cdot 2$$

The factor of 2 accounts for the gather phase (receiving from $N-1$ GPUs) and the broadcast phase (sending to $N-1$ GPUs). As $N$ grows, the master's network bandwidth becomes the bottleneck. The total time scales as $O(N \cdot D)$ — linearly with the number of GPUs. This means adding more GPUs actually makes the communication phase *slower*, which defeats the purpose of scaling.

What we need is an algorithm where the communication time does not grow with $N$.


## Ring-AllReduce: The Bandwidth-Optimal Algorithm

Ring-allreduce solves this by arranging all $N$ GPUs in a logical ring. Each GPU only communicates with its two neighbors — the one before it and the one after it. The algorithm proceeds in two phases: **scatter-reduce** and **allgather**. Each phase takes exactly $N - 1$ steps.

**Setup.** Each GPU's gradient vector $\mathbf{g}_i$ is divided into $N$ equal chunks. We label these chunks $\mathbf{g}_i^{(0)}, \mathbf{g}_i^{(1)}, \ldots, \mathbf{g}_i^{(N-1)}$.

### Phase 1: Scatter-Reduce

In this phase, each GPU accumulates the sum for one particular chunk. After $N - 1$ steps, GPU $k$ will hold the fully reduced (summed) version of chunk $k$.

**Step 1.** Each GPU $i$ sends chunk $(i)$ to GPU $(i+1) \bmod N$, and receives chunk $(i-1) \bmod N$ from GPU $(i-1) \bmod N$. Upon receiving, each GPU adds the received chunk to its own corresponding chunk.

**Step 2.** Each GPU sends the chunk it just updated (the partially reduced chunk) to its next neighbor, and receives and accumulates from the previous neighbor.

This continues for $N - 1$ steps. At the end, each GPU holds one fully reduced chunk — GPU $k$ holds $\sum_{i=0}^{N-1} \mathbf{g}_i^{(k)}$.

Let us trace through a concrete example with $N = 4$ GPUs and a gradient vector split into 4 chunks. We label the chunks with the GPU index as subscript and the chunk index as superscript.

```
Initial state:
  GPU 0: [g0_0, g0_1, g0_2, g0_3]
  GPU 1: [g1_0, g1_1, g1_2, g1_3]
  GPU 2: [g2_0, g2_1, g2_2, g2_3]
  GPU 3: [g3_0, g3_1, g3_2, g3_3]

Step 1: GPU i sends chunk i to GPU (i+1)%4
  GPU 0 sends chunk 0 → GPU 1;  GPU 1 accumulates: chunk 0 = g0_0 + g1_0
  GPU 1 sends chunk 1 → GPU 2;  GPU 2 accumulates: chunk 1 = g1_1 + g2_1
  GPU 2 sends chunk 2 → GPU 3;  GPU 3 accumulates: chunk 2 = g2_2 + g3_2
  GPU 3 sends chunk 3 → GPU 0;  GPU 0 accumulates: chunk 3 = g3_3 + g0_3

Step 2: Each GPU sends its just-updated chunk to the next neighbor
  GPU 1 sends chunk 0 (g0+g1) → GPU 2;  GPU 2 accumulates: chunk 0 = g0_0+g1_0+g2_0
  GPU 2 sends chunk 1 (g1+g2) → GPU 3;  GPU 3 accumulates: chunk 1 = g1_1+g2_1+g3_1
  GPU 3 sends chunk 2 (g2+g3) → GPU 0;  GPU 0 accumulates: chunk 2 = g2_2+g3_2+g0_2
  GPU 0 sends chunk 3 (g3+g0) → GPU 1;  GPU 1 accumulates: chunk 3 = g3_3+g0_3+g1_3

Step 3: One more round to complete the reduce
  GPU 2 sends chunk 0 (g0+g1+g2) → GPU 3;  GPU 3: chunk 0 = g0+g1+g2+g3  ✓
  GPU 3 sends chunk 1 (g1+g2+g3) → GPU 0;  GPU 0: chunk 1 = g1+g2+g3+g0  ✓
  GPU 0 sends chunk 2 (g2+g3+g0) → GPU 1;  GPU 1: chunk 2 = g2+g3+g0+g1  ✓
  GPU 1 sends chunk 3 (g3+g0+g1) → GPU 2;  GPU 2: chunk 3 = g3+g0+g1+g2  ✓

After scatter-reduce:
  GPU 0 holds the fully reduced chunk 1
  GPU 1 holds the fully reduced chunk 2
  GPU 2 holds the fully reduced chunk 3
  GPU 3 holds the fully reduced chunk 0
```

Notice the key property: at every step, every GPU sends exactly one chunk and receives exactly one chunk. All communication happens simultaneously. No GPU is idle, and no GPU is overloaded.


![The allgather phase distributes the fully reduced chunks so every GPU has the complete result.](figures/figure_2.png)
*Ring-AllReduce allgather phase: each GPU already holds one fully reduced chunk and broadcasts it around the ring until every GPU has all chunks.*


### Phase 2: Allgather

After the scatter-reduce phase, each GPU holds exactly one fully reduced chunk, but it needs all $N$ chunks. The allgather phase is structurally identical to scatter-reduce, except instead of accumulating (adding), each GPU simply replaces its chunk with the received one.

In $N - 1$ more steps, the fully reduced chunks propagate around the ring until every GPU holds the complete, fully reduced gradient vector.

### Why is Ring-AllReduce Bandwidth-Optimal?

Let us count the total data transferred. In each of the two phases, every GPU sends one chunk of size $D/N$ per step, for $N - 1$ steps. The total data sent per GPU across both phases is:

$$\text{Data per GPU} = 2 \cdot (N-1) \cdot \frac{D}{N}$$

As $N$ grows large, this approaches $2D$ — the total data transferred per GPU is essentially independent of the number of GPUs. Compare this to the naive approach where the master transfers $2(N-1) \cdot D$ data. The ring-allreduce communication time is:

$$T_{\text{ring}} = 2(N-1) \cdot \frac{D}{N \cdot B}$$

where $B$ is the bandwidth of each link. This simplifies to approximately $\frac{2D}{B}$ for large $N$ — the communication time is determined by the total gradient size and the per-link bandwidth, not by the number of GPUs. This is provably optimal; no algorithm can do better given that every GPU must send and receive $D$ elements worth of information.

This is exactly what we want for scaling. Adding more GPUs does not significantly increase communication time — it primarily increases the total throughput.


## Choosing the Right Batch Size

Now that we understand how gradients are synchronized, let us think about what happens to the effective batch size as we scale. With $N$ GPUs each processing a local batch of size $b$, the effective global batch size is:

$$B_{\text{global}} = N \cdot b$$

If you have 8 GPUs each processing 32 samples, your effective batch size is 256. Scale to 64 GPUs and it becomes 2,048. Scale to 256 GPUs and it becomes 8,192. This raises a critical question: can you just keep increasing the batch size indefinitely?

The answer is no. There is a fundamental trade-off.


![Learning rate scaling rules for large batch training: linear scaling and square root scaling.](figures/figure_3.png)
*Learning rate vs. batch size scaling rules: linear scaling (Goyal et al., 2017) and square root scaling (Hoffer et al., 2017) with warmup.*


### The Critical Batch Size

Research by McCandlish et al. (2018) introduced the concept of a **critical batch size** $B_{\text{crit}}$. Below this threshold, doubling the batch size roughly halves the number of steps needed to reach a given loss — a nearly perfect speedup. Above this threshold, doubling the batch size gives diminishing returns: you use more compute but do not train proportionally faster.

The critical batch size is related to the ratio of the gradient signal to gradient noise:

$$B_{\text{crit}} \approx \frac{B_{\text{noise}}}{1}$$

where $B_{\text{noise}}$ is the "gradient noise scale" — a measure of how much the stochastic gradient varies across different mini-batches. For a given model and dataset, this is roughly constant. For many language models, $B_{\text{crit}}$ is in the range of millions of tokens, which translates to batch sizes in the hundreds or low thousands.

### Learning Rate Scaling Rules

When you increase the batch size, you typically need to adjust the learning rate. The two most widely used rules are:

**Linear scaling (Goyal et al., 2017).** If you multiply the batch size by $k$, multiply the learning rate by $k$:

$$\eta_{\text{large}} = k \cdot \eta_{\text{base}}$$

The intuition is that with a $k\times$ larger batch, each gradient estimate is $k\times$ more accurate (lower variance), so you can take a $k\times$ larger step. This works well when the loss surface is locally smooth and the batch size is not too extreme.

**Square root scaling (Hoffer et al., 2017).** Multiply the learning rate by $\sqrt{k}$:

$$\eta_{\text{large}} = \sqrt{k} \cdot \eta_{\text{base}}$$

This is more conservative and tends to work better for very large batch sizes where linear scaling leads to training instability.

**Warmup.** Regardless of the scaling rule, large-batch training almost always requires a **warmup period** — starting with a small learning rate and linearly increasing it to the target over the first few hundred or thousand steps. Without warmup, the large learning rate combined with the random initial weights causes the optimizer to take dangerously large steps early in training, often leading to divergence.

A practical warmup schedule looks like:

$$\eta(t) = \begin{cases} \eta_{\text{target}} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\ \eta_{\text{target}} & \text{if } t \geq T_{\text{warmup}} \end{cases}$$

where $T_{\text{warmup}}$ is typically 1-5% of total training steps.

### The Practical Trade-Off

The decision of batch size is ultimately a trade-off between three factors:

1. **Statistical efficiency** (samples per unit of learning): Smaller batches have noisier gradients but each step provides more "unique" information per sample processed.
2. **Throughput** (samples per second): Larger batches process more samples per second because they better utilize GPU parallelism and reduce the relative communication overhead.
3. **Communication overhead**: With data parallelism, every step requires an allreduce. Larger local batches mean fewer allreduce operations per epoch.

The sweet spot is usually at or slightly below the critical batch size — large enough to keep all GPUs busy, but not so large that you waste compute on diminishing returns.

```python
# Example: Adjusting learning rate for distributed training
import torch

def get_scaled_lr(base_lr, base_batch_size, global_batch_size, rule='linear'):
    """Scale learning rate for larger batch sizes."""
    k = global_batch_size / base_batch_size
    if rule == 'linear':
        return base_lr * k
    elif rule == 'sqrt':
        return base_lr * (k ** 0.5)
    else:
        raise ValueError(f"Unknown rule: {rule}")

# Base configuration: lr=1e-3, batch_size=256
base_lr = 1e-3
base_bs = 256

# Scaling to 8 GPUs, local batch 256 -> global batch 2048
global_bs = 8 * 256
lr_linear = get_scaled_lr(base_lr, base_bs, global_bs, rule='linear')
lr_sqrt   = get_scaled_lr(base_lr, base_bs, global_bs, rule='sqrt')

print(f"Base LR: {base_lr}")
print(f"Global batch size: {global_bs}")
print(f"Linear scaling LR: {lr_linear}")    # 8e-3
print(f"Sqrt scaling LR:   {lr_sqrt:.6f}")  # ~2.83e-3
```


## GPU Profiling with PyTorch Profiler and TensorBoard

You have set up data parallelism with ring-allreduce, chosen your batch size, and scaled your learning rate. Training is running — but is it running *well*? How do you know if your GPUs are actually computing most of the time, or if they are spending half their time waiting for data loading, communication, or memory operations?

This is where profiling comes in.


![A TensorBoard profiler trace showing compute, communication, and idle time for a distributed training step.](figures/figure_4.png)
*TensorBoard profiler trace showing compute/communication overlap in a distributed training step.*


### Using torch.profiler

PyTorch ships with a built-in profiler that can capture detailed timing information for every operation on both CPU and GPU. Here is how to use it:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Profile 5 training steps after a 2-step warmup
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        with record_function("forward"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        with record_function("backward"):
            optimizer.zero_grad()
            loss.backward()

        with record_function("optimizer"):
            optimizer.step()

        prof.step()  # Signal the profiler that a step is complete
        if step >= 10:
            break
```

After running this, you can visualize the trace in TensorBoard:

```bash
tensorboard --logdir=./log/profiler
```

### Reading the Profiler Trace

The profiler trace is a timeline that shows exactly what each CPU thread and GPU stream was doing at every microsecond. Here is what to look for:

**Compute-bound.** If the GPU utilization bars are mostly filled with CUDA kernels (matmul, convolution, etc.) and there are few gaps, your training is compute-bound. This is the best case — your GPUs are doing useful work most of the time. Optimization here means using faster operations (flash attention, fused kernels).

**Memory-bound.** If you see many small kernels with gaps between them, and the profiler shows significant time in memory operations (cudaMemcpy, allocation), your training is memory-bound. The GPU is waiting for data to arrive from memory rather than computing. Solutions include increasing the arithmetic intensity (larger tiles in matmuls), using mixed precision to reduce memory traffic, and kernel fusion to reduce intermediate memory reads/writes.

**Communication-bound.** In distributed training, you will see NCCL (NVIDIA Collective Communications Library) operations like ncclAllReduce in the trace. If these operations dominate the timeline, your training is communication-bound. Solutions include gradient bucketing, overlapping communication with computation, and reducing the gradient size (gradient compression, mixed precision).

### Overlapping Communication with Computation

One of the most effective optimizations in distributed training is overlapping the allreduce communication with the backward pass computation. The key insight is that gradients become available layer by layer during backpropagation — you do not need to wait for all gradients before starting communication.

PyTorch's `DistributedDataParallel` does this automatically through **gradient bucketing**. It groups gradients into buckets (default size: 25 MB) and launches the allreduce for each bucket as soon as all gradients in that bucket are ready. While the allreduce for early layers is in progress over the network, the backward pass is still computing gradients for later layers on the GPU.

```python
# DDP automatically handles bucketing and overlap
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=25,  # Size of gradient buckets in MB
)
# During backward(), DDP triggers allreduce for each bucket
# as soon as all gradients in that bucket are computed
```

When this overlap works well, the communication is almost entirely hidden behind the computation. The profiler trace will show NCCL operations running on a separate CUDA stream concurrently with backward pass kernels. When it does not overlap well — for example, if your model has one very large layer at the end — you will see the GPU go idle while waiting for communication to finish.


![Communication overhead scaling: ring-allreduce vs naive allreduce as the number of GPUs increases.](figures/figure_5.png)
*Communication overhead scaling with the number of GPUs: ring-allreduce maintains near-constant per-GPU cost, while naive allreduce grows linearly.*


### Practical Profiling Tips

1. **Always profile with realistic data.** Synthetic data may hide data loading bottlenecks that dominate real training.

2. **Profile at the target scale.** Communication patterns at 4 GPUs are very different from 64 GPUs. Profile at (or close to) your target scale.

3. **Look at the GPU utilization percentage.** Anything below 80% means there is significant optimization potential. World-class training pipelines achieve 50-60% MFU (Model FLOPS Utilization) — meaning even the best setups leave room for improvement.

4. **Check the data loader.** A common bottleneck is the CPU data pipeline. If the GPU trace shows regular gaps at the start of each step (before the forward pass), your data loading is the bottleneck. Use `num_workers > 0` in your DataLoader and consider `pin_memory=True`.

5. **Use `record_function` liberally.** Custom annotations make the trace much easier to read. Wrap your forward pass, backward pass, optimizer step, and any custom operations.


## Gradient Bucketing in Practice

Let us look at gradient bucketing more carefully, because it is the mechanism that makes communication overlap possible in PyTorch DDP.

Without bucketing, DDP would either (a) launch a separate allreduce for every single parameter tensor, which has enormous per-operation overhead, or (b) wait for all gradients and do one massive allreduce, which prevents any overlap.

Bucketing is the middle ground. Parameters are assigned to buckets in reverse order of their appearance in the model (matching the order in which gradients are computed during backpropagation). When the last gradient in a bucket is computed, that bucket's allreduce is launched immediately.

The bucket size is a tuning parameter. Smaller buckets allow more overlap but have higher per-operation overhead. Larger buckets have less overhead per byte but delay the start of communication. The default of 25 MB works well for most models, but you may want to tune it:

```python
# Smaller buckets: more overlap potential, more overhead
model = DistributedDataParallel(model, bucket_cap_mb=10)

# Larger buckets: less overhead, less overlap
model = DistributedDataParallel(model, bucket_cap_mb=50)
```

## Putting It All Together

Here is a checklist for efficient distributed training:

1. **Use ring-allreduce** (the default in PyTorch DDP via NCCL). Never implement naive allreduce.
2. **Choose your batch size** near the critical batch size. Scale the learning rate using linear or sqrt scaling, and always use warmup.
3. **Profile your training loop** with `torch.profiler`. Look at GPU utilization, identify whether you are compute-bound, memory-bound, or communication-bound.
4. **Maximize overlap** between communication and computation. Tune the bucket size if needed.
5. **Use mixed precision** (`torch.cuda.amp`) to reduce both computation time and communication volume — FP16 gradients are half the size of FP32.

## What Comes Next

We now understand how data parallel GPUs communicate. Ring-allreduce gives us bandwidth-optimal gradient synchronization. Proper batch size selection and learning rate scaling let us train efficiently at scale. And profiling tells us where the remaining bottlenecks are.

But data parallelism replicates the entire model on every GPU — the weights, the gradients, and the optimizer states. What if the model is so large that it does not fit on a single GPU, even with a batch size of 1? What if the optimizer states alone consume more memory than your GPU has?

This brings us to ZeRO — the Zero Redundancy Optimizer — which takes the elegant observation that data parallelism wastes enormous amounts of memory by storing redundant copies of the same data across all GPUs, and eliminates that redundancy one piece at a time.
