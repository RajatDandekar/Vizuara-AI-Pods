# Pipeline Parallelism

*Split transformer layers across GPUs like an assembly line — from naive schedules to 1F1B, minimizing the pipeline bubble.*

---

## The Assembly Line Analogy

Let us start with an analogy from manufacturing. Imagine a car factory with four stations: welding, painting, assembly, and inspection. If you run the factory with a single car at a time — welding finishes, then painting starts, then assembly, then inspection — three out of four stations are idle at any given moment. That is a 75% waste of capacity. The solution every factory uses is **pipelining**: while station 4 inspects car 1, station 3 assembles car 2, station 2 paints car 3, and station 1 welds car 4. All stations are busy simultaneously.

Pipeline parallelism applies this same idea to training large neural networks. Instead of splitting the data across GPUs (data parallelism) or splitting individual weight matrices (tensor parallelism), we split the **layers** of the model across GPUs. GPU 0 handles the first few transformer layers, GPU 1 handles the next few, and so on. Data flows through the GPUs in sequence, like cars moving through the factory.

The promise is simple: if a model has 32 transformer layers and each layer barely fits in memory, we can split the model across 4 GPUs with 8 layers each. No single GPU needs to hold the entire model. But as we will see, the challenge is not splitting the layers — that is straightforward. The challenge is keeping all GPUs busy at the same time. The idle time in a pipeline is called the **pipeline bubble**, and minimizing it is the central problem of pipeline parallelism.

In this article, we will study four progressively better scheduling strategies: the naive pipeline, GPipe, 1F1B, and interleaved 1F1B. Each one reduces the bubble further.


## Splitting Layers Across GPUs

Suppose we have a transformer with $L$ layers and $P$ GPUs. The simplest approach is to assign $L / P$ consecutive layers to each GPU. If $L = 32$ and $P = 4$:

- **GPU 0**: Layers 1-8
- **GPU 1**: Layers 9-16
- **GPU 2**: Layers 17-24
- **GPU 3**: Layers 25-32

Each group of consecutive layers assigned to a GPU is called a **stage**. During the forward pass, data flows from stage 0 to stage 1 to stage 2 to stage 3. Each stage receives the activation output from the previous stage, runs it through its layers, and sends the result to the next stage. During the backward pass, gradients flow in the opposite direction — from stage 3 back to stage 0.

The key operation between stages is a **point-to-point send/receive**. Unlike allreduce in data parallelism (which involves all GPUs), pipeline communication is between exactly two GPUs — the sender and the receiver. The data transferred is the activation tensor at the boundary between two stages. For a transformer with hidden dimension $h$, sequence length $s$, and micro-batch size $b$, each activation tensor has shape $(b, s, h)$, which in FP16 is:

$$\text{Activation size} = b \times s \times h \times 2 \text{ bytes}$$

For $b = 1$, $s = 4096$, $h = 4096$, this is about 32 MB per transfer — much smaller than the allreduce in data parallelism, which must communicate all gradients. This is one of the advantages of pipeline parallelism: the communication volume is small relative to the computation per stage.

Now the question is: how do we schedule the forward and backward passes across stages?


## The Naive Pipeline and the Bubble Problem

The simplest schedule is the **naive pipeline**. We feed one micro-batch through the entire pipeline — forward pass through all stages, then backward pass through all stages — before starting the next micro-batch.

Let us trace through what happens with $P = 4$ GPUs and a single micro-batch.

**Forward pass:**
- Time step 1: GPU 0 runs forward on layers 1-8. GPUs 1, 2, 3 are idle.
- Time step 2: GPU 1 runs forward on layers 9-16. GPUs 0, 2, 3 are idle.
- Time step 3: GPU 2 runs forward on layers 17-24. GPUs 0, 1, 3 are idle.
- Time step 4: GPU 3 runs forward on layers 25-32 and computes the loss. GPUs 0, 1, 2 are idle.

**Backward pass:**
- Time step 5: GPU 3 runs backward. GPUs 0, 1, 2 are idle.
- Time step 6: GPU 2 runs backward. GPUs 0, 1, 3 are idle.
- Time step 7: GPU 1 runs backward. GPUs 0, 2, 3 are idle.
- Time step 8: GPU 0 runs backward. GPUs 1, 2, 3 are idle.

![The naive pipeline schedule with 4 GPUs: only one GPU is active at a time, creating a massive bubble.](figures/figure_1.png)
*The naive pipeline schedule with 4 GPUs processing a single micro-batch. Only one GPU is active at any given time step, resulting in 75% idle time.*

Each micro-batch takes $2P$ time steps (P forward + P backward), but only **2 of those time steps** involve useful work for any given GPU (one forward, one backward). The idle fraction — the **pipeline bubble** — is:

$$\text{Bubble fraction (naive)} = \frac{P - 1}{P}$$

For $P = 4$ GPUs, the bubble is $3/4 = 75\%$. Three-quarters of the total GPU time is wasted. For $P = 8$, the bubble is $7/8 = 87.5\%$. This is clearly unacceptable.

Let us plug in some concrete numbers. Suppose each stage takes 100 ms for a forward pass and 200 ms for a backward pass (the backward pass is roughly $2\times$ the forward pass because it computes both the gradient with respect to the activations and the gradient with respect to the weights). With 4 GPUs and one micro-batch:

- Total time = $4 \times 100 + 4 \times 200 = 1200$ ms
- Useful work per GPU = $100 + 200 = 300$ ms
- Bubble per GPU = $1200 - 300 = 900$ ms
- Bubble fraction = $900 / 1200 = 75\%$

This tells us that the naive approach with a single micro-batch is hopelessly inefficient. The fix is to send **multiple micro-batches** through the pipeline so that while one GPU is working on micro-batch $k$, another GPU can work on micro-batch $k-1$ or $k+1$.


## GPipe: Micro-Batches to the Rescue

The first major improvement is **GPipe** (Huang et al., 2019). The idea is simple: instead of processing one micro-batch at a time, we split the mini-batch into $M$ micro-batches and pipeline them through the stages. GPipe runs **all forward passes first**, then **all backward passes**.

The schedule works as follows:

1. GPU 0 runs forward on micro-batch 1, then immediately starts forward on micro-batch 2 (while GPU 1 runs forward on micro-batch 1).
2. This continues until all $M$ micro-batches have completed their forward passes through all stages.
3. Then, all $M$ micro-batches run their backward passes in reverse order.

![The GPipe schedule with 4 GPUs and M micro-batches: all forwards first, then all backwards.](figures/figure_2.png)
*GPipe schedule with $P = 4$ stages and $M$ micro-batches. All forward passes are completed before any backward pass begins. The bubble appears at the beginning and end.*

Let us count the bubble. With $P$ stages and $M$ micro-batches, the forward phase takes $P + M - 1$ time steps (it takes $P - 1$ steps for the first micro-batch to "fill" the pipeline, plus $M$ steps for all micro-batches to flow through). Similarly for the backward phase. But the actual work for each GPU is $2M$ time steps ($M$ forward + $M$ backward).

The total number of time steps is:

$$\text{Total time steps} = (P - 1) + M + (P - 1) + M = 2(P - 1 + M)$$

The useful work per GPU is $2M$ time steps. So the bubble fraction is:

$$\text{Bubble fraction (GPipe)} = \frac{2(P - 1)}{2(P - 1 + M)} = \frac{P - 1}{P - 1 + M}$$

Let us plug in numbers. With $P = 4$ and $M = 12$ micro-batches:

$$\text{Bubble fraction} = \frac{4 - 1}{4 - 1 + 12} = \frac{3}{15} = 20\%$$

Compare this to the naive pipeline's 75% bubble. By splitting into 12 micro-batches, we have reduced the bubble from 75% to 20%. If we increase to $M = 32$:

$$\text{Bubble fraction} = \frac{3}{3 + 32} = \frac{3}{35} \approx 8.6\%$$

The trend is clear: as $M$ grows large relative to $P$, the bubble fraction approaches zero. In the limit $M \to \infty$, the bubble vanishes entirely. In practice, $M \geq 4P$ keeps the bubble below 10%.

### The Memory Problem with GPipe

GPipe has an elegant schedule, but it comes with a significant memory cost. During the forward phase, all $M$ micro-batches pass through each stage **before any backward pass begins**. This means every stage must store the activations for all $M$ micro-batches simultaneously — because those activations are needed during the backward pass to compute gradients.

The activation memory per stage in GPipe is:

$$\text{Activation memory (GPipe)} = M \times \text{(activations per micro-batch per stage)}$$

If each micro-batch's activations for a stage cost 100 MB and we have $M = 32$ micro-batches, that is 3.2 GB of activation memory per stage — just for the pipeline. Combined with the model weights, gradients, and optimizer states, this can easily blow up the memory budget.

GPipe addresses this partially by using **activation recomputation** (gradient checkpointing) — discarding intermediate activations and recomputing them during the backward pass. But even with checkpointing, the boundary activations (the outputs passed between stages) must be stored for all $M$ micro-batches, which still grows linearly with $M$.

This brings us to a question: can we start the backward pass earlier, so that we do not need to keep so many activations alive at the same time?


## 1F1B: One Forward, One Backward

The answer is the **1F1B (one-forward-one-backward)** schedule, introduced in PipeDream (Narayanan et al., 2019). The key insight is: instead of running all forwards first and then all backwards, we **interleave** forward and backward passes. As soon as a stage finishes the forward pass for one micro-batch, it can start the backward pass for an earlier micro-batch.

The 1F1B schedule has three phases:

1. **Warmup phase**: Each stage runs forward passes to fill the pipeline, just like GPipe. Stage $k$ runs $P - k$ forward passes.
2. **Steady state (1F1B) phase**: Each stage alternates — one forward pass followed by one backward pass. This is the core of the schedule.
3. **Cooldown phase**: The pipeline drains as the remaining backward passes complete.

![The 1F1B schedule with 4 GPUs: forward and backward passes are interleaved in steady state.](figures/figure_3.png)
*1F1B schedule with $P = 4$ stages and $M$ micro-batches. After the warmup phase, each stage alternates between one forward and one backward pass. The pipeline bubble is the same size as GPipe, but the peak activation memory is dramatically reduced.*

### Why 1F1B Saves Memory

The critical advantage of 1F1B is not in reducing the bubble — the bubble fraction is the same as GPipe, $\frac{P-1}{P-1+M}$. The advantage is in **memory**.

In GPipe, a stage must store activations for all $M$ micro-batches because no backward pass runs until all forward passes complete. In 1F1B, because backward passes start early, a stage only needs to store activations for the micro-batches that have completed forward but not yet backward.

How many micro-batches are "in flight" at any given stage? During steady state, each stage has completed forward passes for some micro-batches and backward passes for others. The maximum number of micro-batches whose activations must be stored at any stage is:

$$\text{Peak activations (1F1B)} = P$$

This is independent of $M$. Whether you process 16 or 1,024 micro-batches, each stage only ever stores at most $P$ micro-batches' worth of activations.

Let us see why. Consider stage $k$ during steady state. It has completed forward passes on some micro-batches but has not yet run backward passes on them. Because backward passes are interleaved with forward passes in a one-for-one rhythm, the backlog of "pending backward" micro-batches is bounded by the pipeline depth $P$.

Let us plug in numbers to make this concrete. With $P = 4$ stages and $M = 32$ micro-batches:

- **GPipe**: Each stage stores activations for $M = 32$ micro-batches. If each costs 100 MB, that is 3,200 MB.
- **1F1B**: Each stage stores activations for at most $P = 4$ micro-batches. That is 400 MB.

This is an **8x reduction** in peak activation memory. For larger $M$, the savings are even more dramatic. This is exactly what we want — we can increase $M$ to shrink the bubble without worrying about memory blowing up.

### 1F1B in Practice

Here is a simplified pseudocode for the 1F1B schedule on stage $k$ (0-indexed) with $M$ micro-batches:

```python
def run_1f1b(stage_id, num_stages, num_microbatches, forward_fn, backward_fn):
    """Run 1F1B pipeline schedule for one stage."""
    P = num_stages
    M = num_microbatches
    k = stage_id

    # Phase 1: Warmup — run (P - 1 - k) extra forward passes
    num_warmup = P - 1 - k
    for i in range(num_warmup):
        forward_fn(microbatch=i)

    # Phase 2: Steady state — alternate 1 forward + 1 backward
    num_steady = M - num_warmup
    for i in range(num_steady):
        forward_fn(microbatch=num_warmup + i)
        backward_fn(microbatch=i)  # backward for earlier microbatch

    # Phase 3: Cooldown — run remaining backward passes
    for i in range(num_warmup):
        backward_fn(microbatch=num_steady + i)
```

Let us trace through stage 0 ($k = 0$) with $P = 4$ and $M = 8$:

- **Warmup**: $P - 1 - 0 = 3$ forward passes (micro-batches 0, 1, 2)
- **Steady state**: $8 - 3 = 5$ iterations, each doing one forward + one backward:
  - Forward(3), Backward(0)
  - Forward(4), Backward(1)
  - Forward(5), Backward(2)
  - Forward(6), Backward(3)
  - Forward(7), Backward(4)
- **Cooldown**: 3 backward passes (micro-batches 5, 6, 7)

At peak, stage 0 has 3 activations stored (after warmup, before any backward starts). After each steady-state step, one activation is consumed (backward) and one is created (forward), keeping the count at 3. During cooldown, the count drops to 0. The peak is $P - 1 = 3$, which is consistent with our claim that peak activations are bounded by $P$.


## Interleaved 1F1B: Reducing the Bubble Further

The 1F1B schedule has the same bubble fraction as GPipe: $\frac{P-1}{P-1+M}$. Can we do better? The answer is yes, using **interleaved 1F1B** (Narayanan et al., 2021), also called the **virtual pipeline** approach.

The idea is clever. Instead of assigning $L/P$ **consecutive** layers to each GPU, we assign each GPU **multiple smaller groups of non-consecutive layers**. Each GPU handles $v$ groups of $L / (P \cdot v)$ layers, where $v$ is the number of **virtual stages** per GPU.

For example, with $L = 32$ layers, $P = 4$ GPUs, and $v = 2$:

- **GPU 0**: Layers 1-4 and Layers 17-20 (virtual stages 0 and 4)
- **GPU 1**: Layers 5-8 and Layers 21-24 (virtual stages 1 and 5)
- **GPU 2**: Layers 9-12 and Layers 25-28 (virtual stages 2 and 6)
- **GPU 3**: Layers 13-16 and Layers 29-32 (virtual stages 3 and 7)

Now the pipeline has $P \times v = 8$ virtual stages instead of 4. Data flows through GPU 0 (layers 1-4), GPU 1 (layers 5-8), GPU 2 (layers 9-12), GPU 3 (layers 13-16), then **back to GPU 0** (layers 17-20), GPU 1 (layers 21-24), GPU 2 (layers 25-28), and GPU 3 (layers 29-32).

![Interleaved 1F1B with non-contiguous layer assignments across 4 GPUs.](figures/figure_4.png)
*Interleaved 1F1B with $v = 2$ virtual stages per GPU. Each GPU handles two non-contiguous groups of layers. The pipeline depth is effectively $P \times v$, but the bubble shrinks because each virtual stage is smaller.*

### Why Does This Help?

The bubble fraction depends on the number of pipeline stages relative to the number of micro-batches. With virtual pipeline stages, the effective number of stages is $P \times v$, but each forward and backward pass through a virtual stage takes only $1/v$ of the time (because it processes $1/v$ as many layers).

The bubble fraction with interleaved 1F1B becomes:

$$\text{Bubble fraction (interleaved)} = \frac{P - 1}{(P - 1 + M) \cdot v}$$

Wait — let us be more precise. The bubble time is determined by the pipeline fill and drain, which costs $(P - 1)$ time units (where one unit is the time for one virtual stage). But each virtual stage's time is $1/v$ of a full stage. So the bubble time in absolute terms is:

$$\text{Bubble time} = (P - 1) \times \frac{t_{\text{stage}}}{v}$$

Meanwhile, the total time per micro-batch through the full pipeline is $P \times v$ virtual stages. The bubble as a fraction of total time is:

$$\text{Bubble fraction (interleaved)} = \frac{P - 1}{v \cdot (P - 1 + M)}$$

This is a factor of $v$ smaller than the standard 1F1B bubble.

Let us plug in numbers. With $P = 4$, $M = 12$, and $v = 2$:

- **Standard 1F1B**: $\frac{3}{3 + 12} = \frac{3}{15} = 20\%$
- **Interleaved 1F1B** ($v = 2$): $\frac{3}{2 \times 15} = \frac{3}{30} = 10\%$

With $v = 4$:

$$\text{Bubble fraction} = \frac{3}{4 \times 15} = \frac{3}{60} = 5\%$$

The bubble shrinks by a factor of $v$. This is a significant improvement.

### The Trade-Off: More Communication

The interleaved schedule does not come for free. Because each GPU now handles non-contiguous layers, the data must make $v$ round trips through the GPUs instead of one. This increases the **number of point-to-point communications** by a factor of $v$. Each communication is smaller (smaller activation tensors from fewer layers), but the total communication volume and latency increase.

In practice, this trade-off is worthwhile when the GPUs are connected by high-bandwidth interconnects (like NVLink within a node). For cross-node communication with slower InfiniBand links, the extra communication overhead may offset the bubble reduction.


## Memory Comparison: GPipe vs. 1F1B

Let us now do a side-by-side comparison of the memory requirements for GPipe and 1F1B, because this is one of the most important practical differences.

Consider a setup with $P = 4$ stages, $M = 16$ micro-batches, and activation memory of 50 MB per micro-batch per stage.

| | GPipe | 1F1B |
|---|---|---|
| **Peak activations per stage** | $M = 16$ micro-batches | $P = 4$ micro-batches |
| **Activation memory per stage** | $16 \times 50 = 800$ MB | $4 \times 50 = 200$ MB |
| **Bubble fraction** | $\frac{3}{19} \approx 15.8\%$ | $\frac{3}{19} \approx 15.8\%$ |
| **Scales with...** | $M$ (bad: larger $M$ = more memory) | $P$ (good: independent of $M$) |

![Memory comparison: GPipe stores M activations while 1F1B stores only P.](figures/figure_5.png)
*Memory comparison between GPipe and 1F1B schedules. GPipe's activation memory grows linearly with the number of micro-batches $M$, while 1F1B keeps memory bounded at $P$ regardless of $M$.*

The implication is profound. With GPipe, there is a tension between the bubble and memory: you want large $M$ to shrink the bubble, but large $M$ increases activation memory. With 1F1B, you can freely increase $M$ to shrink the bubble without any memory penalty. This is exactly what we want.

In practice, most modern pipeline parallel implementations use the 1F1B schedule or its interleaved variant. GPipe is primarily of historical importance — it introduced the idea of micro-batch pipelining, but 1F1B strictly dominates it for training.


## When to Use Pipeline Parallelism

Pipeline parallelism is most useful when:

1. **The model is too large for tensor parallelism alone.** Tensor parallelism works well within a single node (where GPUs are connected by NVLink), but it requires allreduce communication after every transformer block. Pipeline parallelism only requires point-to-point communication between adjacent stages, making it more suitable for **cross-node** parallelism.

2. **You have many transformer layers.** Pipeline parallelism splits layers, so it naturally fits models with many repeated blocks. A 96-layer model can be split into 8 stages of 12 layers each.

3. **Communication bandwidth between nodes is limited.** The per-step communication volume for pipeline parallelism is just one activation tensor (a few tens of MB), while tensor parallelism requires allreduce of the entire activation (potentially hundreds of MB). This makes pipeline parallelism the go-to strategy when GPUs are connected by slower inter-node links.

In the large-scale training systems used for models like GPT-3, LLaMA, and Megatron-Turing NLG, pipeline parallelism is combined with tensor parallelism and data parallelism in a **3D parallelism** setup:

- **Tensor parallelism** within a node (fast NVLink communication)
- **Pipeline parallelism** across nodes (moderate bandwidth needed)
- **Data parallelism** across pipeline-parallel groups (allreduce at the outermost level)

This layered approach ensures that each form of parallelism operates at the communication level it is best suited for.


## Summary

Let us summarize the four pipeline schedules and their trade-offs:

| Schedule | Bubble Fraction | Peak Activations | Key Idea |
|---|---|---|---|
| Naive | $\frac{P-1}{P}$ | 1 micro-batch | One micro-batch at a time |
| GPipe | $\frac{P-1}{P-1+M}$ | $M$ micro-batches | All forwards, then all backwards |
| 1F1B | $\frac{P-1}{P-1+M}$ | $P$ micro-batches | Interleave forward and backward |
| Interleaved 1F1B | $\frac{P-1}{v(P-1+M)}$ | $P$ micro-batches | Non-contiguous layers, $v$ virtual stages |

The progression is clear:

- **Naive pipeline** has an unacceptable bubble of $(P-1)/P$.
- **GPipe** introduces micro-batching to reduce the bubble, but stores $M$ activations.
- **1F1B** achieves the same bubble as GPipe while storing only $P$ activations — decoupling bubble size from memory usage.
- **Interleaved 1F1B** further reduces the bubble by a factor of $v$ at the cost of more communication.

Pipeline parallelism is about partitioning the **depth** of the model across GPUs, complementing tensor parallelism (which partitions the **width**) and data parallelism (which partitions the **data**). Together, these three strategies form the backbone of every large-scale LLM training system in production today.

The final parallelism strategy in our course deals with a different kind of model architecture altogether. Some models, like Mixtral and Switch Transformer, use **Mixture of Experts** — where only a fraction of the model's parameters are activated for each token. Distributing these experts across GPUs introduces its own unique set of challenges. This brings us to **expert parallelism**.
