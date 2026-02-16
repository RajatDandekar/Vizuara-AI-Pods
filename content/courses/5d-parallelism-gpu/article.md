# 5D Parallelism: How We Train Models That Don't Fit on a Single GPU

*Understanding Data, Tensor, Pipeline, Sequence, and Expert Parallelism — the five dimensions that power today's largest AI models*

---

## The "One GPU Is Not Enough" Problem

Let us start with a simple question. How much memory does a GPU have?

NVIDIA's most powerful training GPU, the H100, has 80 GB of memory. That sounds like a lot — until you try to train a large language model on it.

Let us take a concrete example. Meta's LLaMA 3 405B model has 405 billion parameters. Just storing these parameters in 16-bit floating point requires about 810 GB. But that is only the beginning. During training with the Adam optimizer, we also need to store:

- The gradients: another 810 GB
- The optimizer states (momentum and variance in fp32): about 3.2 TB

That brings us to roughly **4.8 terabytes** of memory — just for the model, before we even think about the activations from the forward pass.

A single H100 has 80 GB. We would need **60 GPUs** just to hold the model in memory — and we have not started any computation yet.

![GPU memory vs LLaMA 3 405B training memory requirements](figures/figure_1.png)
*A single H100 GPU has 80 GB of memory. Training LLaMA 3 405B requires ~4,860 GB — that is 60x more.*

So, how do we train a model that is 60 times larger than what a single GPU can hold?

The answer is **parallelism** — splitting the work across thousands of GPUs. But not just in one way. Modern training systems split the work along **five independent dimensions**, each targeting a different bottleneck. This is called **5D Parallelism**.

---

## The Big Picture: Five Dimensions

Before we dive into each dimension, let us get a bird's-eye view. The five dimensions of parallelism are:

| Dimension | What It Splits |
|---|---|
| **Data Parallelism (DP)** | The training data (mini-batch) |
| **Tensor Parallelism (TP)** | Weight matrices within a single layer |
| **Pipeline Parallelism (PP)** | Layers of the model across stages |
| **Context Parallelism (CP)** | Activations along the sequence dimension |
| **Expert Parallelism (EP)** | Experts in Mixture-of-Experts layers |

The key insight is that these dimensions are **orthogonal** — they can be combined multiplicatively. The total number of GPUs factors cleanly as:

![Equation](equations/eq_1.png)

Let us plug in some real numbers. LLaMA 3 was trained on **16,384 H100 GPUs** with the following configuration:

![Equation](equations/eq_2.png)

Each dimension targets a different bottleneck, and each has its own communication pattern and trade-offs.

![5D parallelism overview](figures/figure_2.png)
*The five dimensions of parallelism — TP within a node, PP across nodes, DP as the outermost dimension, with CP and EP as additional overlays.*

Now let us understand each dimension one by one.

---

## Dimension 1: Data Parallelism (DP)

Let us start with the simplest and most widely used form of parallelism.

Imagine you are a professor who needs to grade 1,000 exam papers. You have the answer key. Instead of grading all 1,000 yourself, you make 10 copies of the answer key and give them to 10 teaching assistants. Each TA grades 100 papers independently. At the end, all 10 TAs meet to reconcile their grading and make sure they are aligned.

This is exactly how data parallelism works.

In data parallelism, every GPU gets a **complete copy of the model**. The training data (the mini-batch) is split equally across all GPUs. Each GPU performs the forward pass and backward pass on its own shard of data. After the backward pass, the gradients are **averaged across all GPUs** using a collective communication operation called **all-reduce**.

![Data parallelism](figures/figure_3.png)
*Data Parallelism: Each GPU holds a full copy of the model. The mini-batch is split across GPUs, and gradients are synchronized via all-reduce.*

Mathematically, the gradient update after all-reduce looks as follows:

![Equation](equations/eq_3.png)

where $g_i$ is the gradient computed by GPU $i$, and $N_d$ is the number of data-parallel GPUs.

Let us plug in some simple numbers. Suppose we have 4 GPUs and a global batch size of 32. Each GPU processes 8 samples. If GPU 1 computes a gradient of 0.5, GPU 2 computes 0.3, GPU 3 computes 0.7, and GPU 4 computes 0.1, then:

![Equation](equations/eq_4.png)

Every GPU then uses this averaged gradient $g = 0.4$ to update its model weights. Since they all start with the same weights and apply the same update, the models remain identical. This is exactly what we want.

But there is a problem. With naive data parallelism, every GPU stores:

![Equation](equations/eq_5.png)

where $\Psi$ is the number of model parameters.

Let us compute this for a 7 billion parameter model:

![Equation](equations/eq_6.png)

A 7B model already exceeds a single H100's 80 GB memory — and we have not even counted activations! Every GPU stores a **full redundant copy** of everything. This is very wasteful.

This brings us to ZeRO.

### ZeRO: Eliminating the Redundancy

The insight behind **ZeRO** (Zero Redundancy Optimizer), introduced by Rajbhandari et al. in 2020, is beautifully simple: if we have $N_d$ GPUs all storing the same optimizer states, gradients, and parameters, why not **shard** them so each GPU only stores $1/N_d$ of each?

ZeRO comes in three stages:

**ZeRO Stage 1** — Shard the optimizer states. Each GPU stores only $1/N_d$ of the momentum and variance.

![Equation](equations/eq_7.png)

**ZeRO Stage 2** — Additionally shard the gradients.

![Equation](equations/eq_8.png)

**ZeRO Stage 3 (FSDP)** — Shard everything: parameters, gradients, and optimizer states.

![Equation](equations/eq_9.png)

Let us plug in numbers for our 7B model with 4 GPUs:

- **Naive DP:** 112 GB per GPU (does not fit on H100!)
- **ZeRO Stage 1:** $4 \times 7B + 12 \times 7B / 4$ = 28 + 21 = **49 GB** per GPU
- **ZeRO Stage 2:** $2 \times 7B + 14 \times 7B / 4$ = 14 + 24.5 = **38.5 GB** per GPU
- **ZeRO Stage 3:** $16 \times 7B / 4$ = **28 GB** per GPU

![ZeRO Stages](figures/figure_4.png)
*ZeRO Stages 1, 2, and 3: Progressively sharding optimizer states, gradients, and parameters across GPUs reduces per-GPU memory from 112 GB down to 28 GB for a 7B model with 4 GPUs.*

ZeRO Stage 3 gives us a linear memory reduction with the number of GPUs. Not bad right?

PyTorch's **FSDP** (Fully Sharded Data Parallelism) is essentially the native implementation of ZeRO Stage 3 within the PyTorch framework.

But data parallelism alone — even with ZeRO — has a limitation. It does not reduce the per-GPU compute for each forward and backward pass. For that, we need to split the model itself.

---

## Dimension 2: Tensor Parallelism (TP)

Now let us look at a fundamentally different approach. Instead of replicating the model and splitting the data, what if we **split the model's weight matrices** across GPUs?

Imagine you have a very large multiplication problem: multiplying a 4096×4096 matrix by a vector. Instead of one person doing all 16 million multiplications, you split the matrix into 4 vertical strips of 4096×1024 and give each strip to a different person. Each person multiplies their strip with the input vector, producing a partial result. The partial results are then combined to get the final answer.

This is tensor parallelism, introduced by Shoeybi et al. in the **Megatron-LM** paper (2019).

The key idea is to split weight matrices in two ways:

**Column-Parallel:** Split the weight matrix $W$ into columns across $t$ GPUs:

![Equation](equations/eq_10.png)

Each GPU $i$ computes $Y_i = X \cdot W_i$, where $X$ is the full input (replicated). The outputs $Y_i$ are partial results.

**Row-Parallel:** Split the weight matrix $W$ into rows. The input is already partitioned from the preceding column-parallel layer. Each GPU computes a partial result, and an **all-reduce** sums them to get the final output.

Let us see this with concrete numbers. Suppose we have a weight matrix of shape 4096×4096 and we use TP = 4 GPUs.

**Column-parallel split:**
- Each GPU stores a 4096×1024 slice of $W$
- Input $X$ has shape (batch, seq, 4096) — replicated on all GPUs
- GPU 0 computes $Y_0 = X \cdot W_0$ → output shape (batch, seq, 1024)
- GPU 1 computes $Y_1 = X \cdot W_1$ → output shape (batch, seq, 1024)
- ... and so on

The memory per GPU drops from $4096 \times 4096 = 16.8M$ parameters to $4096 \times 1024 = 4.2M$ parameters — a **4x reduction**. This is exactly what we want.

In a Transformer layer, TP is applied as follows:
- The **attention block** splits Q, K, V projections column-wise — each GPU gets a subset of attention heads. The output projection is row-parallel.
- The **MLP block** uses column-parallel for the first linear layer (the "up-projection") and row-parallel for the second (the "down-projection").

![Tensor Parallelism in a Transformer layer](figures/figure_5.png)
*Tensor Parallelism (TP=4): Q, K, V matrices are split column-wise across 4 GPUs. The MLP up-projection is column-parallel, the down-projection is row-parallel. All-reduce synchronizes after each block.*

The cost of tensor parallelism is **communication**. Each Transformer layer requires **4 all-reduce operations** — 2 in the forward pass and 2 in the backward pass. The volume of each all-reduce is:

![Equation](equations/eq_11.png)

where $b$ is the micro-batch size, $s$ is the sequence length, $h$ is the hidden dimension, and $t$ is the TP degree.

Let us plug in numbers for a single layer of LLaMA 3 405B: $b = 1$, $s = 8192$, $h = 16384$, $t = 8$:

![Equation](equations/eq_12.png)

With 4 all-reduces per layer and 126 layers, that is over **112 GB of communication per training step** — just for TP! This is why tensor parallelism must be confined to GPUs within a single node connected by **NVLink**, which provides 900 GB/s of bandwidth on the H100.

---

## Dimension 3: Pipeline Parallelism (PP)

Now let us understand a third approach. What if instead of splitting weight matrices, we split the model **layer by layer** across GPUs?

Think of an assembly line in a car factory. Station 1 welds the frame, Station 2 installs the engine, Station 3 paints the body, Station 4 does the final inspection. Each station works on a different car at the same time. While Station 1 welds Car 2's frame, Station 2 is installing Car 1's engine, and so on.

In pipeline parallelism, we partition the model's layers into **stages**. If we have 32 layers and 4 GPUs, each GPU gets 8 consecutive layers. Data flows through the pipeline as **micro-batches** — small slices of the original mini-batch.

But here is the catch. With a naive approach, there is a serious problem: the **pipeline bubble**.

When the pipeline first starts, only Stage 1 has work to do. Stages 2, 3, and 4 are idle, waiting for Stage 1 to finish. Similarly, at the end, only Stage 4 is working while the others are idle. This wasted time is called the **bubble**.

The bubble fraction for the basic 1F1B (one-forward-one-backward) schedule is:

![Equation](equations/eq_13.png)

where $p$ is the number of pipeline stages and $m$ is the number of micro-batches.

Let us plug in some numbers. With $p = 4$ stages and $m = 16$ micro-batches:

![Equation](equations/eq_14.png)

This means 15.8% of the total GPU time is wasted in the bubble. To keep this manageable, we need $m \gg p$.

![1F1B Pipeline Schedule](figures/figure_6.png)
*1F1B Pipeline Schedule: Forward passes (blue) and backward passes (orange) alternate after the warm-up phase. The idle bubble regions are shown in gray.*

The solution to large bubbles is **interleaved scheduling**. Instead of each GPU holding one contiguous block of layers, each GPU holds $v$ **virtual stages** — non-contiguous chunks of layers. This reduces the bubble by a factor of $v$:

![Equation](equations/eq_15.png)

With $v = 4$ virtual stages per GPU, our bubble drops from 15.8% to:

![Equation](equations/eq_16.png)

Much better!

![Pipeline Schedule Comparison](figures/figure_7.png)
*Comparison of pipeline schedules: GPipe has large bubbles, 1F1B reduces them, and Interleaved 1F1B with virtual stages shrinks the bubble fraction to just 4.5%.*

The communication cost of pipeline parallelism is relatively light — just **point-to-point sends** of the activation tensor between adjacent stages. The volume per micro-batch is $b \times s \times h$, which for LLaMA 3 is about $1 \times 8192 \times 16384 \approx 128$ MB. This can be handled over InfiniBand between nodes without much trouble.

---

## Dimension 4: Context / Sequence Parallelism (CP)

So far, we have been splitting the model (TP, PP) and the data (DP). But there is another dimension that becomes critical when training with very long sequences: **splitting the sequence itself**.

Imagine you need to read a 500-page book and write a comprehensive summary. Instead of reading all 500 pages yourself, you give 100 pages to each of 5 people. But here is the catch — in a Transformer's attention mechanism, every token needs to "look at" every other token. So every reader needs access to what the other readers have seen.

The problem is that attention memory scales **quadratically** with sequence length. The attention score matrix has shape $(s \times s)$, where $s$ is the sequence length. Going from a context of 8K tokens to 128K tokens means $16\times$ the sequence length, but $256\times$ the attention memory!

This is where Context Parallelism comes in. The idea is to **shard the sequence across GPUs** so that each GPU only processes a chunk of the full sequence.

The most elegant implementation is **Ring Attention**. Here is how it works:

1. Each GPU holds a chunk of the query tokens $Q_i$ — this chunk stays fixed
2. The key and value blocks ($K$ and $V$) **rotate around a ring** of GPUs
3. At each step, a GPU computes attention between its local queries and the current K/V block, while simultaneously sending the K/V block to the next GPU and receiving a new one
4. After CP steps, every GPU has attended to all K/V blocks

The communication is **overlapped with computation**, so the ring rotation is nearly free.

![Ring Attention](figures/figure_8.png)
*Ring Attention: Each GPU holds a fixed chunk of query tokens. Key/Value blocks rotate clockwise around the ring, with communication overlapped behind computation.*

With CP, the activation memory per GPU is reduced proportionally:

![Equation](equations/eq_17.png)

Let us see the impact with numbers. For a 128K context with hidden dimension 16384:

- **Without CP:** The attention score matrix alone is $128,000 \times 128,000 \times 2$ bytes $\approx 30.5$ GB (in fp16) — per layer!
- **With CP = 16:** Each GPU handles $128,000 / 16 = 8,000$ query tokens, and the attention chunk is $8,000 \times 128,000 \times 2 \approx 1.9$ GB per layer.

This is exactly what makes long-context training feasible. LLaMA 3 used CP = 16 during its 128K context extension phase.

There is also a closely related concept called **Sequence Parallelism (SP)** from the Megatron-LM framework. SP works hand-in-hand with Tensor Parallelism — it shards the **non-tensor-parallel operations** (LayerNorm, Dropout) along the sequence dimension. This saves activation memory proportional to the TP degree without adding any extra communication. In modern systems, SP is always enabled whenever TP is used.

---

## Dimension 5: Expert Parallelism (EP)

The fifth and final dimension applies specifically to **Mixture-of-Experts (MoE)** models.

Think of a hospital. Instead of one general practitioner who tries to treat everything, you have specialists — a cardiologist, a neurologist, an orthopedist, a dermatologist. When a patient arrives, a **triage nurse** (the router) evaluates the symptoms and sends the patient to the most appropriate specialist.

In a MoE model, the standard feedforward (MLP) layer is replaced by multiple **expert networks** — each an independent MLP. A **gating network** (the router) looks at each token and decides which top-$k$ experts should process it.

For example, DeepSeek-V3 has **256 routed experts** with top-8 routing — each token is processed by 8 out of 256 experts. The total model has 671 billion parameters, but only 37 billion are activated for any given token.

With Expert Parallelism, we distribute the experts across GPUs. If we have 256 experts and EP = 64 GPUs, each GPU holds $256 / 64 = 4$ experts.

The communication pattern is **all-to-all**: tokens that are routed to experts on other GPUs must be sent there for processing, and the results must be sent back. Each MoE layer requires two all-to-all operations — one to dispatch tokens, one to combine results.

![MoE Expert Parallelism](figures/figure_9.png)
*Expert Parallelism: A gating network routes each token to the appropriate experts. Tokens are dispatched via all-to-all communication, processed by distributed experts, and combined back.*

The key advantage of MoE + EP is that it allows **massive model capacity** (671B parameters) while keeping the **per-token compute** manageable (37B activated). Expert parallelism makes it possible to distribute these hundreds of experts across many GPUs efficiently.

---

## Putting It All Together: The 5D Configuration

Now we have all five pieces of the puzzle. The question is: how do we choose the right configuration?

The answer comes from understanding the **communication hierarchy** of a GPU cluster. Not all GPUs are connected equally:

- **Within a node (8 GPUs):** Connected by NVLink/NVSwitch — ~900 GB/s bandwidth
- **Across nodes:** Connected by InfiniBand — ~400 GB/s bandwidth (fast, but 2x slower than NVLink)
- **Across racks:** Connected by network switches — bandwidth drops further

The rule of thumb is: **place the most communication-hungry parallelism on the fastest interconnect.**

1. **TP (most bandwidth-hungry)** → Within a single node (NVLink)
2. **CP (moderate bandwidth)** → Can span nodes but prefers fast links
3. **PP (latency-tolerant)** → Across nodes (point-to-point only)
4. **DP (most tolerant)** → Outermost dimension (gradient sync can be overlapped)
5. **EP (MoE layers only)** → Applied independently to expert layers

![Cluster Hierarchy](figures/figure_10.png)
*GPU cluster hierarchy: NVLink connects GPUs within a node (900 GB/s), InfiniBand connects nodes (400 GB/s). TP maps to the fastest link, PP across nodes, DP as the outermost dimension.*

Let us walk through two real-world configurations.

**LLaMA 3 405B** (16,384 H100 GPUs):
- TP = 8 (within each 8-GPU node)
- PP = 16 (across 16 nodes — model split into 16 stages of ~8 layers each)
- DP = 128 (128 data-parallel replicas)
- CP = 1 for 8K pre-training; CP = 16 for 128K context extension
- Total: $128 \times 8 \times 16 = 16,384$ GPUs

**DeepSeek-V3** (2,048 H800 GPUs):
- PP = 16 (DualPipe algorithm for near-zero bubbles)
- EP = 64 (experts distributed across 8 nodes)
- DP with ZeRO Stage 1 (for the remaining factor)
- No TP — the MoE architecture distributes parameters via EP instead
- Total: 2,048 GPUs, trained for only **$5.57 million** — remarkably efficient

---

## Memory Math: A Complete Example

Now let us put the math together and verify that a 405B model actually fits on 80 GB GPUs.

With TP = 8, PP = 16, and ZeRO Stage 1 with DP = 128, the memory per GPU is:

**Parameters (fp16):**

![Equation](equations/eq_18.png)

**Gradients (fp16):**

![Equation](equations/eq_19.png)

**Optimizer states (fp32, ZeRO Stage 1):**

![Equation](equations/eq_20.png)

**Total model state per GPU:**

![Equation](equations/eq_21.png)

That leaves about **67 GB** on each 80 GB H100 for activations and workspace — more than enough with activation checkpointing.

This is exactly what we want. A 405 billion parameter model, comfortably sitting on 80 GB GPUs, across 16,384 of them.

![Memory Breakdown](figures/figure_11.png)
*Memory breakdown for one GPU in the LLaMA 3 405B configuration: Model state uses only ~13 GB out of 80 GB, leaving ample room for activations.*

---

## Communication Costs: Where the Time Goes

Let us look at the communication costs for each dimension. This is where the trade-offs become clear.

| Dimension | Communication Type | Volume per Step |
|---|---|---|
| DP (vanilla) | All-reduce | $2\Psi \cdot (N_d - 1) / N_d$ |
| DP (ZeRO-3/FSDP) | All-gather + reduce-scatter | $3\Psi \cdot (N_d - 1) / N_d$ |
| TP | All-reduce (4 per layer) | $4 \cdot b \cdot s \cdot h \cdot L \cdot (t-1) / t$ |
| PP | Point-to-point | $b \cdot s \cdot h$ per micro-batch |
| CP (Ring) | P2P ring | $2 \cdot b \cdot (s / CP) \cdot h \cdot (CP - 1)$ |
| EP | All-to-all | $2 \cdot tokens \cdot h_{expert}$ per MoE layer |

Let us compute the TP communication for one layer of LLaMA 3 405B ($b=1$, $s=8192$, $h=16384$, $t=8$):

![Equation](equations/eq_22.png)

With 126 layers, that is roughly **58 GB of TP communication per training step** — completed in about 64 ms on NVLink at 900 GB/s. This is why TP must stay within a node.

Compare this to PP communication: $1 \times 8192 \times 16384 \approx 128$ MB per micro-batch, sent point-to-point to just one neighbor. Much lighter.

The key tradeoff in 5D parallelism is clear: **more parallelism reduces memory per GPU, but increases communication.** The art is in matching each dimension to the right level of the hardware hierarchy.

![Communication Costs](figures/figure_12.png)
*Communication volume comparison: TP is the most bandwidth-hungry (NVLink required), while PP is lightweight (point-to-point over InfiniBand).*

---

## Conclusion

Let us step back and see the big picture. Each of the five parallelism dimensions targets a **different bottleneck**:

- **Data Parallelism** scales training throughput — process more data per step
- **Tensor Parallelism** splits weight matrices — when a single layer is too large for one GPU
- **Pipeline Parallelism** splits layers across stages — when the model has too many layers
- **Context Parallelism** splits the sequence — when attention memory explodes for long contexts
- **Expert Parallelism** distributes experts — when MoE models have hundreds of specialists

The art of large-scale training is in the **configuration** — matching each parallelism dimension to the right level of the hardware hierarchy. TP on NVLink, PP across nodes, DP on whatever remains.

We have come a long way from the days when a single GPU was enough. Training a model like LLaMA 3 405B requires orchestrating 16,384 GPUs across five dimensions of parallelism, where every GPU must play its part in perfect synchrony. It is like conducting a massive orchestra — each instrument has a different role, but together they produce something no single instrument could achieve alone.

This is truly amazing.

---

## References

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021)
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" (2019)
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022)
- Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
- Jacobs et al., "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models" (2023)
- Grattafiori et al., "The Llama 3 Herd of Models" (2024)
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024)
