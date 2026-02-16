# 5D Parallelism from Scratch

**How modern LLMs are trained across thousands of GPUs — understanding Data, Tensor, Pipeline, Sequence, and Expert Parallelism from first principles.**

---

Let us start with a simple example. Imagine you run a restaurant that has just received an order for 1,000 plates of biryani — but you only have one chef and one stove.

What happens?

Your chef starts cooking. One plate at a time. Each plate takes about 10 minutes. That means it will take roughly 10,000 minutes — about 7 days — to finish all 1,000 plates. Your customers have long left by then.

Now think about what exactly is going wrong here. There are two problems:

1. **The chef is too slow** — one person simply cannot cook 1,000 plates fast enough. This is a *compute bottleneck*.

2. **The kitchen is too small** — even if the chef were faster, a single stove can only hold one pot at a time. This is a *memory bottleneck*.

This is exactly the situation we face when training large language models. The "chef" is a single GPU. The "recipe" is the model. The "1,000 plates" is the massive training dataset. And just like in our kitchen, a single GPU is both too slow and too small.

So how do we scale this kitchen?

We could **hire more chefs** and give each one a copy of the recipe — that is Data Parallelism. We could **split the recipe across chefs** so each one handles a different part of the same dish — that is Tensor Parallelism. We could set up an **assembly line** where one chef preps the ingredients, another cooks the rice, and another adds the spices — that is Pipeline Parallelism. We could **split the order itself** so each chef handles different portions of the long order list — that is Sequence Parallelism. Or we could hire **specialist chefs** — one for appetizers, one for main course, one for desserts — and route each order to the right specialist — that is Expert Parallelism.

These five strategies are exactly what companies like Meta, Google, and DeepSeek use to train models across thousands of GPUs. Let us understand each one from scratch.


![The five dimensions of parallelism used to train modern LLMs.](figures/figure_1.png)
*The five dimensions of parallelism used to train modern LLMs.*


---

## Why Do We Need Parallelism?

Before we dive into each parallelism technique, let us understand *why* we need them in the first place.

Let us take a concrete example. Consider a 7 billion parameter model — something like Llama 2 7B. How much memory does it need?

When we train a model using mixed-precision training with the Adam optimizer, each parameter requires memory for four things:

1. **Model weights** (fp16): 2 bytes per parameter
2. **Gradients** (fp16): 2 bytes per parameter
3. **Optimizer states** (fp32 — Adam stores both the first moment and second moment, plus a fp32 copy of the weights): 12 bytes per parameter

So the total memory per parameter is:


$$
M = P \times (2 + 2 + 12) = 16P \text{ bytes}
$$

Let us plug in some simple numbers to see how this works. For our 7B model:

$$M = 7 \times 10^9 \times 16 = 112 \text{ GB}$$

A single NVIDIA A100 GPU has 80 GB of memory. Our 7B model needs 112 GB just for the weights, gradients, and optimizer states — and we have not even counted the **activations** (the intermediate outputs during the forward pass) which can easily add another 30–50 GB depending on the batch size and sequence length.

So here is the situation: a "small" 7B model already does not fit on a single GPU.


![A 7B model's training memory already exceeds a single A100's 80 GB capacity.](figures/figure_2.png)
*A 7B model's training memory already exceeds a single A100's 80 GB capacity.*


Now imagine a 70B model — that is 10 times larger. Or a 405B model like Llama 3. Or DeepSeek-V3 with its 671B parameters. These models require *thousands* of GPUs working together.

But it is not just about memory. Even if a model somehow fit on a single GPU, the compute time would be enormous. Training Llama 3 405B on a single A100 would take roughly 30 million GPU-hours. At one GPU, that is over 3,400 years.

So, how do we solve this? This naturally brings us to the five strategies that modern LLM training uses. Let us start with the simplest one.


---

## Data Parallelism — "Hire More Chefs"

Let us go back to our restaurant. The simplest way to cook 1,000 plates faster is to hire more chefs. You give each chef a complete copy of the recipe and divide the 1,000 orders among them — 250 plates each if you hire 4 chefs. Each chef cooks independently, and at the end of the round, they share notes on what they learned so that everyone stays in sync.

This is exactly how Data Parallelism works.

Each GPU holds a **full copy of the model**. The training dataset is split into mini-batches, and each GPU processes a different mini-batch. After the forward pass and the backward pass, each GPU has computed its own set of gradients. But we need all GPUs to stay synchronized — they must all have the same model weights.

So what do we do? We **average the gradients** across all GPUs using an operation called AllReduce:


$$
g_{\text{avg}} = \frac{1}{N} \sum_{i=1}^{N} g_i
$$

Here, $g_i$ is the gradient computed on GPU $i$, and $N$ is the total number of GPUs.

Let us plug in some simple numbers. Suppose we have 4 GPUs, and for a single parameter, the gradients computed by each GPU are: $g_1 = 0.5$, $g_2 = 0.3$, $g_3 = 0.7$, $g_4 = 0.1$. The averaged gradient is:

$$g_{\text{avg}} = \frac{0.5 + 0.3 + 0.7 + 0.1}{4} = \frac{1.6}{4} = 0.4$$

All four GPUs now use $g_{\text{avg}} = 0.4$ to update their weights. Since they all started with the same weights and now apply the same update, they remain perfectly synchronized. This is exactly what we want.


![Each GPU trains on different data; gradients are averaged via AllReduce.](figures/figure_3.png)
*Each GPU trains on different data; gradients are averaged via AllReduce.*


### The Problem with Vanilla Data Parallelism

Now the question is: does Data Parallelism solve our memory problem?

Not really. Each GPU still holds a **full copy** of the model weights, gradients, and optimizer states. If a 7B model needs 112 GB, then each GPU still needs 112 GB — regardless of how many GPUs we add.

Data Parallelism increases throughput (we process more data per second), but it does not reduce the per-GPU memory requirement.

This is where a brilliant idea called **ZeRO** (Zero Redundancy Optimizer) comes in.

### ZeRO — Eliminating Redundancy

The key insight behind ZeRO is simple: if we have 4 GPUs, why should each one store the *same* optimizer states, gradients, and weights? That is a 4x redundancy!

ZeRO progressively eliminates this redundancy in three stages:

**ZeRO Stage 1:** Shard the **optimizer states** across GPUs. Each GPU only stores 1/N of the optimizer states. This alone saves roughly 4x memory.

**ZeRO Stage 2:** Shard both the **optimizer states and gradients**. Each GPU only stores 1/N of each.

**ZeRO Stage 3:** Shard **everything** — optimizer states, gradients, and the model weights themselves. Each GPU only stores 1/N of the entire model. When a GPU needs a weight shard it does not own, it fetches it from the GPU that does.

ZeRO Stage 3 is essentially what PyTorch calls **FSDP** (Fully Sharded Data Parallel).


![ZeRO progressively shards optimizer states, gradients, and weights across GPUs.](figures/figure_4.png)
*ZeRO progressively shards optimizer states, gradients, and weights across GPUs.*


Data Parallelism (especially with ZeRO) is the workhorse of distributed training. But it has a fundamental limitation: it does not help when a **single layer** is too large to fit on one GPU. For that, we need to go deeper — we need to split the layers themselves.

This brings us to Tensor Parallelism.

---

## Tensor Parallelism — "Split the Recipe Across Chefs"

In Data Parallelism, each chef has the complete recipe. In Tensor Parallelism, we tear one page of the recipe in half. Chef A handles the first half of the sauce, Chef B handles the second half. They must coordinate at the end of each step — but together they can handle a sauce recipe that neither could fit on their stove alone.

The core idea of Tensor Parallelism is to **split individual weight matrices** across multiple GPUs.

### Splitting a Linear Layer

Let us take a simple linear layer: $Y = XW$, where $X$ is the input and $W$ is the weight matrix. How can we split this computation across 2 GPUs?

**Column-wise split:** We split the weight matrix $W$ along its columns:


$$
W = [W_1 \mid W_2] \quad \Rightarrow \quad Y = X[W_1 \mid W_2] = [XW_1 \mid XW_2]
$$


Each GPU computes its half, and we concatenate the results.

Let us plug in some simple numbers. Suppose $X$ is a $(2 \times 4)$ matrix and $W$ is a $(4 \times 6)$ matrix. We split $W$ into $W_1$ of size $(4 \times 3)$ and $W_2$ of size $(4 \times 3)$. GPU 0 computes $XW_1$, getting a $(2 \times 3)$ result. GPU 1 computes $XW_2$, getting another $(2 \times 3)$ result. We concatenate them to get the full $(2 \times 6)$ output. Each GPU only needed to store a $(4 \times 3)$ weight matrix instead of the full $(4 \times 6)$.

**Row-wise split:** Alternatively, we split $W$ along its rows:


$$
W = \begin{bmatrix} W_1 \\ W_2 \end{bmatrix}, \quad X = [X_1 \mid X_2] \quad \Rightarrow \quad Y = X_1 W_1 + X_2 W_2
$$


Here, each GPU computes a partial sum, and we use an **AllReduce** to add them together.

Let us see a quick example. Take $W$ as a $(4 \times 6)$ matrix and $X$ as a $(2 \times 4)$ matrix. We split $W$ into $W_1$ (top 2 rows, size $2 \times 6$) and $W_2$ (bottom 2 rows, size $2 \times 6$), and correspondingly split $X$ into $X_1$ (first 2 columns, size $2 \times 2$) and $X_2$ (last 2 columns, size $2 \times 2$). GPU 0 computes $X_1 W_1$ (a $2 \times 6$ partial result), GPU 1 computes $X_2 W_2$ (another $2 \times 6$ partial result), and we sum them to get the full $Y$.


![Column-wise splits need concatenation; row-wise splits need AllReduce summation.](figures/figure_5.png)
*Column-wise splits need concatenation; row-wise splits need AllReduce summation.*


### How Transformers Use Tensor Parallelism

Now the question is: how do we apply this to a real Transformer?

The Megatron-LM paper showed an elegant way to partition the MLP block of a Transformer so that only **one AllReduce** is needed per block.

Here is how it works:

1. **First linear layer** ($W_1$): Split **column-wise**. Each GPU computes its portion independently.
2. **GeLU activation**: Applied locally on each GPU — no communication needed!
3. **Second linear layer** ($W_2$): Split **row-wise**. Each GPU computes a partial result.
4. **AllReduce**: Sum the partial results across GPUs.

The beauty of this design is that we only need one AllReduce operation for the entire MLP block. The GeLU activation sits between the two linear layers and requires no communication because each GPU already has the full activations it needs after the column-wise split.

For the **self-attention** layer, the partitioning is even more natural. Each GPU handles a **subset of the attention heads**. Since attention heads are independent of each other, each GPU can compute its assigned heads without any communication until the final output projection.


![Megatron-LM splits the MLP with one AllReduce per block.](figures/figure_6.png)
*Megatron-LM splits the MLP with one AllReduce per block.*


There is one very important thing to understand about Tensor Parallelism: it requires **extremely fast interconnect** between GPUs. Since communication happens *within every single layer*, any latency kills performance. This is why Tensor Parallelism is almost always used within a single node where GPUs are connected by NVLink (900 GB/s on modern H100 nodes), not across nodes where we rely on slower InfiniBand.

Tensor Parallelism splits individual layers. But what if the model has so many layers that no single GPU can hold even a fraction of them? This brings us to Pipeline Parallelism.

---

## Pipeline Parallelism — "The Assembly Line"

Think of a car factory. Station 1 builds the chassis, Station 2 adds the engine, Station 3 paints the body, Station 4 installs the interior. Each station is a specialist — it only handles one stage of the process. As soon as Station 1 finishes one chassis, it immediately starts the next one while Station 2 works on the first car's engine. This is an assembly line, and it is the core idea behind Pipeline Parallelism.

In Pipeline Parallelism, we split the model's layers into **stages**. GPU 0 gets layers 1–8, GPU 1 gets layers 9–16, GPU 2 gets layers 17–24, and GPU 3 gets layers 25–32. Each GPU only stores and computes its assigned layers.

### The Bubble Problem

But there is a problem. Let us think about what happens with a single micro-batch:

1. GPU 0 computes the forward pass for layers 1–8 and sends the result to GPU 1.
2. GPU 1 computes layers 9–16 and sends the result to GPU 2.
3. GPU 2 computes layers 17–24 and sends the result to GPU 3.
4. GPU 3 computes layers 25–32 and starts the backward pass.

While GPU 0 is working, GPUs 1, 2, and 3 are idle. While GPU 3 is doing the backward pass, GPUs 0, 1, and 2 are idle. This idle time is called the **pipeline bubble**.

How bad is the bubble? The fraction of time wasted is:


$$
\text{Bubble fraction} = \frac{P - 1}{P - 1 + M}
$$

where $P$ is the number of pipeline stages and $M$ is the number of micro-batches.

Let us plug in some simple numbers. With $P = 4$ stages and $M = 1$ micro-batch:

$$\text{Bubble} = \frac{4 - 1}{4 - 1 + 1} = \frac{3}{4} = 75\%$$

That is terrible — 75% of the time, GPUs are sitting idle! But now let us try $M = 12$ micro-batches:

$$\text{Bubble} = \frac{4 - 1}{4 - 1 + 12} = \frac{3}{15} = 20\%$$

Much better. The trick is to split each mini-batch into many **micro-batches** and feed them through the pipeline one after another.


![With only 1 micro-batch, 75% of GPU time is wasted in pipeline bubbles.](figures/figure_7.png)
*With only 1 micro-batch, 75% of GPU time is wasted in pipeline bubbles.*


### GPipe and 1F1B

The first solution was **GPipe**: split the mini-batch into many micro-batches, run all forward passes first, then run all backward passes. This reduces the bubble, but all activations from the forward passes must be stored in memory simultaneously — which can be very expensive.

A better approach is the **1F1B** (one-forward-one-backward) schedule. The idea is simple: instead of doing all forward passes and then all backward passes, we **interleave** them. As soon as a GPU finishes a forward micro-batch, it can start a backward micro-batch from an earlier iteration.

This is clever because each GPU no longer needs to store activations for *all* micro-batches at once — it can release activations as soon as the corresponding backward pass is done. This dramatically reduces peak memory usage while keeping the same bubble fraction.


![1F1B interleaves forward and backward passes to reduce peak memory.](figures/figure_8.png)
*1F1B interleaves forward and backward passes to reduce peak memory.*


Pipeline Parallelism solves the depth problem — it lets us split a model with hundreds of layers across many GPUs. But there is still one dimension we have not touched: the **sequence length**. This brings us to Sequence Parallelism.

---

## Sequence Parallelism — "Split the Sentence"

Imagine you need to translate a 500-page book. Instead of one translator reading all 500 pages, you give pages 1–125 to translator A, pages 126–250 to translator B, pages 251–375 to translator C, and pages 376–500 to translator D. Each translator works on their section, and they coordinate when a reference in one section depends on context from another.

In the world of Transformers, "sequence length" is our "number of pages." And as models are asked to process longer and longer contexts — 128K tokens for Llama 3, over 1M tokens for Gemini — the memory required for activations grows dramatically. Attention, in particular, has memory that scales quadratically with the sequence length.

There are two flavors of Sequence Parallelism, and they solve different problems.

### Flavor 1: Megatron-Style Sequence Parallelism

This flavor works hand-in-hand with Tensor Parallelism.

Here is the problem it solves: when we use Tensor Parallelism, the linear layers and attention are split across GPUs — great. But there are parts of the Transformer that are **not** tensor-parallel: specifically, **LayerNorm** and **Dropout**. In vanilla Tensor Parallelism, these operations are simply *replicated* on every GPU. That is wasteful — each GPU stores the full activations for these operations.

Megatron-style Sequence Parallelism instead partitions these operations along the **sequence dimension**. If we have a sequence of 4,096 tokens and 4 GPUs, each GPU handles LayerNorm and Dropout for only 1,024 tokens.

The communication works as follows:
- **Before a Tensor-Parallel region** (e.g., entering the MLP): Gather the activations along the sequence dimension, then scatter along the tensor dimension.
- **After a Tensor-Parallel region** (e.g., exiting the MLP): AllReduce along the tensor dimension, then scatter along the sequence dimension.

This saves significant activation memory without adding any extra communication — it simply replaces the existing AllReduce with a more memory-efficient reduce-scatter + all-gather pattern.


![Megatron-style SP handles LayerNorm and Dropout along the sequence dimension.](figures/figure_9.png)
*Megatron-style SP handles LayerNorm and Dropout along the sequence dimension.*


### Flavor 2: Ring Attention (Context Parallelism)

The second flavor is for truly long sequences — 100K, 500K, or even 1M+ tokens. Here, the *entire* attention computation is too large for a single GPU.

The idea is beautifully simple. We split the sequence into chunks and distribute them across GPUs arranged in a **ring**. Each GPU holds:
- Its own chunk of queries (Q)
- Its own chunk of keys and values (K, V)

Then, the computation proceeds in rounds:

1. Each GPU computes attention between its local Q and local K, V.
2. Each GPU sends its K, V chunk to the next GPU in the ring and receives K, V from the previous GPU.
3. Each GPU computes attention between its local Q and the newly received K, V.
4. Repeat until every Q chunk has seen every K, V chunk.

By the end, each GPU has computed the full attention output for its chunk of the sequence. The communication (passing K, V around the ring) can be overlapped with the computation, making this very efficient.


![Ring Attention passes K,V blocks around a ring so each GPU sees the full sequence.](figures/figure_10.png)
*Ring Attention passes K,V blocks around a ring so each GPU sees the full sequence.*


We have now covered four dimensions. But there is a fifth dimension which is unique to a special class of models — the **Mixture of Experts** architecture. This brings us to Expert Parallelism.

---

## Expert Parallelism — "Specialist Chefs"

Let us go back to our restaurant one last time. So far, all our chefs have been generalists — each one can cook any dish. But what if we instead hired **specialist chefs**? One expert in appetizers, one in grilled dishes, one in curries, one in desserts. A "host" at the front decides which orders go to which specialist.

This is exactly how Mixture of Experts (MoE) models work.

### Mixture of Experts Basics

In a standard Transformer, every token passes through the *same* MLP layer. In an MoE Transformer, the MLP layer is replaced by **N expert MLPs** and a **router** (also called a gating network). The router looks at each incoming token and decides which experts should process it.

The output of an MoE layer is:


$$
y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)
$$


where $g_i(x)$ is the gate weight for expert $i$ and $E_i(x)$ is the output of expert $i$. In practice, only the **top-K** experts are activated for each token — this is what makes MoE *sparse*.

Let us plug in some simple numbers. Suppose we have 8 experts and use top-2 routing. An input token $x$ arrives, and the router produces gate weights:

$$[0.00, \; 0.05, \; 0.60, \; 0.00, \; 0.00, \; 0.35, \; 0.00, \; 0.00]$$

The top-2 experts are Expert 2 (weight 0.60) and Expert 5 (weight 0.35). We renormalize so they sum to 1:

$$g_2 = \frac{0.60}{0.60 + 0.35} = 0.63, \quad g_5 = \frac{0.35}{0.60 + 0.35} = 0.37$$

The final output is:

$$y = 0.63 \times E_2(x) + 0.37 \times E_5(x)$$

Only 2 out of 8 experts were activated. This means the model can have 8x more parameters (one per expert) while only using 2x the compute of a single expert per token. This is exactly what we want — more capacity without proportionally more compute.

### Placing Experts on GPUs

Now, Expert Parallelism simply means: place different experts on different GPUs. If we have 8 experts and 8 GPUs, each GPU hosts one expert.

The communication pattern is called **All-to-All**. It works like this:

1. The router on each GPU decides which expert should handle each token.
2. Tokens are **dispatched** to the GPU hosting the correct expert (All-to-All send).
3. Each GPU processes the tokens assigned to its expert.
4. Results are **collected** back to the original GPUs (All-to-All receive).


![Tokens are routed to specialist experts on different GPUs via All-to-All communication.](figures/figure_11.png)
*Tokens are routed to specialist experts on different GPUs via All-to-All communication.*


### The Load Balancing Challenge

You might be thinking: what happens if the router sends all tokens to the same expert? Then one GPU is overloaded while the others sit idle.

This is a real problem. To prevent it, MoE models add an **auxiliary loss** during training that encourages the router to distribute tokens evenly across experts. Without this loss, models tend to collapse into using just one or two experts — defeating the entire purpose of having multiple specialists.

At the scale of production models, these numbers get impressive. DeepSeek-V3 uses **256 experts** with top-8 routing, meaning each token activates only 8 out of 256 available experts. The model has 671 billion total parameters but only activates about 37 billion per token.

---

## Putting It All Together — The 5D Grid

Now let us understand how all five dimensions work together.

In practice, these parallelism strategies are not used in isolation — they are **composed**. The total number of GPUs used in training is:


$$
N_{\text{total}} = N_{\text{DP}} \times N_{\text{TP}} \times N_{\text{PP}} \times N_{\text{EP}}
$$

(Sequence Parallelism typically shares the same GPU groups as Tensor Parallelism, so it does not add an independent dimension to the count.)

Let us plug in some real numbers. For Llama 3 405B, Meta used:

$$N_{\text{total}} = 128 \times 8 \times 16 = 16{,}384 \text{ GPUs}$$

That is 128-way Data Parallelism, 8-way Tensor Parallelism (one full node of 8 GPUs), and 16-way Pipeline Parallelism.

The key principle behind this composition is the **communication hierarchy**:

- **Tensor Parallelism** needs the fastest links because communication happens within every single layer. This is why TP is always used **within a node**, where GPUs are connected by NVLink at 900 GB/s.

- **Pipeline Parallelism** needs moderate bandwidth because communication happens only at stage boundaries (once per micro-batch per stage). PP is typically used **across nearby nodes** connected by high-speed InfiniBand.

- **Data Parallelism** can tolerate the highest latency because gradient synchronization happens only once per training step. DP spans the **entire cluster**.

- **Sequence Parallelism** piggybacks on the same GPU groups as Tensor Parallelism (within a node).

- **Expert Parallelism** requires All-to-All communication, which can be expensive. In practice, EP groups are carefully chosen to balance communication cost with expert placement.


![How the five parallelism dimensions compose to span 16,384 GPUs in Llama 3 training.](figures/figure_12.png)
*How the five parallelism dimensions compose to span 16,384 GPUs in Llama 3 training.*


Here is a summary of when to use each technique:

| Dimension | What it splits | Why you need it | Communication |
|---|---|---|---|
| **Data Parallel** | Training data | Increase throughput | AllReduce (gradients) |
| **Tensor Parallel** | Weight matrices | Single layer too large | AllReduce / AllGather within layers |
| **Pipeline Parallel** | Model layers | Too many layers for one GPU | Point-to-point at stage boundaries |
| **Sequence Parallel** | Sequence length | Long contexts blow up memory | Reduce-Scatter / Ring pass |
| **Expert Parallel** | MoE experts | Distribute specialist sub-networks | All-to-All token dispatch |

---

## Conclusion

Let us recap what we have learned.

Training a large language model is fundamentally a problem of scale — the model does not fit on one GPU, and a single GPU is too slow. The solution is to split the problem across thousands of GPUs using five complementary strategies:

1. **Data Parallelism** splits the training data across GPUs — each GPU sees different examples but shares the same model.
2. **Tensor Parallelism** splits individual weight matrices — allowing layers that are too large for one GPU.
3. **Pipeline Parallelism** splits the model depth — assigning different layers to different GPUs like an assembly line.
4. **Sequence Parallelism** splits along the sequence length — critical for long-context models.
5. **Expert Parallelism** distributes MoE experts — routing tokens to specialist sub-networks on different GPUs.

Each dimension solves a specific bottleneck, and together they form the "5D grid" that makes it possible to train models with hundreds of billions of parameters across tens of thousands of GPUs.

The next time you read that a model was trained on 16,000 GPUs, you now understand exactly what is happening under the hood.

That is all. Thank you for reading.

---

## References

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" (2019)
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021)
- Li et al., "Sequence Parallelism: Long Sequence Training from System Perspective" (2023)
- Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2022)
- Llama Team, "The Llama 3 Herd of Models" (2024)
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024)
