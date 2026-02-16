# 5D Parallelism: How We Train Models Too Large for Any Single GPU

*A visual, analogy-driven guide to the five dimensions of parallelism that power trillion-parameter training.*

---

Let us start with a simple question. You want to train a 70-billion-parameter language model. You have a shiny new NVIDIA H100 GPU with 80 GB of memory. Can you do it?

Let us do a quick back-of-the-napkin calculation. When you train a model with mixed-precision and the Adam optimizer, you need to store three things in memory: the model parameters (in fp16), the gradients (in fp16), and the optimizer states (in fp32 — that is the parameter copy, the momentum, and the variance). Add it all up and you get roughly **16 bytes per parameter**:

$$M_{\text{training}} \approx 16 \times \Psi \text{ bytes}$$

Let us plug in some simple numbers. For our 70-billion-parameter model: $M = 16 \times 70 \times 10^9 = 1{,}120$ GB. Our GPU has 80 GB. **Not even close.** We would need 14 of these GPUs just to hold the model states — and we have not even counted the memory needed for activations yet.

So what do we do? We split the work across many GPUs. But here is the catch — there are **five fundamentally different ways** to split it. Each one solves a different bottleneck. Together, they form what the community calls **5D Parallelism**.

Think of it this way. Training a massive model is like running a restaurant that needs to serve 10,000 meals an hour. No single kitchen can handle that. You need to scale — but *how* you scale matters enormously. You could open more kitchens, or hire specialist chefs, or build an assembly line. Each strategy has trade-offs. The five dimensions of parallelism are exactly these five scaling strategies:

| Dimension | What Gets Split | Kitchen Analogy |
|---|---|---|
| **Data Parallelism (DP)** | Training batch | Open identical franchise kitchens |
| **Tensor Parallelism (TP)** | Weight matrices within a layer | Multiple chefs split one dish |
| **Pipeline Parallelism (PP)** | Layers across stages | Sandwich assembly line |
| **Context Parallelism (CP)** | Sequence length | Waiters split a long banquet table |
| **Expert Parallelism (EP)** | MoE expert sub-networks | Specialist chefs (sushi, pastry, grill) |

And here is the beautiful part — these dimensions are **multiplicative**. The total number of GPUs you need is:

$$N_{\text{GPUs}} = N_{\text{DP}} \times N_{\text{TP}} \times N_{\text{PP}} \times N_{\text{CP}} \times N_{\text{EP}}$$

Meta trained Llama 3.1 405B on **16,384 H100 GPUs** using four of these five dimensions simultaneously. Let us understand how each one works.


![Five Ways to Split the Work](figures/figure_1.png)
*The five dimensions of parallelism, each mapped to a kitchen scaling strategy.*


---

## Data Parallelism — Open More Kitchens

Let us start with the most intuitive approach. If one kitchen cannot serve all the customers, **open more kitchens.**

Imagine you run a burger restaurant. Business is booming. Your one kitchen can make 100 burgers an hour, but you need 400. The simplest solution? Open three more identical kitchens — same menu, same equipment, same recipes. Each kitchen serves a different group of customers. At the end of each day, all four kitchens share notes: "the new sauce needs less salt," "grill temperature should be higher." They all update their recipes together so that every location stays in sync.

This is exactly how Data Parallelism works. Every GPU gets a **full copy** of the model. The training data is split into chunks — each GPU processes a different chunk. After computing gradients on its own chunk, all GPUs **synchronize** their gradients through an operation called AllReduce. Now every GPU has the same averaged gradient, and they all take the same optimizer step. The result? Four GPUs process four times the data in the same amount of time.


![Data Parallelism](figures/figure_2.png)
*Same model everywhere, different data. Sync after every step.*


### The Problem: Wasteful Copies

But there is a catch. Every kitchen has to own the **full** set of equipment — every pot, every pan, every oven — even if it only uses a fraction of it at any given moment. This is incredibly wasteful.

Remember our calculation? Training a 7-billion-parameter model (like Llama 7B) requires about $16 \times 7 \times 10^9 = 112$ GB of memory. A single 80 GB H100 cannot even hold one copy. And with vanilla Data Parallelism, we need a full copy on *every* GPU. Not great.

### The Fix: ZeRO / FSDP — Share the Equipment

The insight behind ZeRO (Zero Redundancy Optimizer) and FSDP (Fully Sharded Data Parallelism) is simple: **stop duplicating everything.** Instead of every kitchen owning every tool, divide the equipment across kitchens. Each kitchen keeps only its share. When it needs a tool it does not have, it borrows it from a neighbor.

ZeRO does this in three stages, each more aggressive than the last:

- **Stage 1:** Share the optimizer states (the heaviest items) → ~4× memory reduction
- **Stage 2:** Also share the gradients → ~8× memory reduction
- **Stage 3:** Also share the parameters themselves → memory scales linearly with GPU count

With ZeRO Stage 3, the memory per GPU becomes simply the total divided by the number of GPUs. For our 70B model on 64 GPUs: $1{,}120 \div 64 = 17.5$ GB per GPU. **Now it fits comfortably.**

The trade-off? When a GPU needs parameters it does not currently hold, it must fetch them from the others. This adds an extra round of communication (an AllGather operation) during the forward pass. But for most training setups, this is a very good deal — a small communication cost in exchange for a massive memory saving.


![DDP vs FSDP/ZeRO-3](figures/figure_3.png)
*DDP keeps full copies on every GPU. FSDP/ZeRO-3 shards everything — same model, 4x less memory per GPU.*


---

## Tensor Parallelism — Split the Recipe

Data Parallelism clones the whole kitchen. But what if a single dish is so complicated that one chef cannot make it fast enough? What if the recipe itself is the bottleneck?

Imagine you are making a seven-layer wedding cake. It requires: mixing the batter, baking each layer, making three different fillings, assembling the layers, and decorating the top. One pastry chef doing everything will take hours. The solution? **Two chefs work on the same cake at the same time.** Chef A handles the batter and baking. Chef B prepares the fillings and decoration. They combine their work at the end. But crucially — they must be at the **same counter**, constantly coordinating. If Chef A is in a different building, the overhead of running back and forth would negate the benefit.

This is Tensor Parallelism. Instead of splitting the data (different orders for each kitchen), we split the **computation within a single layer**. A large weight matrix is sliced across GPUs — each GPU computes part of the matrix multiply, and the partial results are combined via an AllReduce operation.

In a transformer, every layer has two big matrix multiplications: one in the MLP block and one in the self-attention block. The Megatron-LM framework showed a clever way to split these so that you only need **one AllReduce per block** — not one per matrix. The trick is to split the first matrix by columns and the second by rows. The nonlinearity (GeLU) in between can be applied independently on each GPU, so no communication is needed there. This is exactly what we want.

For self-attention, the split maps perfectly to **attention heads**. If you have 64 attention heads and 8 GPUs, each GPU handles 8 heads. The Q, K, and V projections are split by columns (heads), and the output projection is split by rows.


![Tensor Parallelism](figures/figure_4.png)
*One layer, split across four GPUs. The weight matrix is sliced into columns, each GPU computes its slice, and AllReduce combines the results.*


### The Golden Rule: TP Stays Inside the Node

Here is the critical constraint. Tensor Parallelism requires an AllReduce **inside every single layer** — in both the forward and backward pass. For a model with 80 layers, that is hundreds of AllReduce operations per training step. If these have to go over a slow network link, your GPUs will spend more time waiting for communication than doing useful computation.

This is why TP always stays **within a single server node.** Inside an NVIDIA DGX H100 node, 8 GPUs are connected by NVLink with a bidirectional bandwidth of 900 GB/s. Between nodes, you have InfiniBand at roughly 50–100 GB/s — about 10× slower. Running TP across that boundary would be a disaster.

So the rule is simple: **TP degree = number of GPUs per node**, which is almost always 8.


![NVLink vs InfiniBand](figures/figure_5.png)
*TP lives inside the node (NVLink, 900 GB/s). The inter-node link (InfiniBand, 50 GB/s) is 10x slower — never put TP there.*


---

## Pipeline Parallelism — The Assembly Line

Tensor Parallelism splits one layer across GPUs within a node. But a large model might have 80 or even 120 layers. What if, even after splitting each layer with TP, the whole model still does not fit on a single node? This brings us to Pipeline Parallelism.

Think of a sandwich shop with an assembly line. Station 1 lays out the bread and adds lettuce. Station 2 piles on the meat and cheese. Station 3 adds condiments, wraps the sandwich, and hands it to the customer. Each station works on a **different sandwich** at the same time. While Station 2 is building Sandwich #1, Station 1 is already starting Sandwich #2.

Pipeline Parallelism works the same way. We divide the model's layers into **stages** and assign each stage to a different GPU (or group of GPUs, if we are combining with TP). The first GPU processes layers 0–19, the second handles layers 20–39, the third takes layers 40–59, and so on. Activations flow from one stage to the next via simple point-to-point sends — no expensive AllReduce needed.

But there is a catch. When Sandwich #1 first arrives at Station 1, Stations 2 and 3 have nothing to do — they are idle. And when Sandwich #1 reaches Station 3 at the end, Stations 1 and 2 are now idle. This wasted time is called the **pipeline bubble**, and it is the main downside of Pipeline Parallelism.


![Pipeline Parallelism](figures/figure_6.png)
*Top: the sandwich assembly line analogy. Bottom: the GPU pipeline timeline showing forward passes (blue), backward passes (orange), and the pipeline bubble (grey idle time).*


### The Bubble Problem

How bad is the bubble? There is a simple rule:

$$\text{Bubble fraction} \approx \frac{p - 1}{m}$$

where $p$ is the number of pipeline stages and $m$ is the number of micro-batches.

Let us plug in some numbers. With $p = 4$ stages and $m = 8$ micro-batches: bubble $= 3/8 = 37.5\%$. That means more than a third of GPU time is wasted! But if we increase to $m = 32$ micro-batches: bubble $= 3/32 = 9.4\%$. Much better. The lesson is clear: **you need many more micro-batches than pipeline stages.**

This is why Pipeline Parallelism works well in combination with other dimensions. The outer Data Parallelism provides a large global batch, which can be sliced into many micro-batches to keep the pipeline full.

### Squeezing the Bubble: Zero-Bubble Scheduling

Researchers have found clever ways to shrink the bubble even further. The key insight is that the backward pass can be split into two independent parts: computing the **input gradients** (needed to send back to the previous stage) and computing the **weight gradients** (needed for the optimizer step). By rearranging when each part happens, you can fill in the gaps where GPUs would otherwise be idle.

DeepSeek-V3 uses a variant of this called **DualPipe**, which overlaps computation with communication to get near-zero bubble. This is cutting-edge work from 2024 — the pipeline scheduling problem is still an active area of research.


![1F1B vs Zero-Bubble Schedule](figures/figure_7.png)
*Standard 1F1B schedule (top) vs Zero-Bubble schedule (bottom). Same total work, but the idle time is filled by rearranging weight gradient computation.*


---

## Context Parallelism — The Long Banquet Table

We have split the data, the weight matrices, and the layers. But there is one more bottleneck that has become increasingly critical in the era of long-context models: the **sequence length**.

Here is the problem. Self-attention computes a score between **every pair of tokens** in the input. If your input has $s$ tokens, the attention mechanism builds an $s \times s$ matrix. Memory grows as the **square** of the sequence length. At $s = 4{,}096$ tokens, this is manageable. At $s = 128{,}000$ tokens? The attention scores alone would require hundreds of gigabytes. No single GPU can hold that.

Imagine a banquet table stretching 100 meters long. One waiter cannot serve the whole table — they would spend all their time running back and forth. The solution? Split the table into sections. Waiter 1 takes seats 1–25, Waiter 2 takes 26–50, and so on. Each waiter serves their section efficiently. But here is the trick — sometimes a guest at seat 10 orders the same dish as a guest at seat 80. The waiters need to **pass information** between sections so that every guest is served correctly.

This is exactly how **Ring Attention** works — the core mechanism behind Context Parallelism. Each GPU holds a chunk of the input sequence (specifically, a chunk of the Query tokens). The Key and Value chunks circulate around a ring of GPUs, one step at a time. At each step, a GPU computes attention between its local Queries and whatever Key/Value chunk it currently holds. After all chunks have circulated, every token has attended to every other token — the same result as running attention on a single GPU, but distributed across many.

The beauty is that while a GPU is computing attention on the current K/V chunk, it is simultaneously **sending that chunk to the next GPU** in the ring. Communication hides behind computation.


![Ring Attention](figures/figure_8.png)
*Ring Attention: K/V chunks circulate around a ring of GPUs. After 4 steps, every Q has attended to every K/V. The ring is complete.*


### Quick Note: SP vs CP

You might see two terms that sound similar — let us clear that up:

- **Sequence Parallelism (SP)** is a simpler technique that is always paired with Tensor Parallelism. It shards the activations of operations like LayerNorm and Dropout along the sequence dimension. It adds no extra communication — it just restructures what TP already does.
- **Context Parallelism (CP)** is the heavyweight version we just described. It splits attention itself across GPUs and is a fully independent parallelism dimension.

When people talk about 5D Parallelism, they mean CP, not SP.

---

## Expert Parallelism — Specialist Chefs

The four dimensions above apply to any model — dense transformers, vision models, you name it. But some of the largest models today use a special architecture called **Mixture-of-Experts (MoE)**, and it brings a fifth dimension into play.

Think of a high-end food hall. Instead of one kitchen cooking everything, there are specialist stations: a sushi chef, a pastry chef, a grill master, a pasta chef. When a customer places an order, the **maître d'** (the router) looks at the order and sends it to the right specialist. A steak order goes to the grill, a cake order goes to the pastry station. Most orders only need one or two specialists — the rest stay idle for that order. This is incredibly efficient: you get the variety of a hundred-chef restaurant with the cost of running only a few chefs at a time.

An MoE transformer layer works the same way. Instead of one large MLP block, it has many smaller **expert** sub-networks — DeepSeek-V3 has 256 of them. A small router network looks at each token and decides which 1–2 experts should handle it. Only those experts activate. The rest sit idle for that token.

Now, with 256 experts, you cannot fit them all on one GPU. Expert Parallelism distributes them: if you have 64 GPUs for EP, each GPU holds 4 experts. When a token is routed to Expert #47, it needs to travel to whichever GPU holds that expert, get processed, and travel back. This token-shuffling happens through an **All-to-All** communication pattern — every GPU potentially sends tokens to every other GPU, and receives tokens back.


![Expert Parallelism](figures/figure_9.png)
*The router dispatches tokens to specialist experts via All-to-All communication. Note the load imbalance — one GPU gets more tokens than others.*


### The Load Balancing Challenge

The maître d' must route orders fairly. If every customer orders sushi, the sushi chef is overwhelmed while the grill master stands idle. In MoE models, this is the **load balancing problem** — and it is one of the trickiest practical challenges.

Solutions include adding an auxiliary loss that penalizes uneven routing, setting a hard capacity cap per expert, or dynamically adjusting the router's biases (the approach DeepSeek-V3 uses).

### The DeepSeek-V3 Example

DeepSeek-V3 is a striking example of Expert Parallelism in action. It has **671 billion total parameters** across its experts, but only **37 billion activate** for any given token. You get the brainpower of a 671B model at the computational cost of a 37B one. This is exactly what we want.

---

## Putting It All Together

Now we have all five dimensions. The natural question is: how do you combine them in practice? The answer lies in the **hardware**.

### The Hardware Hierarchy

Not all GPU-to-GPU connections are equal. Inside an NVIDIA DGX H100 server, 8 GPUs are connected by NVLink — a blazing-fast mesh with 900 GB/s of bidirectional bandwidth. Think of these as chefs at the same kitchen counter: they can hand things to each other instantly.

Between servers, you have InfiniBand at 50–100 GB/s — roughly **10× slower**. Think of this as kitchens in different buildings: you can still send things back and forth, but there is a real delay.

The rule of thumb is simple: **chatty parallelism dimensions stay inside the node. Quiet dimensions can cross nodes.**

- **Tensor Parallelism** needs to communicate *inside every layer* → stays on NVLink (within one node)
- **Expert Parallelism** does All-to-All shuffles that are bandwidth-hungry → best within NVLink or a small cluster of nodes
- **Pipeline Parallelism** only sends activations between adjacent stages → point-to-point, lightweight, can cross nodes
- **Context Parallelism** passes K/V chunks in a ring → overlaps with computation, can cross nodes
- **Data Parallelism** synchronizes gradients once per step → can span the entire cluster


![The 5D Hardware Map](figures/figure_10.png)
*How all five dimensions map to a datacenter. TP inside each node (NVLink), PP across nodes (InfiniBand), DP across the cluster. Llama 3.1 405B: 8 × 16 × 128 = 16,384 GPUs.*


### Real-World Configurations

**Llama 3.1 405B** (Meta — 16,384 H100 GPUs):

At a short context of 8K tokens, Meta used TP=8, PP=16, and DP=128. That gives $8 \times 16 \times 128 = 16{,}384$ GPUs. When they scaled to 128K tokens, they turned on Context Parallelism at CP=16, and reduced DP to 8: $8 \times 16 \times 16 \times 8 = 16{,}384$. Same number of GPUs, completely different parallelism mix. The five knobs are re-tuned for the longer menu.

**DeepSeek-V3** (2,048 H800 GPUs):

This is an MoE model, so Expert Parallelism enters the picture. DeepSeek used PP=16 and EP=64 with ZeRO-1 Data Parallelism — and notably **no Tensor Parallelism at all**. Instead, they rely on their DualPipe scheduling to overlap computation with the All-to-All communication that EP requires. It is a very different recipe for a very different model.

---

## Practical Code — Data Parallelism in PyTorch

Enough concepts. Let us look at some actual code. We will implement the simplest dimension — Data Parallelism — using PyTorch's DistributedDataParallel.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train(rank, world_size):
    # Initialize the process group for communication
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Create the model — same on every GPU
    model = nn.Linear(1024, 1024).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    for step in range(100):
        # Each GPU gets different data (different random batch)
        x = torch.randn(32, 1024, device=rank)
        loss = ddp_model(x).sum()

        loss.backward()   # Gradients are AllReduced automatically here!
        optimizer.step()
        optimizer.zero_grad()

# Launch with: torchrun --nproc_per_node=4 script.py
```

Let us understand this code in detail. First, we initialize the NCCL process group — this sets up the GPU-to-GPU communication backend. Each GPU gets a `rank` (0, 1, 2, 3) and knows the total `world_size` (4).

We create a simple model and wrap it with `DDP`. This is the magic line — from this point on, every call to `loss.backward()` will **automatically** trigger an AllReduce across all GPUs to average the gradients. You do not need to write any communication code yourself. One line of wrapping, and your single-GPU training loop becomes multi-GPU.

Each GPU creates its own random batch (`torch.randn`), computes the forward pass, and calls backward. The AllReduce happens behind the scenes during the backward pass — it is overlapped with gradient computation so that communication and computation happen in parallel.

The other four dimensions — TP, PP, CP, and EP — require specialized frameworks like NVIDIA's Megatron-LM or Microsoft's DeepSpeed. But the principle is always the same: **split, compute, communicate.**

---

## Wrapping Up

Let us recap our five dimensions, one last time through the kitchen lens:

1. **Data Parallelism** — Open more kitchens, each serving different customers, sharing notes at the end of the day
2. **Tensor Parallelism** — Two chefs splitting one complex dish at the same counter, combining at the end
3. **Pipeline Parallelism** — An assembly line where each station handles a different stage, sandwiches flowing through
4. **Context Parallelism** — Waiters splitting a long banquet table into sections, passing dishes down the line
5. **Expert Parallelism** — Specialist chefs, each handling only the orders they are best at, routed by the maître d'

The art of training at scale is not just about throwing more GPUs at the problem — it is about choosing the right combination of these five knobs for your model, your hardware, and your sequence length. Llama 3 uses one recipe. DeepSeek-V3 uses a completely different one. Both train massive models efficiently because they match their parallelism strategy to their architecture and hardware.

Looking ahead, next-generation hardware like NVIDIA's GB200 NVL72 — which puts **72 GPUs** in a single NVLink domain — will blur some of these boundaries. When the fast-link domain grows from 8 to 72 GPUs, Tensor Parallelism can stretch further, and some pipeline stages may collapse. The five knobs remain, but the dials shift.

That's it! See you next time.

---

## References

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021)
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)
- Huang et al., "GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism" (2019)
- Qi et al., "Zero Bubble Pipeline Parallelism" (2024)
- Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021)
- Meta AI, "The Llama 3 Herd of Models" (2024)
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024)
- HuggingFace, "The Ultra-Scale Playbook: Training LLMs on GPU Clusters" (2025)
