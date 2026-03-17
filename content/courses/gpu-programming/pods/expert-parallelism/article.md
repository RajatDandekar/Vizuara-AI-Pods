# Expert Parallelism

*Mixture of Experts models activate only a fraction of parameters per token — learn how experts are distributed across GPUs and kept balanced.*

---

Let us start with a thought experiment. Imagine you run a hospital with 100 doctors. Every time a patient walks in — whether they have a broken arm, a heart condition, or a skin rash — you could send them to every single doctor for an opinion. But that would be absurdly wasteful. In practice, a receptionist quickly assesses the patient and routes them to the 2 or 3 specialists who are most relevant. The heart patient sees the cardiologist and perhaps the radiologist. The skin rash patient sees the dermatologist. Each patient only uses a small fraction of the hospital's total expertise, yet the hospital as a whole has enormous capacity.

This is exactly the idea behind **Mixture of Experts (MoE)** models. Instead of activating every parameter for every token, we build a model with many specialized sub-networks — called **experts** — and a lightweight **router** that sends each token to only a handful of them. The result is a model with a massive total parameter count (giving it huge capacity) but a modest computational cost per token (because most of the parameters are dormant for any given input).

Models like Mixtral 8x7B, DeepSeek-V3, and the Switch Transformer all use this architecture. And distributing these experts across GPUs — **expert parallelism** — is the fifth and final dimension of the parallelism grid that powers modern large-scale training.

In this article, we will build MoE from the ground up: the architecture, the routing mechanism, the communication pattern, the load balancing problem, and finally, how expert parallelism combines with the four other parallelism strategies to form the complete 5D parallelism framework.


## The Standard Transformer MLP: Where MoE Begins

To understand Mixture of Experts, we first need to be clear about what it replaces. In a standard transformer block, each token passes through two main components: the **self-attention** layer and the **feed-forward network (MLP)**. The MLP typically consists of two linear projections with an activation function in between:

$$\text{MLP}(\mathbf{x}) = \mathbf{W}_2 \, \sigma(\mathbf{W}_1 \mathbf{x})$$

where $\mathbf{x} \in \mathbb{R}^{d}$ is the input token representation, $\mathbf{W}_1 \in \mathbb{R}^{4d \times d}$ projects up to a larger dimension, $\sigma$ is an activation function like GELU, and $\mathbf{W}_2 \in \mathbb{R}^{d \times 4d}$ projects back down.

For a model with hidden dimension $d = 4096$, this single MLP has about $2 \times 4096 \times 16384 \approx 134$ million parameters. And critically, **every single token** passes through every single one of these parameters. This is what we call a **dense** model — all parameters are active for all inputs.

Now the question is: do we really need all 134 million parameters to process every token? Intuitively, different tokens might benefit from different processing. The token "photosynthesis" probably needs different computations than the token "meanwhile". What if we could have multiple specialized MLPs and route each token to just the relevant ones?


## The MoE Layer: Router + N Expert MLPs

A **Mixture of Experts layer** replaces the single MLP with $N$ independent MLPs — the **experts** — plus a small **router** (also called a **gating network**) that decides which experts each token should visit.

![The MoE layer architecture: a router network examines each token and sends it to the top-k experts out of N total expert MLPs.](figures/figure_1.png)
*The MoE layer: a router examines each token and routes it to the top-k experts (here $k = 2$) out of $N$ total expert MLPs. The outputs of the selected experts are combined using the router's weights.*

Here is how it works for a single token $\mathbf{x}$:

**Step 1: Compute router scores.** The router is a simple linear layer followed by softmax:

$$\mathbf{h} = \mathbf{W}_r \mathbf{x}$$

$$p_i = \frac{e^{h_i}}{\sum_{j=1}^{N} e^{h_j}} \quad \text{for } i = 1, \ldots, N$$

where $\mathbf{W}_r \in \mathbb{R}^{N \times d}$ is the router's weight matrix and $p_i$ is the probability that token $\mathbf{x}$ should be sent to expert $i$.

**Step 2: Select top-k experts.** We pick the $k$ experts with the highest probabilities. Typically $k = 1$ or $k = 2$.

$$\text{TopK} = \text{argtop}_k(p_1, p_2, \ldots, p_N)$$

**Step 3: Compute expert outputs and combine.** Only the selected $k$ experts process the token. The final output is a weighted sum of their outputs:

$$\mathbf{y} = \sum_{i \in \text{TopK}} \tilde{p}_i \cdot \text{Expert}_i(\mathbf{x})$$

where $\tilde{p}_i$ is the renormalized weight among the selected experts:

$$\tilde{p}_i = \frac{p_i}{\sum_{j \in \text{TopK}} p_j}$$

Let us plug in some concrete numbers. Suppose we have $N = 8$ experts, $k = 2$, and the router produces scores $\mathbf{h} = [1.2, 0.3, 2.5, 0.1, 0.8, 3.1, 0.2, 0.5]$. After softmax, the probabilities become approximately:

$$\mathbf{p} = [0.06, 0.02, 0.23, 0.02, 0.04, 0.42, 0.02, 0.03]$$

The top-2 experts are Expert 5 (probability 0.42) and Expert 2 (probability 0.23). After renormalization:

$$\tilde{p}_5 = \frac{0.42}{0.42 + 0.23} = 0.646, \quad \tilde{p}_2 = \frac{0.23}{0.42 + 0.23} = 0.354$$

The final output is $\mathbf{y} = 0.646 \cdot \text{Expert}_5(\mathbf{x}) + 0.354 \cdot \text{Expert}_2(\mathbf{x})$. The other 6 experts are never computed for this token. This is exactly what we want — massive model capacity with sparse computation.


### The Parameter-vs-Compute Trade-Off

This is the magic of MoE. Consider Mixtral 8x7B, which has 8 experts of roughly 7B parameters each. The total parameter count is about 47B (8 experts plus the shared attention layers). But because each token only activates 2 experts, the **active parameters** per token are roughly equivalent to a 13B dense model.

We can express this relationship as:

$$\text{Active FLOPs per token} \approx \frac{k}{N} \times \text{Total expert FLOPs} + \text{Shared FLOPs}$$

For Mixtral: $\frac{2}{8} = 25\%$ of the expert FLOPs are used per token. The model has the capacity of a 47B model but the inference cost of a ~13B model. Not bad, right?


### A Minimal MoE Implementation

Let us see this in code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: a simple linear layer
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # N independent expert MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch, seq_len, d = x.shape

        # Compute router probabilities
        router_logits = self.router(x)                  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1) # (batch, seq_len, num_experts)

        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # both: (batch, seq_len, top_k)

        # Renormalize the top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute weighted sum of selected expert outputs
        output = torch.zeros_like(x)
        for k_idx in range(self.top_k):
            expert_idx = top_k_indices[:, :, k_idx]     # (batch, seq_len)
            weight = top_k_probs[:, :, k_idx].unsqueeze(-1)  # (batch, seq_len, 1)

            for e in range(self.num_experts):
                mask = (expert_idx == e)                 # (batch, seq_len)
                if mask.any():
                    tokens = x[mask]                     # (num_selected, d_model)
                    expert_out = self.experts[e](tokens)  # (num_selected, d_model)
                    output[mask] += weight[mask] * expert_out

        return output
```

Let us understand this code. The router is a single linear layer that maps each token from $d$-dimensional space to $N$ scores — one per expert. We apply softmax to get probabilities, then select the top-$k$ experts. For each selected expert, we gather the tokens that were routed to it, run them through that expert's MLP, and scatter the weighted results back. The key point is that each expert only processes the tokens assigned to it, not all tokens.

This implementation is deliberately pedagogical — production MoE implementations use batched operations and specialized CUDA kernels for the gather-scatter pattern to avoid the Python loops.


## Expert Parallelism: Different Experts on Different GPUs

Now we arrive at the central question: how do we distribute these experts across GPUs?

With $N = 64$ experts (as in DeepSeek-V3), each expert is a full MLP with millions of parameters. Storing all 64 experts on every GPU would defeat the purpose of having sparse computation. The natural solution is **expert parallelism**: place different experts on different GPUs.

If we have $E$ GPUs dedicated to expert parallelism and $N$ experts, each GPU holds $N / E$ experts. For example, with 64 experts on 8 GPUs, each GPU stores 8 experts.

But here is the complication. During the forward pass, a batch of tokens arrives, and the router assigns each token to its top-$k$ experts. These experts might live on different GPUs. Token 1 might need Expert 3 (on GPU 0) and Expert 47 (on GPU 5). Token 2 might need Expert 12 (on GPU 1) and Expert 3 (on GPU 0). Every token needs to be sent to the GPUs that hold its assigned experts, processed there, and then the results need to be sent back.

This requires a communication pattern called **All-to-All**.


## All-to-All Communication: The Heart of Expert Parallelism

In data parallelism, we used **AllReduce** — every GPU sends the same type of data (gradients) and receives the same result. In expert parallelism, we need something fundamentally different. Each GPU needs to send **different data to different GPUs**. GPU 0 has some tokens destined for the experts on GPU 1, different tokens destined for GPU 2, and so on. Simultaneously, GPU 0 needs to receive tokens from all other GPUs that are destined for the experts it holds.

This is the **All-to-All** communication primitive.

![All-to-All communication: each GPU sends different tokens to different GPUs based on routing decisions, and receives tokens destined for its local experts.](figures/figure_2.png)
*All-to-All communication: each GPU partitions its tokens by destination GPU and sends different subsets to different GPUs. After the experts process the tokens, a second All-to-All sends the results back.*

Here is the flow for a single MoE layer with expert parallelism:

1. **Route locally.** Each GPU runs the router on its local tokens, determining which experts each token needs.
2. **All-to-All dispatch.** Each GPU sorts its tokens by destination GPU and performs an All-to-All. After this, each GPU has received all the tokens that need to be processed by its local experts.
3. **Expert computation.** Each GPU runs its local experts on the received tokens.
4. **All-to-All combine.** A second All-to-All sends the expert outputs back to the originating GPUs.
5. **Weighted sum.** Each GPU combines the expert outputs using the router weights.

The communication cost of All-to-All is significant. If each GPU holds $T$ tokens and each token representation is $d$ floats, the total data each GPU must send (and receive) is approximately:

$$\text{Data per GPU} = T \cdot d \cdot k \cdot \text{sizeof(float)}$$

For $T = 4096$ tokens, $d = 4096$, $k = 2$, and FP16: each GPU transfers about $4096 \times 4096 \times 2 \times 2 = 64$ MB per MoE layer. With 20+ MoE layers in a model, this adds up.

This is why expert parallelism typically operates over GPUs connected by fast interconnects (NVLink or NVSwitch within a node), and why the number of expert-parallel GPUs is usually kept modest — 4 to 16.


## The Load Balancing Problem

Now let us address the most subtle and important challenge in MoE: **load balancing**. The router is a learned neural network, and there is nothing preventing it from sending most tokens to a small number of "popular" experts while leaving others idle. In the worst case, the router collapses to always selecting the same 1 or 2 experts, effectively reducing the MoE layer to a dense layer.

![Load imbalance: some experts receive a disproportionate number of tokens while others sit idle, wasting GPU compute.](figures/figure_3.png)
*Load imbalance in MoE: the router sends most tokens to a few "popular" experts while other experts sit idle. This wastes compute and creates a bottleneck on the overloaded GPUs.*

Why does this happen? Think about it from the optimization perspective. During training, if Expert 3 happens to produce slightly better outputs than the others early on, the router learns to send more tokens to Expert 3. More tokens means more gradient updates, which makes Expert 3 even better. Meanwhile, the neglected experts receive fewer tokens, fewer gradients, and fall further behind. This is a positive feedback loop — a **rich-get-richer** dynamic — that leads to **expert collapse**.

This is a serious problem for two reasons:

1. **Wasted capacity.** If only 2 out of 64 experts are used, we have a 64-expert model with the capacity of a 2-expert model. All those extra parameters are dead weight.
2. **GPU utilization collapse.** With expert parallelism, if Expert 3 receives 80% of tokens and sits on GPU 0, then GPU 0 is overloaded while GPUs holding the neglected experts are nearly idle. Training slows to the pace of the most overloaded GPU.


### The Auxiliary Load Balancing Loss

The standard solution is to add an **auxiliary loss** that penalizes uneven expert utilization. The Switch Transformer (Fedus et al., 2022) introduced a widely adopted formulation.

Define two quantities for each expert $i$ across a batch of $T$ tokens:

- $f_i$ = **fraction of tokens** routed to expert $i$:

$$f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[\text{token } t \text{ selects expert } i]$$

- $P_i$ = **average router probability** for expert $i$:

$$P_i = \frac{1}{T} \sum_{t=1}^{T} p_i^{(t)}$$

where $p_i^{(t)}$ is the router's softmax probability of expert $i$ for token $t$.

The auxiliary loss is:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where $\alpha$ is a small coefficient (typically 0.01) and the factor $N$ ensures the loss is properly scaled regardless of the number of experts.

Let us plug in some numbers to see how this works. Suppose $N = 4$ experts and $T = 100$ tokens:

- **Balanced case:** $f_i = 0.25$ for all $i$, $P_i = 0.25$ for all $i$. The loss is $\alpha \cdot 4 \cdot 4 \times (0.25 \times 0.25) = \alpha \cdot 4 \cdot 0.25 = \alpha$.
- **Imbalanced case:** Expert 1 gets 70% of tokens ($f_1 = 0.70, P_1 = 0.60$), and the other three share the rest. The loss becomes $\alpha \cdot 4 \cdot (0.70 \times 0.60 + 0.10 \times 0.13 + 0.10 \times 0.13 + 0.10 \times 0.14) = \alpha \cdot 4 \cdot (0.42 + 0.013 + 0.013 + 0.014) = \alpha \cdot 4 \cdot 0.46 = 1.84\alpha$.

The imbalanced case produces a loss that is 1.84x the balanced case. The gradient from this auxiliary loss pushes the router toward more uniform expert utilization. The coefficient $\alpha$ controls how strongly we enforce balance — too large and we sacrifice model quality, too small and the balancing is ineffective.

The total training loss becomes:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{aux}}$$


### Expert Capacity and Token Overflow

Even with the auxiliary loss, perfect balance is never achieved. In practice, we set a **capacity factor** $C$ that determines how many tokens each expert is allowed to process. If each expert should ideally receive $T \cdot k / N$ tokens (the perfectly balanced share), the expert capacity is:

$$\text{Capacity}_i = C \cdot \frac{T \cdot k}{N}$$

where $C > 1$ provides some headroom. A typical value is $C = 1.25$, meaning each expert can handle 25% more tokens than its fair share.

![Expert capacity: each expert has a maximum buffer size. Overflow tokens are dropped and passed through with a residual connection.](figures/figure_4.png)
*Expert capacity and overflow: each expert has a fixed buffer size. Tokens that exceed the capacity are dropped — their representation passes through unchanged via the residual connection.*

What happens to tokens that exceed capacity? They are **dropped** — their input is passed through to the output unchanged (via the residual connection), as if the MoE layer were an identity function for those tokens. This means some tokens receive no expert processing at all.

Let us work through an example. With $T = 1024$ tokens, $N = 8$ experts, $k = 2$, and $C = 1.25$:

$$\text{Ideal tokens per expert} = \frac{1024 \times 2}{8} = 256$$

$$\text{Capacity per expert} = 1.25 \times 256 = 320$$

If Expert 3 receives 400 tokens (because the router heavily favors it), 80 of those tokens will be dropped. Those 80 tokens get no expert processing in this layer — they simply pass through. This is wasteful, but it prevents any single expert from becoming a computational bottleneck that slows down the entire batch.

The trade-off is:
- **$C$ too small** (e.g., 1.0): Many tokens get dropped, hurting model quality
- **$C$ too large** (e.g., 2.0): More memory is needed per expert buffer, and load imbalance is tolerated, slowing down training
- **Sweet spot**: $C = 1.1$ to $C = 1.5$ works well in practice


### Expert Choice Routing: Flipping the Problem

An elegant alternative, proposed in the **Expert Choice** paper (Zhou et al., 2022), flips the routing direction. Instead of tokens choosing experts, **experts choose tokens**.

Each expert selects its top-$k_e$ tokens based on the router scores:

$$\text{Tokens for Expert}_i = \text{argtop}_{k_e}\big(p_i^{(1)}, p_i^{(2)}, \ldots, p_i^{(T)}\big)$$

where $k_e = C \cdot T / N$ is the fixed number of tokens each expert processes.

This guarantees perfect load balance by construction — every expert processes exactly $k_e$ tokens. There is no capacity overflow, no dropped tokens (from the expert's perspective), and no need for the auxiliary loss.

The trade-off is that different tokens may be processed by a different number of experts — some tokens might be selected by 4 experts (getting extra processing), while others might be selected by none (getting no expert processing). But in practice, this tends to work well because "important" tokens naturally get selected by more experts.


## Combining Expert Parallelism with Other Strategies

Expert parallelism does not exist in isolation. In real training systems, it is one dimension of a multi-dimensional parallelism grid. Let us see how all five strategies compose.

### The 5D Parallelism Grid

Modern large-scale training systems like Megatron-LM and DeepSeek organize GPUs along five independent axes:

![The 5D parallelism grid showing how thousands of GPUs are organized along Data, Tensor, Pipeline, Sequence/Context, and Expert parallelism dimensions.](figures/figure_5.png)
*The 5D parallelism grid: $\text{DP} \times \text{TP} \times \text{PP} \times \text{SP/CP} \times \text{EP}$. Each GPU sits at a unique coordinate in this grid, and each axis determines what is split and how communication happens.*

| Dimension | What is Split | Communication Pattern | Typical Scale |
|---|---|---|---|
| **Data Parallelism (DP)** | Training data (batches) | AllReduce (gradients) | 8 -- 512 GPUs |
| **Tensor Parallelism (TP)** | Weight matrices (columns/rows) | AllReduce within each layer | 2 -- 8 GPUs (within node) |
| **Pipeline Parallelism (PP)** | Transformer layers (stages) | Point-to-point (activations) | 2 -- 16 stages |
| **Sequence/Context Parallelism (SP/CP)** | Sequence dimension | Ring attention / AllGather | 2 -- 8 GPUs |
| **Expert Parallelism (EP)** | MoE experts | All-to-All | 4 -- 16 GPUs |

The total number of GPUs is the product of all dimensions:

$$\text{Total GPUs} = \text{DP} \times \text{TP} \times \text{PP} \times \text{SP/CP} \times \text{EP}$$

Let us work through a concrete example. DeepSeek-V3 trains on 2048 H800 GPUs with the following configuration:

- **EP** = 64 (64-way expert parallelism for the MoE layers)
- **TP** = 1 (no tensor parallelism — the attention is small enough)
- **PP** = 16 (16 pipeline stages)
- **DP** = 2 ($2048 / (64 \times 16 \times 1) = 2$)

Each attention layer uses the full set of GPUs for data parallelism (since attention parameters are replicated), but MoE layers use expert parallelism to distribute the 256 experts across 64 GPUs (4 experts per GPU). The All-to-All communication for expert dispatch happens within groups of 64 GPUs.

### Why Each Dimension Has Its Place

You might wonder: why not just use one parallelism strategy for everything? The answer is that each strategy has different trade-offs in communication cost, memory savings, and compute efficiency:

- **TP** requires the fastest interconnect (NVLink) because it communicates within every layer. It is limited to GPUs within a single node (typically 8).
- **PP** has lower communication bandwidth needs (only activations between stages) but introduces pipeline bubbles. It works well across nodes.
- **DP** requires AllReduce of all gradients, but this can overlap with backward computation. It scales across many nodes.
- **SP/CP** addresses the specific bottleneck of long-sequence attention memory.
- **EP** addresses the specific structure of MoE models, where experts are naturally independent.

The art of large-scale training is choosing the right size for each dimension based on your model architecture, hardware topology, and interconnect bandwidth.

```python
# Conceptual configuration for a 5D parallel training setup
# (This is pseudocode illustrating how frameworks like Megatron-LM configure parallelism)

parallel_config = {
    "data_parallel_size": 2,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 16,
    "context_parallel_size": 1,
    "expert_parallel_size": 64,
}

total_gpus = (
    parallel_config["data_parallel_size"]
    * parallel_config["tensor_parallel_size"]
    * parallel_config["pipeline_parallel_size"]
    * parallel_config["context_parallel_size"]
    * parallel_config["expert_parallel_size"]
)
print(f"Total GPUs required: {total_gpus}")  # 2048

# Expert parallelism group: 64 GPUs share experts via All-to-All
# Pipeline parallelism group: 16 stages in sequence
# Data parallelism group: 2 replicas for gradient averaging
```


## Course Summary: The Complete Parallelism Landscape

We have now covered every dimension of the parallelism grid that makes training models with hundreds of billions — or even trillions — of parameters possible. Let us step back and see the complete picture.

![The complete course overview: from a single GPU to the full 5D parallelism grid, showing how each strategy addresses a specific bottleneck.](figures/figure_6.png)
*The complete parallelism landscape. Each strategy addresses a specific bottleneck: memory (DP + ZeRO, TP, PP), compute (DP), sequence length (SP/CP), or model capacity (EP). Together, they enable training at scales that would be impossible with any single strategy alone.*

We started this course with a simple question: **how do you train a model that does not fit on a single GPU?** The answer, as we have seen, is not a single technique but a layered system of complementary strategies:

1. **GPU Fundamentals** (Pod 1): GPUs are throughput-oriented processors with thousands of cores, designed for the matrix multiplications that dominate deep learning.

2. **Memory Components** (Pod 2): Training memory is consumed by four things — weights, gradients, optimizer states, and activations. Understanding this breakdown is essential for choosing the right parallelism strategy.

3. **Activation Recomputation + Gradient Accumulation + Data Parallelism** (Pod 3): The first line of defense. Recomputation trades compute for memory. Accumulation simulates large batches. Data parallelism replicates the model and splits the data.

4. **Ring-AllReduce + Batch Size + Profiling** (Pod 4): How GPUs communicate efficiently, how to choose the right batch size as you scale, and how to find bottlenecks with profiling.

5. **ZeRO Optimizer** (Pod 5): Eliminates the memory redundancy in data parallelism by partitioning optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3) across GPUs.

6. **Tensor Parallelism** (Pod 6): Splits individual weight matrices across GPUs using column-linear and row-linear sharding, enabling layers that are too large for a single GPU.

7. **Sequence Parallelism** (Pod 7): Splits the sequence dimension for LayerNorm and dropout operations that would otherwise waste memory in tensor-parallel setups.

8. **Context Parallelism** (Pod 8): Extends sequence splitting to the attention computation using Ring Attention, enabling million-token context windows.

9. **Pipeline Parallelism** (Pod 9): Splits transformer layers across GPUs like an assembly line, using microbatch schedules like 1F1B to minimize the pipeline bubble.

10. **Expert Parallelism** (Pod 10 — this article): Distributes MoE experts across GPUs with All-to-All communication, using load balancing techniques to keep all GPUs busy.

Each strategy addresses a different bottleneck. Data parallelism and ZeRO address memory redundancy. Tensor and pipeline parallelism address models whose layers or depth exceed a single GPU. Sequence and context parallelism address the quadratic memory of long-sequence attention. Expert parallelism addresses the unique structure of sparse MoE models.

No single strategy is sufficient at the largest scales. The power comes from their composition — the 5D grid where each dimension independently addresses a different axis of the scaling challenge.


## Closing Thoughts

When you see headlines about models trained on 10,000 GPUs, those numbers are not brute force. They represent a carefully orchestrated system where every GPU knows exactly what slice of the computation it owns, what data it needs to send and receive, and how to stay synchronized with the rest. The five parallelism dimensions — Data, Tensor, Pipeline, Sequence/Context, and Expert — are the conceptual framework behind this orchestration.

Understanding these dimensions does not just help you train large models. It helps you **reason** about the fundamental trade-offs in distributed computation: memory versus communication, compute versus redundancy, load balance versus routing quality. These trade-offs appear everywhere in systems engineering, not just in deep learning.

If you have made it through all ten pods of this course, you now have a solid foundation for understanding how the largest models in the world are trained. The next time you read a systems paper from Google, Meta, or DeepSeek, the parallelism strategies they describe will not be mysterious — they will be familiar tools in your toolkit.

That's it! Thanks for learning with us.