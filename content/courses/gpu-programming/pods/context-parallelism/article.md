# Context Parallelism

*Training with million-token contexts by splitting the sequence across GPUs using Ring Attention and its optimized variants.*

---

Let us start with a concrete scenario. You are training a large language model and you want it to understand entire books, full codebases, or hours of transcribed conversation -- contexts that span one million tokens or more. You load a single long document into your model, and before the forward pass even reaches the second layer, you get an `OutOfMemoryError`. The sequence is just too long.

Where did the memory go? The answer lies in the self-attention mechanism. In a standard transformer, every token attends to every other token. If your sequence has $S$ tokens, the attention score matrix has shape $S \times S$. For a modest context of $S = 8{,}192$ tokens, that is 67 million entries -- manageable. But for $S = 131{,}072$ tokens (128K context), the matrix has over 17 billion entries. And for one million tokens? The attention matrix alone would have $10^{12}$ entries. Even in FP16, that is 2 terabytes -- for a single layer, for a single attention head.

This $O(S^2)$ memory scaling is the fundamental barrier to long-context training. No single GPU on the planet can hold it.

In an earlier pod, we studied **sequence parallelism**, which splits the sequence dimension across GPUs for operations like LayerNorm and dropout. But sequence parallelism deliberately avoids splitting the attention computation itself -- it leaves the full sequence on each GPU during attention, because every token needs to see every other token to compute the attention scores.

**Context parallelism** tackles exactly this problem. It splits the full sequence across GPUs for attention too -- and uses a clever communication pattern called **Ring Attention** to make sure every token still gets to attend to every other token, without ever materializing the full $S \times S$ matrix on any single device.

![Attention memory scaling showing O(seq_len^2) growth as sequence length increases.](figures/figure_1.png)
*Attention memory scaling: as sequence length doubles, the memory for the attention matrix quadruples. Beyond ~64K tokens, the attention matrix alone exceeds the memory of an 80 GB A100.*


## Why Attention is the Bottleneck

Let us make the $O(S^2)$ cost concrete with some numbers. For a single attention head with head dimension $d_h = 128$ and sequence length $S$, the attention computation involves:

1. Computing the attention score matrix: $\mathbf{A} = \mathbf{Q}\mathbf{K}^T$, which has shape $S \times S$
2. Applying softmax across each row of $\mathbf{A}$
3. Computing the output: $\mathbf{O} = \text{softmax}(\mathbf{A}) \cdot \mathbf{V}$

The score matrix $\mathbf{A}$ is the memory hog. Let us compute its size for different sequence lengths, for a single attention head in FP16 (2 bytes per element):

| Sequence Length $S$ | Attention Matrix Size | Memory (FP16) |
|---|---|---|
| 4,096 | $4{,}096 \times 4{,}096$ | 32 MB |
| 32,768 | $32{,}768 \times 32{,}768$ | 2 GB |
| 131,072 | $131{,}072 \times 131{,}072$ | 32 GB |
| 1,048,576 | $1{,}048{,}576 \times 1{,}048{,}576$ | 2 TB |

And this is per head. A typical model with 32 attention heads multiplies these numbers by 32. At $S = 131{,}072$, even a single layer's attention across all heads requires roughly 1 TB of memory. An 80 GB GPU does not stand a chance.

The key insight is: we do not actually need the full $S \times S$ matrix at once. The softmax and the output computation can be done in **chunks** -- processing pieces of the key-value pairs at a time and combining the partial results. This is exactly the idea behind FlashAttention (computing attention without materializing the full matrix in HBM) and, at a distributed level, Ring Attention.


## Ring Attention: The Core Idea

Ring Attention, introduced by Liu et al. (2023), distributes the attention computation across $N$ GPUs by partitioning the sequence into $N$ chunks and arranging the GPUs in a logical **ring**. Each GPU is responsible for computing one chunk of the query tokens' attention output, but it needs to attend to all key-value chunks from all GPUs.

Here is the setup. We have $N$ GPUs and a sequence of $S$ tokens. We split the sequence into $N$ contiguous chunks, each of length $S/N$:

- GPU 0 holds tokens $[0, S/N)$ -- both the query chunk $\mathbf{Q}_0$ and the corresponding key-value chunks $\mathbf{K}_0, \mathbf{V}_0$
- GPU 1 holds tokens $[S/N, 2S/N)$ -- query $\mathbf{Q}_1$, key-value $\mathbf{K}_1, \mathbf{V}_1$
- ...and so on

Each GPU needs to compute the attention output for its own query chunk $\mathbf{Q}_i$ against **all** key-value chunks $\mathbf{K}_0, \mathbf{V}_0, \mathbf{K}_1, \mathbf{V}_1, \ldots, \mathbf{K}_{N-1}, \mathbf{V}_{N-1}$. The naive approach would be to AllGather all the KV pairs to every GPU, but that would require $O(S)$ memory per GPU -- exactly what we are trying to avoid.

Ring Attention avoids this by passing the KV chunks around the ring, one step at a time. At each step, every GPU:

1. Computes a **partial attention** between its local queries and the KV chunk it currently holds
2. **Sends** its current KV chunk to the next GPU in the ring
3. **Receives** a new KV chunk from the previous GPU in the ring
4. Combines the new partial result with its running total using **online softmax**

After $N$ steps, every GPU has seen all $N$ KV chunks and has computed the full attention output for its own query chunk -- without ever needing more than one KV chunk in memory at a time.

![Ring Attention: 4 GPUs arranged in a ring, each holding a query chunk and passing KV chunks to the next neighbor.](figures/figure_2.png)
*Ring Attention: 4 GPUs in a ring. Each GPU holds its own Q chunk and one KV chunk at a time. After each attention computation, the KV chunk is passed to the next neighbor.*


## Ring Attention Step-by-Step

Let us trace through the algorithm with $N = 4$ GPUs and a sequence of $S = 4{,}096$ tokens, so each GPU holds a chunk of 1,024 tokens.

**Initial state.** Each GPU $i$ holds its local query chunk $\mathbf{Q}_i$ (which stays put for the entire algorithm) and its local KV chunk $(\mathbf{K}_i, \mathbf{V}_i)$.

**Step 1.** Each GPU computes partial attention between its queries and its locally held KV chunk:

$$\mathbf{A}_i^{(0)} = \mathbf{Q}_i \mathbf{K}_i^T / \sqrt{d_h}$$

Then it sends $(\mathbf{K}_i, \mathbf{V}_i)$ to GPU $(i+1) \bmod 4$ and receives $(\mathbf{K}_{(i-1) \bmod 4}, \mathbf{V}_{(i-1) \bmod 4})$ from GPU $(i-1) \bmod 4$.

**Step 2.** Each GPU now holds a different KV chunk. It computes partial attention with this new KV chunk, and combines the result with the running total from Step 1 using online softmax. Then it passes the KV chunk along to the next GPU.

**Step 3.** Same procedure -- compute, combine, pass.

**Step 4 (final).** After this step, every GPU has computed attention against all 4 KV chunks. GPU $i$ now holds the complete attention output $\mathbf{O}_i$ for its 1,024 tokens.

Let us write this more precisely. At step $t$, GPU $i$ holds KV chunk from GPU $(i - t) \bmod N$. The partial attention scores are:

$$\mathbf{S}_i^{(t)} = \mathbf{Q}_i \mathbf{K}_{(i-t) \bmod N}^T / \sqrt{d_h}$$

The key question is: how do we combine partial softmax results from different steps?

![Ring Attention step-by-step: 4 steps showing KV chunks rotating around the ring while queries stay fixed.](figures/figure_3.png)
*Ring Attention step-by-step with 4 GPUs. Queries (Q) stay fixed on each GPU. KV chunks rotate around the ring. After 4 steps, every GPU has attended to all KV chunks.*


## Online Softmax: Combining Partial Results

This is the mathematical heart of Ring Attention. Normally, softmax requires knowing the maximum value across the **entire** row (all $S$ positions) to compute the normalization. If we only have partial rows, we cannot compute the final softmax directly. But we can use the **online softmax** trick (Milakov and Gimelshein, 2018), which maintains a running maximum and a running normalizing sum.

Here is the idea. Suppose GPU $i$ has already processed KV chunks from steps $0$ through $t-1$, and it has accumulated:

- $\mathbf{m}_i^{(t-1)}$: the running row-wise maximum of the attention logits seen so far
- $\mathbf{l}_i^{(t-1)}$: the running sum of exponentiated logits (the softmax denominator)
- $\mathbf{O}_i^{(t-1)}$: the running numerator (weighted sum of values)

When a new KV chunk arrives at step $t$, GPU $i$ computes the new logits $\mathbf{S}_i^{(t)}$ and updates as follows:

**Step 1: Update the running maximum.**

$$\mathbf{m}_i^{(t)} = \max\left(\mathbf{m}_i^{(t-1)},\ \text{rowmax}(\mathbf{S}_i^{(t)})\right)$$

**Step 2: Rescale the old accumulator.**

$$\alpha = \exp\left(\mathbf{m}_i^{(t-1)} - \mathbf{m}_i^{(t)}\right)$$

**Step 3: Compute the new chunk's contribution.**

$$\beta = \exp\left(\mathbf{S}_i^{(t)} - \mathbf{m}_i^{(t)}\right)$$

**Step 4: Update the denominator and output.**

$$\mathbf{l}_i^{(t)} = \alpha \cdot \mathbf{l}_i^{(t-1)} + \text{rowsum}(\beta)$$

$$\mathbf{O}_i^{(t)} = \alpha \cdot \mathbf{O}_i^{(t-1)} + \beta \cdot \mathbf{V}_{(i-t) \bmod N}$$

After all $N$ steps, the final output is $\mathbf{O}_i^{(N-1)} / \mathbf{l}_i^{(N-1)}$.

Let us plug in a small numerical example to make sure this makes sense. Suppose we have two KV chunks and a single query token with $d_h = 1$ (just to keep the numbers simple). The full logits across both chunks are $[2.0, 1.0, 3.0, 0.5]$.

**After chunk 1** (logits $[2.0, 1.0]$, values $[v_0, v_1]$):
- $m^{(0)} = 2.0$
- $l^{(0)} = e^{2.0 - 2.0} + e^{1.0 - 2.0} = 1.0 + 0.368 = 1.368$
- $O^{(0)} = 1.0 \cdot v_0 + 0.368 \cdot v_1$

**After chunk 2** (logits $[3.0, 0.5]$, values $[v_2, v_3]$):
- $m^{(1)} = \max(2.0, 3.0) = 3.0$
- $\alpha = e^{2.0 - 3.0} = 0.368$
- $\beta_0 = e^{3.0 - 3.0} = 1.0$, $\beta_1 = e^{0.5 - 3.0} = 0.082$
- $l^{(1)} = 0.368 \times 1.368 + 1.0 + 0.082 = 0.503 + 1.082 = 1.585$
- $O^{(1)} = 0.368 \times (1.0 \cdot v_0 + 0.368 \cdot v_1) + 1.0 \cdot v_2 + 0.082 \cdot v_3$

The final output is $O^{(1)} / l^{(1)}$. You can verify that this gives exactly the same result as computing the full softmax over all four logits at once. This is exactly what we want -- numerically identical attention, computed in chunks, with no full-size matrix ever stored.

Here is a simplified implementation of the online softmax accumulation:

```python
import torch

def ring_attention_step(Q_local, K_chunk, V_chunk,
                        running_max, running_sum, running_out):
    """One step of Ring Attention: process a new KV chunk."""
    d_h = Q_local.shape[-1]

    # Compute attention logits for this chunk
    logits = Q_local @ K_chunk.transpose(-2, -1) / (d_h ** 0.5)

    # Online softmax update
    chunk_max = logits.max(dim=-1, keepdim=True).values
    new_max = torch.maximum(running_max, chunk_max)

    # Rescale old accumulator to new max
    alpha = torch.exp(running_max - new_max)
    # Exponentiate new logits under new max
    beta = torch.exp(logits - new_max)

    # Update running denominator and numerator
    new_sum = alpha * running_sum + beta.sum(dim=-1, keepdim=True)
    new_out = alpha * running_out + beta @ V_chunk

    return new_max, new_sum, new_out
```


## Overlapping Communication with Computation

You might be wondering: if every GPU must send and receive KV chunks at each step, does the communication not slow everything down? This brings us to one of Ring Attention's most elegant properties: the communication can be **completely overlapped** with computation.

Here is why. At each step, while GPU $i$ is computing the partial attention between $\mathbf{Q}_i$ and the current KV chunk, it can simultaneously be sending that KV chunk to GPU $(i+1)$ and receiving the next KV chunk from GPU $(i-1)$. The attention computation (a matrix multiplication) takes significant time on the GPU's compute units, while the KV transfer uses the network interconnect (NVLink or InfiniBand). These are independent hardware resources -- they can operate in parallel.

In practice, if the chunk size is large enough that the attention computation takes longer than the KV transfer, the communication is entirely hidden. The total time per step is determined only by the computation, as if there were no communication at all.

Let us check this with concrete numbers. For a chunk of $S/N = 32{,}768$ tokens (131K sequence across 4 GPUs), head dimension $d_h = 128$, and 32 heads, the KV chunk size per head is:

$$\text{KV chunk size} = 2 \times 32{,}768 \times 128 \times 2 \text{ bytes (FP16)} = 16 \text{ MB per head}$$

Across 32 heads: $32 \times 16 = 512$ MB. Over NVLink at 600 GB/s, this transfer takes roughly 0.85 ms.

The attention computation for one head (a $32{,}768 \times 32{,}768$ matmul plus the softmax and value projection) involves approximately $2 \times 32{,}768^2 \times 128 \approx 275$ billion FLOPs. On an A100 at 312 TFLOP/s FP16, this takes roughly 0.88 ms per head -- but since we have 32 heads running sequentially, the total compute time per step is around 28 ms, far exceeding the 0.85 ms communication. The communication is fully hidden.

This is the same principle we saw with Ring-AllReduce: arrange GPUs in a ring, overlap transfer with computation, and the communication effectively disappears from the critical path.


## The Causal Attention Problem

Everything we have described so far works perfectly for bidirectional (non-causal) attention, where every token attends to every other token. But autoregressive language models use **causal attention**: token $i$ can only attend to tokens $0, 1, \ldots, i$. It cannot look into the future.

This creates a load balancing problem. Consider 4 GPUs with chunks of a causal sequence:

- GPU 0 holds the earliest tokens (positions 0 to $S/4 - 1$). Its queries can only attend to KV chunk 0. The attention to chunks 1, 2, and 3 is entirely masked out.
- GPU 1 holds positions $S/4$ to $S/2 - 1$. Its queries attend to KV chunks 0 and 1, but chunks 2 and 3 are masked.
- GPU 2 attends to chunks 0, 1, and 2.
- GPU 3 attends to all four chunks.

This is a severe imbalance. GPU 0 does roughly $1/4$ the work of GPU 3. During Ring Attention, GPU 0 can skip 3 of the 4 steps entirely (since the attention mask zeros out those chunks), while GPU 3 must work through all 4 steps. The other GPUs sit idle waiting for GPU 3 to finish.

The computational load forms a **triangle**: the causal mask means GPU $i$ performs roughly $(i+1)/N$ of the full attention work. For $N = 4$, the loads are 25%, 50%, 75%, and 100%. The slowest GPU determines the wall-clock time, so the effective utilization is:

$$\text{Utilization} = \frac{\text{Average load}}{\text{Max load}} = \frac{(1 + 2 + 3 + 4) / 4}{4} = \frac{2.5}{4} = 62.5\%$$

That means 37.5% of the GPU-seconds are wasted on idle time. For $N = 8$ GPUs, utilization drops to about 56%. This is clearly unacceptable for large-scale training.

![Causal attention imbalance: the triangular mask means early-token GPUs do far less work than late-token GPUs.](figures/figure_4.png)
*Causal attention load imbalance with contiguous chunking. The triangular causal mask means GPU 0 (earliest tokens) does only 25% of the work, while GPU 3 (latest tokens) does 100%. GPUs 0-2 sit idle waiting for GPU 3.*


## Striped Attention: Load Balancing for Causal Models

**Striped Attention** (Brandon et al., 2023) fixes this load imbalance with a simple but powerful idea: instead of assigning **contiguous** chunks of the sequence to each GPU, assign tokens in a **round-robin** (striped) pattern.

With $N = 4$ GPUs:
- **Contiguous**: GPU 0 gets tokens [0, 1, 2, ...S/4-1], GPU 1 gets [S/4, ...S/2-1], etc.
- **Striped**: GPU 0 gets tokens [0, 4, 8, 12, ...], GPU 1 gets [1, 5, 9, 13, ...], GPU 2 gets [2, 6, 10, 14, ...], GPU 3 gets [3, 7, 11, 15, ...]

Now every GPU holds a mix of early, middle, and late tokens. The causal mask no longer creates a triangular imbalance -- each GPU has roughly the same number of tokens that attend to each KV chunk.

Let us see why this balances the load. Consider a query at position $p$ on GPU $i$ (so $p = i + kN$ for some integer $k$). Under causal masking, this query attends to all positions $\leq p$. The number of positions $\leq p$ that belong to KV chunk $j$ is approximately $\lfloor p/N \rfloor + 1$ for each $j \leq (p \bmod N)$, and $\lfloor p/N \rfloor$ otherwise. The key point is that these counts are nearly identical across all GPU indices $j$ -- the load is spread evenly.

With Striped Attention, every step of Ring Attention involves roughly the same amount of non-masked computation across all GPUs. The utilization jumps from 62.5% (contiguous) to nearly 100% (striped) for 4 GPUs, and from 56% to nearly 100% for 8 GPUs.

![Striped attention: tokens are assigned in round-robin fashion so every GPU holds a mix of early and late positions.](figures/figure_5.png)
*Striped Attention for load balancing. Instead of contiguous chunks, tokens are assigned round-robin. Every GPU holds a mix of early and late positions, so the causal mask creates equal work across all GPUs.*

Striped Attention is used in practice by systems like Llama 3.1's long-context training, where it is combined with Ring Attention to train with 128K-token contexts across hundreds of GPUs.


## Memory Analysis: How Much Do We Save?

Let us now quantify the memory savings from context parallelism. The dominant memory consumer in attention is the attention score matrix. Without any parallelism, a single GPU must hold the full $S \times S$ matrix (per head):

$$\text{Memory (no parallelism)} = O(S^2)$$

With context parallelism across $N$ GPUs, each GPU holds queries of length $S/N$ and processes one KV chunk of length $S/N$ at a time. The partial attention score matrix at each step has shape $(S/N) \times (S/N)$:

$$\text{Memory per GPU (context parallel)} = O\left(\frac{S^2}{N^2}\right)$$

This is a **quadratic** reduction per GPU! Let us plug in numbers for $S = 131{,}072$ and $N = 8$:

- **Without CP**: $131{,}072^2 \times 2 \text{ bytes} \approx 32 \text{ GB per head}$
- **With CP (N=8)**: $(16{,}384)^2 \times 2 \text{ bytes} \approx 512 \text{ MB per head}$

That is a 64x reduction in attention memory per GPU. Combined with FlashAttention (which avoids materializing even the chunk-level attention matrix in HBM), the per-GPU memory for attention becomes $O(S/N)$ -- linear in the chunk size.

The KV memory per GPU is also $O(S/N)$: each GPU stores its own KV chunk plus one incoming KV chunk at a time, which is $2 \times (S/N) \times d_h$ per head.

Here is a practical implementation sketch that shows how context parallelism integrates into a transformer forward pass:

```python
import torch
import torch.distributed as dist

def context_parallel_attention(Q, K, V, rank, world_size):
    """
    Context-parallel attention using Ring Attention.
    Q, K, V: local chunks of shape (batch, seq_len // world_size, num_heads, d_head)
    """
    chunk_len = Q.shape[1]
    d_h = Q.shape[-1]

    # Initialize online softmax accumulators
    running_max = torch.full((Q.shape[0], chunk_len, Q.shape[2], 1),
                             float('-inf'), device=Q.device)
    running_sum = torch.zeros_like(running_max)
    running_out = torch.zeros_like(Q)

    # Current KV chunk (starts with our own)
    kv_k = K.clone()
    kv_v = V.clone()

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1) % world_size

    for step in range(world_size):
        # --- Compute partial attention with current KV chunk ---
        logits = torch.einsum('bshd,bthd->bsht', Q, kv_k) / (d_h ** 0.5)

        # Online softmax update
        chunk_max = logits.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, chunk_max)
        alpha = torch.exp(running_max - new_max)
        beta = torch.exp(logits - new_max)

        running_sum = alpha * running_sum + beta.sum(dim=-1, keepdim=True)
        running_out = alpha * running_out + torch.einsum('bsht,bthd->bshd', beta, kv_v)
        running_max = new_max

        # --- Rotate KV to next GPU (overlapped with compute in practice) ---
        if step < world_size - 1:
            recv_k = torch.empty_like(kv_k)
            recv_v = torch.empty_like(kv_v)
            dist.sendrecv(kv_k, send_rank, recv_k, recv_rank)
            dist.sendrecv(kv_v, send_rank, recv_v, recv_rank)
            kv_k, kv_v = recv_k, recv_v

    # Final normalization
    output = running_out / running_sum
    return output
```

This is a simplified version for pedagogical clarity. Production implementations (like those in Megatron-LM and PyTorch's `torch.distributed`) overlap the send/recv with the attention computation using CUDA streams, and combine this with FlashAttention kernels for maximum efficiency.


## When to Use Context Parallelism

Context parallelism is not always necessary. Here is a practical guide:

- **Sequence length < 8K**: Standard attention (or FlashAttention on a single GPU) is sufficient. No need for context parallelism.
- **Sequence length 8K-64K**: FlashAttention handles the memory, but you may want context parallelism if activation memory is still tight alongside other parallelism dimensions.
- **Sequence length 64K-1M+**: Context parallelism is essential. The attention memory simply cannot fit on a single GPU, regardless of other optimizations.

Context parallelism composes naturally with other parallelism strategies. In Meta's Llama 3.1 training setup, they use:
- **Tensor parallelism** (8-way) within a node
- **Context parallelism** (8-way) for 128K-token sequences
- **Pipeline parallelism** across nodes
- **Data parallelism** across the remaining GPUs

Each parallelism dimension addresses a different bottleneck. Tensor parallelism splits the weight matrices. Context parallelism splits the sequence for attention. Data parallelism splits the training data. And pipeline parallelism splits the layers.


## Summary

Let us recap what we have covered:

1. **The $O(S^2)$ attention bottleneck**: Self-attention memory grows quadratically with sequence length, making long-context training impossible on a single GPU.

2. **Context parallelism**: Splits the sequence across GPUs for the attention computation itself -- going beyond what sequence parallelism (which only handles LayerNorm and dropout) can do.

3. **Ring Attention**: Arranges GPUs in a ring and rotates KV chunks step by step. Online softmax allows each GPU to accumulate partial attention results without ever holding the full attention matrix. Communication is overlapped with computation using independent hardware resources.

4. **Causal attention imbalance**: Contiguous sequence chunks create a triangular load imbalance where early-token GPUs do far less work than late-token GPUs.

5. **Striped Attention**: Assigns tokens round-robin across GPUs, ensuring every GPU has a balanced mix of early and late positions. This restores near-100% utilization under causal masking.

6. **Memory savings**: Per-GPU attention memory drops from $O(S^2)$ to $O(S^2/N^2)$, or $O(S/N)$ when combined with FlashAttention.

Context parallelism splits the sequence for attention. But what about splitting across layers? If your model has 80 transformer layers and your GPU can only hold 20 layers worth of weights and states, you need to distribute different layers to different GPUs -- turning the model into an assembly line. This brings us to **Pipeline Parallelism**, where we will study how to split layers across GPUs, handle the infamous "pipeline bubble" of idle time, and implement scheduling strategies like 1F1B that minimize wasted compute.
