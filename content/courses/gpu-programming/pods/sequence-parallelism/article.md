# Sequence Parallelism

*Why LayerNorm and dropout waste memory in tensor parallelism, and how sequence parallelism fixes it by splitting the sequence dimension.*

---

Let us start with a specific scenario that you will encounter the moment you deploy tensor parallelism in practice. You have a large transformer model — say 30 billion parameters — and you are splitting the weight matrices of each layer across 4 GPUs using tensor parallelism. The MLP columns are split, the attention heads are split, and each GPU computes only its fraction of the matrix multiplications. Everything looks efficient on paper. But then you profile the actual memory usage on each GPU and discover something strange: **LayerNorm and dropout are consuming the same amount of activation memory as if you had no parallelism at all.**

How is that possible? You went through the trouble of splitting the model across GPUs precisely to reduce the per-GPU memory. Yet some layers are completely ignoring the parallelism and forcing every GPU to hold full-sized activations. This is not a small problem — for long sequences and large hidden dimensions, the activation memory consumed by LayerNorm and dropout can dominate the total activation footprint.

This is the problem that **Sequence Parallelism** solves. Introduced in the Megatron-LM v3 paper (Korthikanti et al., 2022), sequence parallelism eliminates this wasted memory by splitting the sequence dimension for the non-tensor-parallel regions of the transformer. And it does so with **zero additional communication cost** — by cleverly replacing one collective operation with two others that together move the same amount of data.

In this article, we will understand exactly why this memory waste happens, how sequence parallelism fixes it, and how the two techniques combine to produce a full transformer layer where every single byte of activation memory is distributed across GPUs.


## The Problem: Replicated Activations in Tensor Parallelism

To understand the problem, let us trace what happens inside a single transformer layer when we use tensor parallelism with $N$ GPUs. Recall that a transformer layer has this structure:

$$\text{Output} = x + \text{MLP}(\text{LayerNorm}(x + \text{Attention}(\text{LayerNorm}(x))))$$

Tensor parallelism splits the heavy operations — the attention projections and the MLP linear layers — across GPUs. Each GPU computes a slice of the matrix multiplication. But what about LayerNorm and dropout? These operations sit *between* the tensor-parallel regions, and they have a fundamental property that causes trouble.

**LayerNorm needs the full hidden dimension.** LayerNorm computes the mean and variance across the hidden dimension of each token. If the hidden dimension were split across GPUs, each GPU would only have a partial view and could not compute the correct statistics. So in standard tensor parallelism, the output of the tensor-parallel region is **allreduced** (gathered and summed) before being fed into LayerNorm. This means LayerNorm receives the full, un-split activation tensor.

**Dropout operates element-wise on the full tensor.** Since dropout follows LayerNorm (or sits in the residual path), it also receives the full activation tensor.

Let us quantify the waste. Consider a transformer with hidden dimension $h = 8192$, sequence length $s = 4096$, micro-batch size $b = 1$, and tensor parallelism degree $N = 4$ GPUs. Each activation tensor has shape $(b, s, h) = (1, 4096, 8192)$.

For the tensor-parallel MLP and attention regions, each GPU stores activations of size:

$$\text{TP activation per GPU} = b \times s \times \frac{h}{N} = 1 \times 4096 \times 2048 = 8\text{M elements}$$

But for LayerNorm and dropout, each GPU stores:

$$\text{Non-TP activation per GPU} = b \times s \times h = 1 \times 4096 \times 8192 = 33.5\text{M elements}$$

Every GPU holds the **full** 33.5 million element tensor for every LayerNorm and dropout layer. With 4 GPUs, the total memory consumed by these non-tensor-parallel regions across the system is $4 \times 33.5\text{M} = 134\text{M}$ elements — four times what is actually needed if we could distribute them.

In a standard transformer layer, there are two LayerNorms, two dropout operations, and one residual addition that operate on the full tensor. These non-tensor-parallel regions can account for a significant fraction of the total activation memory. The Megatron-LM team found that for large models, these replicated activations constitute roughly **one-third** of the total activation memory per layer.

![Tensor parallelism with replicated LayerNorm: each GPU stores the full activation for non-TP operations.](figures/figure_1.png)
*Tensor parallelism with replicated LayerNorm: while the MLP and attention activations are split across GPUs, every GPU redundantly stores the full activation tensor for LayerNorm and dropout — wasting memory.*


## The Insight: Split Along the Sequence Dimension

The key observation is elegantly simple. LayerNorm computes statistics **independently for each token**. The mean and variance are computed across the hidden dimension for a single token — there is no interaction between different tokens in the sequence. The same is true for dropout: each element is dropped independently. And the residual addition is purely element-wise.

This means that while these operations need the **full hidden dimension** (which is why we cannot split them the same way we split the MLP), they **do not need the full sequence**. Each token can be processed completely independently.

So here is the idea: instead of giving every GPU all $s$ tokens with the full hidden dimension $h$, we give each GPU only $s/N$ tokens with the full hidden dimension $h$. Each GPU applies LayerNorm, dropout, and residual addition to its own slice of the sequence. No GPU needs to see any other GPU's tokens for these operations.

This is **sequence parallelism**: splitting the activation tensor along the sequence dimension for the non-tensor-parallel regions of the transformer.

The activation memory for LayerNorm and dropout per GPU becomes:

$$\text{SP activation per GPU} = b \times \frac{s}{N} \times h$$

Let us plug in our numbers from before: $b = 1$, $s = 4096$, $h = 8192$, $N = 4$.

$$\text{SP activation per GPU} = 1 \times \frac{4096}{4} \times 8192 = 1 \times 1024 \times 8192 = 8.4\text{M elements}$$

Compare this to the original 33.5M elements per GPU. We have reduced the activation memory for these regions by a factor of $N = 4$. This is exactly what we want.

![Sequence parallelism: each GPU holds only seq_len/N tokens for LayerNorm and dropout.](figures/figure_2.png)
*Sequence parallelism: instead of every GPU storing the full sequence for LayerNorm and dropout, each GPU stores only $s/N$ tokens. The total memory is the same as before, but now it is distributed instead of replicated.*


## The Communication Pattern: AllGather and Reduce-Scatter

Now comes the question you should be asking: if the activations are split along the sequence dimension in the LayerNorm/dropout regions but split along the hidden dimension in the tensor-parallel regions, how do we transition between these two layouts? We need some form of communication to rearrange the data.

Let us trace the forward pass through a single transformer layer to see what communication is needed.

**Step 1: Sequence-parallel region (LayerNorm).** Each GPU holds a slice of shape $(b, s/N, h)$ — its portion of the sequence, full hidden dimension. It applies LayerNorm to its local tokens.

**Step 2: Transition to tensor-parallel region (Attention/MLP).** The attention and MLP layers need the full sequence on each GPU, but only a slice of the hidden dimension. So we need to go from "each GPU has $s/N$ tokens, full $h$" to "each GPU has all $s$ tokens, $h/N$ of the hidden dimension." The operation that does this is an **AllGather** along the sequence dimension, combined with the column-parallel split of the weight matrix. In practice, this is implemented as a single **AllGather**: each GPU contributes its $(b, s/N, h)$ chunk, and every GPU ends up with the full $(b, s, h)$ tensor, which then enters the column-parallel linear layer that produces a $(b, s, h/N)$ output.

**Step 3: Tensor-parallel region.** Each GPU computes its portion of the attention or MLP on the full sequence but with $h/N$ hidden units. This is standard tensor parallelism.

**Step 4: Transition back to sequence-parallel region.** After the row-parallel linear layer, we need to go from "each GPU has all $s$ tokens with partial sums across $h$" back to "each GPU has $s/N$ tokens, full $h$." The operation that does this is a **Reduce-Scatter**: each GPU's partial result is summed across GPUs (the reduce) and the result is split across GPUs along the sequence dimension (the scatter). Each GPU ends up with shape $(b, s/N, h)$ — its slice of the sequence with the correctly reduced hidden dimension.

Here is the critical insight that makes sequence parallelism free: **In standard tensor parallelism, the transition between layers uses an AllReduce. An AllReduce is mathematically equivalent to a Reduce-Scatter followed by an AllGather** (or equivalently, an AllGather followed by a Reduce-Scatter). The total data moved is the same.

Let us verify this with a bandwidth analysis. For an AllReduce of a tensor with $M$ elements across $N$ GPUs, each GPU sends and receives:

$$\text{AllReduce data per GPU} = 2 \cdot \frac{N-1}{N} \cdot M$$

For a Reduce-Scatter plus an AllGather of the same tensor:

$$\text{Reduce-Scatter data per GPU} = \frac{N-1}{N} \cdot M$$
$$\text{AllGather data per GPU} = \frac{N-1}{N} \cdot M$$
$$\text{Total} = 2 \cdot \frac{N-1}{N} \cdot M$$

The total communication volume is identical. We have not added any extra communication — we have simply **decomposed** the AllReduce into its two constituent operations and placed them at different points in the computation. The Reduce-Scatter happens at the exit of the tensor-parallel region (transitioning to sequence-parallel), and the AllGather happens at the entry of the next tensor-parallel region (transitioning from sequence-parallel).

Let us plug in numbers to make this concrete. Suppose $M = b \times s \times h = 1 \times 4096 \times 8192 = 33.5\text{M}$ elements in BF16 (2 bytes each), and $N = 4$ GPUs.

With standard tensor parallelism (AllReduce):

$$\text{Data per GPU} = 2 \times \frac{3}{4} \times 33.5\text{M} \times 2 \text{ bytes} = 100.5 \text{ MB}$$

With sequence parallelism (Reduce-Scatter + AllGather):

$$\text{Reduce-Scatter} = \frac{3}{4} \times 33.5\text{M} \times 2 \text{ bytes} = 50.25 \text{ MB}$$
$$\text{AllGather} = \frac{3}{4} \times 33.5\text{M} \times 2 \text{ bytes} = 50.25 \text{ MB}$$
$$\text{Total} = 100.5 \text{ MB}$$

Same total bandwidth. Zero extra cost. This is the elegance of the approach.

![Communication pattern: AllGather transitions into TP regions, Reduce-Scatter transitions out.](figures/figure_3.png)
*Communication pattern in sequence parallelism: AllGather collects the full sequence before entering the tensor-parallel region, Reduce-Scatter distributes the result back along the sequence dimension. The total bandwidth equals one AllReduce — exactly what standard tensor parallelism already uses.*


## The Combined TP + SP Transformer Layer

Now let us put everything together and trace through a complete transformer layer with both tensor parallelism and sequence parallelism. This is what a single layer looks like in Megatron-LM v3.

**Entering the layer.** The input activation arrives in sequence-parallel layout: each GPU holds shape $(b, s/N, h)$.

**LayerNorm 1.** Each GPU applies LayerNorm to its local $s/N$ tokens. No communication needed — each token is independent.

**AllGather.** Each GPU gathers the full sequence from all GPUs: $(b, s/N, h) \rightarrow (b, s, h)$. Now every GPU has the full sequence.

**Column-parallel Attention.** Each GPU computes its portion of the attention heads on the full sequence. The $Q$, $K$, $V$ projection matrices are split by columns, so each GPU computes $h/N$ attention dimensions. The output on each GPU has shape $(b, s, h/N)$.

**Row-parallel output projection + Reduce-Scatter.** Each GPU multiplies by its row-slice of the output projection matrix, producing a partial sum of shape $(b, s, h)$. The Reduce-Scatter sums these partial results and distributes the output along the sequence dimension: $(b, s, h) \rightarrow (b, s/N, h)$.

**Dropout + Residual Addition (sequence-parallel).** Each GPU applies dropout and adds the residual connection to its local $s/N$ tokens.

**LayerNorm 2.** Same as LayerNorm 1 — each GPU processes its own $s/N$ tokens locally.

**AllGather.** Same transition as before: $(b, s/N, h) \rightarrow (b, s, h)$.

**Column-parallel MLP (first linear).** Each GPU computes its slice of the up-projection: $(b, s, h) \rightarrow (b, s, 4h/N)$. Activation function (GeLU) is applied locally.

**Row-parallel MLP (second linear) + Reduce-Scatter.** Each GPU computes its portion and the result is reduce-scattered back: $(b, s, 4h/N) \rightarrow (b, s/N, h)$.

**Dropout + Residual Addition (sequence-parallel).** Again, each GPU operates on its local $s/N$ tokens.

Notice the pattern: the layer alternates between two regimes. In the **sequence-parallel regime** (LayerNorm, dropout, residual add), activations are split as $(b, s/N, h)$. In the **tensor-parallel regime** (attention, MLP), activations are split as $(b, s, h/N)$. The AllGather and Reduce-Scatter operations act as bridges between these two regimes.

Here is a simplified PyTorch-style pseudocode that shows this structure:

```python
class TPSPTransformerLayer(nn.Module):
    """Transformer layer with Tensor + Sequence Parallelism."""

    def __init__(self, hidden_dim, num_heads, tp_degree):
        super().__init__()
        self.ln1 = LayerNorm(hidden_dim)           # Runs in SP mode
        self.ln2 = LayerNorm(hidden_dim)           # Runs in SP mode
        self.attn = TPAttention(hidden_dim, num_heads, tp_degree)
        self.mlp = TPMLP(hidden_dim, tp_degree)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_sp):
        # x_sp shape: (batch, seq_len/N, hidden) — sequence-parallel layout

        # --- Attention block ---
        residual = x_sp
        x_sp = self.ln1(x_sp)                     # Local LayerNorm on s/N tokens
        x_full = all_gather(x_sp, dim=1)           # (b, s/N, h) -> (b, s, h)
        x_tp = self.attn(x_full)                   # TP attention: (b, s, h/N) internal
        x_sp = reduce_scatter(x_tp, dim=1)         # (b, s, h) -> (b, s/N, h)
        x_sp = self.dropout(x_sp) + residual       # Local dropout + residual

        # --- MLP block ---
        residual = x_sp
        x_sp = self.ln2(x_sp)                     # Local LayerNorm on s/N tokens
        x_full = all_gather(x_sp, dim=1)           # (b, s/N, h) -> (b, s, h)
        x_tp = self.mlp(x_full)                    # TP MLP: (b, s, 4h/N) internal
        x_sp = reduce_scatter(x_tp, dim=1)         # (b, s, h) -> (b, s/N, h)
        x_sp = self.dropout(x_sp) + residual       # Local dropout + residual

        return x_sp  # (batch, seq_len/N, hidden)
```

Each `all_gather` and `reduce_scatter` call involves inter-GPU communication. But as we showed above, the total communication volume per layer is the same as what standard tensor parallelism already requires. We have simply reorganized *where* that communication happens.


## Memory Savings: A Concrete Accounting

Let us do a careful memory accounting to understand what sequence parallelism actually saves. Consider a single transformer layer with:

- Hidden dimension $h = 12288$ (roughly GPT-3 175B scale)
- Sequence length $s = 2048$
- Micro-batch size $b = 1$
- Tensor parallelism degree $N = 8$ GPUs
- BF16 precision (2 bytes per element)

In a standard transformer layer, the major stored activations (for the backward pass) include: the inputs to each LayerNorm, the dropout masks, the attention scores, and the inputs to each linear layer.

For the **non-tensor-parallel regions** (2 LayerNorms, 2 dropouts, residual connections), each activation has shape $(b, s, h)$:

$$\text{Elements per activation} = 1 \times 2048 \times 12288 = 25.2\text{M}$$

With standard TP (replicated on every GPU):

$$\text{Memory per GPU for non-TP activations} = 5 \times 25.2\text{M} \times 2 \text{ bytes} = 252 \text{ MB}$$

(The factor of 5 accounts for the saved tensors across 2 LayerNorms, 2 dropout masks, and 1 residual.)

With TP + SP (split along sequence):

$$\text{Memory per GPU for SP activations} = 5 \times \frac{25.2\text{M}}{8} \times 2 \text{ bytes} = 31.5 \text{ MB}$$

That is a reduction from 252 MB to 31.5 MB — a factor of $N = 8$ — just for the non-tensor-parallel activations. For a model with 96 layers, this translates to a total saving of:

$$\text{Total saved} = 96 \times (252 - 31.5) \text{ MB} = 96 \times 220.5 \text{ MB} \approx 21 \text{ GB per GPU}$$

That is 21 GB of memory freed up on each GPU, simply by distributing the LayerNorm and dropout activations instead of replicating them. This is memory that can be used for larger batch sizes, longer sequences, or fewer activation recomputation checkpoints.

The tensor-parallel regions (attention, MLP) already have their activations split and are unaffected by this change. So the total activation memory saving from adding sequence parallelism to an existing tensor-parallel setup is:

$$\text{Activation reduction factor for non-TP ops} = N$$

This is the core result. For the operations that were previously replicated, sequence parallelism reduces their activation memory by the tensor parallelism degree.


## Implementation Considerations

There are a few practical details worth noting for anyone implementing this.

**The AllGather and Reduce-Scatter must be fused with the linear layers.** In practice, you do not want to first AllGather the full tensor into a separate buffer and then feed it into the linear layer — that would temporarily double the memory usage. Instead, the AllGather is overlapped with the matrix multiplication: as each chunk arrives, it is immediately consumed by the GEMM (general matrix multiplication). Megatron-LM implements this through custom CUDA kernels.

**Dropout masks must be consistent.** In the sequence-parallel region, each GPU applies dropout only to its own $s/N$ tokens. During the backward pass, the same dropout mask must be applied. Since each GPU generates its own mask independently (using its own random state), this works naturally as long as the random seed is set correctly per GPU.

**The sequence length must be divisible by N.** If the sequence length is not evenly divisible by the tensor parallelism degree, you need to pad the sequence. In practice, training sequences are almost always powers of 2 (1024, 2048, 4096), and TP degrees are small powers of 2 (2, 4, 8), so this is rarely an issue.

Here is a minimal implementation of the communication primitives:

```python
import torch
import torch.distributed as dist

def all_gather_along_seq(x, tp_group):
    """Gather sequence chunks from all GPUs.

    Input:  (batch, seq_len/N, hidden) on each GPU
    Output: (batch, seq_len, hidden)   on each GPU
    """
    world_size = dist.get_world_size(tp_group)
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x, group=tp_group)
    return torch.cat(chunks, dim=1)  # Concatenate along sequence dimension

def reduce_scatter_along_seq(x, tp_group):
    """Reduce partial sums and scatter along sequence dimension.

    Input:  (batch, seq_len, hidden) on each GPU (partial sums)
    Output: (batch, seq_len/N, hidden) on each GPU (fully reduced)
    """
    world_size = dist.get_world_size(tp_group)
    seq_len = x.shape[1]
    chunk_size = seq_len // world_size

    # Split along sequence dimension
    chunks = list(x.split(chunk_size, dim=1))
    output = torch.empty_like(chunks[0])
    dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM, group=tp_group)
    return output
```

And here is how you would use these in a forward pass:

```python
def forward_with_sp(x_sp, ln, linear_col, linear_row, tp_group):
    """Forward pass for one sub-block (attention or MLP) with TP + SP.

    x_sp: (batch, seq/N, hidden) — input in sequence-parallel layout
    """
    # 1. LayerNorm on local sequence chunk (no communication)
    x_sp = ln(x_sp)

    # 2. AllGather: (b, s/N, h) -> (b, s, h)
    x_full = all_gather_along_seq(x_sp, tp_group)

    # 3. Column-parallel linear: (b, s, h) -> (b, s, h/N)
    x_tp = linear_col(x_full)

    # ... activation function, attention, etc. ...

    # 4. Row-parallel linear: (b, s, h/N) -> (b, s, h) [partial sums]
    x_partial = linear_row(x_tp)

    # 5. Reduce-Scatter: (b, s, h) -> (b, s/N, h)
    x_sp = reduce_scatter_along_seq(x_partial, tp_group)

    return x_sp
```


## Putting It All Together: The Full Picture

Let us now look at the complete picture of a transformer layer with combined tensor parallelism and sequence parallelism.

![Full transformer layer combining TP and SP: sequence-parallel regions alternate with tensor-parallel regions.](figures/figure_4.png)
*A complete transformer layer with TP + SP. The layer alternates between sequence-parallel regions (LayerNorm, dropout, residual — activations split along sequence) and tensor-parallel regions (attention, MLP — activations split along hidden dimension). AllGather and Reduce-Scatter bridge the transitions, using exactly the same bandwidth as the AllReduce in standard TP.*

The summary of what each GPU holds at each stage:

| Stage | Layout | Shape per GPU | Dimension Split |
|-------|--------|---------------|-----------------|
| LayerNorm, Dropout, Residual | Sequence-parallel | $(b, s/N, h)$ | Sequence |
| Attention / MLP internals | Tensor-parallel | $(b, s, h/N)$ | Hidden |
| AllGather (SP $\rightarrow$ TP) | Communication | — | — |
| Reduce-Scatter (TP $\rightarrow$ SP) | Communication | — | — |

The communication cost per layer is:

$$\text{Forward: } 2 \times \text{AllGather} + 2 \times \text{ReduceScatter} = 2 \times \text{AllReduce}$$

This is the same as standard tensor parallelism, which also requires 2 AllReduces per layer (one for the attention block, one for the MLP block). The backward pass has the same communication pattern (with the roles of AllGather and Reduce-Scatter swapped), so the total communication cost remains identical.

What we gain is purely in **memory**: every activation tensor in the non-tensor-parallel regions is now $N$ times smaller per GPU. The operations themselves (LayerNorm, dropout, residual add) are trivially parallelized since they are independent across tokens.


## Why This Matters for Scale

The memory savings from sequence parallelism become increasingly important as models get larger. Here is a comparison for different model sizes with $N = 8$ tensor parallelism:

| Model | Hidden $h$ | Layers $L$ | Non-TP Activation (TP only) | Non-TP Activation (TP + SP) | Saved per GPU |
|-------|-----------|-----------|----------------------------|-----------------------------|---------------|
| 7B   | 4096      | 32        | 2.6 GB                     | 0.33 GB                     | 2.3 GB        |
| 30B  | 7168      | 60        | 8.6 GB                     | 1.1 GB                      | 7.5 GB        |
| 175B | 12288     | 96        | 23.6 GB                    | 2.9 GB                      | 20.7 GB       |

(Assuming $s = 2048$, $b = 1$, BF16. Non-TP activations include LayerNorm inputs, dropout masks, and residuals.)

For the 175B model, that is over 20 GB saved per GPU. On an 80 GB A100, this is the difference between fitting the model and not fitting the model — or the difference between needing activation recomputation and not needing it.


## From Sequence Parallelism to Context Parallelism

Sequence parallelism, as we described it here, operates **within a tensor parallelism group**. It splits the sequence dimension among the same $N$ GPUs that are already doing tensor parallelism. The sequence length is divided by the TP degree — typically 2, 4, or 8. This works well for reducing activation memory, but it does not help when the sequence itself is so long that the attention mechanism becomes the bottleneck.

Consider a model processing a 1-million-token context. Even with $N = 8$ tensor parallelism, each GPU still handles $1{,}000{,}000 / 8 = 125{,}000$ tokens during the attention computation. The attention score matrix has shape $(125{,}000, 1{,}000{,}000)$ — that is 125 billion elements. In BF16, that is 250 GB just for the attention scores. No single GPU can hold that.

This is where **Context Parallelism** comes in — a technique that splits the sequence across a *separate* group of GPUs specifically to handle long-context attention. While sequence parallelism splits the sequence for LayerNorm and dropout (operations that are independent per token), context parallelism splits the sequence for attention itself (an operation where tokens interact with each other). This requires a fundamentally different communication pattern — one that we will explore in detail.

That is it for sequence parallelism. The core idea is simple but powerful: LayerNorm and dropout do not need to see the full sequence, so do not give it to them. Split the sequence, save the memory, and pay nothing extra in communication.
