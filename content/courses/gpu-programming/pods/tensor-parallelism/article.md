# Tensor Parallelism

*Split individual weight matrices across GPUs -- column-linear, row-linear, and how tensor parallelism works inside a transformer block.*

---

Let us start with a concrete scenario. You have a transformer model with a hidden dimension of 12,288 and an MLP intermediate dimension of 49,152. A single weight matrix in the MLP has shape $(12288, 49152)$, which means it contains about 600 million parameters. In FP16, that single matrix takes 1.2 GB. The full model has dozens of these matrices, plus attention weights, layer norms, and embeddings. Altogether, the model does not fit on a single GPU.

In earlier pods, we learned about data parallelism and ZeRO, which either replicate or partition the model's states across GPUs. But those approaches share a common assumption: each GPU eventually performs the full forward pass through every layer. What if we could split the work *within* a single layer -- so that no GPU ever needs to hold an entire weight matrix?

This is the core idea behind **tensor parallelism**: take a single weight matrix, slice it into pieces, and distribute those pieces across multiple GPUs. Each GPU computes only a *portion* of the matrix multiplication, and the partial results are combined through a single communication operation. The result is mathematically identical to running the full matrix multiply on one GPU, but the memory and compute are distributed.

The approach was introduced by Shoeybi et al. in the **Megatron-LM** paper (2019) and has since become a standard building block in every large-scale training framework. In this article, we will build tensor parallelism from first principles. We will understand two fundamental strategies -- column-linear and row-linear parallelism -- and then see how Megatron-LM combines them inside a transformer block to minimize communication.


## Matrix Multiplication: The Building Block

Before we split any matrices, let us make sure we have a clear picture of what a linear layer does. A linear layer (without bias) computes:

$$Y = X W$$

where $X$ has shape $(b, k)$ -- $b$ tokens (or batch elements) and $k$ features -- and $W$ has shape $(k, n)$. The output $Y$ has shape $(b, n)$.

In a transformer MLP, the first linear layer projects from hidden dimension $h$ to intermediate dimension $4h$, so $k = h$ and $n = 4h$. The second linear layer projects back: $k = 4h$ and $n = h$.

The question is: how can we split this computation across $T$ GPUs (where $T$ is the tensor parallelism degree)? There are exactly two natural ways to partition the weight matrix $W$: split it along its columns, or split it along its rows.


## Column-Linear Parallelism: Split $W$ Along Columns

The first strategy is to partition $W$ along its **column dimension**. We split $W$ of shape $(k, n)$ into $T$ pieces:

$$W = \begin{bmatrix} W_1 & W_2 & \cdots & W_T \end{bmatrix}$$

where each $W_i$ has shape $(k, n/T)$. GPU $i$ stores only $W_i$.

Now, the full computation $Y = XW$ can be decomposed as:

$$Y = X \begin{bmatrix} W_1 & W_2 & \cdots & W_T \end{bmatrix} = \begin{bmatrix} X W_1 & X W_2 & \cdots & X W_T \end{bmatrix}$$

Each GPU $i$ computes $Y_i = X W_i$ independently, producing a partial output of shape $(b, n/T)$. The full output $Y$ is obtained by **concatenating** all the partial outputs along the last dimension:

$$Y = \text{concat}(Y_1, Y_2, \ldots, Y_T)$$

Let us plug in some simple numbers to see this clearly. Suppose we have $T = 2$ GPUs, $b = 1$ (a single token), $k = 4$, and $n = 4$. The input is $X = [1, 2, 3, 4]$ and the weight matrix is:

$$W = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix}$$

We split $W$ along columns into two halves:

$$W_1 = \begin{bmatrix} 1 & 2 \\ 5 & 6 \\ 9 & 10 \\ 13 & 14 \end{bmatrix}, \quad W_2 = \begin{bmatrix} 3 & 4 \\ 7 & 8 \\ 11 & 12 \\ 15 & 16 \end{bmatrix}$$

GPU 0 computes $Y_1 = X W_1 = [1 \cdot 1 + 2 \cdot 5 + 3 \cdot 9 + 4 \cdot 13, \; 1 \cdot 2 + 2 \cdot 6 + 3 \cdot 10 + 4 \cdot 14] = [90, 100]$.

GPU 1 computes $Y_2 = X W_2 = [1 \cdot 3 + 2 \cdot 7 + 3 \cdot 11 + 4 \cdot 15, \; 1 \cdot 4 + 2 \cdot 8 + 3 \cdot 12 + 4 \cdot 16] = [110, 120]$.

Concatenating: $Y = [90, 100, 110, 120]$. You can verify this matches the full matrix multiply $XW$. This is exactly what we want.

![Column-linear parallelism: the weight matrix is split along columns, each GPU computes a slice of the output, and the results are concatenated.](figures/figure_1.png)
*Column-linear parallelism: weight matrix $W$ is split along columns across 2 GPUs. Each GPU computes a partial output, and the results are concatenated.*

The critical observation about column-linear parallelism is:

- **Input $X$ must be available on every GPU** (replicated).
- **No communication is needed during the forward pass** -- each GPU computes its slice independently.
- The outputs are naturally split across GPUs, which is useful if the next operation can consume split inputs.


## Row-Linear Parallelism: Split $W$ Along Rows

The second strategy is to partition $W$ along its **row dimension**. We split $W$ of shape $(k, n)$ into $T$ pieces:

$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_T \end{bmatrix}$$

where each $W_i$ has shape $(k/T, n)$. But here is the catch: if $W$ is split along rows, then the **input $X$ must also be split** along its last dimension to make the matrix multiply work. GPU $i$ gets the slice $X_i$ of shape $(b, k/T)$.

Each GPU computes a partial product:

$$Y_i = X_i W_i$$

Each $Y_i$ has shape $(b, n)$ -- the full output width! But each $Y_i$ is only a *partial sum*. The full output is:

$$Y = X W = X_1 W_1 + X_2 W_2 + \cdots + X_T W_T = \sum_{i=1}^{T} Y_i$$

To get the final result, we need to **sum** all partial results across GPUs. This is an **allreduce** operation.

Let us trace through the same numbers. With $T = 2$, we split $W$ along rows:

$$W_1 = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \end{bmatrix}, \quad W_2 = \begin{bmatrix} 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix}$$

And split $X = [1, 2, 3, 4]$ correspondingly: $X_1 = [1, 2]$, $X_2 = [3, 4]$.

GPU 0: $Y_1 = X_1 W_1 = [1 \cdot 1 + 2 \cdot 5, \; 1 \cdot 2 + 2 \cdot 6, \; 1 \cdot 3 + 2 \cdot 7, \; 1 \cdot 4 + 2 \cdot 8] = [11, 14, 17, 20]$.

GPU 1: $Y_2 = X_2 W_2 = [3 \cdot 9 + 4 \cdot 13, \; 3 \cdot 10 + 4 \cdot 14, \; 3 \cdot 11 + 4 \cdot 15, \; 3 \cdot 12 + 4 \cdot 16] = [79, 86, 93, 100]$.

Allreduce (sum): $Y = [11 + 79, \; 14 + 86, \; 17 + 93, \; 20 + 100] = [90, 100, 110, 120]$. The same answer as before. This is exactly what we want.

![Row-linear parallelism: the weight matrix is split along rows, each GPU computes a partial sum, and results are combined via allreduce.](figures/figure_2.png)
*Row-linear parallelism: weight matrix $W$ is split along rows across 2 GPUs. Each GPU computes a partial result, and an allreduce (sum) produces the final output on every GPU.*

The key properties of row-linear parallelism are:

- **Input must be split** across GPUs along its feature dimension.
- **Allreduce is required** to combine partial sums -- this is a communication cost.
- **Output is replicated** on every GPU after the allreduce, which is useful if the next operation needs the full output (like a layer norm or residual connection).


## The Megatron-LM Insight: Combining Column and Row

Now the question is: which strategy should we use? Column-linear or row-linear? The brilliant insight of Megatron-LM is that you do not choose one -- you use **both**, one after the other, arranged so that the output of the first feeds naturally into the input of the second, eliminating unnecessary communication.

Consider the MLP block of a transformer. It has two linear layers with a GeLU activation in between:

$$\text{MLP}(X) = \text{GeLU}(X W_1) W_2$$

Here $W_1$ has shape $(h, 4h)$ and $W_2$ has shape $(4h, h)$. Megatron-LM applies tensor parallelism as follows:

1. **First linear ($W_1$): column-parallel.** Split $W_1$ along columns. Each GPU $i$ computes $Z_i = X W_{1,i}$, producing a partial hidden state of shape $(b, 4h/T)$.

2. **GeLU activation: applied locally.** Each GPU applies GeLU to its $Z_i$ independently. This works because GeLU is an element-wise operation -- it does not mix values across the feature dimension. No communication needed.

3. **Second linear ($W_2$): row-parallel.** The column-parallel first layer produced outputs that are split along the feature dimension -- which is exactly the input format that row-parallel needs! Split $W_2$ along rows, where each $W_{2,i}$ has shape $(4h/T, h)$. GPU $i$ computes $Y_i = \text{GeLU}(Z_i) \cdot W_{2,i}$, producing a partial sum of shape $(b, h)$.

4. **Allreduce.** Sum the partial results $Y_i$ across all GPUs to get the final MLP output.

The key insight here is that by choosing column-parallel for the first layer and row-parallel for the second, the intermediate representations naturally have the right shape at every step. The GeLU activations do not require any communication. The **only communication in the entire MLP block is a single allreduce** at the very end.

Let us also account for the residual connection. In a transformer, the output of the MLP is added to the input: $\text{output} = X + \text{MLP}(X)$. Since the allreduce produces the full MLP output on every GPU, and $X$ is already replicated, the residual addition happens locally with no extra communication.

![Megatron-LM MLP block: the first linear layer is column-parallel, GeLU is applied locally, the second linear layer is row-parallel, followed by a single allreduce.](figures/figure_3.png)
*Megatron-LM MLP block: column-parallel first layer, local GeLU, row-parallel second layer, one allreduce. The residual connection adds to the result locally.*


## Tensor Parallelism in Self-Attention

The same column-then-row strategy applies to the self-attention layer, but here the parallelism has an even more natural structure.

In multi-head attention, the input $X$ is projected into queries, keys, and values using three weight matrices:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

where $W_Q, W_K, W_V$ each have shape $(h, h)$. The output of attention is then projected back:

$$\text{Attn}(X) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \cdot W_O$$

where $W_O$ has shape $(h, h)$ is the output projection.

Now, multi-head attention already splits $Q$, $K$, and $V$ into $A$ heads, where each head operates on a slice of dimension $d_k = h / A$. Tensor parallelism exploits this directly: **assign different attention heads to different GPUs**.

With $T$ GPUs, each GPU gets $A/T$ attention heads. The parallelism works as follows:

1. **$W_Q, W_K, W_V$: column-parallel.** Each weight matrix is split along columns. GPU $i$ stores the columns corresponding to its assigned heads and computes $Q_i, K_i, V_i$ -- the queries, keys, and values for its subset of heads.

2. **Attention computation: local.** Each GPU independently computes attention for its heads. Since attention heads are completely independent (they do not share information), no communication is needed. GPU $i$ computes $\text{softmax}(Q_i K_i^T / \sqrt{d_k}) V_i$, which produces an output of shape $(b, s, h/T)$ where $s$ is the sequence length.

3. **Output projection $W_O$: row-parallel.** The attention output on each GPU has dimension $h/T$, which is exactly the input format for a row-parallel linear layer. GPU $i$ multiplies its attention output by its row-shard of $W_O$, producing a partial result of shape $(b, s, h)$.

4. **Allreduce.** Sum the partial results across GPUs.

Again, the residual connection ($X + \text{Attn}(X)$) is computed locally since both $X$ and the allreduced output are available on every GPU.

![Self-attention with tensor parallelism: Q, K, V projections are column-parallel split by attention heads, attention is computed locally, and the output projection is row-parallel with an allreduce.](figures/figure_4.png)
*Tensor-parallel self-attention: Q, K, V are column-parallel (split by heads), each GPU runs attention independently, the output projection is row-parallel, and one allreduce produces the final result.*


## Communication Cost: Two Allreduces Per Layer

Let us count the total communication for one transformer layer. Each layer has:

1. **Self-attention block**: one allreduce (after the output projection)
2. **MLP block**: one allreduce (after the second linear layer)

That gives us **two allreduce operations per transformer layer**. For a model with $L$ layers, the total is $2L$ allreduces during the forward pass, and another $2L$ during the backward pass (for gradient synchronization).

How large is each allreduce? The tensors being reduced have shape $(b, s, h)$ -- the full activations. In FP16 with $b = 1$, $s = 2048$, $h = 12288$:

$$\text{Allreduce size} = 1 \times 2048 \times 12288 \times 2 \text{ bytes} \approx 48 \text{ MB}$$

For a model with 96 layers, the total allreduce volume per forward pass is $2 \times 96 \times 48 \approx 9.2$ GB. This is a significant amount of data, and it must be transferred at every training step.

This is why tensor parallelism works best **within a single node**, where GPUs are connected by high-bandwidth **NVLink** interconnects. NVLink on an A100 provides 600 GB/s of bidirectional bandwidth between GPUs on the same node. Compare this with inter-node networking (typically InfiniBand at 200-400 Gb/s, or about 25-50 GB/s) -- NVLink is roughly 10-20x faster.

The rule of thumb in practice:
- **Tensor parallelism degree $T$ = number of GPUs within a node** (typically 4 or 8)
- **Data parallelism and pipeline parallelism** handle scaling across nodes

Trying to run tensor parallelism across nodes introduces a communication bottleneck at every layer that destroys throughput.


## Implementation: A Column-Parallel Linear Layer

Let us implement a column-parallel linear layer to make the concepts concrete. This is a simplified version of what Megatron-LM uses internally.

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """Linear layer with weight split along the output dimension (columns)."""

    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert out_features % world_size == 0
        self.out_features_per_gpu = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # Each GPU stores only its shard of the weight
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_gpu)
        )

    def forward(self, x):
        # x has shape (batch, seq_len, in_features) -- replicated on all GPUs
        # Each GPU computes its slice of the output
        output = x @ self.weight  # (batch, seq_len, out_features_per_gpu)
        return output  # Output is split across GPUs along last dim

class RowParallelLinear(nn.Module):
    """Linear layer with weight split along the input dimension (rows)."""

    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_gpu = in_features // world_size
        self.rank = rank
        self.world_size = world_size

        # Each GPU stores only its shard of the weight
        self.weight = nn.Parameter(
            torch.randn(self.in_features_per_gpu, out_features)
        )

    def forward(self, x):
        # x has shape (batch, seq_len, in_features_per_gpu) -- split across GPUs
        partial = x @ self.weight  # (batch, seq_len, out_features)
        # Allreduce: sum partial results across all GPUs
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial  # Output is replicated on all GPUs
```

And here is how the tensor-parallel MLP block combines them:

```python
class TensorParallelMLP(nn.Module):
    """MLP block with Megatron-LM style tensor parallelism."""

    def __init__(self, hidden_dim, world_size, rank):
        super().__init__()
        intermediate_dim = 4 * hidden_dim
        # First linear: column-parallel (split output dim)
        self.fc1 = ColumnParallelLinear(
            hidden_dim, intermediate_dim, world_size, rank
        )
        # Second linear: row-parallel (split input dim)
        self.fc2 = RowParallelLinear(
            intermediate_dim, hidden_dim, world_size, rank
        )

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim) -- replicated
        z = self.fc1(x)        # (batch, seq_len, intermediate_dim // world_size) -- split
        z = torch.nn.functional.gelu(z)  # Element-wise, no communication
        y = self.fc2(z)        # (batch, seq_len, hidden_dim) -- allreduce inside
        return y               # Replicated output, ready for residual add
```

Let us walk through this code. The `ColumnParallelLinear` takes the full input $X$ (replicated on every GPU) and multiplies it by its column shard of $W_1$, producing a partial output. The GeLU activation is element-wise, so it runs locally without communication. The `RowParallelLinear` takes the split intermediate activations, multiplies by its row shard of $W_2$, and then calls `dist.all_reduce` to sum the partial results. After the allreduce, every GPU has the identical full output -- ready to be added to the residual.


## The Full Transformer Block

Now let us see the complete picture. A single tensor-parallel transformer layer has:

1. **LayerNorm** -- applied to the replicated input $X$. Each GPU computes this independently (same input, same output). No communication.

2. **Self-attention** (column-parallel Q, K, V; local attention; row-parallel output projection) -- **one allreduce**.

3. **Residual add** -- local, since both the input and the allreduced attention output are replicated.

4. **LayerNorm** -- again local on replicated input.

5. **MLP** (column-parallel first linear; local GeLU; row-parallel second linear) -- **one allreduce**.

6. **Residual add** -- local.

![Full tensor-parallel transformer block showing where communication occurs: one allreduce after attention and one after the MLP.](figures/figure_5.png)
*Full tensor-parallel transformer block. Communication happens at exactly two points: after the attention output projection and after the MLP second linear layer. Everything else -- LayerNorm, GeLU, residual connections, attention computation -- runs locally on each GPU.*

The total communication per layer is **two allreduce operations** in the forward pass and **two allreduce operations** in the backward pass. That is it. Every other computation in the transformer layer happens independently on each GPU.


## Why Tensor Parallelism Shines Within a Node

Let us put concrete numbers on why tensor parallelism is a within-node strategy. Consider an 8-GPU node with NVLink, training a model with $h = 8192$ and $s = 4096$.

Each allreduce transfers a tensor of size $b \times s \times h \times 2$ bytes. With $b = 1$:

$$\text{Tensor size} = 1 \times 4096 \times 8192 \times 2 = 64 \text{ MB}$$

With NVLink at ~300 GB/s effective per GPU, this allreduce takes roughly:

$$\text{Time} \approx \frac{2 \times 64 \text{ MB}}{300 \text{ GB/s}} \approx 0.4 \text{ ms}$$

The factor of 2 accounts for the ring-allreduce algorithm transferring approximately $2\times$ the data. On NVLink, 0.4 ms is well within the time budget -- the compute for each layer takes several milliseconds, so communication can be overlapped or is a small fraction of total time.

Now imagine running the same allreduce over InfiniBand at 25 GB/s:

$$\text{Time} \approx \frac{128 \text{ MB}}{25 \text{ GB/s}} \approx 5 \text{ ms}$$

That is over 10x slower. With two allreduces per layer and 96 layers, you would spend nearly a full second per forward pass just on communication. This would dominate training time.

This is why tensor parallelism is almost always confined to a single node. Across nodes, the preferred strategies are data parallelism, ZeRO, and pipeline parallelism, which have much lower communication frequency.


## Summary

Let us recap the key ideas:

1. **Tensor parallelism** splits individual weight matrices across GPUs, distributing both memory and computation for each layer.

2. **Column-linear parallelism** splits $W$ along columns. Each GPU computes a slice of the output. The outputs are concatenated. No communication needed in the forward pass.

3. **Row-linear parallelism** splits $W$ along rows. Each GPU computes a partial sum. An allreduce combines the results.

4. **Megatron-LM** uses column-parallel for the first linear layer and row-parallel for the second, so the intermediate activations flow naturally between them -- requiring only **one allreduce per sub-block** (attention or MLP).

5. **Self-attention** is naturally parallelizable by heads: assign different heads to different GPUs, with column-parallel QKV projections and a row-parallel output projection.

6. The total cost is **two allreduce operations per transformer layer**, making tensor parallelism best suited for **within-node** communication over fast NVLink.

But there is a hidden cost we have glossed over. Notice that LayerNorm and dropout operate on the **full, replicated** activations. Every GPU stores the complete activation tensor for these operations, even though each GPU only needs its slice for the attention and MLP computations. For long sequences and large batches, this replicated memory adds up significantly. This brings us to **Sequence Parallelism** -- a technique that splits these replicated operations along the sequence dimension to eliminate this wasted memory.
