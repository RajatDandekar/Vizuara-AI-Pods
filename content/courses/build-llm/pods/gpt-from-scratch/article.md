# Building a GPT-Style Model from Scratch: Forward Pass, Loss, and Backpropagation

Let us start with a simple thought experiment. Imagine you are teaching a child to write stories. You do not hand them a grammar textbook or a dictionary. Instead, you sit them down with thousands of books and say: "Read this sentence, and before you turn the page, try to guess the next word." The child reads "The cat sat on the..." and guesses "mat." You reveal the answer. Sometimes they are right, sometimes they are wrong, but over time, something remarkable happens. The child begins to internalize grammar, vocabulary, world knowledge, and even a sense of narrative structure — all from this single, deceptively simple task of predicting the next word.

This is exactly how GPT works.

GPT — Generative Pre-trained Transformer — is one of the most influential language model architectures in modern AI. From GPT-2 to GPT-4, the core mechanism is the same: predict the next token, compute how wrong you were, and adjust your weights to be less wrong next time. In this article, we will build a GPT-style model completely from scratch. We will walk through every component: how text becomes numbers, how those numbers flow through the model (the forward pass), how we measure error (the loss function), and how the model learns from its mistakes (backpropagation).

By the end, you will have a complete, working GPT in PyTorch that you can train on your own text.

What is remarkable is the simplicity. Despite the impressive outputs, the entire training process boils down to three steps that repeat over and over: run the input through the model (forward pass), measure how wrong the prediction was (loss), and adjust the model to be less wrong (backpropagation). Every GPT model ever built — from the original 117 million parameter GPT-2 to the trillion-parameter behemoths — follows this exact loop.

Let us build it from the ground up.


![The GPT training loop: predict, measure error, update weights](figures/figure_1.png)
*The GPT training loop: predict, measure error, update weights*


---

## The GPT Architecture: Why Decoder-Only?

Before we start building, let us understand what GPT actually is at a high level.

The original Transformer architecture from the 2017 paper "Attention Is All You Need" had two parts: an **encoder** (which reads the input) and a **decoder** (which generates the output). This was designed for tasks like translation, where you read a sentence in French and produce a sentence in English.

GPT takes a bold shortcut: it throws away the encoder entirely and keeps only the decoder. Why? Because GPT's job is not to translate between two sequences — it is to generate text one token at a time. It reads what has been written so far and predicts what comes next. For this, a decoder is all you need.

Think about it this way: if you are writing a story, you do not need to first read and encode a separate input in another language. You simply need to look at what you have written so far and decide what word comes next. That is exactly what a decoder-only model does.

The key ingredient that makes this work is **causal masking** (also called autoregressive masking). When the model processes a sequence, each token is only allowed to attend to the tokens that came before it — never to future tokens. This is exactly like our child reading a book: when they are at word number 5, they can look back at words 1 through 4, but they cannot peek ahead at word 6.

Have a look at the diagram below:


![Causal masking: each token can only attend to previous tokens](figures/figure_2.png)
*Causal masking: each token can only attend to previous tokens*


This constraint is what makes GPT autoregressive — it generates tokens one at a time, left to right, each prediction conditioned only on the past.

Now let us build this model, piece by piece.

---

## Token Embeddings and Positional Embeddings: From Text to Vectors

The first challenge is fundamental: neural networks work with numbers, not words. So how do we convert raw text into something a neural network can process?

This happens in two steps.

**Step 1: Tokenization and Token Embeddings**

First, we break the input text into tokens. A token might be a word, a subword, or even a single character, depending on the tokenizer. For simplicity, let us assume we have a vocabulary of 10,000 tokens, and each token is assigned a unique integer ID.

The sentence "The cat sat" might become the token IDs: [42, 891, 1537].

Now, each token ID is converted into a dense vector using a **token embedding table**. This is simply a large matrix of shape (vocabulary_size, embedding_dim). If our embedding dimension is 64, then the token embedding table has shape (10000, 64). To get the embedding for token 42, we simply look up row 42 of this table.


$$
\mathbf{e}_{\text{token}} = \text{EmbeddingTable}[\text{token\_id}] \in \mathbb{R}^{d_{\text{model}}}
$$

Let us plug in some simple numbers. Suppose our vocabulary size is 10,000 and our embedding dimension is 64. Token ID 42 ("The") gets mapped to a vector of 64 numbers — say [0.12, -0.34, 0.56, ...]. Token ID 891 ("cat") gets mapped to a different vector of 64 numbers. These vectors are initially random, but during training, the model learns to place semantically similar tokens close together in this 64-dimensional space. This is exactly what we want.

**Step 2: Positional Embeddings**

There is a problem with token embeddings alone: they contain no information about the order of the tokens. The sequence "cat sat the" would produce the same set of embeddings as "the cat sat" — just in a different order. But order matters enormously in language.

To fix this, we add **positional embeddings**. GPT uses a learned positional embedding table of shape (max_sequence_length, embedding_dim). Position 0 gets one vector, position 1 gets another, and so on.

The final input to the model is the sum of both embeddings:


$$
\mathbf{x}_i = \mathbf{e}_{\text{token}}(i) + \mathbf{e}_{\text{position}}(i)
$$

Let us plug in some simple numbers. Suppose our token embedding for "cat" at position 1 is [0.5, 0.3, -0.1, 0.7] and our positional embedding for position 1 is [0.1, -0.2, 0.4, 0.0]. The combined input vector is:

$$[0.5 + 0.1, \; 0.3 + (-0.2), \; -0.1 + 0.4, \; 0.7 + 0.0] = [0.6, \; 0.1, \; 0.3, \; 0.7]$$

This combined vector now encodes both **what** the token is and **where** it appears. This is exactly what we want.


![Token embeddings plus positional embeddings give the model input](figures/figure_3.png)
*Token embeddings plus positional embeddings give the model input*


---

## The Forward Pass: Step by Step

Now we arrive at the heart of the model. The forward pass is the journey that our input vectors take through the entire network to produce a prediction. Let us trace this path step by step.

### Step 1: Input Embedding (already done)

We combine token embeddings and positional embeddings to get our input matrix **X** of shape (sequence_length, d_model).

### Step 2: Masked Multi-Head Self-Attention

This is the most important component of the Transformer. Let us build the intuition before diving into the math.

Imagine you are reading the sentence "The animal didn't cross the street because it was too tired." When you reach the word "it," your brain automatically looks back at the earlier words and figures out that "it" refers to "the animal." Self-attention is the mechanism that allows the model to do exactly this — each token looks at all previous tokens and decides which ones are most relevant.

For each token, we compute three vectors: a **Query** (what am I looking for?), a **Key** (what do I contain?), and a **Value** (what information do I carry?).

Let us use an analogy. Imagine you are at a library and you have a question (the Query). You walk up to each book on the shelf and check its title (the Key). The degree to which your question matches a book's title determines how much attention you pay to that book's content (the Value). Self-attention works the same way — the Query of one token is compared against the Keys of all other tokens to determine how much of each token's Value to absorb.

These three vectors are computed by multiplying the input by three learned weight matrices:


$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$


The attention scores are then computed as:


$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right) V
$$

Let us break this down and plug in some simple numbers.

Suppose we have 3 tokens and d_k = 4. Our Q and K matrices might look like:

$$Q = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 \end{bmatrix}$$

First, compute QK^T (the raw attention scores):

$$QK^T = \begin{bmatrix} 1 & 1 & 2 \\ 1 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}$$

Divide by sqrt(d_k) = sqrt(4) = 2:

$$\frac{QK^T}{\sqrt{d_k}} = \begin{bmatrix} 0.5 & 0.5 & 1.0 \\ 0.5 & 0.5 & 0.0 \\ 0.5 & 0.0 & 0.5 \end{bmatrix}$$

Now apply the causal mask — set future positions to negative infinity:

$$\text{Masked} = \begin{bmatrix} 0.5 & -\infty & -\infty \\ 0.5 & 0.5 & -\infty \\ 0.5 & 0.0 & 0.5 \end{bmatrix}$$

After softmax (each row sums to 1):

- Row 1: softmax([0.5, -inf, -inf]) = [1.0, 0.0, 0.0]
- Row 2: softmax([0.5, 0.5, -inf]) = [0.5, 0.5, 0.0]
- Row 3: softmax([0.5, 0.0, 0.5]) = [0.37, 0.26, 0.37]

This tells us that token 1 only attends to itself (it has no history), token 2 attends equally to tokens 1 and 2, and token 3 attends to all three with varying weights. This is exactly what we want — each token only looks at the past and present, never the future.

The division by sqrt(d_k) is important. Without it, when d_k is large, the dot products become very large, pushing the softmax into regions where it has extremely small gradients. Dividing by sqrt(d_k) keeps the values in a range where softmax behaves well.

**Multi-Head Attention** simply means we run this attention computation multiple times in parallel (say, 4 or 8 heads), each with different learned weight matrices. Each head can learn to focus on different types of relationships — one head might learn syntax, another might learn co-reference, another might learn semantic similarity. The outputs of all heads are concatenated and projected through a final linear layer.

Why multiple heads? A single attention head can only compute one set of attention weights — it can only "look for" one type of pattern at a time. But language is rich with multiple simultaneous relationships. In the sentence "The cat, which was orange, sat on the warm mat," we simultaneously need to track that "sat" relates to "cat" (subject-verb), that "orange" describes "cat" (adjective-noun), and that "warm" describes "mat." Multiple heads allow the model to track all of these patterns in parallel.


![Multi-head self-attention with causal masking across four heads](figures/figure_4.png)
*Multi-head self-attention with causal masking across four heads*


### Step 3: Residual Connection + Layer Normalization

After the attention block, we add the original input back to the output. This is called a **residual connection** (or skip connection):


$$
\mathbf{x} = \mathbf{x} + \text{Attention}(\mathbf{x})
$$


Then we apply **Layer Normalization**, which normalizes the values across the feature dimension to have zero mean and unit variance:


$$
\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Let us plug in some simple numbers. Suppose after the residual connection, we have a vector x = [4.0, 2.0, 6.0, 0.0]. The mean is mu = (4+2+6+0)/4 = 3.0 and the variance is sigma^2 = ((1)^2 + (-1)^2 + (3)^2 + (-3)^2)/4 = 5.0. So:

$$\text{Normalized} = \frac{[4.0 - 3.0, \; 2.0 - 3.0, \; 6.0 - 3.0, \; 0.0 - 3.0]}{\sqrt{5.0 + 0.00001}} = \frac{[1.0, \; -1.0, \; 3.0, \; -3.0]}{2.236} = [0.447, \; -0.447, \; 1.342, \; -1.342]$$

If gamma = 1 and beta = 0 (the initial values), the output is this normalized vector. The key point is that Layer Normalization keeps the activations from exploding or vanishing as they pass through many layers. This makes training deep networks much more stable.

Why do we need residual connections? Imagine the model has 12 layers. Without residual connections, the gradient signal during backpropagation must pass through all 12 layers to reach the first layer. With residual connections, the gradient has a "highway" — it can flow directly through the skip connections without being attenuated by the intermediate computations. This is critical for training deep networks effectively.

### Step 4: Feed-Forward Network (FFN)

After attention and layer norm, each token passes through a simple two-layer neural network independently:


$$
\text{FFN}(\mathbf{x}) = W_2 \cdot \text{GELU}(W_1 \mathbf{x} + b_1) + b_2
$$


The first layer typically projects from d_model to 4 * d_model (expanding the representation), applies the GELU activation function, and the second layer projects back to d_model.

Let us plug in some simple numbers. If d_model = 4, then W_1 has shape (4, 16) and W_2 has shape (16, 4). Our input vector of dimension 4 gets expanded to dimension 16, passes through GELU (a smooth, non-linear activation function), and gets compressed back to dimension 4. This expansion-compression pattern allows the network to learn complex transformations while keeping the output the same size as the input.

Think of the FFN as the model's "thinking" step. Attention gathers relevant information from across the sequence; the FFN then processes that gathered information through a non-linear transformation. Each token gets its own independent processing here.

Why does the FFN expand to 4x the dimension and then compress back? The expansion creates a much higher-dimensional space where the network can represent more complex relationships. The GELU activation introduces non-linearity, allowing the network to learn patterns that a simple linear transformation could never capture. Then the compression back to d_model keeps the output compatible with the rest of the network. Research has shown that this 4x expansion ratio works well in practice across many model sizes.

### Step 5: Residual Connection + Layer Normalization (again)

We add another residual connection and layer norm:


$$
\mathbf{x} = \text{LayerNorm}(\mathbf{x} + \text{FFN}(\mathbf{x}))
$$


### Step 6: Repeat for N Layers

Steps 2 through 5 form a single **Transformer block**. GPT stacks multiple blocks on top of each other. GPT-2 Small uses 12 blocks, GPT-2 Medium uses 24, and GPT-3 uses 96. The output of one block becomes the input to the next.


![The complete GPT forward pass from tokens to next-token probabilities](figures/figure_5.png)
*The complete GPT forward pass from tokens to next-token probabilities*


### Step 7: Final Linear Projection and Softmax

After all N Transformer blocks, we have a matrix of shape (sequence_length, d_model). To predict the next token, we need to convert this into a probability distribution over our entire vocabulary.

We do this with a single linear layer (no bias) that projects from d_model to vocabulary_size:


$$
\text{logits} = \mathbf{h} W_{\text{vocab}} \in \mathbb{R}^{V}
$$

where h is the output vector for a single token position and V is the vocabulary size. These raw scores are called **logits**.

We then apply softmax to convert logits into probabilities:


$$
P(\text{token}_i) = \frac{e^{\text{logit}_i}}{\sum_{j=1}^{V} e^{\text{logit}_j}}
$$

Let us plug in some simple numbers. Suppose our vocabulary has 5 tokens and the logits for the next position are [2.0, 1.0, 0.5, -1.0, 0.0]. Applying softmax:

$$\text{Denominator} = e^{2.0} + e^{1.0} + e^{0.5} + e^{-1.0} + e^{0.0} = 7.389 + 2.718 + 1.649 + 0.368 + 1.0 = 13.124$$

$$P = \left[\frac{7.389}{13.124}, \; \frac{2.718}{13.124}, \; \frac{1.649}{13.124}, \; \frac{0.368}{13.124}, \; \frac{1.0}{13.124}\right] = [0.563, \; 0.207, \; 0.126, \; 0.028, \; 0.076]$$

The model assigns 56.3% probability to token 0, 20.7% to token 1, and so on. The highest probability token is the model's best guess for the next token.

Notice something important: the softmax gives us probabilities for **every position** in the sequence simultaneously. At position 0, the model predicts what comes after the first token. At position 1, it predicts what comes after the first two tokens. And so on. This is incredibly efficient — in a single forward pass, we get predictions for all positions at once, which is exactly why training can be parallelized across the entire sequence.

This is the forward pass — from raw text all the way to a probability distribution over the next token.

---

## The Loss Function: Cross-Entropy for Next-Token Prediction

Now the question is: we have the model's prediction (a probability distribution over the vocabulary), and we know the actual next token (from the training data). How do we measure how wrong the model is?

This brings us to the **cross-entropy loss**.

The intuition is simple. If the correct next token is "mat" (token ID 3), and our model assigns 90% probability to "mat," the model did a great job — the loss should be low. If the model assigns only 1% probability to "mat," the model did a terrible job — the loss should be high.

Cross-entropy loss is defined as:


$$
\mathcal{L} = -\log P(\text{correct token})
$$


That is it. We take the probability the model assigned to the correct answer, take the log, and negate it.

Why the negative log? Let us think about it:
- If P(correct) = 1.0 (perfect prediction), then loss = -log(1.0) = 0. No penalty.
- If P(correct) = 0.5, then loss = -log(0.5) = 0.693.
- If P(correct) = 0.01 (terrible prediction), then loss = -log(0.01) = 4.605. Large penalty.


![The cross-entropy loss penalizes bad predictions exponentially](figures/figure_6.png)
*The cross-entropy loss penalizes bad predictions exponentially*


Let us plug in a full numerical example. Suppose we have a sequence of 4 tokens: ["The", "cat", "sat", "on"]. Our model processes the first 3 tokens and predicts probabilities for each next position:

- After "The": model predicts P("cat") = 0.3. Loss = -log(0.3) = 1.204
- After "The cat": model predicts P("sat") = 0.7. Loss = -log(0.7) = 0.357
- After "The cat sat": model predicts P("on") = 0.1. Loss = -log(0.1) = 2.303

The total loss is the average:

$$\mathcal{L}_{\text{total}} = \frac{1.204 + 0.357 + 2.303}{3} = \frac{3.864}{3} = 1.288$$

This tells us that, on average, the model is not doing too badly for "sat" (it got 70% right) but is struggling with "on" (only 10% confidence). The gradient from backpropagation will push the model to improve on these weak predictions. This is exactly what we want.

Notice something interesting about this loss function: it penalizes confident wrong answers much more than uncertain wrong answers. If the model assigns 0.01 probability to the correct token, the loss is 4.6 — very high. But if it assigns 0.3, the loss is only 1.2. This creates a strong incentive for the model to avoid being confidently wrong, which is a very desirable property.

For a full training batch, we average the loss across all tokens in all sequences:


$$
\mathcal{L}_{\text{batch}} = -\frac{1}{N} \sum_{i=1}^{N} \log P_{\theta}(t_i \mid t_1, t_2, \ldots, t_{i-1})
$$

where N is the total number of predicted tokens in the batch and theta represents the model parameters.

---

## Backpropagation: How the Model Learns

We have completed the forward pass and computed the loss. Now comes the magic: how does the model actually learn? How does it adjust its millions of parameters to make better predictions?

The answer is **backpropagation** — which is nothing more than the chain rule of calculus applied systematically through the network.

### The Chain Rule

Let us start with the core idea. Suppose we have a simple function y = f(g(x)). The derivative of y with respect to x is:


$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

Let us plug in some simple numbers. If g(x) = 3x and f(g) = g^2, then y = (3x)^2 = 9x^2. Using the chain rule: dy/dg = 2g = 6x, and dg/dx = 3, so dy/dx = 6x * 3 = 18x. If x = 2, the gradient is 36. We can verify: y = 9(4) = 36, and plugging x = 2.001, y = 9(2.001)^2 = 36.036, so the slope is indeed approximately 36. This is exactly what we want.

In a neural network, the "chain" is much longer. The loss depends on the softmax, which depends on the logits, which depend on the final layer norm, which depends on the FFN, which depends on the attention, which depends on the embeddings... and so on. Backpropagation simply applies the chain rule at each step, working backwards from the loss to every parameter in the network.

The beauty of backpropagation is that it is computationally efficient. Rather than computing the gradient for each parameter independently (which would require a separate forward pass for each parameter), backpropagation computes all gradients in a single backward pass. For a model with millions of parameters, this is the difference between feasible and completely impossible. The backward pass takes roughly the same amount of time as two forward passes — a remarkably cheap price for obtaining all the information the model needs to learn.

### How Gradients Flow Through the Transformer

Let us trace the gradient flow backwards through our GPT model:

**1. From Loss to Logits:** The gradient of cross-entropy loss with respect to the logits has a beautifully simple form. If the softmax output is p and the correct token is at index c, then:


$$
\frac{\partial \mathcal{L}}{\partial \text{logit}_i} = p_i - \mathbb{1}_{i=c}
$$

This means: for every token, the gradient is just the predicted probability. For the correct token, we subtract 1. Let us plug in numbers. If our softmax output was [0.563, 0.207, 0.126, 0.028, 0.076] and the correct answer is token 0, the gradient is:

$$[0.563 - 1, \; 0.207, \; 0.126, \; 0.028, \; 0.076] = [-0.437, \; 0.207, \; 0.126, \; 0.028, \; 0.076]$$

The negative gradient at position 0 means "push this logit higher" (increase the probability of the correct answer). The positive gradients at all other positions mean "push these logits lower." This makes sense because this is exactly how we want the model to adjust.

Also notice that the magnitude of the gradient for incorrect tokens is proportional to how much probability the model wasted on them. Token 1 (probability 0.207) gets a gradient of 0.207, while token 3 (probability 0.028) gets a tiny gradient of 0.028. The model gets a stronger push to reduce the probability of tokens it was most wrongly confident about. This is elegant and efficient.

**2. Through the Transformer Blocks:** The gradient flows backwards through each block. Here is where **residual connections** play a crucial role. At each residual connection:


$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{in}}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{out}}} \cdot \left(1 + \frac{\partial \text{Block}(\mathbf{x}_{\text{in}})}{\partial \mathbf{x}_{\text{in}}}\right)
$$

The key term is the "1 +" part. Even if the block's own gradient is tiny, the gradient still flows through with magnitude 1 via the skip connection. Without this, gradients in a 12-layer network would need to pass through 12 matrix multiplications, potentially shrinking to near zero (the **vanishing gradient problem**). The residual connection solves this by providing a gradient highway.

**3. Layer Normalization helps too.** By normalizing activations to have zero mean and unit variance, layer norm prevents the activations (and hence the gradients) from growing uncontrollably. This addresses the **exploding gradient problem** — the opposite of vanishing gradients.


![Residual connections create gradient highways for stable training](figures/figure_7.png)
*Residual connections create gradient highways for stable training*


### The Weight Update: Gradient Descent

Once we have computed the gradient of the loss with respect to every parameter in the network, we update each parameter using gradient descent:


$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}
$$

where eta is the learning rate.

Let us plug in some simple numbers. Suppose one weight in our model is currently theta = 0.5, the gradient is 0.2, and our learning rate is 0.01. The updated weight is:

$$\theta_{\text{new}} = 0.5 - 0.01 \times 0.2 = 0.5 - 0.002 = 0.498$$

The weight decreased slightly because the gradient was positive, meaning this weight was contributing to increasing the loss. By nudging it down, we reduce the loss. Multiply this by millions of parameters, repeat for millions of training steps, and the model gradually learns to predict the next token with increasing accuracy.

In practice, GPT models use the **AdamW optimizer** instead of vanilla gradient descent. AdamW maintains running averages of the gradients and their squares, adapting the learning rate for each parameter individually. This means that parameters which have been getting large, consistent gradients get smaller updates (to avoid overshooting), while parameters with small, noisy gradients get relatively larger updates (to make progress despite the noise). The "W" in AdamW stands for weight decay, which gently pushes all weights toward zero to prevent overfitting. But the core idea remains the same: compute gradients, update weights, repeat.

---

## Code: A Minimal GPT in PyTorch

Enough theory — let us look at some practical implementation now. Here is a complete, minimal GPT model in PyTorch. Every component we discussed is present.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)                  # (B, T, d_model)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
        x = tok_emb + pos_emb                          # (B, T, d_model)
        for block in self.blocks:
            x = block(x)                               # (B, T, d_model)
        x = self.ln_f(x)                               # (B, T, d_model)
        logits = self.head(x)                           # (B, T, vocab_size)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                # Attention + residual
        x = x + self.ffn(self.ln2(x))                 # FFN + residual
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)             # (3, B, n_heads, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = att @ v                                  # (B, n_heads, T, d_k)
        out = out.transpose(1, 2).reshape(B, T, C)    # (B, T, d_model)
        return self.proj(out)
```

Let us understand this code in detail.

**The GPT class** is the top-level model. It creates the token embedding table, positional embedding table, a stack of Transformer blocks, a final layer norm, and the output projection head. In the `forward` method, it looks up embeddings, adds positional information, passes through all blocks, applies final layer norm, and projects to vocabulary size — exactly the forward pass we described.

**The TransformerBlock class** implements one layer of the Transformer. Notice the pattern: layer norm first (this is called "Pre-LN" and is what GPT-2 uses), then attention with a residual connection, then layer norm again, then FFN with a residual connection.

**The CausalSelfAttention class** computes multi-head self-attention with causal masking. The Q, K, V matrices are computed in a single linear layer for efficiency (`self.qkv`), then reshaped and split. The causal mask is created using `torch.triu` (upper triangular matrix of ones), which sets all future positions to True, and these are filled with negative infinity before softmax. After attention, the heads are concatenated and projected back.

A few things worth noting about this implementation. First, computing Q, K, and V in a single linear layer (`self.qkv = nn.Linear(d_model, 3 * d_model)`) rather than three separate layers is a common optimization. It produces the same result but is faster because it batches the matrix multiplications. Second, the reshape and permute operations reorganize the tensor so that each attention head operates independently on its own slice of the dimensions. If d_model = 64 and n_heads = 4, each head gets d_k = 16 dimensions to work with. Third, `torch.triu` with `diagonal=1` creates a matrix of ones above the main diagonal — these are exactly the "future" positions that should be masked.

---

## Putting It All Together: A Tiny Training Loop

Now let us see the complete picture — a training loop that ties together the forward pass, loss computation, and backpropagation.

```python
# --- Hyperparameters ---
VOCAB_SIZE = 256        # Character-level tokenization (ASCII)
D_MODEL = 64            # Embedding dimension
N_HEADS = 4             # Number of attention heads
N_LAYERS = 4            # Number of transformer blocks
MAX_SEQ_LEN = 128       # Maximum sequence length
BATCH_SIZE = 32         # Batch size
LEARNING_RATE = 3e-4    # Learning rate

# --- Create model and optimizer ---
model = GPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training loop ---
for step in range(1000):
    # Get a batch of training data
    # x: input tokens (B, T), y: target tokens (B, T)
    x, y = get_batch(train_data, BATCH_SIZE, MAX_SEQ_LEN)

    # Step 1: Forward pass
    logits = model(x)                           # (B, T, vocab_size)

    # Step 2: Compute loss
    loss = F.cross_entropy(
        logits.view(-1, VOCAB_SIZE),            # Flatten to (B*T, vocab_size)
        y.view(-1)                              # Flatten to (B*T,)
    )

    # Step 3: Backward pass (compute gradients)
    optimizer.zero_grad()                       # Clear old gradients
    loss.backward()                             # Backpropagation

    # Step 4: Update weights
    optimizer.step()                            # Gradient descent

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

Let us understand this code in detail.

Each iteration of the training loop performs exactly four steps:

1. **Forward pass:** We feed input tokens through the model and get logits — raw scores for every possible next token at every position.

2. **Compute loss:** We reshape the logits and targets into flat vectors and compute cross-entropy loss. The `view(-1, VOCAB_SIZE)` reshapes (B, T, vocab_size) into (B*T, vocab_size) so that PyTorch's `cross_entropy` function can process all positions at once.

3. **Backward pass:** `loss.backward()` is where the magic happens. PyTorch automatically computes the gradient of the loss with respect to every parameter in the model using backpropagation. This single line triggers the entire chain rule computation we discussed — from the loss, through the output projection, through every Transformer block, all the way back to the embedding tables.

4. **Update weights:** `optimizer.step()` applies the AdamW update rule to nudge every parameter in the direction that reduces the loss.

The `optimizer.zero_grad()` call before `loss.backward()` is important — without it, PyTorch would accumulate gradients from previous steps, which is not what we want for standard training.


![One training step: forward, loss, backward, update](figures/figure_8.png)
*One training step: forward, loss, backward, update*


As training progresses, you would see the loss decrease from its initial value (around log(vocab_size) = log(256) = 5.5 for random predictions) down toward 1.0 or lower, indicating that the model is learning the patterns in the training data.

---

## Conclusion

We have built a GPT-style language model completely from scratch. Let us recap the journey:

1. **Text becomes vectors** through token embeddings and positional embeddings.
2. **The forward pass** sends these vectors through N Transformer blocks — each consisting of masked multi-head self-attention, layer normalization, a feed-forward network, and residual connections.
3. **The final projection** converts the output into a probability distribution over the entire vocabulary.
4. **Cross-entropy loss** measures how far the model's predictions are from the actual next tokens.
5. **Backpropagation** uses the chain rule to compute gradients for every parameter, and gradient descent updates the weights to reduce the loss.

This is the core of GPT-2, GPT-3, and GPT-4. The differences between these models are primarily about scale — more parameters, more layers, more training data, longer context windows, and more sophisticated training recipes. But the fundamental loop is identical: predict the next token, compute the loss, backpropagate, update weights. Repeat billions of times.

To put the scale in perspective:

- **GPT-2 Small**: 12 layers, 12 heads, d_model = 768, ~117 million parameters
- **GPT-2 Large**: 36 layers, 20 heads, d_model = 1280, ~774 million parameters
- **GPT-3**: 96 layers, 96 heads, d_model = 12288, ~175 billion parameters

Every single one of these uses the exact same architecture we just built — token embeddings, positional embeddings, masked multi-head self-attention, feed-forward networks, layer normalization, residual connections, and cross-entropy loss with backpropagation. The only differences are the numbers.

The fact that this simple objective — next-token prediction — gives rise to models that can write essays, answer questions, translate languages, and even reason about complex problems is one of the most remarkable discoveries in modern AI. There is no explicit programming for any of these abilities. The model is simply trained to predict the next word, and from that single objective, intelligence emerges. This is truly amazing.

That's it! You now have all the pieces to understand and build a GPT model from scratch. The complete code shown in this article is fewer than 50 lines for the model itself. Try training it on your favorite book or dataset and watch the loss decrease as the model begins to learn the structure of language. Start with character-level tokenization on a small text file — you will be surprised at how quickly even a tiny model starts producing coherent text.

Thanks!
