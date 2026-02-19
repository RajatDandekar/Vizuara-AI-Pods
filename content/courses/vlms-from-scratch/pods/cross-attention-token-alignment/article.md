# Cross-Attention & Token Alignment: How Vision-Language Models Learn to See and Speak

Understanding the mechanism that allows language models to "look" at images — from first principles to a working implementation.

---

Let us start with a simple analogy. Imagine a translator at the United Nations. A diplomat is delivering a speech in French, and the translator must produce the equivalent words in English. Here is the key observation: the translator does not just listen once and then speak from memory. At every moment, as they produce each English word, they are actively looking back at the French speaker, re-reading their notes, and deciding which part of the original speech is most relevant to the word they are about to say.

This is exactly what cross-attention does in a neural network.

In modern vision-language models, we face a version of this translation problem. Given an image of a dog catching a frisbee, the model needs to generate the caption "A dog catches a frisbee." But when it is generating the word "frisbee," it should be paying attention to the part of the image where the frisbee actually is — not the background trees, not the grass. Cross-attention is the mechanism that makes this possible.

Before we dive into cross-attention, let us make sure we are clear on the foundation it builds upon: self-attention.


![Self-attention operates within one sequence; cross-attention bridges two.](figures/figure_1.png)
*Self-attention operates within one sequence; cross-attention bridges two.*


## Self-Attention: A Quick Refresher

Let us start by looking at self-attention. If you have seen the transformer architecture before, this will be a quick recap. If not, we will build enough intuition here for you to follow the rest of the article.

The core idea behind self-attention is simple: given a sequence of tokens, each token should be able to "look at" every other token in the sequence and decide how much information to gather from each one.

To do this, we project each token into three different representations:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I actually carry?"

Each token produces a query, a key, and a value by multiplying its embedding with learned weight matrices $W_Q$, $W_K$, and $W_V$.


![In self-attention, Q, K, and V all come from the same input sequence.](figures/figure_2.png)
*In self-attention, Q, K, and V all come from the same input sequence.*


The attention mechanism then computes:


$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Let us plug in some simple numbers to see how this works.

Suppose we have 2 tokens, each with an embedding dimension of 3. Our Q and K matrices are:

$$Q = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

Step 1: Compute $QK^\top$:

$$QK^\top = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}$$

Step 2: Scale by $\sqrt{d_k} = \sqrt{3} \approx 1.73$:

$$\frac{QK^\top}{\sqrt{3}} = \begin{bmatrix} 0.577 & 0.577 \\ 0.577 & 0 \end{bmatrix}$$

Step 3: Apply softmax row-wise:

Row 1: $\text{softmax}([0.577, 0.577]) = [0.5, 0.5]$

Row 2: $\text{softmax}([0.577, 0]) = [0.644, 0.356]$

Step 4: Multiply by V:

$$\text{Output} = \begin{bmatrix} 0.5 & 0.5 \\ 0.644 & 0.356 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 2.0 & 3.0 \\ 1.71 & 2.71 \end{bmatrix}$$

This tells us that token 1 attends equally to both tokens (weights 0.5, 0.5), so its output is the average of the two value vectors. Token 2 attends more to token 1 (weight 0.644) than to itself (weight 0.356). This is exactly what we want — each token produces an output that is a weighted combination of all the values.

The critical observation for what comes next: in self-attention, Q, K, and V are all derived from the **same input sequence**.

Now the question is: what happens when we have **two different sequences** — say, one from an image and one from text?

## From Self to Cross: The Key Twist

This brings us to the main character in our story: cross-attention.

The idea is beautifully simple. Instead of computing Q, K, and V from the same sequence, we split the sources:

- **Q comes from one sequence** (the one that is "asking questions" — typically the text/decoder side)
- **K and V come from a different sequence** (the one that "holds the answers" — typically the image/encoder side)

Let us think of this with an analogy. Imagine a student sitting in a library with a textbook open in front of them. The student has questions in mind (these are the queries). The textbook has an index (these are the keys) and the actual content on each page (these are the values). The student uses their question to look up the most relevant entry in the index, and then reads the corresponding content.

In cross-attention, the text tokens are the student, and the image tokens are the textbook.


![Cross-attention lets each text token query the most relevant image patches.](figures/figure_3.png)
*Cross-attention lets each text token query the most relevant image patches.*


The mathematical formula is almost identical to self-attention:


$$\text{CrossAttention}(Q_{\text{text}}, K_{\text{image}}, V_{\text{image}}) = \text{softmax}\!\left(\frac{Q_{\text{text}} \, K_{\text{image}}^\top}{\sqrt{d_k}}\right) V_{\text{image}}$$

The only difference is where Q, K, and V come from. Let us plug in some numbers.

Suppose we have 2 text tokens and 4 image patches, each with dimension 3:

$$Q_{\text{text}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}, \quad K_{\text{image}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix}, \quad V_{\text{image}} = \begin{bmatrix} 10 \\ 20 \\ 30 \\ 40 \end{bmatrix}$$

Step 1: Compute $Q_{\text{text}} K_{\text{image}}^\top$ (shape: 2 x 4):

$$Q_{\text{text}} K_{\text{image}}^\top = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 \end{bmatrix}$$

Step 2: Scale by $\sqrt{3} \approx 1.73$:

$$\begin{bmatrix} 0.577 & 0 & 0.577 & 0.577 \\ 0 & 0.577 & 0 & 0.577 \end{bmatrix}$$

Step 3: Softmax row-wise:

Row 1: $[0.285, 0.161, 0.285, 0.285] \quad$ (text token 1 attends mostly to image patches 1, 3, and 4)

Row 2: $[0.196, 0.304, 0.196, 0.304] \quad$ (text token 2 attends mostly to image patches 2 and 4)

Step 4: Multiply by V:

$$\text{Output}_1 = 0.285 \times 10 + 0.161 \times 20 + 0.285 \times 30 + 0.285 \times 40 = 25.93$$

$$\text{Output}_2 = 0.196 \times 10 + 0.304 \times 20 + 0.196 \times 30 + 0.304 \times 40 = 26.16$$

Notice something important: our Q had shape (2 x 3) and V had shape (4 x 1), but the output has shape (2 x 1). The output always has the same number of rows as Q. This means the output is aligned with the text tokens, but it carries information gathered from the image. This is exactly what we want — the text tokens are now "enriched" with visual information.

## Token Alignment — Making Two Worlds Speak the Same Language

Now we understand the mechanics of cross-attention. But there is a practical problem we have not addressed yet.

Consider this: a Vision Transformer (ViT) produces image patch embeddings in, say, a 768-dimensional space. A language model like LLaMA works in a 4096-dimensional space. If we try to compute $Q_{\text{text}} K_{\text{image}}^\top$, the dimensions will not match — the dot product requires Q and K to share the same last dimension $d_k$.

This is the **token alignment problem**: the image tokens and text tokens live in completely different embedding spaces. Before cross-attention can work, we need to bring them into a shared space.


![Projection aligns image tokens into the language model's embedding space.](figures/figure_4.png)
*Projection aligns image tokens into the language model's embedding space.*


The simplest approach is a linear projection:


$$Z_{\text{image}} = X_{\text{image}} \, W_{\text{proj}} + b$$

Here, $X_{\text{image}}$ has shape $(n_{\text{patches}} \times d_{\text{vision}})$ and $W_{\text{proj}}$ has shape $(d_{\text{vision}} \times d_{\text{text}})$, so $Z_{\text{image}}$ has shape $(n_{\text{patches}} \times d_{\text{text}})$. Now the image tokens live in the same space as the text tokens.

Let us see this with a simple example. Suppose our ViT produces 4-dimensional embeddings and our language model uses 3-dimensional embeddings:

$$X_{\text{image}} = \begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix}, \quad W_{\text{proj}} = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \\ 0.7 & 0.8 & 0.9 \\ 1.0 & 1.1 & 1.2 \end{bmatrix}$$

$$Z_{\text{image}} = \begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix} \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \\ 0.7 & 0.8 & 0.9 \\ 1.0 & 1.1 & 1.2 \end{bmatrix} = \begin{bmatrix} 7.0 & 8.0 & 9.0 \end{bmatrix}$$

Our 4-dimensional image token has been projected into a 3-dimensional space that is compatible with the text tokens. This is what allows the cross-attention dot product to work.

In practice, there are three main architectures for this alignment step, each with different tradeoffs:


![Three approaches to align image tokens with language embedding space.](figures/figure_5.png)
*Three approaches to align image tokens with language embedding space.*


**Linear Projection (LLaVA).** A single learned matrix $W_{\text{proj}}$ maps each image patch token from the vision dimension to the language dimension. This is the simplest approach and works surprisingly well. The number of output tokens equals the number of input patches.

**MLP Projection.** Two linear layers with a GELU nonlinearity in between. This gives the projection more expressive power — the nonlinearity allows it to learn more complex mappings between the two spaces.

**Cross-Attention Resampler (Q-Former / Perceiver).** This is the most sophisticated approach. Instead of projecting each patch independently, we define a fixed set of learnable query tokens (say, 32 or 64) and let them cross-attend to the full set of image patch tokens. The output is a fixed-length sequence regardless of image resolution. This is used in BLIP-2, Flamingo, and similar architectures.

## Where Cross-Attention Lives in Modern Architectures

Now that we understand what cross-attention does and how token alignment works, let us see how these pieces fit into actual model architectures.

There are three dominant patterns in how modern vision-language models integrate visual information into language processing:


![Three ways to integrate vision into language models.](figures/figure_6.png)
*Three ways to integrate vision into language models.*


**Pattern 1: Encoder-Decoder Cross-Attention.** This is the original transformer design from "Attention Is All You Need." The image encoder processes the visual input, and every layer of the text decoder has a cross-attention sub-layer that attends to the encoder output. This is used in models like the original image captioning transformers and some recent architectures.

**Pattern 2: Gated Cross-Attention (Flamingo-style).** The language model is a standard decoder-only model, but cross-attention layers are inserted between existing self-attention layers. Crucially, these cross-attention layers have a learnable gating mechanism initialized to zero, so at the start of training, the model behaves exactly like the original language model. The visual information is gradually incorporated during training.

**Pattern 3: Prefix Fusion (LLaVA-style).** This is the simplest approach. The image tokens are projected into the language space and simply concatenated with the text tokens as a prefix. The model then uses standard self-attention over the combined sequence. There are no explicit cross-attention layers — the self-attention mechanism implicitly performs cross-modal attention because text tokens can attend to image tokens in the prefix.

Each approach has tradeoffs. Encoder-decoder gives the most explicit control over cross-modal interaction. Gated cross-attention allows using a frozen pretrained language model. Prefix fusion is the simplest to implement but requires the self-attention mechanism to learn cross-modal patterns on its own.

## Multi-Head Cross-Attention

So far, we have looked at cross-attention with a single "head" — one set of Q, K, V projections. In practice, transformers use **multi-head attention**, where we run multiple attention operations in parallel, each with its own learned projections.

Why does this matter? Because different heads can learn to attend to different things. When generating the word "catches" in "A dog catches a frisbee," one head might focus on the dog's mouth region, while another focuses on the frisbee trajectory. Multiple heads give the model the ability to gather different types of information simultaneously.


![Different heads attend to different image regions simultaneously.](figures/figure_7.png)
*Different heads attend to different image regions simultaneously.*


The mathematical formulation is:


$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_O \quad \text{where} \quad \text{head}_i = \text{Attention}(Q W_Q^i, K W_K^i, V W_V^i)
$$


Let us work through a concrete example with 2 heads.

Suppose our model dimension is $d = 4$ and we use $h = 2$ heads, so each head has $d_k = d/h = 2$.

Our text query (1 token, dimension 4): $Q = [1, 0, 1, 0]$

Our image keys (2 patches, dimension 4):
$$K = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}$$

For Head 1, we use the first 2 dimensions:
$$Q_1 = [1, 0], \quad K_1 = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$$

Scores: $Q_1 K_1^\top = [1, 0]$, scaled: $[0.707, 0]$, softmax: $[0.670, 0.330]$

For Head 2, we use the last 2 dimensions:
$$Q_2 = [1, 0], \quad K_2 = \begin{bmatrix} 0 & 0 \\ 1 & 1 \end{bmatrix}$$

Scores: $Q_2 K_2^\top = [0, 1]$, scaled: $[0, 0.707]$, softmax: $[0.330, 0.670]$

Notice how Head 1 focuses more on image patch 1 (weight 0.670) while Head 2 focuses more on image patch 2 (weight 0.670). The two heads are attending to different parts of the image. This is exactly what we want — the model can gather information from multiple regions in parallel.

## Let Us Build It — Implementation from Scratch

Enough theory, let us look at a practical implementation now.

We will implement cross-attention in PyTorch, building it from scratch so you can see exactly how every piece connects to the math we just covered.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention layer.
    Q comes from the decoder (text), K and V from the encoder (image).
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Separate projections: Q from text, K and V from image
        self.W_Q = nn.Linear(d_model, d_model)  # projects text tokens to queries
        self.W_K = nn.Linear(d_model, d_model)  # projects image tokens to keys
        self.W_V = nn.Linear(d_model, d_model)  # projects image tokens to values
        self.W_O = nn.Linear(d_model, d_model)  # output projection

    def forward(self, text_tokens, image_tokens):
        """
        Args:
            text_tokens:  (batch, n_text, d_model)  — the decoder sequence
            image_tokens: (batch, n_image, d_model) — the encoder sequence
        Returns:
            output: (batch, n_text, d_model) — text enriched with image info
        """
        batch_size = text_tokens.size(0)

        # Step 1: Project to Q, K, V
        Q = self.W_Q(text_tokens)   # (batch, n_text, d_model)
        K = self.W_K(image_tokens)  # (batch, n_image, d_model)
        V = self.W_V(image_tokens)  # (batch, n_image, d_model)

        # Step 2: Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (batch, num_heads, n_text, n_image)

        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights shape: (batch, num_heads, n_text, n_image)

        # Step 4: Multiply by values
        context = torch.matmul(attn_weights, V)
        # context shape: (batch, num_heads, n_text, d_k)

        # Step 5: Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_O(context)  # (batch, n_text, d_model)

        return output, attn_weights
```

Let us understand this code in detail.

The `__init__` method creates four linear layers: $W_Q$, $W_K$, $W_V$ for projecting the inputs, and $W_O$ for projecting the concatenated heads. Notice that $W_Q$ is applied to `text_tokens` while $W_K$ and $W_V$ are applied to `image_tokens` — this is the core difference from self-attention.

In the `forward` method, the key steps are:

1. **Project** text tokens into queries and image tokens into keys and values.
2. **Reshape** into multiple heads — we split $d_{\text{model}}$ into $h$ heads, each with dimension $d_k$.
3. **Compute scores** using the scaled dot product $QK^\top / \sqrt{d_k}$.
4. **Apply softmax** to get attention weights.
5. **Multiply by V** to gather information from the image.
6. **Concatenate** all heads and apply the output projection.

Now let us build a simple VLM block that uses this cross-attention layer:

```python
class VisionLanguageBlock(nn.Module):
    """
    A single transformer block for a vision-language model.
    Includes: self-attention (text), cross-attention (text -> image), FFN.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()

        # Self-attention for text tokens
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = CrossAttention(d_model, num_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, text_tokens, image_tokens):
        # 1. Self-attention over text (with residual)
        text_normed = self.norm1(text_tokens)
        text_tokens = text_tokens + self.self_attn(
            text_normed, text_normed, text_normed
        )[0]

        # 2. Cross-attention: text queries image (with residual)
        text_normed = self.norm2(text_tokens)
        cross_out, attn_weights = self.cross_attn(text_normed, image_tokens)
        text_tokens = text_tokens + cross_out

        # 3. Feed-forward (with residual)
        text_tokens = text_tokens + self.ffn(self.norm3(text_tokens))

        return text_tokens, attn_weights
```

This is the standard pattern in encoder-decoder vision-language models. Each decoder block has three sub-layers: self-attention (text talks to text), cross-attention (text queries image), and a feed-forward network. Each sub-layer uses a residual connection and layer normalization.

## What the Model Actually Looks At — Visualizing Attention

One of the most compelling aspects of cross-attention is that we can directly visualize what the model is paying attention to. The attention weights $\alpha_{ij}$ tell us exactly how much text token $i$ attends to image patch $j$.


![Cross-attention maps reveal which image regions each word attends to.](figures/figure_8.png)
*Cross-attention maps reveal which image regions each word attends to.*


Here is how you can extract and visualize these attention weights:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_cross_attention(image, attn_weights, text_tokens, patch_grid=(14, 14)):
    """
    Visualize cross-attention weights overlaid on the original image.

    Args:
        image: original image as numpy array (H, W, 3)
        attn_weights: (num_heads, n_text, n_patches) from cross-attention
        text_tokens: list of token strings
        patch_grid: (rows, cols) of image patches
    """
    # Average across attention heads
    avg_attn = attn_weights.mean(dim=0)  # (n_text, n_patches)

    fig, axes = plt.subplots(1, len(text_tokens), figsize=(4 * len(text_tokens), 4))

    for idx, token in enumerate(text_tokens):
        # Reshape attention to 2D grid matching image patches
        attn_map = avg_attn[idx].reshape(patch_grid).detach().numpy()

        # Resize attention map to match image dimensions
        attn_resized = np.array(
            plt.cm.jet(attn_map / attn_map.max())[:, :, :3]  # normalize and colormap
        )
        attn_resized = np.kron(
            attn_resized, np.ones((image.shape[0] // patch_grid[0],
                                    image.shape[1] // patch_grid[1], 1))
        )

        axes[idx].imshow(image, alpha=0.5)
        axes[idx].imshow(attn_resized, alpha=0.5)
        axes[idx].set_title(f'"{token}"', fontsize=14)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
```

This visualization reveals something fascinating: the model learns to ground language in vision without any explicit supervision telling it which image regions correspond to which words. The cross-attention mechanism discovers these correspondences purely from the training objective.

Common patterns you will observe:

- **Nouns** (dog, car, person) attend to the corresponding object regions
- **Verbs** (catches, runs, sits) attend to action-relevant regions — often the area between the agent and the object
- **Adjectives** (red, large, fluffy) attend to the same region as their associated noun but with slightly different emphasis
- **Spatial words** (above, behind, next to) attend to the boundary regions between objects

## Practical Considerations and Common Pitfalls

Before you go and implement cross-attention in your own projects, let us discuss a few practical points that are often overlooked.

**Computational Cost.** The attention score matrix has shape $(n_{\text{text}} \times n_{\text{image}})$. If your image has 196 patches (14x14 from a ViT) and your text has 512 tokens, that is a 512 x 196 matrix per head per layer. This is generally cheaper than self-attention over the text alone (512 x 512), but it adds up across layers.

**No Causal Masking Needed.** In the decoder's self-attention, we use a causal mask to prevent tokens from attending to future tokens. In cross-attention, no such mask is needed — every text token is allowed to attend to every image patch. The image is fully observed; there is no "future" to hide.

**Positional Encodings.** Image tokens need 2D positional information (row and column in the patch grid), while text tokens use 1D positional encodings (sequential position). These are typically added to the tokens before they enter the cross-attention layer, and the attention mechanism learns to use them implicitly.

**Attention Collapse.** Sometimes, all text tokens end up attending to the same image patch — usually a "background" patch with average features. This can happen when the temperature (the $\sqrt{d_k}$ scaling) is too aggressive or when training data lacks diversity. Solutions include using a larger $d_k$, adding dropout to attention weights, or using entropy regularization.

**Memory-Efficient Attention.** Modern implementations use FlashAttention or similar memory-efficient algorithms that fuse the softmax and matmul operations to avoid materializing the full attention matrix. This works for cross-attention just as well as for self-attention.

## Conclusion

Let us recap what we have covered.

**Self-attention** allows tokens within a single sequence to communicate with each other — Q, K, and V all come from the same input.

**Cross-attention** is the mechanism that connects two different sequences — Q comes from one sequence (typically text), while K and V come from another (typically image). The output has the same length as the query sequence, enriched with information from the key-value sequence.

**Token alignment** solves the dimensionality mismatch between vision and language embedding spaces. Whether through a simple linear projection, an MLP, or a sophisticated cross-attention resampler, the goal is the same: bring image tokens into a space where they can interact with text tokens.

Together, cross-attention and token alignment form the bridge that allows vision-language models to "see and speak" — to ground language in visual perception and generate text that is faithful to the visual input.

That's it!

Here is a summary of key formulas:

**Scaled Dot-Product Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Cross-Attention (Q from text, K/V from image):**
$$\text{CrossAttention}(Q_{\text{text}}, K_{\text{image}}, V_{\text{image}}) = \text{softmax}\!\left(\frac{Q_{\text{text}} \, K_{\text{image}}^\top}{\sqrt{d_k}}\right) V_{\text{image}}$$

**Multi-Head Attention:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_O$$

---

**References:**

1. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
2. Alayrac, J., et al. "Flamingo: a Visual Language Model for Few-Shot Learning." NeurIPS 2022.
3. Liu, H., et al. "Visual Instruction Tuning." NeurIPS 2023 (LLaVA).
4. Li, J., et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML 2023.
5. Jaegle, A., et al. "Perceiver: General Perception with Iterative Attention." ICML 2021.
