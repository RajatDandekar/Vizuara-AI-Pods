# Vision Transformers from Scratch: How Treating Images as Sentences Changed Computer Vision

*We break down the Vision Transformer (ViT) paper step by step — from image patches to self-attention — with intuition, math, and a full PyTorch implementation.*

---

## The Big Idea: Reading Images Like Sentences

Let us start with a simple experiment. Take the sentence: **"The cat sat on the mat."** When you read it, you process it one word at a time — *The*, then *cat*, then *sat*, and so on. Each word is a small, discrete token, and you build up meaning by understanding how these tokens relate to each other.

Now look at a photograph of a cat sitting on a mat. How do you process it? You might think you take in the whole image at once, but that is not quite what happens. Your eyes actually fixate on different *regions* — the cat's face, the mat's texture, the background — and your brain pieces together the full scene from these local glimpses.

Here is the key question: **What if we could teach a computer to look at images the same way it reads sentences — one piece at a time?**

This is exactly the idea behind the **Vision Transformer (ViT)**. Instead of processing an image with convolutional filters (the traditional approach), ViT chops the image into small patches and treats each patch like a word in a sentence. Then it feeds these "visual words" into a Transformer — the same architecture that powers GPT and BERT — and lets self-attention figure out how the patches relate to each other.

The result? A model that can "read" images with remarkable accuracy. Let us understand how this works, step by step.


![Transformers process sentences as word tokens and images as patch tokens — the same architecture handles both.](figures/figure_1.png)
*Transformers process sentences as word tokens and images as patch tokens — the same architecture handles both.*


---

## A Quick Refresher: Why CNNs Were King

Before Vision Transformers came along, **Convolutional Neural Networks (CNNs)** dominated computer vision for nearly a decade. Let us quickly understand why — and what their limitations are.

A CNN works by sliding small filters (say, 3×3 pixels) across an image. Each filter detects a specific local pattern — an edge, a corner, a texture. As you stack more convolutional layers, the network learns to combine these local patterns into larger, more abstract features: edges become textures, textures become object parts, and object parts become full objects.

The key strength of CNNs is **locality** — each filter only looks at a small neighborhood of pixels. This is a powerful inductive bias because nearby pixels tend to be related (a pixel on a dog's ear is more related to its neighboring pixels than to a pixel in the sky).

But this strength is also a limitation. A single convolutional layer with a 3×3 filter can only "see" a 3×3 region. To understand how a dog's ear relates to its tail (which might be hundreds of pixels away), you need to stack many layers so the **receptive field** gradually grows to cover the entire image.

Think of it this way: a CNN reads an image with a magnifying glass. It sees fine local detail beautifully, but needs to zoom out layer by layer to understand the big picture.

Now the question is: **what if we skip the magnifying glass entirely and let the model see all parts of the image at once, from the very first layer?**

This is precisely what the Vision Transformer does. And this brings us to the core idea.


![CNNs build global context gradually through stacked layers; ViT has global context from the very first layer.](figures/figure_2.png)
*CNNs build global context gradually through stacked layers; ViT has global context from the very first layer.*


---

## The Core Idea: Images as Sequences of Patches

The central insight of the ViT paper (Dosovitskiy et al., 2020) is beautifully simple: **reshape an image into a sequence of patches, and feed them to a standard Transformer encoder.**

Let us walk through this step by step.

**Step 1: Divide the image into patches.**

Take an input image of size 224 × 224 pixels with 3 color channels (RGB). Now divide it into a grid of non-overlapping square patches, each of size 16 × 16 pixels.

How many patches do we get?

$$N = \left(\frac{224}{16}\right)^2 = 14 \times 14 = 196 \text{ patches}$$

So our 224 × 224 image becomes a grid of 196 patches.

**Step 2: Flatten each patch into a vector.**

Each patch is a 16 × 16 × 3 block of pixels. We flatten it into a single vector:

$$\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}$$


Let us plug in the numbers. Each patch vector has $16 \times 16 \times 3 = 768$ dimensions. So our image has been transformed from a 224 × 224 × 3 tensor into a sequence of **196 vectors, each of dimension 768**.

Does this remind you of something? In NLP, a sentence is a sequence of word tokens, each represented as a vector (an embedding). Here, an image is a sequence of patch tokens, each represented as a flattened pixel vector. **The image has become a sentence.**

Let us plug in some simple numbers to make sure this is clear. Suppose we had a tiny 8 × 8 grayscale image (1 channel) with a patch size of 4. The number of patches would be $(8/4)^2 = 4$ patches, and each patch vector would have $4 \times 4 \times 1 = 16$ dimensions. So our tiny image becomes a sequence of 4 tokens, each of dimension 16. This is exactly the same structure as a 4-word sentence with 16-dimensional word embeddings.


![A 224×224 image becomes a sequence of 196 patch vectors, each with 768 dimensions.](figures/figure_3.png)
*A 224×224 image becomes a sequence of 196 patch vectors, each with 768 dimensions.*


---

## Patch Embedding and Position Embedding

We now have 196 raw patch vectors, but raw pixel values are not great representations for a Transformer to work with. We need to project them into a richer, learned embedding space — just like word embeddings in NLP.

### Linear Projection (Patch Embedding)

We multiply each flattened patch vector by a learnable weight matrix $\mathbf{E}$ to project it into the model's hidden dimension $D$:

$$\mathbf{z}_0^i = \mathbf{x}_p^i \, \mathbf{E} + \mathbf{e}_{pos}^i$$

For ViT-Base, $D = 768$. Since our flattened patches are also 768-dimensional (when $P = 16$ and $C = 3$), the projection matrix $\mathbf{E}$ is 768 × 768. Each patch gets transformed from a raw pixel vector into a learned representation — much like how a word embedding layer transforms a one-hot word vector into a dense embedding.

Let us plug in some simple numbers. Suppose $D = 4$ for a toy model, and we have a flattened patch vector $\mathbf{x}_p = [0.1, 0.5, 0.3, 0.2, 0.8, 0.4]$ (6-dimensional, meaning $P^2 \cdot C = 6$). Our projection matrix $\mathbf{E}$ would be $6 \times 4$. After multiplication, we get a 4-dimensional embedding vector. This is exactly what we want — a compact, learnable representation of each patch.

### The [CLS] Token

Here is a clever trick borrowed from BERT. We prepend a special learnable token called the **[CLS] token** to the beginning of our patch sequence. This token does not correspond to any image patch — instead, it serves as a "summary" token that will accumulate information from all patches through self-attention.

After adding the [CLS] token, our sequence grows from 196 to **197 tokens**.

### Position Embeddings

There is one more problem. Transformers process all tokens in parallel — they have no built-in notion of order or position. If we shuffled all 196 patches randomly, the Transformer would produce the same output. But clearly, the position of a patch matters — a patch from the top-left corner of an image carries different spatial information than one from the bottom-right.

To fix this, we add **learnable position embeddings** to each token. Each of the 197 positions gets its own learnable vector $\mathbf{e}_{pos}^i$, which is added to the patch embedding. This way, the model learns that token 0 is the [CLS] token, token 1 is the top-left patch, token 2 is the next patch to its right, and so on.

The beautiful part? These position embeddings are learned from data. After training, if you visualize them, you find that nearby patches have similar position embeddings — the model has learned a 2D spatial structure entirely on its own, without us encoding it explicitly.


![Each patch is linearly projected, combined with a position embedding, and prepended with a [CLS] token.](figures/figure_4.png)
*Each patch is linearly projected, combined with a position embedding, and prepended with a [CLS] token.*


---

## The Transformer Encoder: Self-Attention on Patches

Now we feed our 197 tokens into a standard Transformer encoder. This is where the magic happens — self-attention allows every patch to communicate with every other patch, building a rich, global understanding of the image.

### Inside a Transformer Block

Each Transformer block has two main components:

1. **Multi-Head Self-Attention (MHSA)** — lets patches attend to each other
2. **MLP (Feed-Forward Network)** — processes each token independently

Both are wrapped with **Layer Normalization** and **residual connections**. The full block looks like this:

$$\mathbf{z}'_l = \text{MHSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$

$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$

Let us break down each component.

### Self-Attention: How Patches Talk to Each Other

Self-attention is the core mechanism. For each token, it computes three vectors:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I carry?"

These are computed by multiplying the input by three learned weight matrices:

$$Q = \mathbf{z} W^Q, \quad K = \mathbf{z} W^K, \quad V = \mathbf{z} W^V$$

Then, the attention output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Let us see what this means with a concrete example. Imagine we have just 3 patches (to keep it simple), each with $d_k = 2$ dimensions:

- Patch 1 (dog's face): $Q_1 = [1, 0]$, $K_1 = [1, 0]$, $V_1 = [0.9, 0.1]$
- Patch 2 (dog's body): $Q_2 = [0.8, 0.2]$, $K_2 = [0.9, 0.1]$, $V_2 = [0.7, 0.3]$
- Patch 3 (sky background): $Q_3 = [0, 1]$, $K_3 = [0.1, 0.9]$, $V_3 = [0.2, 0.8]$

First, compute the attention scores: $QK^T / \sqrt{d_k}$. For Patch 1:

- Score with Patch 1: $(1 \times 1 + 0 \times 0) / \sqrt{2} = 0.71$
- Score with Patch 2: $(1 \times 0.9 + 0 \times 0.1) / \sqrt{2} = 0.64$
- Score with Patch 3: $(1 \times 0.1 + 0 \times 0.9) / \sqrt{2} = 0.07$

After softmax: $[0.41, 0.38, 0.21]$. Patch 1 (dog's face) attends most strongly to itself and the dog's body, and barely attends to the sky background. This is exactly what we want — the model learns that related patches should pay more attention to each other.

The $\sqrt{d_k}$ in the denominator is a scaling factor that prevents the dot products from becoming too large (which would push softmax into regions with tiny gradients).

### Multi-Head Attention

Instead of computing a single attention, ViT uses **multiple heads** — each head learns to focus on different types of relationships:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O$$


For ViT-Base: $D = 768$ and $h = 12$ heads, so each head works with $d_k = 768 / 12 = 64$ dimensions. One head might learn to attend to spatially nearby patches, another might attend to patches with similar colors, and yet another might focus on patches containing similar textures.

### The MLP Block

After self-attention, each token passes through a two-layer MLP with a GELU activation:

$$\text{MLP}(\mathbf{z}) = \text{GELU}(\mathbf{z} W_1 + b_1) W_2 + b_2$$


The inner dimension is typically $4 \times D = 4 \times 768 = 3072$. This expansion-then-compression pattern gives each token a chance to process the information gathered from self-attention through a richer, higher-dimensional space.

ViT-Base stacks **12 of these blocks** in sequence. By the final block, each token has been refined through 12 rounds of global self-attention and local processing — building an incredibly rich representation of the image.


![One Transformer encoder block: self-attention followed by MLP, both with residual connections. ViT-Base stacks 12 of these.](figures/figure_5.png)
*One Transformer encoder block: self-attention followed by MLP, both with residual connections. ViT-Base stacks 12 of these.*


Now let us visualize what self-attention actually looks like when applied to image patches.


![A patch containing the dog's face attends strongly to the dog's body and nearby ground, but weakly to the distant sky.](figures/figure_6.png)
*A patch containing the dog's face attends strongly to the dog's body and nearby ground, but weakly to the distant sky.*


---

## The Classification Head

After passing through all 12 Transformer blocks, we have 197 refined token representations. But we only need one output — a class prediction.

This is where the [CLS] token pays off. Remember, this special token has been attending to all 196 patches across all 12 layers, accumulating a global summary of the entire image. We simply extract this token and pass it through a small classification head:

$$\hat{y} = \text{MLP}(\text{LN}(\mathbf{z}_L^0))$$


Here, $\mathbf{z}_L^0$ is the [CLS] token's representation at the final layer $L$. The MLP head is just a single linear layer that maps from dimension $D$ to the number of classes (e.g., 1000 for ImageNet).

Training uses standard **cross-entropy loss**, just like any image classifier. Nothing fancy here — the innovation is entirely in the architecture, not the training objective.

Now we have all the pieces of the puzzle. Let us put them together and see the full architecture.


![The complete Vision Transformer: image patches enter the Transformer encoder, and the [CLS] token's output predicts the class.](figures/figure_7.png)
*The complete Vision Transformer: image patches enter the Transformer encoder, and the [CLS] token's output predicts the class.*


---

## ViT Variants and the Power of Scale

The original ViT paper introduced three model sizes:

| Model | Layers | Hidden Size $D$ | MLP Size | Heads | Parameters |
|-------|--------|-----------------|----------|-------|------------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

Now here is the most important finding from the paper. When trained on ImageNet alone (1.3M images), ViT **underperforms** well-tuned CNNs like ResNet. But when pre-trained on larger datasets — ImageNet-21k (14M images) or the massive JFT-300M (300M images) — ViT **surpasses** all CNNs.

This makes sense because CNNs have built-in inductive biases — locality (nearby pixels are related) and translation equivariance (a cat is a cat regardless of where it appears). These biases act as helpful shortcuts when data is limited. Transformers have no such biases — they must learn spatial relationships entirely from data. With enough data, this flexibility becomes a strength rather than a weakness.


![With limited data, CNNs win thanks to inductive biases; with large-scale data, ViT surges ahead.](figures/figure_8.png)
*With limited data, CNNs win thanks to inductive biases; with large-scale data, ViT surges ahead.*



![The three ViT variants scale from 86M to 632M parameters by increasing depth and width.](figures/figure_9.png)
*The three ViT variants scale from 86M to 632M parameters by increasing depth and width.*


A later paper called **DeiT (Data-efficient Image Transformers)** showed that with the right training recipe — strong data augmentation, regularization, and knowledge distillation from a CNN teacher — ViT can achieve competitive results even on ImageNet-1k alone. This opened the floodgates for Vision Transformers in practical settings where massive pre-training datasets are not available.

---

## Full PyTorch Implementation

Enough theory — let us build a Vision Transformer from scratch in PyTorch. We will implement ViT-Base step by step.

### Patch Embedding

First, let us create the patch embedding layer. This takes an image and converts it into a sequence of embedded patch tokens with position information:

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 196

        # Linear projection of flattened patches (implemented as a Conv2d for efficiency)
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Learnable [CLS] token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

    def forward(self, x):
        B = x.shape[0]  # batch size

        # Project patches: (B, 3, 224, 224) -> (B, 768, 14, 14) -> (B, 768, 196) -> (B, 196, 768)
        x = self.projection(x).flatten(2).transpose(1, 2)

        # Prepend [CLS] token: (B, 196, 768) -> (B, 197, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.position_embeddings
        return x
```

Let us understand this code. The key trick is using `nn.Conv2d` with `kernel_size=patch_size` and `stride=patch_size` — this extracts non-overlapping patches and projects them in a single operation, which is mathematically equivalent to flattening each patch and multiplying by the weight matrix $\mathbf{E}$, but much more efficient. The `[CLS]` token is a learnable parameter that gets prepended to every image's patch sequence.

### Transformer Block

Next, let us implement a single Transformer encoder block:

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Multi-Head Self-Attention with residual connection
        ln_x = self.ln1(x)
        attn_out, _ = self.attn(ln_x, ln_x, ln_x)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x
```

Notice how clean this is. Layer normalization is applied *before* each sub-layer (this is called "Pre-Norm" and is what ViT uses), and residual connections add the input back after each sub-layer. The MLP expands the dimension by a factor of 4 (768 → 3072) and then compresses it back.

### The Full Vision Transformer

Now let us put everything together:

```python
class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_channels=3,
        num_classes=1000, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4.0, dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Stack Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(depth)]
        )

        # Classification head
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)       # (B, 197, 768)
        x = self.dropout(x)
        x = self.blocks(x)            # (B, 197, 768)
        cls_token = x[:, 0]           # (B, 768) — extract [CLS] token
        x = self.ln(cls_token)
        x = self.head(x)              # (B, num_classes)
        return x
```

This is the entire Vision Transformer. The forward pass is remarkably straightforward: embed the patches, pass through 12 Transformer blocks, extract the [CLS] token, and classify.

### Let Us Test It

```python
# Create a ViT-Base model
model = VisionTransformer(
    img_size=224, patch_size=16, in_channels=3,
    num_classes=1000, embed_dim=768, depth=12, num_heads=12
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.1f}M")

# Forward pass with a random image
dummy_image = torch.randn(1, 3, 224, 224)
output = model(dummy_image)
print(f"Input shape:  {dummy_image.shape}")
print(f"Output shape: {output.shape}")
```

Running this gives:

```
Total parameters: 86.6M
Input shape:  torch.Size([1, 3, 224, 224])
Output shape: torch.Size([1, 1000])
```

86.6 million parameters — matching the ViT-Base specification. A single 224 × 224 image goes in, and a 1000-dimensional logits vector comes out (one score per ImageNet class). This is exactly what we want.

---

## Wrapping Up

Let us recap the three key ideas behind the Vision Transformer:

1. **Images can be treated as sequences of patches.** By dividing an image into a grid of 16 × 16 patches and flattening them, we transform a 2D visual input into a 1D sequence — just like a sentence.

2. **A standard Transformer encoder works on patch tokens.** No special vision-specific modules needed. Self-attention gives every patch global context from layer one, and the [CLS] token aggregates the final representation.

3. **Scale matters.** ViT needs large-scale pre-training data to outperform CNNs. Without inductive biases like locality, the model relies on data to learn spatial relationships — and with enough data, it learns them better than any hand-coded bias.

The impact of ViT has been enormous. It opened the door to an entire family of vision transformers — DeiT, Swin Transformer, BEiT, MAE, DINO — that now dominate computer vision benchmarks. The idea that the same Transformer architecture can handle both language and vision was a unifying moment for deep learning.

Here is the link to the original paper: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020).

That's it!

---

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020)
- Vaswani et al., "Attention Is All You Need" (2017)
- Touvron et al., "Training Data-efficient Image Transformers & Distillation through Attention (DeiT)" (2021)
- He et al., "Deep Residual Learning for Image Recognition (ResNet)" (2016)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
