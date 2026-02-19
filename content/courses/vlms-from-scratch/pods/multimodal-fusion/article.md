# Multimodal Fusion Architectures: How AI Learns to See, Read, and Listen — All at Once

*From early fusion to cross-attention — building the bridges that connect vision and language inside modern AI systems.*

---

Let us start with a simple scenario. Imagine you are a doctor trying to diagnose a patient. You look at the chest X-ray on your screen — the lungs look slightly hazy. You read through the patient's medical history — three months of persistent cough, recent weight loss. The patient tells you they feel short of breath, especially at night.

No single piece of information is enough. The X-ray alone could mean a dozen things. The text report alone is ambiguous. The patient's verbal description alone is vague. But when you combine all three — the visual, the textual, and the auditory — the picture snaps into focus.

This is **multimodal fusion**, and it is precisely what we want our AI systems to do.


![Multimodal fusion mirrors how humans combine senses to make decisions.](figures/figure_1.png)
*Multimodal fusion mirrors how humans combine senses to make decisions.*


The question we will answer in this article is: **How do neural networks combine information from different modalities?** We will start with the simplest approaches and build our way up to the architectures that power modern vision-language models like LLaVA and Flamingo.

---

## What Does "Multimodal" Mean?

Before we build anything, let us be precise about what "modality" means. A **modality** is a distinct channel of information. Images are one modality. Text is another. Audio, video, depth maps, LiDAR point clouds — each of these is a separate modality.

Now, here is the crucial challenge. Each modality lives in a completely different representational space:

- An **image** is a grid of pixels — a 3D tensor of shape (height, width, channels).
- **Text** is a sequence of discrete tokens — integers drawn from a vocabulary.
- **Audio** is a 1D waveform — a continuous signal sampled at regular intervals.


![Each modality speaks a different language — fusion bridges them.](figures/figure_2.png)
*Each modality speaks a different language — fusion bridges them.*


These are fundamentally different data structures. You cannot simply add an image to a sentence. So the first job of a multimodal system is to find a way to bring these different representations into a common space where they can meaningfully interact.

We denote a multimodal input as:


$$X = \{x^{(v)}, x^{(t)}, x^{(a)}\}$$

where $x^{(v)}$ is the visual input, $x^{(t)}$ is the textual input, and $x^{(a)}$ is the audio input.

Let us plug in some concrete dimensions. Suppose our image is a 224x224 RGB image, so $x^{(v)} \in \mathbb{R}^{224 \times 224 \times 3}$. Our text is a sequence of 20 tokens, so $x^{(t)} \in \mathbb{Z}^{20}$. Our audio is a 1-second clip sampled at 16 kHz, so $x^{(a)} \in \mathbb{R}^{16000}$. Three completely different shapes. Three completely different data types. The question is: how do we get them to talk to each other?

---

## The Three Fusion Strategies

There are three canonical approaches to combining information from different modalities. Understanding these three strategies is the foundation for understanding every modern multimodal architecture.


![The three canonical strategies for combining modalities.](figures/figure_3.png)
*The three canonical strategies for combining modalities.*


Let us go through each one.

### Early Fusion: Mixing Everything from the Start

The simplest approach is **early fusion**. Take the raw (or lightly processed) inputs from each modality, concatenate them, and feed the combined representation into a single shared network.

Think of it like cooking — you throw all the ingredients into one pot from the very beginning and let them cook together.

The mathematical formulation is straightforward:


$$z_{\text{early}} = f_{\text{shared}}\left(\left[x^{(v)} ; x^{(t)}\right]\right)$$

Here, the semicolon denotes concatenation. We take the visual input and the textual input, concatenate them along some dimension, and pass the result through a shared function $f_{\text{shared}}$ (typically a neural network).

Let us work through a simple numerical example. Suppose after some initial processing, our image is represented as a vector $x^{(v)} = [0.3, 0.7, 0.1]$ and our text is represented as $x^{(t)} = [0.5, 0.2]$. Early fusion simply concatenates these:

$$[x^{(v)} ; x^{(t)}] = [0.3, 0.7, 0.1, 0.5, 0.2]$$

This 5-dimensional vector is then passed through the shared network. This is exactly what we want — the shared network can now learn interactions between all five features, regardless of which modality they came from.

**The advantage** of early fusion is that the model can learn cross-modal interactions from the very first layer. If there is a subtle relationship between a pixel pattern and a word, the network has the full depth of processing to discover it.

**The disadvantage** is practical: images and text have very different sizes and structures. A 224x224 image has 150,528 pixel values. A typical text input might have a few hundred token embeddings. Concatenating these raw inputs creates a lopsided, awkward input tensor.

Here is a simple early fusion implementation in PyTorch:

```python
import torch
import torch.nn as nn

class EarlyFusionModel(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim, num_classes):
        super().__init__()
        # Shared network processes the concatenated input
        self.shared = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_features, text_features):
        # Concatenate along the feature dimension
        combined = torch.cat([image_features, text_features], dim=-1)
        return self.shared(combined)

# Example: 512-dim image features + 256-dim text features
model = EarlyFusionModel(image_dim=512, text_dim=256, hidden_dim=384, num_classes=10)
img = torch.randn(1, 512)   # Batch of 1 image
txt = torch.randn(1, 256)   # Batch of 1 text
output = model(img, txt)
print(f"Output shape: {output.shape}")  # (1, 10)
```

### Late Fusion: Independent Experts, Final Committee

The opposite approach is **late fusion**. Here, each modality is processed independently by its own specialized encoder. Only at the very end are the outputs combined — typically through concatenation, addition, or a small fusion head.

Think of it as two experts writing independent reports on the same case. Neither expert sees the other's work. At the end, a manager reads both reports and makes a final decision.


$$z_{\text{late}} = g\left(\left[f_v(x^{(v)}) ; f_t(x^{(t)})\right]\right)$$

Here, $f_v$ is a vision encoder (like a CNN or ViT), $f_t$ is a text encoder (like a Transformer), and $g$ is a small fusion head that combines the outputs.

Let us plug in numbers. Suppose our vision encoder produces $f_v(x^{(v)}) = [0.8, -0.3]$ and our text encoder produces $f_t(x^{(t)}) = [0.1, 0.9]$. We concatenate to get $[0.8, -0.3, 0.1, 0.9]$, then pass through the fusion head $g$.

If $g$ is simply a linear layer with weights $W = [0.5, 0.2, 0.3, 0.4]$ and bias $b = 0.1$:

$$z_{\text{late}} = 0.5(0.8) + 0.2(-0.3) + 0.3(0.1) + 0.4(0.9) + 0.1 = 0.4 - 0.06 + 0.03 + 0.36 + 0.1 = 0.83$$

This tells us that the fused representation is 0.83 — a single number that combines information from both modalities. In practice, the fusion head would output a vector, not a scalar, but the principle is the same.


![Late fusion lets each modality be processed by a specialist encoder.](figures/figure_4.png)
*Late fusion lets each modality be processed by a specialist encoder.*


**The advantage** of late fusion is flexibility. You can use the best available encoder for each modality — a pretrained ViT for images, a pretrained LLM for text — and combine their outputs without modifying either encoder. This also makes it easy to swap encoders.

**The disadvantage** is that cross-modal interactions are limited. The vision encoder never sees the text, and the text encoder never sees the image. They can only interact at the very end, in the fusion head. If the task requires fine-grained cross-modal reasoning (e.g., "Is the red car in front of or behind the blue truck?"), late fusion may struggle.

### Mid-level Fusion: Cross-Attention — The Best of Both Worlds

Now we come to the approach that dominates modern multimodal architectures: **mid-level fusion through cross-attention**.

The idea is elegant. Instead of waiting until the end to combine modalities, we let them interact at intermediate layers of the network. Specifically, we use the **cross-attention mechanism** — the same mechanism that powers the decoder in the original Transformer.

Think of two translators sitting side by side, working on different documents. As they translate, they constantly glance at each other's work to maintain consistency. Each translator has their own primary task, but they can selectively attend to the other's output whenever they need context.

The cross-attention formula is:


$$\text{CrossAttn}(Q_t, K_v, V_v) = \text{softmax}\left(\frac{Q_t K_v^\top}{\sqrt{d_k}}\right) V_v$$

Here, $Q_t$ are the **queries** from the text modality, and $K_v$, $V_v$ are the **keys** and **values** from the visual modality. The text tokens ask questions (queries), and the visual tokens provide answers (keys and values).

Let us work through a concrete numerical example. Suppose we have 2 text tokens and 3 visual tokens, with $d_k = 2$:

$$Q_t = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad K_v = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad V_v = \begin{bmatrix} 0.5 & 0.3 \\ 0.1 & 0.8 \\ 0.9 & 0.2 \end{bmatrix}$$

Step 1: Compute $Q_t K_v^\top$:

$$Q_t K_v^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix}$$

Step 2: Divide by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:

$$\frac{Q_t K_v^\top}{\sqrt{2}} = \begin{bmatrix} 0.707 & 0 & 0.707 \\ 0.707 & 0.707 & 0 \end{bmatrix}$$

Step 3: Apply softmax row-wise. For the first row: $\text{softmax}([0.707, 0, 0.707]) \approx [0.39, 0.19, 0.39]$ (the two equal values get equal weight, and the zero value gets less weight).

Step 4: Multiply by $V_v$ to get the output — each text token is now a weighted combination of the visual tokens. The first text token attends roughly equally to visual tokens 1 and 3 (the ones with high scores), picking up a blend of their values.

This is exactly what we want. The text token "selectively attends" to the most relevant visual tokens. If the text says "red car," the cross-attention mechanism will learn to focus on the visual tokens that correspond to red car pixels.


![Cross-attention lets text tokens selectively attend to visual regions.](figures/figure_5.png)
*Cross-attention lets text tokens selectively attend to visual regions.*


Here is a clean implementation of a cross-attention layer:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Queries come from one modality, Keys/Values from another
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x_query, x_kv):
        B, N, D = x_query.shape
        _, M, _ = x_kv.shape

        q = self.q_proj(x_query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_kv).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_kv).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: text queries attend to visual keys
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)

# Example: 10 text tokens query 196 visual patches, both 512-dim
cross_attn = CrossAttention(dim=512, num_heads=8)
text_tokens = torch.randn(1, 10, 512)
visual_patches = torch.randn(1, 196, 512)
output = cross_attn(text_tokens, visual_patches)
print(f"Output shape: {output.shape}")  # (1, 10, 512)
```

---

## Encoders — Translating Each Modality into a Common Language

Before we can fuse modalities, each modality needs its own **encoder** to convert raw data into a sequence of embedding vectors. Think of encoders as translators: they take input in one language (pixels, tokens, waveforms) and produce output in a common language (embedding vectors of dimension $d$).

For **vision**, the dominant encoder today is the **Vision Transformer (ViT)**. It splits the image into non-overlapping patches (typically 16x16 pixels), treats each patch as a "visual token," and processes the full sequence with a Transformer encoder. The output is a sequence of patch embeddings.

For **language**, we use a standard **Transformer** — either an encoder (BERT-style) for understanding tasks, or a decoder (GPT-style) for generation tasks. The output is a sequence of token embeddings.

But here is the problem: the vision encoder might produce 768-dimensional embeddings, while the language model expects 4096-dimensional inputs. We need a **projection layer** to map each modality's output into a shared space:


$$
h_v = W_v \cdot \text{ViT}(x^{(v)}) + b_v, \quad h_t = W_t \cdot \text{LLM}(x^{(t)}) + b_t
$$


Here, $W_v$ is a weight matrix that projects the ViT output from 768 dimensions to the shared dimension, and $W_t$ does the same for the LLM.

Let us work a simple example. Suppose the ViT produces a 3-dimensional embedding $\text{ViT}(x^{(v)}) = [0.5, -0.2, 0.8]$ and we want to project it to 2 dimensions using:

$$W_v = \begin{bmatrix} 0.3 & -0.1 & 0.5 \\ 0.2 & 0.4 & -0.3 \end{bmatrix}, \quad b_v = \begin{bmatrix} 0.1 \\ 0.0 \end{bmatrix}$$

Then:

$$h_v = \begin{bmatrix} 0.3(0.5) + (-0.1)(-0.2) + 0.5(0.8) + 0.1 \\ 0.2(0.5) + 0.4(-0.2) + (-0.3)(0.8) + 0.0 \end{bmatrix} = \begin{bmatrix} 0.15 + 0.02 + 0.40 + 0.1 \\ 0.10 - 0.08 - 0.24 + 0.0 \end{bmatrix} = \begin{bmatrix} 0.67 \\ -0.22 \end{bmatrix}$$

This tells us that the raw 3D ViT embedding has been projected into a 2D shared space. The language embeddings would be projected into the same 2D space, and now we can meaningfully combine them.


![Projection layers map different encoders into a shared space.](figures/figure_6.png)
*Projection layers map different encoders into a shared space.*


---

## The LLaVA Architecture — A Complete Walkthrough

Now let us look at a real multimodal architecture that puts all of these ideas together. **LLaVA** (Large Language-and-Vision Assistant) is one of the simplest and most elegant multimodal architectures. It is the "hello world" of vision-language models.

The key insight behind LLaVA is surprisingly simple: **treat visual patches as if they were text tokens**. If a ViT produces 196 patch embeddings and your text has 20 token embeddings, just concatenate them into a sequence of 216 tokens and feed the whole thing into an LLM.

Here is the step-by-step pipeline:

**Step 1:** The image passes through a pretrained CLIP ViT encoder. This produces a sequence of visual patch embeddings — for example, 576 patches, each of dimension 1024.

**Step 2:** These visual embeddings pass through a **linear projection layer** (or a small MLP) that maps them from the ViT's dimension (1024) to the LLM's dimension (4096). Now the visual tokens speak the same language as text tokens.

**Step 3:** The projected visual tokens are **prepended** to the text tokens. If we have 576 visual tokens and 50 text tokens, the LLM now sees a sequence of 626 tokens.

**Step 4:** This combined sequence is fed into a large language model (like LLaMA or Vicuna). The LLM processes all tokens with self-attention — including the visual tokens. It then generates text output autoregressively.


![LLaVA fuses vision and language by treating image patches as text tokens.](figures/figure_7.png)
*LLaVA fuses vision and language by treating image patches as text tokens.*


The mathematical formulation is clean:


$$
y = \text{LLM}\left(\left[W \cdot \text{ViT}(I) \;;\; \text{Embed}(T)\right]\right)
$$


Here, $W$ is the projection matrix, $I$ is the input image, $T$ is the input text, and the semicolon denotes sequence concatenation.

Let us trace through the dimensions. Suppose:
- $\text{ViT}(I) \in \mathbb{R}^{576 \times 1024}$ (576 patches, 1024-dim each)
- $W \in \mathbb{R}^{4096 \times 1024}$, so $W \cdot \text{ViT}(I) \in \mathbb{R}^{576 \times 4096}$
- $\text{Embed}(T) \in \mathbb{R}^{50 \times 4096}$ (50 text tokens, 4096-dim each)
- Concatenation: $[W \cdot \text{ViT}(I) ; \text{Embed}(T)] \in \mathbb{R}^{626 \times 4096}$

The LLM now processes a 626-token sequence, where the first 576 tokens are visual and the last 50 are textual. From the LLM's perspective, it is just processing a longer-than-usual sequence of tokens. This is exactly what we want — the self-attention mechanism inside the LLM naturally learns to correlate visual tokens with text tokens.

Here is a simplified implementation:

```python
import torch
import torch.nn as nn

class SimpleLLaVA(nn.Module):
    def __init__(self, vit_dim=1024, llm_dim=4096, vocab_size=32000):
        super().__init__()
        # Visual projection: map ViT output to LLM dimension
        self.visual_proj = nn.Linear(vit_dim, llm_dim)
        # Text embedding
        self.text_embed = nn.Embedding(vocab_size, llm_dim)
        # Simplified LLM: just a few Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=llm_dim, nhead=32, dim_feedforward=11008,
                batch_first=True
            ),
            num_layers=4
        )
        self.lm_head = nn.Linear(llm_dim, vocab_size)

    def forward(self, visual_features, text_ids):
        # Step 1-2: Project visual features to LLM dimension
        vis_tokens = self.visual_proj(visual_features)  # (B, 576, 4096)
        # Step 3: Embed text and concatenate
        txt_tokens = self.text_embed(text_ids)           # (B, N, 4096)
        combined = torch.cat([vis_tokens, txt_tokens], dim=1)  # (B, 576+N, 4096)
        # Step 4: Process through LLM
        hidden = self.transformer(combined)
        logits = self.lm_head(hidden)
        return logits

# Example usage
model = SimpleLLaVA()
vis = torch.randn(1, 576, 1024)    # ViT output
txt = torch.randint(0, 32000, (1, 50))  # Text token IDs
out = model(vis, txt)
print(f"Output shape: {out.shape}")  # (1, 626, 32000)
```

The beauty of LLaVA is its simplicity. The projection layer is the only new component — everything else is a pretrained ViT and a pretrained LLM. This makes it easy to train, easy to understand, and surprisingly effective.

But there is a limitation. Because LLaVA concatenates all visual tokens with text tokens, the sequence length grows rapidly. A single high-resolution image can produce hundreds of visual tokens. If we want to process multiple images in a conversation, the sequence length explodes. This brings us to a different approach.

---

## The Flamingo Architecture — Cross-Attention Fusion in Action

**Flamingo**, developed by DeepMind, takes a different path. Instead of concatenating visual tokens with text tokens, Flamingo uses **gated cross-attention layers** that are inserted between the frozen LLM layers.

The key innovations are:

**1. The Perceiver Resampler.** Instead of feeding all 576 visual patch tokens into the LLM, Flamingo uses a small Transformer-based module called the Perceiver Resampler. This takes in the variable-length visual tokens and outputs a fixed number of "visual summary" tokens (typically 64). Think of it as a bottleneck that compresses the visual information into a manageable number of tokens.

**2. Gated Cross-Attention.** Between every pair of LLM layers, Flamingo inserts a cross-attention layer where the LLM tokens (queries) attend to the visual summary tokens (keys and values). Crucially, these cross-attention layers are **gated**:


$$
h' = h + \tanh(\alpha) \cdot \text{CrossAttn}(h, v)
$$


Here, $h$ is the LLM hidden state, $v$ is the visual tokens, and $\alpha$ is a learnable scalar initialized to **zero**. Because $\tanh(0) = 0$, the cross-attention output starts as zero — meaning the model initially behaves exactly like the original frozen LLM. As training progresses, $\alpha$ gradually moves away from zero, allowing the visual information to flow in.

Let us plug in numbers. At the start of training, $\alpha = 0$:

$$h' = h + \tanh(0) \cdot \text{CrossAttn}(h, v) = h + 0 \cdot \text{CrossAttn}(h, v) = h$$

The output is just $h$ — no visual information at all. After some training, suppose $\alpha = 1.0$:

$$h' = h + \tanh(1.0) \cdot \text{CrossAttn}(h, v) = h + 0.76 \cdot \text{CrossAttn}(h, v)$$

Now 76% of the cross-attention output is added to the hidden state. This is a clever training trick — it prevents the randomly initialized cross-attention layers from corrupting the pretrained LLM at the start of training.


![Flamingo uses gated cross-attention to inject vision into a frozen LLM.](figures/figure_8.png)
*Flamingo uses gated cross-attention to inject vision into a frozen LLM.*


Here is a gated cross-attention block in PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        # Gate initialized to zero — starts with no visual influence
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, text_hidden, visual_tokens):
        # Cross-attention: text queries attend to visual keys/values
        attn_out = self.cross_attn(self.norm(text_hidden), visual_tokens)
        # Gated residual: tanh(gate) starts at 0, gradually opens
        return text_hidden + torch.tanh(self.gate) * attn_out

# Usage: interleave between LLM layers
gated_xattn = GatedCrossAttention(dim=4096, num_heads=32)
text_h = torch.randn(1, 50, 4096)    # LLM hidden states
vis_summary = torch.randn(1, 64, 4096)  # Perceiver resampler output
output = gated_xattn(text_h, vis_summary)
print(f"Gate value: {torch.tanh(gated_xattn.gate).item():.4f}")  # ~0.0 at init
print(f"Output shape: {output.shape}")  # (1, 50, 4096)
```

The Flamingo approach has a major advantage over LLaVA: the LLM's sequence length stays the same regardless of the number of visual tokens. The visual information enters through cross-attention side-channels, not through concatenation. This makes Flamingo far more efficient for multi-image inputs.

---

## Training Multimodal Models

Now that we understand the architectures, the question is: **how do we train these models?** Modern vision-language models typically use a **two-stage training** approach.

**Stage 1: Pretraining — Align the Modalities.** In this stage, the goal is to align the visual and textual representations so that an image of a dog and the text "a photo of a dog" end up close to each other in the shared embedding space. The dominant approach here is **contrastive learning** (CLIP-style).

The contrastive loss pushes matching image-text pairs together and non-matching pairs apart:


$$\mathcal{L}_{\text{contrastive}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp\left(\text{sim}(v_i, t_i) / \tau\right)}{\sum_{j=1}^{N}\exp\left(\text{sim}(v_i, t_j) / \tau\right)}$$

Here, $\text{sim}(v_i, t_i)$ is the cosine similarity between the visual embedding $v_i$ and its matching text embedding $t_i$, $\tau$ is a temperature parameter, and $N$ is the batch size.

Let us work through a small example with $N = 3$ and $\tau = 0.1$. Suppose our similarity matrix (cosine similarities between all image-text pairs) is:

|  | $t_1$ | $t_2$ | $t_3$ |
|---|---|---|---|
| $v_1$ | 0.9 | 0.2 | 0.1 |
| $v_2$ | 0.3 | 0.8 | 0.2 |
| $v_3$ | 0.1 | 0.1 | 0.7 |

The diagonal entries (0.9, 0.8, 0.7) are the matching pairs — these should be high. The off-diagonal entries should be low.

For the first image $v_1$:

$$\frac{\exp(0.9 / 0.1)}{\exp(0.9/0.1) + \exp(0.2/0.1) + \exp(0.1/0.1)} = \frac{\exp(9)}{\exp(9) + \exp(2) + \exp(1)} = \frac{8103.1}{8103.1 + 7.39 + 2.72} \approx 0.999$$

The loss for this pair is $-\log(0.999) \approx 0.001$ — very small, because the matching pair has much higher similarity than the non-matching pairs. This is exactly what we want.

**Stage 2: Fine-tuning — Teach the Model to Follow Instructions.** After pretraining, the model can associate images with text but cannot answer questions or follow instructions. In this stage, we fine-tune on instruction-following datasets (e.g., "What is in this image?" + image + "A golden retriever playing in the park"). The loss here is standard next-token prediction on the text output.


![Modern VLMs train in two stages: align, then instruct.](figures/figure_9.png)
*Modern VLMs train in two stages: align, then instruct.*


A practical consideration is **which parts to freeze and which to train**. In LLaVA:
- Stage 1: Freeze the ViT and the LLM; only train the projection layer
- Stage 2: Freeze the ViT; train the projection layer and the LLM together

In Flamingo:
- Freeze the ViT and the LLM entirely; only train the Perceiver Resampler and the gated cross-attention layers

Freezing pretrained components is crucial — it preserves the knowledge they have already learned and dramatically reduces the amount of data and compute needed for training.

---

## When Does Fusion Strategy Matter?

Now that we have seen all three fusion strategies in action, the natural question is: when should you use which one?


![Different tasks favor different fusion strategies.](figures/figure_10.png)
*Different tasks favor different fusion strategies.*


Here are some practical guidelines:

**Use early fusion** when both modalities are small and tightly coupled. For example, fusing sensor readings with categorical metadata in a tabular model. The simplicity is hard to beat.

**Use late fusion** when you need modular, swappable encoders and the task is primarily about matching or ranking. CLIP is essentially a late fusion model — each modality is encoded independently, and the only interaction is a cosine similarity at the end.

**Use cross-attention (mid-level) fusion** when the task requires fine-grained cross-modal reasoning. Visual question answering ("What color is the third car from the left?"), image captioning, and multimodal instruction following all benefit from the deep cross-modal interaction that cross-attention provides.

The trend in the field is clearly toward cross-attention and token-level fusion. Modern systems like GPT-4V, Gemini, and LLaVA-NeXT all use variants of mid-level fusion, because the tasks that matter most — reasoning, following instructions, generating grounded descriptions — require rich cross-modal interaction.

---

## Summary

Let us recap what we have covered.

**Multimodal fusion** is the problem of combining information from different modalities (vision, text, audio) into a unified representation. There are three canonical strategies:

1. **Early fusion** — concatenate raw inputs and process jointly. Simple but inflexible.
2. **Late fusion** — encode each modality independently, combine at the end. Flexible but limited cross-modal interaction.
3. **Mid-level fusion (cross-attention)** — allow modalities to interact at intermediate layers. The dominant approach in modern VLMs.

We walked through two real architectures:
- **LLaVA** — projects visual patches into the LLM's token space and concatenates them with text tokens. Simple, effective, and easy to train.
- **Flamingo** — uses a Perceiver Resampler and gated cross-attention to inject visual information into a frozen LLM. More efficient for multi-image inputs.

Training happens in two stages: **alignment** (contrastive pretraining) and **instruction tuning** (next-token prediction on instruction data).

The choice of fusion strategy depends on your task. For fine-grained cross-modal reasoning, cross-attention is king. For retrieval and ranking, late fusion works well. For small-scale problems, early fusion is the simplest path.

**References:**
- Liu et al., "Visual Instruction Tuning" (LLaVA), 2023. [Paper](https://arxiv.org/abs/2304.08485)
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", 2022. [Paper](https://arxiv.org/abs/2204.14198)
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021. [Paper](https://arxiv.org/abs/2103.00020)

That's it!
