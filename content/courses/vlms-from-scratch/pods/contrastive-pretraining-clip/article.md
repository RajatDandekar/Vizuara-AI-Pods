# Contrastive Pretraining (CLIP-style): Teaching Machines to See and Read at the Same Time

## How CLIP learned to connect images and text in a shared space -- from first principles to implementation

Let us start with a simple thought experiment. Imagine a librarian who has spent years looking at millions of photographs, each with a short caption written on the back. Over time, this librarian develops an extraordinary skill: show them any new photograph they have never seen, and they can describe it. Give them any description, and they can find the right photograph.

This is essentially what CLIP (Contrastive Language-Image Pretraining) does. Published by OpenAI in 2021, CLIP learned to connect images and text by training on 400 million image-text pairs scraped from the internet. The result was remarkable -- a model that could classify images into categories it had never been explicitly trained on, simply by understanding the relationship between visual content and natural language.

But how does a machine learn to "see" and "read" at the same time? This brings us to the core idea behind CLIP: **contrastive pretraining**.


![CLIP learns to match images and text by embedding them into the same space.](figures/figure_1.png)
*CLIP learns to match images and text by embedding them into the same space.*


## What is Contrastive Learning?

Before we dive into CLIP, let us understand the fundamental idea behind contrastive learning.

Think about how you organize your music library. You instinctively group similar songs together -- jazz in one playlist, rock in another. Songs that sound similar end up near each other, and songs that sound completely different end up far apart.

Contrastive learning does something very similar, but in a mathematical embedding space. The idea is simple:

1. Take a pair of items that **should** be similar (a positive pair)
2. Take pairs of items that **should not** be similar (negative pairs)
3. Learn an embedding function that pulls positive pairs close together and pushes negative pairs apart

For CLIP specifically, a positive pair is an image and its correct caption. A negative pair is an image with someone else's caption.


![Contrastive learning pulls matching pairs together while pushing others apart.](figures/figure_2.png)
*Contrastive learning pulls matching pairs together while pushing others apart.*


This is the key insight: we do not need manually assigned labels like "dog" or "cat." Instead, the captions that naturally accompany images on the internet provide all the supervision we need. This is why CLIP is called a **self-supervised** (or more precisely, **naturally supervised**) method.

## The CLIP Architecture

Now let us understand how CLIP is actually built. The architecture is surprisingly elegant -- it consists of just two components:

1. **An Image Encoder** -- This takes an image and converts it into a fixed-size vector (say, 512 dimensions). CLIP uses either a Vision Transformer (ViT) or a ResNet for this.

2. **A Text Encoder** -- This takes a text caption and converts it into a fixed-size vector of the same dimension. CLIP uses a standard Transformer for this.

Both encoders are independent -- they do not share any parameters. Each one is specialized for its own modality (images or text), but they both output vectors that live in the same shared embedding space.


![CLIP uses two separate encoders that project into a shared embedding space.](figures/figure_3.png)
*CLIP uses two separate encoders that project into a shared embedding space.*


After encoding, both vectors are normalized to have unit length (they lie on the surface of a hypersphere). The similarity between an image and a text is then computed as their **cosine similarity**, which for unit vectors is simply the dot product:


$$\text{sim}(I, T) = \frac{I \cdot T}{\|I\| \|T\|} = I \cdot T \quad \text{(when } \|I\| = \|T\| = 1\text{)}$$

Let us plug in some simple numbers to see how this works. Suppose our embeddings are 3-dimensional (instead of 512) for simplicity. Let the image embedding be $$I = [0.6, 0.8, 0.0]$$ and the text embedding be $$T = [0.5, 0.7, 0.5]$$. First, we normalize them:

$$\|I\| = \sqrt{0.36 + 0.64 + 0} = 1.0$$, $$\|T\| = \sqrt{0.25 + 0.49 + 0.25} = \sqrt{0.99} \approx 0.995$$.

The normalized text embedding becomes approximately $$T' = [0.503, 0.703, 0.503]$$.

The cosine similarity is: $$\text{sim} = 0.6 \times 0.503 + 0.8 \times 0.703 + 0.0 \times 0.503 = 0.302 + 0.562 + 0 = 0.864$$.

This value of 0.864 (close to 1.0) tells us that these two embeddings are quite similar -- which is exactly what we want for a matching image-text pair.

Here is a simple PyTorch implementation of this dual-encoder architecture:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # Learnable temperature parameter (log-scale for stability)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    def forward(self, images, texts):
        # Encode both modalities
        image_features = self.image_encoder(images)   # (batch, embed_dim)
        text_features = self.text_encoder(texts)       # (batch, embed_dim)

        # Normalize to unit vectors
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute scaled cosine similarity matrix
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T  # (batch, batch)

        return logits
```

Let us understand this code. The `image_encoder` and `text_encoder` are separate neural networks that produce embedding vectors. We normalize these vectors using `F.normalize` so they have unit length. Then we compute the similarity matrix by taking the matrix product of image features with the transpose of text features. The `logit_scale` is a learned temperature parameter -- we will explain why this matters shortly.

## The Contrastive Loss Function: The Heart of CLIP

Now we come to the most important part -- the loss function that teaches CLIP to align images and text.

Here is the key insight: in a batch of $$N$$ image-text pairs, we create an $$N \times N$$ similarity matrix. The entry at position $$(i, j)$$ is the cosine similarity between image $$i$$ and text $$j$$. The diagonal entries are the **positive pairs** (image $$i$$ with its own caption $$i$$), and all off-diagonal entries are **negative pairs**.


![The similarity matrix turns every batch into a self-supervised classification problem.](figures/figure_4.png)
*The similarity matrix turns every batch into a self-supervised classification problem.*


The loss function treats each row of this matrix as a classification problem: for each image, which text is the correct match? This is exactly a cross-entropy loss where the "correct class" is the diagonal entry:


$$\mathcal{L}_{i \to t} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}$$

Let us plug in some simple numbers with $$N = 3$$ and $$\tau = 0.1$$. Suppose our similarity matrix looks like this:

| | Text 1 | Text 2 | Text 3 |
|---|---|---|---|
| Image 1 | **0.9** | 0.2 | 0.1 |
| Image 2 | 0.3 | **0.8** | 0.15 |
| Image 3 | 0.1 | 0.25 | **0.85** |

For Image 1, we need the softmax probability of the correct match (Text 1):

$$\frac{\exp(0.9 / 0.1)}{\exp(0.9/0.1) + \exp(0.2/0.1) + \exp(0.1/0.1)} = \frac{\exp(9)}{\exp(9) + \exp(2) + \exp(1)} = \frac{8103.1}{8103.1 + 7.39 + 2.72} = \frac{8103.1}{8113.2} = 0.9988$$

The negative log of this is $$-\log(0.9988) = 0.0012$$. This is very small -- which tells us that Image 1 is correctly matched to Text 1 with high confidence. This is exactly what we want.

Similarly, CLIP also computes the loss in the text-to-image direction (treating each column as a classification problem). The final loss is the average of both directions:


$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i}\right)$$

Let us complete the example. The text-to-image loss follows the same pattern but transposes the matrix. For Text 1: $$\frac{\exp(9)}{\exp(9) + \exp(3) + \exp(1)} = \frac{8103.1}{8103.1 + 20.09 + 2.72} = 0.9972$$. The loss is $$-\log(0.9972) = 0.0028$$.

The symmetric CLIP loss for this batch would average the image-to-text and text-to-image losses. Since both are very small in this example, our model is doing a good job at matching the pairs. This is exactly what we want.

## Why Temperature Matters

You might have noticed the temperature parameter $$\tau$$ in the loss function. But why do we need it?

The temperature controls how "sharp" or "flat" the softmax distribution is. Let us see this with a concrete example.

Consider the same similarities for Image 1: $$[0.9, 0.2, 0.1]$$.

With **low temperature** $$\tau = 0.07$$:

$$\text{softmax} = \left[\frac{\exp(12.86)}{\sum}, \frac{\exp(2.86)}{\sum}, \frac{\exp(1.43)}{\sum}\right] \approx [0.99998, 0.00001, 0.00001]$$

With **high temperature** $$\tau = 1.0$$:

$$\text{softmax} = \left[\frac{\exp(0.9)}{\sum}, \frac{\exp(0.2)}{\sum}, \frac{\exp(0.1)}{\sum}\right] = \left[\frac{2.46}{4.67}, \frac{1.22}{4.67}, \frac{1.11}{4.67}\right] \approx [0.53, 0.26, 0.24]$$


![Lower temperature sharpens the softmax, making the model more decisive.](figures/figure_5.png)
*Lower temperature sharpens the softmax, making the model more decisive.*


With low temperature, the model is very confident -- it assigns nearly all probability to the correct match. With high temperature, the distribution is much flatter. CLIP initializes $$\tau$$ at 0.07 and learns it during training. The model typically converges to a very low temperature, meaning it learns to be decisive about its matches.

An interesting detail: CLIP actually parameterizes this as $$\exp(t)$$ where $$t$$ is the learnable parameter, and clips the value to prevent $$\tau$$ from getting too small (which would cause numerical instability).

## Training at Scale

One of the most important insights from the CLIP paper is that **scale matters enormously** for contrastive learning.

CLIP was trained on a dataset of 400 million image-text pairs collected from the internet, called WIT (WebImageText). To put this in perspective, ImageNet -- the standard benchmark -- has only 1.2 million labeled images. CLIP's dataset is more than 300 times larger.


![CLIP's performance comes from training on 400M image-text pairs from the web.](figures/figure_6.png)
*CLIP's performance comes from training on 400M image-text pairs from the web.*


But it is not just the data -- the batch size is also critical. CLIP was trained with batch sizes of 32,768. Why does this matter? In contrastive learning, each batch provides $$N - 1$$ negative pairs for every positive pair. With a batch size of 32,768, each image is compared against 32,767 wrong captions. This is a massive amount of contrastive signal in every training step.

The training used mixed-precision (FP16) computation across hundreds of GPUs. The image encoder (ViT-L/14) has about 300 million parameters, and the text encoder has about 63 million parameters. The full training took several weeks on a large GPU cluster.

## Zero-Shot Classification: The Magic of CLIP

Now let us come to the main character in our story -- the part that made CLIP truly revolutionary.

Once CLIP is trained, it can classify images into **any** set of categories without ever seeing a single labeled example. This is called **zero-shot classification**, and it works like this:

1. Define your categories (e.g., "dog", "cat", "car", "tree")
2. Create text prompts: "a photo of a dog", "a photo of a cat", etc.
3. Encode each text prompt using CLIP's text encoder
4. Encode the input image using CLIP's image encoder
5. Compute cosine similarity between the image and each text prompt
6. The category with the highest similarity is the prediction


![Zero-shot classification compares image embeddings against text prompt embeddings.](figures/figure_7.png)
*Zero-shot classification compares image embeddings against text prompt embeddings.*


This is remarkable. A traditional image classifier needs thousands of labeled examples per class. CLIP needs zero -- just a text description of each class.

Here is how you would implement zero-shot classification:

```python
import torch
import torch.nn.functional as F

def zero_shot_classify(image, class_names, clip_model, image_preprocess, tokenizer):
    """
    Classify an image into one of the given classes using CLIP.

    Args:
        image: input image (PIL or tensor)
        class_names: list of class names, e.g. ["dog", "cat", "car"]
        clip_model: pretrained CLIP model
        image_preprocess: image preprocessing function
        tokenizer: text tokenizer
    """
    # Create text prompts for each class
    prompts = [f"a photo of a {cls}" for cls in class_names]

    # Encode the image
    image_input = image_preprocess(image).unsqueeze(0)
    image_features = clip_model.image_encoder(image_input)
    image_features = F.normalize(image_features, dim=-1)

    # Encode all text prompts
    text_inputs = tokenizer(prompts)
    text_features = clip_model.text_encoder(text_inputs)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarities
    similarities = (image_features @ text_features.T).squeeze(0)

    # Convert to probabilities
    probs = F.softmax(similarities * clip_model.logit_scale.exp(), dim=-1)

    # Return the top prediction
    top_idx = probs.argmax().item()
    return class_names[top_idx], probs[top_idx].item()
```

The beauty of this approach is that you can change the classes at any time. Want to classify animals? Use "dog", "cat", "bird". Want to classify emotions? Use "happy", "sad", "angry". The model does not need to be retrained -- you just change the text prompts.

In the original paper, CLIP achieved 76.2% accuracy on ImageNet zero-shot -- without ever seeing a single ImageNet training image. This is competitive with fully supervised ResNet-50, which was trained on all 1.2 million labeled ImageNet images. Not bad right?

## Practical Implementation: Building Mini-CLIP

Let us now build a simplified version of CLIP to see these ideas in action. We will train on CIFAR-10 with synthetic captions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# --- Image Encoder (Simple CNN) ---
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.conv(x).squeeze(-1).squeeze(-1)
        return self.projection(x)

# --- Text Encoder (Simple Embedding + Transformer) ---
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, max_len=32):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens):
        x = self.token_embed(tokens) + self.pos_embed[:, :tokens.size(1)]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling over sequence
        return self.projection(x)

# --- Contrastive Loss ---
def clip_loss(image_features, text_features, temperature=0.07):
    """Symmetric contrastive loss (InfoNCE)."""
    # Normalize
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Similarity matrix
    logits = (image_features @ text_features.T) / temperature

    # Labels: diagonal entries are the correct matches
    labels = torch.arange(logits.size(0), device=logits.device)

    # Symmetric cross-entropy loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2

# --- Training Loop ---
# image_encoder = ImageEncoder(embed_dim=128)
# text_encoder = TextEncoder(embed_dim=128)
# optimizer = torch.optim.AdamW(
#     list(image_encoder.parameters()) + list(text_encoder.parameters()),
#     lr=3e-4, weight_decay=0.01
# )
# for epoch in range(num_epochs):
#     for images, labels in dataloader:
#         text_tokens = generate_captions(labels)  # "a photo of a dog", etc.
#         img_feats = image_encoder(images)
#         txt_feats = text_encoder(text_tokens)
#         loss = clip_loss(img_feats, txt_feats)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
```

Let us understand this code in detail. The `ImageEncoder` is a simple CNN that takes 32x32 CIFAR-10 images and produces 128-dimensional embeddings. The `TextEncoder` uses token embeddings, positional embeddings, and a small Transformer to convert text tokens into 128-dimensional embeddings. The `clip_loss` function computes the symmetric InfoNCE loss we described earlier.

The key line is `logits = (image_features @ text_features.T) / temperature`. This creates the $$N \times N$$ similarity matrix, scaled by temperature. The labels `torch.arange(N)` tell the cross-entropy loss that the diagonal entries are the correct matches.


![Mini-CLIP learns meaningful representations in just a few epochs.](figures/figure_8.png)
*Mini-CLIP learns meaningful representations in just a few epochs.*


## Limitations and Extensions

While CLIP is powerful, it is not perfect. There are several well-documented limitations:

**Counting and spatial reasoning.** If you ask CLIP to distinguish "three dogs" from "five dogs," or "a cat on top of a box" from "a cat inside a box," it often fails. The model learns coarse semantic alignment, not fine-grained compositional understanding.

**Negation.** CLIP struggles with negation. "A photo with no people" and "a photo with people" may receive similar similarity scores, because the model picks up on the word "people" regardless of "no."

**Domain-specific performance.** CLIP was trained on internet data, which has biases. It performs well on common objects and scenes but poorly on specialized domains like medical imaging or satellite imagery.

Several improvements have been proposed:

- **SigLIP** replaces the softmax-based loss with a sigmoid loss, which allows each image-text pair to be evaluated independently rather than relative to the batch. This removes the need for large batch sizes.
- **OpenCLIP** is an open-source reproduction that has trained CLIP models on various open datasets, making the technology accessible to everyone.
- **ALIGN** by Google trained on an even noisier dataset of 1.8 billion image-text pairs with minimal filtering, showing that scale can compensate for data quality.


![CLIP's representations power a wide ecosystem of downstream applications.](figures/figure_9.png)
*CLIP's representations power a wide ecosystem of downstream applications.*


The most impactful legacy of CLIP is as a **foundation model component**. Modern vision-language models like LLaVA, GPT-4V, and Gemini all use CLIP (or CLIP-like) vision encoders as their "eyes." The representations learned through contrastive pretraining have proven to be remarkably transferable across a huge range of tasks.

## Conclusion

Contrastive pretraining, as demonstrated by CLIP, introduced a fundamental shift in how we train vision models. Instead of relying on expensive manually-labeled datasets, CLIP showed that natural language supervision -- the captions that already exist alongside images on the internet -- provides a richer, more scalable form of supervision.

The core idea is elegant: embed images and text into a shared space, pull matching pairs together, push non-matching pairs apart. This simple objective, applied at sufficient scale, produces representations that generalize remarkably well to tasks the model has never seen.

For further reading, refer to the original paper: "Learning Transferable Visual Models From Natural Language Supervision" by Radford et al. (2021). You can also explore the OpenCLIP repository for open-source implementations and pretrained models.

That's it!
