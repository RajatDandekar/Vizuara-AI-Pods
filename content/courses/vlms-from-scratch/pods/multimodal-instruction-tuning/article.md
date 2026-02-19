# Multimodal Instruction Tuning: The Training Recipe That Teaches LLMs to See

*From frozen projectors to GPT-4 data generation — the complete training recipe behind LLaVA and beyond.*

---

Let us start where we left off. In the previous article on Multimodal Fusion Architectures, we built the LLaVA architecture from scratch — a CLIP vision encoder, a projection layer, and an LLM, all wired together so that image patches become tokens the language model can process. We traced through the dimensions. We wrote the code. We understood why it works.

But right now, that architecture is useless.

The projection layer is initialized with random weights. If you feed it an image of a golden retriever and ask "What is in this picture?", the model will produce nonsense — random token sequences that have nothing to do with dogs, parks, or anything visual. The architecture is correct, but the model has not learned to use it.

This brings us to the central question of this article: **How do we train a multimodal model to actually follow instructions about images?**

The answer turns out to be a carefully designed two-stage training recipe, a clever data generation pipeline, and a set of design choices that each have measurable impact on the final model. Let us dig into every detail.


![Untrained vs trained VLM — same architecture, different training.](figures_v2/figure_1.png)
*Untrained vs trained VLM — same architecture, different training.*


---

## The Two-Stage Training Recipe — Every Detail

The training procedure for LLaVA is deceptively simple on the surface: two stages, each with a different objective. But the hyperparameter choices, the freezing strategy, and the data composition at each stage all matter enormously. Let us go through each stage with the precision it deserves.

### Stage 1: Feature Alignment Pretraining

In this first stage, we have one goal: teach the projection layer to produce visual tokens that the LLM can interpret. Think of it as calibrating a translator — before we ask them to handle complex negotiations, they need to get the basic vocabulary right.

**What is frozen:** Both the CLIP vision encoder and the LLM are completely frozen. Their weights do not change at all. Only the projection layer (a 2-layer MLP with ~20M parameters) receives gradients and updates.

**The training data:** 558K image-caption pairs filtered from the CC3M dataset. Each example is an image paired with a short descriptive caption like "a black cat sitting on a wooden table" or "sunset over the Pacific Ocean."

**The loss function:** Standard autoregressive next-token prediction on the caption, conditioned on the visual tokens:

$$\mathcal{L}_{\text{stage1}} = -\sum_{t=1}^{T} \log \, p_\theta(x_t \mid x_{<t}, H_v)$$

Here, $x_t$ is the $t$-th caption token, $x_{<t}$ are all previous caption tokens, and $H_v$ are the projected visual tokens. The only trainable parameters $\theta$ are in the projection layer.

Let us plug in some simple numbers. Suppose our caption is three tokens: ["a", "red", "car"]. After the forward pass, the model predicts:

- $p(\text{"a"} \mid H_v) = 0.6$
- $p(\text{"red"} \mid H_v, \text{"a"}) = 0.3$
- $p(\text{"car"} \mid H_v, \text{"a"}, \text{"red"}) = 0.5$

Then:

$$\mathcal{L} = -[\log(0.6) + \log(0.3) + \log(0.5)] = -[-0.51 + (-1.20) + (-0.69)] = 2.40$$

This loss of 2.40 tells us the model is uncertain — especially about "red". As training progresses, the projection layer adjusts its weights so that visual tokens from a red car image make the word "red" more predictable. After thousands of updates, this loss drops below 1.0.

**The hyperparameters** are critical and often overlooked:

- **Learning rate:** $1 \times 10^{-3}$ (relatively high, because the projector is randomly initialized)
- **Batch size:** 256
- **Optimizer:** AdamW with weight decay 0
- **Epochs:** 1 (just a single pass through the data)
- **Duration:** ~5.5 hours on 8x A100 GPUs

The high learning rate is intentional. Since the projector starts from random initialization, we need aggressive updates to quickly move it from noise to a reasonable mapping. The frozen LLM and ViT are safe from these large gradients.


![Stage 1: only the projector trains.](figures_v2/figure_2.png)
*Stage 1: only the projector trains.*


### Stage 2: Visual Instruction Tuning

In the second stage, the goal shifts dramatically. We no longer just want the model to describe images — we want it to follow complex instructions, answer questions, and reason about visual content.

**What changes:** The projection layer **and** the entire LLM are now unfrozen and trained together. Only the vision encoder remains frozen.

**The training data:** 665K instruction-following examples (in LLaVA-1.5). These include visual question answering, detailed image descriptions, and multi-step reasoning about images.

**The loss function:** The same next-token prediction, but now applied to instruction-following responses:

$$\mathcal{L}_{\text{stage2}} = -\sum_{t=1}^{T} \log \, p_\theta(x_t \mid x_{<t}, H_v, \text{instruction})$$

**The hyperparameters change dramatically:**

- **Learning rate:** $2 \times 10^{-5}$ — this is **50 times smaller** than Stage 1
- **Batch size:** 128
- **Optimizer:** AdamW with weight decay 0
- **Epochs:** 1
- **Duration:** ~20 hours on 8x A100 GPUs

The learning rate drop from $10^{-3}$ to $2 \times 10^{-5}$ is one of the most important design choices. The LLM has been pretrained on trillions of text tokens — its weights encode vast knowledge about language, reasoning, and the world. If we used a large learning rate, we would destroy this knowledge in the first few gradient updates. The small learning rate lets the LLM gently adapt to visual inputs without forgetting how to use language.

Let us see why numerically. Suppose a weight in the LLM has value $w = 0.5$ and the gradient is $g = 0.01$. With the Stage 1 learning rate:

$$w_{\text{new}} = 0.5 - (1 \times 10^{-3})(0.01) = 0.5 - 0.00001 = 0.49999$$

With the Stage 2 learning rate:

$$w_{\text{new}} = 0.5 - (2 \times 10^{-5})(0.01) = 0.5 - 0.0000002 = 0.4999998$$

The Stage 2 update is 50 times smaller. This is exactly what we want — gentle adjustments that preserve the LLM's knowledge while teaching it to incorporate visual information.


![Stage 2: LLM and projector train together.](figures_v2/figure_3.png)
*Stage 2: LLM and projector train together.*


---

## Training Dynamics — What Happens Under the Hood

Now let us look at what actually happens during training. Understanding the dynamics gives us intuition for why certain design choices matter.

### Loss Curves

In Stage 1, the loss drops rapidly. Captioning is a relatively simple task — the model just needs to learn that images of dogs produce words about dogs. The projector, starting from random initialization, quickly finds a mapping that makes captions predictable. Typical Stage 1 training shows the loss dropping from ~7.0 to ~2.0 within the first epoch.

In Stage 2, the loss drops more slowly and from a lower starting point. Instruction following is fundamentally harder than captioning — the model needs to reason about spatial relationships, read text in images, count objects, and generate coherent multi-sentence responses. The loss starts around ~1.5 and drops to ~0.8 over the training epoch.

### Why Freezing in Stage 1 is Critical

Here is an experiment that reveals why the two-stage approach is necessary. What happens if we skip Stage 1 and go directly to instruction tuning with the LLM unfrozen?

The projector starts with random weights. Its outputs are noise — random vectors in the LLM's embedding space. When the LLM sees these noisy visual tokens, it generates random outputs, producing large gradients. These gradients flow back through the LLM, updating its weights based on garbage visual signal. The result: the LLM's language capabilities degrade.

With the two-stage approach, Stage 1 ensures the projector first learns a reasonable mapping. By the time we unfreeze the LLM in Stage 2, the visual tokens are meaningful — they actually encode visual information. The gradients flowing through the LLM are now informative rather than destructive.

### Learning Rate Schedule

Both stages use a cosine learning rate schedule with a warmup period. The schedule follows:

$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

Here, $\eta_{\max}$ is the peak learning rate, $\eta_{\min}$ is the minimum (typically 0), $t$ is the current step, and $T$ is the total number of steps.

Let us plug in numbers for Stage 2. With $\eta_{\max} = 2 \times 10^{-5}$, $\eta_{\min} = 0$, and at the midpoint $t = T/2$:

$$\eta(T/2) = 0 + \frac{1}{2}(2 \times 10^{-5})(1 + \cos(\pi/2)) = \frac{1}{2}(2 \times 10^{-5})(1 + 0) = 1 \times 10^{-5}$$

At the midpoint of training, the learning rate has decayed to half the peak value. By the end, $\cos(\pi) = -1$, so $\eta(T) = 0$. The cosine schedule provides a smooth, gradual decay that helps the model converge to a good solution without oscillating.


![Training loss curves for Stage 1 (fast) and Stage 2 (gradual).](figures_v2/figure_4.png)
*Training loss curves for Stage 1 (fast) and Stage 2 (gradual).*


---

## The GPT-4 Data Generation Pipeline

Now let us look at where the instruction-following data comes from. This is one of the cleverest parts of the LLaVA pipeline, and it deserves a detailed walkthrough.

### The Problem

We need tens of thousands of examples where a human asks a question about an image and gets a detailed, accurate response. Creating these manually would require hiring annotators to look at each image, think of interesting questions, and write comprehensive answers. At the scale we need (158K+ examples), this is prohibitively expensive and slow.

### The Solution: GPT-4 as a Data Generator

LLaVA's insight is that we already have rich textual descriptions of images — the COCO dataset provides human-written captions and object bounding box annotations for over 100K images. We can feed these text descriptions to GPT-4 (a text-only model) and ask it to generate instruction-following conversations as if it were looking at the actual image.

GPT-4 never sees a single pixel. It works entirely from the text metadata. But because COCO captions are detailed and accurate, GPT-4 can generate surprisingly high-quality instruction data.

### The Three Data Types

LLaVA generates three types of instruction data, each serving a different purpose:

**1. Conversation Data (58K examples)**

Multi-turn dialogues where a user asks progressively deeper questions about an image.

Consider an image with COCO captions: "A woman in a blue dress standing next to a bicycle on a cobblestone street" and bounding boxes: [person, bicycle, building, flower pot].

GPT-4 generates a conversation like:

> **User:** What do you see in this image?
> **Assistant:** The image shows a woman wearing a blue dress standing on a cobblestone street next to a bicycle. There are buildings in the background and a flower pot visible nearby, suggesting this is likely a European city street.
> **User:** What might the woman be doing?
> **Assistant:** Based on the scene, she appears to be either about to ride the bicycle or has just stopped. The cobblestone street and surrounding architecture suggest she might be exploring a historic district, possibly as a tourist or during a daily commute.

**2. Detailed Description Data (23K examples)**

Long, paragraph-length descriptions that comprehensively cover everything in the image.

For the same image, GPT-4 might generate:

> **Assistant:** This image depicts an outdoor urban scene on a cobblestone street. A woman wearing a blue dress is standing beside a bicycle. The street is lined with multi-story buildings featuring traditional European architecture. A flower pot with colorful blooms is positioned near the building entrance. The lighting suggests the photo was taken during late afternoon, with warm sunlight casting long shadows across the cobblestones. The overall composition gives the scene a charming, relaxed atmosphere.

**3. Complex Reasoning Data (77K examples)**

Questions that require multi-step reasoning, inference, or world knowledge.

> **User:** What season do you think this photo was taken in, and what clues support your answer?
> **Assistant:** Several clues suggest this photo was taken in late spring or summer. First, the woman is wearing a light blue dress without any jacket or coat, indicating warm weather. Second, the flower pot contains blooming flowers, which is typical of warmer months. Third, the quality of light — warm and at a low angle — is consistent with late afternoon in summer. The cobblestone street appears dry, suggesting it has not rained recently, which is more common in summer months in European cities.


![GPT-4 generates three types of instruction data from image descriptions.](figures_v2/figure_5.png)
*GPT-4 generates three types of instruction data from image descriptions.*


### Why This Data Works So Well

The quality of GPT-4's generated data comes from three properties:

1. **Diversity of questions:** GPT-4 naturally generates varied question types — from simple identification ("What is in this image?") to complex inference ("What might happen next?").

2. **Reasoning chains in answers:** GPT-4 does not just state facts — it explains its reasoning. "The woman appears to be a tourist because of the bicycle rental and the camera around her neck." This teaches the VLM to reason, not just recognize.

3. **Consistency with visual content:** Because GPT-4 works from accurate COCO captions and bounding boxes, the generated data stays grounded in what the image actually contains. There is minimal hallucination in the training data itself.

---

## Ablation Studies — What Actually Matters

One of the most valuable contributions of the LLaVA papers is a thorough set of ablation studies. These experiments isolate individual design choices and measure their impact. Let us walk through the key findings.

### What Happens If You Skip Stage 1?

If you go directly to instruction tuning without the feature alignment stage, performance drops across all benchmarks. On VQAv2, the accuracy drops from 80.0% to 73.6% — a significant gap. On GQA, the drop is from 63.3% to 57.1%.

This confirms our earlier analysis: the projector must learn a reasonable visual-to-language mapping before the LLM can meaningfully adapt to visual inputs.

### Linear Projector vs MLP Projector

LLaVA-1.0 used a single linear layer as the projector. LLaVA-1.5 upgraded to a two-layer MLP with GELU activation. The difference matters.

A linear projector computes:

$$h = W \cdot z + b$$

An MLP projector computes:

$$h = W_2 \cdot \text{GELU}(W_1 \cdot z + b_1) + b_2$$

Let us see why the MLP is better with a concrete example. Suppose we have two different visual patches: $z_A = [1.0, 0.5]$ (representing a "bright red" patch) and $z_B = [0.5, 1.0]$ (representing a "dark blue" patch). We want the projector to map both to their correct positions in the LLM's embedding space.

With a linear projector $W = \begin{bmatrix} 0.3 & 0.7 \\ 0.8 & 0.2 \end{bmatrix}$:

$$h_A = W \cdot z_A = [0.3(1.0) + 0.7(0.5),\; 0.8(1.0) + 0.2(0.5)] = [0.65, 0.90]$$

$$h_B = W \cdot z_B = [0.3(0.5) + 0.7(1.0),\; 0.8(0.5) + 0.2(1.0)] = [0.85, 0.60]$$

The linear mapping is limited — it can only scale and rotate the input space. With a non-linear MLP, the GELU activation allows the projector to bend the space, creating more flexible mappings. On benchmarks, the MLP projector improves performance by 1-3% across the board. Not dramatic, but consistent and free.

### Effect of Instruction Data Volume

More instruction data generally helps, but the returns diminish. Going from 80K to 158K examples improves VQAv2 by ~2%. Going from 158K to 665K (as in LLaVA-1.5) improves it by another ~1.5%. The first doubling matters more than the second.

What matters even more than volume is **quality and diversity**. The LLaVA-1.5 paper showed that mixing academic VQA datasets (like VQAv2, GQA, TextVQA training sets) with GPT-4-generated data produces better results than using either source alone. Academic datasets provide grounding in specific visual skills. GPT-4 data provides reasoning depth and instruction diversity.

### Effect of Image Resolution

Increasing the input resolution from 224x224 to 336x336 improves performance significantly on tasks that require reading text or understanding small objects. TextVQA accuracy jumps from 46.1% to 61.3% — a 15-point improvement from resolution alone.

The reason is straightforward: at 224x224, each ViT patch covers a 16x16 = 256 pixel region (for ViT-L/14). Small text that is only 10 pixels tall gets compressed into a single patch. At 336x336, the same text spans more patches, giving the model more information to work with.


![Ablation studies: every design choice has measurable impact.](figures_v2/figure_6.png)
*Ablation studies: every design choice has measurable impact.*


---

## Evaluation — What Do Multimodal Benchmarks Actually Test?

When we see that LLaVA-1.5 scores 80.0% on VQAv2 and 63.3% on GQA, what do these numbers actually mean? Understanding what each benchmark tests — and what it misses — is essential for interpreting results.

### VQAv2: The Generalist Benchmark

VQAv2 contains over 200K images with 1.1M questions. The questions are deliberately designed so that the same question on different images has different answers (to prevent language-only shortcuts).

What it tests well: basic object recognition ("What animal is in the photo?"), attribute identification ("What color is the car?"), counting ("How many people are there?"), and simple spatial relationships.

What it misses: complex multi-step reasoning, detailed scene understanding, and nuanced interpretation. A model can score 70%+ on VQAv2 by mastering simple pattern matching without any deep visual reasoning.

### GQA: Compositional Reasoning

GQA goes deeper. It uses scene graph annotations to generate questions that require multiple reasoning steps: "Is the person to the left of the table holding a cup?" To answer correctly, the model must locate the table, find the person to its left, and check what they are holding.

Scores on GQA are typically 10-15% lower than VQAv2, reflecting the increased difficulty of compositional reasoning.

### TextVQA: Can the Model Read?

TextVQA presents images containing text (signs, labels, book covers, screens) and asks questions that require reading. "What does the sign say?" or "What is the phone number on the building?"

This benchmark directly tests the model's OCR ability. It is where image resolution has the largest impact — higher resolution means more readable text in the visual patches.

### MM-Vet: The Integrated Challenge

MM-Vet is the most comprehensive benchmark. It evaluates six core capabilities: recognition, knowledge, OCR, spatial awareness, language generation, and math. Each question may require multiple capabilities simultaneously.

For example: "What is the total cost of the items shown?" requires OCR (reading prices), counting (how many items), and math (addition). This is why MM-Vet scores are typically the lowest — the model must combine multiple skills in a single response.


![What each benchmark actually evaluates.](figures_v2/figure_7.png)
*What each benchmark actually evaluates.*


### How to Interpret Scores

Here is a practical guide for reading benchmark results:

- A model scoring 80%+ on VQAv2 can handle everyday visual questions reliably.
- A model scoring 65%+ on GQA can do basic compositional reasoning.
- A model scoring 60%+ on TextVQA can read reasonably clear text in images.
- A model scoring 35+ on MM-Vet has integrated visual reasoning ability, though it will still make frequent errors on complex questions.

When comparing models, look at the benchmark where the models differ most. If Model A beats Model B on TextVQA but loses on GQA, it means A has better OCR but weaker compositional reasoning. This kind of nuanced interpretation matters far more than average scores.

---

## Failure Modes — Where Instruction Tuning Breaks

Understanding when a model fails is just as important as understanding when it succeeds. Instruction-tuned VLMs have several well-documented failure modes.

### Hallucination

The most common failure. The model describes objects or details that simply are not in the image. You show it a photo of an empty table, and it says "A cat is sitting on the table." This happens because the LLM's language prior is extremely strong — it has seen millions of sentences about cats on tables, and this prior can override the visual evidence.

Hallucination rates vary significantly across models. Models trained on higher-quality data hallucinate less. The POPE benchmark specifically measures object hallucination: "Is there a [object] in the image?" Models like LLaVA-1.5 achieve ~86% accuracy on POPE, meaning they still hallucinate about 14% of the time on simple yes/no questions.

### Spatial Reasoning Errors

Ask "Is the red ball to the left or right of the blue box?" and the model frequently gets it wrong. The problem is fundamental: ViT processes image patches with position embeddings, but the absolute spatial layout is partially lost by the time information reaches the LLM. The model knows both objects are present but struggles with their exact spatial relationship.

### Counting Errors

"How many birds are in the tree?" is surprisingly difficult. When multiple similar objects appear in an image, the self-attention mechanism tends to merge their representations rather than keeping separate counts. Models become increasingly unreliable as the count goes above 5-6 objects.

### OCR Limitations

Reading text in images fails when the text is small (under ~15 pixels tall at 336x336 resolution), rotated, handwritten, or partially occluded. LLaVA-NeXT's high-resolution approach significantly improves OCR, but cursive and artistic fonts remain challenging.


![Common VLM failure modes: hallucination, spatial, counting, OCR.](figures_v2/figure_8.png)
*Common VLM failure modes: hallucination, spatial, counting, OCR.*


---

## Scaling Laws — From LLaVA to LLaVA-NeXT

The LLaVA family provides a clean case study in how scaling different axes — model size, data, and resolution — affects multimodal performance.

### Model Size Scaling

Increasing the LLM backbone from 7B to 13B parameters improves all benchmarks consistently. The VQAv2 score goes from 78.5% (7B) to 80.0% (13B). The improvement is most pronounced on complex reasoning tasks — MM-Vet improves from 31.1 to 36.1, a 16% relative gain.

Scaling further to 34B (using Yi-34B) pushes MM-Vet to 51.4. The pattern is clear: larger LLMs bring stronger reasoning capability to the multimodal setting, because they have a deeper understanding of language and world knowledge.

### Data Scaling

LLaVA-1.0 used 158K instruction examples. LLaVA-1.5 scaled to 665K by mixing GPT-4-generated data with academic VQA datasets. This 4x data increase improved performance across all benchmarks, with the largest gains on tasks that require diverse visual skills.

### Resolution Scaling — The LLaVA-NeXT Revolution

The biggest single improvement came from increasing image resolution. LLaVA-NeXT introduced the "AnyRes" approach: instead of resizing all images to a fixed size, it splits high-resolution images into multiple tiles, encodes each tile separately, and concatenates the visual tokens.

The number of visual tokens scales with resolution:

$$N_{\text{tokens}} = \left\lceil \frac{H}{P} \right\rceil \times \left\lceil \frac{W}{P} \right\rceil \times k$$

Here, $H$ and $W$ are the image height and width, $P$ is the patch size (14 for ViT-L/14), and $k$ is the number of tiles.

Let us plug in numbers. For a standard 336x336 image with patch size 14 and $k = 1$ tile:

$$N_{\text{tokens}} = \left\lceil \frac{336}{14} \right\rceil \times \left\lceil \frac{336}{14} \right\rceil \times 1 = 24 \times 24 \times 1 = 576 \text{ tokens}$$

For a 672x672 image split into $k = 4$ tiles of 336x336 each, plus a 336x336 thumbnail:

$$N_{\text{tokens}} = 576 \times 4 + 576 = 2880 \text{ tokens}$$

The model now has 5x more visual tokens to work with. This dramatically improves performance on detail-sensitive tasks. TextVQA jumps from 61.3% to 67.1%. MM-Vet goes from 36.1 to 43.9.

The trade-off is compute: more tokens means more memory and slower inference. But for tasks that require fine visual detail, the improvement is well worth it.


![Scaling model size, data, and resolution each improve different capabilities.](figures_v2/figure_9.png)
*Scaling model size, data, and resolution each improve different capabilities.*


---

## Practical Recipe — Instruction-Tune Your Own VLM

Let us put everything together into a concrete recipe for someone who wants to instruction-tune their own vision-language model.

### Step 1: Choose Your Base Models

You need a vision encoder and an LLM. The standard choices are:

- **Vision encoder:** CLIP ViT-L/14-336 (widely available, well-tested)
- **LLM:** LLaMA-2-7B or 13B (good balance of capability and cost), Vicuna (fine-tuned LLaMA with better instruction following), or Mistral-7B

### Step 2: Build the Projector

A two-layer MLP with GELU activation. Match the hidden dimension to the LLM's dimension:

```python
import torch.nn as nn

projector = nn.Sequential(
    nn.Linear(1024, 4096),   # CLIP dim -> LLM dim
    nn.GELU(),
    nn.Linear(4096, 4096),   # LLM dim -> LLM dim
)
# Total: ~8.4M parameters
```

### Step 3: Stage 1 — Alignment

Train the projector on image-caption data with the ViT and LLM frozen:

```python
# Stage 1 training configuration
config_stage1 = {
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 1,
    "optimizer": "AdamW",
    "weight_decay": 0,
    "warmup_ratio": 0.03,
    "lr_schedule": "cosine",
    "trainable": ["projector"],      # Only projector updates
    "frozen": ["vision_encoder", "llm"],
    "data": "558K image-caption pairs",
}
# Approximate time: 5-6 hours on 8x A100
# Approximate time: ~24 hours on 1x A100
```

### Step 4: Prepare Instruction Data

You have several options:

1. **Use existing datasets:** LLaVA-Instruct-150K is publicly available
2. **Generate with GPT-4:** Feed image descriptions to GPT-4 with appropriate prompts
3. **Mix academic VQA data:** Combine VQAv2, GQA, TextVQA, OCR-VQA training sets with GPT-4 data

For best results, mix all three sources.

### Step 5: Stage 2 — Instruction Tuning

Train the projector and LLM together on instruction data:

```python
# Stage 2 training configuration
config_stage2 = {
    "learning_rate": 2e-5,           # 50x smaller than Stage 1!
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "AdamW",
    "weight_decay": 0,
    "warmup_ratio": 0.03,
    "lr_schedule": "cosine",
    "trainable": ["projector", "llm"],  # Both update
    "frozen": ["vision_encoder"],
    "data": "665K instruction examples",
}
# Approximate time: 18-20 hours on 8x A100
# Approximate time: ~80 hours on 1x A100
```

### Step 6: Evaluate

Run your model on standard benchmarks and compare to published baselines. Key benchmarks: VQAv2, GQA, TextVQA, MM-Vet, POPE (for hallucination).

### Common Mistakes to Avoid

1. **Using the same learning rate for both stages.** The LLM needs a much smaller learning rate in Stage 2.
2. **Skipping Stage 1.** The projector must be aligned before the LLM adapts.
3. **Training for too many epochs.** One epoch per stage is sufficient — more leads to overfitting on the relatively small instruction datasets.
4. **Ignoring data quality.** 100K high-quality instruction examples outperform 500K noisy ones.
5. **Using low resolution.** If your task involves reading text or fine details, use at least 336x336.


![The complete VLM instruction tuning workflow.](figures_v2/figure_10.png)
*The complete VLM instruction tuning workflow.*


---

## Key Takeaways

Let us summarize the essential insights from this deep dive into multimodal instruction tuning:

1. **The training recipe matters more than the architecture.** The LLaVA architecture is simple — a projection layer between a ViT and an LLM. What makes it work is the two-stage training procedure with carefully tuned hyperparameters.

2. **Two-stage training is essential.** Stage 1 aligns the visual and language spaces using captioning data with a high learning rate. Stage 2 teaches instruction following with a 50x smaller learning rate. Skipping Stage 1 drops performance by 6-7 points.

3. **Data quality trumps quantity.** GPT-4-generated instruction data provides reasoning depth that simple captioning data cannot. Mixing academic VQA datasets with GPT-4 data gives the best results.

4. **Know your benchmarks.** VQAv2 tests general recognition, GQA tests compositional reasoning, TextVQA tests OCR, and MM-Vet tests integrated skills. Each tells you something different about your model.

5. **Know your failure modes.** Hallucination, spatial reasoning errors, counting mistakes, and OCR limitations are well-documented. Higher resolution and better data help, but these problems are far from solved.

6. **Scaling helps.** Larger LLMs improve reasoning, more data improves diversity, and higher resolution improves fine-grained understanding. Each axis benefits different capabilities.

That's it!

Here is the link to the original LLaVA paper: [Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)

LLaVA-1.5 paper: [Improved Baselines with Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2310.03744)

LLaVA-NeXT: [LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge (Liu et al., 2024)](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

For evaluation benchmarks: [VQAv2](https://visualqa.org/), [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html), [TextVQA](https://textvqa.org/), [MM-Vet](https://github.com/yuweihao/MM-Vet)
