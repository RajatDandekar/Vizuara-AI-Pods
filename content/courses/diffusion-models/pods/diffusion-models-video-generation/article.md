# From Still to Motion: How Diffusion Models Learned to Generate Videos

*Extending image diffusion to the temporal dimension — from 2D noise to coherent video, one frame at a time.*

---

## The Flipbook Intuition

Let us start with a simple example. Imagine you are flipping through a flipbook — each page has a slightly different drawing, and when you flip fast enough, the drawings come to life as a smooth animation.


![A flipbook is just a low-tech video — both are sequences of coherent images.](figures/figure_1.png)
*A flipbook is just a low-tech video — both are sequences of coherent images.*


This is exactly what a video is — a sequence of images (called **frames**) played back rapidly enough that our eyes perceive smooth motion. A typical video plays at 24 to 30 frames per second.

Now, we have already seen in our previous articles how diffusion models can generate stunning images from pure noise. The forward process gradually adds Gaussian noise to a clean image until it becomes indistinguishable from random noise. The reverse process then learns to denoise, step by step, to produce a brand-new image.

Refer to our article on Diffusion Models here: https://vizuara.substack.com/p/what-exactly-are-diffusion-models

The main question is: **Can we extend this same idea to generate entire videos?**

Instead of denoising a single 2D image, can we denoise an entire sequence of frames — a 3D block of data — such that the result is not just a collection of pretty pictures, but a coherent, temporally consistent video?

This is exactly what we will explore in this article.

---

## What Makes Video Different from Images?

Let us understand why generating video is fundamentally harder than generating images.

A single image lives in a 2D spatial grid. If the image is 256 pixels wide and 256 pixels tall with 3 color channels (RGB), it contains 256 × 256 × 3 = **196,608 values**.

A video adds a new dimension: **time**. A 4-second video at 8 frames per second, with the same 256 × 256 resolution, has 32 frames. That means 32 × 256 × 256 × 3 = **6,291,456 values** — roughly **32 times more data** than a single image.


![Video adds a time dimension, resulting in 32x more data than a single image.](figures/figure_2.png)
*Video adds a time dimension, resulting in 32x more data than a single image.*


But the challenge is not just about scale. There are three core problems:

**1. Temporal Coherence**

Objects must move smoothly between frames. If a cat is sitting on a sofa in frame 1, it cannot suddenly teleport to the kitchen in frame 2. Lighting must stay consistent. Colors should not flicker. Physics must be respected.

Think of it this way. If I asked you to draw a cat sitting on a sofa, you might produce a beautiful image. But if I asked you to draw 30 images of that cat jumping off the sofa, each frame must smoothly connect to the next. The cat cannot teleport, change color, or suddenly grow a third ear between frames.

**2. Computational Cost**

As we just saw, videos have 10 to 100 times more data than images. This means our neural networks need significantly more memory, compute, and training time. Training a video diffusion model from scratch can require thousands of GPUs running for weeks.

**3. Training Data**

Curating high-quality video datasets is much harder than curating image datasets. Videos need to be properly trimmed, filtered for quality, and annotated with text descriptions. While we have billions of captioned images on the internet, high-quality captioned video datasets are much scarcer.

---

## The Naive Approach — Why Frame-by-Frame Fails

A thought might come to your mind: "Why not just generate each frame independently using an image diffusion model?"

This sounds reasonable. We have excellent image diffusion models — why not just run them 30 times and stitch the results together?

The problem is that each frame would be independently sampled from the noise distribution. There is no mechanism to enforce consistency between frames. Each frame is essentially a separate roll of the dice.

Imagine asking 30 different artists to each draw one frame of a cat jumping off a sofa, without showing them what the other artists drew. You might get 30 beautiful individual frames, but when you play them back as a video, the result would be chaos — the cat would change shape, color, and position randomly from frame to frame.


![Independent frame generation fails at temporal coherence; joint video diffusion succeeds.](figures/figure_3.png)
*Independent frame generation fails at temporal coherence; joint video diffusion succeeds.*


This is exactly why we need architectures that understand time — models that can jointly generate all frames while maintaining coherence across them.

---

## Extending the Diffusion Framework to Video

Now let us see how we can mathematically extend the diffusion framework from images to video.

Recall from our earlier article that in image diffusion, the forward process adds noise to a clean image $x_0$ over $T$ timesteps:


$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t\, \mathbf{I})
$$


For video, the idea is beautifully simple. Instead of $x$ representing a single image, we let $\mathbf{v}$ represent an entire video clip — a 3D tensor with dimensions (number of frames) × (height) × (width) × (channels).

The forward process is exactly the same — we add Gaussian noise to the entire video:


$$
q(\mathbf{v}_t \mid \mathbf{v}_{t-1}) = \mathcal{N}(\mathbf{v}_t;\, \sqrt{1 - \beta_t}\, \mathbf{v}_{t-1},\, \beta_t\, \mathbf{I})
$$


Notice that this is exactly the same equation we saw for images. The only difference is that $\mathbf{v}$ now represents an entire video clip rather than a single image. The noise is added independently to every pixel in every frame.

Let us plug in some simple numbers to see how this works. Suppose we have a tiny video with just 2 frames, each of size 2 × 2 pixels (single channel for simplicity). Let our clean video be:

Frame 1: [[0.8, 0.6], [0.4, 0.9]]
Frame 2: [[0.7, 0.5], [0.5, 0.8]]

At diffusion timestep $t$ with $\beta_t = 0.1$, the mean scaling factor is $\sqrt{1 - 0.1} = \sqrt{0.9} \approx 0.949$. So the mean of the noisy video becomes:

Frame 1 mean: [[0.759, 0.569], [0.380, 0.854]]
Frame 2 mean: [[0.664, 0.475], [0.475, 0.759]]

Each pixel then gets an independent Gaussian noise sample added with standard deviation $\sqrt{0.1} \approx 0.316$. After many such steps, the entire video becomes pure noise — just as we saw with images.

The reverse process objective is also the same — we train a neural network $\epsilon_\theta$ to predict the noise that was added:


$$
\mathcal{L} = \mathbb{E}_{\mathbf{v}_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(\mathbf{v}_t, t)\|^2\right]
$$


The difference is that our neural network must now understand both **spatial patterns** (what objects look like within a frame) and **temporal patterns** (how objects move across frames).


![The same forward/reverse diffusion math applies, but now v is an entire video tensor.](figures/figure_4.png)
*The same forward/reverse diffusion math applies, but now v is an entire video tensor.*


So the math is essentially unchanged. The real challenge — and the real innovation — lies in the **architecture** of $\epsilon_\theta$. How do we design a neural network that can jointly reason about space and time?

---

## The Architecture — How Neural Networks Learn Space and Time

This brings us to the core architectural innovation behind video diffusion models: **factorized spatial-temporal attention**.

Let us understand this with an analogy. Imagine an editor reviewing a movie. First, they examine each frame individually to check the visual quality — are the colors right? Is the composition good? Is the lighting correct? This is **spatial processing**. Then, they play the frames in sequence to check if the motion is smooth — does the actor move naturally? Are there any jumps or glitches? This is **temporal processing**.

Our neural network does exactly the same thing, in alternation.

### Spatial Layers — What We Already Know

Image diffusion models use a U-Net architecture with 2D convolutional layers and spatial self-attention. These layers operate on each frame independently. They handle texture, shape, color, and spatial composition — everything about what a single frame looks like.

If you have seen the architecture of Stable Diffusion, you know these layers well. They are the workhorses that make individual frames look beautiful.

### Temporal Layers — The New Ingredient

The key innovation is to **insert temporal layers** between the spatial layers. These are:

- **Temporal convolutions:** 1D convolutions that operate along the time axis. For each spatial position (pixel), they look at how that pixel's value changes across frames.
- **Temporal self-attention:** Each pixel in frame $t$ attends to the same spatial location across all other frames. This allows the model to learn temporal patterns — how things move, how lighting evolves, how objects enter and exit the scene.

The temporal self-attention works as follows:


$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k} }\right) \cdot V
$$


Here, $Q$, $K$, and $V$ are projected from features at the **same spatial position** across all $T$ frames. This means that for a pixel at position $(i, j)$, the query comes from frame $t$, and the keys and values come from all frames $1, 2, \ldots, T$.

Let us plug in some simple numbers. Suppose we have 3 frames, and the feature value at position $(i, j)$ in each frame is:

Frame 1: $q_1 = 1.0$, Frame 2: $q_2 = 1.5$, Frame 3: $q_3 = 0.8$

With $d_k = 1$ for simplicity, the attention scores for frame 2 attending to all frames are:

$\text{scores} = [q_2 \cdot k_1,\; q_2 \cdot k_2,\; q_2 \cdot k_3] = [1.5,\; 2.25,\; 1.2]$

After softmax: $[0.27,\; 0.57,\; 0.16]$

This tells us that frame 2 attends most strongly to itself (0.57) and to frame 1 (0.27), while attending less to frame 3 (0.16). This is exactly what we want — neighboring frames should influence each other more strongly.


![Spatial layers (blue) handle each frame independently; temporal layers (orange) enforce consistency across frames.](figures/figure_5.png)
*Spatial layers (blue) handle each frame independently; temporal layers (orange) enforce consistency across frames.*


### Factorized Attention — Why It Works

Now the question is: why not just do full 3D attention over the entire space-time volume?

The answer is computational cost. Full 3D self-attention over all $(T \times H \times W)$ positions has complexity $O((T \cdot H \cdot W)^2)$. For a 16-frame video at 32 × 32 spatial resolution, this means attending over $16 \times 32 \times 32 = 16{,}384$ positions, giving $16{,}384^2 \approx 268$ million attention computations.

Factorized attention breaks this into two cheaper steps:
1. **Spatial attention** within each frame: $O(T \cdot (H \cdot W)^2)$ — attend over $32 \times 32 = 1{,}024$ positions, done 16 times = $16 \times 1{,}024^2 \approx 16.8$ million
2. **Temporal attention** at each spatial position: $O(H \cdot W \cdot T^2)$ — attend over 16 frames, done $1{,}024$ times = $1{,}024 \times 16^2 \approx 262{,}000$

Total: roughly **17 million** operations instead of **268 million** — about **16 times cheaper**. For higher resolutions, the savings are even more dramatic.

---

## Training Strategies — Joint and Fine-tuned

Now let us understand how these video diffusion models are actually trained. There are two main approaches.

### Approach 1: Training from Scratch

The most straightforward approach is to train the entire video diffusion model — both spatial and temporal layers — from scratch on video data. This allows the model to learn video-native representations from the ground up.

The downside is that this is extremely expensive. Video datasets are smaller and noisier than image datasets, and training a model with billions of parameters on video data requires enormous computational resources.

### Approach 2: Fine-tuning a Pretrained Image Model

The more practical and widely used approach is to leverage a pretrained image diffusion model and **add temporal layers** on top of it. This is the approach used by Stable Video Diffusion, Imagen Video, and many other successful models.

The process works as follows:

1. Start with a powerful pretrained image diffusion model (e.g., Stable Diffusion)
2. Keep the spatial layers frozen (or partially frozen) — these already know how to generate beautiful images
3. Insert randomly initialized temporal attention and temporal convolution layers between the spatial layers
4. Train only the temporal layers on video data

This is like taking an artist who already knows how to paint beautiful still images, and teaching them animation. They do not need to relearn how to draw — they only need to learn how things move.

Many models also use a **joint image-video training objective** to prevent the spatial capabilities from degrading:


$$
\mathcal{L}_{\text{joint} } = \lambda_{\text{img} } \cdot \mathbb{E}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right] + \lambda_{\text{vid} } \cdot \mathbb{E}\!\left[\|\epsilon - \epsilon_\theta(\mathbf{v}_t, t)\|^2\right]
$$


Here, the first term trains on single images and the second term trains on video clips. The weights $\lambda_{\text{img}}$ and $\lambda_{\text{vid}}$ control the balance.

Let us plug in some simple numbers. Suppose $\lambda_{\text{img}} = 0.3$ and $\lambda_{\text{vid}} = 0.7$, the image loss is 2.5, and the video loss is 1.8. Then:

$\mathcal{L}_{\text{joint}} = 0.3 \times 2.5 + 0.7 \times 1.8 = 0.75 + 1.26 = 2.01$

The video loss is weighted more heavily because learning temporal coherence is the primary objective. But keeping the image loss ensures the model does not forget how to generate high-quality individual frames.


![Pretrained spatial layers are frozen (blue); only new temporal layers (orange) are trained on video data.](figures/figure_6.png)
*Pretrained spatial layers are frozen (blue); only new temporal layers (orange) are trained on video data.*


---

## Latent Video Diffusion — Making It Practical

Now let us address the elephant in the room: **computational cost**.

Running diffusion directly in pixel space for video is enormously expensive. Remember, a 16-frame video at 256 × 256 resolution has over 3 million pixel values. Running hundreds of denoising steps on this tensor is simply not practical.

The solution? **Latent Video Diffusion** — run the diffusion process in a compressed latent space instead.

Remember the variational autoencoders from our earlier article? They make a comeback here. Instead of running diffusion on the full-resolution video, we first compress the video into a much smaller representation using an encoder, run all the diffusion in this compact latent space, and then decode back to pixels at the very end.

The encoder compresses the video:


$$
\mathbf{z} = \mathcal{E}(\mathbf{v}), \quad \mathbf{v} \in \mathbb{R}^{T \times H \times W \times 3}, \quad \mathbf{z} \in \mathbb{R}^{T \times h \times w \times c}, \quad h \ll H,\; w \ll W
$$


Let us plug in some real numbers. Suppose our video $\mathbf{v}$ has 16 frames at 256 × 256 resolution with 3 channels. That is $16 \times 256 \times 256 \times 3 = 3{,}145{,}728$ values.

After encoding with a spatial downsampling factor of 8 and a latent channel size of 4, we get $\mathbf{z}$ with dimensions $16 \times 32 \times 32 \times 4 = 65{,}536$ values.

This is a **48× compression**! Now our diffusion process operates on a tensor that is 48 times smaller, making training and inference dramatically faster.


![A 3D VAE compresses the video 48x; diffusion runs entirely in the compact latent space.](figures/figure_7.png)
*A 3D VAE compresses the video 48x; diffusion runs entirely in the compact latent space.*


Some models use a 2D VAE (compressing each frame independently) while more recent models use a **3D VAE** that compresses both spatially and temporally. A 3D VAE can also compress along the time axis — for example, reducing 16 frames to 4 latent frames — giving even higher compression ratios.

---

## Text-to-Video — Conditioning on Language

We have seen how to generate videos from noise. But how do we tell the model **what** to generate? This is where text conditioning comes in.

The mechanism is very similar to what you might have seen in text-to-image models like Stable Diffusion. We encode the text prompt into a vector representation, and then use **cross-attention** to let the video generation process "look at" the text at every denoising step.

The process works as follows:

1. **Text encoding:** The text prompt (e.g., "a cat jumping off a sofa") is passed through a text encoder such as CLIP or T5 to produce a sequence of text embeddings.

2. **Cross-attention:** In both the spatial and temporal attention layers of the U-Net, we add cross-attention blocks where the queries come from the video features and the keys/values come from the text embeddings:


$$
\text{CrossAttn}(Q_{\text{video} }, K_{\text{text} }, V_{\text{text} }) = \text{softmax}\!\left(\frac{Q_{\text{video} }\, K_{\text{text} }^T}{\sqrt{d_k} }\right) \cdot V_{\text{text} }
$$


Here, $Q_{\text{video}}$ is projected from the video features at each layer, while $K_{\text{text}}$ and $V_{\text{text}}$ are projected from the text encoder's output.

Let us trace through a simple example. Suppose the text "a dog running" is encoded into a sequence of 4 token embeddings (in practice, this is 77 tokens for CLIP). Each video feature at a spatial-temporal position computes attention scores over these 4 text tokens. If the video feature at a position depicting the dog's legs has high attention to the "running" token, the model knows to generate motion-appropriate features at that location. This is exactly what we want.

3. **Classifier-free guidance:** During training, the text conditioning is randomly dropped (replaced with an empty string) some percentage of the time. At inference, the model generates two predictions — one conditioned on the text and one unconditional — and the final prediction is extrapolated away from the unconditional one:

$\hat{\epsilon} = \epsilon_{\text{uncond}} + w \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$

where $w > 1$ is the guidance scale (typically 7 to 15). Higher values produce videos that more closely match the text, at the cost of some diversity.


![Text embeddings from CLIP/T5 are injected via cross-attention to guide video generation.](figures/figure_8.png)
*Text embeddings from CLIP/T5 are injected via cross-attention to guide video generation.*


---

## Landmark Models — A Brief Tour

Now let us look at the key models that have defined this field. Each one introduced important ideas that built upon the previous work.

**Video Diffusion Models (VDM) — Ho et al., 2022**

This was the first work to apply diffusion models directly to video generation. The authors introduced the factorized space-time U-Net architecture that we discussed earlier — spatial layers processing each frame independently, interleaved with temporal attention layers operating across frames. VDM demonstrated that the image diffusion framework could be naturally extended to video with minimal architectural changes.

**Imagen Video — Ho et al., 2022**

Building on VDM, Imagen Video introduced a **cascaded** approach: a base model generates a low-resolution, low-frame-rate video, and then a series of super-resolution models progressively upscale it in both space and time. This allowed generation at 1280 × 768 resolution — a significant leap. The key insight was that it is easier to first get the global motion and structure right at low resolution, and then fill in the fine details.

**Make-A-Video — Singer et al., 2022 (Meta)**

Make-A-Video introduced a clever trick: learn **spatial appearance** from text-image pairs (which are abundant) and learn **motion** from unlabeled video (which does not need text captions). This decoupling meant the model could leverage the massive text-image datasets available while only needing raw video for learning temporal dynamics.

**Stable Video Diffusion (SVD) — Blattmann et al., 2023**

SVD took the practical approach of fine-tuning the open-source Stable Diffusion image model for video. It demonstrated the power of the "pretrain on images, fine-tune for video" paradigm that we discussed earlier. SVD is primarily an **image-to-video** model — you give it a single image, and it generates a short video clip showing that image coming to life.

**Sora — OpenAI, 2024**

Sora represented a paradigm shift. Instead of using a U-Net architecture, it adopted a **Diffusion Transformer (DiT)** — replacing the convolutional U-Net entirely with a Transformer. Sora can generate up to 1-minute videos at high resolution, demonstrating that Transformer-based architectures scale better for video generation.


![Key milestones in video diffusion: from VDM's first U-Net to Sora's Transformer architecture.](figures/figure_9.png)
*Key milestones in video diffusion: from VDM's first U-Net to Sora's Transformer architecture.*


---

## The Diffusion Transformer (DiT) for Video

The shift from U-Net to Transformer is one of the most important recent trends in video generation. Let us understand why and how it works.

### Why Transformers?

U-Nets have served image and video diffusion extremely well. But they have a limitation: their architecture is relatively fixed. You choose the number of layers, channels, and attention heads, and that is your model. Scaling up means making the U-Net wider or deeper, which quickly hits diminishing returns.

Transformers, on the other hand, exhibit clean **scaling laws** — performance improves predictably as you increase the model size and training data. This was demonstrated by Peebles and Xie (2023) for image diffusion, and Sora extended it to video.

### Spacetime Patches

The key idea behind DiT for video is **spacetime patching**. Instead of processing the video through convolutional layers, we cut the video into small 3D cubes of space-time and treat each cube as a token — just like Vision Transformers (ViT) cut images into 2D patches.


$$
\text{Given } \mathbf{v} \in \mathbb{R}^{T \times H \times W \times 3},\; \text{create patches } p_i \in \mathbb{R}^{t_p \times h_p \times w_p \times 3},\; \text{project to tokens } z_i = \text{Linear}(\text{flatten}(p_i))
$$


Here, $t_p$, $h_p$, $w_p$ are the patch dimensions in time, height, and width respectively.

Let us plug in some numbers. Suppose we have a 16-frame video at 256 × 256 resolution, and we use patches of size $2 \times 16 \times 16$ (2 frames tall, 16 pixels wide, 16 pixels high). The number of tokens is:

$\frac{16}{2} \times \frac{256}{16} \times \frac{256}{16} = 8 \times 16 \times 16 = 2{,}048 \text{ tokens}$

Each token is a flattened vector of size $2 \times 16 \times 16 \times 3 = 1{,}536$ values, which is then linearly projected to the Transformer's hidden dimension (e.g., 1024).

Think of it as cutting a video into small cubes of space-time and then letting each cube talk to every other cube through self-attention. This is different from the factorized approach — here, spatial and temporal information are mixed from the very beginning.


![DiT cuts the video into 3D spacetime patches, tokenizes them, and processes all tokens through Transformer blocks.](figures/figure_10.png)
*DiT cuts the video into 3D spacetime patches, tokenizes them, and processes all tokens through Transformer blocks.*


The advantage of this approach is simplicity and scalability. There is no need to carefully design separate spatial and temporal layers — the Transformer's self-attention naturally captures both spatial and temporal relationships. And because Transformers scale predictably, we can simply make the model bigger to get better results.

---

## Practical Implementation

Enough theory — let us look at some practical code. We will use the Hugging Face `diffusers` library to generate a short video from a single image using Stable Video Diffusion.

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# Load the Stable Video Diffusion pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Load a conditioning image
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
)
image = image.resize((1024, 576))

# Generate the video
# num_frames: how many frames to generate
# decode_chunk_size: for memory efficiency
generator = torch.manual_seed(42)
frames = pipe(
    image,
    num_frames=25,
    decode_chunk_size=8,
    generator=generator
).frames[0]

# Save as video file
export_to_video(frames, "generated_video.mp4", fps=7)
print("Video saved!")
```

Let us understand this code in detail.

First, we load the Stable Video Diffusion pipeline using `from_pretrained`. The model `stable-video-diffusion-img2vid-xt` is the extended version that generates 25 frames (about 3-4 seconds at 7 fps). We use `float16` precision to reduce memory usage.

Next, we load and resize a conditioning image. This is the image that the model will "animate" — it serves as the first frame, and the model generates the subsequent frames showing plausible motion.

Then we call the pipeline with our image. The `num_frames` parameter controls how many frames to generate. The `decode_chunk_size` parameter controls how many latent frames are decoded to pixels at once — this is a memory optimization since decoding all frames simultaneously would require too much GPU memory.

Finally, we export the generated frames as an MP4 video file. Not bad, right?

---

## Open Challenges and What Lies Ahead

Video diffusion models have made remarkable progress, but several challenges remain.

**Video Length.** Most current models generate 2 to 4 seconds of video. Generating longer, coherent videos (minutes or hours) remains an open problem. Some approaches use autoregressive extension — generating a few seconds, then using the last frame as conditioning for the next chunk — but this often leads to quality degradation over time.

**Physics Consistency.** Current models often violate basic physics. Objects pass through each other, gravity works inconsistently, and reflections behave incorrectly. Teaching models to understand physical laws — rather than just statistical patterns — is an active area of research.

**Fine-grained Control.** Users want to control not just what appears in the video, but how things move — camera angles, character poses, object trajectories. Current text conditioning provides only coarse control. Techniques like ControlNet for video are being actively developed.

**Resolution and Quality.** There is a fundamental tradeoff between resolution, video length, and computational cost. Generating high-resolution, long videos in real-time is still far from practical for most users.

The field is evolving rapidly, and the line between video generation and world simulation is beginning to blur. Does this remind you of world models in reinforcement learning, where the agent learns an internal model of how the world works? We will explore this fascinating connection in a future article.

See you next time!

---

## References

- Ho et al., "Video Diffusion Models" (2022)
- Ho et al., "Imagen Video: High Definition Video Generation with Diffusion Models" (2022)
- Singer et al., "Make-A-Video: Text-to-Video Generation without Text-Video Data" (2022)
- Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets" (2023)
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)
- Brooks et al., "Video Generation Models as World Simulators" (Sora technical report, 2024)
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)