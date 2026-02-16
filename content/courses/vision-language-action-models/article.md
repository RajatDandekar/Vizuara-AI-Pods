# Vision Language Action Models (VLAs)

*How a single AI model can see, understand language, and control a robot — all at once*

Vizuara AI

---

## The Dream: One Brain for Seeing, Understanding, and Acting

Let us start with a simple example. Imagine you walk into your kitchen and say to a robot: "Pick up the red cup and place it next to the plate."

For you, this is trivially easy. Your eyes see the scene, your brain understands the instruction, and your hands execute the motion — all in one fluid, coordinated effort. But for a robot, this single sentence requires three incredibly difficult capabilities:

1. **Vision** — The robot needs to look at the scene through its camera and understand what it sees. Where is the red cup? Where is the plate? What obstacles are in the way?

2. **Language** — The robot needs to parse your instruction. What does "pick up" mean in terms of motor commands? What does "next to" mean spatially?

3. **Action** — The robot needs to translate all of this understanding into precise motor commands — joint angles, gripper forces, and movement trajectories.


![A robot arm in a kitchen scene with a red cup and plate on a table. Three arrows pointing to the robot's camera (labeled "Vision"), a speech bubble (labeled "Language"), and the robot's gripper (labeled "Action"). The arrows converge into a single brain icon labeled "VLA" in the center.](figures/figure_1.png)
*A robot arm in a kitchen scene with a red cup and plate on a table. Three arrows pointing to the robot's camera (labeled "Vision"), a speech bubble (labeled "Language"), and the robot's gripper (labeled "Action"). The arrows converge into a single brain icon labeled "VLA" in the center.*


For decades, roboticists built separate systems for each of these. A computer vision module to detect objects. A natural language processing module to parse commands. A motion planning module to generate trajectories. Each module was developed independently, and then they were stitched together in a fragile pipeline.

**But what if we could build a single model — one neural network — that does all three at once?**

This is exactly the idea behind **Vision-Language-Action Models**, or **VLAs**. And they are rapidly becoming the most exciting development in robot learning.

---

## The Old Way: Pipelines That Break

Let us understand why the traditional pipeline approach was so problematic.

In the classical robotics stack, you would have something like this:

1. A **perception module** (say, a YOLO object detector) that identifies objects in the scene
2. A **language parser** that converts the instruction into a symbolic plan
3. A **task planner** that sequences the sub-goals
4. A **motion planner** (like RRT or trajectory optimization) that generates the path
5. A **controller** that executes the path on the robot's joints


![A pipeline diagram showing five sequential boxes: "Object Detector" → "Language Parser" → "Task Planner" → "Motion Planner" → "Controller". Red arrows between each box are labeled "Error propagation". Below it, a single box labeled "VLA Model" with inputs "Image + Text" on the left and output "Robot Actions" on the right, labeled "End-to-end".](figures/figure_10.png)
*A pipeline diagram showing five sequential boxes: "Object Detector" → "Language Parser" → "Task Planner" → "Motion Planner" → "Controller". Red arrows between each box are labeled "Error propagation". Below it, a single box labeled "VLA Model" with inputs "Image + Text" on the left and output "Robot Actions" on the right, labeled "End-to-end".*


Now the question is: what goes wrong with this pipeline?

**Everything.** Each module introduces errors, and these errors accumulate. If the object detector misidentifies the red cup, everything downstream fails. If the language parser cannot handle a slightly unusual phrasing, the task planner receives garbage. If the motion planner assumes a rigid object but the cup is slightly deformable, the grasp fails.

This is what researchers call the **error propagation problem**. Each stage in the pipeline can only be as good as the stage before it.

But there is a deeper problem. These modules do not share representations. The vision module does not know what the language module needs. The planner does not understand the visual scene. There is no shared understanding of the world.

This brings us to the key insight that launched the VLA revolution.

---

## The Key Insight: Large Language Models Already Understand the World

Here is a surprising fact. Models like GPT-4 and PaLM have been trained on trillions of tokens from the internet. In the process of learning to predict the next word, they have absorbed an enormous amount of **world knowledge**:

- They know that cups go on tables
- They know that you pick things up by grasping them
- They know spatial relationships like "next to," "on top of," and "behind"
- They know that red cups are typically cylindrical and have handles

Now, Vision-Language Models (VLMs) like PaLM-E, LLaVA, and PaLI-X extend this by also understanding images. They can look at a picture and reason about it in natural language.

**The VLA insight is simple but profound:** If a VLM already understands vision and language, why not teach it to also output robot actions?

Instead of building three separate systems, we take a single pre-trained VLM and fine-tune it to additionally produce motor commands. The model sees an image, reads an instruction, and directly outputs the sequence of actions the robot should take.


![A diagram showing the evolution from separate models to VLAs. Left side: three separate boxes labeled "Vision Model (ViT)", "Language Model (LLM)", and "Action Policy (MLP)" connected with arrows. Right side: a single large box labeled "Vision-Language-Action Model (VLA)" with image, text, and proprioception inputs on the left, and action outputs on the right. An arrow labeled "Evolution" connects the two sides.](figures/figure_11.png)
*A diagram showing the evolution from separate models to VLAs. Left side: three separate boxes labeled "Vision Model (ViT)", "Language Model (LLM)", and "Action Policy (MLP)" connected with arrows. Right side: a single large box labeled "Vision-Language-Action Model (VLA)" with image, text, and proprioception inputs on the left, and action outputs on the right. An arrow labeled "Evolution" connects the two sides.*


This is exactly what we want. One model. One set of weights. One shared understanding of the world.

---

## How VLAs Work: The Architecture

Let us now look at the architecture of a VLA in detail. While specific models differ, they all share a common blueprint with three components:

### 1. Vision Encoder

The vision encoder converts camera images into a sequence of visual tokens that the language model can process. Most VLAs use a **Vision Transformer (ViT)** variant:

- **SigLIP** — Used in OpenVLA and PaLI-based models. Trained with a sigmoid loss for image-text matching.
- **DINOv2** — A self-supervised ViT that produces rich spatial features without needing text supervision.

The image is divided into patches (typically 16×16 pixels), and each patch is converted into a token embedding. A 224×224 image becomes a sequence of 196 visual tokens.


![Vision encoder pipeline diagram. An input image of a robot workspace is divided into a 14x14 grid of patches. Each patch passes through a ViT encoder to produce 196 visual token embeddings, shown as a sequence of colored vectors.](figures/figure_2.png)
*Vision encoder pipeline diagram. An input image of a robot workspace is divided into a 14x14 grid of patches. Each patch passes through a ViT encoder to produce 196 visual token embeddings, shown as a sequence of colored vectors.*


### 2. Language Model Backbone

This is the "brain" of the VLA — a large pre-trained language model that fuses the visual tokens with the language instruction tokens. The visual tokens are simply prepended to the text tokens, and the entire sequence is processed by the transformer.

Popular backbones include:
- **PaLM-E** (562B parameters) — Used in RT-2
- **PaLI-X** (55B parameters) — Also used in RT-2
- **Llama 2** (7B parameters) — Used in OpenVLA
- **Gemma** (2B parameters) — Used in PaliGemma / Pi-0

The language model processes the interleaved visual and text tokens and produces output tokens that represent robot actions.


![Token sequence diagram showing how visual tokens (blue squares) and text tokens (green squares) are concatenated into a single sequence and fed through a Transformer. The output tokens (red squares) represent discretized robot actions. Labels show "Image tokens [v1, v2, ..., v196]" + "Text tokens [pick, up, the, red, cup]" → "Transformer" → "Action tokens [a1, a2, ..., a7]".](figures/figure_3.png)
*Token sequence diagram showing how visual tokens (blue squares) and text tokens (green squares) are concatenated into a single sequence and fed through a Transformer. The output tokens (red squares) represent discretized robot actions. Labels show "Image tokens [v1, v2, ..., v196]" + "Text tokens [pick, up, the, red, cup]" → "Transformer" → "Action tokens [a1, a2, ..., a7]".*


### 3. Action Head: From Tokens to Motor Commands

This is where VLAs diverge in their approach. The fundamental challenge is: **how do you represent continuous robot actions in a format that a language model can produce?**

There are three main approaches:

**Approach 1: Action Tokenization (RT-2 style)**

Discretize each action dimension into bins. For example, if a robot arm has 7 degrees of freedom, and each is discretized into 256 bins, then an action is represented as 7 integers. These integers are mapped to tokens in the language model's vocabulary.


$$
a_{\mathrm{token}} = \mathrm{discretize}(a_{\mathrm{cont}}, n_{\mathrm{bins}}) = \left\lfloor \frac{a - a_{\min}}{a_{\max} - a_{\min}} \cdot (n_{\mathrm{bins}} - 1) \right\rfloor
$$

**Approach 2: Diffusion Action Head (Octo style)**

Instead of discretizing, use a diffusion model to generate continuous actions. The action head takes the transformer's output as conditioning and iteratively denoises a random sample into a precise action.

**Approach 3: Flow Matching (Pi-0 style)**

Similar to diffusion but uses straighter denoising trajectories, resulting in faster inference. We will cover this in detail in the next article on Pi-0.


![A comparison diagram showing three approaches side by side. Left: "Action Tokenization" with continuous action being binned into discrete tokens (like a histogram). Middle: "Diffusion Head" showing iterative denoising from noise to action. Right: "Flow Matching" showing a straighter path from noise to action. Each approach has pros/cons listed below.](figures/figure_4.png)
*A comparison diagram showing three approaches side by side. Left: "Action Tokenization" with continuous action being binned into discrete tokens (like a histogram). Middle: "Diffusion Head" showing iterative denoising from noise to action. Right: "Flow Matching" showing a straighter path from noise to action. Each approach has pros/cons listed below.*


---

## The Pioneers: RT-2, Octo, and OpenVLA

Now let us look at the three landmark VLA models that shaped this field.

### RT-2: The First True VLA (Google DeepMind, 2023)

RT-2 was the first model to demonstrate that a large Vision-Language Model could be directly used as a robot controller. The key innovation was breathtakingly simple: **represent robot actions as text tokens.**

RT-2 takes a pre-trained VLM (either PaLM-E at 12B parameters or PaLI-X at 55B parameters) and co-fine-tunes it on a mixture of:
- Original vision-language data (to retain general knowledge)
- Robot demonstration data (to learn motor control)

The robot's actions — 7 numbers representing arm position, rotation, and gripper state — are converted to text strings like `"1 128 91 241 5 101 127"` and appended to the training data as if they were language.


![RT-2 architecture diagram. Input: camera image + text instruction "Pick up the can". The image passes through a ViT encoder, the instruction is tokenized, both feed into PaLM-E/PaLI-X. Output: action tokens "1 128 91 241 5 101 127" representing the 7-DOF robot action. Arrows show the data flow.](figures/figure_5.png)
*RT-2 architecture diagram. Input: camera image + text instruction "Pick up the can". The image passes through a ViT encoder, the instruction is tokenized, both feed into PaLM-E/PaLI-X. Output: action tokens "1 128 91 241 5 101 127" representing the 7-DOF robot action. Arrows show the data flow.*


What made RT-2 truly remarkable were its **emergent capabilities**. Because the VLM backbone had been trained on internet-scale data, RT-2 could:

- **Reason about unseen objects:** "Pick up the object that is the most unhealthy" → picks up chips over fruit
- **Understand symbols:** "Move the can to the Taylor Swift poster" → understands pop culture references
- **Follow spatial instructions:** "Place the object at the top-left corner" → understands spatial layout

None of these capabilities were explicitly trained. They emerged from the VLM's pre-existing world knowledge. This is exactly what we want — a robot that can leverage the vast knowledge encoded in internet-scale pre-training.

### Octo: The Open-Source Generalist (UC Berkeley, 2024)

Octo was designed with a different philosophy. While RT-2 used massive closed-source VLMs, Octo aimed to be an **open-source, modular generalist policy** that anyone could fine-tune.

Key design choices:
- **Transformer backbone** with a readout token mechanism
- **Diffusion action head** for continuous action generation
- Trained on the **Open X-Embodiment dataset** (800k+ episodes from 22 robot types)
- Designed to be fine-tuned on your own robot with as few as 50 demonstrations


![Octo architecture diagram. Multiple input modalities (image observations, language instruction, proprioception) feed into a Transformer encoder. A special "readout" token attends to all input tokens. The readout feeds into a diffusion action head that generates continuous actions through iterative denoising.](figures/figure_6.png)
*Octo architecture diagram. Multiple input modalities (image observations, language instruction, proprioception) feed into a Transformer encoder. A special "readout" token attends to all input tokens. The readout feeds into a diffusion action head that generates continuous actions through iterative denoising.*


### OpenVLA: Democratizing VLAs (Stanford/Berkeley, 2024)

OpenVLA brought the VLA paradigm to the open-source community with a fully open 7B parameter model. Built on:

- **Prismatic VLM** backbone combining SigLIP and DINOv2 vision encoders
- **Llama 2 (7B)** as the language backbone
- Trained on 970k episodes from the Open X-Embodiment dataset
- **Action tokenization** (like RT-2) with 256 bins per dimension


![OpenVLA architecture diagram. Two vision encoders (SigLIP and DINOv2) process the input image in parallel, producing complementary visual features. These are projected and concatenated with text tokens from the instruction. The combined sequence passes through Llama 2 (7B). Output: discretized action tokens.](figures/figure_7.png)
*OpenVLA architecture diagram. Two vision encoders (SigLIP and DINOv2) process the input image in parallel, producing complementary visual features. These are projected and concatenated with text tokens from the instruction. The combined sequence passes through Llama 2 (7B). Output: discretized action tokens.*


OpenVLA demonstrated that a 7B model — small enough to run on a single GPU — could match or exceed RT-2's performance on many benchmarks. It also showed that fine-tuning with **LoRA** (Low-Rank Adaptation) on just 10-20 demonstrations could adapt the model to a new robot or task.

---

## Training VLAs: The Two-Stage Recipe

How do you actually train a VLA? The recipe has become surprisingly standardized:

**Stage 1: Pre-train the VLM on internet-scale data**

The vision encoder and language model are trained on billions of image-text pairs from the internet. This gives the model rich visual understanding and language comprehension. Models like PaLI-X, LLaVA, and Prismatic are used off-the-shelf for this stage.

**Stage 2: Fine-tune on robot demonstration data**

The pre-trained VLM is then fine-tuned on robot data. The training objective is simple:



$$
\mathcal{L}_{\mathrm{VLA}} = -\sum_{t=1}^{T} \log p_{\theta}(a_t \mid o_t, l)
$$

where $$o_t$$ is the observation (image + proprioception) at time $$t$$, $$l$$ is the language instruction, and $$a_t$$ is the ground-truth action. This is just the standard cross-entropy loss for next-token prediction — the same loss used to train language models!


![Two-stage training pipeline. Stage 1 (left): "Internet Pre-training" showing web images and text flowing into a VLM with a "General Knowledge" label. Stage 2 (right): "Robot Fine-tuning" showing robot demonstrations (image + action pairs) flowing into the same VLM, now with additional action tokens in the vocabulary. An arrow connects Stage 1 to Stage 2 labeled "Transfer".](figures/figure_8.png)
*Two-stage training pipeline. Stage 1 (left): "Internet Pre-training" showing web images and text flowing into a VLM with a "General Knowledge" label. Stage 2 (right): "Robot Fine-tuning" showing robot demonstrations (image + action pairs) flowing into the same VLM, now with additional action tokens in the vocabulary. An arrow connects Stage 1 to Stage 2 labeled "Transfer".*


A critical trick is **co-fine-tuning**: mixing robot data with some of the original vision-language data during Stage 2. This prevents the model from forgetting its general knowledge while learning to output actions. RT-2 showed that this ratio matters — too much robot data causes catastrophic forgetting of VLM capabilities.

---

## The Open X-Embodiment Dataset

One of the most important developments for VLAs has been the creation of large-scale, multi-robot datasets. The **Open X-Embodiment (OXE)** dataset, a collaboration between 21 institutions, contains:

- **1 million+ robot episodes**
- **22 different robot embodiments** (from single arms to bimanual systems)
- **527 different skills**

This dataset has become the ImageNet of robot learning. Models trained on OXE show dramatic improvements in generalization compared to models trained on data from a single robot.


![Open X-Embodiment dataset overview. A central database icon labeled "OXE Dataset: 1M+ episodes" surrounded by icons of different robot types: Franka Panda, UR5, Kuka, ALOHA bimanual, Google Robot, Stretch, etc. Each robot icon shows the number of episodes contributed. Below: a bar chart showing skill diversity across manipulation, navigation, and locomotion.](figures/figure_9.png)
*Open X-Embodiment dataset overview. A central database icon labeled "OXE Dataset: 1M+ episodes" surrounded by icons of different robot types: Franka Panda, UR5, Kuka, ALOHA bimanual, Google Robot, Stretch, etc. Each robot icon shows the number of episodes contributed. Below: a bar chart showing skill diversity across manipulation, navigation, and locomotion.*


The key finding: **positive transfer across embodiments.** A model trained on data from Robot A performs better on Robot B than if it had only been trained on Robot B's data alone. The shared visual-semantic representations learned from diverse robots generalize across morphologies.

---

## Practical Implementation: Running a VLA

Let us look at how you would actually use a VLA in practice. Here is a simplified example of running inference with OpenVLA:

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# Load the pre-trained OpenVLA model
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b")

# Given a camera observation and language instruction
image = get_camera_image()           # RGB image from robot camera
instruction = "Pick up the red cup"  # Natural language command

# Process inputs and generate action
inputs = processor(images=image, text=instruction, return_tensors="pt")
action_tokens = model.generate(**inputs, max_new_tokens=7)

# Decode action tokens back to continuous values
action = processor.decode_actions(action_tokens)
# action = [x, y, z, roll, pitch, yaw, gripper]

# Send to robot controller
robot.execute(action)
```

Let us understand this code step by step. First, we load the pre-trained OpenVLA model — all 7 billion parameters, stored in bfloat16 to fit in GPU memory. The processor handles tokenization for both images and text. We then pass the camera image and language instruction through the model, which generates 7 action tokens. These tokens are decoded back into continuous action values and sent to the robot.

This is the beauty of VLAs — the same API you use for chatting with a language model is now used to control a robot.

---

## What Can VLAs Do Today?

The results from VLAs have been truly impressive:

- **RT-2** achieved 62% success on novel semantic reasoning tasks (vs 32% for RT-1)
- **Octo** showed positive cross-embodiment transfer across 9 robot platforms
- **OpenVLA** matched RT-2 performance with a 7B model (vs 55B for PaLI-X)
- **Pi-0** demonstrated the first successful generalist policy for complex dexterous tasks like folding laundry

But the most exciting aspect is the **emergent capabilities**. VLAs can understand instructions that were never in the robot training data. They can reason about objects, materials, spatial relationships, and even cultural references — all because they inherited this knowledge from internet pre-training.

---

## Limitations and the Road Ahead

VLAs are powerful, but they are far from perfect. Let us be honest about the current limitations:

1. **Inference speed** — A 55B parameter model cannot run at the 50Hz control frequency many robots need. Smaller models (7B) and action chunking help, but latency remains a challenge.

2. **Data hunger** — While VLAs need less robot data than training from scratch, they still require thousands of demonstrations. Collecting robot data is expensive.

3. **Safety** — A model that generalizes to unseen situations might also generalize in unexpected, dangerous ways. There are no guarantees.

4. **Long-horizon tasks** — VLAs predict one action (or one action chunk) at a time. Multi-step tasks requiring planning over minutes or hours remain difficult.

5. **Dexterous manipulation** — Most VLAs work with simple parallel-jaw grippers. Multi-fingered dexterous manipulation requires much richer action spaces.

Now the question is: will scaling solve these problems? Will bigger models trained on more data eventually give us truly general-purpose robots?

The evidence from language models suggests yes — but robotics has unique challenges that language does not. You cannot download a trillion robot demonstrations from the internet. The physical world does not forgive hallucinations.

---

## What Comes Next?

The VLA paradigm has opened a Pandora's box of possibilities. In the next few articles, we will explore:

- **Pi-0 from Physical Intelligence** — How flow matching enables dexterous manipulation
- **RT-1 and RT-2 in depth** — The full story of Google's Robotics Transformers
- **World Models** — Can robots learn to imagine the future before acting?

The journey from separate perception-planning-control pipelines to unified VLAs mirrors the journey from separate NLP modules to end-to-end language models. And just as GPT transformed language AI, VLAs may transform robotics.

We are truly at the beginning of something remarkable. See you next time!

---

*References:*
- *Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (2023)*
- *Octo Model Team, "Octo: An Open-Source Generalist Robot Policy" (2024)*
- *Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model" (2024)*
- *Black et al., "Pi0: A Vision-Language-Action Flow Model for General Robot Control" (2024)*