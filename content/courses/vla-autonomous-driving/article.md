# Vision-Language-Action Models for Autonomous Driving: From Pixels and Words to Steering Wheels

*How combining vision, language understanding, and action generation is reshaping autonomous driving — explained from scratch.*

---

Let us start with a simple example. Imagine you are sitting in the passenger seat, teaching your friend to drive in a new city.

You look out the window and point at things: "That is a school zone up ahead." You describe the situation: "The car in front is braking — I think there is a pedestrian crossing." And you give instructions: "Slow down and move to the right lane."

Your friend does something remarkable. They **see** the road through their eyes, **understand** your words, and **act** on them — adjusting the steering wheel, easing off the gas, signaling a lane change. All of this happens in one seamless loop: vision, language, action.

This is exactly what a **Vision-Language-Action (VLA)** model does for autonomous driving.

Now, you might be thinking: "We already have self-driving cars. Tesla, Waymo, Cruise — they have been working on this for years. What is new here?"

Great question. Traditional self-driving systems work like an assembly line. They break the driving task into separate modules:

1. **Perception:** Detect objects — cars, pedestrians, traffic lights
2. **Prediction:** Forecast where those objects will move
3. **Planning:** Decide what the car should do next
4. **Control:** Execute the plan — turn the wheel, apply the brakes

Each module is built and optimized independently. The perception team builds their detector, the prediction team builds their forecaster, and so on. This modular approach has been the standard for over a decade.

But here is the problem: **errors cascade.** If the perception module misclassifies a construction cone as a pedestrian, every downstream module makes decisions based on that mistake. The prediction module forecasts a "pedestrian" walking into the road, the planner slams the brakes, and the car stops in the middle of a highway. Not ideal.

The question that researchers began asking around 2023 was bold: **Can we replace this entire pipeline with a single model that sees, understands, and acts?**


![Traditional pipelines chain four separate modules; a VLA replaces them with one unified model.](figures/figure_1.png)
*Traditional pipelines chain four separate modules; a VLA replaces them with one unified model.*


This is precisely the promise of VLAs. Instead of four separate modules, you have **one model** that takes in camera images and a language description of the driving context, and directly outputs the trajectory the car should follow. Vision. Language. Action. All in one.

Let us now understand how this works — piece by piece.

---

## The Building Blocks: Vision, Language, and Action

A VLA model has three core components, each handling one modality. Let us look at each of them.

### The Vision Backbone — How the Model "Sees"

Think of the vision backbone as a **translator**. It takes the raw, noisy, pixel-level camera feed — millions of numbers representing colors — and converts it into a compact, meaningful description.

How does it do this? Most modern VLAs use a **Vision Transformer (ViT)**. The idea is beautifully simple:

1. Take the camera image and divide it into small patches (say, 16×16 pixels each)
2. Treat each patch as a "word" — just like how language models process words
3. Pass all these patch "words" through a transformer with self-attention
4. The output: a sequence of **feature tokens** — each one a compact vector summarizing what that part of the image contains

The key insight is that the model does not need to read every single pixel. After training, it learns to focus on what matters — pedestrians stepping off the curb, brake lights turning red, lane markings curving to the right. The rest gets compressed away.


![The vision encoder converts raw camera images into compact feature tokens via patch embedding and self-attention.](figures/figure_2.png)
*The vision encoder converts raw camera images into compact feature tokens via patch embedding and self-attention.*


For autonomous driving, there is one additional complication: you do not have just one camera. Most self-driving cars have **4 to 8 cameras** covering a 360-degree view. The vision encoder processes each camera independently (or sometimes jointly) and produces feature tokens for the entire surround view.

### The Language Backbone — Why Words Matter for Driving

Now, you might wonder: "Why does a self-driving car need to understand language? It is not having a conversation."

Language serves two critical purposes in a VLA:

**Purpose 1: Understanding commands.** Imagine telling your car: "Take the scenic route along the coast" or "Avoid the highway — there is construction." These are high-level instructions that a traditional perception module cannot process, but a language model can.

**Purpose 2: Reasoning about scenes.** This is the more powerful use. The language backbone allows the model to internally "think" about what it sees. When the model encounters a school zone, it does not just detect the sign — it **understands** what "school zone" means: children might dart into the road, speed limits are lower, extra caution is needed.

Pre-trained large language models like LLaMA or Gemma already have this world knowledge baked in from training on billions of words from the internet. They know what a "construction zone" is, what "yielding to an emergency vehicle" means, and why "black ice" is dangerous. We do not need to teach the model these concepts from scratch — we just need to connect the language backbone to the vision and action components.

Language, in this sense, is the **thinking layer** of the VLA. It converts raw visual perception into structured reasoning.

### The Action Head — Two Flavors

Now we come to the most interesting part: how does the model actually **drive?**

In autonomous driving, an "action" typically means one of two things:
- **Low-level controls:** A steering angle, an acceleration value, a braking force
- **Trajectory waypoints:** A sequence of future positions the car should follow — for example, 64 points over the next 6.4 seconds

The question is: how do we get a language model — which is designed to output **words** — to output **numbers** like steering angles or waypoint coordinates?

There are two main approaches being used today.

**Approach 1: Action Tokenization.** This is the clever trick. We take the continuous action space (steering angles ranging from -45° to +45°) and **discretize** it into bins, just like converting a continuous variable into categories. Each bin gets its own token, which is added to the model's vocabulary alongside regular words. Now the model can "say" a steering angle the same way it says a word.

The tokenization formula is straightforward:


$$a_{\text{token}} = \text{round}\left(\frac{a - a_{\min}}{a_{\max} - a_{\min}} \times (N_{\text{bins}} - 1)\right)$$

Let us plug in some simple numbers to see how this works. Suppose we want to tokenize a steering angle of $a = 15°$, with a range of $[-45°, 45°]$ and $N_{\text{bins}} = 256$:

$$a_{\text{token}} = \text{round}\left(\frac{15 - (-45)}{45 - (-45)} \times 255\right) = \text{round}\left(\frac{60}{90} \times 255\right) = \text{round}(170) = 170$$

So a steering angle of 15° gets mapped to token 170 out of 256 possible tokens. A straight-ahead steering of 0° would map to token 128 (the middle). This is exactly what we want — the model now treats driving actions as just another type of "word" in its vocabulary.

This approach is used by RT-2 and EMMA.

**Approach 2: Diffusion Action Decoder.** Instead of forcing the language model to output discrete tokens, we attach a separate neural network — a **diffusion decoder** — that takes the language model's internal representations and generates smooth, continuous trajectories. How does diffusion work? The core idea is simple: start with random noise and iteratively refine it into a clean output — like sculpting a trajectory out of marble, chipping away noise with each step. This is the same principle behind image generation models like DALL-E, except here the model generates sequences of waypoints instead of pixels. NVIDIA's Alpamayo uses this approach, and we will see why shortly.


![A VLA fuses vision, language, and action into one unified architecture.](figures/figure_3.png)
*A VLA fuses vision, language, and action into one unified architecture.*


---

## Two Paradigms: End-to-End vs. Dual-System VLAs

Before we dive into specific architectures, let us understand an important distinction that has emerged in the field. Not all VLAs are built the same way. There are two paradigms:

### End-to-End VLA — One Model Does Everything

In the end-to-end approach, a single unified model handles perception, reasoning, **and** action generation. Camera images and text go in, trajectory waypoints come out. There is no separate planner or controller — the VLA does it all.

The advantage? Simplicity and emergent reasoning. Because everything is trained end-to-end, the model can discover shortcuts and relationships that a modular system would miss. For instance, the model might learn that when it sees a specific type of road marking AND the language description mentions "highway exit," it should start decelerating — without anyone explicitly programming this rule.

The challenge? The model must think and act simultaneously, and if it makes a mistake, there is no safety net.

### Dual-System VLA — Slow Thinking + Fast Acting

Think of how a Formula 1 team works. The **strategist** on the pit wall analyzes data, considers tire degradation, weather forecasts, and competitor positions to formulate a race strategy. This is deep, slow thinking. Meanwhile, the **driver** in the cockpit executes split-second maneuvers — braking at 300 km/h, overtaking on a narrow straight — using reflexes and instinct. This is fast acting.

A Dual-System VLA works the same way:
- The **VLM** (the strategist) takes in camera images and produces scene descriptions, causal reasoning, and high-level plans. "There are construction cones encroaching into the left lane. I should nudge right to maintain safe clearance."
- A **lightweight classical planner** (the driver) takes this reasoning and converts it into safe, kinematically-feasible trajectories — making sure the car does not swerve too hard or violate physics.

The advantage? Safety guarantees. The fast planner ensures the car always follows physically possible paths. The slow VLM provides the intelligence and reasoning.

DriveVLM-Dual and Mobileye's VLSA (Vision-Language-Semantic-Action) system use this approach. Mobileye's system is particularly interesting — the VLSA module provides structured semantic guidance to the planner, while a separate formal safety layer ensures no collision.


![End-to-end VLAs unify reasoning and control; dual-system VLAs separate slow deliberation from fast execution.](figures/figure_4.png)
*End-to-end VLAs unify reasoning and control; dual-system VLAs separate slow deliberation from fast execution.*


Now the question is: which approach is better? The honest answer is that both have their place, and many cutting-edge systems actually combine elements of both. Let us look at how these ideas play out in real architectures.

---

## How VLAs are Trained

Before we look at specific architectures, let us understand how these models learn to drive.

### The Pre-training Foundation

VLAs do not start from scratch. This is a crucial point. Training a model to understand images, text, **and** driving from nothing would require an astronomical amount of driving data — far more than exists today.

Instead, VLAs build on top of **pre-trained Vision-Language Models (VLMs)** — models that have already learned to understand images and text from billions of internet examples.

Here is an analogy. Imagine you are hiring a brilliant translator who already speaks 50 languages. They understand grammar, context, cultural nuances, and can hold conversations on any topic. Now you just need to teach them **one more skill** — driving. You do not need to teach them what a "stop sign" looks like or what "construction zone" means — they already know. You just need to show them what actions to take when they encounter these situations.

This is exactly how VLA training works. The pre-trained VLM brings:
- **Visual understanding:** It can already recognize objects, scenes, and spatial relationships
- **World knowledge:** It knows what "school zone" means, why ambulances have sirens, and that ice is slippery
- **Reasoning ability:** It can follow logical chains like "pedestrian is walking toward crosswalk → pedestrian might cross → I should slow down"

We just need to teach it the **action** part.

### Fine-tuning on Driving Data

The training data consists of triplets: **(camera images, language description, expert driver's actions)**. The language description might be a scene caption ("A busy intersection with a pedestrian crossing") or a driving command ("Turn left at the traffic light").

The training objective is simple: **predict the expert's action given the image and text.** This is essentially **behavioral cloning** — the model learns to imitate what a skilled human driver would do in every situation.

The loss function is the standard cross-entropy loss over action tokens:


$$\mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(a_t \mid I_t, \ell)$$


Let us plug in some simple numbers. Suppose we have $T = 3$ timesteps, and at each step the model assigns the following probabilities to the expert's actual action:
- $p_\theta(a_1 \mid I_1, \ell) = 0.8$ (the model is fairly confident)
- $p_\theta(a_2 \mid I_2, \ell) = 0.5$ (less confident)
- $p_\theta(a_3 \mid I_3, \ell) = 0.9$ (very confident)

Then:

$$\mathcal{L} = -[\log(0.8) + \log(0.5) + \log(0.9)]$$
$$= -[(-0.22) + (-0.69) + (-0.11)]$$
$$= -(-1.02) = 1.02$$

A lower loss means the model assigns higher probabilities to the expert's actions — which is exactly what we want. As training progresses, this loss decreases and the model learns to drive more like the expert.


![VLA training starts from a pre-trained VLM and adds driving capability through behavioral cloning on expert demonstrations.](figures/figure_5.png)
*VLA training starts from a pre-trained VLM and adds driving capability through behavioral cloning on expert demonstrations.*


### The Scale of Modern Driving Datasets

Now, here is something truly remarkable about how fast this field is moving. Let us look at the scale of datasets used to train driving VLAs.

The **nuScenes** dataset, released in 2019, was groundbreaking at the time. It contained about **5.5 hours** of driving data from Boston and Singapore.

The **Waymo Open Dataset** (2020) scaled this up to about **570 hours** across multiple US cities.

NVIDIA's **Physical AI AV dataset** (2025) contains **1,727 hours** of driving across 25 countries and 2,500+ cities, with multi-camera, LiDAR, and radar coverage.

And the training data for **Alpamayo**? **80,000 hours** of multi-camera video, over **1 billion images**, and **700,000 Chain-of-Causation reasoning annotations**.

That is a four-order-of-magnitude increase in just five years. This is exactly what we want — more data means better generalization, and better generalization means safer driving.


![Driving dataset scale has grown by four orders of magnitude in five years.](figures/figure_6.png)
*Driving dataset scale has grown by four orders of magnitude in five years.*


This exponential growth in data is one of the key reasons VLAs have suddenly become viable for autonomous driving.

---

## Landmark VLA Architectures

Now let us look at the models that brought VLAs from a research idea to production-ready systems. We will cover four architectures, each representing a major step forward.

### RT-2 — The Pioneer (2023)

Our story begins not with driving, but with **robotics**. In 2023, Google DeepMind released **RT-2 (Robotic Transformer 2)**, the first model to demonstrate that a pre-trained VLM can directly output physical actions.

The architecture is elegantly simple. RT-2 takes a pre-trained VLM — either **PaLM-E** (562B parameters) or **PaLI-X** (55B parameters) — and fine-tunes it on robot manipulation data. The key trick? Actions are **tokenized** and appended to the model's text vocabulary. So instead of outputting just words, the model can output action tokens like `[move_gripper_x: 0.3]` `[move_gripper_y: -0.1]`.

The result that stunned the field was **emergent generalization**. When researchers asked the robot to "pick up the object that is closest to the blue ball" — an instruction it had never seen during training — it could do it. The robot combined spatial reasoning (understanding "closest to") with visual grounding (identifying the blue ball) and motor execution (grasping the object) in one seamless inference pass.

Now, RT-2 was designed for robotic arms picking up objects, not for cars navigating highways. But it proved something fundamental: **a single VLM can bridge the gap from seeing to doing.** This insight is what launched the entire VLA-for-driving movement.


![RT-2 treats physical actions as tokens in the language model's vocabulary, enabling a VLM to directly control a robot.](figures/figure_7.png)
*RT-2 treats physical actions as tokens in the language model's vocabulary, enabling a VLM to directly control a robot.*


### EMMA — Waymo's End-to-End Driving Model (2024)

If RT-2 proved the concept in robotics, **EMMA** brought it to driving. Developed by Waymo and built on top of **Gemini** (Google's multimodal foundation model), EMMA is one of the most complete end-to-end driving VLAs.

EMMA takes in **multi-view camera images** (front, left, right, rear) along with a text prompt describing the driving task. Here is the key innovation: it represents **everything as text tokens** — including the output trajectory.

Want the model to plan a path? The trajectory waypoints are serialized as text: `(2.1, 0.3), (4.5, 0.2), (6.8, -0.1), ...` — each waypoint is just a pair of numbers that the language model generates autoregressively, one token at a time.

But EMMA does more than just planning. By changing the text prompt, the same model can perform:
- **Motion planning:** "Plan a safe trajectory for the next 8 seconds"
- **Object detection:** "Identify all vehicles and pedestrians in the scene"
- **Road graph estimation:** "Describe the lane structure at this intersection"
- **Scene understanding:** "What is happening in this driving scene?"

This multi-task capability is what makes EMMA powerful. It achieved **state-of-the-art results** on the Waymo Open Motion Dataset, outperforming many specialized models that were designed for individual tasks.


![EMMA represents driving as language modeling — trajectory waypoints are generated as text tokens by the Gemini foundation model.](figures/figure_8.png)
*EMMA represents driving as language modeling — trajectory waypoints are generated as text tokens by the Gemini foundation model.*


### DriveVLM — Reasoning Before Acting (2024)

EMMA showed that a VLM can plan trajectories. But how does it make decisions? In a complex scene with multiple lanes, pedestrians, and construction zones, how does the model decide what to do? The decision process is a black box.

**DriveVLM** addresses this with an elegant idea: **make the model show its work.**

Instead of directly predicting a trajectory from an image, DriveVLM uses a three-stage **chain-of-thought** pipeline:

1. **Scene Description:** "I see a four-lane road with construction cones on the left side. There is a red sedan ahead in my lane, and a pedestrian waiting at the crosswalk on the right."
2. **Scene Analysis:** "The construction cones are encroaching into my lane. The red sedan is decelerating. The pedestrian appears to be waiting but might step into the road."
3. **Hierarchical Planning:** "I should slow down to match the red sedan's speed, nudge slightly right to avoid the construction cones, and be prepared to stop if the pedestrian enters the crosswalk."

This is the Dual-System VLA approach in action. The VLM does the reasoning (the "slow thinking"), and then a traditional motion planner converts this high-level plan into a safe, smooth trajectory (the "fast acting"). The combination is called **DriveVLM-Dual**.

The beauty of this approach is **interpretability**. When the car makes a decision, you can read exactly why. This is invaluable for debugging, for safety validation, and — crucially — for regulators who need to understand why an autonomous vehicle did what it did.

### NVIDIA Alpamayo — The State of the Art (2025–2026)

Now let us come to the main character in our story.

Announced at CES 2026, **NVIDIA Alpamayo** is the first industry-scale, open-source reasoning VLA for autonomous driving. It represents the most ambitious attempt yet to bring VLAs from the research lab to the real road.

Let us break down what makes Alpamayo special.

**Architecture.** Alpamayo-R1 is a 10.5 billion parameter model with two components:
- A **Cosmos-Reason VLM backbone** (8.2B parameters) that handles perception and reasoning
- A **diffusion-based action decoder** (2.3B parameters) that generates smooth trajectories

This is a hybrid approach. The VLM does the thinking (understanding the scene, reasoning about what to do), and the diffusion decoder does the acting (producing precise trajectory waypoints). The diffusion decoder is crucial because it generates **continuous, smooth** trajectories — no discretization artifacts from tokenization.

**Input.** Four cameras — front-wide, front-tele, cross-left, cross-right — each at 10 Hz, with a 0.4-second history window (4 frames per camera). Plus 1.6 seconds of egomotion history (where the car has been recently). This gives the model both spatial awareness (multiple viewpoints) and temporal awareness (how things are changing over time).

**Output.** Two things simultaneously:
1. A **6.4-second future trajectory** — 64 waypoints at 10 Hz, each specifying position (x, y, z) and rotation. This tells the car exactly where to go for the next 6.4 seconds.
2. A **Chain-of-Causation reasoning trace** — a natural language explanation of the model's decision.

The Chain-of-Causation is Alpamayo's signature feature. Here is a real example from NVIDIA's technical blog. The model encounters a construction zone and outputs:

> **Reasoning:** "Nudge to the left to increase clearance from the construction cones encroaching into the lane."
>
> **Trajectory:** [64 waypoints curving gently to the left]

The model does not just drive — it **explains why it drives that way.** This is exactly what we want for safety-critical applications. And this is not a post-hoc explanation bolted on after the fact. The reasoning trace is generated jointly with the trajectory, meaning the model's reasoning actually influences its actions.

**Training scale.** Alpamayo was trained on:
- 80,000 hours of multi-camera driving video
- Over 1 billion images
- 700,000 Chain-of-Causation annotations (structured causal explanations paired with driving scenarios)

**Real-world deployment.** The Mercedes-Benz CLA sedan, shipping in 2026, will be the **first production vehicle** with Alpamayo-enhanced driving capabilities under the "MB.Drive Assist Pro" branding. Uber and Lucid Motors have also announced integrations. This is not a research demo — it is going into real cars that real people will drive.

**Open source.** The model weights and inference code are available on HuggingFace (non-commercial license for weights, Apache 2.0 for code). This means anyone with an NVIDIA GPU (24GB+ VRAM) can run Alpamayo locally.


![Alpamayo combines a Cosmos-Reason VLM with a diffusion action decoder to jointly generate reasoning traces and smooth trajectories.](figures/figure_9.png)
*Alpamayo combines a Cosmos-Reason VLM with a diffusion action decoder to jointly generate reasoning traces and smooth trajectories.*



![Alpamayo generates human-readable causal reasoning alongside its planned trajectory.](figures/figure_10.png)
*Alpamayo generates human-readable causal reasoning alongside its planned trajectory.*


---

## The Critical Challenges

VLAs are exciting, but let us be honest about the challenges that remain. Autonomous driving is a safety-critical application, and there are hard problems that VLAs still need to solve.

### Latency — Thinking Fast Enough

Here is a simple calculation. At 60 miles per hour, a car travels **27 meters every second.** If the VLA takes 500 milliseconds to think — a reasonable inference time for a 10B parameter model — the car has already moved **13.5 meters** before the model even produces its first action.

You cannot have a GPS that takes 30 seconds to recalculate your route while you are about to miss the exit. The same applies here — the model must think fast enough for its decisions to be relevant.

Current solutions include:
- **Dual-System architecture:** Let the fast planner handle real-time control at 100Hz while the VLM reasons at a slower 2-5Hz cadence. Alpamayo uses this approach — the diffusion decoder runs fast, while the VLM reasons at the rate it can.
- **Model distillation:** Train a smaller, faster "student" model to mimic the larger VLA's behavior
- **Asynchronous planning:** The VLM plans ahead for the next 6+ seconds, so even if it runs slowly, the planned trajectory is still valid for a while

### Safety and the Long Tail

The biggest challenge in autonomous driving is not the 99% of driving that is easy — it is the **0.01% that is deadly**. These are called **long-tail scenarios**: a mattress falls off a truck on the highway, a child chases a ball into the road, an emergency vehicle approaches from an unexpected direction, a road sign is obscured by graffiti.

Traditional perception systems struggle with these scenarios because they have never been trained on them. How do you train an object detector to recognize "mattress falling off truck" when there are only a handful of such examples in any dataset?

This is where VLAs have a genuine advantage. Because the language backbone has been trained on billions of words from the internet, it has **conceptual knowledge** about unusual situations. It knows what a mattress is, what happens when objects fall from vehicles, and why you should swerve or brake. Even if the vision model has never seen a mattress on a highway, the language model can reason about it.

But VLAs also introduce a new risk: **hallucination.** The model might "see" a pedestrian that does not exist and brake unnecessarily. Or it might confuse a shadow for an obstacle. This is the same problem that affects chatbots — language models can generate confident-sounding nonsense — but in a driving context, the consequences are far more severe.


![Language reasoning helps VLAs handle rare scenarios that rule-based systems cannot anticipate.](figures/figure_11.png)
*Language reasoning helps VLAs handle rare scenarios that rule-based systems cannot anticipate.*


### Sim-to-Real Transfer

You cannot train a self-driving car by crashing it into things and learning from mistakes. Real-world data collection is expensive and dangerous. So most VLAs rely heavily on **simulation** — virtual environments like CARLA, nuPlan, or NVIDIA's AlpaSim where the model can practice driving billions of miles without risking anyone's safety.

The problem is the **domain gap.** Simulated camera images do not look exactly like real camera images. Simulated physics is an approximation. The behavior of simulated pedestrians and drivers is overly simplistic.

VLAs offer a partial solution here. Because language descriptions are **domain-agnostic** (the word "pedestrian crossing" means the same thing in simulation and reality), the language backbone helps bridge the sim-to-real gap. The model does not need to learn that "this specific texture pattern means a pedestrian" — it can learn that "a humanoid shape near a crosswalk means a pedestrian is about to cross."

NVIDIA's **AlpaSim** addresses this more directly. It is a microservice-based simulator with 900 reconstructed real-world scenes rendered using Omniverse. Research has shown that AlpaSim rollouts can **reduce variance in real-world metrics by up to 83%**, making simulation-based evaluation much more predictive of actual on-road performance.

---

## Practical Implementation — A Minimal VLA for Driving

Enough theory — let us look at some practical implementation. We will walk through a simplified VLA inference loop to understand how the pieces fit together in code.

The following pseudocode shows the core inference loop of a VLA model for driving:

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Load pre-trained VLA model and processor
model = AutoModelForCausalLM.from_pretrained("nvidia/Alpamayo-R1-10B")
processor = AutoProcessor.from_pretrained("nvidia/Alpamayo-R1-10B")

def vla_inference(camera_images, egomotion_history, prompt):
    """
    Run VLA inference: images + text → reasoning + trajectory.

    camera_images: dict of 4 camera views, each (4, 3, 320, 576) — 4 frames at 10Hz
    egomotion_history: (16, 12) — 16 waypoints, each with (x,y,z) + 9D rotation
    prompt: str — driving instruction or scene context
    """
    # Preprocess inputs: tokenize text, encode images
    inputs = processor(
        images=camera_images,
        text=prompt,
        ego_history=egomotion_history,
        return_tensors="pt"
    )

    # Forward pass — model produces reasoning + trajectory
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    # Decode the reasoning trace (text tokens)
    reasoning = processor.decode(outputs.reasoning_tokens)

    # Decode the trajectory (64 waypoints × 12D each)
    trajectory = outputs.trajectory  # shape: (64, 12)

    return reasoning, trajectory
```

Let us understand this code step by step. The `processor` handles the multi-modal input preparation — it tokenizes the text prompt, encodes the four camera images into patch tokens, and formats the egomotion history. The `model.generate()` call runs the VLM backbone, producing both reasoning tokens (text) and trajectory outputs (from the diffusion decoder). Finally, we decode both outputs into human-readable reasoning and a 64-waypoint trajectory.

Now let us look at the action tokenization approach, which is simpler and used by models like RT-2 and EMMA:

```python
import numpy as np

def tokenize_action(action, action_min, action_max, n_bins=256):
    """Convert a continuous action value to a discrete token."""
    normalized = (action - action_min) / (action_max - action_min)
    token = int(round(normalized * (n_bins - 1)))
    return max(0, min(n_bins - 1, token))  # clamp to valid range

def detokenize_action(token, action_min, action_max, n_bins=256):
    """Convert a discrete token back to a continuous action value."""
    normalized = token / (n_bins - 1)
    action = normalized * (action_max - action_min) + action_min
    return action

# Example: tokenize steering angle of 15 degrees
steering = tokenize_action(15.0, action_min=-45.0, action_max=45.0)
print(f"Steering 15° → token {steering}")  # Output: token 170

# Detokenize back
recovered = detokenize_action(170, action_min=-45.0, action_max=45.0)
print(f"Token 170 → {recovered:.1f}°")  # Output: 15.0°
```

Here you can see that the tokenization and detokenization are exact inverses of each other — the continuous steering angle of 15° maps to token 170, and token 170 maps back to 15.0°. This is exactly what we want — no information is lost in the round-trip conversion. In practice, the model would output token 170 as part of its generated sequence, and the post-processing step would convert it back to a physical steering command.

---

## The Road Ahead — What's Next for VLAs in Driving

We have covered the foundations, the architectures, and the challenges. Let us now look at where this field is heading.

**World Models.** The next frontier for VLAs is the ability to **imagine the future before acting.** Current VLAs react to what they see right now. A VLA with a world model could simulate: "If I change lanes now, the truck behind me will need to brake. If I wait 3 seconds, there will be a gap." NVIDIA's Cosmos foundation model, which serves as the backbone for Alpamayo, is already moving in this direction — it was designed as a world foundation model for physical AI.

**Closed-Loop Training.** A major limitation of behavioral cloning is **covariate shift**: the model is trained on expert demonstrations, but at deployment time, its own imperfect actions lead to states the expert never visited. NVIDIA's **RoaD algorithm**, released alongside Alpamayo, addresses this by training the VLA in closed-loop simulation — the model drives in AlpaSim, encounters its own mistakes, and learns to recover from them. This is significantly more data-efficient than traditional reinforcement learning.

**Multi-Agent Reasoning.** Driving is inherently a multi-agent problem. Other drivers have intentions, preferences, and sometimes irrational behaviors. Future VLAs will use language to reason about other agents: "The car in the adjacent lane has been drifting toward me — the driver might be distracted. I should increase my lateral clearance."

**Regulation and Interpretability.** Perhaps the most important advantage of VLAs for the autonomous driving industry is not technical — it is regulatory. When a VLA makes a decision, it can **explain why** in natural language. This is enormously valuable for safety certifications, accident investigations, and building public trust. Alpamayo's Chain-of-Causation is a concrete step toward this: every driving decision comes with a human-readable justification.

**The Convergence of Robotics and Driving.** Here is a fascinating trend: the same VLA architecture that controls a robot arm picking up objects (RT-2) is now controlling a car navigating a highway (Alpamayo). The abstraction is the same — see, think, act. This convergence means that advances in one domain directly benefit the other. A breakthrough in robotic manipulation reasoning could improve autonomous driving reasoning, and vice versa.


![VLAs for driving have evolved from research prototypes to production-ready systems in under three years.](figures/figure_12.png)
*VLAs for driving have evolved from research prototypes to production-ready systems in under three years.*


---

## Conclusion

Let us take a step back and look at the big picture.

We started this article with a simple analogy — teaching your friend to drive. They see the road, understand your instructions, and act accordingly. Vision, language, action.

We then built up the VLA architecture piece by piece: a vision encoder that converts camera images into feature tokens, a language model that reasons about the scene, and an action head that produces trajectories — either through tokenization or diffusion decoding.

We explored two paradigms: end-to-end VLAs that unify everything in one model, and dual-system VLAs that separate slow reasoning from fast acting.

We walked through the landmark architectures: RT-2 that proved the concept, EMMA that brought it to driving, DriveVLM that added interpretable reasoning, and Alpamayo that is putting it into production cars.

And we were honest about the challenges: latency, safety, hallucination, and the sim-to-real gap.

The field is moving at an extraordinary pace. Three years ago, the idea of a single neural network that could see a construction zone, reason "I should nudge left to maintain clearance from those cones," and then smoothly steer the car — all in one forward pass — was a research aspiration. Today, it is shipping in a Mercedes-Benz.

That's it!

---

## References

- Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (2023)
- Hwang et al., "EMMA: End-to-End Multimodal Model for Autonomous Driving" (Waymo, 2024)
- Tian et al., "DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models" (2024)
- NVIDIA, "Alpamayo: Open AI Models for Autonomous Vehicle Development" (2025)
- Shao et al., "OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model" (AAAI 2026)
- Jiang et al., "Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future" (Survey, arXiv:2512.16760)
- Mobileye, "Vision-Language-Semantic-Action (VLSA) for Autonomous Driving" (CES 2026)
