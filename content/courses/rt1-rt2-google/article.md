# RT-1 and RT-2: Google's Robotics Transformers

*How Google taught robots to understand 700 tasks — and then gave them the power of language models*

Vizuara AI

---

## The Robot That Understands "Pick Up the Unhealthy Option"

Let us start with a remarkable demonstration. A robot sits at a table with an apple and a bag of chips in front of it. A human says: "Pick up the object that is the most unhealthy."

The robot reaches for the chips.

This sounds trivially easy for a human, but think about what the robot just did. It was **never trained** on the concept of "unhealthy food." It was never given a dataset mapping foods to health ratings. Yet it correctly identified chips as less healthy than an apple and executed a precise grasping motion.


![A robot arm at a table with an apple and a bag of chips. A speech bubble shows the instruction Pick up the most unhealthy option. The robot is reaching toward the chips. A thought bubble near the robot shows semantic reasoning: chips equals processed food equals unhealthy. Clean illustration style.](figures/figure_1.png)
*A robot arm at a table with an apple and a bag of chips. A speech bubble shows the instruction Pick up the most unhealthy option. The robot is reaching toward the chips. A thought bubble near the robot shows semantic reasoning: chips equals processed food equals unhealthy. Clean illustration style.*


This is **RT-2** — the Robotics Transformer 2 — and it represents one of the most important breakthroughs in robot learning. But to understand RT-2, we first need to understand where it came from. Let us start with its predecessor: **RT-1**.

---

## The Problem: Why Was Robot Learning So Hard?

Before RT-1, the state of robot learning was frustrating. Most learned robot policies could handle only a handful of tasks, required careful engineering for each new skill, and failed catastrophically when encountering anything slightly outside their training distribution.

The core problem was data. In natural language processing, models can train on trillions of words from the internet. In computer vision, there are billions of labeled images. But in robotics, every single demonstration must be physically collected by a real robot.


![Data scale comparison. Three horizontal bars showing data availability. NLP: trillions of tokens from web text, very long bar. Computer Vision: billions of images from ImageNet and web scraping, long bar. Robotics: thousands of demonstrations from physical collection, very short bar. The gap is dramatic and visually striking.](figures/figure_10.png)
*Data scale comparison. Three horizontal bars showing data availability. NLP: trillions of tokens from web text, very long bar. Computer Vision: billions of images from ImageNet and web scraping, long bar. Robotics: thousands of demonstrations from physical collection, very short bar. The gap is dramatic and visually striking.*


This creates a vicious cycle: without large datasets, models cannot generalize. Without generalization, each new task requires a new dataset. Without efficient data collection, building large datasets is impractical.

Now the question is: can we break this cycle? Can we do for robotics what GPT did for language?

---

## RT-1: The Robotics Transformer

In 2022, Google's robotics team made a bold bet. Instead of training a specialist model for each task, they would collect a **massive, multi-task dataset** and train a single model on everything.

### The Data Collection Effort

Google deployed a fleet of **13 robots** in office kitchen environments. Over **17 months**, these robots collected over **130,000 demonstration episodes** across **700+ task instructions**.

The tasks ranged from simple ("pick up the can") to complex ("move the apple near the cloth") and covered manipulation skills like picking, placing, opening drawers, and pushing objects.


![Fleet of robot arms in a kitchen environment collecting demonstrations. Multiple robots shown at workstations with varied objects like cans, fruits, drawers, and cloths. An overlay showing statistics: 13 robots, 17 months, 130K episodes, 700 plus tasks. Clean infographic style.](figures/figure_2.png)
*Fleet of robot arms in a kitchen environment collecting demonstrations. Multiple robots shown at workstations with varied objects like cans, fruits, drawers, and cloths. An overlay showing statistics: 13 robots, 17 months, 130K episodes, 700 plus tasks. Clean infographic style.*


### The RT-1 Architecture

RT-1's architecture is elegant and purposeful. Let us walk through each component:

**1. EfficientNet Vision Encoder**

The input image from the robot's camera is processed by an EfficientNet — a lightweight convolutional neural network that extracts visual features. This produces a set of image feature tokens.

**2. TokenLearner**

Here is a clever trick. The visual features from EfficientNet produce many tokens (81), but processing all of them through a transformer is expensive. **TokenLearner** compresses these 81 tokens down to just **8 tokens** using learned spatial attention. This dramatically reduces computation while retaining the important information.

**3. Transformer Decoder**

The 8 compressed visual tokens are combined with the tokenized language instruction and fed into a Transformer decoder. The transformer processes this multimodal sequence and outputs **action tokens**.


![RT-1 architecture diagram. Flow from left to right: Camera image enters EfficientNet CNN which produces 81 image tokens. These pass through TokenLearner which compresses them to 8 tokens. The 8 visual tokens plus tokenized text instruction enter a Transformer decoder. Output: 11 discretized action tokens. Each component is a labeled box with arrows showing data flow.](figures/figure_3.png)
*RT-1 architecture diagram. Flow from left to right: Camera image enters EfficientNet CNN which produces 81 image tokens. These pass through TokenLearner which compresses them to 8 tokens. The 8 visual tokens plus tokenized text instruction enter a Transformer decoder. Output: 11 discretized action tokens. Each component is a labeled box with arrows showing data flow.*


### Action Tokenization: The Key Innovation

How do you represent continuous robot actions in a format that a transformer can produce? RT-1's answer: **discretization**.

Each action dimension is divided into 256 bins. For example, the robot's x-position might range from -0.5m to 0.5m. This range is divided into 256 equal bins, and the continuous value is mapped to the nearest bin index.

The full action has 11 dimensions:
- 3 for end-effector position (x, y, z)
- 3 for end-effector rotation (roll, pitch, yaw)
- 1 for gripper open/close
- 3 for base movement (x, y, yaw)
- 1 for mode switching (base vs arm)

So one action is represented as 11 integers, each in the range [0, 255]. The transformer simply needs to predict these 11 classification outputs.

$$
a_i = \left\lfloor \frac{a_i^{\mathrm{raw}} - a_i^{\min}}{a_i^{\max} - a_i^{\min}} \cdot 255 \right\rfloor, \quad i = 1, \ldots, 11
$$

This is exactly what we want. It turns a continuous control problem into a sequence prediction problem — the same kind of problem that transformers are extraordinarily good at.

### RT-1 Results

The results were striking:

- **97% success rate** on seen tasks in the training environment
- **76% success rate** on previously unseen task instructions
- **Massive improvement** over prior methods like BC-Z (Behavioral Cloning Zero-shot) and Gato


![Bar chart comparing RT-1 versus baseline methods. Four grouped bars for: Seen Tasks, Unseen Instructions, Unseen Objects, Distractor Robustness. RT-1 shown in blue consistently outperforming BC-Z in orange, Gato in green, and SayCan in purple. RT-1 bars are tallest in all categories. Clean chart with clear labels.](figures/figure_4.png)
*Bar chart comparing RT-1 versus baseline methods. Four grouped bars for: Seen Tasks, Unseen Instructions, Unseen Objects, Distractor Robustness. RT-1 shown in blue consistently outperforming BC-Z in orange, Gato in green, and SayCan in purple. RT-1 bars are tallest in all categories. Clean chart with clear labels.*


But RT-1 had a fundamental limitation. It could only generalize to new **combinations** of things it had seen before. Ask it to handle a truly novel concept — something never in the robot training data — and it would fail.

This brings us to RT-2.

---

## RT-2: When Robots Learn to Think

The key insight behind RT-2 is almost embarrassingly simple: **what if we just used a really large Vision-Language Model and taught it to output robot actions?**

Vision-Language Models (VLMs) like PaLM-E and PaLI-X have been trained on billions of image-text pairs from the internet. They understand objects, scenes, spatial relationships, abstract concepts, cultural references, and much more. All of this knowledge is stored in their weights.

RT-2's innovation: represent robot actions as **text tokens** and fine-tune a VLM to output them alongside its normal language outputs.

### Actions as Text

In RT-2, a robot action is represented as a simple string of numbers:

```
"1 128 91 241 5 101 127"
```

Each number corresponds to one of the 256 bins for each action dimension. The VLM is trained to generate this string of numbers as if it were generating any other text response.


![RT-2 concept diagram. Input side shows a camera image of a table with objects plus text instruction Pick up the can. These feed into a large VLM box labeled PaLM-E 12B or PaLI-X 55B. Output side shows the model generating text: 1 128 91 241 5 101 127 as action tokens. An arrow shows these numbers being decoded into continuous robot joint commands.](figures/figure_5.png)
*RT-2 concept diagram. Input side shows a camera image of a table with objects plus text instruction Pick up the can. These feed into a large VLM box labeled PaLM-E 12B or PaLI-X 55B. Output side shows the model generating text: 1 128 91 241 5 101 127 as action tokens. An arrow shows these numbers being decoded into continuous robot joint commands.*


The beauty of this approach is that the VLM does not need to learn about the physical world from scratch. It already knows what a "can" looks like, where "the top-left corner" of a table is, and what "unhealthy food" means. It just needs to learn a new output format: the language of robot actions.

### The Architecture

RT-2 comes in two variants:

1. **RT-2-PaLM-E** — Built on PaLM-E (12B parameters), which is itself a combination of PaLM (language) and ViT (vision)
2. **RT-2-PaLI-X** — Built on PaLI-X (55B parameters), a vision-language model pre-trained on web-scale image-text data

Both variants are **co-fine-tuned**: they train on a mixture of:
- Original vision-language data (so the model retains general knowledge)
- Robot action data (so the model learns to output motor commands)


![RT-2 architecture showing two variants side by side. Left: RT-2-PaLM-E with PaLM language model and ViT vision encoder, 12B parameters. Right: RT-2-PaLI-X with PaLI-X VLM, 55B parameters. Both show input image plus text instruction flowing in and action token string flowing out. Co-fine-tuning data mixture shown below both: web data plus robot data.](figures/figure_6.png)
*RT-2 architecture showing two variants side by side. Left: RT-2-PaLM-E with PaLM language model and ViT vision encoder, 12B parameters. Right: RT-2-PaLI-X with PaLI-X VLM, 55B parameters. Both show input image plus text instruction flowing in and action token string flowing out. Co-fine-tuning data mixture shown below both: web data plus robot data.*


### Chain-of-Thought Reasoning for Robots

One of the most fascinating aspects of RT-2 is that it can perform **chain-of-thought reasoning** — just like language models do when solving math problems.

When prompted with: "I need to place the object on {target location}. First I need to plan: ..."

RT-2 can reason step by step about what actions to take before outputting the motor commands. This brings a form of deliberate planning to robot control.


![Chain-of-thought reasoning example. A robot viewing a table with objects. The model's internal reasoning shown as a thought process: Step 1 I see an apple and a plate. Step 2 The instruction says to move the apple to the plate. Step 3 I need to reach toward the apple first. Then: action tokens output. Clean diagram showing reasoning before action.](figures/figure_7.png)
*Chain-of-thought reasoning example. A robot viewing a table with objects. The model's internal reasoning shown as a thought process: Step 1 I see an apple and a plate. Step 2 The instruction says to move the apple to the plate. Step 3 I need to reach toward the apple first. Then: action tokens output. Clean diagram showing reasoning before action.*


### Emergent Capabilities: The Magic of Transfer

The most remarkable results from RT-2 are the **emergent capabilities** — abilities that were never explicitly trained but emerged from the VLM's internet-scale knowledge:

**Semantic reasoning:** "Pick up the object that is the most unhealthy" — the robot picks chips over fruit. The concept of "unhealthy food" came from the VLM's pre-training, not the robot data.

**Symbol grounding:** "Move the can to the Taylor Swift poster" — the robot understands a pop culture reference that was never in any robot dataset.

**Spatial reasoning:** "Place the apple at the top-left corner of the table" — fine-grained spatial understanding transferred from vision-language pre-training.

**Semantic category reasoning:** "Pick up something that is a fruit" — understanding categorical relationships between objects.


![Grid of four emergent capability examples. Top-left: semantic reasoning with unhealthy food choice. Top-right: symbol grounding with Taylor Swift reference. Bottom-left: spatial reasoning with top-left corner instruction. Bottom-right: category reasoning with pick up a fruit instruction. Each panel shows the robot, the instruction, and the correct action being taken.](figures/figure_8.png)
*Grid of four emergent capability examples. Top-left: semantic reasoning with unhealthy food choice. Top-right: symbol grounding with Taylor Swift reference. Bottom-left: spatial reasoning with top-left corner instruction. Bottom-right: category reasoning with pick up a fruit instruction. Each panel shows the robot, the instruction, and the correct action being taken.*


None of these were in the robot training data. They emerged purely from the VLM's pre-existing knowledge. This is truly amazing.

---

## RT-1 vs RT-2: Head-to-Head

Let us compare the two models directly:

| **Aspect** | **RT-1** | **RT-2** |
|-----------|---------|---------|
| **Architecture** | EfficientNet + TokenLearner + Transformer | Full VLM (PaLM-E / PaLI-X) |
| **Parameters** | ~35M | 12B / 55B |
| **Training data** | Robot data only | Web data + Robot data |
| **Seen tasks** | 97% success | 95% success |
| **Unseen reasoning** | 32% success | 62% success |
| **Inference speed** | Fast (~5Hz) | Slow (~1-3Hz) |
| **Key strength** | Speed, reliability | Generalization, reasoning |

On tasks the models have seen before, RT-1 and RT-2 perform comparably. But on tasks requiring novel reasoning — understanding new concepts, following complex instructions, handling unseen objects — RT-2 dramatically outperforms RT-1.

The trade-off is speed. RT-1's 35M parameters can run at 5Hz on robot hardware. RT-2's 55B parameters struggle to run at even 1-3Hz.

---

## The Open X-Embodiment Story

The impact of RT-1 and RT-2 extended far beyond Google's labs. In 2023, Google led a collaboration with 21 research institutions to create the **Open X-Embodiment (OXE)** dataset.

This dataset combines robot demonstrations from 22 different robot embodiments — from Google's own robots to Franka Pandas, UR5 arms, and mobile manipulators — creating a dataset of over 1 million episodes.

Models trained on this diverse data — called **RT-1-X** and **RT-2-X** — showed a remarkable finding: **training on data from many different robots improves performance on each individual robot.** The cross-embodiment transfer works.

RT-2-X showed a **50% improvement** in emergent skill evaluation compared to RT-2 trained on Google robot data alone.


![Open X-Embodiment ecosystem diagram. A central hub labeled OXE Dataset 1M episodes connected to icons of 22 different robot types contributed by different institutions: Google, Stanford, Berkeley, CMU, and others. Arrows show data flowing in and improved policies flowing out. Key statistic: 50 percent improvement from cross-embodiment training.](figures/figure_9.png)
*Open X-Embodiment ecosystem diagram. A central hub labeled OXE Dataset 1M episodes connected to icons of 22 different robot types contributed by different institutions: Google, Stanford, Berkeley, CMU, and others. Arrows show data flowing in and improved policies flowing out. Key statistic: 50 percent improvement from cross-embodiment training.*


---

## Practical Code: Action Tokenization

Let us implement the core action tokenization mechanism used by both RT-1 and RT-2:

```python
import numpy as np

class ActionTokenizer:
    """Discretize continuous robot actions into tokens."""

    def __init__(self, action_ranges, num_bins=256):
        """
        Args:
            action_ranges: list of (min, max) for each action dimension
            num_bins: number of discrete bins per dimension
        """
        self.ranges = action_ranges
        self.num_bins = num_bins

    def tokenize(self, action):
        """Convert continuous action to discrete tokens."""
        tokens = []
        for i, (a_min, a_max) in enumerate(self.ranges):
            # Clip to valid range
            val = np.clip(action[i], a_min, a_max)
            # Map to [0, num_bins - 1]
            token = int((val - a_min) / (a_max - a_min) * (self.num_bins - 1))
            tokens.append(token)
        return tokens

    def detokenize(self, tokens):
        """Convert discrete tokens back to continuous action."""
        action = []
        for i, (a_min, a_max) in enumerate(self.ranges):
            val = tokens[i] / (self.num_bins - 1) * (a_max - a_min) + a_min
            action.append(val)
        return np.array(action)

# Example: 7-DOF robot arm
ranges = [
    (-0.5, 0.5),    # x position (meters)
    (-0.5, 0.5),    # y position
    (0.0, 0.5),     # z position
    (-np.pi, np.pi), # roll
    (-np.pi, np.pi), # pitch
    (-np.pi, np.pi), # yaw
    (0.0, 1.0),     # gripper (0=closed, 1=open)
]

tokenizer = ActionTokenizer(ranges)

# Continuous action
action = [0.1, -0.2, 0.3, 0.5, -0.1, 0.0, 0.8]

# Tokenize
tokens = tokenizer.tokenize(action)
print(f"Tokens: {tokens}")  # e.g., [153, 76, 153, 168, 120, 127, 204]

# Detokenize (slight quantization error)
recovered = tokenizer.detokenize(tokens)
print(f"Recovered: {recovered}")
```

Let us understand this code. The `ActionTokenizer` maps each continuous action dimension to a discrete bin index. With 256 bins, the quantization error is at most $$\frac{a_{\max} - a_{\min}}{255}$$ per dimension — typically less than 4mm for position and 1.4 degrees for rotation. This is precise enough for most manipulation tasks.

For RT-2, these token indices are simply converted to text strings: `"153 76 153 168 120 127 204"` — and the VLM is trained to generate them as language.

---

## What Came After: The Road Ahead

RT-1 and RT-2 opened the floodgates. Their key contributions — large-scale data collection, action tokenization, and leveraging VLMs — became the foundation for an entire field:

- **RT-H** (2024) — Adds human-in-the-loop corrections during deployment
- **Octo** (2024) — Open-source generalist policy inspired by RT-1's architecture
- **OpenVLA** (2024) — Open-source VLA inspired by RT-2's approach
- **Pi0** (2024) — Uses flow matching instead of action tokenization
- **RT-2-X** — Cross-embodiment scaling of RT-2

The central lesson from Google's Robotics Transformer line is this: **scale matters.** Scale your data (from tens to hundreds of thousands of demonstrations), scale your model (from millions to billions of parameters), and scale your pre-training (from robot data alone to internet-scale knowledge). Each axis of scaling unlocked qualitatively new capabilities.

In the next article, we will zoom out and look at the broader landscape of **Developments in Robotics Foundation Models** — how the ideas pioneered by RT-1 and RT-2 are being extended by labs around the world. See you next time!

---

*References:*
- *Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale" (2022)*
- *Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (2023)*
- *Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models" (2023)*