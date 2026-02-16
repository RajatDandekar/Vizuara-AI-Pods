# Vizuara Substack Writing Style Profile

> Comprehensive style analysis based on 9 reference articles covering Reinforcement Learning, RLHF, VAEs, Diffusion Models, Imitation Learning, and Score Matching.

---

## 1. Article Structure

### Opening Pattern: "Let us start with..." + Concrete Example

Nearly every article opens by immediately grounding the reader in a tangible, relatable scenario before introducing any formalism. The dominant opener is **"Let us start/begin with/by..."** followed by a concrete example:

- *"Let us start with a simple example. Imagine that you have collected handwriting samples from all the students in your class (100)."* — VAEs article
- *"Let us take a simple example: Suppose we want to train a robot to pour a 'heart shape' in a coffee (latte)"* — Imitation Learning article
- *"Let us start by looking at value functions."* — Three Horsemen article
- *"Let us start by understanding how Reinforcement Learning is applied to language models."* — RLHF article

Occasionally, articles open with a clean definition followed immediately by real-world examples:

- *"Diffusion is the natural tendency of particles (like molecules, heat, or even information) to move and spread out until they are evenly distributed."* — Diffusion Models article
- *"EBMs define a probability density via an energy function which assigns lower energy to more likely configurations."* — Score Matching article

### Section Flow Pattern

The dominant flow follows this structure:

```
Intuitive Example / Analogy
    → Concept Introduction (informal)
        → Mathematical Formulation (step-by-step)
            → More Examples / Diagrams
                → Practical Implementation (Code + Results)
                    → Transition to Next Topic / Cliffhanger
```

Key observations:
- **Theory and practice alternate.** Articles do not front-load all theory — they interleave explanations with examples, diagrams, and code.
- **Math is sandwiched between intuition.** Before every equation, there's a motivating example. After every equation, there's an interpretation.
- **Each major section follows a mini arc:** question posed → intuition built → math shown → practical example → takeaway.

### Transitions Between Sections

Transitions are almost always **question-driven** or use the phrase **"This brings us to..."**:

- *"So, how do we solve these equations? This naturally brings us to the following topic."* — Three Horsemen article
- *"But how do we choose actions that are the right actions in every state?"* — Three Horsemen article
- *"Now the question is, how do we generate the actual sample for the digit 5 once we pass this to the decoder?"* — VAEs article
- *"We are yet to understand how to train our score function... This brings us to Score-based Generative Models."* — Score Matching article
- *"But why are we discussing about this now?"* — Diffusion Models article

Other transitional patterns:
- *"Now let us look at..."* / *"Now let us understand..."*
- *"Enough theory, let us look at some practical implementation now."*
- *"Let us come to the main character in our story."*
- *"Now we will look at..."*

### How Articles Conclude

Articles typically end in one of two ways:

1. **Brief sign-off:**
   - *"That's it!"* — Diffusion Models article, DSM article
   - *"Thanks!"* — VAEs article

2. **GitHub/Colab links + resources:**
   - *"Refer to this GitHub repo for reproducing the code: [link]"* — Policy Gradient article
   - *"Here is the link to the original paper..."* — Score Matching article, DSM article

### Article Length

- **Shortest:** ~6,700 characters / ~1,500 words (DSM article)
- **Longest:** ~30,000 characters / ~5,000 words (Three Horsemen article)
- **Typical range:** 10,000–20,000 characters / 2,000–4,000 words
- Longer articles are appropriate when the concept is more complex

### Figures, Code Blocks, and Equations

- **Figures:** Extremely heavy usage — every article has 10–30+ image/diagram references. Diagrams appear every few paragraphs. They include: concept diagrams, architecture schematics, training result plots, animated GIFs, backup diagrams, before/after visualizations.
- **Code blocks:** 2–5 per article, placed in the "Practical Implementation" sections. Typically 10–30 lines of Python.
- **Equations:** Interspersed throughout, always preceded by verbal motivation and followed by verbal interpretation. Multi-step derivations are broken into labeled steps (Step 1, Step 2, etc.).

---

## 2. Voice & Tone

### Formality Level: Conversational Professor

The writing reads like a **friendly, enthusiastic tutor** explaining concepts one-on-one. It is neither rigidly academic nor flippantly casual. The tone sits at the intersection of accessibility and substance — concepts are simplified without being dumbed down.

- Academic enough: Includes formal equations, references to seminal papers, proper terminology
- Casual enough: Uses contractions, exclamation marks, emoji-like text smileys `:)`, and informal asides

### Use of First Person

**"We" and "Let us"** dominate — creating a collaborative, guided-journey feel:

- *"Let us start by understanding..."*
- *"We can simply use the softmax function..."*
- *"Now we have all the pieces of the puzzle ready..."*
- *"So, we have taken care of the following part..."*

**"I"** appears sparingly and authentically — usually to share personal confusion or enthusiasm:

- *"I have personally not understood the requirement of this, but I am just explaining it here anyways."* — RL Intro article
- *"I was initially very confused where this word comes from, but it means..."* — Imitation Learning article
- *"I did not stop at this. I used the same approach to train a GPT-2 model from scratch..."* — RLHF article
- *"The way I think about it is that..."* — Diffusion Models article

### How the Reader is Addressed

The reader is spoken to directly with **"you"**, creating an interactive dialogue:

- *"You might be thinking that, but I never defined the action space..."* — RL Intro article
- *"You might be wondering that: well, this is not very good."* — RL Intro article
- *"You might have seen something like this before."* — Policy Gradient article
- *"Can you imagine what will happen if we have an environment which has a huge number of states?"* — Three Horsemen article
- *"You might have noticed that these equations are difficult to solve..."* — Three Horsemen article
- *"You might have guessed what we are about to do next."* — DSM article
- *"Can you tell me what is wrong in the about diagram? What am I missing?"* — Imitation Learning article

### Humor and Personality

Humor is **light, occasional, and disarming** — never forced:

- *"Unless you are James Bond :)"* — RL Intro article (on remembering poker information)
- *"If only it was this easy to run reinforcement learning problems :)"* — RL Intro article
- *"Yes, we are taking Batman as our example :)"* — Diffusion Models article
- *"Brace yourselves!"* — Policy Gradient article (before a complex derivation)
- *"Okay, enough mathematics!"* — RLHF article
- *"It almost looks like a hiker who is drunk and trying to navigate their way in the terrain."* — Score Matching article
- *"Our 'drunk hiker' does quite well..."* — Score Matching/DSM articles
- *"Our lunar lander just falls off - it is not oriented properly towards the end."* — RL Intro article

### Confidence Level

**Assertive but honest.** Claims are stated directly. When results are shown, the author confidently validates them:

- *"This is exactly what we want."* (appears across nearly every article — a signature phrase)
- *"This clearly tells us that introducing a baseline not only reduces the variance but also allows for faster convergence."*
- *"We can clearly see that the reward increases with time..."*
- *"This is truly amazing."* — on the value of a 20-year-old research paper

Hedging appears only when genuinely uncertain:

- *"I have personally not understood the requirement of this..."*
- *"I was initially very confused..."*
- *"The generated distribution does not match the true distribution perfectly with the density, but..."*

---

## 3. Technical Explanation Style

### How New Concepts are Introduced: Analogy-First / Example-First

This is the most distinctive feature of the writing. **Every concept is introduced through a concrete analogy or real-world example before any formalism:**

| Concept | Analogy/Example Used |
|---|---|
| Variational Autoencoders | Handwriting samples from classmates |
| Latent variables | "Secret recipe" that determines handwriting |
| Forward diffusion | Perfume spreading across a room; sugar in water |
| Gaussian noise on images | Batman image with pixels modified |
| Score function | Compass pointing toward high-probability regions |
| Langevin sampling | Foggy landscape treasure hunt |
| Local minima problem | Getting stuck in a pothole on a mountain |
| Denoising score matching | Invisible magnets on a tabletop |
| Policy gradient | Climbing a mountain |
| On-policy vs off-policy | Learning to ride a bike vs watching a pro cyclist |
| PPO clipping | Runner with a coach, don't change style too much |
| Importance sampling | Estimating country's average age from samples |
| Reward hacking | LLM gaming positive sentiment words |
| Multimodal demonstrations | Robot navigating around a tree left or right |
| Bellman equation | Playing Pac-Man with a routine strategy |
| Discounting | Rs. 100 now vs Rs. 100 five years later |
| Markov property | Chess board configuration, cannonball physics |

### How Math is Motivated Before Being Shown

Math is never dropped in cold. The pattern is:

1. **Pose a question:** *"But how do we calculate the above probability?"*
2. **Build intuition with words/diagrams:** Verbal explanation + visual
3. **Transition to math:** *"Let us do a bit of mathematics."* / *"The mathematical representation for this can be written as:"*
4. **Show the equation**
5. **Interpret it immediately:** *"Here, G(t) is the total return..."* / *"This means that..."*

Example from Policy Gradient article:
> *"We are going to do a small derivation right now through which you will understand how this gradient is calculated. We are going to perform this derivation in multiple steps."*

Then proceeds with: Step 1, Step 2, Step 3... Step 7, each with a single equation and a verbal explanation.

### Numerical Worked Examples for Every Formula

**CRITICAL RULE:** Whenever a formula, equation, or mathematical approach is introduced, it MUST be followed by a simple numerical worked example. Students learn best when they can plug in concrete numbers and trace through the computation.

The pattern is:
1. Show the formula
2. Immediately say: *"Let us plug in some simple numbers to see how this works."*
3. Assign small, easy-to-follow values to each variable (e.g., 2, 3, 0.5)
4. Walk through the computation step by step
5. Interpret the result: *"This tells us that..."*

Example pattern:
> The return is calculated as: $$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}$$
>
> Let us plug in some simple numbers. Suppose our rewards are $r_t = 1$, $r_{t+1} = 2$, $r_{t+2} = 3$, and $\gamma = 0.9$:
>
> $$G_t = 1 + 0.9 \times 2 + 0.81 \times 3 = 1 + 1.8 + 2.43 = 5.23$$
>
> This tells us that our total discounted return from time step $t$ is 5.23. Notice how the later rewards contribute less due to discounting — this is exactly what we want.

This applies to ALL formulas — action tokenization formulas, loss functions, probability distributions, update rules, etc. No formula should appear without a concrete numerical walkthrough.

### Depth of Mathematical Derivations

**Moderate depth, maximum clarity.** Derivations are:
- Broken into labeled, bite-sized steps (Step 1, Step 2, etc.)
- Each step has one equation + one sentence of explanation
- Complex proofs are referenced to external resources rather than fully derived:
  - *"I will include a book in the resources section where you can find the derivation..."*
  - *"For further reading, please refer to the book: The Principles of Diffusion Models..."*
  - *"You can refer to my notes on Generalized Advantage Estimation here:"*

### How Code Relates to Math

**Code always follows math.** The pattern is:

```
Mathematical derivation → "Let us implement this practically" → Code → Results/Visualization
```

Code is presented as the practical realization of the preceding math:

- *"Here you can see that we are using the exact same Q-Learning formula which we had seen before."* — Three Horsemen article
- *"In the above step, you can see that we are calculating the log probabilities for the actions, multiplying them by the Q-value, and then summing it up for all the states in the episode. This is exactly what the REINFORCE algorithm says."* — Policy Gradient article

### Use of Intuition-Building Before Formalism

Extremely strong. Typical pattern:
1. Real-world analogy
2. Visual diagram
3. Informal verbal explanation
4. *Then* the equation

Example from Score Matching article — introducing Langevin dynamics:
> *"Imagine that you are dropped into a thick fog on a vast landscape. Your goal is to find the deepest valley because that is where the treasure is hidden."*

Then: walking strategy → slope concept → gradient descent analogy → formal equation.

---

## 4. Visual & Formatting Patterns

### When and Why Figures are Used

Figures appear **before and after explanations** to anchor understanding. They serve multiple purposes:

1. **Concept diagrams:** Agent-environment interface, encoder-decoder architecture, policy iteration flow
2. **Step-by-step process illustrations:** Forward diffusion pipeline, RLHF training flow, backup diagrams
3. **Result visualizations:** Training reward curves, latent space plots, generated samples
4. **Real-world photos:** Chess players, robots, gazelle calves, coffee latte art — to ground analogies
5. **Animated GIFs:** Diffusion process, decoder sampling, Langevin dynamics trajectories
6. **Comparison diagrams:** Monte Carlo vs TD methods, REINFORCE vs baseline

Figures are placed **immediately after** the verbal description they illustrate, usually preceded by:
- *"Have a look at the diagram below:"*
- *"This can be shown in the following figure:"*
- *"Let us look at a simple visualization to understand this better."*
- *"The schematic below explains..."*

### Figure Caption Style

Figures generally **do not have formal captions.** Instead, the surrounding text acts as the caption — the figure is introduced by the preceding sentence and interpreted by the following sentence. Occasionally, figures include alt-text descriptions sourced from the web (e.g., *"GamesCrafters :: Games"*, *"Pick And Place Robot, For Industrial at ₹ 550000/number in Ahmedabad"*).

### Code Block Placement

Code blocks appear in **dedicated "Practical Implementation" sections**, usually in the second half of an article. They are:
- Preceded by a setup sentence: *"The following is the piece of code which we will use..."*
- Followed by a line-by-line explanation
- Sometimes preceded by installation instructions: *"pip install gymnasium[box2d] - That's All!"*

### Use of Bold, Italics, Blockquotes

- **Bold:** Used for key terms on first introduction (*"This is called **exploitation**"*), section headers, and emphasis
- **Italics:** Rarely used standalone — mostly in inline emphasis
- **Blockquotes/Quotes:** Used for important principles or key insights:
  - *"In the field of reinforcement learning, the central role of value estimation is the most important thing that researchers learned from 1960 to 1990"*
  - *"Point-estimate policies typically fail to learn multimodal targets..."*
  - *"The robot needs visual data from a camera to perceive the world around it."*

### Bullet/List Usage

- **Numbered lists** for: steps in algorithms, enumerated elements (4 elements of RL, 3 problems with RL for robotics), ordered processes
- **Bulleted lists** for: properties, characteristics, action/observation descriptions, advantages/disadvantages
- Lists are **short** — typically 2–6 items, each 1–2 lines

---

## 5. Signature Phrases & Patterns

### Recurring Transitional Phrases

| Phrase | Frequency |
|---|---|
| *"Let us start by/with..."* | Almost every article opening |
| *"Let us look at..."* / *"Let us understand..."* | Very frequent within sections |
| *"Now let us..."* | Very frequent for transitions |
| *"This brings us to..."* | Frequent for major topic transitions |
| *"Now the question is..."* / *"The main question is..."* | Frequent — drives narrative forward |
| *"Have a look at..."* | Frequent — introduces figures |
| *"This is exactly what we want."* | Signature validation phrase — appears in nearly every article |
| *"This makes sense because..."* / *"This does not make sense."* | Frequent — validates/invalidates approaches |
| *"Let us see how..."* | Frequent — transitions to proof or implementation |

### How Questions are Posed to the Reader

Questions are a **primary narrative device** — used to create suspense, guide thinking, and transition between sections:

- **Rhetorical/guiding:** *"But how do we choose actions that are the right actions in every state?"*
- **Direct challenge:** *"Can you tell me what is wrong in the about diagram? What am I missing?"*
- **Recall prompt:** *"Does this remind you of something?"*
- **Thinking prompt:** *"You might be thinking that..."* / *"A thought might come to your mind which says..."*
- **Setup for next section:** *"So, what is the practical solution?"* / *"So then, how do we select the correct sample?"*
- **Connecting to prior knowledge:** *"Can we look at the Bellman equation and understand how we can use it to find an optimal policy?"*
- **Engagement:** *"How would you write the rewards and returns for this problem?"*

### Opening and Closing Patterns

**Openings:**
- "Let us start by/with..." (dominant)
- "Let us take a simple example" (common variant)
- Brief definition → immediate example (alternate)

**Closings:**
- "That's it!" (concise articles)
- "Thanks!" (friendly sign-off)
- GitHub/Colab link + resource recommendations

### Catchphrases and Distinctive Expressions

- **"This is exactly what we want."** — The single most characteristic phrase. Appears after validating a result, loss function, or approach. Used as a confirmation stamp.
- **"Not bad right?"** — After showing results that exceed expectations
- **"Awesome!"** — Expressing genuine excitement about results
- **"Brace yourselves!"** — Before complex mathematical sections
- **"Okay, enough mathematics!"** — Transitioning away from derivations
- **"Believe it or not..."** — Introducing surprising connections
- **"If only it was this easy..."** / **"Well, not quite."** — Setting up complications
- **"Our 'drunk hiker'..."** — Recurring playful character for Langevin dynamics
- **"This is truly amazing."** — Expressing wonder at research contributions
- **"What is common in all these examples?"** — Recurring synthesis question after multiple examples

---

## 6. Code Style

### Primary Language

**Python exclusively.** All code examples are in Python.

### Libraries Used

- **PyTorch** (nn.Module, torch.Tensor, F.softmax, F.log_softmax) — for neural network implementations
- **NumPy** — for numerical operations
- **OpenAI Gymnasium** — for RL environments
- **Standard library typing** — for type hints (tt.List[float])

### Comment Density and Style

**Moderate comment density.** Comments are:
- Explanatory, not redundant: `# Calculate discounted returns (working backwards)`
- Action-oriented: `# Sample an action from the probability distribution`
- Section-marking: `# --- Action Selection ---`, `# --- Environment Step ---`, `# --- Loss Calculation and Optimization ---`
- Occasionally conversational: `# .item() extracts the value from the tensor`

### Pedagogical vs Production-Ready

**Strongly pedagogical.** Code is:
- Minimal — focused on illustrating the core algorithm, not edge cases
- Linear/sequential — easy to follow top-to-bottom
- Self-contained — each snippet demonstrates one concept
- Accompanied by detailed prose explanation before and after
- Often followed by "Let us understand this code in detail" with a walkthrough

Example of pedagogical code with walkthrough:
```python
import gymnasium as gym

# Create the environment
env = gym.make("LunarLander-v3", render_mode="human")

env.reset()

for step in range(200):
    env.render()
    env.step(env.action_space.sample())

env.close()
```
Followed by: *"Let us understand this code in detail. The first step is the creation of our environment using the gym.make( ) function. Next we have env.reset( ), which is used to reset the episode..."*

### Import Patterns

- Standard top-of-block imports
- Only essential libraries imported
- No complex dependency management or configuration

### Variable Naming

- **Descriptive:** `episode_rewards`, `batch_states`, `batch_actions`, `returns_sum`, `returns_count`, `sample_action`, `sample_observation`
- **Mathematical convention preserved:** `G` for return, `Q` for Q-values, `GAMMA` for discount factor, `ALPHA` for step size, `EPSILON` for exploration rate
- **Constants in CAPS:** `GAMMA`, `ALPHA`, `EPSILON`
- **Snake_case** for function and variable names: `calc_qvals`, `state_t`, `next_state_discrete`

### Code-to-Explanation Ratio

For every code block, there is typically **2–3x more explanation text** than code. The code is a launching point for discussion, not a standalone artifact.

---

## 7. Overall Narrative Strategy (Meta-Pattern)

The writing follows a consistent **"Guided Discovery"** pedagogy:

1. **Hook with a concrete scenario** the reader can visualize
2. **Build intuition** through analogy, diagrams, and informal language
3. **Formalize** with mathematics — step by step, never in bulk
4. **Validate** with "This is exactly what we want" or similar confirmation
5. **Complicate** — introduce a limitation or question ("But wait... this doesn't work because...")
6. **Resolve** with the next technique/concept
7. **Implement** in code with results
8. **Wrap up** with a concise summary and key takeaways

This creates a narrative arc that mirrors a detective story: each section poses a problem, builds toward a solution, and then reveals a new layer of complexity that motivates the next section. Each article is **self-contained** — all necessary background is explained within the article itself.

---

## 8. Standalone Article Rule

Each article must be **fully self-contained**. The reader should never need to have read any other article to understand the current one. When a concept depends on prerequisite knowledge:

- **Explain it inline** with a brief, intuitive recap — enough for the reader to follow along
- **Do NOT** reference "the previous article," "as we saw last time," or "in the next lecture"
- **Do NOT** use cliffhangers that point to a future article
- If a related topic exists elsewhere, provide a brief self-contained explanation rather than linking to another article

The goal is that any reader can pick up any single article and walk away with a complete understanding of the topic.

---

## Summary: The Vizuara Voice in One Paragraph

The Vizuara style is that of an **enthusiastic, patient tutor** who believes that every complex concept has an intuitive core waiting to be unlocked. Articles open with concrete, often playful examples (handwriting, Batman images, coffee latte art, drunk hikers), then progressively layer in mathematical formalism — always step by step, always interpreted immediately. The dominant voice is first-person plural ("Let us..."), creating a collaborative journey. Questions drive the narrative forward, functioning as both transitions and engagement hooks. The signature validation phrase "This is exactly what we want" appears like a recurring drumbeat of confirmation. Code is pedagogical and always follows theory, never precedes it. Each article is **fully self-contained** — all prerequisite knowledge is explained inline so any reader can pick it up independently. The overall effect is that of a lecture you actually want to attend — clear, progressive, visual, and genuinely excited about the elegance of the ideas being taught.
