# Tiny Recursive Models: Less is More for AI Reasoning

How a 7-million parameter network outperforms trillion-parameter LLMs on reasoning benchmarks — by thinking again, and again, and again.

---

## The Sudoku Insight

Let us start with a simple example. Imagine that you are solving a very hard Sudoku puzzle — one of those extreme-difficulty grids where only 17 numbers are given out of 81 cells.

How do you approach it?

You probably do not sit down, stare at the grid once, and fill in every single cell from left to right. That would be impossible. Instead, you do something much smarter: you **scan** the grid, fill in the few cells where you are confident, and then **go back to the beginning** and scan again.

And here is the magical part — on the second pass, things that were invisible before suddenly become obvious. That "3" you placed in row 2 now eliminates possibilities in column 5, which reveals a "7" in box 6, which constrains row 8. Each pass through the grid reveals new information that was hidden in the previous pass.

This is **recursive reasoning**: repeatedly applying the same thinking process to the same problem, where each application builds on the insights from the last.


![Single-pass models guess once; recursive models refine their answer with each pass.](figures/figure_1.png)
*Single-pass models guess once; recursive models refine their answer with each pass.*


Now, here is an interesting question. Modern large language models — GPT-4, Gemini, Claude — are essentially **single-pass** machines. They process the input once through their layers and produce an answer. They do not go back, re-examine, and refine.

What if, instead of making our models bigger, we made them **think longer**?

This is the core idea behind Tiny Recursive Models (TRM), a remarkable piece of research by Alexia Jolicoeur-Martineau (2025). And the results are stunning.

---

## The Scaling Wall

Before we dive into how TRM works, let us understand the problem it solves.

There is a benchmark called **ARC-AGI** (Abstraction and Reasoning Corpus for Artificial General Intelligence). It was designed to test whether AI systems can perform the kind of abstract reasoning that comes naturally to humans.

Each ARC-AGI task gives you a few input-output examples of grid transformations, and you must figure out the underlying rule and apply it to a new input. The grids contain colored cells, and the transformations involve patterns like rotations, reflections, counting, symmetry, and object manipulation.

These tasks are tricky even for humans. But they are designed to test something fundamental: can you **generalize** from a few examples?

Now, the current approach in AI is to throw massive models at such problems. Deepseek R1, o3-mini, Gemini 2.5 Pro — these models have hundreds of billions (sometimes trillions) of parameters. They use chain-of-thought prompting, massive test-time compute budgets, and enormous training datasets.

And yet, on ARC-AGI-1, the best LLM (Gemini 2.5 Pro) achieves only about 37% accuracy.

Here comes the surprise.

A model with **7 million parameters** — that is 0.01% the size of these frontier LLMs — achieves **44.6%** on ARC-AGI-1 and **7.8%** on ARC-AGI-2, outperforming all of them.

Not bad right?


![TRM with 7M parameters outperforms LLMs with 100B+ parameters on ARC-AGI-1.](figures/figure_2.png)
*TRM with 7M parameters outperforms LLMs with 100B+ parameters on ARC-AGI-1.*


How is this possible? What does this tiny model know that the giants do not?

The answer is not about what it knows — it is about **how it thinks**.

---

## The Predecessor: Hierarchical Reasoning Models

Before we look at TRM, let us understand its predecessor: the **Hierarchical Reasoning Model (HRM)**.

HRM was inspired by biological systems. In neuroscience, there is evidence that the brain processes information at multiple timescales — some neural circuits operate at high frequency (fast, reactive processing) and others at low frequency (slow, deliberate reasoning). Think of it as the difference between your reflexes and your deep thinking.

HRM translated this idea into a neural network architecture with **two separate networks**:

1. A **high-frequency network** that processes information rapidly and updates at every step
2. A **low-frequency network** that operates more slowly and provides higher-level guidance

These two networks recursed at different rates, creating a hierarchical reasoning loop. With only 27 million parameters, HRM achieved impressive results — it could solve Sudoku puzzles, navigate mazes, and handle ARC-AGI tasks far better than you would expect from a model of its size.


![HRM uses two separate networks recursing at different frequencies — 27M parameters total.](figures/figure_3.png)
*HRM uses two separate networks recursing at different frequencies — 27M parameters total.*


But HRM had three significant problems:

**Problem 1: Shaky mathematical foundations.** HRM invoked the Implicit Function Theorem to justify training with only 1-step gradient approximations. This theorem assumes that the recursion converges to a fixed point — but with only 4 recursion steps, this convergence was unlikely to actually hold.

**Problem 2: Computational waste.** HRM used a Q-learning mechanism for deciding when to stop recursing. This required **two forward passes** per optimization step — one for the actual reasoning and one for the Q-learning update. This doubled the computational cost.

**Problem 3: Unjustified complexity.** The biological motivation was elegant, but when researchers ran ablation studies, it was unclear how much each component actually contributed. Was the hierarchy truly necessary, or was it just adding complexity without proportional benefit?

This naturally brings us to the question: can we keep the power of recursive reasoning while stripping away all the unnecessary complexity?

---

## Enter the Tiny Recursive Model

The Tiny Recursive Model (TRM) answers this question with a resounding yes — and the simplification is dramatic.

Instead of two networks with 4 layers each, TRM uses **one network** with just **2 layers** and **7 million parameters**. That is a 75% reduction in parameters and a massive reduction in architectural complexity.

But the real insight is not just about making the model smaller. It is about understanding **what actually matters** in recursive reasoning.

TRM maintains two features that flow through the recursion loop:

1. **y** — the current embedded solution. At any point during recursion, you can decode y (via argmax) to get the model's current best answer.

2. **z** — a latent reasoning feature. This is like a **scratchpad** or internal chain-of-thought. It holds all the intermediate reasoning that the model needs but that is not directly part of the answer.

Let us go back to our Sudoku analogy. Think of **y** as your current best guess for every cell in the grid. And think of **z** as all the mental notes you keep in your head — "this row already has a 3 and a 7, so this cell cannot be 3 or 7." You cannot see these notes in the final answer, but they are essential for getting there.

The recursion loop is beautifully simple:


$$z \leftarrow \text{net}(x, y, z) \quad \text{(update reasoning — repeat } n \text{ times)}$$

$$y \leftarrow \text{net}(y, z) \quad \text{(refine answer)}$$

That is it. The same tiny 2-layer network is applied repeatedly. First, it updates the latent reasoning z by looking at the input x, the current solution y, and the previous reasoning state z. This happens n times (typically n = 6). Then, it refines the answer y using the updated reasoning z.


![TRM: one tiny 2-layer network processes input (x), solution (y), and reasoning (z) in a loop.](figures/figure_4.png)
*TRM: one tiny 2-layer network processes input (x), solution (y), and reasoning (z) in a loop.*


Let us plug in some simple numbers to build intuition. Imagine a tiny 3×3 grid puzzle where the goal is to fill in missing values.

Suppose the input grid is:

| 1 | ? | 3 |
|---|---|---|
| ? | 2 | ? |
| 3 | ? | 1 |

Initially, y (the solution) is all zeros for the missing cells, and z (the reasoning) is also initialized to zeros.

**Pass 1:** The network sees the input x and the blank solution y. It notices that row 1 has 1 and 3, so position (1,2) must be 2. The reasoning z gets updated with constraints: "row 1 needs 2, column 2 has 2 in row 2." The solution y gets partially updated: position (1,2) → 2.

**Pass 2:** Now the network sees the updated y with the 2 filled in. This new information propagates — z updates with "column 2 now has 2 in rows 1 and 2, so row 3 column 2 must be something else." More cells get resolved.

**Pass 3:** With even more cells filled, the remaining unknowns become trivially solvable. The solution y converges to the correct answer.

This is recursive reasoning in action — each pass makes the next pass easier.


![Each recursion pass fills in more of the puzzle as the reasoning state z accumulates constraints.](figures/figure_5.png)
*Each recursion pass fills in more of the puzzle as the reasoning state z accumulates constraints.*


---

## Deep Supervision: The Training Secret

Having a recursive architecture is one thing. Training it effectively is another.

The naive approach would be: run all the recursions, check the final answer, compute the loss, and backpropagate. But this has a serious problem — if the model has to do 18 recursions (3 supervision steps × 6 recursions each), the gradient has to flow through all 18 steps. This makes training unstable and slow.

TRM uses a clever trick called **deep supervision**. Instead of only checking the final answer, it checks the answer at **multiple intermediate points** during the recursion.

Here is how it works. Let us say we have T = 3 supervision steps, each containing n = 6 recursion iterations:

**Supervision Step 1:**
- Run 5 recursions **without** accumulating gradients (let the model reason freely)
- Run 1 recursion **with** gradients (this is the step we learn from)
- Compare the current prediction to the ground truth and compute the loss
- Backpropagate and update the model

**Supervision Step 2:**
- Same thing: 5 free recursions + 1 gradient recursion
- Compare prediction to ground truth again
- Backpropagate again

**Supervision Step 3:**
- Same pattern one more time
- By now, the model has had 18 total recursions (3 × 6) and 3 gradient updates

The analogy here is perfect: it is like a teacher who checks your Sudoku progress every few minutes, not just grading the final submission. If you are going in the wrong direction at minute 5, you get corrected immediately rather than wasting the next 20 minutes going further down the wrong path.


![Deep supervision checks predictions at 3 intermediate points. Gray circles = free recursions, orange = gradient step.](figures/figure_6.png)
*Deep supervision checks predictions at 3 intermediate points. Gray circles = free recursions, orange = gradient step.*


The loss function has two components. The first is a standard **softmax cross-entropy loss** that measures how close the prediction is to the ground truth:


$$\mathcal{L}_{\text{pred}} = -\sum_{i} y_i^{\text{true}} \log(\hat{y}_i)$$

Let us plug in some simple numbers. Suppose we have 3 possible values for a cell (1, 2, or 3). The true label is class 2, so $y^{\text{true}} = [0, 1, 0]$. The model predicts probabilities $\hat{y} = [0.1, 0.7, 0.2]$.

$$\mathcal{L}_{\text{pred}} = -(0 \times \log(0.1) + 1 \times \log(0.7) + 0 \times \log(0.2)) = -\log(0.7) = 0.357$$

This tells us the model is doing reasonably well (probability 0.7 for the correct class) but can still improve.

The second component is a **halting loss** — a simplified version of Adaptive Computation Time (ACT). It is a binary cross-entropy that encourages the model to learn when it has already found the correct answer:


$$\mathcal{L}_{\text{halt}} = -[q \log(\hat{q}) + (1-q) \log(1-\hat{q})]$$

Here, $q = 1$ if the prediction matches the ground truth, and $q = 0$ otherwise. The model outputs $\hat{q}$, its own estimate of whether it has the right answer.

If the model correctly predicts $q = 1$ (meaning "I am done, my answer is correct") and $\hat{q} = 0.9$:

$$\mathcal{L}_{\text{halt}} = -[1 \times \log(0.9) + 0 \times \log(0.1)] = -\log(0.9) = 0.105$$

The total loss is simply the sum: $\mathcal{L} = \mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{halt}} = 0.357 + 0.105 = 0.462$.

When the halting signal $\hat{q} > 0$, the model is allowed to stop early during training — no need to keep recursing on a sample it has already solved. This saves computation and prevents the model from wasting time on easy examples.

---

## Inside the Network

Now let us look at what is inside this tiny 2-layer network.

Each layer uses a combination of modern techniques that have proven effective in transformer architectures:

1. **RMSNorm** — a normalization technique that stabilizes activations without the overhead of full layer normalization
2. **SwiGLU activation** — a gated activation function that has become standard in modern language models (used in LLaMA, Gemini, etc.)
3. **Rotary position embeddings** — these encode spatial position information directly into the attention mechanism, which is critical for grid-based reasoning tasks

The model comes in two variants:

**MLP variant** (for small, fixed-size contexts like 9×9 Sudoku): Instead of self-attention, it uses a simple matrix multiplication of size [L, L] where L is the sequence length (81 for Sudoku). Since the context size is fixed and small, a linear layer is cheaper and more effective than attention.

**Attention variant** (for large, variable-size contexts like 30×30 grids): This uses standard self-attention, which excels when the context length L is much larger than the hidden dimension D.


![MLP variant (left) for small fixed grids; Attention variant (right) for large variable contexts.](figures/figure_7.png)
*MLP variant (left) for small fixed grids; Attention variant (right) for large variable contexts.*


Here is a remarkable number: with T = 3 supervision steps, n = 6 recursions per step, and 2 layers per recursion, the model achieves an **effective depth of 42 layers** per supervision step.


$$\text{Effective depth} = T \times (n + 1) \times n_{\text{layers}} = 3 \times 7 \times 2 = 42$$

Let us think about what this means. A standard 42-layer transformer would require tens or hundreds of millions of parameters — each layer having its own unique weights. TRM achieves the same depth of processing with only 2 layers of weights, applied 21 times. The parameters are **shared across all recursions**.

This is the essence of "Less is More": fewer parameters, more computation.

---

## Results: David vs Goliath

Now let us look at the results, and this is where things get truly remarkable.

TRM was evaluated on four benchmarks of increasing difficulty:

**Sudoku-Extreme:** These are the hardest possible Sudoku puzzles — trained on only 1,000 examples and tested on 423,000. The TRM (MLP variant, 5M parameters) achieves **87.4%** accuracy. The previous HRM model (27M parameters) managed only 55.0%. And LLMs? Deepseek R1, Claude, o3 — they all score **0.0%**. Not a single puzzle solved correctly.

**Maze-Hard:** Navigating 30×30 mazes where the shortest path is over 110 steps long. TRM (attention variant, 19M parameters) scores **85.3%** vs HRM's 74.5%. LLMs again: **0.0%**.

**ARC-AGI-1:** The abstract reasoning benchmark. TRM (7M parameters) achieves **44.6%**, beating HRM (40.3%) and the best LLM, Gemini 2.5 Pro (37.0%).

**ARC-AGI-2:** The harder version. TRM scores **7.8%**, beating HRM (5.0%) and Gemini 2.5 Pro (4.9%).


![TRM dominates across all four benchmarks. LLMs score 0% on Sudoku-Extreme and Maze-Hard.](figures/figure_8.png)
*TRM dominates across all four benchmarks. LLMs score 0% on Sudoku-Extreme and Maze-Hard.*


Let us pause and appreciate what these numbers mean.

On Sudoku-Extreme and Maze-Hard, frontier LLMs with hundreds of billions of parameters — models that cost millions of dollars to train — cannot solve a single instance. A 5-million parameter model trained on a **single GPU in under 36 hours** solves 87.4% of them.

This is not a marginal improvement. This is a fundamentally different capability.

The training efficiency is also worth noting:

- Sudoku: less than 36 hours on a single L40S GPU
- Mazes: less than 24 hours on 4 L40S GPUs
- ARC-AGI: approximately 3 days on 4 H100 GPUs

These are modest compute budgets by today's standards. This is exactly what we want — strong performance from efficient models.

---

## What Matters and What Doesn't

One of the most valuable contributions of the TRM paper is a thorough ablation study that reveals which design choices actually matter. Let us go through the key findings on the Sudoku-Extreme benchmark.

**Full backpropagation is the single most important factor.** When the authors switched from HRM's 1-step gradient approximation to full backpropagation through the recursion (enabled by deep supervision), accuracy jumped from 56.5% to 87.4% — a gain of **+30.9 percentage points**. This is the difference between a mediocre model and a great one.

**Fewer layers with more recursion beats more layers with fewer recursions.** Using 2 layers with more recursion depth outperformed using 4 layers with fewer recursions by **+7.9 percentage points**. This is counterintuitive — we are used to thinking that more layers means more capacity means better performance. But on small datasets, the extra parameters lead to overfitting. Recursive depth, by contrast, adds computational depth without adding parameters.

**MLP beats attention on small fixed contexts.** For the 9×9 Sudoku grid (81 tokens), replacing self-attention with a simple MLP improved accuracy by **+13.0 percentage points**. The lesson: use the right tool for the right job. Self-attention is powerful but brings overhead that is unnecessary when the context size is fixed and small.

**EMA (Exponential Moving Average) stabilizes training.** Maintaining an exponential moving average of the model weights with a decay of 0.999 improved accuracy by **+7.5 percentage points**. This is especially important when training on very small datasets (only 1,000 examples), where training can be noisy and unstable.

**Two features (y + z) are essential.** When the authors removed the latent reasoning feature z and forced the model to encode everything in a single feature, generalization degraded. Both features — the explicit solution y and the implicit reasoning scratchpad z — are necessary.


![Full backpropagation through recursion is the single most impactful design choice (+30.9%).](figures/figure_9.png)
*Full backpropagation through recursion is the single most impactful design choice (+30.9%).*


The authors also documented what **did not** work:

- **Mixture of Experts:** Too much capacity led to overfitting — the model memorized the training set instead of learning to reason
- **Fixed-point iteration:** Theoretically appealing but slowed training in practice
- **Weight-tied embeddings:** Too constraining for the model to learn effectively
- **Removing the latent feature z:** Forced the model to encode all reasoning directly in the solution, which hurt generalization

---

## The Big Picture: Parameters vs Compute

Now let us step back and ask a deeper question: **why does recursion work so well?**

The fundamental insight is that recursion trades **parameters for compute**. Instead of encoding all the reasoning ability into a massive set of unique weights (the LLM approach), TRM applies a small set of shared weights many times.

This mirrors how humans solve puzzles. You do not have billions of neurons dedicated specifically to Sudoku. You use the **same reasoning circuits** — constraint propagation, elimination, pattern matching — and apply them repeatedly. Your brain is a recursive reasoner.


![The LLM approach uses many parameters in one pass; TRM uses few parameters in many passes.](figures/figure_10.png)
*The LLM approach uses many parameters in one pass; TRM uses few parameters in many passes.*


This connects to a broader pattern emerging across AI research:

- **Diffusion models** generate images through iterative refinement — starting from noise and repeatedly denoising
- **Chain-of-thought prompting** encourages LLMs to reason step by step rather than jumping to an answer
- **System 2 thinking** (from Kahneman's framework) represents slow, deliberate, iterative reasoning as opposed to fast, reflexive System 1

TRM provides perhaps the cleanest evidence yet that for reasoning tasks, **how long you think matters more than how big your brain is**.

The paper leaves us with some fascinating open questions. Why does recursion fundamentally outperform deeper networks? Is there a theoretical explanation for why shared weights applied many times generalize better than unique weights applied once? And can this recursive approach be extended beyond classification to generative tasks like text and image generation?

---

## Conclusion

Let us take stock of what we have learned.

The Tiny Recursive Model achieves state-of-the-art reasoning performance with **0.01%** of the parameters of frontier LLMs. It does this through three key ideas:

1. **Recursive architecture:** One tiny network applied many times, maintaining a solution feature y and a reasoning scratchpad z
2. **Deep supervision:** Checking and correcting predictions at multiple intermediate points during recursion
3. **Simplicity over complexity:** 2 layers instead of 4, one network instead of two, MLP instead of attention where appropriate

The results speak for themselves: 87.4% on extreme Sudoku (LLMs: 0%), 85.3% on hard mazes (LLMs: 0%), and 44.6% on ARC-AGI-1 (best LLM: 37%) — all from a model that fits on a single GPU and trains in hours rather than weeks.

The lesson is profound and counterintuitive: **Less is More.** Recursive depth beats parameter count. Thinking longer beats thinking bigger.

The next time you are stuck on a hard problem, remember — you do not need a bigger brain. You just need to think again.

See you next time!

---

## References

- Jolicoeur-Martineau, A., "Less is More: Recursive Reasoning with Tiny Networks" (2025), arXiv:2510.04871
- Jolicoeur-Martineau, A., "Hierarchical Reasoning Model" (2024)
- Chollet, F., "On the Measure of Intelligence" (2019) — introduces the ARC benchmark
- Graves, A., "Adaptive Computation Time for Recurrent Neural Networks" (2016)
