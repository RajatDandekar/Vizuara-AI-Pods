# Building a Reasoning Model from Scratch
## How to teach a small language model to think step-by-step using reinforcement learning with verifiable rewards

---

Let us start with a simple example. Suppose you ask a language model the following math question:

> "A bakery sells 3 types of bread. Each type costs \$4. If you buy 2 of each type, and you have a \$5 coupon, how much do you pay?"

A standard language model might immediately respond with:

> "You pay \$19."

That is incorrect. The model jumped to an answer without working through the problem. Now, let us see how a reasoning model handles the same question:

> *\<think\> Let me break this down. There are 3 types of bread, each costs \$4. I buy 2 of each type, so I buy 2 x 3 = 6 loaves total. The total cost is 6 x \$4 = \$24. I have a \$5 coupon, so I pay \$24 - \$5 = \$19. Wait, let me re-check: 3 types, 2 of each = 6 loaves. 6 x 4 = 24. 24 - 5 = 19. Yes, \$19. \</think\>*
>
> **The answer is \$19.**

Interesting. In this case both happen to arrive at the same number, but notice the critical difference. The reasoning model showed its work, verified the computation, and double-checked itself. For harder problems, this step-by-step thinking is what separates a correct answer from a wrong one.


![Base model gives an instant guess; the reasoning model thinks step by step.](figures/figure_1.png)
*Base model gives an instant guess; the reasoning model thinks step by step.*


So, how do we teach a model to produce these intermediate reasoning steps? How do we go from a standard language model to one that thinks before it answers?

This brings us to the core topic of this article.

---

## What Does "Reasoning" Mean for a Language Model?

In cognitive science, there is a well-known distinction between two modes of thinking:

**System 1** is fast, automatic, and intuitive. When you see "2 + 2", you immediately know the answer is 4. There is no deliberation — the answer just appears.

**System 2** is slow, deliberate, and effortful. When you see "What is 17 x 23?", you do not instantly know the answer. You need to break it down: 17 x 20 = 340, 17 x 3 = 51, total = 391.

A standard language model operates mostly in System 1 mode. It generates the next token based on pattern matching from its training data. It does not "think through" the problem.

A reasoning model, on the other hand, operates in System 2 mode. It generates a chain of intermediate tokens — called **reasoning tokens** or a **chain-of-thought** — before arriving at the final answer. These tokens represent the model "thinking out loud."


![System 1 gives fast answers. System 2 reasons through problems step by step.](figures/figure_2.png)
*System 1 gives fast answers. System 2 reasons through problems step by step.*


The key insight is this: reasoning is not something that emerges from pre-training alone. It is a behavior that must be **taught** through a deliberate training process. The question is: what does that training process look like?

---

## The Training Pipeline — Three Stages

The recipe for building a reasoning model follows three stages:

**Stage 1: Supervised Fine-Tuning (SFT) on Chain-of-Thought Data.** We take a pre-trained language model and fine-tune it on examples that include step-by-step reasoning traces. This teaches the model the *format* of reasoning.

**Stage 2: Reinforcement Learning with Verifiable Rewards.** We use RL to let the model discover its own reasoning strategies. The reward comes from checking whether the final answer is correct. This teaches the model the *quality* of reasoning.

**Stage 3: Rejection Sampling and Distillation.** We use the trained large model to generate many solutions, keep only the correct ones, and use these to train a smaller model. This makes reasoning efficient and accessible.


![Three-stage pipeline: format first, quality second, efficiency third.](figures/figure_3.png)
*Three-stage pipeline: format first, quality second, efficiency third.*


Let us now dive into each stage, starting with supervised fine-tuning.

---

## Stage 1: Supervised Fine-Tuning on Chain-of-Thought Data

The first step is to teach the model what a reasoning trace looks like. We do this by fine-tuning on examples that include explicit chain-of-thought steps wrapped in special tags.

The data format looks like this:

**Prompt:** "What is 15% of 80?"

**Target completion:**
```
<think>
To find 15% of 80, I need to multiply 80 by 0.15.
80 x 0.15 = 12.
Let me verify: 10% of 80 is 8, and 5% of 80 is 4. So 15% = 8 + 4 = 12. Correct.
</think>

The answer is 12.
```

The `<think>` and `</think>` tags mark the boundary of the reasoning trace. Everything inside these tags is the chain-of-thought. Everything after is the final answer.


![SFT teaches the model to generate the full reasoning-then-answer sequence.](figures/figure_4.png)
*SFT teaches the model to generate the full reasoning-then-answer sequence.*


The training objective is standard next-token prediction. We compute the cross-entropy loss over the entire sequence — including the reasoning tokens.


$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x)$$

Here, $x$ is the prompt, $y_t$ is the token at position $t$, and $T$ is the total number of tokens in the completion (including both the reasoning trace and the final answer).

Let us plug in some simple numbers to see how this works. Suppose our completion has 4 tokens, and the model assigns the following probabilities to each correct token:

| Token | Correct Token | Model Probability |
|-------|---------------|-------------------|
| $y_1$ | "To" | 0.8 |
| $y_2$ | "find" | 0.7 |
| $y_3$ | "15%" | 0.5 |
| $y_4$ | "of" | 0.9 |

The loss is:

$$\mathcal{L} = -[\log(0.8) + \log(0.7) + \log(0.5) + \log(0.9)]$$
$$= -[-0.223 + (-0.357) + (-0.693) + (-0.105)]$$
$$= -[-1.378] = 1.378$$

This tells us that the average per-token loss is $1.378 / 4 = 0.345$. As the model gets better at predicting the reasoning tokens, this loss will decrease. This is exactly what we want.

Here is a minimal code snippet for the SFT training step:

```python
import torch
import torch.nn.functional as F

def sft_training_step(model, tokenizer, prompt, completion, optimizer):
    """One step of supervised fine-tuning on a chain-of-thought example."""
    # Tokenize the full sequence: prompt + completion
    full_text = prompt + completion
    tokens = tokenizer(full_text, return_tensors="pt").input_ids

    # Forward pass: get logits for every position
    logits = model(tokens[:, :-1]).logits  # shape: (1, seq_len-1, vocab_size)
    targets = tokens[:, 1:]                # shift targets by 1

    # Cross-entropy loss over all tokens
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

Let us understand this code. The key idea is simple: we feed the entire sequence (prompt + reasoning + answer) into the model, and we train the model to predict each token given all previous tokens. The cross-entropy loss measures how far the model's predictions are from the actual tokens.

But here is the important question: **is SFT enough?**

The answer is no. SFT teaches the model the *format* of reasoning — it learns to produce tokens inside `<think>` tags. But it does not necessarily teach the model to reason *well*. The model might produce plausible-looking but incorrect reasoning chains. It might take unnecessary detours or skip critical steps.

To improve the *quality* of reasoning, we need reinforcement learning. This brings us to Stage 2.

---

## Stage 2: Reinforcement Learning with Verifiable Rewards

This is where the real magic happens.

The key insight from the DeepSeek-R1 paper is that we can use reinforcement learning to teach a model to reason, using a beautifully simple reward signal: **check if the final answer is correct.**

For math problems, this is straightforward. If the model says the answer is 19 and the ground truth is 19, the reward is 1. If the model says 24, the reward is 0. There is no ambiguity, no need for human annotators, no reward model to train. We call these **verifiable rewards** because the correctness can be verified automatically.

Let us now understand the RL algorithm that makes this work: **Group Relative Policy Optimization (GRPO)**.

### GRPO: The Core Algorithm

In standard policy gradient methods like PPO, you need a separate value network (critic) to estimate the baseline and compute advantages. This doubles the memory requirement and adds complexity.

GRPO, introduced in the DeepSeek-Math paper, eliminates the critic entirely. The idea is elegant:

1. For each prompt, sample **G completions** from the current policy.
2. Compute the reward for each completion.
3. Use the **group statistics** (mean and standard deviation of the rewards) as the baseline.
4. The advantage of each completion is simply how much better or worse it is relative to the group.

The group-relative advantage is computed as:


$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

Here, $r_i$ is the reward for the $i$-th completion, and $\mathbf{r} = [r_1, r_2, \ldots, r_G]$ is the vector of all rewards in the group.

Let us plug in some simple numbers. Suppose we sample $G = 4$ completions for a math problem, and the rewards are:

$$\mathbf{r} = [1, 0, 1, 0]$$

The mean is $\text{mean}(\mathbf{r}) = 0.5$, and the standard deviation is $\text{std}(\mathbf{r}) = 0.5$.

The advantages are:

$$\hat{A}_1 = \frac{1 - 0.5}{0.5} = 1.0, \quad \hat{A}_2 = \frac{0 - 0.5}{0.5} = -1.0$$
$$\hat{A}_3 = \frac{1 - 0.5}{0.5} = 1.0, \quad \hat{A}_4 = \frac{0 - 0.5}{0.5} = -1.0$$

This tells us that completions 1 and 3 (which got the right answer) have positive advantages, and completions 2 and 4 (which got the wrong answer) have negative advantages. The policy update will increase the probability of generating completions like 1 and 3, and decrease the probability of generating completions like 2 and 4. This is exactly what we want.


![GRPO: sample a group, reward each, and update relative to the group.](figures/figure_5.png)
*GRPO: sample a group, reward each, and update relative to the group.*


Now, the GRPO objective uses a **clipped surrogate** loss, similar to PPO. This prevents the policy from changing too drastically in a single update:


$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \min\!\left(r_i(\theta)\,\hat{A}_i,\;\; \text{clip}\!\left(r_i(\theta),\, 1-\epsilon,\, 1+\epsilon\right)\hat{A}_i\right)$$

Here, $r_i(\theta) = \frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}$ is the probability ratio between the current and old policies, and $\epsilon$ is the clipping parameter (typically 0.2).

Let us work through a concrete example. Suppose for one completion:
- Probability ratio $r(\theta) = 1.5$ (the current policy is 50% more likely to generate this completion)
- Advantage $\hat{A} = 1.0$ (this was a good completion)
- Clipping parameter $\epsilon = 0.2$

The unclipped objective is: $r(\theta) \times \hat{A} = 1.5 \times 1.0 = 1.5$

The clipped ratio is: $\text{clip}(1.5, 0.8, 1.2) = 1.2$

The clipped objective is: $1.2 \times 1.0 = 1.2$

We take the minimum: $\min(1.5, 1.2) = 1.2$

The clipping kicks in here because the ratio moved too far from 1.0. This prevents the policy from making overly aggressive updates towards this completion, even though it was good. This is the same "do not change your running style too much" idea from PPO.


![Clipping prevents the policy from moving too far in one update.](figures/figure_6.png)
*Clipping prevents the policy from moving too far in one update.*


Here is the core GRPO computation in PyTorch:

```python
import torch

def grpo_loss(log_probs, old_log_probs, rewards, epsilon=0.2):
    """
    Compute GRPO loss for a group of completions.

    Args:
        log_probs: Log probabilities under current policy, shape (G,)
        old_log_probs: Log probabilities under old policy, shape (G,)
        rewards: Rewards for each completion, shape (G,)
        epsilon: Clipping parameter
    """
    # Step 1: Compute group-relative advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Step 2: Compute probability ratios
    ratios = torch.exp(log_probs - old_log_probs)

    # Step 3: Clipped surrogate objective
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    loss = -torch.min(unclipped, clipped).mean()

    return loss
```

Let us understand this code. In Step 1, we normalize the rewards to get the advantages — completions above the group mean get positive advantages, below get negative. In Step 2, we compute how much the current policy differs from the old policy for each completion. In Step 3, we apply the clipped surrogate objective. The `torch.min` ensures we take the more conservative estimate, preventing large policy updates.

---

## The KL Divergence Penalty

There is one more crucial ingredient. During RL training, the model's policy will gradually drift away from the initial reference policy. If the drift is too large, the model might "forget" useful behaviors it learned during pre-training and SFT, or it might find degenerate strategies that exploit the reward signal.

To prevent this, we add a KL divergence penalty that measures how far the current policy has moved from the reference policy:


$$R_{\text{total}} = R_{\text{outcome}} - \beta \cdot D_{\text{KL}}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right)$$

Here, $R_{\text{outcome}}$ is the reward from checking the answer, $\beta$ is a weight that controls the strength of the penalty, and $D_{\text{KL}}$ measures the divergence between the current policy $\pi_\theta$ and the frozen reference policy $\pi_{\text{ref}}$.

Let us plug in some simple numbers. Suppose for a given completion:
- The outcome reward is $R_{\text{outcome}} = 1.0$ (correct answer)
- The KL divergence is $D_{\text{KL}} = 2.0$ (the policy has drifted significantly)
- The penalty weight is $\beta = 0.1$

The total reward is:

$$R_{\text{total}} = 1.0 - 0.1 \times 2.0 = 1.0 - 0.2 = 0.8$$

The KL penalty reduced the effective reward from 1.0 to 0.8. This tells the model: "Yes, you got the right answer, but you drifted quite far from your original behavior. The reward is slightly lower to discourage excessive drift."

Now consider a much larger drift with $D_{\text{KL}} = 15.0$:

$$R_{\text{total}} = 1.0 - 0.1 \times 15.0 = 1.0 - 1.5 = -0.5$$

Even though the model got the correct answer, the total reward is negative because the drift is too extreme. This forces the model to find reasoning strategies that are both correct *and* stay close to the reference policy. This is exactly what we want.


![The KL penalty prevents the model from deviating too far from its original behavior.](figures/figure_7.png)
*The KL penalty prevents the model from deviating too far from its original behavior.*


---

## Stage 3: Rejection Sampling and Distillation

After the RL training phase, we have a large model that reasons well. But large models are expensive to run. Can we transfer this reasoning ability to a smaller model?

This is where rejection sampling and distillation come in. The process is straightforward:

1. **Generate:** Take the large, RL-trained reasoning model and generate many solutions (say, 64) for each problem in the training set.
2. **Filter:** Keep only the solutions where the final answer is correct. This gives us a curated dataset of high-quality reasoning traces.
3. **Distill:** Fine-tune a smaller model (using standard SFT) on this curated dataset.

The result is a smaller model that has learned the reasoning patterns of the larger model. DeepSeek found that distilling from their 671B parameter model to a 7B parameter model produced remarkably strong reasoning performance — in some cases outperforming much larger models that were trained with RL directly.


![Generate many solutions, keep the correct ones, train a smaller model.](figures/figure_8.png)
*Generate many solutions, keep the correct ones, train a smaller model.*


This is a powerful finding. It suggests that the *data generated by RL training* might be even more valuable than the RL training procedure itself. A well-curated dataset of reasoning traces can go a long way.

---

## Practical Implementation — Putting It All Together

Let us now look at how all of these pieces come together in a practical training pipeline. We will use a small language model and train it on the GSM8K math reasoning dataset.

Here is the complete training pipeline:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Small but capable base model
DATASET = "gsm8k"                    # Grade-school math reasoning
GROUP_SIZE = 4                        # Number of completions per prompt
EPSILON = 0.2                         # Clipping parameter
BETA = 0.1                            # KL penalty weight
LR = 1e-6                             # Learning rate

# --- Load model and reference ---
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
ref_model.eval()  # Freeze the reference model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def extract_answer(completion):
    """Extract the numerical answer from a completion."""
    import re
    match = re.search(r'(?:answer is|####)\s*(\-?[\d,]+)', completion)
    return match.group(1).replace(',', '') if match else None

def compute_reward(completion, ground_truth):
    """Compute reward: 1 if correct, 0 otherwise."""
    predicted = extract_answer(completion)
    return 1.0 if predicted == str(ground_truth) else 0.0

def grpo_training_step(model, ref_model, prompt, ground_truth, optimizer):
    """One GRPO training step for a single prompt."""
    # Step 1: Generate G completions
    completions = []
    for _ in range(GROUP_SIZE):
        output = model.generate(
            tokenizer(prompt, return_tensors="pt").input_ids,
            max_new_tokens=512, do_sample=True, temperature=0.7
        )
        completions.append(tokenizer.decode(output[0], skip_special_tokens=True))

    # Step 2: Compute rewards
    rewards = torch.tensor([compute_reward(c, ground_truth) for c in completions])

    # Step 3: Compute group-relative advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Step 4: Compute loss (simplified)
    loss = compute_grpo_loss(model, ref_model, completions, advantages, EPSILON, BETA)

    # Step 5: Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), rewards.mean().item()
```

Let us understand the flow. For each prompt from GSM8K, we generate 4 completions from the current model. We check each completion against the ground truth answer. We compute advantages — correct completions get positive advantages, incorrect ones get negative. Then we update the policy using the clipped surrogate loss with the KL penalty.

Over the course of training, we observe something remarkable:


![Training reward rises sharply as the model discovers effective reasoning strategies.](figures/figure_9.png)
*Training reward rises sharply as the model discovers effective reasoning strategies.*


The reward curve tells a clear story. Initially, the model barely gets any answers right — it is generating random reasoning traces. Then, around step 200-800, there is a rapid improvement phase where the model discovers that step-by-step reasoning actually leads to correct answers. After that, it continues to refine its reasoning strategies more slowly.

---

## Emergent Behaviors — What RL Discovers

Perhaps the most fascinating aspect of training reasoning models with RL is what the model learns to do *on its own*. We never explicitly teach these behaviors — they emerge because they help the model get more rewards.

**Self-verification.** The model learns to check its own work:

> *"...so the total is 3 x 7 = 21. Let me verify: 7 + 7 + 7 = 21. Yes, that is correct."*

**Backtracking.** The model learns to catch and correct its own mistakes:

> *"...the area is length x width = 5 x 3 = 18. Wait, 5 x 3 = 15, not 18. Let me recalculate. The area is 15 square meters."*

**Extended thinking for hard problems.** The model naturally produces longer reasoning chains for more difficult problems. A simple addition might get 2-3 reasoning steps, while a multi-step word problem might get 10-15 steps.


![RL training produces emergent behaviors: self-check, backtrack, and scale thinking.](figures/figure_10.png)
*RL training produces emergent behaviors: self-check, backtrack, and scale thinking.*


These emergent behaviors are remarkable because they mirror how human experts solve problems. An experienced mathematician does not just compute — they verify, they backtrack when something feels wrong, and they invest more effort on harder problems.

The model was never told to do any of this. It discovered these strategies purely because they lead to higher rewards. This is the power of reinforcement learning with verifiable rewards — by defining *what* success looks like (correct answers), we let the model figure out *how* to get there.

---

## Conclusion

Let us summarize the complete pipeline for building a reasoning model from scratch:

1. **Start with a pre-trained language model** that already has world knowledge and language capabilities.
2. **Supervised fine-tuning** on chain-of-thought data teaches the model the *format* of reasoning — how to produce `<think>` traces.
3. **Reinforcement learning with GRPO** teaches the model the *quality* of reasoning — using verifiable rewards (correct/incorrect) and group-relative baselines.
4. **Rejection sampling and distillation** transfers reasoning ability from large models to small ones efficiently.

The key insight is this: we do not teach the model *how* to reason. We simply reward it for getting the right answer, and it discovers reasoning strategies on its own. The chain-of-thought, the self-verification, the backtracking — all of it emerges from the RL training process.

This is the approach behind DeepSeek-R1, and it represents one of the most exciting developments in modern AI. The recipe is surprisingly simple: a good base model, verifiable rewards, and a well-designed RL algorithm.

For further reading, refer to the original DeepSeek-R1 paper: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025). The GSM8K dataset is available at https://github.com/openai/grade-school-math.

Here is the link to our Google Colab notebooks where you can build this entire pipeline from scratch: [Vizuara Teaching Notebooks].

That's it!

---

## References

1. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025.
2. DeepSeek-AI. "DeepSeek-Math: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024.
3. Cobbe et al. "Training Verifiers to Solve Math Word Problems." (GSM8K) 2021.
4. Schulman et al. "Proximal Policy Optimization Algorithms." 2017.
5. Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." 2022.
6. Kahneman, D. "Thinking, Fast and Slow." Farrar, Straus and Giroux. 2011.
