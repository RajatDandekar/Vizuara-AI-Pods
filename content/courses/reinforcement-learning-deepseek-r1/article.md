# Reinforcement Learning in DeepSeek-R1 — How Pure RL Taught a Language Model to Think

## Understanding GRPO, rule-based rewards, and the "aha moment" from first principles

---

Let us start with a simple example. Imagine you are a teacher and you want your students to solve math problems — not just give the final answer, but show their entire reasoning process.

One approach is to hand them a textbook full of perfectly worked solutions. "Study these examples," you say. "Learn how to write your reasoning step by step." The students read the textbook, memorize the patterns, and start writing solutions that *look* like the worked examples. This is essentially what **Supervised Fine-Tuning (SFT)** does — you show the model thousands of human-written reasoning traces and say: "Write like this."

But here is the problem. The students are copying the *style* of reasoning without truly understanding *how* to reason. They can mimic the format, but when they encounter a genuinely new problem, they struggle.

Now consider a second approach. You give the students a stack of practice problems — and an answer key. No worked solutions, no step-by-step guides. Just: "Here is the problem. Here is the correct answer. Figure out how to get there on your own."

At first, the students flounder. Their reasoning is messy, disorganized, sometimes wrong. But over hundreds of practice problems, something remarkable happens — they start developing their own reasoning strategies. They learn to break problems into steps, to double-check their work, even to pause mid-solution and say, "Wait, let me reconsider this approach."

This is the core insight behind **DeepSeek-R1**: you do not need to teach a language model *how* to reason by showing it reasoning examples. You can teach it to reason purely through **Reinforcement Learning** — by letting it practice, checking its answers, and rewarding it when it gets them right.


![Figure 1](figures/figure_1.png)

**Figure 1:** The fundamental difference between teaching reasoning through SFT (left) and pure RL (right). SFT copies existing reasoning patterns; pure RL lets the model discover its own.

---

## Quick Refresher: RL for Language Models

Before we dive into what makes DeepSeek-R1 special, let us quickly revisit how Reinforcement Learning is applied to language models. If you have read our previous article on RLHF, this will be familiar.

In the RL framework for language models:

- **State:** The sequence of tokens generated so far
- **Action:** The next token to generate
- **Policy:** The LLM itself — it outputs probabilities over the vocabulary for the next token
- **Reward:** A score measuring how good the complete response is

The standard RLHF pipeline involves **four** separate models:

1. **Policy Model** — the LLM being trained
2. **Reference Model** — a frozen copy of the original LLM (to prevent the policy from drifting too far)
3. **Reward Model** — a separate neural network trained on human preferences to score completions
4. **Value Model (Critic)** — estimates the expected future reward for each state, used to compute advantages

The policy is updated using **Proximal Policy Optimization (PPO)**, which uses the advantage estimates from the value model to determine which actions to reinforce.

The policy gradient objective looks as follows:


$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} \log \pi_\theta(a_t | s_t) \cdot A_t \right]
$$


Let us plug in some simple numbers to see how this works. Suppose we have a completion with 3 tokens, and the log probabilities and advantages are:

- Token 1: $\log \pi = -0.5$, advantage $A_1 = 2.0$
- Token 2: $\log \pi = -1.0$, advantage $A_2 = -0.5$
- Token 3: $\log \pi = -0.3$, advantage $A_3 = 1.5$

The objective becomes:

$$J = (-0.5)(2.0) + (-1.0)(-0.5) + (-0.3)(1.5) = -1.0 + 0.5 - 0.45 = -0.95$$

The gradient will push the policy to increase the probability of Token 1 (positive advantage) and Token 3 (positive advantage), while decreasing the probability of Token 2 (negative advantage). This is exactly what we want — reinforce good actions, penalize bad ones.


![Figure 2](figures/figure_2.png)

**Figure 2:** The standard RLHF pipeline requires four separate models. This is computationally expensive — you need to load and run all four models during training.

Now the question is — can we make this simpler? Can we get rid of one of these models? This brings us to DeepSeek-R1.

---

## What Makes DeepSeek-R1 Different?

DeepSeek-R1 introduced two key innovations that simplify the entire pipeline:

**Innovation 1: No SFT on reasoning traces.** Instead of collecting thousands of human-written chain-of-thought examples, DeepSeek-R1 lets the model discover reasoning through pure RL. The model is given math and logic problems, and is rewarded only for getting the correct final answer.

**Innovation 2: GRPO instead of PPO.** DeepSeek-R1 replaces PPO with a new algorithm called **Group Relative Policy Optimization (GRPO)**. The key difference? GRPO completely eliminates the value model. Instead of training a separate critic network to estimate advantages, GRPO computes advantages by comparing multiple completions against each other.

This means DeepSeek-R1 needs only **three** models instead of four:

1. **Policy Model** — the LLM being trained
2. **Reference Model** — frozen copy for KL constraint
3. **Reward Function** — rule-based (not even a learned model!)


![Figure 3](figures/figure_3.png)

**Figure 3:** DeepSeek-R1 eliminates the value model entirely, reducing the training infrastructure from four models to three.

This is a significant simplification. Training a value model is expensive — it has the same size as the policy model, and it needs to be updated alongside the policy. By removing it, GRPO cuts memory requirements and training complexity substantially.

But how do we estimate the advantage without a value model? This brings us to the heart of GRPO.

---

## Group Relative Policy Optimization (GRPO) — The Core Idea

Let us understand GRPO with an analogy.

Imagine you are a teacher grading a class quiz. You have two options for assigning grades:

**Option A (PPO approach):** You hire an expert who has studied thousands of quizzes and can predict, for any student's partial answer, what their final score will be. This expert watches each student's progress and says: "Based on what I have seen so far, I predict this student will score 7 out of 10." You use these predictions to decide which students to encourage and which to correct.

**Option B (GRPO approach):** You skip the expert entirely. Instead, you give the same quiz to a group of students. When the results come in, you simply rank them relative to each other. The student who scored highest gets the most encouragement. The student who scored lowest gets the most correction. The group average becomes the natural baseline.

GRPO takes Option B. Let us walk through the algorithm step by step.

**Step 1: Sample a group of completions.** For a given prompt $q$, generate $G$ different completions from the current policy $\pi_{\theta_{old}}$. Let us say $G = 5$.

**Step 2: Score each completion.** Use a reward function to assign a reward $r_i$ to each completion.

**Step 3: Normalize rewards within the group.** Compute the mean $\mu_G$ and standard deviation $\sigma_G$ of the rewards, and normalize:


$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}
$$

This normalized reward serves as the advantage estimate. No value model needed.

Let us plug in some simple numbers. Suppose we sample $G = 5$ completions for a math problem, and their rewards are:

$$\mathbf{r} = [0.2, \; 0.8, \; 0.5, \; 0.1, \; 0.9]$$

First, compute the mean: $\mu_G = \frac{0.2 + 0.8 + 0.5 + 0.1 + 0.9}{5} = \frac{2.5}{5} = 0.5$

Next, compute the standard deviation: $\sigma_G = \sqrt{\frac{(0.2-0.5)^2 + (0.8-0.5)^2 + (0.5-0.5)^2 + (0.1-0.5)^2 + (0.9-0.5)^2}{5}} = \sqrt{\frac{0.09 + 0.09 + 0 + 0.16 + 0.16}{5}} = \sqrt{0.1} \approx 0.316$

Now normalize each reward:

- $\hat{A}_1 = \frac{0.2 - 0.5}{0.316} = -0.95$ — this completion is below average, discourage it
- $\hat{A}_2 = \frac{0.8 - 0.5}{0.316} = +0.95$ — above average, encourage it
- $\hat{A}_3 = \frac{0.5 - 0.5}{0.316} = 0.0$ — exactly average, neutral
- $\hat{A}_4 = \frac{0.1 - 0.5}{0.316} = -1.27$ — worst in the group, strong discouragement
- $\hat{A}_5 = \frac{0.9 - 0.5}{0.316} = +1.27$ — best in the group, strong encouragement

This is exactly what we want. The best completion in the group gets the strongest positive signal, the worst gets the strongest negative signal, and the average completion gets zero gradient.

**Step 4: Update the policy.** Use the normalized advantages in a clipped objective (similar to PPO) with a KL penalty:


$$
J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q)} \; \mathbb{E}_{\{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \; \frac{1}{G} \sum_{i=1}^{G} \left[ \min\!\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A}_i, \;\; \text{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)},\; 1{-}\varepsilon,\; 1{+}\varepsilon\right) \hat{A}_i \right) - \beta \; D_{KL}\!\left(\pi_\theta \| \pi_{ref}\right) \right]
$$

This looks intimidating, but let us break it down term by term:

- $\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$ — the **probability ratio** between the new and old policy. If the new policy makes a completion more likely, this ratio is greater than 1.
- $\text{clip}(\cdot, 1-\varepsilon, 1+\varepsilon)$ — the **clipping** mechanism from PPO. It prevents the ratio from moving too far from 1, keeping updates conservative. Typically $\varepsilon = 0.2$.
- $\hat{A}_i$ — the **group-normalized advantage** we just computed.
- $\beta \; D_{KL}(\pi_\theta \| \pi_{ref})$ — the **KL penalty** that prevents the policy from drifting too far from the reference model.

The $\min$ of the clipped and unclipped terms ensures that the policy never gets too large an update, even when the advantage is very high. If you remember our PPO discussion from the RLHF article — the runner with the coach who says "don't change your style too much" — this is the same idea.


![Figure 4](figures/figure_4.png)

**Figure 4:** The GRPO workflow. For each prompt, sample G completions, score them, normalize rewards within the group to get advantages, and update the policy.

---

## Why GRPO Works — The Intuition

You might be thinking: "This is just REINFORCE with baseline, right?" You are not far off. Let us see the connection.

Remember from our Policy Gradient article that the REINFORCE with baseline method subtracts a baseline $b(s)$ from the return to reduce variance:


$$
\nabla J = \mathbb{E}\!\left[\nabla \log \pi_\theta(a|s) \;\cdot\; (R - b) \right]
$$


In GRPO, the baseline is the **mean reward of the group**: $b = \mu_G$. Subtracting the group mean is exactly the baseline trick.

But GRPO goes one step further — it also divides by the standard deviation $\sigma_G$. This is **variance normalization**, and it has a powerful effect on training stability.

Let us see why with a numerical example. Suppose we have two different prompts:

- **Prompt A** (easy): rewards are $[0.9, 0.95, 0.85, 0.92, 0.88]$ — all completions score high
- **Prompt B** (hard): rewards are $[0.1, 0.05, 0.15, 0.02, 0.08]$ — all completions score low

Without normalization (just subtracting the mean), the advantages for Prompt A would be tiny (since all rewards are close to the mean), while the advantages for Prompt B would also be tiny. The gradient signals would be weak for both.

With GRPO's normalization (subtract mean AND divide by std):

- **Prompt A:** mean = 0.9, std = 0.037. The completion with reward 0.95 gets advantage $\frac{0.95 - 0.9}{0.037} = +1.35$
- **Prompt B:** mean = 0.08, std = 0.047. The completion with reward 0.15 gets advantage $\frac{0.15 - 0.08}{0.047} = +1.49$

Now the best completion in each group gets a strong gradient signal, regardless of the absolute difficulty level. The model learns equally well from easy and hard prompts. This is exactly what we want.

The key insight is this: by generating multiple completions per prompt, the model compares **its own attempts against each other** rather than against some external value function's estimate. The group itself becomes the baseline, and no separate critic is needed.

---

## Rule-Based Rewards — No Reward Model Needed

Now we understand how the policy is updated. But we still need to answer: where do the rewards come from?

In standard RLHF, a **learned reward model** scores completions. This reward model is trained on human preference data — pairs of completions where humans indicate which one is better. Training a good reward model is expensive, requires large amounts of human annotation, and the model can be "gamed" (the policy learns to exploit quirks in the reward model rather than actually improving).

DeepSeek-R1 takes a radically simpler approach for reasoning tasks: **rule-based verifiable rewards**. No learned reward model at all.

The total reward for each completion has two components:

**1. Accuracy Reward:** Is the final answer correct?

For math problems, this is straightforward — extract the final numerical answer from the completion and compare it to the ground truth. If correct, $r_{accuracy} = 1$. If incorrect, $r_{accuracy} = 0$.

For code problems, run the generated code against test cases. If all test cases pass, $r_{accuracy} = 1$.

**2. Format Reward:** Did the model structure its response properly?

DeepSeek-R1 requires the model to put its reasoning inside `<think>...</think>` tags and its final answer outside. If the completion follows this format, $r_{format} = 1$. Otherwise, $r_{format} = 0$.

The total reward is simply:


$$
r_i = r_{accuracy}(o_i) + r_{format}(o_i)
$$


Let us work through a concrete example. Suppose we sample $G = 5$ completions for the prompt "What is 17 × 23?", where the correct answer is 391:

| Completion | Answer Given | Correct? | Has `<think>` tags? | $r_{accuracy}$ | $r_{format}$ | $r_{total}$ |
|---|---|---|---|---|---|---|
| $o_1$ | 391 | Yes | Yes | 1 | 1 | 2 |
| $o_2$ | 391 | Yes | No | 1 | 0 | 1 |
| $o_3$ | 389 | No | Yes | 0 | 1 | 1 |
| $o_4$ | 391 | Yes | Yes | 1 | 1 | 2 |
| $o_5$ | 400 | No | No | 0 | 0 | 0 |

Rewards: $\mathbf{r} = [2, 1, 1, 2, 0]$. Mean: $\mu_G = 1.2$, std: $\sigma_G \approx 0.75$.

Normalized advantages:
- $\hat{A}_1 = \frac{2 - 1.2}{0.75} = +1.07$ — correct answer, good format — strong encouragement
- $\hat{A}_2 = \frac{1 - 1.2}{0.75} = -0.27$ — correct but missing format — slight discouragement
- $\hat{A}_3 = \frac{1 - 1.2}{0.75} = -0.27$ — wrong answer but good format — slight discouragement
- $\hat{A}_4 = \frac{2 - 1.2}{0.75} = +1.07$ — correct and formatted — strong encouragement
- $\hat{A}_5 = \frac{0 - 1.2}{0.75} = -1.60$ — wrong answer, no format — strong discouragement

The model learns that getting the right answer AND using the proper format is the best strategy. This is exactly what we want.


![Figure 5](figures/figure_5.png)

**Figure 5:** The rule-based reward system in DeepSeek-R1. Two simple checks — accuracy and format — replace the need for a learned reward model.

Why does this work so well? Because math, code, and logic puzzles have **objectively verifiable answers**. You do not need a neural network to tell you whether $17 \times 23 = 391$. A simple string comparison suffices. This eliminates an entire model from the training pipeline and removes the risk of reward hacking.

---

## The "Aha Moment" — Emergent Reasoning

Now we come to perhaps the most fascinating finding of the DeepSeek-R1 paper.

During RL training — with no supervised examples of reasoning, no human-written chain-of-thought, just a reward signal for correct answers — the model spontaneously develops sophisticated reasoning behaviors.

Early in training, the model's responses are terse and often wrong. It guesses at answers without showing any work. But as RL training progresses, something remarkable emerges:

- The model starts **breaking problems into steps**
- It begins **allocating more thinking tokens to harder problems** — spending more time on difficult questions
- It learns to **try alternative approaches** when the first attempt fails
- Most strikingly, it develops the habit of **self-correction** — pausing mid-solution to reconsider

The DeepSeek team documented a now-famous example they call the **"aha moment."** During training, the model produced a response that included something like:

> "Wait, let me reconsider. I made an error in my previous step. Let me re-examine this from the beginning..."

Nobody taught the model to do this. There was no training example that said "when you are stuck, reconsider your approach." The model discovered this strategy on its own, purely because reconsidering and correcting errors leads to more correct final answers — which means higher rewards.

This is like our student analogy from the beginning. A student who practices enough problems with an answer key will naturally develop the habit of double-checking their work — not because a teacher told them to, but because they learned that checking prevents mistakes.


![Figure 6](figures/figure_6.png)

**Figure 6:** The "aha moment" — during pure RL training, the model spontaneously learns to pause, reconsider, and self-correct its reasoning. This behavior emerges without any supervised examples.

This emergent reasoning capability is what makes DeepSeek-R1 so significant. It suggests that the ability to reason is not something that needs to be *taught* through examples — it can be *discovered* through reinforcement.

---

## The Full DeepSeek-R1 Training Pipeline

You might be wondering — if pure RL is so powerful, why not just do RL from the start and call it a day?

The DeepSeek team actually tried this (they call it DeepSeek-R1-Zero). It works surprisingly well for math and logic, but it has problems: the model sometimes mixes languages mid-response, produces hard-to-read outputs, and struggles with general tasks like writing or open-ended QA.

To address these issues, the full DeepSeek-R1 uses a **four-stage training pipeline**:

**Stage 1: Cold Start (Small-Scale SFT)**

A small number of high-quality chain-of-thought examples (a few thousand) are used to teach the model the basic `<think>...</think>` format. This is not about teaching reasoning — it is about teaching the model the *structure* of a response.

**Stage 2: Reasoning-Oriented RL**

This is the main event. Large-scale GRPO training on thousands of math, code, science, and logic problems. The model learns to reason through pure RL with rule-based rewards. This is where the "aha moments" happen.

**Stage 3: Rejection Sampling + SFT**

The RL-trained model generates many candidate solutions. The best ones (verified correct) are kept and combined with general-purpose data (writing, QA, translation). The model is fine-tuned on this curated dataset. This stage fixes readability issues and adds general capabilities.

**Stage 4: All-Scenario RL**

A final round of RL covering both reasoning tasks AND general tasks. This stage uses both rule-based rewards (for reasoning) and a learned reward model (for general helpfulness and harmlessness).


![Figure 7](figures/figure_7.png)

**Figure 7:** The four-stage DeepSeek-R1 training pipeline. Stage 2 (Reasoning RL) is where the core reasoning capability develops.

Why four stages? The answer is balance. Pure RL produces powerful reasoners but poor communicators. SFT produces fluent writers but shallow thinkers. The multi-stage pipeline combines the best of both approaches — Stage 2 builds deep reasoning, and Stages 3–4 refine it into a model that is both smart and readable.

---

## Practical Implementation — GRPO in Code

Enough theory — let us look at a practical implementation. We will write a simplified version of the GRPO training loop in PyTorch.

**Step 1: Sampling G completions from the policy**

```python
import torch
import torch.nn.functional as F

def sample_completions(model, tokenizer, prompt, num_samples=5, max_length=256):
    """
    Sample G completions from the current policy for a given prompt.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    completions = []

    for _ in range(num_samples):
        output = model.generate(
            input_ids,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=1.0,
            top_k=50
        )
        completion_text = tokenizer.decode(output[0], skip_special_tokens=True)
        completions.append(completion_text)

    return completions
```

Let us understand this code. For a given prompt, we call `model.generate` with `do_sample=True` so that each call produces a different completion. We repeat this $G$ times (here `num_samples=5`) to get our group of completions. The `temperature=1.0` ensures we get diverse samples.

**Step 2: Computing group-normalized advantages**

```python
def compute_group_advantages(rewards):
    """
    Normalize rewards within the group to get advantage estimates.
    This replaces the value model in standard PPO.
    """
    rewards = torch.tensor(rewards, dtype=torch.float32)

    # Group statistics
    mean_reward = rewards.mean()
    std_reward = rewards.std()

    # Avoid division by zero
    if std_reward < 1e-8:
        return torch.zeros_like(rewards)

    # Normalize: subtract mean, divide by std
    advantages = (rewards - mean_reward) / std_reward

    return advantages
```

This is the heart of GRPO. Notice how simple it is — just two lines of actual computation. We subtract the group mean (the baseline trick from REINFORCE) and divide by the standard deviation (variance normalization). No neural network, no training loop, no gradient computation. This tiny function replaces an entire value model.

**Step 3: Computing the GRPO loss**

```python
def grpo_loss(policy_model, ref_model, prompts, completions, advantages,
              epsilon=0.2, beta=0.01):
    """
    Compute the GRPO loss with clipping and KL penalty.
    """
    total_loss = 0.0

    for prompt, completion, advantage in zip(prompts, completions, advantages):
        # Log probabilities under current and old policy
        log_prob_new = get_log_prob(policy_model, prompt, completion)
        log_prob_old = get_log_prob(policy_model, prompt, completion).detach()
        log_prob_ref = get_log_prob(ref_model, prompt, completion)

        # Probability ratio
        ratio = torch.exp(log_prob_new - log_prob_old)

        # Clipped objective (same as PPO)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        policy_loss = -torch.min(unclipped, clipped)

        # KL penalty against reference model
        kl_penalty = beta * (log_prob_new - log_prob_ref)

        total_loss += policy_loss + kl_penalty

    return total_loss / len(prompts)
```

Let us walk through this code. For each prompt-completion pair, we compute the ratio of the new policy probability to the old policy probability. If this ratio is too far from 1 (meaning the policy changed too much), the clipping mechanism kicks in and limits the update. The KL penalty term prevents the policy from drifting too far from the reference model.

Here you can see that we are using the exact same clipping mechanism from PPO. The only difference is that the advantage $\hat{A}_i$ comes from group normalization rather than a value model. This is the elegance of GRPO — it keeps the stability benefits of PPO while removing the need for a critic.

---

## Results and Impact

Does this approach actually work? The results speak for themselves.

DeepSeek-R1 achieves performance comparable to OpenAI's o1 model on major reasoning benchmarks:

- **AIME 2024** (competition math): DeepSeek-R1 scores **79.8%** pass@1, comparable to OpenAI o1's 79.2%
- **MATH-500** (math problem solving): DeepSeek-R1 achieves **97.3%**, matching o1's 96.4%
- **Codeforces** (competitive programming): DeepSeek-R1 reaches a rating of **2,029**, in the 96.3rd percentile

Perhaps even more impressively, the DeepSeek team showed that the reasoning capability can be **distilled** into smaller models. By generating reasoning traces from the full R1 model and fine-tuning smaller models (1.5B to 70B parameters) on these traces, even tiny models gain significant reasoning ability. The distilled 14B model outperforms many larger models on math benchmarks.


![Figure 8](figures/figure_8.png)

**Figure 8:** DeepSeek-R1 matches or exceeds OpenAI o1 on major reasoning benchmarks, despite using a simpler training approach.

---

## Conclusion

Let us recap what we have learned. DeepSeek-R1 demonstrated three powerful ideas:

1. **Pure RL can teach reasoning.** You do not need expensive human-annotated reasoning traces. Given enough practice problems and a simple reward signal, an LLM can discover how to reason on its own.

2. **GRPO simplifies the infrastructure.** By computing advantages through group normalization instead of a value model, GRPO cuts the number of models from four to three — reducing memory, compute, and engineering complexity.

3. **Rule-based rewards work beautifully for verifiable tasks.** When you can check whether an answer is correct with a simple comparison, you do not need a learned reward model. This eliminates reward hacking and makes the training pipeline more robust.

The combination of these ideas produced a model that matches the state-of-the-art in reasoning while being conceptually simpler and more efficient to train.

But here is an open question: what about tasks where we *cannot* easily verify the answer? Creative writing, open-ended reasoning, ethical dilemmas — these do not have a simple ground truth to check against. How do we apply RL to improve LLMs on these tasks? We will explore this in a future article.

---

**References:**

- DeepSeek-AI et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992)
