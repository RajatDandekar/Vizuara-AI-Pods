# Group-Relative Policy Optimization (GRPO) — From Scratch

*How DeepSeek eliminated the critic network and made RLHF simpler, cheaper, and better*

Let us start with a simple example. Imagine that you are a teacher who has just received 30 essays from your students on the topic "Why is the sky blue?" Now you need to grade them.

One approach is to assign an absolute score to each essay. You read an essay and say, "This deserves a 7 out of 10." But this is hard — what does a 7 really mean? You need a detailed rubric, and even then, you might not be consistent across all 30 essays.

There is a much easier approach. You read all 30 essays, and then you rank them relative to each other. You say, "This essay is above average compared to the group," or "This one is below average." You do not need an absolute scale — you just need to compare within the group.

This is the core idea behind Group-Relative Policy Optimization (GRPO). Instead of training a separate neural network (a "critic") to predict how good each response is on an absolute scale, GRPO generates multiple responses to the same prompt and compares them relative to each other. The group average becomes the baseline.

But before we dive into GRPO, let us first understand why it was needed. This brings us to the standard approach for training language models with reinforcement learning.


![Relative scoring within a group is simpler than absolute scoring.](figures/figure_1.png)
*Relative scoring within a group is simpler than absolute scoring.*


## Quick Recap: RLHF and the Role of the Critic

Let us start by understanding how Reinforcement Learning is applied to language models.

When we frame a language model as an RL agent, the mapping is straightforward:

- **State**: The prompt plus all tokens generated so far
- **Action**: The next token to generate
- **Policy**: The language model itself — it outputs a probability distribution over the vocabulary for the next token
- **Reward**: A score that tells us how good the complete response is

The standard pipeline for training language models with human feedback (RLHF) has three stages:

1. **Supervised Fine-Tuning (SFT)**: Train the model on high-quality demonstrations
2. **Reward Model Training**: Train a separate model to predict human preferences
3. **RL Fine-Tuning**: Use the reward model to provide rewards, and optimize the policy using PPO

The RL fine-tuning step using PPO is where things get interesting — and expensive.

PPO (Proximal Policy Optimization) works by collecting responses from the current policy, scoring them with the reward model, and then updating the policy to increase the probability of good responses while decreasing the probability of bad ones. The key mechanism is the clipped surrogate objective:


$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

Here, $r_t(\theta)$ is the ratio of the new policy probability to the old policy probability for a given token. The advantage $\hat{A}_t$ tells us how much better this action was compared to what we expected. The clipping ensures that we do not make updates that are too large.

Let us plug in some simple numbers to see how this works. Suppose $r_t(\theta) = 1.3$ (meaning the new policy is 30% more likely to produce this token), $\hat{A}_t = 2.0$ (the action was much better than expected), and $\epsilon = 0.2$:

- Unclipped term: $1.3 \times 2.0 = 2.6$
- Clipped ratio: $\text{clip}(1.3, 0.8, 1.2) = 1.2$
- Clipped term: $1.2 \times 2.0 = 2.4$
- Final: $\min(2.6, 2.4) = 2.4$

The clipping brought the objective down from 2.6 to 2.4, preventing an overly aggressive update. This is exactly what we want.

But here is the critical part: to compute the advantage $\hat{A}_t$, PPO uses **Generalized Advantage Estimation (GAE)**, which requires a learned **value function** (also called the **critic**):


$$
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \text{where} \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$


This value function $V(s)$ is a separate neural network that predicts the expected cumulative reward from a given state. For language models, this critic network has the same architecture as the LLM itself, just with a scalar output head instead of vocabulary logits.

Let us plug in some numbers here too. Suppose $\gamma = 0.99$, $\lambda = 0.95$, $r_t = 1$, $V(s_t) = 5.0$, $V(s_{t+1}) = 5.5$:

- $\delta_t = 1 + 0.99 \times 5.5 - 5.0 = 1 + 5.445 - 5.0 = 1.445$

This tells us that the actual reward plus the discounted future value was 1.445 higher than what the critic predicted — so this action was better than expected.


![PPO requires four models — the critic alone doubles the memory.](figures/figure_2.png)
*PPO requires four models — the critic alone doubles the memory.*


## The Problem: Why the Critic is Expensive

Now let us understand why the critic is such a big problem for LLM training.

When training a 7 billion parameter language model with PPO, you need to keep **four models** in GPU memory simultaneously:

1. **Policy model**: The LLM being trained (~7B parameters)
2. **Reference model**: A frozen copy of the initial LLM for KL divergence (~7B parameters)
3. **Reward model**: Typically a fine-tuned LLM with a scalar head (~7B parameters)
4. **Value model (Critic)**: Same architecture as the policy, with a scalar head (~7B parameters)

That is roughly 28 billion parameters total, just for a 7B model. The critic alone accounts for 25% of the total memory.

But the problems go beyond memory. The critic must be trained jointly with the policy, which introduces additional instability. If the critic gives poor value estimates, the advantage estimates will be noisy, and the policy updates will be unreliable.

Now the question is: **Can we compute advantages without a critic at all?**


![GRPO eliminates the critic, saving 25% of GPU memory.](figures/figure_3.png)
*GRPO eliminates the critic, saving 25% of GPU memory.*


## The Key Insight: Group-Relative Baselines

Let us come back to our teacher grading essays.

Recall that the teacher found it much easier to grade essays by comparing them within a group rather than assigning absolute scores. GRPO does exactly the same thing.

Here is how it works:

**Step 1: Sample a group of responses.** For each prompt $q$, generate $G$ different completions $\{o_1, o_2, \ldots, o_G\}$ from the current policy $\pi_{\theta_{\text{old}}}$.

**Step 2: Score each response.** Pass each completion through the reward model to get rewards $\{r_1, r_2, \ldots, r_G\}$.

**Step 3: Normalize within the group.** Compute the advantage for each response by subtracting the group mean and dividing by the group standard deviation:


$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, r_2, \ldots, r_G)}{\text{std}(r_1, r_2, \ldots, r_G)}
$$

That is it. No critic network. No value function. No GAE. Just simple statistics within the group.

Let us plug in some simple numbers. Suppose we sample $G = 5$ responses for a prompt, and the reward model gives scores $\{2.0, 3.5, 1.0, 4.0, 2.5\}$:

- Mean: $(2.0 + 3.5 + 1.0 + 4.0 + 2.5) / 5 = 13.0 / 5 = 2.6$
- Std: $\sqrt{((2.0-2.6)^2 + (3.5-2.6)^2 + (1.0-2.6)^2 + (4.0-2.6)^2 + (2.5-2.6)^2)/5}$
- Std: $\sqrt{(0.36 + 0.81 + 2.56 + 1.96 + 0.01)/5} = \sqrt{5.7/5} = \sqrt{1.14} \approx 1.07$

Now the advantages are:
- $\hat{A}_1 = (2.0 - 2.6) / 1.07 = -0.56$ (below average)
- $\hat{A}_2 = (3.5 - 2.6) / 1.07 = +0.84$ (above average)
- $\hat{A}_3 = (1.0 - 2.6) / 1.07 = -1.50$ (well below average)
- $\hat{A}_4 = (4.0 - 2.6) / 1.07 = +1.31$ (best in group)
- $\hat{A}_5 = (2.5 - 2.6) / 1.07 = -0.09$ (roughly average)

Response 4 gets the highest advantage, response 3 gets the lowest. The policy will be updated to make responses like response 4 more likely and responses like response 3 less likely. This is exactly what we want.

The beauty of this approach is that it is **self-calibrating**. If all responses are good (high rewards), the advantages will still distinguish the best from the merely good. If all responses are bad, it will still identify the least bad one. The group provides its own baseline.


![GRPO normalizes rewards within a group to compute advantages without a critic.](figures/figure_4.png)
*GRPO normalizes rewards within a group to compute advantages without a critic.*


## The Full GRPO Objective

Now let us look at the complete GRPO objective function. It combines the group-relative advantages with PPO-style clipping and a KL divergence penalty:


$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q),\; \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left(\min\left(\frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})} \hat{A}_i, \; \text{clip}\left(\frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i\right) - \beta \, D_{\text{KL}}\left(\pi_\theta \| \pi_{\text{ref}}\right)\right]
$$

This looks complicated, but let us break it down term by term.

**The outer expectation** samples prompts $q$ from the dataset and generates $G$ completions for each prompt using the old policy.

**The double sum** averages over all $G$ responses and over all tokens within each response. The $\frac{1}{|o_i|}$ normalization ensures that longer responses are not weighted more heavily than shorter ones.

**The importance sampling ratio** is computed per-token:


$$
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, \, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, \, o_{i,<t})}
$$

This is the ratio of the probability of token $o_{i,t}$ under the new policy versus the old policy. If this ratio is greater than 1, the new policy is more likely to produce this token than the old policy.

**The clipping** works identically to PPO — it prevents the policy from changing too much in a single update. If the ratio exceeds $1 + \epsilon$ or falls below $1 - \epsilon$, it is clipped.

**The advantage** $\hat{A}_i$ is the group-relative advantage we derived earlier. Notice that it is the **same** for all tokens in a given response — the entire response gets a single advantage score, not per-token advantages.

**The KL penalty** prevents the policy from drifting too far from the reference model:


$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - 1
$$

This is an approximation of the KL divergence. Let us verify that it works correctly. If $\pi_\theta = \pi_{\text{ref}}$ (the policy has not drifted at all), then the ratio $\frac{\pi_{\text{ref}}}{\pi_\theta} = 1$:

$D_{\text{KL}} = 1 - \log(1) - 1 = 1 - 0 - 1 = 0$

The KL divergence is zero when the policies are identical. This is exactly what we want.

Now suppose $\frac{\pi_{\text{ref}}}{\pi_\theta} = 2$ (the reference model is twice as likely to produce this token):

$D_{\text{KL}} = 2 - \log(2) - 1 = 2 - 0.693 - 1 = 0.307$

The KL divergence is positive, so the penalty term will push the policy back toward the reference. This makes sense because a large divergence means the policy has drifted significantly.

Notice a key difference from PPO: in PPO, the KL divergence is used as a **constraint** (you stop training if KL gets too large). In GRPO, the KL divergence is added directly to the **loss function** as a penalty term scaled by $\beta$. This is simpler to implement and tune.

## GRPO vs PPO: Side-by-Side

Let us put the two methods side by side to see exactly what changed.

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Advantage estimation** | Learned value function (GAE) | Group mean/std of rewards |
| **Critic network** | Required (same size as policy) | Not needed |
| **Memory cost** | Policy + Critic + Reward + Reference | Policy + Reward + Reference |
| **Advantage granularity** | Per-token | Per-response |
| **KL divergence** | Constraint (early stopping) | Penalty term in loss |
| **Samples per prompt** | 1 (typically) | G (typically 4-64) |
| **Training stability** | Depends on critic quality | Depends on group size |

The fundamental trade-off is clear: GRPO trades the learned, per-token advantages of a critic for simple, per-response advantages computed from group statistics. This makes training simpler and cheaper, but it requires sampling multiple completions per prompt.


![GRPO replaces the critic with group-level reward normalization.](figures/figure_5.png)
*GRPO replaces the critic with group-level reward normalization.*


## GRPO in Practice: DeepSeek-R1

GRPO was developed by the DeepSeek team and used to train their reasoning models, including DeepSeek-R1. Let us look at how they applied GRPO in practice.

The DeepSeek-R1 training pipeline has four stages:

1. **Cold-start SFT**: Fine-tune the base model on a small set of chain-of-thought examples to bootstrap reasoning ability.

2. **RL with rule-based rewards (GRPO)**: This is where GRPO shines. The rewards are computed by simple, verifiable rules — for math problems, check if the final answer is correct; for code, run the test cases. No reward model needed at this stage.

3. **Rejection sampling + SFT**: Sample many responses, keep only the best ones (using the RL-trained model), and fine-tune on this curated dataset.

4. **RL with mixed rewards (GRPO again)**: A second round of GRPO training using both rule-based rewards (for reasoning) and a learned reward model (for helpfulness and safety).

One of the most remarkable findings from DeepSeek-R1 is that GRPO training with rule-based rewards alone led to the **emergence of chain-of-thought reasoning**. The model spontaneously learned to "think step by step" without ever being explicitly told to do so. It also developed self-verification behaviors — checking its own work before giving a final answer.

This is truly amazing. The model learned to reason just by being rewarded for getting the right answer, with GRPO providing the optimization signal.

For the group size, DeepSeek used $G = 64$ completions per prompt during GRPO training. This means that for each prompt, 64 different responses were generated, scored, and their advantages were computed relative to the group.


![DeepSeek-R1 uses two rounds of GRPO — emergent reasoning appears in stage 2.](figures/figure_6.png)
*DeepSeek-R1 uses two rounds of GRPO — emergent reasoning appears in stage 2.*


## Practical Implementation

Now let us implement GRPO in PyTorch. We will write the core components step by step.

First, the advantage computation — this is the heart of GRPO:

```python
import torch

def compute_grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute group-relative advantages from a batch of rewards.

    Args:
        rewards: Tensor of shape (G,) — rewards for G completions
                 of the same prompt
    Returns:
        advantages: Tensor of shape (G,) — normalized advantages
    """
    mean_reward = rewards.mean()
    std_reward = rewards.std()

    # Avoid division by zero if all rewards are identical
    if std_reward < 1e-8:
        return torch.zeros_like(rewards)

    advantages = (rewards - mean_reward) / std_reward
    return advantages
```

Let us understand this code in detail. We compute the mean and standard deviation of the rewards within the group. Then we subtract the mean and divide by the standard deviation. If all rewards are identical (standard deviation is zero), we return zero advantages — there is nothing to learn from a group where every response is equally good.

Next, the GRPO loss computation:

```python
def compute_grpo_loss(
    log_probs: torch.Tensor,       # (G, T) log probs under current policy
    old_log_probs: torch.Tensor,   # (G, T) log probs under old policy
    ref_log_probs: torch.Tensor,   # (G, T) log probs under reference policy
    advantages: torch.Tensor,       # (G,) per-response advantages
    mask: torch.Tensor,             # (G, T) attention mask
    epsilon: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """
    Compute the GRPO loss.
    """
    # Per-token importance sampling ratio
    ratio = torch.exp(log_probs - old_log_probs)  # (G, T)

    # Expand advantages to match token dimension
    adv = advantages.unsqueeze(1)  # (G, 1) -> broadcasts to (G, T)

    # Clipped surrogate objective
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    policy_loss = torch.min(surr1, surr2)  # (G, T)

    # KL divergence penalty (per-token)
    kl_ratio = torch.exp(ref_log_probs - log_probs)  # pi_ref / pi_theta
    kl_div = kl_ratio - torch.log(kl_ratio) - 1.0    # (G, T)

    # Combine: maximize policy objective, minimize KL divergence
    per_token_loss = policy_loss - beta * kl_div  # (G, T)

    # Average over valid tokens (using mask), then over group
    per_response = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)
    loss = -per_response.mean()  # Negate because we minimize

    return loss
```

Let us walk through the key parts. The importance sampling ratio $r_{i,t}$ is computed by exponentiating the difference in log probabilities — this is numerically more stable than dividing probabilities directly. The clipped surrogate is identical to PPO. The KL divergence uses the approximation we discussed earlier. Finally, we mask out padding tokens and average over the valid tokens in each response, then average over the group.

Here is a simplified training loop that shows how these pieces fit together:

```python
def grpo_training_step(
    policy_model, ref_model, reward_model, tokenizer,
    prompts, optimizer, G=8, epsilon=0.2, beta=0.04
):
    """One GRPO training step."""
    policy_model.train()
    total_loss = 0.0

    for prompt in prompts:
        # Step 1: Generate G completions from the OLD policy
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        completions = []
        for _ in range(G):
            with torch.no_grad():
                output = policy_model.generate(
                    input_ids, max_new_tokens=256,
                    do_sample=True, temperature=1.0
                )
            completions.append(output)

        # Step 2: Score each completion with the reward model
        rewards = torch.tensor([
            reward_model.score(prompt, tokenizer.decode(c[0]))
            for c in completions
        ])

        # Step 3: Compute group-relative advantages
        advantages = compute_grpo_advantages(rewards)

        # Step 4: Compute log probs under current, old, ref policies
        # (simplified — real impl would batch this)
        log_probs = get_log_probs(policy_model, completions)
        with torch.no_grad():
            old_log_probs = log_probs.detach()
            ref_log_probs = get_log_probs(ref_model, completions)

        # Step 5: Compute GRPO loss
        mask = create_completion_mask(completions, input_ids)
        loss = compute_grpo_loss(
            log_probs, old_log_probs, ref_log_probs,
            advantages, mask, epsilon, beta
        )

        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    optimizer.zero_grad()

    return total_loss / len(prompts)
```

The training loop follows the same pattern we discussed: sample G completions, score them, normalize within the group, and compute the clipped loss with KL penalty. Not bad, right?

## Conclusion

Let us summarize what we have learned.

GRPO eliminates the critic network by replacing learned value estimates with a simple statistical trick: sample multiple responses, compute their rewards, and normalize within the group. This gives us three major benefits:

1. **Lower memory cost**: No critic means 25% less GPU memory.
2. **Simpler training**: No need to jointly train a critic alongside the policy.
3. **Effective optimization**: Group-relative advantages are surprisingly effective, especially when combined with rule-based rewards for verifiable tasks.

DeepSeek demonstrated that GRPO can train state-of-the-art reasoning models, with chain-of-thought reasoning emerging purely from RL training without explicit supervision. This is a powerful validation of the approach.

The key insight to remember is this: sometimes, you do not need an absolute scale to evaluate quality. Comparing within a group is enough. This is exactly what GRPO does, and it works remarkably well.

Here is the link to our Google Colab notebooks where you can implement GRPO from scratch: [Vizuara Teaching Notebooks].

That's it!

---

## References

1. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
2. Shao, Z. et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
3. Schulman, J. et al. "Proximal Policy Optimization Algorithms." 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
4. Ouyang, L. et al. "Training language models to follow instructions with human feedback." NeurIPS 2022.
5. Sutton, R. S. and Barto, A. G. "Reinforcement Learning: An Introduction." MIT Press, 2018.
