# Reinforcement Learning from Human Feedback (RLHF) — Theory and Implementation from Scratch

A complete guide to aligning language models with human preferences: from reward modeling to PPO, with full code implementations

Vizuara AI

---

## Why RLHF Matters

Let us start with a simple example. Suppose you ask a language model the following question:

**Prompt:** "Explain quantum computing to a 10-year-old."

A base language model, which has only been trained to predict the next word, might give you a response like this:

*"Quantum computing leverages the principles of quantum mechanics, specifically superposition and entanglement of qubits, enabling parallel state exploration in Hilbert space..."*

This is technically correct, but completely useless for a 10-year-old.

Now, imagine the same model after it has been aligned with human preferences:

*"Imagine you have a magic coin. A normal coin is either heads or tails. But a quantum coin can be BOTH heads and tails at the same time! A quantum computer uses these magic coins to solve puzzles much faster than a regular computer."*

This is exactly what we want. The second response is helpful, clear, and appropriate for the audience.

But how do we get from the first response to the second? How do we teach a language model what humans actually prefer?

This is the problem that Reinforcement Learning from Human Feedback (RLHF) solves.


![Base LLM vs RLHF-aligned LLM: alignment transforms raw capability into helpful behavior.](figures/figure_1.png)
*Base LLM vs RLHF-aligned LLM: alignment transforms raw capability into helpful behavior.*


In this article, we will build the entire RLHF pipeline from scratch. We will understand every component — reward modeling, policy gradients, PPO, and KL penalties — with full mathematical derivations and practical implementations.

---

## Reinforcement Learning Meets Language Models

Let us start by understanding how Reinforcement Learning is applied to language models.

Consider the sentence: "I am going ___"

What is the next word? The next word depends on the previous set of words. So here, the **state** is the combination of the previous words.

The **action** is the next word in the sequence.

The **reward** denotes the "quality" of the action given the input state. We will look at how to quantify this quality later.

What about the **policy**? The policy is the probability of taking an action for the current state. For large language models, the policy is the probability of predicting the "next token" given the "current tokens."

This is the LLM itself.

In other words, in a large language model, the agent is the same as the policy, which is the LLM itself. This might sound a bit unusual, but it is true.

Let us write this down mathematically. Given a prompt $x$ made up of tokens, the LLM generates a completion $y$ token by token. The probability of generating the full completion is:


$$
\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_1, y_2, \ldots, y_{t-1})
$$


Let us plug in some simple numbers to see how this works. Suppose our model generates a 3-token completion, and the probabilities at each step are:

- $\pi_\theta(y_1 | x) = 0.8$ (first token "Pune")
- $\pi_\theta(y_2 | x, y_1) = 0.7$ (second token "is")
- $\pi_\theta(y_3 | x, y_1, y_2) = 0.9$ (third token "great")

Then the probability of the full completion is:

$\pi_\theta(y|x) = 0.8 \times 0.7 \times 0.9 = 0.504$

This tells us that our model assigns a probability of about 50.4% to this particular completion. The higher this probability, the more confident the model is about this sequence. This is exactly what we want — the model should assign high probability to good completions and low probability to poor ones.


![The LLM acts as both the agent and the policy in the RL framework.](figures/figure_2.png)
*The LLM acts as both the agent and the policy in the RL framework.*


---

## The Reward Problem — Why Humans Are Better at Comparing

Now the question is: how do we define the reward?

For objective questions, it is easy to assign rewards because we can judge the answers easily.

**Prompt:** What is 2 + 2?
**Answer:** 4
**Reward:** High (easy to verify)

But for subjective questions, humans are not good at finding a common ground for agreement.

**Prompt:** Explain RLHF like I am a 5-year-old.
**Answer:** RLHF is used for aligning models.
**Reward:** Subjective (not easy to verify)

But we are good at **comparing**.

If you show a human two different explanations side by side, they can immediately tell you which one is better. "This explanation is clearer, more engaging, and age-appropriate." We might disagree on whether an explanation deserves a 7 out of 10 or an 8 out of 10, but we can almost always agree on which of two explanations is better.


![Humans struggle to score responses absolutely but excel at pairwise comparisons.](figures/figure_3.png)
*Humans struggle to score responses absolutely but excel at pairwise comparisons.*


This insight is the foundation of RLHF. Instead of asking humans to assign numerical scores, we ask them to compare pairs of responses and pick the better one.

---

## Building a Reward Model

Now the question becomes: how do we turn these human comparisons into something a neural network can learn from?

The answer is the **Reward Model**.

The reward model is built from the LLM architecture itself with two main differences:

1. The hidden states are not projected into the vocabulary.
2. Only the final hidden state is passed to a linear layer to get a single scalar value as the reward.


![The reward model repurposes the LLM backbone, replacing the token prediction head with a scalar reward output.](figures/figure_4.png)
*The reward model repurposes the LLM backbone, replacing the token prediction head with a scalar reward output.*


To train our reward model, we need to define the loss function. The mathematical framework comes from the Bradley-Terry model, which says that the probability of preferring response $y_w$ over response $y_l$ is:


$$
P(y_w \succ y_l \mid x) = \sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)
$$


Here, $\sigma$ is the sigmoid function, $r_\theta(x, y_w)$ is the reward for the preferred response, and $r_\theta(x, y_l)$ is the reward for the rejected response.

Let us plug in some simple numbers to see how this works. Suppose our reward model assigns a score of $r = 3.0$ to the preferred response and $r = 1.0$ to the rejected response:

$P(y_w \succ y_l) = \sigma(3.0 - 1.0) = \sigma(2.0) = \frac{1}{1 + e^{-2.0}} = \frac{1}{1 + 0.135} = 0.881$

This tells us that the model believes there is an 88.1% chance that the preferred response is indeed better. This is exactly what we want — a well-trained reward model should assign high probability to the correct preference ordering.

The loss function is then:


$$\mathcal{L}_{\text{RM}} = -\log\sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)$$

Let us understand this by taking two cases.

**Case 1:** The reward model correctly ranks the preferred response higher ($r(y_w) > r(y_l)$).

Using our example where the difference is $+2.0$:

$\mathcal{L} = -\log(\sigma(2.0)) = -\log(0.881) = 0.127$

The loss is small. This is exactly what we want — when the model gets the ranking right, it should not be penalized much.

**Case 2:** The reward model incorrectly ranks the rejected response higher ($r(y_w) < r(y_l)$).

Suppose $r(y_w) = 1.0$ and $r(y_l) = 3.0$, so the difference is $-2.0$:

$\mathcal{L} = -\log(\sigma(-2.0)) = -\log(0.119) = 2.128$

The loss is very high. This is exactly what we want — when the model gets the ranking wrong, it should be heavily penalized.


![The sigmoid function maps reward differences to probabilities; the loss heavily penalizes incorrect rankings.](figures/figure_5.png)
*The sigmoid function maps reward differences to probabilities; the loss heavily penalizes incorrect rankings.*


---

## Policy Gradient — The Engine of RLHF

Now that we have a reward model that can score any completion, we need a way to use these rewards to improve our language model. This is where policy gradient methods come in.

The core idea is simple: we want to find the policy parameters $\theta$ that maximize the expected total reward. The performance measure is:

$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t r_t\right]$

Think of this as climbing a mountain. We are standing at some point on a landscape, and we want to move in the direction that takes us to the highest peak. The gradient of the performance measure tells us which direction to step.

The famous policy gradient theorem gives us the gradient of this performance measure:


$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A(s_t, a_t)\right]
$$


Here, $\nabla_\theta \log \pi_\theta(a_t | s_t)$ is the gradient of the log-probability of the action taken, and $A(s_t, a_t)$ is the advantage function — how much better this action was compared to what we expected.

Let us plug in some simple numbers. Suppose at time step $t$, the log-probability of the taken action is $\log \pi = -1.2$, and the advantage is $A = 2.5$ (the action was much better than expected). The gradient contribution from this step pushes us to increase the probability of this action by a factor proportional to $2.5$.

Now, if the advantage were $A = -1.0$ (the action was worse than expected), the gradient would push us to decrease the probability of this action. This is exactly what we want — reinforce good actions, discourage bad ones.


![Policy gradient tells the agent which direction to step in parameter space to improve performance.](figures/figure_6.png)
*Policy gradient tells the agent which direction to step in parameter space to improve performance.*


The intuition is powerful: if an action led to a good outcome (positive advantage), increase the probability of taking that action in similar states. If an action led to a bad outcome (negative advantage), decrease its probability.

---

## From REINFORCE to PPO

The basic REINFORCE algorithm uses the policy gradient directly, but it has a significant problem: high variance. Because we are sampling trajectories and estimating expectations, the estimates can fluctuate wildly between batches.

There is another practical issue. In the basic approach, after every gradient update, we need to throw away all our collected data and sample new trajectories from the updated policy. This is called **on-policy** learning, and it is very expensive for large language models because generating text is slow.

What if we could reuse trajectories from a slightly older version of our policy? This is where **importance sampling** comes in.

The idea is straightforward. Suppose we have two probability distributions $P$ and $Q$. We want to compute the expected value under $P$, but sampling from $P$ is expensive. If we can sample from $Q$ instead and re-weight each sample by the ratio $P(x)/Q(x)$, we get the correct expected value.

Applied to policy gradients, we multiply by the ratio of the current policy to the old policy:


$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

Let us plug in some simple numbers. Suppose the old policy assigned probability $0.3$ to action $a_t$, and our current policy assigns probability $0.45$:

$r_t(\theta) = \frac{0.45}{0.3} = 1.5$

This ratio of $1.5$ tells us that the current policy is 50% more likely to take this action compared to the old policy. We multiply the advantage by this ratio to correct for the distribution mismatch.

But here is the problem: if this ratio becomes too large or too small, the gradient updates become unstable. Imagine the ratio is $10.0$ — the current policy is 10 times more likely to take this action. Multiplying the advantage by 10 would cause a huge update, potentially breaking the policy.

This is exactly the problem that **Proximal Policy Optimization (PPO)** solves.

Let us understand PPO with the help of an analogy. Imagine that you are a runner who is trying to improve daily. You have an "old style" of running and you are exploring a "new style." You have a coach who gives you feedback.

The coach says: "Yes, the new style is better, but don't change too much in one session. Make gradual improvements."

PPO implements this "coaching" mathematically by clipping the importance sampling ratio:


$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \cdot A_t, \; \text{clip}\bigl(r_t(\theta), 1 - \epsilon, 1 + \epsilon\bigr) \cdot A_t\right)\right]$$

Here, $\epsilon$ is typically set to $0.2$, meaning we allow the ratio to vary between $0.8$ and $1.2$.

Let us plug in some simple numbers to see the clipping in action. Suppose $A_t = 2.0$ (a good action) and $r_t = 1.5$:

- Unclipped: $r_t \cdot A_t = 1.5 \times 2.0 = 3.0$
- Clipped: $\text{clip}(1.5, 0.8, 1.2) \cdot A_t = 1.2 \times 2.0 = 2.4$
- PPO takes the minimum: $\min(3.0, 2.4) = 2.4$

The clipping reduced the objective from $3.0$ to $2.4$, preventing an overly aggressive update. This is exactly what we want — stable, conservative improvements.

Now consider $A_t = -1.0$ (a bad action) and $r_t = 0.5$:

- Unclipped: $0.5 \times (-1.0) = -0.5$
- Clipped: $\text{clip}(0.5, 0.8, 1.2) \cdot (-1.0) = 0.8 \times (-1.0) = -0.8$
- PPO takes the minimum: $\min(-0.5, -0.8) = -0.8$

Here, PPO actually makes the penalty stronger, ensuring the bad action is properly discouraged. This makes sense because when the current policy is already moving away from a bad action (ratio < 1), we want to encourage that direction.


![PPO clips the objective to prevent the policy from changing too much in a single update.](figures/figure_7.png)
*PPO clips the objective to prevent the policy from changing too much in a single update.*


---

## Reward Hacking and the KL Penalty

There is one more critical problem we need to solve. Without any constraints, the language model will learn to "hack" the reward model.

Let us say our reward model is trained to give positive rewards to helpful, polite responses. During RL training, our LLM might figure out that using words like "happy," "love," "wonderful," and "thank you" in every response gets high rewards — regardless of whether these words are relevant to the prompt.

The model might start producing outputs like: "I am so happy and wonderful to help you with this love-ly question, thank you for asking!" This sounds absurd, but without constraints, reward hacking is a real and common failure mode.


![Without constraints, the LLM learns to exploit the reward model instead of genuinely improving.](figures/figure_8.png)
*Without constraints, the LLM learns to exploit the reward model instead of genuinely improving.*


We need to prevent our model from deviating too far from our reference model which already works decently. This is done by measuring the difference in the log probabilities of the same token predicted by our language model and the frozen reference model.

The metric for measuring this difference is the **KL divergence**. The modified reward becomes:


$$r_{\text{total}}(x, y) = r_{\text{RM}}(x, y) - \beta \cdot \text{KL}\bigl(\pi_\theta(\cdot | x) \;\|\; \pi_{\text{ref}}(\cdot | x)\bigr)$$

Here, $r_{\text{RM}}$ is the reward from our reward model, and the KL term penalizes the current policy for diverging from the frozen reference model. The coefficient $\beta$ controls how strongly we penalize divergence.

Let us plug in some simple numbers. Suppose $r_{\text{RM}} = 3.5$, $\text{KL} = 0.8$, and $\beta = 0.1$:

$r_{\text{total}} = 3.5 - 0.1 \times 0.8 = 3.5 - 0.08 = 3.42$

The KL penalty is small because the current policy is still close to the reference. Now suppose after many training steps, the model has diverged significantly and $\text{KL} = 15.0$:

$r_{\text{total}} = 3.5 - 0.1 \times 15.0 = 3.5 - 1.5 = 2.0$

The KL penalty has significantly reduced the effective reward, discouraging the model from straying too far. This is exactly what we want — the model should improve, but not at the cost of becoming a degenerate reward-hacking machine.

In practice, the KL divergence for each token is calculated as:

$\text{KL}_t = \log \pi_\theta(y_t | s_t) - \log \pi_{\text{ref}}(y_t | s_t)$

This is simply the difference in the log probabilities assigned to the same token by the current model and the frozen reference model. If both models agree, the KL is close to zero. If the current model deviates significantly, the KL grows large and the penalty kicks in.

---

## The Complete RLHF Pipeline

Now we have all the pieces of the puzzle ready. Let us put them together into the complete RLHF pipeline.


![The four stages of RLHF: pre-training, supervised fine-tuning, reward modeling, and PPO optimization.](figures/figure_9.png)
*The four stages of RLHF: pre-training, supervised fine-tuning, reward modeling, and PPO optimization.*


**Stage 1: Pre-training.** Train the base language model on a large text corpus using next-token prediction. This gives the model broad language capabilities but no alignment.

**Stage 2: Supervised Fine-Tuning (SFT).** Fine-tune the base model on high-quality human demonstrations. A human writes ideal responses to prompts, and the model learns to imitate them. This gives the model a good starting point.

**Stage 3: Reward Model Training.** Collect comparison data — humans compare pairs of model outputs and select the better one. Train a reward model to predict these preferences.

**Stage 4: RL Fine-Tuning (PPO).** Use the reward model to provide rewards and optimize the SFT model using PPO with a KL penalty against the frozen SFT model. This produces the final aligned model.

---

## Practical Implementation — GPT-2 Sentiment Alignment

Enough theory, let us look at some practical implementation now.

We will implement the entire RLHF pipeline to fine-tune a GPT-2 micro model to align its outputs towards positive sentiments. Here is our setup:

**Our model:** GPT-2 micro — 4 layers, 3 heads per layer, model dimension of 128 (approximately 1 million parameters).

**Our dataset:** 50,000 tweets for sentiment analysis.

**Our methodology:** We will focus specifically on the reinforcement learning step, assuming the model has already been pre-trained and fine-tuned.

First, let us set up the model and generate initial completions:

```python
import torch
import torch.nn.functional as F

# Model configuration
model_dim = 128
n_layers = 4
n_heads = 3
vocab_size = 50257  # GPT-2 tokenizer vocabulary

# Load pre-trained and SFT-fine-tuned model
model = GPT2Model(model_dim, n_layers, n_heads, vocab_size)
model.load_state_dict(torch.load("sft_model.pt"))

# Create a frozen copy as the reference model
ref_model = GPT2Model(model_dim, n_layers, n_heads, vocab_size)
ref_model.load_state_dict(model.state_dict())
ref_model.eval()  # Freeze reference model

# Reward model (same architecture, scalar output head)
reward_model = RewardModel(model_dim, n_layers, n_heads, vocab_size)
reward_model.load_state_dict(torch.load("reward_model.pt"))
reward_model.eval()
```

Now let us look at how the rewards are calculated. This is where the KL penalty comes in:

```python
def calculate_rewards(model, ref_model, reward_model,
                      prompts, completions, beta=0.1):
    """
    Calculate total rewards with KL penalty.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Get reward from the reward model
        full_text = prompt + completion
        r_reward = reward_model(full_text)  # Scalar reward

        # Calculate KL penalty per token
        model_logprobs = model.get_log_probs(prompt, completion)
        ref_logprobs = ref_model.get_log_probs(prompt, completion)
        kl_per_token = model_logprobs - ref_logprobs
        kl_penalty = kl_per_token.sum()

        # Total reward = reward - beta * KL
        total_reward = r_reward - beta * kl_penalty
        rewards.append(total_reward)

    return torch.stack(rewards)
```

The training loop uses the policy gradient method:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
GAMMA = 0.99
num_iterations = 1000

for iteration in range(num_iterations):
    # Step 1: Sample completions from current policy
    prompts = sample_prompts(dataset, batch_size=32)
    completions = model.generate(prompts, max_length=50)

    # Step 2: Calculate rewards with KL penalty
    rewards = calculate_rewards(
        model, ref_model, reward_model,
        prompts, completions, beta=0.1
    )

    # Step 3: Calculate log probabilities of selected actions
    log_probs = model.get_log_probs(prompts, completions)

    # Step 4: Calculate discounted returns
    returns = calculate_returns(rewards, gamma=GAMMA)

    # Step 5: Policy gradient loss
    # Negative because we maximize (gradient ascent)
    loss = -(log_probs * returns).mean()

    # Step 6: Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iteration % 100 == 0:
        print(f"Iteration {iteration}, "
              f"Mean Reward: {rewards.mean():.3f}")
```

After training, we can generate aligned outputs:

```python
# Generate completions after RLHF training
test_prompts = [
    "Today I feel",
    "The weather is",
    "My friend told me",
]

for prompt in test_prompts:
    completion = model.generate(prompt, max_length=30)
    reward = reward_model(prompt + completion)
    print(f"Prompt: {prompt}")
    print(f"Completion: {completion}")
    print(f"Reward: {reward:.2f}\n")
```


![The reward curve shows steady improvement during RLHF training, plateauing as the model converges.](figures/figure_10.png)
*The reward curve shows steady improvement during RLHF training, plateauing as the model converges.*


We can see that the reward is clearly rising with iterations, which means that our training is working perfectly. The model learns to generate outputs that the reward model scores highly, while the KL penalty prevents it from deviating too far from the reference model.


![After RLHF training, the model generates responses with consistently positive sentiment.](figures/figure_11.png)
*After RLHF training, the model generates responses with consistently positive sentiment.*


---

## Text Summarization with PPO

Now let us scale up to a more practical application: text summarization using RLHF with PPO.

We will use a pretrained 124M parameter GPT-2 model and fine-tune it for the Reddit Post Summarization task, which was one of the earliest preference fine-tuning benchmarks.

The pipeline has three steps:

**Step 1: Supervised Fine-Tuning.** We fine-tune GPT-2 on the CarperAI/openai_summarize_tldr dataset, which contains Reddit posts paired with human-written summaries.

**Step 2: Reward Model Training.** We train a reward model on the CarperAI/openai_summarize_comparisons dataset, which contains pairs of summaries where humans have indicated which one they prefer.

**Step 3: RL Fine-Tuning with PPO.** We use the trained reward model to optimize the SFT model using PPO with KL penalty. The key difference from our earlier vanilla policy gradient implementation is the clipping mechanism:

```python
# PPO update step (simplified)
for epoch in range(ppo_epochs):
    # Calculate current log probs and ratio
    curr_log_probs = model.get_log_probs(prompts, completions)
    ratio = torch.exp(curr_log_probs - old_log_probs)

    # Clipped surrogate objective
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    ppo_loss = -torch.min(unclipped, clipped).mean()

    # Update
    optimizer.zero_grad()
    ppo_loss.backward()
    optimizer.step()
```


![The summarization RLHF pipeline: SFT, reward model training, and PPO optimization.](figures/figure_12.png)
*The summarization RLHF pipeline: SFT, reward model training, and PPO optimization.*


The simplicity of the PPO clipping mechanism proved very effective in aligning large language models to human preferences, and hence it is used in modern RLHF implementations, including the models behind ChatGPT, Claude, and many others.

You can use this GitHub repo for replicating the RLHF pipeline: https://github.com/RajatDandekar/Hands-on-RL-Bootcamp-Vizuara

Navigate to the "RLHF-Part 1 Folder" and run the files for supervised fine-tuning, reward model training, and PPO fine-tuning under the "summarize_rlhf" folder.

---

## Wrapping Up

Let us quickly recap what we have covered in this article:

1. **The Alignment Problem:** Base language models are capable but not aligned with human preferences.
2. **Reward Modeling:** Train a neural network to predict human preferences from pairwise comparisons using the Bradley-Terry model.
3. **Policy Gradients:** The mathematical framework for optimizing a policy to maximize expected rewards.
4. **PPO:** A stable, clipped variant of policy gradients that prevents wild policy updates.
5. **KL Penalty:** A regularization mechanism that prevents the model from hacking the reward model.
6. **The Full Pipeline:** Pre-training, SFT, reward model training, and PPO fine-tuning come together to produce aligned language models.

RLHF bridges the gap between "what a model can say" and "what a model should say." It is the technique that transforms a raw language model into a helpful, harmless, and honest assistant.

That's it!

---

**References:**

1. Ouyang, L. et al. "Training language models to follow instructions with human feedback." NeurIPS 2022.
2. Schulman, J. et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
3. Christiano, P. et al. "Deep Reinforcement Learning from Human Preferences." NeurIPS 2017.
4. Ziegler, D. et al. "Fine-Tuning Language Models from Human Preferences." arXiv:1909.08593, 2019.
5. Stiennon, N. et al. "Learning to Summarize with Human Feedback." NeurIPS 2020.
