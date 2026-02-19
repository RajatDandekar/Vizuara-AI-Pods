# Policy Gradient Methods: Teaching Your Agent to Climb Mountains

*From REINFORCE to Actor-Critic -- how gradient ascent on policy parameters unlocks continuous and high-dimensional action spaces.*

## Why Not Just Use Q-Tables?

Let us start with a simple example. Imagine that you have a robotic arm with 7 joints, and each joint can rotate to any angle between 0 and 360 degrees. Your goal is to train this arm to pick up a cup of coffee from a table.

Now, think about the Q-learning approach we have used before. In Q-learning, for every state, we look at the action value function Q and ask: which action gives me the maximum Q-value?

But here is the problem. If each joint can take any angle, the number of possible actions is effectively infinite. You cannot store Q-values for every possible combination of 7 joint angles in a table. It is like trying to write down every possible sentence in the English language -- you simply cannot do it.

This brings us to a fundamental question: Can we directly learn the policy -- the mapping from states to actions -- without going through value functions first?

Think about it this way. In Q-learning, we find the policy indirectly. We first estimate the value function Q, and then use Q to derive the policy. It is like finding "A" by first estimating "B" and then using "B" to find "A". But what if we could estimate "A" directly?

This is exactly what policy gradient methods do. Instead of learning a value function and deriving the policy from it, we directly parameterize the policy and optimize it.

We introduce a policy parameter $\theta$ and write the policy as $\pi(a|s, \theta)$. In general, $\theta$ belongs to a D-dimensional space, $\theta \in \mathbb{R}^D$.


$$
\pi(a|s, \theta) = P(\text{action} = a \mid \text{state} = s, \text{parameters} = \theta)
$$


This tells us that the policy is a probability distribution over actions, given a state and a set of parameters. For example, if we have a state where a cart is tilting to the left, the policy might output a 70% probability of pushing right and a 30% probability of pushing left.

Let us plug in some simple numbers. Suppose we have a state $s$ and two possible actions: left and right. If our policy outputs $\pi(\text{left}|s, \theta) = 0.3$ and $\pi(\text{right}|s, \theta) = 0.7$, then the agent will push right 70% of the time. This is exactly what we want -- a direct mapping from states to action probabilities.


![Policy gradient methods learn the action probabilities directly, bypassing the Q-function entirely.](figures/figure_1.png)
*Policy gradient methods learn the action probabilities directly, bypassing the Q-function entirely.*


Let us look at policy parameterization now.

## Policy Parameterization: From Preferences to Probabilities

How do we parameterize the policy?

Let us say for a state $s$ there are five possible actions. We can assign a numerical preference to each action. These preferences capture how "desirable" each action is -- a higher preference means the action is more likely to be chosen.

The preferences can be captured in a functional form denoted by $h(a, \theta)$, where $\theta$ is our policy parameter vector.

If we assume $h(a, \theta)$ to be a simple linear function, then:

$$h(a, \theta) = \theta_1 \cdot \phi_1(s, a) + \theta_2 \cdot \phi_2(s, a) + \theta_3 \cdot \phi_3(s, a)$$

Here, $\phi_1, \phi_2, \phi_3$ are features that describe the state-action pair, and $\theta_1, \theta_2, \theta_3$ are weights that we learn. The policy parameter is a vector of these values and it represents something like "weights" which have to be tuned to match the preferences.

Now the question is, how do we convert the preferences into probabilities? Remember that our policy will predict the probabilities of all the possible actions.

We can simply use the softmax function to convert the preferences to probabilities:


$$
\pi(a|s, \theta) = \frac{e^{h(a, \theta)}}{\sum_{a'} e^{h(a', \theta)}}
$$

Let us plug in some simple numbers to see how this works. Suppose we have three actions with preferences $h(a_1) = 2.0$, $h(a_2) = 1.0$, and $h(a_3) = 0.5$. Then:

$$e^{2.0} = 7.39, \quad e^{1.0} = 2.72, \quad e^{0.5} = 1.65$$

The sum is $7.39 + 2.72 + 1.65 = 11.76$. So our probabilities are:

$$\pi(a_1) = \frac{7.39}{11.76} = 0.63, \quad \pi(a_2) = \frac{2.72}{11.76} = 0.23, \quad \pi(a_3) = \frac{1.65}{11.76} = 0.14$$

This tells us that action $a_1$ has the highest probability (63%) because it has the highest preference. This is exactly what we want -- actions with higher preferences get higher probabilities, but we still maintain some exploration by giving non-zero probability to all actions.


![Softmax converts raw preferences into a valid probability distribution.](figures/figure_2.png)
*Softmax converts raw preferences into a valid probability distribution.*


Fun Fact: $h(a, \theta)$ was a deep neural network when AlphaGo beat the human Go champion.

One of the fundamental tasks in policy gradient methods is to train our policy to find the values of the policy parameters by maximizing a performance measure. But what exactly is this performance measure? Let us understand.

## The Performance Measure: What Are We Climbing?

Our objective is to maximize something which captures the performance of our policy parameter. This something is called a performance measure and is denoted by $J(\theta)$.

The whole objective of the reinforcement learning process is to optimize the cumulative rewards received in the entire episode. This is exactly what the value function captures.

Hence, in policy gradient methods, the standard performance measure is the value function of the initial state of the episode:

$$J(\theta) = V^{\pi_\theta}(s_0)$$

Think of it this way. Imagine you are standing at the base of a mountain. Your position on the mountain is determined by your policy parameters $\theta$, and the height of the mountain at your position is the performance measure $J(\theta)$. Your goal is to climb to the peak.


![Gradient ascent moves the policy parameters toward the peak of the performance measure.](figures/figure_3.png)
*Gradient ascent moves the policy parameters toward the peak of the performance measure.*


You might have seen something like this before. Yes, this is very similar to the gradient descent algorithm in machine learning. The only difference here is that since we are maximizing our objective, we are performing an ascent instead of a descent.

The policy parameter update rule is then given according to the gradient ascent formula:


$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$


Let us plug in some simple numbers. Suppose $\theta = 1.5$, $\alpha = 0.1$, and $\nabla_\theta J(\theta) = 2.0$. Then:

$$\theta_{\text{new}} = 1.5 + 0.1 \times 2.0 = 1.7$$

This tells us that we move our parameter from 1.5 to 1.7 in the direction that increases our performance measure. This is exactly what we want.

So you can see from the above formula that once we understand how to climb the mountain, we will be able to successfully update our policy parameters.

But to climb the mountain, we need to find out, for every step taken in the policy parameter direction, how much the performance measure changes by. How will we find the gradient of the performance measure?

## Finding the Gradient: A Step-by-Step Derivation

We are going to do a small derivation right now through which you will understand how this gradient is calculated. We are going to perform this derivation in multiple steps.

**Step 1: Substituting the value function in the performance measure**


$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ G(\tau) \right]
$$


Here, $\tau$ signifies the trajectories and $G$ is the return for each trajectory. It is important to understand that the trajectory is not fixed -- for the same policy, we can have multiple different trajectories because the policy output is probabilistic in nature.

**Step 2: Expanding the expectation**


$$
J(\theta) = \sum_\tau P(\tau; \theta) \cdot G(\tau)
$$


Here, $P(\tau; \theta)$ denotes the probability of sampling a specific trajectory from all possible trajectories for the given policy.

**Step 3: Bring the gradient under the summation**


$$
\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau; \theta) \cdot G(\tau)
$$


The above equation simply comes from this rule: The gradient of a sum is equal to the sum of gradients.

**Step 4: The Log-Derivative Trick**

This is the key insight in the entire derivation. We use the identity:

$$\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta)$$

This comes from the rule: The gradient of the log of a function is equal to the gradient of the function divided by the function itself. Rearranging gives us the above.


$$
\nabla_\theta J(\theta) = \sum_\tau P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta) \cdot G(\tau)
$$


**Step 5: Return to expectation form**


$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) \cdot G(\tau) \right]
$$


**Step 6: Expanding the trajectory probability**

Now, the probability of a trajectory can be written as:

$$P(\tau; \theta) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)$$

Taking the log and then the gradient with respect to $\theta$, the initial state distribution $p(s_0)$ and the transition dynamics $p(s_{t+1}|s_t, a_t)$ do not depend on $\theta$, so they vanish. We are left with:

$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**Step 7: Final expression**


$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G(\tau) \right]
$$


The gradient of the performance measure of the policy is given by the expected value over all possible trajectories of the summation of the grad-log action probabilities for each trajectory times the return received for the trajectory.

The above equation is the single most important equation to understand policy gradient methods. It is very closely linked with the Policy Gradient Theorem.

The first proof for the Policy Gradient Theorem came about in 1999 in Richard Sutton's group, who is one of the pioneers in the field.

Let us plug in some simple numbers with a tiny example. Consider a single trajectory with $T = 2$ time steps. The agent is in state $s_0$, takes action $a_0$ with $\log \pi_\theta(a_0|s_0) = -0.5$, then transitions to $s_1$, takes action $a_1$ with $\log \pi_\theta(a_1|s_1) = -1.2$. The total return is $G(\tau) = 3.0$.

The gradient contribution from this trajectory is:

$$\nabla_\theta \log \pi_\theta(a_0|s_0) \cdot 3.0 + \nabla_\theta \log \pi_\theta(a_1|s_1) \cdot 3.0$$

If $\nabla_\theta \log \pi_\theta(a_0|s_0) = 0.8$ and $\nabla_\theta \log \pi_\theta(a_1|s_1) = -0.3$, then the gradient is:

$$0.8 \times 3.0 + (-0.3) \times 3.0 = 2.4 - 0.9 = 1.5$$

This positive gradient tells us to increase $\theta$ because this trajectory had a good return. This is exactly what we want.


![The policy gradient weights each action's log-probability gradient by the trajectory return.](figures/figure_4.png)
*The policy gradient weights each action's log-probability gradient by the trajectory return.*


Now let us try to understand the intuition behind this equation and what it really says.

## Intuition: Reinforcing Good Actions

Let us say that we have an old policy (O). We take one step in the environment and we collect the reward. Now we want to update the policy based on this step.

The new policy is denoted by N.

Using the formula for the gradient of the performance measure, we can find the direction in which we want to move to improve our performance.

Here is the key insight. When we update the policy, we are moving along the performance curve:

- If we have a return which is very good, then we are going to take longer steps, moving towards the peak of the mountain. This means that we are going to **increase** the probability of taking the specific action $a_t$ for the given state $s_t$.

- Similarly, if we have a return which is close to zero or negative, then we are going to take steps which move us away from the peak. This means that we are going to **decrease** the probability of taking that action.


![Actions producing high returns are reinforced; actions producing low returns are penalized.](figures/figure_5.png)
*Actions producing high returns are reinforced; actions producing low returns are penalized.*


Because actions which give good returns are reinforced and actions which give bad returns are penalized, this is called the **REINFORCE** algorithm.

## The REINFORCE Algorithm

The REINFORCE algorithm is beautifully simple. Here is how it works:

1. Initialize the policy parameters $\theta$ randomly
2. Generate a complete episode using the current policy $\pi_\theta$
3. For each time step $t$ in the episode, compute the return $G_t$
4. Update: $\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$
5. Repeat from step 2


![The REINFORCE algorithm collects a full episode before updating the policy.](figures/figure_6.png)
*The REINFORCE algorithm collects a full episode before updating the policy.*


Our policy will be a neural network. Let us define it:

```python
class PGN(nn.Module):
    """
    Policy Gradient Network
    """
    def __init__(self, input_size: int, n_actions: int):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

This network takes the state as input and outputs logits for each action. The logits are then passed through a softmax function to get the action probabilities. This is exactly the policy parameterization we discussed earlier -- the neural network learns the preference function $h(a, \theta)$.

## The Variance Problem and REINFORCE with Baseline

Imagine that you want to find out the average age of your country. You go ahead, collect a lot of samples, and find out the average.

Your estimate is going to heavily depend on which people you did sample. For example, if you only sample people above 60 years of age, you will wrongly think that the average age of the country is more than 60. Different samples will give you very different estimates -- this is the problem of **high variance**.

This is exactly the problem with the REINFORCE method. Because we are sampling trajectories and calculating the expected value, the main issue with the REINFORCE algorithm is that of variance. Some trajectories might have very high returns and some might have very low returns, and the gradient estimate swings wildly between updates.

To solve this problem, people have modified the REINFORCE method with something called **REINFORCE with Baseline**.

In the REINFORCE with Baseline method, a constant value is subtracted from the return. This constant value is dependent on the state.

The gradient of the performance measure is updated as follows:


$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \right]
$$


Here, $b(s_t)$ represents the baseline.

Let us plug in some simple numbers to see why this helps. Suppose we have two trajectories:

- Trajectory 1: $G_1 = 100$, $\nabla_\theta \log \pi = 0.5$. Contribution: $0.5 \times 100 = 50$
- Trajectory 2: $G_2 = 90$, $\nabla_\theta \log \pi = -0.3$. Contribution: $-0.3 \times 90 = -27$

The gradient fluctuates between 50 and -27. Now, if we use a baseline $b = 95$ (the mean return):

- Trajectory 1: $0.5 \times (100 - 95) = 0.5 \times 5 = 2.5$
- Trajectory 2: $-0.3 \times (90 - 95) = -0.3 \times (-5) = 1.5$

The gradient now fluctuates between 2.5 and 1.5. Much more stable. This tells us that the baseline does not change the expected gradient (it is still pointing in the right direction), but it dramatically reduces the variance. This is exactly what we want.

A common practice is to use the value function itself as the baseline. When we subtract the value function from the return, we get what is called the **advantage**:

$$A(s_t, a_t) = G_t - V(s_t)$$

The advantage tells us: "How much better was this action compared to what we expected?" If the advantage is positive, the action was better than average. If negative, it was worse than average.


![Subtracting a baseline reduces variance and accelerates convergence.](figures/figure_7.png)
*Subtracting a baseline reduces variance and accelerates convergence.*


## Actor-Critic Methods

When the baseline is the value function $V(s)$, we get something special. The policy (which decides actions) is the **actor**, and the value function (which evaluates states) is the **critic**.

Think of it like a student and a teacher. The student (actor) takes actions -- answers questions, solves problems. The teacher (critic) evaluates the student's performance -- not just whether the answer was right, but whether it was better or worse than expected.

The advantage function captures this:


$$
A(s, a) = Q(s, a) - V(s)
$$


Let us plug in some simple numbers. Suppose for state $s$, the value function estimates $V(s) = 10$. The agent takes action $a$ and the actual Q-value turns out to be $Q(s, a) = 13$. Then:

$$A(s, a) = 13 - 10 = 3$$

This positive advantage (+3) tells us that action $a$ was significantly better than what we expected in state $s$. So we should increase the probability of taking action $a$ in state $s$.

Now suppose a different action $a'$ gives $Q(s, a') = 7$:

$$A(s, a') = 7 - 10 = -3$$

This negative advantage (-3) tells us that action $a'$ was worse than expected. We should decrease its probability.

In actor-critic methods, we maintain two networks:

1. **The Actor**: The policy network $\pi_\theta(a|s)$ that selects actions
2. **The Critic**: The value network $V_\phi(s)$ that estimates state values


![The actor selects actions while the critic evaluates them, forming a feedback loop.](figures/figure_8.png)
*The actor selects actions while the critic evaluates them, forming a feedback loop.*


The actor-critic approach is powerful because it combines the direct policy optimization of policy gradient methods with the lower variance of value-based methods. This makes sense because the critic helps stabilize the actor's learning by providing a better signal than raw returns.

## Practical Implementation: CartPole with REINFORCE

Now let us implement these methods for a practical problem. We will take the example of the CartPole problem from OpenAI Gymnasium.

There are only two actions in the CartPole problem: left or right. Our main objective is to move the cart in such a way that the pole balanced on top does not fall over for a long time.

**Step 1: Defining the Q-value calculation function**

We need to define a function to estimate the returns based on the rewards collected during an episode:

```python
def calc_qvals(rewards: tt.List[float]) -> tt.List[float]:
    """
    Calculates discounted rewards for an episode.
    Args:
        rewards: A list of rewards obtained during one episode.
    Returns:
        A list of discounted total rewards (Q-values).
    """
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    # The result is now in reverse order, so we reverse it back
    return list(reversed(res))
```

The way the Q-values are calculated is interesting. First, we calculate the return from the last state and then use that information to calculate the return of the previous state, and so on, until we reach the first state of the episode.

**Step 2: Collecting states, actions, and rewards**

```python
# --- Action Selection ---
# Convert state to a tensor and add a batch dimension
state_t = torch.as_tensor(state, dtype=torch.float32)

# Get action probabilities from the network
logits_t = net(state_t)
probs_t = F.softmax(logits_t, dim=0)

# Sample an action from the probability distribution
# .item() extracts the value from the tensor
action = torch.multinomial(probs_t, num_samples=1).item()

# --- Environment Step ---
next_state, reward, done, truncated, _ = env.step(action)

# --- Store Experience ---
batch_states.append(state)
batch_actions.append(action)
episode_rewards.append(reward)
```

From the above code, we can see that the actions are sampled from the probability distribution which our policy neural network generates. This action is then passed to the `env.step()` function to collect the next state and the reward.

**Step 3: Loss calculation and optimization**

```python
# --- Loss Calculation and Optimization ---
optimizer.zero_grad()

logits_t = net(states_t)
log_prob_t = F.log_softmax(logits_t, dim=1)

# Get the log probabilities for the actions that were taken
log_prob_actions_t = log_prob_t[range(len(batch_states)), actions_t]

# Calculate the policy loss: - (log_prob * Q-value)
# We want to maximize (log_prob * Q-value), which is:
# minimizing its negative.
loss_t = -(qvals_t * log_prob_actions_t).sum()

loss_t.backward()
optimizer.step()
```

In the above step, you can see that we are calculating the log probabilities for the actions, multiplying them by the Q-value, and then summing it up for all the states in the episode. This is exactly what the REINFORCE algorithm says.

We multiply this gradient with a negative sign because, for the Python library, we need a loss which we can minimize, not maximize.

Now we have all the pieces of the puzzle ready. Let us see how the rewards change with time.


![REINFORCE converges to the maximum reward of 500 after approximately 120,000 steps.](figures/figure_9.png)
*REINFORCE converges to the maximum reward of 500 after approximately 120,000 steps.*


We can clearly see that the reward increases with time and reaches a value of 500, which is excellent.

Now let us run the code using the REINFORCE with Baseline method. We are going to use a baseline which is the mean of all the Q-values encountered in the episode:

```python
def calc_qvals(rewards: tt.List[float]) -> tt.List[float]:
    """
    Calculates discounted rewards for an episode and applies a baseline.
    The baseline is the mean of the discounted rewards for the episode.
    Subtracting it helps to reduce variance.
    """
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    mean_q = np.mean(res)
    return [q - mean_q for q in res]
```

Now let us compare the rewards for both methods:


![The baseline method reaches maximum reward 30% faster than vanilla REINFORCE.](figures/figure_10.png)
*The baseline method reaches maximum reward 30% faster than vanilla REINFORCE.*


We can see that the REINFORCE with Baseline method reaches a reward of 500 much faster, in approximately 86,000 steps as opposed to 120,000 steps required for the simple REINFORCE method.

Let us check the variance of the Q-values:


![The baseline dramatically reduces Q-value variance from 600+ to under 200.](figures/figure_11.png)
*The baseline dramatically reduces Q-value variance from 600+ to under 200.*


The Q-value variance is significantly less for the REINFORCE with Baseline method compared to the simple REINFORCE method. This clearly tells us that introducing a baseline not only reduces the variance but also allows for faster convergence.

## Key Takeaways

Let us summarize the journey we have taken:

1. **Policy parameterization** allows us to directly optimize the policy without going through value functions, making it possible to handle continuous action spaces.

2. **The policy gradient theorem** gives us a way to compute the gradient of the performance measure -- and the key trick is the log-derivative identity.

3. **REINFORCE** is the simplest policy gradient algorithm: collect a full episode, weight each action by its return, and update.

4. **Baselines** (especially the value function) reduce variance without introducing bias, leading to faster and more stable training.

5. **Actor-Critic** methods combine the best of both worlds: the actor directly optimizes the policy while the critic provides low-variance feedback.

These methods form the foundation for modern algorithms like PPO (Proximal Policy Optimization) and TRPO (Trust Region Policy Optimization), which are used in training large language models through RLHF.

Refer to this GitHub repo for reproducing the code: https://github.com/RajatDandekar/Hands-on-RL-Bootcamp-Vizuara

## References

1. Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). "Policy Gradient Methods for Reinforcement Learning with Function Approximation." *Advances in Neural Information Processing Systems*.
2. Williams, R. J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning*, 8, 229-256.
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
