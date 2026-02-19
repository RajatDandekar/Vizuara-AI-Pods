# Value Functions and Q-Learning: From Bellman's Insight to Learning Optimal Behavior

*How a recursive equation from the 1950s became the foundation for teaching machines to make decisions*

---

Let us start with a simple scenario. Imagine that you are playing a board game on a grid. Your goal is to reach the treasure at the top-right corner while avoiding obstacles.

Now, at any point during the game, you find yourself at a specific cell on the grid. A thought naturally comes to your mind: "How good is it to be in this cell?"

If you are right next to the treasure, the answer is obvious — it is very good. If you are far away with obstacles between you and the treasure, the answer is less encouraging. But here is the interesting part: even cells that are far from the treasure can be "good" if they lie on a clear path towards it.

This idea — assigning a number to every position that tells us "how good it is to be here" — is the foundation of value functions. And it turns out that this simple idea, when combined with a beautiful recursive equation from the 1950s, gives us one of the most powerful tools in all of reinforcement learning.


![Every state has a value — cells closer to the goal are worth more.](figures/figure_1.png)
*Every state has a value — cells closer to the goal are worth more.*


---

## State Value Functions

The value of a state is defined as the expected total future reward that an agent can accumulate, starting from that state and following a given policy thereafter.

This is also called the **state value function**.

Simply put, it answers the question: "If I am in this state right now and I keep following my current strategy, how much total reward can I expect to receive?"

Let us do a bit of mathematics. The state value function is mathematically defined as follows:


$$
V^{\pi}(s) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s\right]
$$


Here, $G_t$ is the total return received from the state $s$, and the expectation is taken over the policy $\pi$.

Let us plug in some simple numbers to see how this works. Suppose we have three states arranged in a chain: A, B, and C. State C is the terminal state.

- From state A, following our policy, we move to state B and receive a reward of +1.
- From state B, we move to state C and receive a reward of +2.
- State C is terminal, so we stop.

The value of state B is simply the reward we get from B: $V(B) = 2$.

The value of state A is the reward from A plus the value of state B: $V(A) = 1 + 2 = 3$.

This makes sense because if we start from state A, we will collect a total reward of $1 + 2 = 3$ before the episode ends.

But wait — in most problems, rewards received far in the future should be worth less than rewards received right now. Think of it this way: Rs. 100 are more valuable today compared to Rs. 100 five years later due to inflation. Similarly, immediate rewards are more valuable compared to rewards received later.

We capture this using a **discount factor** $\gamma$ (gamma), which is a number between 0 and 1. The discounted return is defined as:


$$
G_t = r_t + \gamma \, r_{t+1} + \gamma^2 \, r_{t+2} + \gamma^3 \, r_{t+3} + \cdots
$$


Let us plug in some simple numbers. Suppose our rewards are $r_t = 1$, $r_{t+1} = 2$, $r_{t+2} = 3$, and $\gamma = 0.9$:

$$G_t = 1 + 0.9 \times 2 + 0.81 \times 3 = 1 + 1.8 + 2.43 = 5.23$$

This tells us that our total discounted return from time step $t$ is 5.23. Notice how the later rewards contribute less due to discounting — the reward of 3 at time $t+2$ only contributes 2.43 after discounting. This is exactly what we want.


![Discounted returns accumulate backward from the terminal state.](figures/figure_2.png)
*Discounted returns accumulate backward from the terminal state.*


---

## Action Value Functions

State value functions tell us how good it is to be in a state. But they do not tell us which action to take.

This brings us to the **action value function**, also called the **Q-function**.

The action value function tells us: "If I am in state $s$, take action $a$, and then follow policy $\pi$ thereafter, what is the expected total return?"

It is mathematically defined as follows:


$$
Q^{\pi}(s, a) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s, A_t = a\right]
$$


The action value function is denoted by $Q$, and it is written as $Q(s, a)$, meaning the action value is calculated for a specific state and a specific action taken from that state.

Now the question is, how is this different from the state value function?

Let us take a simple example. Imagine you are standing at a junction with two paths: Path Left and Path Right.

- If you go Left, you reach a state with value 5 and collect a reward of +2 along the way.
- If you go Right, you reach a state with value 8 and collect a reward of +1 along the way.

Using $\gamma = 0.9$:

$$Q(s, \text{Left}) = 2 + 0.9 \times 5 = 2 + 4.5 = 6.5$$

$$Q(s, \text{Right}) = 1 + 0.9 \times 8 = 1 + 7.2 = 8.2$$

So the Q-value for going Right (8.2) is higher than going Left (6.5). If we are trying to maximize our return, we should go Right. This is exactly what Q-values are for — they help us decide which action is the best in every state.

The relationship between V and Q is straightforward: the state value is the expected Q-value over all actions according to the policy:

$$V^{\pi}(s) = \sum_{a} \pi(a \mid s) \, Q^{\pi}(s, a)$$


![Q-values tell us the expected return for each action from a state.](figures/figure_3.png)
*Q-values tell us the expected return for each action from a state.*


---

## The Bellman Equation

Now we arrive at the most important equation in all of reinforcement learning.

Let us come to the main character in our story — Mr. Richard Bellman.

Richard Bellman was a brilliant mathematician who worked on optimal control theory in the 1950s. He was known for his clear and simple insights, and he was a critic of unnecessarily complex mathematics.

The Bellman equation is actually very simple. It states that:

**The value of being in a state is equal to the expected reward for acting according to the policy, plus the expected value of wherever you end up next.**

Imagine that you are playing Pac-Man with a fixed routine: if a ghost is nearby, move away; if there is a pellet, move towards it. This routine is your policy $\pi$.

The question you ask is: "If I start in this specific maze location with this policy, how many points will I score on average?"

The Bellman equation says: you do not need to simulate the entire game from this point. Instead, just look at the immediate reward from your next move, and then add the value of wherever that move takes you.

Mathematically, the Bellman equation for the state value function is:


$$
V^{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \, V^{\pi}(s') \right]
$$


This reads: the value of state $s$ equals the sum over all possible actions (weighted by the policy) of the sum over all possible next states and rewards (weighted by the transition probabilities) of the immediate reward plus the discounted value of the next state.

Let us plug in some simple numbers to see how this works. Consider two states, A and B:

- From state A, there is only one action: move to state B with reward +1.
- From state B, there is only one action: stay in state B with reward +2.
- Discount factor: $\gamma = 0.9$.

For state B:

$$V(B) = 2 + 0.9 \times V(B)$$

$$V(B) - 0.9 \times V(B) = 2$$

$$0.1 \times V(B) = 2$$

$$V(B) = 20$$

For state A:

$$V(A) = 1 + 0.9 \times V(B) = 1 + 0.9 \times 20 = 1 + 18 = 19$$

So the value of state A is 19 and the value of state B is 20. This makes sense — state B is slightly more valuable because you start collecting the +2 rewards immediately, while from state A, you first have to make the +1 transition.

Why was this groundbreaking? Before the Bellman equation, people struggled to solve long-term planning problems efficiently. The Bellman equation showed that you can break any long-horizon problem into just one step at a time. Even if you cannot comprehend the totality of the problem, solving it one step at a time in a recursive manner can help you solve for the value of every state. Believe it or not, this insight is at the heart of all the modern deep reinforcement learning algorithms.


![The Bellman equation decomposes value into immediate reward plus future value.](figures/figure_4.png)
*The Bellman equation decomposes value into immediate reward plus future value.*


---

## The Bellman Optimality Equation

Now we know how to evaluate a given policy using the Bellman equation. But here is the natural next question:

**How do we find the best policy?**

The optimal state value function $V^*(s)$ gives the maximum possible value for each state, over all possible policies:


$$
V^*(s) = \max_{a} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \, V^*(s') \right]
$$


And the optimal action value function $Q^*(s, a)$ is:


$$
Q^*(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \, \max_{a'} Q^*(s', a') \right]
$$


The key difference from the regular Bellman equation is that instead of averaging over all actions according to a policy, we take the **maximum** over all actions. We always pick the best action.

Let us plug in some simple numbers. Suppose we are in state $s$ and have two actions:

- Action 1: leads to next state with $V^* = 10$, immediate reward $r = 3$.
- Action 2: leads to next state with $V^* = 6$, immediate reward $r = 8$.

With $\gamma = 0.9$:

$$\text{Value via Action 1} = 3 + 0.9 \times 10 = 3 + 9 = 12$$

$$\text{Value via Action 2} = 8 + 0.9 \times 6 = 8 + 5.4 = 13.4$$

So $V^*(s) = \max(12, 13.4) = 13.4$, and the optimal action is Action 2. Notice how Action 2 wins even though its next state has a lower value — the higher immediate reward more than compensates. This is why we need to consider the full picture: immediate reward plus discounted future value.

But here is the catch: the Bellman optimality equation is recursive. The variable we are trying to solve for ($V^*$) appears on both sides of the equation. And in practical problems, you might have millions of states. You cannot just "solve" the system of equations directly.

So, how do we solve these equations? This naturally brings us to Q-Learning.


![The optimal policy always picks the action leading to the highest value.](figures/figure_5.png)
*The optimal policy always picks the action leading to the highest value.*


---

## Q-Learning: Learning from Experience

Q-Learning was developed by Chris Watkins in 1989. This algorithm initially went under the radar, but it was resurrected in 2013 when DeepMind used a variant of Q-Learning to play Atari games at superhuman levels.

The beauty of Q-Learning is that it does not require a model of the environment. You do not need to know the transition probabilities. The agent simply interacts with the environment, collects experience, and updates its Q-values after every single step.

The Q-Learning update rule is:


$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \, \max_{a'} Q(s', a') - Q(s, a) \right]
$$


Let us break this down piece by piece:

- $Q(s, a)$ is the current estimate of the Q-value for state $s$ and action $a$.
- $\alpha$ is the **learning rate** — how much we trust the new information versus the old estimate.
- $r + \gamma \max_{a'} Q(s', a')$ is the **TD target** — the reward we just received plus the best Q-value we can get from the next state. This is our "improved estimate."
- $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ is the **TD error** — the difference between our improved estimate and our current estimate. If positive, we underestimated; if negative, we overestimated.

Let us plug in some simple numbers to see how the Q-table gets updated. Suppose we have two states ($s_1$, $s_2$) and two actions (Left, Right). Initially, all Q-values are 0. Learning rate $\alpha = 0.1$, discount factor $\gamma = 0.9$.

**Update 1:** In state $s_1$, take action Right, get reward $r = +5$, land in state $s_2$.

$$Q(s_1, \text{Right}) \leftarrow 0 + 0.1 \times [5 + 0.9 \times \max(0, 0) - 0]$$

$$Q(s_1, \text{Right}) \leftarrow 0 + 0.1 \times [5 + 0 - 0] = 0.5$$

**Update 2:** In state $s_2$, take action Left, get reward $r = +10$, reach terminal state.

$$Q(s_2, \text{Left}) \leftarrow 0 + 0.1 \times [10 + 0 - 0] = 1.0$$

**Update 3:** Back in state $s_1$, take action Right again, get reward $r = +5$, land in $s_2$.

$$Q(s_1, \text{Right}) \leftarrow 0.5 + 0.1 \times [5 + 0.9 \times 1.0 - 0.5]$$

$$Q(s_1, \text{Right}) \leftarrow 0.5 + 0.1 \times [5 + 0.9 - 0.5] = 0.5 + 0.54 = 1.04$$

See how the Q-value for $(s_1, \text{Right})$ increased from 0.5 to 1.04? This happened because the agent now knows that state $s_2$ has a high-value action (Left with Q = 1.0), so the value of being in $s_1$ and going Right — which leads to $s_2$ — also increases. This is the Bellman equation at work, learned from experience.

Now, an important property of Q-Learning is that it is an **off-policy** algorithm. Let us understand this with a simple analogy.

Imagine that you are learning to ride a bicycle. The way you ride now is your current policy. You are trying to get better at your style of riding. You take notes about what happens when you follow your current style. You are learning the same policy which you are following. This is called **on-policy** learning.

Now imagine that you are still riding the bike, but you are watching a video of a professional cyclist. You still ride around and try things, but when you update your understanding, you ask, "What would have happened if I had balanced perfectly like the expert?" So you are behaving one way (riding around) but learning about another way (the optimal policy). This is called **off-policy** learning.

Q-Learning is off-policy because the agent follows an epsilon-greedy policy (exploring sometimes), but the update rule always uses $\max_{a'} Q(s', a')$ — which is the greedy (optimal) action. The behavior policy explores, but the learning target is always the best possible action.


![Q-learning updates values after every single step, no waiting for episode end.](figures/figure_6.png)
*Q-learning updates values after every single step, no waiting for episode end.*


---

## Exploration vs Exploitation

There is one more important piece we need to understand before we can implement Q-Learning: the **epsilon-greedy strategy**.

The question is: how does the agent choose actions during training?

If the agent always picks the action with the highest Q-value (pure exploitation), it might miss discovering better actions that it has not tried yet.

If the agent always picks random actions (pure exploration), it will never use what it has learned.

The epsilon-greedy strategy balances both: with probability $\epsilon$, the agent picks a random action (exploration), and with probability $1 - \epsilon$, it picks the action with the highest Q-value (exploitation).

For example, if $\epsilon = 0.1$, then out of 100 moves, roughly 10 will be random exploratory moves. This makes sure that all actions are tried, and the estimated Q-values will converge to the true values.

A common practice is to start with a high epsilon (like 1.0) and gradually decay it towards a small value (like 0.01). Early in training, the agent explores heavily. As it learns more, it exploits its knowledge more and more.


![Exploration prevents the agent from getting stuck in suboptimal behavior.](figures/figure_7.png)
*Exploration prevents the agent from getting stuck in suboptimal behavior.*


---

## Practical Implementation: Q-Learning for FrozenLake

Now let us put everything together with a practical implementation. We will use the FrozenLake environment from OpenAI Gymnasium.

FrozenLake is a 4x4 grid where our agent must navigate from the start (top-left) to the goal (bottom-right) while avoiding holes in the ice. The ice is slippery, so the agent does not always move in the direction it chooses.

Here is the complete code for training a Q-Learning agent on FrozenLake:

```python
import gymnasium as gym
import numpy as np

# --- Hyperparameters ---
ALPHA = 0.1        # Learning rate
GAMMA = 0.99       # Discount factor
EPSILON = 1.0       # Initial exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
NUM_EPISODES = 10000

# --- Create Environment ---
env = gym.make("FrozenLake-v1", is_slippery=True)

# --- Initialize Q-table ---
Q = np.zeros((env.observation_space.n, env.action_space.n))

# --- Training Loop ---
rewards_per_episode = []

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < EPSILON:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])         # Exploit

        # Take action and observe result
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        td_target = reward + GAMMA * np.max(Q[next_state]) * (1 - terminated)
        td_error = td_target - Q[state, action]
        Q[state, action] += ALPHA * td_error

        total_reward += reward
        state = next_state

    # Decay epsilon
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    rewards_per_episode.append(total_reward)

print("Training complete!")
print(f"Success rate (last 100 episodes): {np.mean(rewards_per_episode[-100:]):.2%}")
```

Let us understand this code in detail.

First, we create the FrozenLake environment and initialize a Q-table with zeros. The Q-table has 16 rows (one for each state in the 4x4 grid) and 4 columns (one for each action: Left, Down, Right, Up).

In the training loop, for each episode, the agent starts at the initial state and keeps taking actions until the episode ends. The action selection follows the epsilon-greedy strategy: with probability $\epsilon$, it picks a random action; otherwise, it picks the action with the highest Q-value.

After each step, we apply the Q-Learning update rule. Notice the `(1 - terminated)` term — this ensures that when we reach a terminal state, we do not add the future value (because there is no future from a terminal state).

After each episode, we decay epsilon so the agent gradually shifts from exploration to exploitation.

Now let us visualize the training progress:

```python
import matplotlib.pyplot as plt

# Calculate moving average of rewards
window = 100
moving_avg = [np.mean(rewards_per_episode[max(0, i-window):i+1])
              for i in range(len(rewards_per_episode))]

plt.figure(figsize=(10, 5))
plt.plot(moving_avg)
plt.xlabel("Episode")
plt.ylabel("Success Rate (Moving Average)")
plt.title("Q-Learning on FrozenLake")
plt.grid(True)
plt.show()
```

When you run this code, you will see that the success rate starts near 0 and gradually increases as the agent learns. After a few thousand episodes, the agent typically reaches a success rate of 70-80% (the environment is stochastic due to the slippery ice, so 100% is not achievable).

We can also visualize the learned Q-table to see the policy:

```python
# Display the learned policy
policy = np.argmax(Q, axis=1)
action_names = ['Left', 'Down', 'Right', 'Up']
print("\nLearned Policy:")
for i in range(4):
    for j in range(4):
        state = i * 4 + j
        print(f"{action_names[policy[state]]:>6}", end=" ")
    print()
```

Not bad right? Our agent has learned a policy that navigates the slippery frozen lake with a decent success rate, all by interacting with the environment step by step.


![The agent learns to navigate the frozen lake after a few thousand episodes.](figures/figure_8.png)
*The agent learns to navigate the frozen lake after a few thousand episodes.*


---

## From Tables to Neural Networks

Now, our FrozenLake environment has only 16 states and 4 actions. The Q-table is tiny — just 16 rows and 4 columns.

But can you imagine what will happen if we have an environment which has a huge number of states?

For example, the game of chess has approximately $10^{46}$ possible states. It would be impossible for us to tabulate Q-values for all these states in an array. We would need more storage than all the atoms in the observable universe.

So, what is the practical solution?

Can we learn a function to represent the Q-values for state-action pairs based on a limited amount of experience and then extrapolate to unseen states?

Does this remind you of something? This is exactly what neural networks do — they learn functions from data and generalize to new inputs.

This idea of replacing the Q-table with a neural network is what led to **Deep Q-Networks (DQN)**, the algorithm that DeepMind used in 2013 to play Atari games directly from raw pixel inputs. The network takes the state as input and outputs Q-values for all possible actions.


![When the state space is huge, a neural network replaces the Q-table.](figures/figure_9.png)
*When the state space is huge, a neural network replaces the Q-table.*


---

## Summary

Let us recap the journey we have taken in this article:

1. **State value functions** $V^{\pi}(s)$ tell us how good it is to be in a state under a given policy.
2. **Action value functions** $Q^{\pi}(s, a)$ tell us how good it is to take a specific action in a state.
3. The **Bellman equation** decomposes value into immediate reward plus discounted future value — a beautiful recursive insight.
4. The **Bellman optimality equation** finds the best possible values by always picking the maximum action.
5. **Q-Learning** uses the Bellman optimality equation to learn Q-values directly from experience, without needing a model of the environment.
6. The **epsilon-greedy** strategy balances exploration and exploitation during learning.

The key insight that runs through everything is Bellman's recursive one-step-at-a-time decomposition. You do not need to think about the entire future — just think about the next step, and let the recursion handle the rest.

That's it!

Refer to this GitHub repo for reproducing the code: https://github.com/RajatDandekar/Hands-on-RL-Bootcamp-Vizuara

---

### References

1. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
2. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD Thesis, University of Cambridge.
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
4. Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602.
