# Building DQN Atari Agents: When Q-Learning Learned to See

How DeepMind combined Q-learning with convolutional neural networks to play Atari games at superhuman levels — and why this 2013 paper changed everything.

## The Tabletop Problem

Let us start with a question that came up naturally in our discussion of Q-learning.

We learned that Q-learning maintains a table of Q-values — one entry for every state-action pair. The agent looks up this table, picks the action with the highest Q-value, and updates the table after receiving a reward. Simple and elegant.

But what happens when the state space becomes enormous?

Consider the game of Breakout. The screen is 210 pixels tall and 160 pixels wide. Each pixel can take one of 128 possible color values. The total number of possible screen configurations is:


$$|S| = 128^{210 \times 160} = 128^{33600}$$

Let us plug in some numbers to appreciate how absurd this is. Even if we simplified the game to a tiny 4x4 grayscale grid with just 2 possible values per pixel (black or white), the state space would be $2^{16} = 65{,}536$ states. For a 10x10 grid with 4 colors, we get $4^{100} \approx 10^{60}$ states — more than the number of atoms in the observable universe.

This tells us that storing Q-values in a table is absolutely impossible for pixel-based observations. We need a fundamentally different approach.

So, what is the practical solution?

Can we learn a function that takes in any state and outputs Q-values, without having to store every single state-action pair?


![Tabular Q-learning breaks down when the state space explodes.](figures/figure_1.png)
*Tabular Q-learning breaks down when the state space explodes.*


## Function Approximation — The Key Idea

Instead of memorizing every face you have ever seen, your brain learns general features — the shape of eyes, the curve of a nose, the structure of a mouth — and combines them to recognize new faces instantly. You do not need a separate memory for every face.

This is exactly the idea behind function approximation in reinforcement learning.

Instead of maintaining a table, we use a parameterized function $Q(s, a; \theta)$ to approximate the true Q-values. Here, $\theta$ represents the parameters (weights) of the function. The goal is:


$$
Q(s, a; \theta) \approx Q^*(s, a)
$$


Let us plug in some simple numbers to understand this. Suppose we have 3 states and 2 actions. The true optimal Q-values might be:

| State | Action 0 | Action 1 |
|-------|----------|----------|
| s1    | 2.0      | 5.0      |
| s2    | 3.0      | 1.0      |
| s3    | 4.0      | 4.5      |

Instead of storing all 6 values, we train a function (say, a neural network) that takes a state as input and outputs the Q-values for all actions. If we feed it state $s_1$, it should output something close to $[2.0, 5.0]$. The network learns the underlying patterns and can generalize — it can estimate Q-values even for states it has never seen before.

Now the question is: what kind of function should we use?

If our states are raw pixel images — like Atari game screens — then we need a function that is good at processing spatial patterns in images. This brings us to convolutional neural networks.


![A neural network replaces the Q-table entirely.](figures/figure_2.png)
*A neural network replaces the Q-table entirely.*


## The 2013 DeepMind Paper — Why It Was Groundbreaking

The Q-learning algorithm was developed by Chris Watkins in 1992. It was a beautiful theoretical result, but for over two decades, it remained limited to problems with small, discrete state spaces.

In 2013, a small London-based startup called DeepMind published a paper titled "Playing Atari with Deep Reinforcement Learning" by Volodymyr Mnih and colleagues. This paper combined Q-learning with deep convolutional neural networks to create what we now call **Deep Q-Networks** or **DQN**.

What made this paper so remarkable?

First, the agent received only raw pixel values as input — no hand-crafted features, no game-specific knowledge. Second, the same neural network architecture and the same set of hyperparameters were used across all games. Third, the agent achieved superhuman performance on many of the 49 Atari games it was tested on.

In 2015, an extended version of this work was published in Nature — one of the most prestigious scientific journals in the world. This was a strong signal that deep reinforcement learning had arrived.

This result was so impressive that Google acquired DeepMind for approximately \$500 million shortly after the 2013 paper.


![DQN bridged a 20-year gap between Q-learning and deep learning.](figures/figure_3.png)
*DQN bridged a 20-year gap between Q-learning and deep learning.*


## The DQN Architecture

Let us now look at the neural network architecture that made all this possible.

The input to the DQN is not a single game frame but a stack of 4 consecutive frames. Why 4 frames? Because a single frame does not capture motion. If you see a screenshot of Pong with the ball in the middle of the screen, you cannot tell whether the ball is moving left or right. But if you see 4 consecutive frames, the ball's trajectory becomes clear.

Each frame is preprocessed: converted to grayscale and resized to 84x84 pixels. So the input tensor has the shape $(4, 84, 84)$ — 4 channels of 84x84 images.

This input is then passed through three convolutional layers:

**Conv Layer 1:** 32 filters of size 8x8 with stride 4. This layer detects low-level features like edges and simple shapes.

**Conv Layer 2:** 64 filters of size 4x4 with stride 2. This layer combines low-level features into more complex patterns.

**Conv Layer 3:** 64 filters of size 3x3 with stride 1. This layer captures high-level spatial relationships.

After the convolutional layers, the output is flattened and passed through a fully connected layer with 512 neurons. The final output layer produces one Q-value for each possible action.

The output size of a convolutional layer is given by:


$$
o = \frac{i - k + 2p}{s} + 1
$$


where $i$ is the input size, $k$ is the kernel size, $p$ is the padding (0 in DQN), and $s$ is the stride.

Let us trace through the actual dimensions. Starting with an 84x84 input:

- After Conv1 (8x8 kernel, stride 4): $o = \frac{84 - 8}{4} + 1 = 20$. Output: 32 channels of 20x20.
- After Conv2 (4x4 kernel, stride 2): $o = \frac{20 - 4}{2} + 1 = 9$. Output: 64 channels of 9x9.
- After Conv3 (3x3 kernel, stride 1): $o = \frac{9 - 3}{1} + 1 = 7$. Output: 64 channels of 7x7.

The flattened size is $64 \times 7 \times 7 = 3136$ neurons. This is exactly what gets fed into the fully connected layer.


![The DQN architecture processes raw pixels into action Q-values.](figures/figure_4.png)
*The DQN architecture processes raw pixels into action Q-values.*


Here is the PyTorch implementation of this network:

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # x shape: (batch, 4, 84, 84), pixel values in [0, 1]
        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.size(0), -1)  # Flatten
        return self.fc(flat)
```

The network takes in a batch of 4-frame stacks and outputs Q-values for each action. This is exactly what we want — a function that maps raw pixel observations to action values.

## Experience Replay — Learning from Memories

Now that we have a neural network that can approximate Q-values, we need to train it. But training a neural network with Q-learning turns out to be surprisingly tricky.

Let us understand the problem. In standard Q-learning, the agent plays the game step by step. At each step, it observes a state, takes an action, receives a reward, and transitions to a new state. If we train the network on each transition as it happens, we run into two problems.

**Problem 1: Correlated data.** Consecutive game frames are highly correlated — frame 100 looks almost identical to frame 101. Neural networks trained on correlated data tend to forget earlier experiences. This is called catastrophic forgetting.

**Problem 2: Non-stationary distribution.** As the agent's policy improves, the distribution of states it visits changes. The network has to keep adapting to a shifting data distribution.

The solution is brilliantly simple: **experience replay**.

Imagine you are studying for an exam. If you only study the last chapter you read, you will forget the earlier ones. Instead, you keep flashcards of all chapters and review them in random order. This way, you retain knowledge from all chapters equally.

Experience replay works the same way. The agent stores every transition $(s_t, a_t, r_t, s_{t+1})$ in a large circular buffer called the replay memory. During training, instead of learning from the most recent transition, the agent samples a random mini-batch of transitions from this buffer.

The transitions are sampled uniformly at random:


$$
(s_j, a_j, r_j, s_{j+1}) \sim \text{Uniform}(\mathcal{D})
$$


Let us see this with a concrete example. Suppose our replay buffer has capacity 5 and we have stored the following transitions:

| Index | State | Action | Reward | Next State |
|-------|-------|--------|--------|------------|
| 0     | s_A   | left   | 0      | s_B        |
| 1     | s_B   | right  | 1      | s_C        |
| 2     | s_C   | left   | 0      | s_D        |
| 3     | s_D   | right  | -1     | s_E        |
| 4     | s_E   | left   | 5      | s_F        |

If we sample a mini-batch of size 2, we might randomly pick transitions at indices 1 and 4. This gives us one transition from the middle of an episode and one from the end — completely uncorrelated. This is exactly what we want.


![Experience replay breaks correlations by sampling random memories.](figures/figure_5.png)
*Experience replay breaks correlations by sampling random memories.*


Here is the implementation:

```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.buffer)
```

## The Target Network — Stability Through Patience

There is one more problem we need to solve. Let us look at the Q-learning update rule to understand it.

In Q-learning, we update the Q-value towards a target:

$$\text{target} = r + \gamma \max_{a'} Q(s', a'; \theta)$$

Notice that the target itself depends on the same network parameters $\theta$ that we are trying to update. Every time we update $\theta$ to make $Q(s, a; \theta)$ closer to the target, the target itself shifts because it also depends on $\theta$.

Imagine you are trying to hit a target with an arrow, but every time you release the arrow, the target moves. You would never converge — you would just keep chasing a moving target.

The solution is to freeze the target.

DQN maintains two separate copies of the neural network:

1. **Online network** (parameters $\theta$): This is the network we train at every step.
2. **Target network** (parameters $\theta^-$): This is a frozen copy that provides stable targets. It is updated only every $C$ steps by copying the online network's weights.

The DQN loss function is:


$$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

This is a mean squared error loss. The term inside the square is called the **temporal difference (TD) error**.

Let us compute this for a single transition to make it concrete. Suppose we have:
- Current state $s$, action $a = 1$, reward $r = 3$, next state $s'$
- The online network predicts: $Q(s, a=1; \theta) = 7.0$
- The target network predicts: $Q(s', a'=0; \theta^-) = 5.0$, $Q(s', a'=1; \theta^-) = 8.0$
- Discount factor $\gamma = 0.99$

The target value is: $r + \gamma \max_{a'} Q(s', a'; \theta^-) = 3 + 0.99 \times 8.0 = 3 + 7.92 = 10.92$

The TD error is: $10.92 - 7.0 = 3.92$

The loss for this transition is: $(3.92)^2 = 15.37$

This tells us that our online network's prediction of 7.0 is too low — the target says the true value should be around 10.92. The gradient descent step will push the prediction upwards.

The target network is updated by simply copying the weights:


$$\theta^- \leftarrow \theta \quad \text{every } C \text{ steps}$$

In the original DQN paper, $C = 10{,}000$ steps. This means the target stays frozen for 10,000 training steps before being refreshed.


![The target network provides stable Q-value targets.](figures/figure_6.png)
*The target network provides stable Q-value targets.*


## The DQN Training Algorithm

Now we have all the pieces of the puzzle ready. Let us put them together into the complete DQN training algorithm.

The algorithm works as follows:

1. Initialize the online network $Q(s, a; \theta)$ with random weights
2. Initialize the target network $Q(s, a; \theta^-)$ as a copy of the online network
3. Initialize the replay buffer $\mathcal{D}$ with capacity $N$
4. For each episode:
   - Reset the environment and get initial observation
   - For each time step:
     - With probability $\epsilon$, select a random action; otherwise, select $a = \arg\max_a Q(s, a; \theta)$
     - Execute action $a$, observe reward $r$ and next state $s'$
     - Store transition $(s, a, r, s')$ in the replay buffer
     - Sample a random mini-batch from the replay buffer
     - Compute the loss using the target network
     - Perform gradient descent on the online network
     - Every $C$ steps, copy the online network weights to the target network


![The DQN training loop combines exploration, replay, and stable targets.](figures/figure_7.png)
*The DQN training loop combines exploration, replay, and stable targets.*


Here is the core training loop:

```python
import torch.optim as optim
import torch.nn.functional as F

GAMMA, BATCH_SIZE, BUFFER_SIZE = 0.99, 32, 100_000
LR, TARGET_UPDATE = 1e-4, 1000
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.1, 100_000

online_net = DQN(n_actions=4).to(device)
target_net = DQN(n_actions=4).to(device)
target_net.load_state_dict(online_net.state_dict())
optimizer = optim.Adam(online_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_SIZE)
step = 0

for episode in range(num_episodes):
    state, done = env.reset(), False
    episode_reward = 0

    while not done:
        # Epsilon-greedy action selection
        eps = max(EPS_END, EPS_START - step / EPS_DECAY)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = online_net(state.unsqueeze(0).to(device))
                action = q.argmax(1).item()

        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        step += 1

        # Training step
        if len(replay_buffer) >= BATCH_SIZE:
            s, a, r, s2, d = replay_buffer.sample(BATCH_SIZE)
            q_val = online_net(s.to(device))
            q_val = q_val.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = target_net(s2.to(device)).max(1)[0]
                target = r.to(device) + GAMMA * next_q * (~d).to(device)

            loss = F.mse_loss(q_val, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network periodically
        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(online_net.state_dict())
```

In the above code, you can see that we are using the exact same components we discussed: epsilon-greedy exploration, experience replay, and the target network for computing stable targets. This is exactly what the DQN algorithm describes.

## Epsilon-Greedy Exploration in DQN

Let us briefly discuss the exploration strategy used by DQN.

We know from our discussion of Q-learning that we need a balance between exploration (trying new actions) and exploitation (using what we already know). DQN uses the epsilon-greedy strategy, where with probability $\epsilon$ the agent selects a random action, and with probability $1 - \epsilon$ the agent selects the action with the highest Q-value.

The key insight in DQN is that epsilon is not fixed — it is annealed over time. At the start of training, $\epsilon = 1.0$, meaning the agent takes completely random actions. Over the course of training, epsilon is linearly decreased to $\epsilon = 0.1$.

The schedule is:


$$\epsilon(t) = \max\left(\epsilon_{\text{end}},\; \epsilon_{\text{start}} - t \cdot \frac{\epsilon_{\text{start}} - \epsilon_{\text{end}}}{\text{decay\_steps}}\right)$$

Let us plug in some numbers. With $\epsilon_{\text{start}} = 1.0$, $\epsilon_{\text{end}} = 0.1$, and $\text{decay\_steps} = 100{,}000$:

- At $t = 0$: $\epsilon = 1.0$ (fully random)
- At $t = 50{,}000$: $\epsilon = 1.0 - 50{,}000 \times \frac{0.9}{100{,}000} = 1.0 - 0.45 = 0.55$
- At $t = 100{,}000$: $\epsilon = 0.1$ (mostly greedy)
- At $t = 200{,}000$: $\epsilon = 0.1$ (stays at minimum)

This makes sense because early in training, the Q-values are random and unreliable. Exploring randomly is the best strategy. As training progresses and the Q-values become more accurate, the agent should exploit its learned knowledge more often. This is exactly what epsilon annealing achieves.


![Epsilon anneals from random exploration to mostly exploitation.](figures/figure_8.png)
*Epsilon anneals from random exploration to mostly exploitation.*


## Preprocessing — How DQN Sees the World

The raw output from an Atari game is a 210x160 RGB image — three color channels, each with values from 0 to 255. This is too large and contains too much redundant information for efficient learning.

DQN applies the following preprocessing steps:

**Step 1: Grayscale conversion.** Convert the RGB image to a single grayscale channel. This reduces the input size by 3x. For most Atari games, color provides minimal additional information — the positions and shapes of objects are what matter.

**Step 2: Resize to 84x84.** The 210x160 image is cropped to a square region and resized to 84x84 pixels. This further reduces the input size while retaining enough detail for the game.

**Step 3: Frame stacking.** Stack the 4 most recent frames together to form a single $(4, 84, 84)$ tensor. This provides the network with temporal information — it can "see" how objects have moved across the last 4 frames.


![Preprocessing transforms raw game frames into network-ready input.](figures/figure_9.png)
*Preprocessing transforms raw game frames into network-ready input.*


Here is the preprocessing code:

```python
import cv2
import numpy as np

class AtariPreprocessor:
    def __init__(self, frame_stack=4, resize_shape=(84, 84)):
        self.frame_stack = frame_stack
        self.resize_shape = resize_shape
        self.frames = deque(maxlen=frame_stack)

    def preprocess_frame(self, frame):
        # Convert RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        resized = cv2.resize(gray, self.resize_shape)
        # Normalize pixel values to [0, 1]
        return resized.astype(np.float32) / 255.0

    def reset(self, frame):
        processed = self.preprocess_frame(frame)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return torch.tensor(np.array(self.frames))

    def step(self, frame):
        processed = self.preprocess_frame(frame)
        self.frames.append(processed)
        return torch.tensor(np.array(self.frames))
```

## Practical Implementation — Training a DQN Agent on Pong

Let us now put everything together and train a DQN agent to play the game of Pong.

Pong is a classic game where two paddles hit a ball back and forth. The agent controls one paddle and the opponent is controlled by the game AI. The score ranges from -21 (complete loss) to +21 (complete win). A random agent scores about -21 because it almost never hits the ball.

To set up the environment, we use the Gymnasium library with the Atari Learning Environment (ALE):

```python
import gymnasium as gym

# Create the Pong environment
env = gym.make("PongNoFrameskip-v4")

# Get number of actions
n_actions = env.action_space.n  # 6 actions in Pong
print(f"Number of actions: {n_actions}")
print(f"Observation space: {env.observation_space.shape}")
```

With the network, replay buffer, preprocessing, and training loop we defined earlier, we train the agent for approximately 1 million frames. After training, the reward curve should look something like this:


![DQN learns to win at Pong after training on raw pixels.](figures/figure_10.png)
*DQN learns to win at Pong after training on raw pixels.*


We can clearly see that the reward increases with time. In the beginning, the agent performs terribly (reward near -21), but after about 400,000 frames of training, it starts winning rallies. By 800,000 frames, it consistently wins games with a reward near +20. Not bad right?

The remarkable thing is that the agent learned all of this from raw pixels alone — no one told it what a paddle is, what a ball is, or what the rules of Pong are. It figured out the optimal strategy purely through trial, error, and the reward signal.

## Results Across Games — The Power of Generalization

DQN was tested on 49 Atari games using the exact same architecture and hyperparameters. The results were remarkable.

The agent achieved superhuman performance on 29 out of 49 games. On games like Breakout, Pong, and Space Invaders — games that require quick reactions and pattern recognition — DQN excelled dramatically. In Breakout, the agent even discovered the optimal strategy of tunneling through the bricks to bounce the ball behind the wall, a strategy that many human players do not think of.

However, DQN struggled on games like Montezuma's Revenge, which requires long-term planning, remembering a sequence of steps, and dealing with very sparse rewards (the agent might play for thousands of frames before receiving any reward). This highlights a fundamental limitation: DQN is great at reactive play but weak at strategic planning.


![DQN achieves superhuman performance on many Atari games.](figures/figure_11.png)
*DQN achieves superhuman performance on many Atari games.*


## Why DQN Matters — The Bigger Picture

DQN was not just a cool demo — it was a paradigm shift. It demonstrated for the first time that a single reinforcement learning agent could learn directly from high-dimensional sensory input (raw pixels) to achieve human-level performance across a diverse set of tasks.

The three key innovations — **function approximation with deep CNNs**, **experience replay**, and **target networks** — are now standard components in modern deep reinforcement learning. These same principles appear in algorithms like:

- **Double DQN:** Addresses overestimation bias in Q-value targets
- **Dueling DQN:** Separates state-value and advantage estimation
- **Prioritized Experience Replay:** Samples important transitions more often
- **Rainbow:** Combines all improvements into a single agent

The success of DQN showed the world that deep learning and reinforcement learning could be combined to solve problems that neither approach could handle alone. It opened the door to AlphaGo, robotic manipulation, and the modern era of reinforcement learning for language models that we explore in later articles.

That's it! We have covered the complete DQN algorithm from scratch — from the limitation of tabular Q-learning, through function approximation, to the three innovations (CNNs, experience replay, target networks) that made deep Q-learning practical. The same architecture that played Pong from raw pixels in 2013 laid the foundation for the deep reinforcement learning revolution that followed.

## References

1. Mnih, V., et al. (2013). "Playing Atari with Deep Reinforcement Learning." arXiv preprint arXiv:1312.5602.
2. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529–533.
3. GitHub Repository: [https://github.com/RajatDandekar/Hands-on-RL-Bootcamp-Vizuara](https://github.com/RajatDandekar/Hands-on-RL-Bootcamp-Vizuara)
