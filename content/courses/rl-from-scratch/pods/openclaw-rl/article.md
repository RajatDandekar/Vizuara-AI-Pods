# OpenClaw-RL: Train a Personalized AI Agent Simply by Talking to It

*How reinforcement learning turns your everyday conversations into training signal — and why your AI assistant should get better the more you use it.*

---

## What is OpenClaw?

Let us start by understanding what OpenClaw is.

Imagine having a single AI assistant that lives on your own machine. It connects to your WhatsApp, Telegram, Slack, and Discord. It can control your browser, read your files, manage your calendar, and even build new skills on the fly.

This is exactly what OpenClaw does.

OpenClaw is an open-source personal AI agent framework. It runs entirely on your own hardware — no cloud dependency, no API keys to external providers, and your data never leaves your machine.


![OpenClaw connects a self-hosted LLM to 50+ services as your personal AI agent.](figures/figure_1.png)
*OpenClaw connects a self-hosted LLM to 50+ services as your personal AI agent.*


Here is what makes OpenClaw powerful:

1. **50+ integrations**: Gmail, GitHub, Spotify, Google Calendar, and many more — all accessible through a single conversational interface.

2. **Browser control**: OpenClaw can navigate websites, fill out forms, and extract information from web pages — all autonomously.

3. **Persistent memory**: Unlike standard chatbots, OpenClaw remembers your context across conversations. It knows what you discussed yesterday.

4. **Autonomous skill-building**: When OpenClaw encounters a task it cannot handle, it can build a new skill for itself. It identifies capability gaps and writes the code to fill them.

5. **Self-hosted LLM**: At its core, OpenClaw wraps any self-hosted language model as an OpenAI-compatible API. You can use Qwen3-4B, LLaMA, DeepSeek, or any model that fits your hardware.

This is truly amazing. You get a personal AI agent that controls your entire digital life — and it all runs locally.

But there is one fundamental limitation.

The language model behind OpenClaw is generic. It was trained on internet-scale data to serve everyone. It does not know *your* preferences. It does not know that you prefer Python over JavaScript. It does not know that you like concise answers, not long essays. It does not know your coding style, your writing tone, or the way you like your emails drafted.

This brings us to the central question of this article: **Can we make OpenClaw learn from you?**

---

## The Problem: Your AI Assistant Never Learns

OpenClaw gives us a powerful personal agent. But there is a fundamental problem: the model behind it never learns from you.

Let us take a concrete example.

Suppose you use OpenClaw every day to help you write code. Every single time, you tell it: "Don't use TypeScript, I want plain JavaScript." You correct it again and again. You give it explicit feedback in natural language.

But the next day, it suggests TypeScript again. Sound familiar? :)

Have a look at the diagram below:

![Without RL, your feedback is lost; with OpenClaw-RL, every conversation makes the model better.](figures/figure_2.png)
*Without RL, your feedback is lost; with OpenClaw-RL, every conversation makes the model better.*


This happens because current language models are **frozen after training**. Once the weights are set, they do not change based on your interactions. Your corrections, your preferences, your feedback — it all disappears into the void.

You might be thinking: "But what about fine-tuning? Can't I just fine-tune the model on my data?"

Yes, you could. But traditional fine-tuning requires you to:
- Manually collect and label training data
- Stop the model, run a training job, and restart it
- Repeat this process every time your preferences change

This is not practical for a personal assistant that you use every day.

What we really want is an AI that improves **continuously**, **in the background**, **from your natural conversations** — without you having to do anything special.

This is exactly what OpenClaw-RL does.

OpenClaw-RL is a fully asynchronous reinforcement learning framework that wraps your OpenClaw agent and turns your everyday conversations into training signal. You just use your assistant normally. Behind the scenes, the system collects your interactions, evaluates the model's responses, and trains the model to better match your preferences.

The model literally gets better the more you use it.

---

## The Big Picture: Four Asynchronous Components

So how does OpenClaw-RL actually work?

The key insight is that the system is built from **four independent components**, and none of them block each other. While you are chatting with your assistant, training is happening in the background. The model serves your requests and improves simultaneously.

Let us look at each component.


![The four async components of OpenClaw-RL run independently on dedicated GPUs.](figures/figure_3.png)
*The four async components of OpenClaw-RL run independently on dedicated GPUs.*


**Component 1: Agent Serving.** This is the model that you talk to. It wraps your self-hosted LLM as an OpenAI-compatible API and serves your requests in real time. It also records the token-level log-probabilities for every response it generates — these will be needed for training later. By default, this uses 4 GPUs.

**Component 2: Rollout Collection.** This component intercepts your live conversations. It tracks each conversation as a session, classifies every turn, and identifies which turns can be used for training. It uses your next message as the natural "feedback signal" for the model's previous response. We will dive deeper into this in the next section.

**Component 3: PRM Judging.** The Process Reward Model (PRM) is a separate model that evaluates how good each of the agent's responses was. It runs asynchronously — it does not need to evaluate responses in real time. It uses majority voting across multiple evaluations for robustness. By default, this uses 2 GPUs.

**Component 4: Policy Training.** This is where the actual learning happens. The trainer takes the evaluated samples and updates the model weights using reinforcement learning. It runs continuously in the background. When it is ready to push a new set of weights, the serving component pauses briefly, loads the new weights, and resumes — no data corruption, no service interruption.

The beauty of this design is that **you never notice any of it**. You just chat with your assistant. Everything else happens behind the scenes.

Under the hood, OpenClaw-RL is built on the **Slime** distributed RL framework, which handles the heavy lifting of distributing training and inference across multiple GPUs.

---

## Rollout Collection: Turning Conversations into Training Data

Now let us understand how everyday conversations become training data.

This is one of the most clever parts of OpenClaw-RL. Most RL systems require pre-collected datasets with explicit labels. OpenClaw-RL does not. It uses your live conversations directly.

Here is how it works.

**Session-aware tracking.** Every conversation you have with OpenClaw is tracked as a session. The system maintains the correct turn ordering within each session. This is important because in a multi-turn conversation, the context of earlier turns affects the meaning of later turns.

**Turn classification.** Not every message in a conversation is useful for training. The rollout collector classifies each API message into two categories:

- **Main-line turns** (trainable): These are the core interaction turns where the agent generates a substantive response and you provide meaningful feedback.
- **Side turns** (non-trainable): These are metadata requests, system messages, or other turns that do not carry useful training signal.

**The next-state signal.** This is the key idea. In traditional RL, you need a separate reward signal to tell the agent how well it did. In OpenClaw-RL, **the next thing you say after the model responds IS the reward signal**.

Let us see this with an example.


![A conversation decomposed into trainable turns with next-state feedback signals.](figures/figure_4.png)
*A conversation decomposed into trainable turns with next-state feedback signals.*


Notice what happened here:

- When the agent responded with JavaScript (Turn 2), the user corrected it with "No, I said Python" (Turn 3). This correction **is** the negative feedback signal for Turn 2.
- When the agent fixed its response (Turn 4), the user said "Perfect, thanks!" (Turn 5). This **is** the positive feedback signal for Turn 4.

No manual labeling was needed. No thumbs-up or thumbs-down buttons. The feedback was embedded naturally in the conversation.

The rollout collector processes these signals and submits ready training samples to the trainer as they become available. It does not wait for the full conversation to end — samples are submitted incrementally.

All conversations and PRM evaluations are logged to JSONL files for analysis and debugging. If you ever want to inspect what the system learned from your conversations, you can look at these logs.

---

## Binary RL with GRPO: Scoring Responses as Good or Bad

Now that we have our conversation data, the question is: how do we actually score the agent's responses and use those scores to improve the model?

This brings us to the first of OpenClaw-RL's two learning paradigms: **Binary RL with GRPO**.

Let us briefly recap how GRPO works. GRPO stands for Group-Relative Policy Optimization. In standard RL for language models, an algorithm called PPO (Proximal Policy Optimization) is used. PPO requires training a separate **critic network** — essentially a second neural network that estimates how good each response is. This doubles the memory and compute requirements.

GRPO eliminates the critic entirely. Instead, it **samples a group of G responses** for the same prompt, computes a reward for each one, and normalizes the rewards within the group to get advantages. This cuts memory and compute by roughly 50%. Not bad right? :)

The advantage for the i-th response in the group is computed as:


$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

Let us plug in some simple numbers to see how this works.

Suppose we have G = 4 responses to the same user prompt, and the PRM assigns rewards: $$r_1 = +1, r_2 = -1, r_3 = +1, r_4 = 0$$.

The mean reward is $$(1 + (-1) + 1 + 0) / 4 = 0.25$$, and the standard deviation is $$0.83$$.

Now we normalize each reward:

- $$\hat{A}_1 = (1 - 0.25) / 0.83 = 0.90$$
- $$\hat{A}_2 = (-1 - 0.25) / 0.83 = -1.51$$
- $$\hat{A}_3 = (1 - 0.25) / 0.83 = 0.90$$
- $$\hat{A}_4 = (0 - 0.25) / 0.83 = -0.30$$

This tells us that responses 1 and 3 were above average (positive advantage — increase their probability), response 2 was well below average (strong negative advantage — decrease its probability), and response 4 was slightly below average. This is exactly what we want.

Now, where do these rewards come from? This is where the **Process Reward Model (PRM)** comes in.

The PRM is a separate language model that evaluates each (response, next-state) pair. It looks at what the agent said, then looks at what the user said next, and judges whether the response was good, neutral, or bad.

But a single evaluation can be noisy. To make the scoring robust, OpenClaw-RL runs the PRM **multiple times** on the same (response, next-state) pair and uses **majority voting**. Have a look at the diagram below:

![Majority voting across PRM evaluations produces robust reward signals.](figures/figure_5.png)
*Majority voting across PRM evaluations produces robust reward signals.*


With the advantages computed, the model is updated using the GRPO clipped surrogate loss:


$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\min\left(\rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) \hat{A}_t\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

Here, $$\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\text{ref}}(a_t \mid s_t)$$ is the ratio between the current policy and the reference policy, $$\hat{A}_t$$ is the group-relative advantage, $$\epsilon_{\text{low}}$$ and $$\epsilon_{\text{high}}$$ are the clipping bounds, and $$\beta$$ controls the KL penalty. KL divergence measures how much the updated policy has drifted from the original — if the model changes too drastically, this penalty pulls it back. This prevents the kind of reward hacking where the model learns to exploit the reward signal without actually improving.

Let us plug in some numbers. Suppose $$\rho_t = 1.4$$, $$\hat{A}_t = 0.90$$, $$\epsilon_{\text{low}} = 0.2$$, $$\epsilon_{\text{high}} = 0.28$$.

- Unclipped term: $$1.4 \times 0.90 = 1.26$$
- Clip bounds: $$[1 - 0.2, 1 + 0.28] = [0.8, 1.28]$$
- Clipped ratio: $$\text{clip}(1.4, 0.8, 1.28) = 1.28$$
- Clipped term: $$1.28 \times 0.90 = 1.15$$
- Final loss contribution: $$\min(1.26, 1.15) = 1.15$$

The clipping prevented the policy from jumping too aggressively from $$\rho_t = 1.4$$ — it was limited to an effective ratio of 1.28. This keeps training stable.

Notice the **asymmetric clipping**: $$\epsilon_{\text{high}} = 0.28 > \epsilon_{\text{low}} = 0.2$$. This is called the **clip-higher** technique. It allows larger positive updates (encouraging good behavior) while being more conservative on negative updates. The optimal range for $$\epsilon_{\text{high}}$$ is between 0.28 and 0.315.

One final important detail: OpenClaw-RL provides an **"at-least-one guarantee"** — every conversation session contributes at least one effective training sample. This ensures that no interaction goes to waste.

---

## On-Policy Distillation: Learning *What* to Improve

Binary RL tells the model whether a response was good or bad. But it does not tell the model **what** to improve.

Think about it. If you tell someone "that was bad" without any further explanation, they know they made a mistake, but they have no idea what to do differently. A much better teacher would say: "That was bad because you used JavaScript instead of Python — next time, check the language preference first."

This brings us to the second learning paradigm in OpenClaw-RL: **On-Policy Distillation (OPD)**.

The core idea is beautiful. Instead of reducing the user's feedback to a scalar number (+1 or -1), OPD extracts a **textual hint** from the user's next message and uses it to create a richer, **token-level** training signal.

Here is how it works, step by step.

**Step 1:** The agent responds to your prompt.

**Step 2:** You give feedback in your next message. For example: "No, I asked for Python not JavaScript."

**Step 3:** A judge model analyzes your feedback and extracts a short **hindsight hint** — a concise description of what should have been different. For example: "The user wants Python code, not JavaScript."

**Step 4:** The system constructs an **enhanced prompt**: your original prompt + the hint. This enhanced prompt is what the model *should have seen* to get it right the first time.

**Step 5:** The same model is run on the enhanced prompt to produce a **teacher distribution** — the log-probabilities it would assign to each token if it had known the hint.

**Step 6:** The token-level advantage is computed as the gap between what the teacher assigns and what the student originally assigned:


$$A_t = \log \pi_{\text{teacher}}(a_t \mid s + \text{hint}) - \log \pi_\theta(a_t \mid s)$$

Let us plug in some simple numbers. Suppose at token position $$t$$, the teacher (who has the hint) assigns $$\log \pi_{\text{teacher}} = -0.5$$ to the correct token. The student (without the hint) assigns $$\log \pi_\theta = -2.3$$.

Then $$A_t = -0.5 - (-2.3) = 1.8$$.

This large positive advantage tells the student: "You should **strongly** increase the probability of this token — the teacher who had the hint was much more confident about it than you were."

At another token position, suppose the teacher assigns $$-1.2$$ and the student assigns $$-1.0$$. Then $$A_t = -1.2 - (-1.0) = -0.2$$. This small negative value means: "You were actually slightly better than the teacher here — barely any correction needed."

This is the power of OPD: it provides **directional guidance at every token position**, not just a single scalar for the entire response.


![On-Policy Distillation extracts directional token-level signals from conversation feedback.](figures/figure_6.png)
*On-Policy Distillation extracts directional token-level signals from conversation feedback.*


There are two important engineering details that make OPD work well in practice:

**Hint quality filtering.** The judge generates multiple hints (m votes), and only the longest, most informative hint is kept. Trivial hints like "the response was wrong" are discarded — they do not add directional value beyond what Binary RL already provides.

**Memory optimization.** When computing the teacher's log-probabilities, only the response-suffix probabilities are calculated, not the full sequence. This significantly reduces peak GPU memory usage.

OPD uses the exact same PPO-style clipped surrogate loss as Binary RL — the only difference is that the advantages are token-level and directional instead of being a single broadcasted scalar.

Now, when should you use Binary RL versus OPD? The following table summarizes the key differences:

![Choose the right paradigm based on your feedback style.](figures/figure_7.png)
*Choose the right paradigm based on your feedback style.*


The rule of thumb is simple: if you mostly give implicit feedback (thumbs up, task success/failure), use Binary RL. If you give concrete textual corrections ("I wanted Python, not JavaScript"), use OPD.

---

## The RLAnything Closed Loop: Co-Optimizing Everything

So far, we have treated the Process Reward Model as a fixed judge. The policy improves, but the PRM stays the same.

But what if the reward model itself could improve alongside the policy?

This is the idea behind **RLAnything**, the theoretical framework that underpins OpenClaw-RL. In the RLAnything framework, three components — the **policy**, the **reward model**, and the **environment** — all co-optimize simultaneously.

Let us understand how this works.

**Integrated feedback.** Instead of relying solely on the PRM's step-wise evaluations, RLAnything combines them with outcome rewards. The integrated reward for step $$i$$ of trajectory $$\tau$$ is:


$$R_{\tau_i} = O_\tau + \frac{\lambda}{m}\sum_{j=1}^{m} S_{\tau_i, j}$$

Here, $$O_\tau \in \{-1, +1\}$$ is the outcome reward (did the overall task succeed?), $$S_{\tau_i, j} \in \{-1, +1\}$$ is the j-th PRM evaluation for step $$i$$, $$m$$ is the number of PRM evaluations, and $$\lambda$$ (default 1) balances outcome vs. step-wise signals.

Let us plug in some numbers. Suppose the task succeeded ($$O_\tau = +1$$), and 3 PRM evaluations scored this particular step as $$[+1, +1, -1]$$.

$$R_{\tau_i} = 1 + \frac{1}{3}(1 + 1 + (-1)) = 1 + \frac{1}{3} = 1.33$$

The outcome reward (+1) is augmented by the PRM consensus (+0.33). This gives us a richer signal than either source alone.

**Consistency feedback for the reward model.** Here is the clever part. How does the reward model itself improve? Through a self-consistency signal:


$$R_{S_{\tau_i, j}} = R_{\tau_i} \cdot S_{\tau_i, j}$$

Let us see what this means with our numbers. If $$R_{\tau_i} = 1.33$$ (the step was good) and the PRM evaluation said $$S = +1$$ (good), the consistency reward is $$1.33 \times 1 = 1.33$$. The PRM was correct, so it gets rewarded.

But if the PRM said $$S = -1$$ (bad), the consistency reward is $$1.33 \times (-1) = -1.33$$. The PRM was wrong — it said the step was bad when it was actually good. So it gets penalized and learns to judge more accurately.

Over time, this creates a virtuous cycle: a better policy generates more informative trajectories for the reward model, and a better reward model provides more precise supervision for the policy.

**Automatic environment adaptation.** The third component in the loop is the environment itself. If the policy succeeds at more than 80% of tasks, the environment automatically generates harder tasks. If the policy succeeds at less than 20%, easier tasks are generated. This keeps the difficulty balanced, ensuring the model always has room to improve.


![The RLAnything closed loop where policy, reward model, and environment co-evolve.](figures/figure_8.png)
*The RLAnything closed loop where policy, reward model, and environment co-evolve.*


There is a beautiful theoretical result behind this. The **reward precision** — the probability that the PRM correctly distinguishes a good step from a bad step — approaches 1 as the number of PRM evaluations $$m$$ increases. Specifically:

$$A \geq 1 - \exp\left(-\frac{m(\mu - 1)^2}{4}\right)$$

where $$\mu = p_+ + p_-$$ is the sum of the PRM's true positive and true negative rates. As long as the PRM is better than random ($$\mu > 1$$), the precision converges exponentially with more evaluations.

Let us plug in some numbers. Suppose the PRM correctly identifies good steps 70% of the time ($$p_+ = 0.7$$) and correctly identifies bad steps 65% of the time ($$p_- = 0.65$$). Then $$\mu = 0.7 + 0.65 = 1.35$$. With $$m = 5$$ evaluations:

$$A \geq 1 - \exp\left(-\frac{5 \times (1.35 - 1)^2}{4}\right) = 1 - \exp(-0.153) = 1 - 0.858 = 0.142$$

With $$m = 20$$ evaluations: $$A \geq 1 - \exp(-0.613) = 1 - 0.542 = 0.458$$. With $$m = 100$$: $$A \geq 1 - \exp(-3.06) = 0.953$$. The precision climbs rapidly — even a mediocre PRM becomes highly reliable with enough evaluations. This is exactly what we want.

---

## Engineering Details: Overlong Reward Shaping and the GRPO-TCR Recipe

Before we look at the practical setup, let us cover two important engineering details that make OpenClaw-RL work well in practice.

**Overlong reward shaping.** When a language model generates very long responses, it can exhaust the context window and produce truncated, incoherent outputs. OpenClaw-RL addresses this with a smooth penalty function that discourages overly long responses:


$$r_{\text{length}}(y) = \begin{cases} 0 & \text{if } |y| \leq L_{\max} - L_{\text{cache}} \\ \frac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}} & \text{if } L_{\max} - L_{\text{cache}} < |y| \leq L_{\max} \\ -1 & \text{if } |y| > L_{\max} \end{cases}$$

Let us plug in some numbers. Suppose $$L_{\max} = 1000$$ tokens and $$L_{\text{cache}} = 200$$ tokens.

- If the response is 700 tokens: $$r = 0$$. No penalty — well within limits.
- If the response is 900 tokens: $$r = (800 - 900) / 200 = -0.5$$. Moderate penalty — the response is getting close to the limit.
- If the response is 1100 tokens: $$r = -1$$. Full penalty — the response exceeded the limit.

Notice the smooth transition in the middle zone. Instead of a harsh cutoff at $$L_{\max}$$, the penalty increases linearly as the response approaches the limit. This gives the model a gentle nudge to be more concise, rather than a sudden cliff.

**The GRPO-TCR recipe.** OpenClaw-RL uses a specific combination of techniques that has been empirically shown to be highly effective:

- **T** — Token-level loss aggregation (each token weighted equally, regardless of trajectory length)
- **C** — Clip-higher (asymmetric clipping bounds that expand exploration)
- **R** — Overlong Reward shaping (the smooth penalty we just discussed)

This combination is powerful enough that **4B-parameter models can match the agentic reasoning performance of 32B-parameter models**. The GRPO-TCR recipe was identified through extensive experiments in the "Demystifying Reinforcement Learning in Agentic Reasoning" paper.

**Graceful weight updates.** When the trainer finishes a training step and produces new weights, the serving component needs to load them. OpenClaw-RL handles this gracefully: training submission pauses during weight updates, then resumes automatically. This ensures no data corruption and no service interruption.

---

## Practical Setup: Getting Your Own Personalized Agent Running

Enough theory. Let us look at how to actually set up OpenClaw-RL.

**Hardware requirements.** By default, OpenClaw-RL uses 8 GPUs:
- 4 GPUs for the actor (policy serving and training)
- 2 GPUs for rollout generation
- 2 GPUs for the PRM

These allocations are configurable via environment variables: `NUM_GPUS`, `ACTOR_GPUS`, `ROLLOUT_GPUS`, `PRM_GPUS`.

**Step 1: Start the RL server.** Choose your optimization method:

For Binary RL (best for implicit feedback like thumbs up/down):
```bash
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

For On-Policy Distillation (best for rich textual feedback):
```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh
```

Once running, the model is served as an OpenAI-compatible API at `http://<HOST_IP>:30000/v1`.

**Step 2: Configure OpenClaw.** Point OpenClaw to your RL server by editing your `openclaw.json`:

```json
{
  "models": {
    "providers": {
      "qwen": {
        "baseUrl": "http://<HOST_IP>:30000/v1",
        "apiKey": "apiKey",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3-4b",
            "name": "Qwen3 4B",
            "reasoning": true,
            "contextWindow": 32768,
            "maxTokens": 8192
          }
        ]
      }
    }
  }
}
```

Replace `<HOST_IP>` with the IP address of the machine running the RL server, and set the `apiKey` to match the `SGLANG_API_KEY` you configured on the server.

**Step 3: Start chatting.** That is it. Start using OpenClaw normally. The RL server will automatically:
1. Collect your conversation trajectories
2. Run PRM evaluations on the agent's responses
3. Compute rewards and advantages
4. Train the model in the background
5. Push updated weights to the serving component

Your agent gets better the more you use it.


![Choose your optimization method based on your natural feedback style.](figures/figure_9.png)
*Choose your optimization method based on your natural feedback style.*


---

## Conclusion

Let us step back and look at what we have covered.

**OpenClaw** gives you a powerful personal AI agent — self-hosted, private, connected to all your services. **OpenClaw-RL** closes the loop by turning your everyday conversations into training signal, making the model learn your preferences continuously.

The system is built from four asynchronous components — agent serving, rollout collection, PRM judging, and policy training — that run independently and never block each other. You chat normally while the model improves in the background.

Two learning paradigms cover the full spectrum of human feedback. **Binary RL** (GRPO) works with implicit signals like task success or failure, scoring responses as good or bad through majority-voted PRM evaluations. **On-Policy Distillation** works with rich textual feedback, extracting hindsight hints and computing token-level directional advantages.

The **RLAnything** framework takes this further by co-optimizing the policy, the reward model, and the environment simultaneously — a closed loop where every component makes the others better.

The roadmap ahead is exciting: broader model support beyond Qwen3-4B, scalable agentic RL infrastructure for general agents (with computer-use coming first), and extending learning beyond the policy to skills and persistent memory.

The era of personal AI agents that truly learn from you is here.

---

## References

- Wang, Yinjie et al., "OpenClaw-RL" (2026) — https://github.com/Gen-Verse/OpenClaw-RL
- Yu, Zhaochen et al., "Demystifying Reinforcement Learning in Agentic Reasoning" (2025) — https://arxiv.org/abs/2510.11701
- Wang, Yinjie et al., "RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System" (2026) — https://arxiv.org/abs/2602.02488
- OpenClaw — https://openclaw.ai
- Slime RL Framework — https://github.com/THUDM/slime
- Open-AgentRL — https://github.com/Gen-Verse/Open-AgentRL
