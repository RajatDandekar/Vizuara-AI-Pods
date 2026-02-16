# DeepSeek-R1 Explained From Scratch

**How Reinforcement Learning Teaches Language Models to Reason Step by Step**

---

Let us start with a simple example. Imagine two students — Student A and Student B — preparing for a math exam.

Student A opens a solutions manual and memorizes every solved problem. For each question, Student A copies the step-by-step solution into a notebook and rehearses it over and over again. On exam day, if a problem looks exactly like one from the manual, Student A does well. But if the problem is slightly different? Student A is stuck.

Student B takes a different approach. Student B attempts the problems on their own. They have no solutions manual — only the answer key at the back of the book. After every attempt, Student B checks: "Did I get the right answer?" If yes, Student B reinforces that approach. If no, Student B backtracks, rethinks, and tries again.

Over time, something interesting happens. Student B starts developing their own problem-solving strategies. They learn to break big problems into smaller steps. They learn to verify their intermediate results. They even develop an inner voice that says, "Wait, let me re-check that step."


![Student A (SFT) memorizes solutions vs Student B (RL) develops reasoning through practice](figures/figure_1.png)
*Student A memorizes solved examples (Supervised Fine-Tuning), while Student B practices independently and checks answers (Reinforcement Learning) — developing deeper reasoning ability.*


This is exactly what DeepSeek-R1 does.

In January 2025, DeepSeek released a paper that showed something remarkable: you can teach a large language model to reason step by step using **reinforcement learning alone** — without any human-written chain-of-thought examples. The model learns to think, verify, and self-correct, all by itself.

Let us understand how.

---

## Why Do LLMs Need to "Think"?

Standard large language models work by predicting the next token. Given a sequence of words, they output the most probable next word. This works incredibly well for many tasks — translation, summarization, creative writing.

But there is a problem. Let us take a concrete example:

**Prompt:** What is 17 × 23?

A standard LLM might get this right, because it has seen many multiplication examples during training. The answer, 391, might be pattern-matched from memory.

But now ask:

**Prompt:** What is 347 × 892?

This is much harder to pattern-match. The answer is 309,524, and a standard LLM will very likely get it wrong — because it is trying to produce the answer in a single shot, without doing any intermediate computation.

Now, what if the model could "think out loud"? What if, instead of jumping straight to the answer, it could write out intermediate steps?


![Standard LLM vs Reasoning LLM comparison](figures/figure_2.png)
*A standard LLM tries to answer in one shot and gets it wrong. A reasoning LLM writes out intermediate steps inside think tags and arrives at the correct answer.*


This is the idea behind **chain-of-thought reasoning**. If the model writes out its intermediate steps, it can solve much harder problems. OpenAI demonstrated this with their o1 model in 2024, which showed impressive reasoning capabilities.

But there was a catch. Training o1 required massive amounts of **human-labeled reasoning data** — experts manually writing out step-by-step solutions for thousands of problems.

This brings us to the main question: **Can we teach an LLM to reason without hand-crafted reasoning examples?**

DeepSeek-R1 answers this with a resounding yes.

---

## DeepSeek-R1-Zero: Pure RL, No Supervision

Let us come to the main character in our story: **DeepSeek-R1-Zero**.

The idea is beautifully simple. Take a powerful base model (DeepSeek-V3-Base, with 671 billion total parameters), and train it using **only reinforcement learning** — no supervised fine-tuning whatsoever.

But if there is no supervised data, how does the model know what "good reasoning" looks like?

It doesn't need to. The reward is entirely **rule-based**:

1. **Accuracy reward:** Did you get the correct final answer? If yes, you get a reward of 1. If no, you get 0.
2. **Format reward:** Did you put your reasoning inside `<think>...</think>` tags and your final answer in the correct format? If yes, you get an additional reward.

That is it. No neural reward model. No human preferences. Just: "Were you right?" and "Did you show your work?"

The total reward for a single response is:


$$
r = r_{\text{accuracy}} + r_{\text{format}}
$$

Let us plug in some simple numbers to see how this works.

Consider a math problem where the model produces an answer:

- **Case 1:** The model gets the answer right and uses `<think>` tags properly: $r = 1.0 + 1.0 = 2.0$
- **Case 2:** The model gets the answer wrong but uses the correct format: $r = 0.0 + 1.0 = 1.0$
- **Case 3:** The model gets the answer right but skips the think tags: $r = 1.0 + 0.0 = 1.0$

This tells us that the highest reward is given only when the model both reasons properly (in the correct format) **and** arrives at the correct answer. This is exactly what we want.

Now, why did DeepSeek avoid using a neural reward model? Because of **reward hacking**. If you recall from our discussion on RLHF, language models can learn to "game" neural reward models by exploiting their weaknesses. By using rule-based rewards with verifiable answers (math problems have definite correct answers, code either passes tests or doesn't), there is nothing to hack.

The results were dramatic. Over the course of RL training, R1-Zero's performance on AIME 2024 (a prestigious math competition) went from **15.6% to 71.0%**.

But here is what is even more fascinating: the model's responses **naturally grew longer** as training progressed. Nobody told the model to think more. Nobody provided examples of long reasoning chains. The model discovered on its own that thinking longer leads to better answers and higher rewards.


![DeepSeek-R1-Zero training curves](figures/figure_3.png)
*DeepSeek-R1-Zero: AIME 2024 accuracy rises from 15.6% to 71.0% during RL training, while response length naturally grows — the model learns to "think longer" on its own.*


---

## The "Aha Moment"

During the training of R1-Zero, something truly remarkable happened.

At an intermediate checkpoint, the model was solving a math problem. It started down one approach, ran into a dead end, and then produced the following text:

> *"Wait, wait. Wait. That's an aha moment I can flag here."*

The model then **re-examined its own reasoning**, found the error in its initial approach, and corrected course to arrive at the right answer.

Let us pause and appreciate what happened here. Nobody taught the model to say "wait." Nobody provided examples of self-reflection. Nobody programmed a "re-check your work" instruction. The model **spontaneously developed** the ability to catch its own mistakes and try alternative approaches — purely from the RL reward signal.


![The aha moment — emergent self-reflection in R1-Zero](figures/figure_4.png)
*The "aha moment": the model starts down one approach, realizes it is messy, spontaneously self-corrects ("Wait, let me reconsider..."), and finds the right solution. This self-reflection emerged purely from RL training.*


Think back to our student analogy. Student B, who practices on their own, eventually develops that inner voice — the one that says "Hold on, let me re-check that step." R1-Zero developed the same thing.

This is one of the most important findings in the paper: **complex reasoning behaviors can emerge from simple reward signals.** You don't need to explicitly teach self-reflection — it arises naturally when the model is incentivized to get the right answer.

---

## GRPO: The RL Algorithm Behind R1

Now the question is: what RL algorithm is used to train R1-Zero?

If you have read our article on RLHF, you will recall **PPO (Proximal Policy Optimization)**. PPO is the workhorse of modern RLHF — it is used to align models like ChatGPT.

But PPO has a major cost: it requires a **critic network** (also called a value function) that is typically the same size as the policy model. For a model with 671 billion parameters, maintaining a separate 671B critic network is extremely expensive.

DeepSeek introduced a clever alternative: **GRPO — Group Relative Policy Optimization.**

The core idea is simple and elegant. Instead of training a separate critic network to estimate how good a response is, GRPO does the following:

**For each question, sample G different answers from the model.** Compute the reward for each answer. Then, use the group's statistics (mean and standard deviation) to figure out which answers were relatively better or worse.

The advantage for the $i$-th answer is calculated as:


$$
A_i = \frac{r_i - \text{mean}(r_1, r_2, \ldots, r_G)}{\text{std}(r_1, r_2, \ldots, r_G)}
$$

Let us plug in some simple numbers to see how this works.

Suppose we sample $G = 5$ answers for a math question, and their rewards are: $[1, 0, 1, 1, 0]$.

The mean reward is: $\text{mean} = (1 + 0 + 1 + 1 + 0) / 5 = 0.6$

The standard deviation is: $\text{std} = 0.49$

Now, for an answer that got reward 1 (correct answer):

$$A = \frac{1 - 0.6}{0.49} = 0.82$$

This is a **positive advantage** — it tells the model: "This approach worked, do more of this."

For an answer that got reward 0 (wrong answer):

$$A = \frac{0 - 0.6}{0.49} = -1.22$$

This is a **negative advantage** — it tells the model: "This approach failed, do less of this."

This is exactly what we want. The model reinforces correct reasoning paths and suppresses incorrect ones — all without needing a separate critic network.

Once we have the advantages, the policy is updated using a loss function that looks very similar to PPO:


$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)} A_i,\; \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i\right) + \beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

Let us break this down term by term:

- **The ratio** $\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}$: How much has our policy changed since we collected the samples? This is the same importance sampling ratio we saw in PPO.

- **The clipping** $\text{clip}(\ldots, 1-\epsilon, 1+\epsilon)$: Don't let the policy change too much in a single step. Recall the runner analogy from PPO — the coach says "don't change your style too drastically."

- **The KL penalty** $\beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$: Keep the model from drifting too far from the reference model. This prevents reward hacking, just as we saw in RLHF.

The key difference from PPO? **No critic network.** The advantages come directly from comparing answers within the group. This makes GRPO significantly cheaper to run at scale.


![PPO vs GRPO comparison](figures/figure_5.png)
*PPO requires an expensive critic model the same size as the policy. GRPO eliminates the critic entirely by sampling G outputs per question and computing advantages from group statistics.*


---

## From R1-Zero to R1: The Full Pipeline

R1-Zero is impressive, but it has some issues. The model sometimes mixes languages mid-reasoning (starting in English and switching to Chinese). Its responses can be hard to read, with poor formatting and structure. Sometimes it skips the reasoning entirely.

To fix these issues, DeepSeek built the full **DeepSeek-R1** model using a four-stage training pipeline. Let us walk through each stage.

### Stage 1: Cold-Start SFT

Instead of starting from scratch like R1-Zero, DeepSeek collected about **1,000 long chain-of-thought examples**. These were carefully curated to show high-quality reasoning with proper formatting — clear steps, readable language, proper use of `<think>` tags.

The base model is fine-tuned on these examples. Think of it as giving Student B a few "model answers" before the exam — not enough to memorize, but enough to understand what good reasoning looks like.

### Stage 2: Reasoning-Oriented RL

Now the RL training begins. GRPO is applied on math, code, and science tasks, using the same accuracy and format rewards as R1-Zero.

But there is one addition. To prevent language mixing, a **language consistency reward** is added:


$$
r_{\text{lang}} = \frac{\text{number of target language words in reasoning}}{\text{total number of words in reasoning}}
$$

Let us plug in some simple numbers. Suppose the model produces 100 tokens in its `<think>` block, and 85 of those tokens are in English (the target language):

$$r_{\text{lang}} = \frac{85}{100} = 0.85$$

If all 100 tokens are in English: $r_{\text{lang}} = 1.0$. If the model mixes in 40 Chinese tokens: $r_{\text{lang}} = 0.6$.

This tells the model that it should maintain consistent language throughout its reasoning. Simple, but effective.

### Stage 3: Rejection Sampling + SFT

After RL training, the model generates a large number of solutions. DeepSeek keeps only the correct ones — about **600,000 reasoning samples** (math, code, science) and **200,000 general samples** (writing, translation, Q&A).

The model is then fine-tuned on this curated 800K-sample dataset. This is essentially the model learning from its own best work.

### Stage 4: Secondary RL (All Scenarios)

A final round of RL is applied, but this time covering **all** task types — not just reasoning. For reasoning tasks, rule-based rewards are used (same as before). For general tasks like helpfulness and harmlessness, a neural reward model is used.

This last stage ensures that DeepSeek-R1 is not just a reasoning specialist, but a well-rounded model.


![DeepSeek-R1 four-stage training pipeline](figures/figure_6.png)
*The four-stage training pipeline: Cold-Start SFT (1,000 examples) → Reasoning RL (GRPO on math/code/science) → Rejection Sampling + SFT (800K curated samples) → All-Scenario RL (reasoning + general tasks).*


---

## Distillation: Reasoning for Everyone

DeepSeek-R1 has 671 billion total parameters (with 37 billion activated at any time via mixture-of-experts). This is not a model you can run on your laptop.

So DeepSeek did something very clever: **distillation**. They used DeepSeek-R1 to generate about 800,000 high-quality reasoning samples, and then fine-tuned smaller open-source models on these samples.

The key insight is that **SFT alone is sufficient** for distillation — you do not need to run RL on the smaller models. The reasoning patterns are already captured in R1's outputs; the smaller models just need to learn to imitate them.

The results are truly amazing:

- **DeepSeek-R1-Distill-Qwen-7B** (7 billion parameters) achieves **55.5% on AIME 2024** — this surpasses QwQ-32B-Preview, a model more than 4x its size.
- **DeepSeek-R1-Distill-Qwen-32B** achieves **72.6% on AIME** and **94.3% on MATH-500**.
- Even the tiny **1.5B parameter model** achieves **28.9% on AIME**, outperforming GPT-4o on mathematical reasoning.

Not bad, right?

DeepSeek released distilled models in six sizes: **1.5B, 7B, 8B, 14B, 32B, and 70B** — all with open weights.


![Distilled models punch above their weight on AIME 2024](figures/figure_7.png)
*Distilled models punch above their weight: the 1.5B model beats GPT-4o, and the 7B model surpasses QwQ-32B — a model more than 4x its size.*


---

## Key Results

Let us now look at how the full DeepSeek-R1 model compares to OpenAI's o1 across key benchmarks:

| Benchmark | DeepSeek-R1 | OpenAI o1-1217 |
|-----------|-------------|----------------|
| AIME 2024 (pass@1) | 79.8% | 79.2% |
| MATH-500 | 97.3% | 96.4% |
| Codeforces (Elo) | 2,029 | 2,061 |
| GPQA Diamond | 71.5% | 75.7% |
| MMLU | 90.8% | 91.8% |

DeepSeek-R1 matches or comes very close to o1 on nearly every benchmark — and on AIME and MATH-500, it actually edges ahead.

With majority voting (sampling 64 answers and taking the most common one), R1 reaches **86.7% on AIME** — this is a remarkable score for a math competition that challenges the brightest high school students.

![DeepSeek-R1 vs OpenAI o1 radar chart comparing performance across AIME 2024, MATH-500, Codeforces, GPQA Diamond, and MMLU benchmarks](figures/figure_8.png)
*DeepSeek-R1 vs OpenAI o1: Head-to-Head comparison across reasoning benchmarks*

And here is what makes this truly significant: **DeepSeek-R1 is open-weight.** The model weights are publicly available. Anyone can download, run, fine-tune, and build on top of it. This stands in sharp contrast to OpenAI's o1, which remains closed.

---

## Code: Trying a Distilled R1 Model

Enough theory, let us look at some practical implementation now.

Here is how you can run one of the distilled DeepSeek-R1 models using the Hugging Face Transformers library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the 7B distilled model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

# Ask a reasoning question
prompt = "How many prime numbers are there between 50 and 80?"
messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate with thinking
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:])
print(response)
```

Let us understand this code in detail. First, we load the 7B distilled model from Hugging Face. This is the smallest model that achieves really strong reasoning performance. We use `bfloat16` precision to reduce memory usage and `device_map="auto"` to automatically place the model on available GPUs.

Next, we construct a prompt asking a mathematical question. The chat template formats the message in the way the model expects. When we call `model.generate()`, the model will first produce its reasoning inside `<think>` tags, and then give the final answer.

The output will look something like this:

```
<think>
Let me list the prime numbers between 50 and 80.
I need to check each odd number in this range.
51 = 3 × 17, not prime
53 - let me check: not divisible by 2, 3, 5, 7. 53 is prime.
55 = 5 × 11, not prime
57 = 3 × 19, not prime
59 - not divisible by 2, 3, 5, 7. 59 is prime.
61 - not divisible by 2, 3, 5, 7. 61 is prime.
...
</think>

There are 7 prime numbers between 50 and 80: 53, 59, 61, 67, 71, 73, and 79.
```

You can see that the model writes out its reasoning step by step — checking each candidate, verifying divisibility, and arriving at the answer. This is exactly the kind of chain-of-thought reasoning that DeepSeek-R1 was trained to produce.

---

## Conclusion

Let us summarize what we have covered.

DeepSeek-R1 demonstrated that **reinforcement learning alone can teach language models to reason.** No hand-crafted chain-of-thought data is needed — just a simple reward signal: "Did you get the right answer?"

The **"aha moment"** showed us that complex behaviors like self-reflection and error correction can **emerge** from simple reward signals. The model learned to pause, re-examine its reasoning, and try alternative approaches — all on its own.

**GRPO** makes this scalable by eliminating the expensive critic network from PPO, replacing it with group-based advantage estimation.

And through **distillation**, DeepSeek made these reasoning capabilities accessible to everyone — with models as small as 1.5 billion parameters outperforming GPT-4o on mathematical reasoning.

This brings us to a fascinating question. If reinforcement learning can teach a model to reason about mathematics and code — domains where the answer is verifiable — can it also teach a model to reason about the physical world? About science? About creative problems where the "right answer" is not so clear-cut?

We will explore this in the next lecture.

---

## References

- DeepSeek-AI et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)
