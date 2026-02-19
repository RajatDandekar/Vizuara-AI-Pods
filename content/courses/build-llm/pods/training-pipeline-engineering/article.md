Training Pipeline Engineering: From Raw Text to a Trained Language Model
How tokenization, data loading, and optimization work together to turn architecture into intelligence.
Vizuara AI

## (1) The Factory That Builds Intelligence

Let us start with a simple analogy. Imagine a car manufacturing factory.

Raw steel, rubber, and glass arrive at the loading dock. But you cannot simply throw raw materials at the assembly line and hope a car appears. First, the raw materials must be **processed** — steel is cut into panels, rubber is molded into tires, glass is shaped into windshields. Then, a **conveyor belt** feeds these processed parts to the workers in organized batches — not too many at once, not too few. Finally, the workers on the assembly line use **precise tools and techniques** to assemble everything, adjusting their methods as they learn what works best.

Training a Large Language Model follows exactly the same process:

1. **Raw text** (books, articles, code) arrives at the loading dock
2. The **tokenizer** processes raw text into numerical tokens — our processed parts
3. The **data loader** organizes tokens into structured batches and feeds them to the model — our conveyor belt
4. The **optimizer** adjusts the model's weights to minimize loss — our skilled assembly workers


![The training pipeline: raw text enters, a trained model exits.](figures/figure_1.png)
*The training pipeline: raw text enters, a trained model exits.*


If any one of these stages fails, the entire pipeline breaks. A perfect architecture with a broken tokenizer will produce garbage. A great tokenizer with a poorly tuned optimizer will never converge. Every stage matters.

To put this in perspective, consider the numbers involved. Training LLaMA 2 (70B parameters) required processing roughly 2 trillion tokens, split across 2,048 GPUs, running continuously for about 1,720,000 GPU-hours. At every single step of that massive computation, the training pipeline must work flawlessly — tokenization must be consistent, batches must flow without interruption, and the optimizer must update 70 billion weights without a single numerical explosion. One broken component at step 500,000 can waste weeks of computation.

In this article, we will build the entire training pipeline from scratch — starting from raw text and ending with a fully trained model. We will understand not just **what** each component does, but **why** it is designed that way.

Let us begin with the first stage: tokenization.

## (2) Tokenization: Teaching Machines to Read

Here is a fundamental problem: neural networks operate on numbers, not text. When we write "The cat sat on the mat", a neural network has absolutely no idea what to do with these characters. We need to convert text into numbers.

But **how** do we convert text into numbers? This brings us to the art of tokenization.

### Character-Level Tokenization

The simplest approach is character-level tokenization. Every character gets a unique number:

```
"cat" → [99, 97, 116]    (c=99, a=97, t=116)
```

This gives us a tiny vocabulary (just ~256 characters for English), but sequences become extremely long. The word "transformer" becomes 11 tokens. An entire book becomes millions of tokens. The model must learn from individual characters that "t", "r", "a", "n", "s", "f", "o", "r", "m", "e", "r" somehow form a meaningful word.

This makes training very slow and very hard. The sentence "The quick brown fox jumps over the lazy dog" is 9 words but 43 characters — nearly 5x more tokens for the same content.

### Word-Level Tokenization

The opposite extreme is word-level tokenization. Every word gets its own number:

```
"The cat sat" → [1, 2, 3]    (The=1, cat=2, sat=3)
```

This produces short sequences, but the vocabulary becomes enormous. English has over 170,000 words in active use, and when you add names, technical terms, misspellings, and multiple languages, the vocabulary can exceed millions. A vocabulary of 1 million tokens means our embedding matrix alone needs 1 million rows — this is extremely memory-intensive.

And what happens when the model encounters a word it has never seen? It simply cannot process it.

### The Sweet Spot: Subword Tokenization (BPE)

What we really want is something in between — a method that keeps common words as single tokens but can break rare words into smaller, meaningful pieces. This is exactly what **Byte Pair Encoding (BPE)** does.

Consider "unhappiness". Character-level produces 11 tokens. Word-level produces 1 token — but only if the model has seen "unhappiness" before. BPE might produce 3 tokens: `[un, happi, ness]`, each carrying meaning: "un" (negation), "happi" (related to "happy"), "ness" (noun suffix). The model can now generalize — when it encounters "unfairness", it recognizes "un" and "ness" from prior training, even if it has never seen the full word.


![The tokenization spectrum: BPE finds the sweet spot.](figures/figure_2.png)
*The tokenization spectrum: BPE finds the sweet spot.*


### How BPE Works: A Step-by-Step Example

Let us walk through BPE with a concrete example. Suppose our entire training corpus is:

```
"low low low low low lowest lowest newer newer newer wider wider wider wider"
```

**Step 1: Start with characters.** We break every word into individual characters and add a special end-of-word marker `_`:

```
l o w _    (appears 5 times: from "low" x5)
l o w e s t _    (appears 2 times)
n e w e r _    (appears 3 times)
w i d e r _    (appears 4 times)
```

Our initial vocabulary is: `{l, o, w, e, s, t, n, r, i, d, _}`

**Step 2: Count all adjacent pairs.** We look at every pair of consecutive tokens and count how often each pair appears:

```
(l, o) → 7 times    (5 from "low" + 2 from "lowest")
(o, w) → 7 times
(w, _) → 5 times
(w, e) → 5 times    (2 from "lowest" + 3 from "newer")
(e, r) → 7 times    (3 from "newer" + 4 from "wider")
(r, _) → 7 times
...
```

**Step 3: Merge the most frequent pair.** Several pairs are tied at 7. Let us pick `(l, o)`. We merge it into a new token `lo`:

```
lo w _    (5 times)
lo w e s t _    (2 times)
n e w e r _    (3 times)
w i d e r _    (4 times)
```

Vocabulary: `{l, o, w, e, s, t, n, r, i, d, _, lo}`

**Step 4: Repeat.** Count pairs again, merge the most frequent. Next we might merge `(e, r)` into `er`:

```
lo w _    (5 times)
lo w e s t _    (2 times)
n e w er _    (3 times)
w i d er _    (4 times)
```

Then merge `(er, _)` into `er_`, then `(lo, w)` into `low`, and so on.

We keep repeating this process until we reach our desired vocabulary size.

The beauty of BPE is that common words like "low" become single tokens, while rare words like "lowest" are split into meaningful pieces like "low" + "est". The model can then learn that "est" is a suffix that means "the most".

### Vocabulary Size Tradeoffs

In practice, modern LLMs use vocabularies between 32,000 and 100,000 tokens. GPT-2 uses 50,257 tokens. LLaMA uses 32,000 tokens.

Why not go larger? Because every token needs an embedding vector stored in memory. If each embedding is 4,096 dimensions and we use 32-bit floats, then:


$$
\text{Embedding Memory} = V \times d \times 4 \text{ bytes}
$$

Let us plug in some numbers. For a vocabulary of $$V = 50{,}000$$ tokens and an embedding dimension of $$d = 4{,}096$$:


$$
\text{Memory} = 50{,}000 \times 4{,}096 \times 4 = 819{,}200{,}000 \text{ bytes} \approx 819 \text{ MB}
$$

That is nearly 1 GB just for the embedding table. If we doubled the vocabulary to 100,000, we would need about 1.6 GB. This is a significant portion of GPU memory that could otherwise be used for larger batch sizes or longer sequences.

This is exactly why choosing the right vocabulary size is so critical. Too small and every word gets split into too many pieces, making sequences very long. Too large and we waste memory on tokens that appear very rarely.

### BPE vs WordPiece: Two Flavors of Subword Tokenization

BPE is not the only subword algorithm. BERT uses **WordPiece**, which differs in how it selects which pair to merge. BPE merges the most **frequent** pair. WordPiece merges the pair that maximizes the **likelihood** of the training corpus — it prefers pairs that appear together more often than chance would predict.

Suppose `(e, r)` appears 100 times, while `e` appears 500 times and `r` appears 200 times. BPE simply sees 100 and compares it to other counts. WordPiece computes:

$$
\text{Score} = \frac{\text{freq}(e, r)}{\text{freq}(e) \times \text{freq}(r)} = \frac{100}{500 \times 200} = \frac{100}{100{,}000} = 0.001
$$

Compare this with `(q, u)` appearing 50 times, where `q` appears 60 times and `u` appears 300 times:

$$
\text{Score} = \frac{50}{60 \times 300} = \frac{50}{18{,}000} \approx 0.0028
$$

Even though `(q, u)` appears less frequently (50 vs 100), WordPiece scores it higher (0.0028 vs 0.001) because `q` and `u` co-occur far more than chance predicts. This makes sense — `q` is almost always followed by `u` in English, so merging them is a strong signal.

GPT models use BPE. BERT uses WordPiece. LLaMA uses a BPE variant built on SentencePiece. The important takeaway is the shared principle: find subword units that balance vocabulary size against sequence length.

## (3) Building the Dataset: From Tokens to Training Pairs

Now that we can convert raw text into tokens, we need to create training data from it. Language models are trained on the **next-token prediction** task: given a sequence of tokens, predict the next one.

Let us take a concrete example. Suppose our tokenized text is:

```
[The, cat, sat, on, the, mat]
 [0]  [1]  [2] [3] [4]  [5]
```

With a context length of 4, we create input-output pairs using a sliding window:

```
Input: [The, cat, sat, on]  → Target: [cat, sat, on, the]
Input: [cat, sat, on, the]  → Target: [sat, on, the, mat]
```


![Sliding window creates input-target pairs for next-token prediction.](figures/figure_3.png)
*Sliding window creates input-target pairs for next-token prediction.*


Notice something important: the target is simply the input shifted by one position to the right. Each token in the input is being trained to predict the very next token. This is the heart of language model training.

### Why Context Length Matters

The context length determines how many previous tokens the model can "see" when predicting the next token. GPT-2 uses a context length of 1,024. GPT-4 supports up to 128,000.

A longer context length means the model can capture longer-range dependencies — for example, resolving a pronoun that refers to a character introduced 5 paragraphs earlier. But longer contexts require more memory and computation because of the self-attention mechanism, which scales quadratically:


$$
\text{Attention Memory} \propto T^2
$$


where $$T$$ is the context length. If we double the context length from 1,024 to 2,048, the attention memory quadruples. This is why context length is always a careful engineering tradeoff.

Let us see this with numbers. Suppose the attention computation for $$T = 1{,}024$$ takes 1 unit of memory:


$$
\text{At } T = 2{,}048: \quad \frac{2{,}048^2}{1{,}024^2} = \frac{4{,}194{,}304}{1{,}048{,}576} = 4\times \text{ the memory}
$$

This tells us that doubling the context length increases attention memory by 4x. This is exactly why researchers invest so much effort into efficient attention mechanisms.

### The Dataset Class in PyTorch

In code, our dataset looks like this:

```python
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_length]
        y = self.tokens[idx + 1 : idx + self.context_length + 1]
        return torch.tensor(x), torch.tensor(y)
```

This is clean and simple. Each call to `__getitem__` returns an input-target pair created by the sliding window. The `__len__` method tells us how many such pairs exist: if we have 10,000 tokens and a context length of 1,024, we get 10,000 - 1,024 = 8,976 training samples.

## (4) The Data Loader: Feeding the Model Efficiently

We now have a dataset of input-target pairs, but we cannot feed them to the model one at a time. Training on individual samples is extremely noisy and slow. Instead, we group samples into **batches**.

### Why Batching Matters

When we compute the loss on a single sample, the gradient we get is a very noisy estimate of the true gradient. It points in roughly the right direction but wobbles a lot. When we average the loss over a batch of, say, 32 samples, the gradient becomes much smoother and more reliable.

Think of it this way: if you ask one person whether a movie is good, you get a very noisy signal. If you ask 32 people, the average opinion is much more reliable. This is exactly what batching does for gradients.


![Larger batches produce smoother, more reliable gradients.](figures/figure_4.png)
*Larger batches produce smoother, more reliable gradients.*


But there is a tradeoff. Larger batches:
- Use more GPU memory (each sample in the batch needs to be stored)
- Can lead to worse generalization (the model may converge to sharper minima)
- Are more computationally efficient per sample (GPUs are optimized for parallel operations)

In practice, batch sizes for LLM training range from 32 to 2,048, often using gradient accumulation to simulate very large effective batch sizes.

### Gradient Accumulation: Faking a Bigger Batch

Now the question is: what if we want a batch size of 512, but our GPU can only hold 32 samples at once? This is where **gradient accumulation** comes in.

Instead of updating weights after every batch of 32, we compute gradients for 16 consecutive mini-batches of 32, **accumulate** (sum) the gradients, and then perform a single weight update. The result is mathematically equivalent to a batch size of $$32 \times 16 = 512$$.

Think of it like collecting votes. If you cannot fit 512 people in one room, you can collect votes from 16 rooms of 32 people each and then tally them up. The final result is the same.

Let us see this with numbers. Suppose we have 3 weights and the gradients from two mini-batches are:

```
Mini-batch 1 gradients: [0.1, -0.3, 0.2]
Mini-batch 2 gradients: [0.3, -0.1, 0.4]
```

Accumulated gradient (averaged): $$\left[\frac{0.1 + 0.3}{2},\; \frac{-0.3 + (-0.1)}{2},\; \frac{0.2 + 0.4}{2}\right] = [0.2,\; -0.2,\; 0.3]$$

This is exactly what we would get if we had processed all 64 samples in a single batch — smoother, more reliable updates without needing the GPU memory for the full batch.

### Padding and Collation

There is a subtle detail worth mentioning. In our `TextDataset`, every sample has the same length, so batching is straightforward. But in many real-world scenarios, sequences have different lengths. The data loader must **pad** shorter sequences with a special token (usually token ID 0) so all sequences in a batch match the longest one. The model then uses an **attention mask** to ignore padding positions during computation — preventing it from wasting capacity learning to predict padding tokens.

### Shuffling

If we feed training samples in order, the model sees the same patterns in the same sequence every epoch. This can cause the model to memorize the order of training data rather than learning general patterns. Shuffling randomizes the order, which provides a better estimate of the true gradient and helps generalization.

### The PyTorch DataLoader

PyTorch provides the `DataLoader` class that handles batching, shuffling, and parallel data loading for us:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,           # Our TextDataset from above
    batch_size=32,     # Number of samples per batch
    shuffle=True,      # Randomize order each epoch
    drop_last=True,    # Drop incomplete final batch
    num_workers=4      # Parallel data loading processes
)
```

The `drop_last=True` parameter is important: if we have 100 samples and a batch size of 32, the last batch would only have 4 samples. This uneven batch can cause issues with certain training techniques, so we simply drop it.

The `num_workers=4` parameter tells PyTorch to use 4 separate processes to prepare data in parallel while the GPU is busy computing. This prevents the GPU from sitting idle waiting for data.

## (5) Optimization: How the Model Learns

Now we arrive at the heart of training: optimization. Data flows through the pipeline in neat batches, the model produces predictions, and we compute the loss. The question is: **how** do we update the weights to reduce that loss?

### Stochastic Gradient Descent (SGD)

The simplest optimizer is SGD. The idea is beautifully intuitive: compute the gradient of the loss with respect to each weight, then take a small step in the opposite direction (because the gradient points uphill, and we want to go downhill).


$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$


where $$\theta_t$$ are the weights at step $$t$$, $$\eta$$ is the learning rate, and $$\nabla L(\theta_t)$$ is the gradient of the loss.

Let us plug in some numbers. Suppose a weight has value $$\theta_t = 0.5$$, the gradient is $$\nabla L = 0.2$$, and the learning rate is $$\eta = 0.01$$:


$$
\theta_{t+1} = 0.5 - 0.01 \times 0.2 = 0.5 - 0.002 = 0.498
$$


The weight moved slightly from 0.5 to 0.498, in the direction that reduces the loss. This makes sense because the positive gradient tells us the loss increases when the weight increases, so we decrease the weight.

### The Problem with Vanilla SGD

SGD has a critical problem: it treats all directions equally. Imagine you are walking down a narrow valley — steep walls on either side but a gentle slope forward. SGD will oscillate wildly between the walls (because the gradient is large in that direction) while making very slow progress along the valley floor (because the gradient is small there).


![Momentum smooths out oscillations and accelerates convergence.](figures/figure_5.png)
*Momentum smooths out oscillations and accelerates convergence.*


### SGD with Momentum

Momentum fixes this by adding a "velocity" to the weight updates. Instead of using only the current gradient, we maintain a running average of past gradients:


$$
v_{t+1} = \beta v_t + \nabla L(\theta_t)
$$


$$
\theta_{t+1} = \theta_t - \eta \, v_{t+1}
$$

Here, $$\beta$$ is the momentum coefficient (typically 0.9), and $$v_t$$ is the velocity.

Let us work through a numerical example. Suppose $$\beta = 0.9$$, $$\eta = 0.01$$, $$v_0 = 0$$, $$\theta_0 = 0.5$$, and the gradients over three steps are 0.2, 0.2, 0.2 (consistent gradient):

**Step 1:**


$$
v_1 = 0.9 \times 0 + 0.2 = 0.2
$$


$$
\theta_1 = 0.5 - 0.01 \times 0.2 = 0.498
$$


**Step 2:**


$$
v_2 = 0.9 \times 0.2 + 0.2 = 0.38
$$


$$
\theta_2 = 0.498 - 0.01 \times 0.38 = 0.49420
$$


**Step 3:**


$$
v_3 = 0.9 \times 0.38 + 0.2 = 0.542
$$


$$
\theta_3 = 0.4942 - 0.01 \times 0.542 = 0.48878
$$


Notice how the velocity builds up over time: 0.2, 0.38, 0.542. When the gradient consistently points in the same direction, momentum accumulates and the updates get larger — the optimizer accelerates. When the gradient oscillates (as in the zigzag scenario), the oscillating components cancel out, and the optimizer moves more smoothly. This is exactly what we want.

### Adam: The Optimizer That Changed Deep Learning

While momentum helps with the direction, it still uses the same learning rate for every weight. Adam (Adaptive Moment Estimation) goes further by adapting the learning rate for each weight individually.

Adam maintains two running averages:
- **First moment** ($$m_t$$): the mean of past gradients (like momentum)
- **Second moment** ($$v_t$$): the mean of past squared gradients (measures how variable the gradient has been)

The Adam update rule has four steps:

**Step 1: Update first moment (mean of gradients)**


$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$


**Step 2: Update second moment (mean of squared gradients)**


$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$


**Step 3: Bias correction (because $$m$$ and $$v$$ are initialized to zero)**


$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Step 4: Update weights**


$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

The typical hyperparameters are $$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$, $$\epsilon = 10^{-8}$$, and $$\eta = 0.001$$.

Now the question is: **why does dividing by $$\sqrt{\hat{v}_t}$$ help?**

Think of it this way. If a weight has had very large gradients (large $$\hat{v}_t$$), the denominator is large, and the effective step size is small — the optimizer is cautious. If a weight has had very small gradients (small $$\hat{v}_t$$), the denominator is small, and the effective step size is large — the optimizer is aggressive. Adam automatically adjusts the learning rate for each weight based on its gradient history.

### Adam: Numerical Worked Example

Let us trace through one step of Adam with concrete numbers. Suppose:

- $$\theta_0 = 0.5$$, $$m_0 = 0$$, $$v_0 = 0$$
- Current gradient: $$g_1 = 0.1$$
- $$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$, $$\eta = 0.001$$, $$\epsilon = 10^{-8}$$

**Step 1:** Update first moment


$$
m_1 = 0.9 \times 0 + 0.1 \times 0.1 = 0.01
$$


**Step 2:** Update second moment


$$
v_1 = 0.999 \times 0 + 0.001 \times 0.1^2 = 0.001 \times 0.01 = 0.00001
$$


**Step 3:** Bias correction at $$t = 1$$


$$
\hat{m}_1 = \frac{0.01}{1 - 0.9^1} = \frac{0.01}{0.1} = 0.1
$$



$$
\hat{v}_1 = \frac{0.00001}{1 - 0.999^1} = \frac{0.00001}{0.001} = 0.01
$$


Notice how bias correction dramatically amplified both values. This is because at $$t = 1$$, the running averages have only seen one gradient and are heavily biased toward zero. The correction fixes this.

**Step 4:** Update weight


$$
\theta_1 = 0.5 - 0.001 \times \frac{0.1}{\sqrt{0.01} + 10^{-8}} = 0.5 - 0.001 \times \frac{0.1}{0.1} = 0.5 - 0.001 = 0.499
$$

The weight moved from 0.5 to 0.499. The effective step size was exactly $$\eta = 0.001$$ here because we only had one gradient observation. As training progresses, the per-weight adaptive rates will diverge based on each weight's gradient history.

### Weight Decay

There is one more important component: weight decay. Without any regularization, weights can grow very large, causing the model to overfit. Weight decay adds a small penalty that shrinks the weights slightly at every step:


$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_{t-1}
$$

where $$\lambda$$ is the weight decay coefficient (typically 0.01 or 0.1).

Let us see the effect with numbers. If $$\theta_{t-1} = 2.0$$ and $$\lambda = 0.01$$:


$$
\text{Weight decay term} = 0.01 \times 2.0 = 0.02
$$


So the weight is pulled toward zero by 0.02 at each step, in addition to the gradient-based update. Larger weights get pulled more strongly. This prevents any single weight from becoming too dominant. This is exactly what we want for generalization.

## (6) Learning Rate Scheduling: The Art of Pacing

Setting the learning rate is not a one-time decision. The optimal learning rate changes during training. At the beginning, we want to be cautious (the model's weights are random and gradients are unreliable). In the middle, we want to make large, confident steps. Toward the end, we want to take smaller steps to fine-tune our position.

This brings us to learning rate scheduling.

### Warmup: Why Transformers Need a Gentle Start

When a Transformer model begins training, its weights are randomly initialized. The attention mechanism computes softmax over random logits, producing essentially random attention patterns. The gradients at this stage can be very large and point in unreliable directions.

If we start with a large learning rate, these noisy, unreliable gradients cause enormous weight updates that can destabilize training entirely — the loss might explode to infinity and never recover.

The solution is **linear warmup**: start with a very small learning rate and linearly increase it over the first few thousand steps. This gives the model time to "settle in" — the attention patterns start to form, the gradients become more meaningful, and the model reaches a region of the loss landscape where larger steps are safe.

Let us see this concretely. Suppose our peak learning rate is $$\eta_{\max} = 3 \times 10^{-4}$$ and we use 1,000 warmup steps. The learning rate at step $$t$$ is:

$$
\eta_t = \eta_{\max} \times \frac{t}{\text{warmup\_steps}}
$$

At step 1: $$\eta_1 = 3 \times 10^{-4} \times \frac{1}{1000} = 3 \times 10^{-7}$$ — extremely small, the model barely moves. At step 500: $$\eta_{500} = 1.5 \times 10^{-4}$$ — halfway to peak. At step 1,000: $$\eta_{1000} = 3 \times 10^{-4}$$ — full speed. This linear ramp gives the model 1,000 steps to stabilize before taking big strides. In practice, warmup is typically 1-10% of total training steps.

### Cosine Decay: Slowing Down Gracefully

After warmup, we want to gradually reduce the learning rate so the model can fine-tune its weights without overshooting good solutions. Cosine decay does this smoothly:


$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi \, t}{T}\right)\right)
$$


where $$\eta_{\max}$$ is the peak learning rate, $$\eta_{\min}$$ is the minimum, $$t$$ is the current step, and $$T$$ is the total number of training steps.

Let us plug in numbers. Suppose $$\eta_{\max} = 3 \times 10^{-4}$$, $$\eta_{\min} = 1 \times 10^{-5}$$, $$T = 10{,}000$$ steps:

At step $$t = 0$$ (start of cosine phase):


$$
\eta_0 = 1 \times 10^{-5} + \frac{1}{2}(3 \times 10^{-4} - 1 \times 10^{-5})(1 + \cos(0)) = 1 \times 10^{-5} + \frac{1}{2}(2.9 \times 10^{-4})(2) = 3 \times 10^{-4}
$$

At step $$t = 5{,}000$$ (halfway):


$$
\eta_{5000} = 1 \times 10^{-5} + \frac{1}{2}(2.9 \times 10^{-4})(1 + \cos(\pi/2)) = 1 \times 10^{-5} + \frac{1}{2}(2.9 \times 10^{-4})(1) = 1.55 \times 10^{-4}
$$

At step $$t = 10{,}000$$ (end):


$$
\eta_{10000} = 1 \times 10^{-5} + \frac{1}{2}(2.9 \times 10^{-4})(1 + \cos(\pi)) = 1 \times 10^{-5} + \frac{1}{2}(2.9 \times 10^{-4})(0) = 1 \times 10^{-5}
$$

The learning rate starts at $$3 \times 10^{-4}$$, smoothly decreases to $$1.55 \times 10^{-4}$$ at the halfway point, and gently lands at $$1 \times 10^{-5}$$ at the end. The cosine curve gives us a slow start to the decay, fast decay in the middle, and a slow finish — exactly the pacing profile we want.


![Warmup followed by cosine decay: the standard LLM learning rate schedule.](figures/figure_6.png)
*Warmup followed by cosine decay: the standard LLM learning rate schedule.*


The combination of linear warmup + cosine decay has become the standard learning rate schedule for training Transformers. Nearly every modern LLM — GPT, LLaMA, Mistral — uses some variant of this approach.

Why cosine specifically, and not a straight line? The cosine curve decays slowly at first, then faster in the middle, then slowly again at the end. The model spends more time at high learning rates (making bold progress) and more time at low learning rates (fine-tuning). A linear decay reduces the rate at a constant pace — spending equal time at every value, which is suboptimal. The cosine schedule naturally allocates training time where it is most useful.

## (7) Gradient Clipping: Taming the Explosions

Even with a carefully tuned learning rate schedule, gradients can sometimes explode. In deep Transformers, gradients can grow exponentially during backpropagation. A single bad batch can produce an enormous gradient that wipes out the progress of thousands of steps.

**Max-norm gradient clipping** provides a safety net. Before applying the gradient update, we check the total gradient norm. If it exceeds a threshold, we scale all gradients down proportionally:


$$
\text{If } \|g\| > c: \quad g \leftarrow c \cdot \frac{g}{\|g\|}
$$

where $$\|g\|$$ is the L2 norm of the full gradient vector and $$c$$ is the clipping threshold (typically 1.0).

Let us work through an example. Suppose we have gradients for three weights: $$g = [3.0, 4.0, 0.0]$$ and a clip threshold $$c = 1.0$$.

**Step 1:** Compute the gradient norm:


$$
\|g\| = \sqrt{3.0^2 + 4.0^2 + 0.0^2} = \sqrt{9 + 16} = \sqrt{25} = 5.0
$$


**Step 2:** Since $$\|g\| = 5.0 > c = 1.0$$, we clip:


$$
g_{\text{clipped}} = 1.0 \times \frac{[3.0, 4.0, 0.0]}{5.0} = [0.6, 0.8, 0.0]
$$

**Step 3:** Verify the clipped norm:


$$
\|g_{\text{clipped}}\| = \sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = \sqrt{1.0} = 1.0
$$

The gradient direction is preserved (3:4:0 became 0.6:0.8:0), but the magnitude is capped at 1.0. We have not lost any directional information — only prevented a catastrophically large step. This is exactly what we want.


![Gradient clipping caps the magnitude while preserving direction.](figures/figure_7.png)
*Gradient clipping caps the magnitude while preserving direction.*


In practice, gradient clipping is applied at every training step and is nearly universal in LLM training. A single line of code provides this critical safety mechanism:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## (8) Mixed Precision Training: Doing More with Less Memory

Before we assemble the full training loop, there is one more critical technique used in virtually every modern LLM training run: **mixed precision training**.

By default, neural network weights and activations are stored as 32-bit floating point numbers (FP32), using 4 bytes each. But for most of the forward and backward pass, we do not need that much precision. **FP16** (16-bit) and **BF16** (bfloat16) use only 2 bytes per number — half the memory.

Let us see what this means in practice. Consider a model with 1 billion parameters. In FP32:

$$
\text{Memory} = 1{,}000{,}000{,}000 \times 4 \text{ bytes} = 4 \text{ GB}
$$

In FP16 or BF16:

$$
\text{Memory} = 1{,}000{,}000{,}000 \times 2 \text{ bytes} = 2 \text{ GB}
$$

That is a 2 GB saving just for model weights. The activations stored during the forward pass also use half the memory. And because FP16 operations are faster on modern GPUs — NVIDIA's Tensor Cores perform FP16 matrix multiplications at roughly 2x the speed of FP32 — training becomes both smaller and faster.

Now the question is: if FP16 is so great, why not use it everywhere?

The problem is that FP16 has a very limited range — the largest representable number is about 65,504 and the smallest positive number is about $$6 \times 10^{-8}$$. During training, gradients can easily fall outside this range, causing underflow or overflow.

The solution is **mixed** precision: use FP16 for the heavy computation (matrix multiplications) but keep a **master copy** of the weights in FP32 for the actual weight update. This gives us the speed of FP16 while maintaining numerical stability where it matters most.

A **loss scaler** multiplies the loss by a large factor (say, 1024) before backpropagation, pushing small gradients away from the FP16 underflow zone, then scales them back down before the weight update.

In PyTorch, enabling mixed precision training takes just a few lines:

```python
scaler = torch.amp.GradScaler()

with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(batch_x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

The `autocast` context manager casts operations to BF16 where safe and keeps FP32 where precision matters. The `GradScaler` handles loss scaling. This is remarkably little code for a technique that can cut memory nearly in half and speed up training by 1.5-2x. This is exactly why every modern training pipeline uses it.

## (9) Putting It All Together: The Complete Training Loop

Now we have all the pieces of the puzzle ready. Let us combine everything — tokenizer, dataset, data loader, optimizer, learning rate schedule, and gradient clipping — into a complete, working training loop.

```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# --- (1) Tokenization ---
# In practice, you would use a trained BPE tokenizer (e.g., tiktoken)
import tiktoken
enc = tiktoken.get_encoding("gpt2")
raw_text = open("training_data.txt", "r").read()
tokens = enc.encode(raw_text)

# --- (2) Build Dataset ---
class TextDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
    def __len__(self):
        return len(self.tokens) - self.context_length
    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_length]
        y = self.tokens[idx + 1 : idx + self.context_length + 1]
        return x, y

CONTEXT_LENGTH = 256
dataset = TextDataset(tokens, CONTEXT_LENGTH)

# --- (3) Create DataLoader ---
BATCH_SIZE = 32
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=True)

# --- (4) Initialize Model and Optimizer ---
model = TransformerLM(vocab_size=enc.n_vocab,
                      d_model=512, n_heads=8,
                      n_layers=6, context_length=CONTEXT_LENGTH)
model = model.to("cuda")

# Adam optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=3e-4, betas=(0.9, 0.999),
                               weight_decay=0.01)

# --- (5) Learning Rate Schedule: Warmup + Cosine Decay ---
TOTAL_STEPS = len(dataloader) * 10       # 10 epochs
WARMUP_STEPS = int(0.1 * TOTAL_STEPS)    # 10% warmup

def get_lr(step):
    if step < WARMUP_STEPS:
        return 3e-4 * step / WARMUP_STEPS      # Linear warmup
    # Cosine decay
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return 1e-5 + 0.5 * (3e-4 - 1e-5) * (1 + math.cos(math.pi * progress))

# --- (6) Training Loop ---
MAX_GRAD_NORM = 1.0
step = 0

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to("cuda")
        batch_y = batch_y.to("cuda")

        # Forward pass
        logits = model(batch_x)                         # (B, T, V)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),            # (B*T, V)
            batch_y.view(-1)                             # (B*T,)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=MAX_GRAD_NORM)

        # Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Weight update
        optimizer.step()
        step += 1

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
```

Let us walk through what this code does:

**Tokenization:** We use `tiktoken` (the tokenizer used by GPT models) to convert raw text into a sequence of integer token IDs. This is our BPE step.

**Dataset:** Our `TextDataset` class creates input-target pairs using the sliding window approach. Each input is a chunk of `context_length` tokens, and the target is the same chunk shifted one position to the right.

**DataLoader:** We wrap the dataset in a `DataLoader` with batch size 32 and shuffling enabled. This feeds the model organized batches of data.

**Optimizer:** We use `AdamW` (Adam with decoupled weight decay) with the standard hyperparameters: $$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$, and weight decay of 0.01.

**Learning Rate Schedule:** The `get_lr` function implements linear warmup for the first 10% of training, followed by cosine decay for the remaining 90%.

**The Loop:** For each batch, we forward pass, compute cross-entropy loss, backpropagate, clip gradients, update the learning rate, and step the optimizer.

This is the fundamental pattern behind every modern language model. The architectures may vary, the scale may differ, but the pipeline remains: **tokenize, batch, forward, loss, backward, clip, step**.


![The complete training pipeline: from raw text to trained model.](figures/figure_8.png)
*The complete training pipeline: from raw text to trained model.*


## (10) Conclusion: The Pipeline Is What Turns Architecture into Intelligence

We started with a factory analogy, and it is worth returning to it now. A factory can have the most advanced assembly line in the world, but without a reliable supply chain and skilled workers, it produces nothing of value.

The same is true for language models. The Transformer architecture is elegant — self-attention, multi-head attention, layer normalization — but the architecture alone is just a blueprint. It is the **training pipeline** that breathes intelligence into it.

We covered the full journey:

1. **Tokenization** converts raw text into a numerical vocabulary using BPE, finding the sweet spot between character-level and word-level representations.

2. **The dataset** creates input-target pairs via a sliding window, teaching the model to predict the next token from context.

3. **The data loader** organizes these pairs into shuffled batches, giving the GPU a steady, efficient stream of training data.

4. **Adam** adapts the learning rate for each weight individually, using the history of past gradients to make intelligent updates.

5. **Warmup + cosine decay** paces the training — cautious at first, aggressive in the middle, gentle at the end.

6. **Gradient clipping** provides a safety net against exploding gradients that could derail training.

7. **Mixed precision training** cuts memory usage nearly in half and speeds up computation, making large-scale training practical on modern hardware.

Every one of these components is essential. Skip tokenization, and the model cannot read. Remove batching, and training takes forever. Use vanilla SGD instead of Adam, and convergence is painfully slow. Skip gradient clipping, and a single bad batch can undo hours of training.

The training pipeline is not glamorous — it does not appear in paper abstracts. But it is what separates a randomly initialized matrix from a model that can write poetry, debug code, and explain quantum mechanics.

The architecture provides the **capacity** to learn. The training pipeline provides the **process** of learning. Together, they produce intelligence.

That's it! Thanks for reading.
