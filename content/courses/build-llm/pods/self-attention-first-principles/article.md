# Self-Attention from First Principles: QKV, Multi-Head Attention, and Positional Encoding

Let us start with a simple observation. When you read the sentence "The cat sat on the mat because **it** was soft," your brain does something remarkable. Your eyes encounter the word "it," and almost instantly, they jump backward to "mat" — not to "cat," not to "sat" — because your brain figured out what "it" refers to. You did not read the sentence one word at a time in isolation. You were constantly connecting words to other words, sometimes far apart, based on meaning.

This is attention.

Now imagine you are trying to build a machine that reads sentences. How would you teach it to make these connections? How would you tell the machine that "it" in one context refers to "mat" but in another sentence might refer to "cat"? This is the fundamental problem that the self-attention mechanism solves, and it is the beating heart of the Transformer architecture that powers GPT, BERT, and virtually every modern language model.

In this article, we will build up self-attention entirely from scratch. We will start with the intuition, move into the mathematics step by step, plug in concrete numbers, and by the end, you will understand every single matrix multiplication that makes Transformers tick.


![Attention as word connections in reading](figures/figure_1.png)
*Attention as word connections in reading*


---

## Why Do We Need Attention?

Before Transformers came along, the dominant approaches for processing text were Recurrent Neural Networks (RNNs) and LSTMs. Let us understand why they were not enough.

**The Sequential Bottleneck.** An RNN processes a sentence word by word, left to right. It maintains a hidden state — think of it as a "memory vector" — that gets updated at every time step. By the time the RNN reaches the 50th word in a sentence, the information about the 1st word has been compressed, distorted, and often lost inside this single memory vector. This is like trying to remember a 50-item grocery list by only keeping a running mental note — by the time you reach the end, you have probably forgotten the first few items.

There is another cost to this sequential nature: you cannot parallelize the computation. The hidden state at time step 10 depends on the hidden state at time step 9, which depends on time step 8, and so on. You have to wait for each step to finish before starting the next. On modern GPUs that thrive on parallelism, this is a massive waste of computational power. Training an RNN on a long document is like forcing a 100-lane highway to operate as a single-lane road.


![The sequential bottleneck of RNNs](figures/figure_2.png)
*The sequential bottleneck of RNNs*


**The Fixed-Window Problem.** Another approach is using convolutional models that look at a fixed window of words at a time — say 5 words. But what if the word you need to attend to is 20 words away? You would need to stack many layers of convolutions just to let information travel that far. This is inefficient. Think of it like a game of telephone: the message gets distorted every time it passes through another layer. The word "it" at position 25 needs to hear from "mat" at position 5, but by the time the signal travels through 4 or 5 convolutional layers, it has been mixed and diluted with everything in between.

**The Attention Solution.** What if, instead of processing words sequentially or in small windows, we let every word look at every other word directly? That is the core idea of self-attention. In a single operation, the word "it" can look at "cat," "sat," "mat," and every other word in the sentence, and decide which ones are relevant. No information bottleneck. No fixed window. Every word has a direct line of sight to every other word.

And here is the truly powerful part: the entire operation can run in parallel. Every word computes its attention simultaneously, so we can fully leverage modern GPU hardware. Where an RNN processes a 100-word sentence in 100 sequential steps, self-attention does it in one.

This is exactly what we want.


![Three approaches to processing sequences](figures/figure_3.png)
*Three approaches to processing sequences*


---

## Queries, Keys, and Values: The Core Intuition

Now let us build the mechanism. Self-attention is built around three simple concepts: **Queries**, **Keys**, and **Values**. Before we dive into any mathematics, let us understand what each one means intuitively.

Think of it like searching in a library.

- **Query (Q):** This is your search question. When the word "it" is trying to figure out what it refers to, it creates a Query: "What am I looking for?"
- **Key (K):** This is the label on each book. Every word in the sentence creates a Key that says: "Here is what I can offer. Here is what I am about."
- **Value (V):** This is the actual content of the book. Once you find a matching book (Key matches Query), you pull out the Value — the actual information that word carries.

Here is the beautiful part: every word in the sentence plays all three roles simultaneously. The word "mat" has its own Query (when it wants to attend to other words), its own Key (so other words can find it), and its own Value (the information it provides when found).

Now the question is: why do we need three separate representations? Why not just use the word embedding directly? This is a subtle but crucial point. Think about it this way — in real life, the question you ask is not the same as the label you put on yourself, and neither of those is the same as the content you actually carry. When you search Google, your search query ("best Italian restaurant near me") is very different from the keywords on a restaurant's website ("family-owned trattoria, homemade pasta"), and both of those are different from the actual information on the page (the menu, hours, reviews). The Query captures what a word *needs*, the Key captures what a word *advertises*, and the Value captures what a word *provides*. By learning these three projections separately, the model gets enormous flexibility in how words find and use each other.


![The library analogy for Query, Key, and Value](figures/figure_4.png)
*The library analogy for Query, Key, and Value*


**How are Q, K, V computed?** Each word starts as an embedding vector — a numerical representation of its meaning. We then multiply this embedding by three separate learned weight matrices to produce Q, K, and V:


$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$


Here, $X$ is the input embedding for a word, and $W^Q$, $W^K$, $W^V$ are weight matrices that the model learns during training. The key insight is that these weight matrices are **learned** — the model figures out the best way to create Queries, Keys, and Values through training on data. Before training, these weight matrices are initialized randomly, so the Q, K, V projections are essentially meaningless. But as the model sees millions of sentences and adjusts the weights through backpropagation, the matrices learn to project embeddings into a space where semantically related words produce similar Queries and Keys. The model discovers, entirely on its own, the right way to ask questions and label answers.

Let us plug in some simple numbers. Suppose the embedding for the word "mat" is a 4-dimensional vector:

$$x_{\text{mat}} = [1.0, \; 0.5, \; -0.3, \; 0.8]$$

And suppose our weight matrix $W^Q$ is a $4 \times 2$ matrix (projecting from 4 dimensions down to $d_k = 2$):

$$W^Q = \begin{bmatrix} 0.2 & 0.1 \\ 0.4 & -0.3 \\ 0.1 & 0.5 \\ -0.2 & 0.3 \end{bmatrix}$$

Then the Query for "mat" is:

$$q_{\text{mat}} = x_{\text{mat}} \cdot W^Q = [1.0, 0.5, -0.3, 0.8] \cdot \begin{bmatrix} 0.2 & 0.1 \\ 0.4 & -0.3 \\ 0.1 & 0.5 \\ -0.2 & 0.3 \end{bmatrix}$$

$$q_{\text{mat}} = [(1.0)(0.2) + (0.5)(0.4) + (-0.3)(0.1) + (0.8)(-0.2), \;\; (1.0)(0.1) + (0.5)(-0.3) + (-0.3)(0.5) + (0.8)(0.3)]$$

$$q_{\text{mat}} = [0.2 + 0.2 - 0.03 - 0.16, \;\; 0.1 - 0.15 - 0.15 + 0.24]$$

$$q_{\text{mat}} = [0.21, \;\; 0.04]$$

This same process happens for every word, and with $W^K$ and $W^V$ too. Each word gets its own Q, K, and V vectors. Now suppose we also compute the Key for "cat" (using a different embedding and the same $W^K$) and get $k_{\text{cat}} = [0.18, \; 0.10]$, and the Key for "mat" turns out to be $k_{\text{mat}} = [0.22, \; 0.03]$. When the word "it" creates its Query and compares it against these Keys, the dot product with "mat" will be higher if the Query and Key happen to align — and that alignment is exactly what the model learns to produce during training.

---

## Scaled Dot-Product Attention: The Full Mechanism

Now we have Queries, Keys, and Values for every word. The next question is: how do we use them to compute attention?

The idea is beautifully simple:

1. **Compare each Query against all Keys** to find out how relevant each word is.
2. **Turn these relevance scores into probabilities** using softmax.
3. **Use these probabilities to take a weighted sum of Values.**

The formula for this is:


$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$


Let us break this down piece by piece.

**Step 1: Compute $QK^T$ (the raw attention scores).**

The dot product $QK^T$ measures how similar each Query is to each Key. If a Query and a Key point in the same direction, their dot product will be large — meaning that word is highly relevant. If they point in different directions, the dot product will be small. You can think of this geometrically: the dot product is proportional to the cosine of the angle between two vectors multiplied by their magnitudes. Two vectors pointing in the same direction give a large positive number. Two vectors at 90 degrees give zero. Two vectors pointing in opposite directions give a large negative number. So the model is essentially asking: "How closely does what I am looking for match what this word is offering?"

**Step 2: Scale by $\sqrt{d_k}$.**

Why do we divide by $\sqrt{d_k}$? Without this scaling, when $d_k$ is large, the dot products can become very large numbers. Large inputs to the softmax function push it into regions where the gradients are extremely small (the softmax output becomes almost one-hot). This makes learning very slow. Dividing by $\sqrt{d_k}$ keeps the values in a reasonable range.

Let us plug in some numbers to see why this matters. Suppose $d_k = 64$ (as in the original Transformer). If each element of Q and K is drawn from a standard normal distribution (mean 0, variance 1), then each element of $QK^T$ is the sum of 64 products of two standard normal variables. The variance of each such product is 1, and since we sum 64 of them, the variance of the dot product is 64. That means the standard deviation is $\sqrt{64} = 8$, so typical values of $QK^T$ might be anywhere from -16 to +16. When you push values like +16 into softmax, $e^{16} \approx 8.9 \times 10^6$ — the softmax output is essentially 1.0 for one word and 0.0 for all others. The gradient for the non-selected words vanishes, and learning stalls. Dividing by $\sqrt{64} = 8$ brings the values back to the range -2 to +2, where softmax produces nicely spread probability distributions. This is exactly what we want.

**Step 3: Apply softmax.**

Softmax converts the raw scores into a probability distribution — all values become positive and sum to 1. This tells us what fraction of attention each word should receive.

**Step 4: Multiply by V.**

We take a weighted sum of Value vectors, where the weights are the attention probabilities. Words that received high attention contribute more to the output. The result is a new representation for each word that now contains information from the words it attended to most. The word "it," after attention, might carry information borrowed mostly from "mat" — effectively resolving the coreference.


![Steps of scaled dot-product attention](figures/figure_5.png)
*Steps of scaled dot-product attention*


### Full Numerical Worked Example

Let us work through a complete example with 3 words and $d_k = 2$. Suppose after projecting with $W^Q$, $W^K$, $W^V$, we get the following vectors for the sentence "I love math":

**Queries:**

$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \quad \text{(rows: "I", "love", "math")}$$

**Keys:**

$$K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix} \quad \text{(rows: "I", "love", "math")}$$

**Values:**

$$V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \quad \text{(rows: "I", "love", "math")}$$

**Step 1: Compute $QK^T$.**

$$QK^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix}$$

Let us compute each element:

- Row 1 of Q $\cdot$ Col 1 of $K^T$: $(1)(0) + (0)(1) = 0$
- Row 1 of Q $\cdot$ Col 2 of $K^T$: $(1)(1) + (0)(0) = 1$
- Row 1 of Q $\cdot$ Col 3 of $K^T$: $(1)(1) + (0)(1) = 1$
- Row 2 of Q $\cdot$ Col 1 of $K^T$: $(0)(0) + (1)(1) = 1$
- Row 2 of Q $\cdot$ Col 2 of $K^T$: $(0)(1) + (1)(0) = 0$
- Row 2 of Q $\cdot$ Col 3 of $K^T$: $(0)(1) + (1)(1) = 1$
- Row 3 of Q $\cdot$ Col 1 of $K^T$: $(1)(0) + (1)(1) = 1$
- Row 3 of Q $\cdot$ Col 2 of $K^T$: $(1)(1) + (1)(0) = 1$
- Row 3 of Q $\cdot$ Col 3 of $K^T$: $(1)(1) + (1)(1) = 2$

$$QK^T = \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$

**Step 2: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$.**

$$\frac{QK^T}{\sqrt{d_k}} = \begin{bmatrix} 0 & 0.707 & 0.707 \\ 0.707 & 0 & 0.707 \\ 0.707 & 0.707 & 1.414 \end{bmatrix}$$

**Step 3: Apply softmax (row-wise).**

For Row 1: $e^0 = 1.0$, $e^{0.707} = 2.028$, $e^{0.707} = 2.028$. Sum $= 5.056$.

$$\text{softmax}(\text{Row 1}) = \left[\frac{1.0}{5.056}, \;\frac{2.028}{5.056}, \;\frac{2.028}{5.056}\right] = [0.198, \; 0.401, \; 0.401]$$

For Row 2: $e^{0.707} = 2.028$, $e^{0} = 1.0$, $e^{0.707} = 2.028$. Sum $= 5.056$.

$$\text{softmax}(\text{Row 2}) = [0.401, \; 0.198, \; 0.401]$$

For Row 3: $e^{0.707} = 2.028$, $e^{0.707} = 2.028$, $e^{1.414} = 4.113$. Sum $= 8.169$.

$$\text{softmax}(\text{Row 3}) = \left[\frac{2.028}{8.169}, \;\frac{2.028}{8.169}, \;\frac{4.113}{8.169}\right] = [0.248, \; 0.248, \; 0.503]$$

So the attention weights matrix is:

$$A = \begin{bmatrix} 0.198 & 0.401 & 0.401 \\ 0.401 & 0.198 & 0.401 \\ 0.248 & 0.248 & 0.503 \end{bmatrix}$$

Have a look at Row 3 ("math"). The word "math" pays the most attention to itself (0.503) and roughly equal attention to "I" and "love" (0.248 each). This makes sense because in our simple setup, the Query for "math" was $[1, 1]$ and the Key for "math" was also $[1, 1]$ — the best match.

Now look at Row 1 ("I"). The word "I" pays almost no attention to itself (0.198) and equal attention to "love" and "math" (0.401 each). Why? Because the Query for "I" is $[1, 0]$ and the Key for "I" is $[0, 1]$ — they point in perpendicular directions, so the dot product is zero. In the language of our library analogy, what "I" is searching for does not match its own label, so it looks elsewhere. This is a small but powerful example of how Q and K let the model learn that a word's "needs" are different from what it "offers."

**Step 4: Multiply attention weights by V.**

$$\text{Output} = A \cdot V = \begin{bmatrix} 0.198 & 0.401 & 0.401 \\ 0.401 & 0.198 & 0.401 \\ 0.248 & 0.248 & 0.503 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$$

Row 1 of the output:

$$[0.198(1) + 0.401(3) + 0.401(5), \;\; 0.198(2) + 0.401(4) + 0.401(6)]$$
$$= [0.198 + 1.203 + 2.005, \;\; 0.396 + 1.604 + 2.406]$$
$$= [3.406, \;\; 4.406]$$

Row 2:

$$[0.401(1) + 0.198(3) + 0.401(5), \;\; 0.401(2) + 0.198(4) + 0.401(6)]$$
$$= [0.401 + 0.594 + 2.005, \;\; 0.802 + 0.792 + 2.406]$$
$$= [3.000, \;\; 4.000]$$

Row 3:

$$[0.248(1) + 0.248(3) + 0.503(5), \;\; 0.248(2) + 0.248(4) + 0.503(6)]$$
$$= [0.248 + 0.744 + 2.515, \;\; 0.496 + 0.992 + 3.018]$$
$$= [3.507, \;\; 4.506]$$

**Final output:**

$$\text{Output} = \begin{bmatrix} 3.406 & 4.406 \\ 3.000 & 4.000 \\ 3.507 & 4.506 \end{bmatrix}$$

Notice something important: the output for "math" (Row 3: $[3.507, 4.506]$) is closest to the Value of "math" ($[5, 6]$) because "math" attended most to itself. The output for "love" (Row 2: $[3.000, 4.000]$) is a more balanced mixture. Each word's output is now a context-enriched representation — it knows about the other words around it. This is exactly what we want.


![Attention weights heatmap for 'I love math'](figures/figure_8.png)
*Attention weights heatmap for 'I love math'*


---

## Multi-Head Attention: Learning Multiple Relationships

Now here is a question: what if different words need to attend to each other for different reasons?

Consider the sentence: "The animal didn't cross the street because **it** was too tired."

- One type of attention might focus on **coreference**: figuring out that "it" refers to "animal."
- Another might focus on **syntax**: connecting "was" to its subject "it."
- A third might focus on **semantics**: linking "tired" to "animal" (animals get tired, streets do not).

A single attention mechanism computes one set of attention weights — it can only learn one pattern. But what if we want the model to learn all three patterns simultaneously?

This brings us to **Multi-Head Attention**.

The idea is simple: instead of running attention once with one set of Q, K, V, we run it **multiple times in parallel**, each with different learned weight matrices. Each parallel run is called a **head**.

Here is an analogy that makes this concrete. Imagine you are reading a newspaper article about a political scandal. One part of your brain tracks *who did what to whom* (the agents and actions). Another part tracks *the timeline* (when events happened relative to each other). A third part tracks *the emotional tone* (which statements are accusations versus defenses). You are processing the same text, but through multiple lenses simultaneously. Multi-head attention gives the Transformer this same ability — each head is a different "lens" through which to read the sentence.


![Multi-Head Attention with parallel heads](figures/figure_6.png)
*Multi-Head Attention with parallel heads*


Here is how it works mathematically. If we have $h$ heads:

1. For each head $i$, we compute:


$$
\text{head}_i = \text{Attention}(Q W_i^Q, \; K W_i^K, \; V W_i^V)
$$


2. We concatenate all heads and project:


$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O
$$


where $W^O$ is a final output projection matrix.

Let us plug in some numbers to see the concatenation step. Suppose we have $h = 2$ heads, each producing a 2-dimensional output for a single word. Head 1 produces $[0.5, \; -0.3]$ and Head 2 produces $[0.8, \; 0.1]$. The concatenation gives us a 4-dimensional vector:

$$\text{Concat} = [0.5, \; -0.3, \; 0.8, \; 0.1]$$

Then $W^O$ (a $4 \times 4$ matrix) projects this back to the model dimension. If $W^O$ is the identity-like matrix that simply passes values through, the output would be $[0.5, -0.3, 0.8, 0.1]$. In practice, $W^O$ is a learned matrix that mixes the information from different heads together, letting the model decide how to combine the different "lenses" into a single coherent representation.

**Why does this work?** Each head has its own set of weight matrices ($W_i^Q, W_i^K, W_i^V$), so each head can learn to focus on a different relationship. In practice, researchers have found that different heads do specialize:

- Some heads learn **positional patterns** (attend to the previous word or the next word).
- Some heads learn **syntactic roles** (subject-verb agreement).
- Some heads learn **semantic similarity** (words with related meanings).

This specialization is not programmed — it emerges naturally from training. The model discovers on its own that it is useful to have one head track syntax while another tracks meaning. This is truly amazing.

Let us plug in some numbers to see the dimensionality. In the original Transformer paper ("Attention Is All You Need," Vaswani et al., 2017):

- The model dimension is $d_{\text{model}} = 512$.
- The number of heads is $h = 8$.
- Each head operates on dimension $d_k = d_{\text{model}} / h = 512 / 8 = 64$.

So each head takes the 512-dimensional input, projects it down to 64 dimensions, runs attention, and produces a 64-dimensional output. The 8 outputs are concatenated back to $8 \times 64 = 512$ dimensions, and the final projection $W^O$ maps this back to 512 dimensions.

The total computation is roughly the same as a single full-size attention — we just split it into 8 parallel, smaller attentions. But the representational power is significantly greater because each head can learn its own specialty. A single 512-dimensional attention can only learn one attention pattern per layer, but eight 64-dimensional heads can learn eight distinct patterns. It is like having eight specialized employees instead of one generalist — the total salary (computation) is the same, but the team accomplishes far more.

---

## Positional Encoding: Giving the Transformer a Sense of Order

There is a subtle problem we have been ignoring. Look back at the self-attention mechanism. Does it care about the order of words?

The answer is **no**. If you shuffle the words in the sentence, the attention weights would change (because the Q, K, V vectors change based on the word), but the mechanism itself has no built-in concept of "this word comes first" or "this word comes after that word." The operation is **permutation equivariant**: rearranging the input just rearranges the output in the same way. There is no notion of sequence order baked in.

This is a problem. The sentences "dog bites man" and "man bites dog" contain the exact same words but mean very different things. We need a way to inject positional information.

Let us think about why this matters with a concrete example. Consider two sentences: "Alice thanked Bob" and "Bob thanked Alice." In both sentences, the words are identical — only the order changes. But the meaning is completely different: who did the thanking and who received the thanks flips entirely. Without some way to encode position, the self-attention mechanism would produce the exact same output for both sentences, because it would see the same set of words and the same set of pairwise relationships. The model would be blind to word order, and that is clearly unacceptable for language understanding.

The solution from the original Transformer paper is **sinusoidal positional encoding**. The idea is to add a special vector to each word's embedding that encodes its position in the sentence. This way, "dog" at position 1 gets a different representation than "dog" at position 5.

But why not just use simple integers? Why not just add the number 1 to the first word, 2 to the second word, and so on? The problem is scale. If your sentence has 500 words, the positional value for the last word would be 500 — completely dwarfing the carefully learned embedding values that typically live in the range -1 to +1. You could normalize, but then nearby positions (like 499 and 500) become almost indistinguishable. What we want is a scheme where every position has a unique "fingerprint," nearby positions have similar fingerprints, and the values stay in a bounded range regardless of sequence length. Sines and cosines give us exactly this.

The formula for the positional encoding at position $pos$ is:


$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$


$$
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Here, $pos$ is the position of the word in the sentence (0, 1, 2, ...), $i$ is the dimension index, and $d_{\text{model}}$ is the total embedding dimension.

**Why sines and cosines?** This choice is not arbitrary. There are two key reasons:

1. **Relative positions can be represented as linear transformations.** For any fixed offset $k$, the encoding at position $pos + k$ can be expressed as a linear function of the encoding at position $pos$. This means the model can easily learn to attend to relative positions (e.g., "the word 2 positions back"). Mathematically, this works because of the trigonometric identity: $\sin(a + b) = \sin(a)\cos(b) + \cos(a)\sin(b)$. The encoding at position $pos + k$ is a fixed rotation of the encoding at position $pos$ — and rotations are linear operations that a weight matrix can easily learn.

2. **Generalization to longer sequences.** Since sinusoidal functions are defined for all integers, the model can handle sequences longer than any it saw during training.

The different frequencies across dimensions mean that early dimensions capture coarse position (slowly changing sine waves) while later dimensions capture fine-grained position (rapidly oscillating sine waves). Think of it like the hands on a clock: the second hand changes rapidly and tells you the fine-grained time, the minute hand changes more slowly, and the hour hand changes very slowly. Together, they uniquely identify any moment in time. Similarly, the multiple frequencies in the positional encoding uniquely identify any position in the sequence.


![Sinusoidal positional encoding heatmap](figures/figure_7.png)
*Sinusoidal positional encoding heatmap*


### Worked Example: Computing Positional Encodings

Let us compute the positional encoding for position $pos = 3$ with $d_{\text{model}} = 4$ (for simplicity). We have 4 dimensions, so $i = 0, 1$.

**Dimension 0** ($2i = 0$, so $i = 0$):

$$PE_{(3, 0)} = \sin\!\left(\frac{3}{10000^{0/4}}\right) = \sin\!\left(\frac{3}{10000^0}\right) = \sin\!\left(\frac{3}{1}\right) = \sin(3) = 0.141$$

**Dimension 1** ($2i+1 = 1$, so $i = 0$):

$$PE_{(3, 1)} = \cos\!\left(\frac{3}{10000^{0/4}}\right) = \cos(3) = -0.990$$

**Dimension 2** ($2i = 2$, so $i = 1$):

$$PE_{(3, 2)} = \sin\!\left(\frac{3}{10000^{2/4}}\right) = \sin\!\left(\frac{3}{10000^{0.5}}\right) = \sin\!\left(\frac{3}{100}\right) = \sin(0.03) = 0.030$$

**Dimension 3** ($2i+1 = 3$, so $i = 1$):

$$PE_{(3, 3)} = \cos\!\left(\frac{3}{10000^{2/4}}\right) = \cos(0.03) = 1.000$$

So the positional encoding for position 3 is:

$$PE_3 = [0.141, \; -0.990, \; 0.030, \; 1.000]$$

This vector gets **added** to the word embedding at position 3. If the word at position 3 has an embedding $[0.5, 0.3, -0.1, 0.8]$, then the input to the Transformer becomes:

$$[0.5 + 0.141, \; 0.3 + (-0.990), \; -0.1 + 0.030, \; 0.8 + 1.000] = [0.641, \; -0.690, \; -0.070, \; 1.800]$$

Notice how different positions would produce different encoding vectors, so the same word at different positions gets a different input representation. This is exactly what we want.

Let us also compute $PE_0$ for comparison:

$$PE_0 = [\sin(0), \cos(0), \sin(0), \cos(0)] = [0, 1, 0, 1]$$

And $PE_1$:

$$PE_1 = [\sin(1), \cos(1), \sin(0.01), \cos(0.01)] = [0.841, \; 0.540, \; 0.010, \; 1.000]$$

The first two dimensions change rapidly across positions (high-frequency sinusoidal), while the last two change very slowly (low-frequency). This multi-frequency representation allows the model to distinguish between nearby positions (using the rapidly changing dimensions) and far-apart positions (using the slowly changing ones).

It is worth noting that many modern Transformers have moved beyond sinusoidal encoding. Models like GPT use **learned positional embeddings** — they simply create a trainable embedding vector for each position, just like word embeddings. Others, like RoPE (Rotary Position Embedding) used in LLaMA, encode relative positions directly into the attention computation by rotating Q and K vectors. But the original sinusoidal encoding remains the clearest way to understand the core idea: positions need explicit representation, and the encoding must allow the model to reason about both absolute and relative positions.

---

## Putting It Together: The Transformer Block

Now that we understand self-attention, multi-head attention, and positional encoding, let us see how they fit together inside a Transformer.

A single Transformer block consists of four components stacked in sequence:

1. **Multi-Head Self-Attention:** The layer we just learned. Each word attends to all other words.
2. **Add & Norm (Residual Connection + Layer Normalization):** We add the input to the output of the attention layer (a residual or "skip" connection), then normalize.
3. **Feed-Forward Network (FFN):** A simple two-layer neural network applied independently to each position.
4. **Add & Norm again:** Another residual connection followed by normalization.


![Architecture of a single Transformer block](figures/figure_9.png)
*Architecture of a single Transformer block*


**Why residual connections?** The idea, borrowed from ResNets, is that the model can always fall back to the identity function — if the attention or FFN layer does not help, the information passes through unchanged. This makes training much more stable, especially in deep networks. Think of it like a highway with an exit ramp: the information can take the exit (go through attention) and pick up new context, or it can stay on the highway (the skip connection) and pass through unchanged. In practice, the output is a blend of both — the original information plus whatever the attention layer added. This ensures that stacking more layers never makes things worse, because the worst case is just passing the input through.

**Why Layer Normalization?** It keeps the activations at a consistent scale across dimensions, which speeds up training and reduces sensitivity to initialization. Without normalization, the values flowing through the network can grow or shrink uncontrollably as they pass through dozens of layers, making the model fragile and difficult to train.

The Feed-Forward Network is typically:


$$
\text{FFN}(x) = \max(0, \; x W_1 + b_1) \, W_2 + b_2
$$


Let us plug in some numbers to see what this does. Suppose the input to the FFN for one word is a 4-dimensional vector $x = [0.5, \; -0.3, \; 0.8, \; 0.1]$, and suppose $W_1$ is a $4 \times 8$ matrix that projects up to 8 dimensions, and $W_2$ is an $8 \times 4$ matrix that projects back down. The hidden layer computes $\max(0, \; xW_1 + b_1)$ — the ReLU activation zeros out any negative values — and then $W_2$ projects back to 4 dimensions. The FFN acts as a small "thinking step" applied independently to each word's representation. While self-attention mixes information *between* words, the FFN processes the information *within* each word — you can think of it as each word "digesting" the contextual information it just gathered from attention.

This is just a ReLU-activated hidden layer followed by a linear projection. In the original Transformer, the hidden dimension is 2048 (4 times the model dimension of 512).

The full Transformer encoder stacks 6 of these blocks on top of each other. The Transformer decoder has a similar structure but adds a **cross-attention** layer between the self-attention and FFN, which allows the decoder to attend to the encoder's outputs. But that is a story for another time — for now, the encoder block above captures the essential architecture.

---

## Code: Self-Attention in PyTorch

Enough theory, let us look at some practical implementation now. Here is a minimal, self-contained implementation of scaled dot-product attention and multi-head attention in PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projection matrices for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, attention_weights
```

Let us understand this code in detail.

- **Lines 1-4:** We import PyTorch and the math module. Nothing fancy.
- **Lines 6-15:** The `__init__` method sets up the four weight matrices: $W^Q$, $W^K$, $W^V$, and $W^O$. Each is a linear layer of size `d_model x d_model`. The dimension per head $d_k$ is computed as `d_model // num_heads`.
- **Lines 17-21:** The `forward` method starts by projecting the input `x` into Q, K, V. The `.view()` and `.transpose()` reshape the tensors so that we have separate heads: the shape goes from `(batch, seq, d_model)` to `(batch, heads, seq, d_k)`.
- **Lines 23-25:** This is the core: we compute $QK^T / \sqrt{d_k}$, apply softmax, and multiply by V. This is the exact formula we derived earlier.
- **Lines 27-29:** We reshape the output back from `(batch, heads, seq, d_k)` to `(batch, seq, d_model)` by concatenating the heads, then apply the output projection $W^O$.

You can test it:

```python
# Create a random input: batch=1, sequence length=10, d_model=64
x = torch.randn(1, 10, 64)

# Multi-head attention with 8 heads
mha = MultiHeadSelfAttention(d_model=64, num_heads=8)
output, attn_weights = mha(x)

print(f"Input shape:            {x.shape}")        # [1, 10, 64]
print(f"Output shape:           {output.shape}")    # [1, 10, 64]
print(f"Attention weights shape: {attn_weights.shape}")  # [1, 8, 10, 10]
```

The attention weights have shape `[1, 8, 10, 10]` — that is 8 heads, each producing a $10 \times 10$ matrix of attention weights (one weight for every pair of positions in the 10-word sequence). You can visualize these to see what each head has learned.

---

## Conclusion

Let us recap the journey we have taken.

We started with a simple observation: when reading a sentence, we constantly connect distant words based on meaning. We saw why sequential models like RNNs and fixed-window models like CNNs struggle with this — they either compress everything into a single bottleneck or require many layers to propagate information.

Self-attention solves this by letting every word attend to every other word directly. The mechanism is elegant: each word produces a **Query** (what it is looking for), a **Key** (what it offers), and a **Value** (the information it carries). The dot product between Queries and Keys determines attention weights, which are used to compute a weighted sum of Values. The separation of Q, K, and V is what gives the model its power — a word's needs, its identity, and its content are all represented independently, letting the model learn rich, flexible attention patterns.

We then saw how **Multi-Head Attention** runs multiple attention computations in parallel, allowing the model to learn different types of relationships — syntax, semantics, coreference — all at once. Each head is a different lens through which the model reads the input, and the combined view is far richer than any single perspective.

Because self-attention has no inherent notion of word order, we add **Positional Encodings** — sinusoidal signals that encode each word's position using different frequencies across dimensions, much like the hands of a clock work at different speeds to uniquely identify each moment in time.

Finally, we saw how these pieces assemble into a **Transformer block**: self-attention, followed by a residual connection and normalization, then a feed-forward network, and another residual connection and normalization. Each block enriches the representation, and stacking multiple blocks lets the model build increasingly abstract and powerful representations of the input.

The entire Transformer architecture, which powers GPT, BERT, and all their successors, is built by stacking these blocks. The self-attention mechanism is the key innovation — it replaced recurrence and convolutions with a single, parallelizable operation that can model relationships between any two positions in a sequence, regardless of distance.

That's it! If you found this useful, share it with someone who wants to understand Transformers from first principles.

Thanks!
