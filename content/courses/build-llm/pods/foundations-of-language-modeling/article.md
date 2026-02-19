# Foundations of Language Modeling
## From N-grams to Neural LMs to Transformers — How Machines Learned to Predict the Next Word

---

Let us start with a simple game. I will give you the beginning of a sentence, and your job is to guess the next word.

"The cat sat on the ___"

If you said "mat," you are in good company — most people do. Let us try another one.

"I went to the grocery ___"

Almost certainly, you said "store." One more:

"The president of the United ___"

"States." Of course.

Now here is the key insight: **you are already a language model.** Your brain, trained on decades of reading and conversation, assigns probabilities to the next word based on the words that came before. You did not consciously calculate anything — you just *knew* that "mat" was more likely than "airplane" after "The cat sat on the."

The central question of this article is: **How do we teach a machine to do the same thing?**

It turns out that this question has been asked and answered three different times over the past three decades, each time with a more powerful idea. First, researchers tried **counting** (N-grams). Then they tried **learning** (Neural Language Models). Finally, they tried **attending** (Transformers). Each generation solved the fundamental limitations of the previous one, and together they form the foundation on which GPT, BERT, LLaMA, and every modern large language model is built.

Let us trace this journey from the beginning.


![Three eras of language modeling: counting, learning, attending](figures/figure_1.png)
*Three eras of language modeling: counting, learning, attending*


---

## N-gram Language Models — Counting Words

### The Core Idea — Predict by Counting

Let us think about this intuitively. Suppose you have a friend who always talks about food. You have heard them say "I want pizza" about 50 times, "I want pasta" about 30 times, and "I want sushi" about 20 times. If they start a sentence with "I want," what would you predict comes next? Probably "pizza," because you have *counted* how often each word follows "I want" in your experience.

This is exactly the idea behind N-gram language models. We count how often words appear together, and we use those counts to estimate probabilities.

Formally, a **language model** assigns a probability to a sequence of words. Given a sentence with words $w_1, w_2, \ldots, w_n$, we want to compute:

$$P(w_1, w_2, \ldots, w_n)$$

Using the **chain rule of probability**, we can decompose this joint probability into a product of conditional probabilities:


$$
P(w_1, w_2, \ldots, w_n) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1, w_2) \cdots P(w_n \mid w_1, \ldots, w_{n-1})
$$


Let us plug in some simple numbers to see how this works. Consider the sentence "I want pizza now" (4 words). We want:

$$P(\text{"I want pizza now"}) = P(\text{"I"}) \times P(\text{"want"} \mid \text{"I"}) \times P(\text{"pizza"} \mid \text{"I want"}) \times P(\text{"now"} \mid \text{"I want pizza"})$$

Suppose from a large corpus we estimate: $P(\text{"I"}) = 0.05$, $P(\text{"want"} \mid \text{"I"}) = 0.08$, $P(\text{"pizza"} \mid \text{"I want"}) = 0.12$, and $P(\text{"now"} \mid \text{"I want pizza"}) = 0.25$. Then:

$$P(\text{"I want pizza now"}) = 0.05 \times 0.08 \times 0.12 \times 0.25 = 0.00012$$

This tells us that this particular sentence has a probability of 0.012% — small, but positive. A nonsensical sentence like "I want pizza elephant" would get an even smaller probability. This is exactly what we want — the model assigns higher probability to sentences that make sense.

### The Markov Assumption — Short Memory

But there is a problem. Look at the last term: $P(\text{"now"} \mid \text{"I want pizza"})$. To estimate this, we need to count how many times the exact sequence "I want pizza" appeared in our training data, and then how many times it was followed by "now." For short sentences this is manageable, but what about a 20-word sentence? We would need to condition on a 19-word history — and we will almost never have seen that exact 19-word sequence in our data.

The solution is the **Markov assumption**: instead of conditioning on the entire history, we only look at the last $(n-1)$ words. This gives us the **N-gram** model.

A **bigram** model (n=2) only looks at the previous word:

$$P(w_i \mid w_1, \ldots, w_{i-1}) \approx P(w_i \mid w_{i-1})$$

A **trigram** model (n=3) looks at the previous two words:

$$P(w_i \mid w_1, \ldots, w_{i-1}) \approx P(w_i \mid w_{i-2}, w_{i-1})$$

And the bigram probability is simply estimated by counting:


$$
P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1},\; w_i)}{\text{Count}(w_{i-1})}
$$

Let us work through a concrete example. Suppose our entire training corpus is these three sentences:

- "the cat sat on the mat"
- "the cat ate the fish"
- "the dog sat on the mat"

Now we want to compute $P(\text{"sat"} \mid \text{"cat"})$. We count:
- Count("cat", "sat") = how many times "cat" is followed by "sat"? Looking at the corpus: "the **cat sat**" appears in sentence 1 and... actually in sentence 1 only. Wait — sentence 2 has "cat ate", and sentence 3 has "dog sat." So Count("cat", "sat") = 1.
- Count("cat") = how many times does "cat" appear? In sentences 1 and 2. So Count("cat") = 2.

Therefore:

$$P(\text{"sat"} \mid \text{"cat"}) = \frac{1}{2} = 0.5$$

Similarly, $P(\text{"ate"} \mid \text{"cat"}) = \frac{1}{2} = 0.5$. Given that we just saw "cat," the model thinks "sat" and "ate" are equally likely. This makes sense given our tiny corpus.


![Unigram, bigram, and trigram sliding windows over a sentence](figures/figure_2.png)
*Unigram, bigram, and trigram sliding windows over a sentence*


Here is a simple Python implementation of bigram counting:

```python
from collections import defaultdict

# Our tiny corpus
corpus = [
    "the cat sat on the mat",
    "the cat ate the fish",
    "the dog sat on the mat"
]

# Count bigrams and unigrams
bigram_counts = defaultdict(int)
unigram_counts = defaultdict(int)

for sentence in corpus:
    words = sentence.split()
    for i in range(len(words)):
        unigram_counts[words[i]] += 1
        if i < len(words) - 1:
            bigram_counts[(words[i], words[i+1])] += 1

# Compute bigram probability
def bigram_prob(w1, w2):
    return bigram_counts[(w1, w2)] / unigram_counts[w1]

print(f"P('sat' | 'cat') = {bigram_prob('cat', 'sat')}")  # 0.5
print(f"P('ate' | 'cat') = {bigram_prob('cat', 'ate')}")  # 0.5
print(f"P('sat' | 'dog') = {bigram_prob('dog', 'sat')}")  # 1.0
```

### Where N-grams Break — The Sparsity Problem

N-gram models are elegant in their simplicity, but they have a fundamental flaw. What happens when we encounter a word pair we have never seen before?

Suppose someone asks: what is $P(\text{"ran"} \mid \text{"cat"})$? In our corpus, "cat" was never followed by "ran." So Count("cat", "ran") = 0, and the probability is zero. But "the cat ran" is a perfectly valid English sentence! A probability of zero means the model considers it *impossible* — which is clearly wrong.

This is the **sparsity problem.** And it gets much worse as we increase n. If our vocabulary has 50,000 words, then:
- Bigrams: $50{,}000^2 = 2.5$ billion possible pairs
- Trigrams: $50{,}000^3 = 125$ trillion possible triples

No corpus in the world is large enough to observe even a fraction of these combinations. Most entries in the count table will be zero — not because those sequences are impossible, but because we simply have not seen them yet.

Researchers developed clever smoothing techniques to redistribute some probability mass to unseen N-grams. The simplest is **Laplace smoothing** (also called add-one smoothing): instead of using raw counts, we add 1 to every count. If our vocabulary size is $V$, the smoothed bigram probability becomes:

$$P_{\text{smooth}}(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i) + 1}{\text{Count}(w_{i-1}) + V}$$

Let us see how this helps. Using our earlier corpus, what is the smoothed probability $P_{\text{smooth}}(\text{"ran"} \mid \text{"cat"})$? We have Count("cat", "ran") = 0, Count("cat") = 2, and suppose our vocabulary has $V = 6$ words (the, cat, dog, sat, ate, ran). Then:

$$P_{\text{smooth}}(\text{"ran"} \mid \text{"cat"}) = \frac{0 + 1}{2 + 6} = \frac{1}{8} = 0.125$$

No longer zero — the model now assigns a 12.5% probability instead of an impossible 0%. But notice the cost: the probability of "sat" given "cat" drops from $\frac{1}{2} = 0.5$ down to $\frac{1 + 1}{2 + 6} = \frac{2}{8} = 0.25$. We have stolen probability mass from things we *have* observed and redistributed it to things we have not. More advanced techniques like **Kneser-Ney smoothing** do this redistribution more cleverly, but they are still patches, not solutions.

Let us also appreciate just how badly sparsity scales in practice. Consider building a trigram model for English. Even a modest vocabulary of 50,000 words produces $50{,}000^3 = 125$ trillion possible trigrams. The entire English Wikipedia contains roughly 4 billion words — meaning we observe at most 4 billion trigram instances to fill 125 trillion slots. That is one observation for every 31,250 possible trigrams. The vast majority of our count table is filled with zeros, and no amount of smoothing can conjure genuine understanding from absence.

But the most fundamental limitation of N-grams is deeper than sparsity: **they have no notion of similarity.** To an N-gram model, "cat" and "dog" are completely unrelated symbols — just different row indices in a table. Learning that "the cat sat on the mat" tells the model absolutely nothing about "the dog sat on the rug." Each word is just an arbitrary index in a vocabulary — there is no understanding that cats and dogs are both animals, or that mats and rugs are both things you sit on.

Think of it this way. Imagine you are learning vocabulary in a foreign language, and someone teaches you "le chat dort" means "the cat sleeps." An N-gram model is like memorizing that exact phrase but gaining zero insight about "le chien dort" ("the dog sleeps") because it treats "chat" and "chien" as completely unrelated symbols. A human, on the other hand, would reason: "chat and chien are both animals, so the sentence probably has the same structure." N-gram models cannot do this. Every word combination must be observed independently — there is no transfer of knowledge between similar words.

So, what if instead of counting words, we could *learn* representations that capture what words actually mean? This brings us to one of the most important ideas in the history of natural language processing.

---

## Neural Language Models — Learning Representations

### Bengio's Breakthrough (2003) — Words as Vectors

In 2003, Yoshua Bengio and his colleagues published a paper called "A Neural Probabilistic Language Model" that changed everything. The core idea was deceptively simple: **represent each word as a dense vector of real numbers** (called an **embedding**), and use a neural network to predict the next word from these embeddings.

Let us understand why this is revolutionary with an analogy. Imagine you are in a foreign city and you need to find a restaurant. The N-gram approach is like having a massive phone book where you look up the exact address. If the restaurant is not in the book, you are stuck. The neural approach is like having a *map* — even if the exact restaurant is not marked, you can see that there is a cluster of restaurants in a particular neighborhood and walk there. The "map" is the embedding space, and similar words live in the same neighborhood.

The Bengio model works as follows. Given the previous $(n-1)$ words, we:

1. **Look up** the embedding vector for each context word using an embedding matrix $C$
2. **Concatenate** these vectors into one long vector $x$
3. **Pass** $x$ through a hidden layer with a tanh activation
4. **Produce** a probability distribution over the entire vocabulary using softmax

Mathematically:


$$
P(w_t \mid w_{t-n+1}, \ldots, w_{t-1}) = \text{softmax}(W \cdot h + b), \quad \text{where} \quad h = \tanh(H \cdot x + d), \quad x = [C(w_{t-n+1}); \ldots; C(w_{t-1})]
$$


Let us plug in some numbers with a toy example. Suppose we have a vocabulary of just 3 words: {"cat"=0, "sat"=1, "mat"=2}, and we are using bigrams (n=2) with an embedding dimension of 2.

Our embedding matrix $C$ might be:

$$C = \begin{bmatrix} 0.2 & 0.8 \\ 0.5 & 0.1 \\ 0.9 & 0.3 \end{bmatrix}$$

So the embedding for "cat" (index 0) is $[0.2, 0.8]$. Now, to predict what comes after "cat," our input $x = C(\text{"cat"}) = [0.2, 0.8]$.

Suppose the hidden layer weight matrix $H$ is $\begin{bmatrix} 0.3 & -0.1 \\ 0.4 & 0.2 \end{bmatrix}$ and bias $d = [0, 0]$. Then:

$$h = \tanh\left(\begin{bmatrix} 0.3 & -0.1 \\ 0.4 & 0.2 \end{bmatrix} \begin{bmatrix} 0.2 \\ 0.8 \end{bmatrix}\right) = \tanh\left(\begin{bmatrix} -0.02 \\ 0.24 \end{bmatrix}\right) = \begin{bmatrix} -0.02 \\ 0.235 \end{bmatrix}$$

After the output layer and softmax, we get a probability distribution over {"cat", "sat", "mat"} — say $[0.15, 0.60, 0.25]$. The model predicts "sat" is most likely after "cat." This is exactly what we want.

The crucial difference from N-grams: **every parameter is learned.** The embeddings, the hidden layer weights, and the output weights are all trained jointly by showing the network millions of sentences and adjusting weights to maximize the probability of the correct next word. Think of the difference this way: an N-gram model is like a massive phone book — you either find the exact entry you need, or you are out of luck. A neural language model is like understanding the *structure* of phone numbers — you know that numbers in the same area code probably correspond to the same city, even if you have never seen that specific number before. The neural model generalizes because it has learned patterns, not just memorized entries.


![Bengio's neural language model: embeddings, hidden layer, and softmax output](figures/figure_3.png)
*Bengio's neural language model: embeddings, hidden layer, and softmax output*


### The Power of Shared Representations

Now here is why Bengio's idea is a game-changer. During training, the model learns that "cat" and "dog" appear in similar contexts — "the cat sat," "the dog sat," "the cat ran," "the dog ran." As a result, their embedding vectors end up close together in the embedding space. This means that **anything the model learns about "cat" automatically transfers to "dog."**

Let us see this with numbers. Suppose after training, the embeddings are:
- "cat" = $[0.82, 0.15, 0.91]$
- "dog" = $[0.79, 0.18, 0.88]$
- "pizza" = $[0.10, 0.95, 0.22]$

The distance between "cat" and "dog" is $\sqrt{(0.82 - 0.79)^2 + (0.15 - 0.18)^2 + (0.91 - 0.88)^2} = \sqrt{0.0027} = 0.052$. Tiny. The distance between "cat" and "pizza" is $\sqrt{(0.82 - 0.10)^2 + (0.15 - 0.95)^2 + (0.91 - 0.22)^2} = \sqrt{1.165} = 1.079$. Much larger. The neural network sees "cat" and "dog" as nearly the same thing, but "cat" and "pizza" as completely different. This is exactly what we want.

If the model has seen "the cat sat on the mat" many times but has never seen "the dog sat on the rug," it can still assign a reasonable probability to that sentence — because "dog" is close to "cat" and "rug" is close to "mat" in the embedding space. The sparsity problem is solved. An N-gram model would assign probability zero to this unseen sentence; a neural language model assigns a reasonable probability because it has learned that the *pattern* is the same — an animal sitting on a surface.

In 2013, Tomas Mikolov and colleagues at Google took this idea further with **Word2Vec** — an extremely efficient method for training word embeddings on massive corpora. Word2Vec came in two flavors: **Skip-gram**, which predicts surrounding context words from a center word, and **CBOW** (Continuous Bag of Words), which predicts the center word from its context. Both are remarkably fast to train — Word2Vec could process billions of words in a matter of hours on a single machine, a dramatic improvement over earlier methods.

Word2Vec showed that the resulting embedding spaces capture remarkable semantic relationships. The most famous example:

$$\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$$

This is not a trick — it is a natural consequence of learning representations from context. Words that are used in similar ways end up at similar locations in the vector space, and the directions between them encode meaningful relationships like gender, tense, and plurality. Other examples abound: $\vec{\text{Paris}} - \vec{\text{France}} + \vec{\text{Italy}} \approx \vec{\text{Rome}}$, and $\vec{\text{walking}} - \vec{\text{walk}} + \vec{\text{swim}} \approx \vec{\text{swimming}}$. The embedding space has learned geography, grammar, and semantics — all from simply predicting words in context.


![Word embedding space with semantic clusters and King-Queen vector arithmetic](figures/figure_4.png)
*Word embedding space with semantic clusters and King-Queen vector arithmetic*


### Recurrent Neural Networks — Unlimited Context?

Bengio's model was a huge step forward, but it still uses a **fixed context window** — just like N-grams, it only looks at the last $(n-1)$ words. If we set $n = 5$, the model cannot use any information from 6 words ago. Can we do better?

This is where **Recurrent Neural Networks (RNNs)** come in. The idea is beautiful in its simplicity: process one word at a time, and carry a **hidden state** $h_t$ forward that summarizes everything the model has seen so far. At each time step, the hidden state is updated based on the current input word and the previous hidden state:


$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b)
$$


Let us work through a numerical example. Suppose we have a 2-dimensional hidden state and are processing the 3-word sentence "cat sat mat." We set the initial hidden state $h_0 = [0, 0]$.

Let $W_{hh} = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}$, $W_{xh} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$, and bias $b = [0, 0]$.

If the embedding for "cat" is $x_1 = [0.2, 0.8]$:

$$h_1 = \tanh\left(\begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}\begin{bmatrix} 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 0.2 \\ 0.8 \end{bmatrix}\right) = \tanh\left(\begin{bmatrix} 0.2 \\ 0.8 \end{bmatrix}\right) = \begin{bmatrix} 0.197 \\ 0.664 \end{bmatrix}$$

Now for "sat" with $x_2 = [0.5, 0.1]$:

$$h_2 = \tanh\left(\begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}\begin{bmatrix} 0.197 \\ 0.664 \end{bmatrix} + \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}\right) = \tanh\left(\begin{bmatrix} 0.599 \\ 0.432 \end{bmatrix}\right) = \begin{bmatrix} 0.537 \\ 0.408 \end{bmatrix}$$

Notice how $h_2$ encodes information about *both* "cat" and "sat" — it carries the memory forward. In theory, this means the RNN has access to the entire history of the sentence, no matter how long.

But there is a catch. During training, we compute gradients through the chain of hidden states using **backpropagation through time**. The gradient of the loss with respect to early hidden states involves multiplying many weight matrices together. If the weights are less than 1, the gradient shrinks exponentially — this is the **vanishing gradient problem.** In practice, this means that RNNs struggle to remember information from more than about 10-20 steps back.

Researchers developed more sophisticated architectures — **LSTMs** (Long Short-Term Memory) and **GRUs** (Gated Recurrent Units) — that use learned gates to control what information to remember and what to forget. These extended the effective memory to roughly 100-200 tokens, which is much better but still limited.

To see why long-range dependencies are so hard for RNNs, consider this sentence: "The author who wrote the bestselling series of fantasy novels that were adapted into blockbuster films **was** born in England." The subject "author" and the verb "was" are separated by 15 words. To get "was" right (and not "were"), the model needs to remember that the subject is singular — but all those intervening words about novels and films keep overwriting the hidden state. It is like a game of telephone: by the time the message reaches the end, the original signal has been diluted beyond recognition. LSTMs and GRUs help by providing a more direct pathway for information to flow, but even they struggle when the dependency spans hundreds of tokens.

And there is another fundamental limitation: RNNs process words **sequentially**, one at a time. You cannot process word 5 until you have processed words 1 through 4. This means RNNs cannot take advantage of modern parallel hardware like GPUs, making them painfully slow to train on large datasets. To put this in perspective, training a large RNN on a 1-billion-word corpus might take weeks on a GPU, because each word must wait for the previous word to be processed. This sequential bottleneck becomes the key obstacle to scaling up language models.

We need a model that can look at *all* words simultaneously and decide which ones matter most. This brings us to the most important architecture in modern AI.


![RNN unrolled through time with the vanishing gradient illustrated as fading arrows](figures/figure_5.png)
*RNN unrolled through time with the vanishing gradient illustrated as fading arrows*


---

## Transformers — Attention Is All You Need

### The Core Intuition — Who Should I Pay Attention To?

Consider this sentence: "The cat sat on the mat because **it** was comfortable."

What does "it" refer to? The mat. How did you figure that out? You did not read the sentence word by word and carry a running memory — you **looked back** at all the previous words and decided that "mat" was the most relevant one for understanding "it." You selectively paid attention to the right word.

Now consider: "The cat sat on the mat because **it** was hungry."

Now "it" refers to the cat! Same sentence structure, different meaning of "it" — and you figured it out by **attending** to a different word.

This selective, context-dependent backward glance is the core intuition behind **self-attention**, the mechanism that powers the Transformer architecture. Instead of processing words one at a time and hoping the hidden state remembers what is important, self-attention lets every word directly look at every other word and decide how much to "pay attention" to it.

Now the question is: why does this solve the long-range dependency problem that crippled RNNs? The answer is beautifully simple. In an RNN, information from word 1 must pass through every intermediate hidden state to reach word 50 — a chain of 49 transformations, each one slightly degrading the signal. In self-attention, word 50 can attend *directly* to word 1 in a single step. The path length between any two words is always 1, regardless of how far apart they are in the sentence. This is like the difference between passing a note through 49 people in a chain versus simply looking across the room and making eye contact. The signal arrives intact.

### The Mechanics — Queries, Keys, and Values

Now let us understand the mechanics of how self-attention works. The idea uses three concepts that are easy to understand with a library analogy.

Imagine you walk into a library looking for information about "machine learning" (this is your **Query** — what you are looking for). Every book on the shelf has a title and a set of keywords on its spine (these are the **Keys** — what each item advertises about itself). When you find books whose keywords match your query, you pull them off the shelf and read their contents (these are the **Values** — the actual information each item carries).

In self-attention, every word in the sentence plays all three roles simultaneously:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I have to offer?"
- **Value (V):** "What information do I carry?"

These three representations are created by multiplying the word's embedding by three learned weight matrices $W_Q$, $W_K$, and $W_V$.

The attention formula is:


$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

This looks dense, so let us break it down step by step with actual numbers. Suppose we have 3 words: "the", "cat", "sat", each with a $d_k = 2$ dimensional representation. After projecting through $W_Q$, $W_K$, $W_V$, suppose we get:

$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{bmatrix}$$

**Step 1: Compute $QK^\top$** — the dot product between every query and every key.

$$QK^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$

Each entry tells us how "relevant" one word is to another. Notice that "sat" (row 3) has the highest score with itself (2) — this makes sense, as a word is always relevant to itself.

**Step 2: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:**

$$\frac{QK^\top}{\sqrt{d_k}} = \begin{bmatrix} 0 & 0.707 & 0.707 \\ 0.707 & 0 & 0.707 \\ 0.707 & 0.707 & 1.414 \end{bmatrix}$$

Why do we divide by $\sqrt{d_k}$? Without this scaling, when $d_k$ is large, the dot products grow large too, pushing the softmax into regions where it produces nearly one-hot outputs and the gradients become very small. Dividing by $\sqrt{d_k}$ keeps the values in a range where softmax produces useful gradients.

**Step 3: Apply softmax row-wise** (each row sums to 1):

$$\text{softmax} = \begin{bmatrix} 0.244 & 0.378 & 0.378 \\ 0.378 & 0.244 & 0.378 \\ 0.262 & 0.262 & 0.476 \end{bmatrix}$$

These are the **attention weights** — each row tells us how much each word attends to every other word.

**Step 4: Multiply by V** to get the final output:

$$\text{Output} = \begin{bmatrix} 0.244 & 0.378 & 0.378 \\ 0.378 & 0.244 & 0.378 \\ 0.262 & 0.262 & 0.476 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 0.433 & 0.567 \\ 0.567 & 0.433 \\ 0.500 & 0.500 \end{bmatrix}$$

Each word's output is a **weighted combination** of all words' values, where the weights are determined by how relevant each word is to the query. This is exactly what we want. The model learns to route information between words based on their content, not their position.


![Step-by-step attention computation with Q, K, V matrices and actual numbers](figures/figure_6.png)
*Step-by-step attention computation with Q, K, V matrices and actual numbers*


Here is a minimal PyTorch implementation of self-attention:

```python
import torch
import torch.nn.functional as F

def self_attention(x, W_q, W_k, W_v):
    """
    x: input embeddings (seq_len, d_model)
    W_q, W_k, W_v: projection matrices (d_model, d_k)
    """
    Q = x @ W_q  # (seq_len, d_k)
    K = x @ W_k  # (seq_len, d_k)
    V = x @ W_v  # (seq_len, d_k)

    d_k = Q.shape[-1]
    scores = Q @ K.T / (d_k ** 0.5)     # (seq_len, seq_len)
    weights = F.softmax(scores, dim=-1)  # attention weights
    output = weights @ V                  # (seq_len, d_k)
    return output

# Example: 3 words, embedding dim = 4, attention dim = 2
x = torch.randn(3, 4)
W_q = torch.randn(4, 2)
W_k = torch.randn(4, 2)
W_v = torch.randn(4, 2)

out = self_attention(x, W_q, W_k, W_v)
print(f"Input shape:  {x.shape}")    # (3, 4)
print(f"Output shape: {out.shape}")  # (3, 2)
```

### Multi-Head Attention and Positional Encoding

A single attention head learns one type of relationship between words. But language has many different types of relationships happening simultaneously — syntactic structure, semantic similarity, coreference, and more. This is why Transformers use **multi-head attention**: they run several attention operations in parallel, each with its own set of $W_Q$, $W_K$, $W_V$ weight matrices.


$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O
$$


where each $\text{head}_i = \text{Attention}(Q W_Q^i, K W_K^i, V W_V^i)$.

Let us make this concrete with a small numerical example. Suppose our model dimension is $d_{\text{model}} = 4$ and we have $h = 2$ attention heads. Each head operates on $d_k = d_{\text{model}} / h = 2$ dimensions. Head 1 might produce an output of shape $(3, 2)$, say $\begin{bmatrix} 0.4 & 0.6 \\ 0.5 & 0.5 \\ 0.3 & 0.7 \end{bmatrix}$, and Head 2 produces $\begin{bmatrix} 0.8 & 0.2 \\ 0.1 & 0.9 \\ 0.6 & 0.4 \end{bmatrix}$. We concatenate these along the last dimension to get a $(3, 4)$ matrix, and then multiply by $W^O$ (a $4 \times 4$ matrix) to produce the final output. The key idea: each head has learned to attend to different relationships, and the concatenation combines these complementary perspectives.

Let us see why this multi-perspective view matters. In the sentence "The cat sat on the mat because it was comfortable," one attention head might learn to connect "it" with "mat" (coreference), while another head connects "comfortable" with "mat" (attribute), and a third head captures the syntactic relationship between "sat" and "on." Each head specializes in a different type of linguistic relationship, and their outputs are concatenated and projected back to the model dimension.

Now, there is one more piece we need. Notice something important: in the attention computation above, we never used the **position** of the words. "The cat sat" would give the same attention weights as "sat cat the" — the operation is permutation-invariant. But word order matters enormously in language!

Transformers solve this by adding **positional encodings** to the input embeddings. The original paper by Vaswani et al. used sinusoidal functions:


$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Let us compute this for positions 0 and 1 with $d_{\text{model}} = 4$ (so $i = 0, 1$):

For **position 0:**
- $PE_{(0, 0)} = \sin(0 / 10000^{0/4}) = \sin(0) = 0$
- $PE_{(0, 1)} = \cos(0 / 10000^{0/4}) = \cos(0) = 1$
- $PE_{(0, 2)} = \sin(0 / 10000^{2/4}) = \sin(0) = 0$
- $PE_{(0, 3)} = \cos(0 / 10000^{2/4}) = \cos(0) = 1$
- So $PE_0 = [0, 1, 0, 1]$

For **position 1:**
- $PE_{(1, 0)} = \sin(1 / 10000^{0}) = \sin(1) = 0.841$
- $PE_{(1, 1)} = \cos(1 / 10000^{0}) = \cos(1) = 0.540$
- $PE_{(1, 2)} = \sin(1 / 10000^{0.5}) = \sin(0.01) = 0.01$
- $PE_{(1, 3)} = \cos(1 / 10000^{0.5}) = \cos(0.01) = 1.0$
- So $PE_1 = [0.841, 0.540, 0.01, 1.0]$

These vectors are different for each position — so when we add them to the word embeddings, the model can distinguish "cat" in position 1 from "cat" in position 5. The use of sin and cos at different frequencies is elegant because it allows the model to learn to attend to relative positions: the vector difference $PE_{pos+k} - PE_{pos}$ is the same regardless of $pos$, which means the model can learn patterns like "two words apart" in a position-independent way.

A complete **Transformer block** consists of:
1. Multi-Head Self-Attention
2. Add & Layer Normalize (residual connection)
3. Feed-Forward Network (two linear layers with a ReLU/GELU activation)
4. Add & Layer Normalize (another residual connection)

Multiple Transformer blocks are stacked on top of each other. Each block refines the representations, building increasingly abstract understanding of the input. The original Transformer paper used 6 blocks; GPT-3 uses 96.


![A single Transformer block: attention, feed-forward, and residual connections](figures/figure_7.png)
*A single Transformer block: attention, feed-forward, and residual connections*


---

## From Language Model to GPT — Putting It All Together

Now we have all the pieces. Let us see how they come together to build a modern language model like GPT.

A Transformer-based language model is a **decoder** — it takes a sequence of tokens and predicts the next token. Specifically, it uses **masked self-attention**, which means that when predicting the word at position $t$, the model can only attend to positions $1, 2, \ldots, t$ — it cannot peek at the future. This ensures that the model makes genuine predictions rather than just copying the answer.

The training objective is **next-token prediction**: given all the words so far, predict the next word. We measure how well the model does this using the **cross-entropy loss**:


$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t \mid w_1, w_2, \ldots, w_{t-1})
$$

Let us work through a concrete example. Suppose our model is processing the 3-token sequence "the cat sat" and we want to compute the loss.

At position 1, the model predicts the probability of "the" (given just the start token). Suppose $P(\text{"the"}) = 0.10$.
At position 2, the model predicts "cat" given "the." Suppose $P(\text{"cat"} \mid \text{"the"}) = 0.05$.
At position 3, the model predicts "sat" given "the cat." Suppose $P(\text{"sat"} \mid \text{"the cat"}) = 0.30$.

The cross-entropy loss is:

$$\mathcal{L} = -[\log(0.10) + \log(0.05) + \log(0.30)]$$
$$= -[-2.303 + (-2.996) + (-1.204)]$$
$$= -[-6.503]$$
$$= 6.503$$

A loss of 6.503 tells us the model is not yet very confident in its predictions — which makes sense for an untrained model. As training progresses, the predicted probabilities for the correct words increase, driving the loss down. A perfect model would assign probability 1.0 to each correct word, giving a loss of 0. In practice, we never reach zero because language is inherently uncertain — there are often multiple valid next words.

This is the beauty of the language modeling objective: it is **self-supervised.** We do not need any labels — the training data *is* the labels. Every word in a sentence serves as the prediction target for the words before it. This means we can train on the entire internet, on every book ever written, on every conversation ever had. The scale of available training data is practically unlimited.

Let us bring the entire journey together. We started with N-grams — count tables that could predict the next word by looking up frequencies but crumbled when they encountered novel combinations. Neural language models replaced the count table with learned embeddings and a neural network, solving sparsity through shared representations. RNNs extended the context window but struggled with long-range dependencies and sequential processing. And Transformers introduced self-attention, allowing every word to directly attend to every other word, with full parallelism and no vanishing gradients.


![The evolution: N-grams count, Neural LMs learn, Transformers attend](figures/figure_8.png)
*The evolution: N-grams count, Neural LMs learn, Transformers attend*


The Transformer unlocked something extraordinary: **scaling.** Once you have an architecture that can process sequences in parallel and attend to any position, you can make it bigger — more layers, more attention heads, wider hidden dimensions — and train it on more data. And remarkably, performance keeps improving as you scale up.

Why could we not scale RNNs the same way? Because of the sequential bottleneck. If you double the dataset, an RNN takes roughly twice as long to train — each token must wait for the previous one. A Transformer, by contrast, processes all tokens in a sequence simultaneously. Doubling the dataset doubles the data but the per-step computation stays parallelizable across thousands of GPU cores. This is why training GPT-3 on 300 billion tokens was even feasible — with RNNs, it would have taken years instead of weeks.

This is the insight that led to GPT-2 (1.5 billion parameters), GPT-3 (175 billion parameters), and the entire generation of large language models that followed.

---

## Conclusion

Let us retrace our steps. We began with a simple game — guessing the next word — and discovered that this game is the foundation of language modeling. We saw three progressively more powerful answers:

1. **N-grams** answer by counting: how often does word B follow word A? Simple and fast, but crippled by sparsity and the inability to recognize that similar words should behave similarly.

2. **Neural Language Models** answer by learning: represent words as dense vectors where similarity is captured automatically. Bengio's 2003 breakthrough and Mikolov's Word2Vec showed that learned embeddings generalize where count tables fail. RNNs extended the context but hit the vanishing gradient wall.

3. **Transformers** answer by attending: let every word look at every other word simultaneously and decide what matters. Self-attention solved long-range dependencies and enabled full parallelism, unlocking the age of scaling.

Every modern LLM — GPT, BERT, LLaMA, Claude, Gemini — is built on the Transformer architecture. Understanding these foundations is not just an academic exercise; it is the key to understanding why these models work, where they struggle, and where they might go next.

The next time your phone's autocomplete suggests the perfect word, remember: behind that suggestion is a journey from simple counting, to learned representations, to the attention mechanism — decades of ideas, each building on the last.

---

**Key Papers:**
- Shannon, C. (1948). "A Mathematical Theory of Communication"
- Bengio, Y. et al. (2003). "A Neural Probabilistic Language Model"
- Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Vaswani, A. et al. (2017). "Attention Is All You Need"
- Radford, A. et al. (2018). "Improving Language Understanding by Generative Pre-Training" (GPT)

That's it! Thanks for reading.
