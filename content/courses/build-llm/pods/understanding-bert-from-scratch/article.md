# Understanding BERT from Scratch
## How a simple idea — reading in both directions — changed NLP forever

Let us start with a simple game. Look at this sentence:

**"The ___ sat on the mat and purred loudly."**

Can you fill in the blank? Of course — it is a **cat**. But how did you figure that out?

You probably looked at the words *after* the blank — "purred loudly" — and instantly thought of a cat. You did not just look at "The" and try to guess. You used the words on **both sides** of the blank to figure out the answer.

Now let us try another example. What goes in the blank here?

**"I went to the ___ to deposit my savings."**

You said **bank**, right? Now try this one:

**"I sat on the ___ of the river and watched the sunset."**

You said **bank** again — but a completely different meaning! The words *around* the blank told you which "bank" was meant.

This is something we do effortlessly — we read in **both directions**. We look at the words before and after to understand the meaning of every single word in a sentence.

Now the question is: **Can we build a model that reads a sentence the way we do — looking in both directions?**

Before 2018, most language models could only read in **one direction** — left to right. They would read "The" and try to predict the next word, then read "The cat" and predict the next word, and so on. They never got to peek at the words that came after.


![Unidirectional models miss right-side context; bidirectional models capture the full picture.](figures/figure_1.png)
*Unidirectional models miss right-side context; bidirectional models capture the full picture.*


This simple idea — reading in both directions — is the key insight behind **BERT** (Bidirectional Encoder Representations from Transformers), and it changed the field of Natural Language Processing forever.

---

## The Problem with Previous Approaches

To appreciate why BERT was such a breakthrough, let us first understand what came before it.

### Static Word Embeddings (Word2Vec)

Before deep contextual models, the dominant approach for representing words was **Word2Vec** (Mikolov et al., 2013). The idea was simple: train a model to represent each word as a fixed vector of numbers. Words that appear in similar contexts end up with similar vectors.

For example, "king" and "queen" would have vectors that are close together, and "cat" and "dog" would also be close.

This was a huge step forward. But there was a fundamental problem.

**Every word gets exactly ONE vector, regardless of context.**

This means the word "bank" in "river bank" gets the *exact same* representation as "bank" in "I went to the bank to deposit money." The model has no way to distinguish between the two.

This is clearly not how language works. The meaning of a word depends entirely on the context in which it appears.

### ELMo: A Step in the Right Direction

In 2018, Peters et al. introduced **ELMo** (Embeddings from Language Models). ELMo was a major improvement because it used **bidirectional LSTMs** to create context-dependent word representations.

The way ELMo worked was as follows: it trained a left-to-right LSTM and a right-to-left LSTM **separately**, and then simply **concatenated** their hidden states.

This gave each word a representation that depended on its context — a significant advance over Word2Vec.

But ELMo had a limitation: the left-to-right and right-to-left models were trained independently. They never truly interacted with each other during training. The bidirectionality was shallow — it was stitched together at the end, not learned jointly.


![From static embeddings to deep bidirectional representations.](figures/figure_2.png)
*From static embeddings to deep bidirectional representations.*


This brings us to the main character in our story.

---

## The Transformer Encoder — BERT's Engine

Before we understand BERT, we need to understand the engine that powers it: the **Transformer Encoder**.

The Transformer was introduced in the landmark paper "Attention Is All You Need" (Vaswani et al., 2017). The key mechanism inside the Transformer is called **self-attention**.

### Self-Attention: The Core Idea

Imagine you are reading a long email, and you encounter the sentence: "The delivery arrived late, but **it** was in perfect condition." When you read the word "it," your eyes instinctively jump back to "delivery" to figure out what "it" refers to.

Self-attention does exactly this — for every word in the input, it looks at **all other words** to figure out which ones are relevant.

But how does it decide which words to pay attention to? This is where the three key players come in: **Query**, **Key**, and **Value**.

Think of it like a library:

- The **Query** is your search question — "What am I looking for?"
- The **Key** is the label on each book — "What information does this book contain?"
- The **Value** is the actual content of the book — "Here is the information you need."

For every word, we create a Query vector, a Key vector, and a Value vector by multiplying the word's embedding with three learned weight matrices.

Here is how self-attention works step by step:

**Step 1:** Compute the dot product of the Query with all Keys. This tells us how relevant each word is to the current word.

**Step 2:** Divide by the square root of the key dimension to keep the numbers stable.

**Step 3:** Apply softmax to get attention weights that sum to 1.

**Step 4:** Multiply each Value by its attention weight and sum them up.

The mathematical formula for this is:


$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let us plug in some simple numbers to see how this works.

Suppose we have 3 words and our key dimension is $d_k = 2$. Let us say:

$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$$

**Step 1:** Compute $QK^T$:

$$QK^T = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 2 & 1 & 1 \end{bmatrix}$$

**Step 2:** Divide by $\sqrt{d_k} = \sqrt{2} \approx 1.41$:

$$\frac{QK^T}{\sqrt{2}} = \begin{bmatrix} 0.71 & 0 & 0.71 \\ 0.71 & 0.71 & 0 \\ 1.41 & 0.71 & 0.71 \end{bmatrix}$$

**Step 3:** Apply softmax row-wise. For the first row: $\text{softmax}(0.71, 0, 0.71) \approx (0.39, 0.22, 0.39)$.

**Step 4:** Multiply by V: $0.39 \times [1,2] + 0.22 \times [3,4] + 0.39 \times [5,6] = [3.0, 4.0]$.

This tells us that for the first word, the output is a weighted combination of all three Value vectors. The first and third words received equal attention (0.39 each), while the second word received less (0.22). This is exactly what we want.

### Multi-Head Attention

Instead of computing attention once, BERT uses **multi-head attention**. The idea is simple: instead of one librarian searching for information, we have **multiple librarians** — each looking for a different type of relationship.

For example, one attention head might learn to track grammatical relationships (subject-verb), while another might track semantic relationships (synonyms and antonyms).

Each head computes attention independently, and their outputs are concatenated and projected through a linear layer.

### The Transformer Encoder Block

A single Transformer encoder block consists of:

1. **Multi-Head Self-Attention** layer
2. **Add & Layer Normalization** (residual connection)
3. **Feed-Forward Network** (two linear layers with a ReLU activation)
4. **Add & Layer Normalization** (another residual connection)

The residual connections are important — they allow information to flow directly through the network, making it easier to train deep models.


![A single Transformer encoder block with residual connections.](figures/figure_3.png)
*A single Transformer encoder block with residual connections.*



![Self-attention lets every word attend to every other word in the sentence.](figures/figure_4.png)
*Self-attention lets every word attend to every other word in the sentence.*


BERT stacks **multiple** such encoder blocks on top of each other. This depth is what allows BERT to build rich, hierarchical representations of language.

---

## BERT's Architecture — Putting It Together

Now that we understand the Transformer encoder, let us see how BERT uses it.

BERT comes in two sizes:

| Model | Layers | Hidden Size | Attention Heads | Parameters |
|---|---|---|---|---|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### Input Representation

One of the elegant aspects of BERT is how it constructs the input representation. Every input token is represented as the **sum of three embeddings**:

1. **Token Embeddings:** The embedding for the actual token. BERT uses **WordPiece tokenization**, which breaks words into subword units. For example, "playing" might be split into "play" and "##ing".

2. **Segment Embeddings:** BERT can take two sentences as input (for tasks like question answering). The segment embedding tells the model which sentence each token belongs to — Sentence A or Sentence B.

3. **Position Embeddings:** Since the Transformer has no built-in sense of word order (unlike LSTMs), we add position embeddings to tell the model where each token sits in the sequence. BERT uses learned position embeddings (not sinusoidal).

There are also two special tokens:
- **[CLS]** is added at the very beginning of the input. Its final hidden state is used as the aggregate representation for classification tasks.
- **[SEP]** is added between the two sentences and at the end.


![BERT's input is the sum of token, segment, and position embeddings.](figures/figure_5.png)
*BERT's input is the sum of token, segment, and position embeddings.*


The input representation for each token can be written as:


$$
\text{Input}_i = \text{TokenEmbed}(\text{token}_i) + \text{SegmentEmbed}(\text{segment}_i) + \text{PositionEmbed}(i)
$$


Let us plug in some simple numbers to see how this works. Suppose we are using BERT-Base with a hidden size of 768. For the token "dog" at position 2 in Sentence A:

- $\text{TokenEmbed}(\text{"dog"})$ produces a vector of 768 numbers, say $[0.12, -0.34, 0.56, \ldots]$
- $\text{SegmentEmbed}(\text{A})$ produces another 768-dimensional vector, say $[0.01, 0.02, -0.01, \ldots]$
- $\text{PositionEmbed}(2)$ produces yet another 768-dimensional vector, say $[0.05, -0.08, 0.03, \ldots]$

We simply add them element-wise: $[0.12 + 0.01 + 0.05, -0.34 + 0.02 + (-0.08), 0.56 + (-0.01) + 0.03, \ldots] = [0.18, -0.40, 0.58, \ldots]$

This combined vector is what enters the first Transformer encoder layer. This is exactly what we want — a single vector that captures the identity of the token, which sentence it belongs to, and where it sits in the sequence.

---

## Pre-training Objective 1: Masked Language Modeling (MLM)

Now comes the clever part. How do we train BERT to actually understand language?

Remember the fill-in-the-blank game we played at the beginning? BERT's first pre-training objective is essentially the same game, at scale.

Here is how it works: we take a sentence, **randomly mask 15% of the tokens**, and ask BERT to predict the original tokens at those masked positions.

For example:

- **Input:** "The cat [MASK] on the mat"
- **Target:** "sat"

This is called **Masked Language Modeling (MLM)**, and it forces BERT to learn deep bidirectional representations. To predict the masked word, the model *must* look at both the left context ("The cat") and the right context ("on the mat").

You might be thinking: "But wait — why do we need to mask at all? If BERT sees all words in both directions, can't it just learn to look at every word directly?"

Exactly! If we did not mask any words, each word could trivially "see itself" through the layers, and BERT would not learn anything useful. Masking forces the model to actually reason about the context.

### The 80-10-10 Rule

There is one subtlety. During fine-tuning, there are no [MASK] tokens in the input — real sentences do not have blanks in them. If BERT is only trained to predict [MASK] tokens, it might struggle when it sees real text.

To solve this, BERT uses a clever strategy for the 15% of tokens selected for prediction:

- **80%** of the time: replace the token with [MASK]
- **10%** of the time: replace the token with a random word
- **10%** of the time: keep the original word unchanged

This way, the model cannot simply learn that [MASK] = "something is hidden here." It has to maintain a good representation for every token, because any token might be the one it needs to predict.


![Masked Language Modeling: predict the hidden word from its bidirectional context.](figures/figure_6.png)
*Masked Language Modeling: predict the hidden word from its bidirectional context.*


The loss function for MLM is the standard cross-entropy loss over the masked positions:


$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(w_i \mid \mathbf{w}_{\text{context}}; \theta)$$

Let us plug in some simple numbers. Suppose we have a sentence with 10 tokens, and 2 of them are masked. For the first masked token, our model predicts probabilities over a small vocabulary of 5 words:

| Word | Predicted Probability |
|---|---|
| sat | 0.72 |
| slept | 0.15 |
| lay | 0.08 |
| jumped | 0.03 |
| ran | 0.02 |

If the true word is "sat," the loss for this token is:

$$\mathcal{L}_1 = -\log(0.72) = 0.33$$

If the second masked token has the true word predicted with probability 0.85:

$$\mathcal{L}_2 = -\log(0.85) = 0.16$$

The total MLM loss for this sentence is: $\mathcal{L}_{\text{MLM}} = 0.33 + 0.16 = 0.49$

A lower loss means the model is getting better at predicting the masked words. This makes sense because a perfect model would predict probability 1.0 for each correct word, giving a loss of $-\log(1.0) = 0$.

---

## Pre-training Objective 2: Next Sentence Prediction (NSP)

BERT does not just understand individual sentences. It also learns to understand the **relationship between two sentences**.

Many NLP tasks require understanding sentence pairs — for example, "Does sentence B answer the question in sentence A?" or "Does sentence B follow logically from sentence A?"

To learn this, BERT uses a second pre-training objective called **Next Sentence Prediction (NSP)**.

Here is how it works:

- Take two sentences, A and B
- **50% of the time:** B is the actual next sentence that follows A in the original text (label: **IsNext**)
- **50% of the time:** B is a random sentence from the corpus (label: **NotNext**)

The model uses the hidden state of the **[CLS]** token — which sits at the very beginning of the input — to make this binary prediction.


![Next Sentence Prediction teaches BERT to understand inter-sentence relationships.](figures/figure_7.png)
*Next Sentence Prediction teaches BERT to understand inter-sentence relationships.*


The loss function for NSP is binary cross-entropy:


$$\mathcal{L}_{\text{NSP}} = -\left[y \log(p) + (1-y) \log(1-p)\right]$$

Let us plug in some simple numbers. Suppose for a given pair, the true label is $y = 1$ (IsNext) and the model predicts $p = 0.85$:

$$\mathcal{L}_{\text{NSP}} = -[1 \times \log(0.85) + 0 \times \log(0.15)] = -\log(0.85) = 0.16$$

Now suppose the true label is $y = 0$ (NotNext) and the model predicts $p = 0.20$ (meaning it correctly thinks this is NOT the next sentence):

$$\mathcal{L}_{\text{NSP}} = -[0 \times \log(0.20) + 1 \times \log(0.80)] = -\log(0.80) = 0.22$$

Both losses are small, which means the model is doing a good job. This is exactly what we want.

### Total Pre-training Loss

The total pre-training loss is simply the sum of both objectives:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

BERT was pre-trained on a massive corpus: **BooksCorpus** (800 million words) and **English Wikipedia** (2,500 million words). Training took 4 days on 4 Cloud TPUs for BERT-Base and 16 Cloud TPUs for BERT-Large.

---

## Fine-tuning BERT for Downstream Tasks

This is where BERT truly shines. After pre-training, BERT can be fine-tuned for almost any NLP task with **minimal modifications**.

The key insight is this: the pre-trained BERT already "understands" language deeply — it knows about grammar, semantics, relationships between words, and even some world knowledge. All we need to do is add a thin, task-specific layer on top and fine-tune the entire model on a small labeled dataset.

Let us look at how BERT is fine-tuned for three common tasks:

### 1. Text Classification (e.g., Sentiment Analysis)

For classification tasks, we take the hidden state of the **[CLS]** token from the final layer and pass it through a single linear layer followed by softmax.

The [CLS] token acts as an aggregate representation of the entire input — because of self-attention, it has "attended to" every other token in the sentence.

### 2. Named Entity Recognition (NER)

For token-level tasks like NER (identifying names, locations, organizations in text), we use the **output representation of each token** and pass each one through a classification layer.

Each token independently predicts its entity type: Person, Location, Organization, or None.

### 3. Question Answering

For question answering (e.g., the SQuAD benchmark), we input the question and the passage as two segments: **[CLS] Question [SEP] Passage [SEP]**. The model predicts the **start position** and **end position** of the answer span within the passage.


![BERT adapts to classification, token labeling, and question answering with minimal changes.](figures/figure_8.png)
*BERT adapts to classification, token labeling, and question answering with minimal changes.*


The beauty of this approach is that the same pre-trained BERT model is used for all these tasks. The only thing that changes is the thin output layer on top. This is the **pre-train and fine-tune** paradigm that BERT popularized, and it has become the default approach in modern NLP.

---

## Practical Implementation: Sentiment Classification with BERT

Enough theory, let us look at some practical implementation now.

We will fine-tune BERT for **movie review sentiment classification** using the HuggingFace Transformers library. This is one of the most common NLP tasks and a great way to see BERT in action.

First, let us load the BERT tokenizer and model:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Positive or Negative
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# Model parameters: 109,483,778
```

Let us understand this code. We load a pre-trained BERT-Base model with a classification head that has 2 output labels (positive and negative). The model has approximately 109 million parameters — most of which were learned during pre-training.

Now, let us see how BERT tokenizes a movie review:

```python
text = "This movie was absolutely fantastic! The acting was superb."

# Tokenize the input
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
)

# Let us see the tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print("Tokens:", tokens)
# ['[CLS]', 'this', 'movie', 'was', 'absolutely', 'fantastic', '!',
#  'the', 'acting', 'was', 'superb', '.', '[SEP]']

print("Token IDs:", inputs['input_ids'][0].tolist())
# [101, 2023, 3185, 2001, 7078, 10392, 999, 1996, 3772, 2001, 25408, 1012, 102]
```

Notice how the tokenizer automatically adds the [CLS] token at the beginning and the [SEP] token at the end. Each token is converted to a numeric ID that maps to a row in the embedding matrix.

Now let us write a simple training loop to fine-tune BERT:

```python
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# --- Prepare a small training batch (for illustration) ---
texts = [
    "This movie was absolutely fantastic!",
    "Terrible film. Waste of time.",
    "I loved every minute of this masterpiece.",
    "The worst movie I have ever seen."
]
labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Tokenize all texts
encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
labels_tensor = torch.tensor(labels)

# --- Fine-tuning ---
optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    outputs = model(
        input_ids=encoded['input_ids'],
        attention_mask=encoded['attention_mask'],
        labels=labels_tensor
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
# Epoch 1, Loss: 0.7124
# Epoch 2, Loss: 0.6531
# Epoch 3, Loss: 0.5847
```

We can see that the loss is clearly decreasing with each epoch, which means our model is learning. In practice, you would train on thousands of examples for multiple epochs, but the core loop remains exactly the same.

Finally, let us test our model on a new review:

```python
model.eval()

test_text = "An incredible journey that moved me to tears. Highly recommend!"
test_inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**test_inputs)
    prediction = torch.softmax(outputs.logits, dim=1)
    label = "Positive" if prediction[0][1] > 0.5 else "Negative"

print(f"Review: {test_text}")
print(f"Prediction: {label} (confidence: {prediction[0][1].item():.2f})")
# Review: An incredible journey that moved me to tears. Highly recommend!
# Prediction: Positive (confidence: 0.73)
```

Not bad right? With just 4 training examples and 3 epochs, BERT is already leaning in the right direction. With a full dataset like IMDB (50,000 reviews), BERT achieves over 93% accuracy on sentiment classification.

---

## Why BERT Was a Breakthrough

When BERT was released in October 2018, it achieved **state-of-the-art results on 11 different NLP benchmarks simultaneously**, including:

- **GLUE benchmark** (General Language Understanding Evaluation)
- **SQuAD v1.1 and v2.0** (Question Answering)
- **SWAG** (Sentence-pair inference)

BERT's contribution goes beyond just the numbers. Here are the three key reasons why BERT changed the field:

**1. Bidirectional pre-training works.** BERT proved that training a model to use both left and right context simultaneously produces far richer representations than unidirectional approaches. This seems obvious in hindsight, but the technical challenge of making it work was significant.

**2. The pre-train and fine-tune paradigm.** BERT showed that you can train one large model on a general language understanding task, and then fine-tune it for almost any specific task with minimal effort. This democratized NLP — researchers with limited compute could now achieve competitive results by fine-tuning a pre-trained model.

**3. Transfer learning for NLP.** BERT did for NLP what ImageNet did for computer vision. It showed that a single pre-trained model can serve as the foundation for countless downstream tasks. This spawned an entire ecosystem of models built on the same idea.

BERT also inspired a family of successors, each improving on different aspects:

- **RoBERTa** (Liu et al., 2019): Removed NSP, trained with more data and longer
- **ALBERT** (Lan et al., 2019): Parameter sharing for efficiency
- **DistilBERT** (Sanh et al., 2019): Knowledge distillation for a smaller, faster model
- **ELECTRA** (Clark et al., 2020): Replaced MLM with a more sample-efficient replaced token detection objective

---

## References

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)
- Vaswani et al., "Attention Is All You Need" (2017)
- Peters et al., "Deep Contextualized Word Representations" (ELMo, 2018)
- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (Word2Vec, 2013)
- Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
- HuggingFace Transformers: https://huggingface.co/transformers

That's it!
