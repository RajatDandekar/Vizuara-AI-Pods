# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

*How one paper changed the way machines understand language — from reading left-to-right to reading the whole sentence at once*

Vizuara AI

---

## The Fill-in-the-Blank Game

Let us start with a simple game that all of us have played in school — fill in the blank.

**"The cat sat on the ___"**

Easy, right? You probably guessed "mat" or "floor" or "chair" almost instantly.

But here is the interesting part: **how** did you guess so quickly?

You read the words "The cat sat on the" and your brain immediately understood the context from **all** the surrounding words. You knew it was about a cat, you knew it was sitting, and you knew it had to be sitting **on** something — a surface.

Now imagine a very strange rule: you are ONLY allowed to read words from left to right, and you must guess the next word **without ever looking ahead.** You read "The" — okay, could be anything. You read "The cat" — now it is about a cat. You read "The cat sat" — the cat is sitting. You read "The cat sat on" — sitting on something. You read "The cat sat on the" — and now you must guess the next word.

This is still manageable. But now consider a harder example:

**"I went to the ___ to deposit my money"**

If you can only read left-to-right up to the blank, you see: "I went to the ___." The blank could be "bank," "store," "park," "hospital" — almost anything!

But if you can also read what comes **after** the blank — "to deposit my money" — suddenly it is obvious. The answer is "bank," and specifically a **financial** bank, not a riverbank.

This is the core insight behind BERT. Before BERT, language models could only read in one direction. BERT was the first model to say: **"Why not read the entire sentence at once?"**


![Unidirectional vs Bidirectional reading comparison](figures/figure_1.png)
*Before BERT, models could only read left-to-right and were blocked from seeing future context. BERT reads the entire sentence bidirectionally.*


---

## Why Reading Left-to-Right Is Not Enough

Let us look at another example to really drive this home.

Consider the word **"bank"** in these two sentences:

1. "I sat on the **bank** of the river and watched the sunset."
2. "I went to the **bank** to withdraw some cash."

The word "bank" means completely different things in these two sentences. But here is the crucial point — the meaning of "bank" depends on the words that come **after** it.

In the first sentence, the words "of the river" tell us it is a riverbank. In the second sentence, the words "to withdraw some cash" tell us it is a financial institution.

If you can only read left-to-right, by the time you reach "bank," you do not have access to the disambiguating words that come after it. You are stuck guessing.

This is exactly the problem that existed before BERT.

**GPT** (2018, by OpenAI) was a powerful language model, but it read text strictly left-to-right. Each word could only attend to the words that came before it. This is called a **unidirectional** or **autoregressive** model.

**ELMo** (2018, by Allen AI) tried to fix this by training two separate models — one reading left-to-right, and one reading right-to-left — and then concatenating their outputs. But the two directions were never deeply fused. Each direction was computed independently, and they were only combined at the very end.

**BERT** (2018, by Google) took a fundamentally different approach. Instead of reading in one direction (or two shallow directions), BERT allowed **every word to attend to every other word in the sentence simultaneously.** The context flows in all directions, at every layer, from the very beginning.


![GPT vs ELMo vs BERT sentence processing comparison](figures/figure_2.png)
*Comparing how GPT, ELMo, and BERT process a sentence. GPT reads left-to-right only. ELMo uses two shallow directions concatenated at the end. BERT uses deep bidirectional attention where every word attends to every other word at every layer.*


This was a game-changer. But how did BERT achieve this? What architecture makes it possible to read everything at once?

This brings us to the Transformer.

---

## The Architecture: Transformer Encoder Stack

BERT uses the **encoder** part of the Transformer architecture. If you have read about the original Transformer from the "Attention Is All You Need" paper (Vaswani et al., 2017), you will know that it has two parts: an encoder and a decoder. BERT only uses the encoder.

Why? Because the decoder is designed for generation — predicting one token at a time, left-to-right. But BERT does not generate text. BERT **understands** text. It reads the entire input and produces a rich representation of every word in context.

Each Transformer encoder layer consists of two main components:

1. **Multi-Head Self-Attention** — every word looks at every other word
2. **Feed-Forward Network** — processes each word's representation independently

These two components are stacked on top of each other, with residual connections and layer normalization between them.

### Self-Attention: The Heart of BERT

Let us understand self-attention with a concrete example.

Consider the sentence: **"The cat sat on the mat"**

When processing the word "sat," the self-attention mechanism asks: **"Which other words in this sentence are most relevant to understanding 'sat'?"**

The answer: "cat" is very relevant (who is sitting?), and "mat" is relevant (where is the cat sitting?). Words like "the" and "on" are less important.

Self-attention computes this relevance using three vectors for each word:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What information do I contain?"
- **V (Value):** "What information should I pass along?"

The attention score between two words is computed by taking the dot product of the Query of one word with the Key of another word:


$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Here, $Q$ is the matrix of query vectors for all words, $K$ is the matrix of key vectors, $V$ is the matrix of value vectors, and $d_k$ is the dimension of the key vectors. The division by $\sqrt{d_k}$ prevents the dot products from becoming too large.

Let us plug in some simple numbers to see how this works.

Suppose we have just 3 words and $d_k = 2$. Let our Q, K, and V matrices be:

$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

$$K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}$$

$$V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

**Step 1:** Compute $QK^T$:

$$QK^T = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 2 & 1 & 1 \end{bmatrix}$$

**Step 2:** Divide by $\sqrt{d_k} = \sqrt{2} \approx 1.41$:

$$\frac{QK^T}{\sqrt{2}} = \begin{bmatrix} 0.71 & 0 & 0.71 \\ 0.71 & 0.71 & 0 \\ 1.41 & 0.71 & 0.71 \end{bmatrix}$$

**Step 3:** Apply softmax to each row. For the first row $[0.71, 0, 0.71]$:

The denominator is $e^{0.71} + e^{0} + e^{0.71} = 2.03 + 1 + 2.03 = 5.06$

$$\text{softmax} = \left[\frac{2.03}{5.06},\ \frac{1.00}{5.06},\ \frac{2.03}{5.06}\right] \approx [0.39,\ 0.22,\ 0.39]$$

This tells us that word 1 pays **equal attention** (0.39) to words 1 and 3, and less attention (0.22) to word 2. This is exactly what we want — the attention weights tell us how much each word should influence the representation of the current word.

**Step 4:** Multiply by V to get the final output:

$$\text{Output}_{\text{word 1}} = 0.39 \times [1,0] + 0.22 \times [0,1]$$
$$+ \ 0.39 \times [1,1] = [0.78,\ 0.61]$$

The output for word 1 is a weighted combination of all the value vectors, where the weights are determined by relevance. This is exactly what we want.

### Multi-Head Attention

Instead of computing attention once, BERT computes it **multiple times in parallel** — these are called attention "heads." Each head can learn to focus on different types of relationships:

- One head might learn **syntactic** relationships (subject-verb agreement)
- Another head might learn **semantic** relationships (word meaning)
- Another might learn **positional** relationships (nearby words)

BERT-Base uses **12 attention heads**, and BERT-Large uses **16 attention heads.**

The outputs of all heads are concatenated and passed through a linear layer.

### BERT Model Sizes

BERT comes in two sizes:

| | BERT-Base | BERT-Large |
|---|---|---|
| Layers | 12 | 24 |
| Hidden Size | 768 | 1024 |
| Attention Heads | 12 | 16 |
| Parameters | 110M | 340M |


![BERT-Base architecture](figures/figure_3.png)
*The BERT-Base architecture: 12 stacked Transformer encoder layers, each containing Multi-Head Self-Attention and a Feed-Forward Network, with 110M total parameters.*


Now we understand the engine that powers BERT. But how does text actually enter this engine?

---

## Input Representation: How BERT Reads Text

Before we can feed text into BERT, we need to convert it into numbers. BERT's input representation is the sum of **three** different embeddings:

**1. Token Embeddings** — What is the word?

BERT uses **WordPiece tokenization.** This is a clever trick: instead of having one entry per word in the vocabulary, frequent words are kept whole, and rare words are broken into sub-word pieces.

For example:
- "I" → `I`
- "love" → `love`
- "unbelievable" → `un`, `##believ`, `##able`

The `##` prefix means "this piece is a continuation of the previous word." This allows BERT to handle any word — even words it has never seen before — by breaking them into known pieces.

**2. Segment Embeddings** — Which sentence does this word belong to?

BERT often processes **pairs** of sentences (for tasks like question answering or natural language inference). The segment embedding tells the model whether a token belongs to Sentence A or Sentence B.

**3. Position Embeddings** — Where is this word in the sequence?

Since the Transformer has no built-in notion of word order (unlike RNNs), we need to explicitly tell it the position of each word. BERT uses learned position embeddings for positions 0 through 511.

BERT also uses two special tokens:
- **[CLS]** — placed at the very beginning. Its output representation is used for classification tasks.
- **[SEP]** — placed between sentences and at the end, to mark sentence boundaries.

The final input embedding for each token is:

$$\text{Input} = \text{Token Embedding} + \text{Segment Embedding} + \text{Position Embedding}$$


![BERT input representation](figures/figure_4.png)
*BERT's input representation is the sum of three embeddings: Token Embeddings (the word itself), Segment Embeddings (which sentence it belongs to), and Position Embeddings (where it appears in the sequence).*


Now we know how text enters the model and how the model processes it. But the critical question remains: **how does BERT actually learn?**

This brings us to BERT's two pre-training objectives.

---

## Pre-training Objective 1: Masked Language Modeling (MLM)

Remember the fill-in-the-blank game from the beginning? That is exactly what Masked Language Modeling is.

During pre-training, BERT randomly selects **15% of the tokens** in the input and masks them. The model then tries to predict the original token at each masked position.

For example:

**Original:** "The cat sat on the mat"

**Masked:** "The [MASK] sat on the mat"

BERT must predict that [MASK] = "cat" by looking at ALL the surrounding words — "The," "sat," "on," "the," "mat" — from both directions simultaneously.

This is the key insight that makes bidirectional training possible. GPT cannot do this because it reads left-to-right — if you mask a word, GPT can only use the words to the left. BERT uses words from **both** sides.

### The 80-10-10 Rule

There is a subtle problem. During pre-training, BERT sees [MASK] tokens. But during fine-tuning, there are no [MASK] tokens — the model sees normal text. This creates a mismatch.

To mitigate this, BERT does not always replace selected tokens with [MASK]. Instead, for the 15% of tokens selected:

- **80% of the time:** Replace with [MASK] (e.g., "The [MASK] sat")
- **10% of the time:** Replace with a **random** word (e.g., "The dog sat")
- **10% of the time:** Keep the **original** word unchanged (e.g., "The cat sat")

This forces the model to maintain a good representation for ALL tokens, not just masked ones, because it never knows which tokens have been tampered with.

### The MLM Loss Function

The loss is computed only over the masked positions. The mathematical representation is:


$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid \mathbf{x}_{\setminus \mathcal{M}}; \theta)$$

Here, $\mathcal{M}$ is the set of masked positions, $x_i$ is the true token at position $i$, $\mathbf{x}_{\setminus \mathcal{M}}$ represents all the non-masked tokens, and $\theta$ are the model parameters. The model computes $P(x_i \mid \mathbf{x}_{\setminus \mathcal{M}})$ by taking the output representation at position $i$ and passing it through a softmax over the entire vocabulary.

Let us plug in some simple numbers to see how this works.

Suppose we have a tiny vocabulary of 5 words: {cat, dog, mat, sat, the}, and the masked position should be "cat."

The model outputs a probability distribution over the vocabulary at the masked position:

$$P(\text{cat}) = 0.72, \quad P(\text{dog}) = 0.15, \quad P(\text{mat}) = 0.08$$
$$P(\text{sat}) = 0.03, \quad P(\text{the}) = 0.02$$

The loss for this single masked position is:

$$\mathcal{L} = -\log(0.72) = -(-0.328) = 0.328$$

If the model had been less confident and assigned $P(\text{cat}) = 0.20$, the loss would be:

$$\mathcal{L} = -\log(0.20) = 1.609$$

The loss is much higher when the model is wrong. This is exactly what we want — the loss penalizes the model for assigning low probability to the correct token.


![Masked Language Modeling training](figures/figure_5.png)
*Masked Language Modeling: BERT processes a sentence with 15% of tokens masked, then predicts the original token at each masked position using a softmax over the entire vocabulary.*


---

## Pre-training Objective 2: Next Sentence Prediction (NSP)

Many important NLP tasks — question answering, natural language inference — require understanding the **relationship between two sentences.** Masked Language Modeling alone does not capture inter-sentence relationships.

To teach BERT about sentence relationships, the authors added a second pre-training objective: **Next Sentence Prediction.**

The task is simple. Given two sentences A and B, predict: **Does B actually follow A in the original text?**

During training, 50% of the time B is the **real** next sentence (label: **IsNext**), and 50% of the time B is a **random** sentence from the corpus (label: **NotNext**).

For example:

**IsNext (positive pair):**
- Sentence A: "I went to the store."
- Sentence B: "I bought some milk."
- Label: **IsNext** (B logically follows A)

**NotNext (negative pair):**
- Sentence A: "I went to the store."
- Sentence B: "Penguins live in Antarctica."
- Label: **NotNext** (B is random, unrelated to A)

The [CLS] token's output representation is fed into a binary classifier that predicts IsNext or NotNext.


![Next Sentence Prediction training](figures/figure_6.png)
*Next Sentence Prediction: BERT learns whether Sentence B is the real next sentence (IsNext) or a random sentence (NotNext) using the [CLS] token output.*


The total pre-training loss is the sum of both objectives:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

BERT was pre-trained on two massive datasets:
- **BooksCorpus** (800M words) — 11,038 unpublished books
- **English Wikipedia** (2,500M words) — text only, no tables or lists

Together, that is **3.3 billion words** of pre-training data.

---

## The Pre-train and Fine-tune Paradigm

Now here is what truly made BERT revolutionary. BERT introduced a **two-stage paradigm** that changed the entire field of NLP:

**Stage 1: Pre-training** — Train the model on massive amounts of unlabeled text using MLM and NSP. This is expensive (days on many TPUs) but done **only once.** Google did this for us and released the pre-trained weights.

**Stage 2: Fine-tuning** — Take the pre-trained BERT, add a small task-specific layer on top, and train on a **small** labeled dataset for your specific task. This is cheap — typically a few hours on a single GPU.

The beauty is that the same pre-trained BERT can be fine-tuned for completely different tasks just by changing the output layer. Have a look at how BERT adapts to four different tasks:

**1. Sentence Classification** (e.g., sentiment analysis): Use the [CLS] output → pass through a linear classifier → positive or negative.

**2. Sentence Pair Classification** (e.g., natural language inference): Feed both sentences with [SEP] in between → use [CLS] output → entailment, contradiction, or neutral.

**3. Question Answering** (e.g., SQuAD): Feed the question and passage → for each token in the passage, predict whether it is the **start** or **end** of the answer span.

**4. Token Classification** (e.g., Named Entity Recognition): For each token, predict its entity label (Person, Organization, Location, etc.).


![BERT fine-tuning for four different downstream tasks](figures/figure_7.png)
*The same pre-trained BERT body can be fine-tuned for four different tasks by simply changing the small task-specific head on top: Sentiment Analysis, NLI, Question Answering, and NER.*


This is exactly what we want. A single model, pre-trained once, can be adapted to dozens of different tasks with minimal effort.

---

## Practical Implementation

Enough theory, let us look at some practical implementation now.

Thanks to the HuggingFace Transformers library, using BERT is remarkably simple. Let us see how to load a pre-trained BERT model and use it for sentiment classification.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# --- Load pre-trained BERT ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

# --- Tokenize input ---
text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

print("Token IDs:", inputs['input_ids'])
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

# --- Get prediction ---
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1)

print(f"Probabilities: Negative={probs[0][0]:.3f}, Positive={probs[0][1]:.3f}")
print(f"Prediction: {'Positive' if prediction.item() == 1 else 'Negative'}")
```

Let us understand this code in detail.

First, we load the **tokenizer** and the **model.** The tokenizer converts human-readable text into token IDs that BERT understands. The model is a pre-trained BERT with a classification head on top.

Next, we tokenize our input text. The tokenizer automatically adds the [CLS] and [SEP] tokens, converts words to WordPiece tokens, and pads the sequence. For our example "This movie was absolutely fantastic!", the tokens would look like: `[CLS] this movie was absolutely fantastic ! [SEP]`.

Finally, we pass the tokenized input through the model. The model outputs logits (raw scores) for each class. We apply softmax to convert these to probabilities, and then take the argmax to get the predicted class.

Now let us look at how to fine-tune BERT on a custom sentiment dataset:

```python
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

# --- Simple Sentiment Dataset ---
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

# --- Training Loop ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Example data
texts = ["I loved this film!", "Terrible movie.", "Best experience ever!", "Awful acting."]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

dataset = SentimentDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Fine-tuning
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
```

Let us understand this code in detail.

We create a simple `SentimentDataset` class that tokenizes text and returns input IDs, attention masks, and labels. The attention mask tells BERT which tokens are real and which are padding — this is important so the model does not attend to padding tokens.

The training loop is straightforward: for each batch, we compute the forward pass, calculate the loss (cross-entropy between predicted and true labels), backpropagate, and update the weights.

Notice the learning rate: **2e-5.** This is a key detail. During fine-tuning, we use a very small learning rate because BERT is already pre-trained — we do not want to destroy the learned representations. We just want to gently nudge them toward our specific task.

Not bad, right? With just 20 lines of training code, we can fine-tune one of the most powerful language understanding models ever built.

---

## The Impact: BERT Changed Everything

When BERT was released in October 2018, it did not just improve the state of the art — it **shattered** it. BERT achieved new state-of-the-art results on **11 NLP benchmarks simultaneously**, including:

- **SQuAD 1.1** (question answering): surpassed human-level performance
- **GLUE benchmark** (8 diverse NLP tasks): +7.7% improvement over the previous best
- **MultiNLI** (natural language inference): 86.7% accuracy

The pre-train → fine-tune paradigm that BERT introduced became the default approach in NLP. Almost overnight, the field shifted from training task-specific models from scratch to fine-tuning pre-trained models.

BERT also spawned an entire family of successors:

- **RoBERTa** (Facebook, 2019) — trained longer, removed NSP, used more data
- **ALBERT** (Google, 2019) — parameter-efficient version with shared weights
- **DistilBERT** (Hugging Face, 2019) — 60% smaller, 97% of performance
- **ELECTRA** (Google, 2020) — replaced masking with a more efficient "replaced token detection" objective
- **DeBERTa** (Microsoft, 2020) — improved attention mechanism with disentangled position and content

Even GPT learned from BERT. While GPT went in a different direction (scaling up autoregressive generation), many of BERT's insights — large-scale pre-training, transfer learning, the Transformer architecture — became foundational to all modern large language models.

Today, BERT and its variants are still widely used for tasks where understanding is more important than generation: search ranking, text classification, named entity recognition, and sentence embeddings. Google Search itself uses BERT to better understand search queries.

This is truly amazing. A single paper from 2018 reshaped the entire landscape of natural language processing.

In the next article, we will look at how the ideas from BERT evolved into even more powerful models — from RoBERTa's training insights to the emergence of GPT-3 and the modern era of large language models. See you next time!

---

## References

- Devlin, J. et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
- Vaswani, A. et al., "Attention Is All You Need" (2017)
- Radford, A. et al., "Improving Language Understanding by Generative Pre-Training" (GPT, 2018)
- Peters, M. et al., "Deep Contextualized Word Representations" (ELMo, 2018)
- Liu, Y. et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
- Lan, Z. et al., "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (2019)
- Clark, K. et al., "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" (2020)
