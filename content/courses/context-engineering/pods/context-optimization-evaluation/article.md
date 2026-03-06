# Context Optimization & Evaluation

*How to compress, budget, measure, and continuously improve the information you feed your LLM*

---

Let us start with a simple analogy. Imagine you are a chef preparing a lunchbox for a VIP client. Your kitchen has hundreds of ingredients — fresh vegetables, sauces, spices, proteins, garnishes. But the lunchbox has a fixed size. You cannot fit everything. So you face two fundamental questions: **which ingredients make the cut**, and **how do you know the meal actually tastes good?**

This is precisely the challenge we face after assembling context for a Large Language Model. In the earlier parts of this course, we learned how to retrieve relevant documents, design effective prompts, and build memory architectures. But assembling context is only half the battle. The other half is **optimizing** that context to fit within tight token budgets, and then **evaluating** whether the context actually helps the LLM produce better answers.

Without optimization, you waste tokens on redundant or low-value information. Without evaluation, you are flying blind — you have no idea whether your context engineering decisions are actually working.

So here is the question we are asking in this article: **How do we make the most of every token in our context window, and how do we rigorously measure whether our context is doing its job?**

This brings us to Context Optimization and Evaluation.

---

## The Token Budget Problem

Let us begin with the most fundamental constraint: the context window is finite, and every token you spend on one thing is a token you cannot spend on something else.

Think of it like packing a suitcase for a two-week trip. The airline gives you a 23 kg weight limit. You have clothes, toiletries, electronics, books, snacks, and gifts to bring. Everything has weight. If you pack three heavy novels, you might have to leave behind a jacket. The total weight of everything you pack must stay under the limit.

For an LLM, the "weight limit" is the context window size, measured in tokens. A typical modern model gives you 128K tokens to work with. That sounds like a lot, but it fills up fast when you start adding system prompts, conversation history, retrieved documents, tool outputs, and the user's query — all while reserving space for the model's output.

We can express this as a budget constraint:


$$
T_{\text{total
$$
 = T_{\text{system}} + T_{\text{history}} + T_{\text{RAG}} + T_{\text{tools}} + T_{\text{user}} + T_{\text{reserved}}}}

where each $T$ represents the token allocation for that component, and the sum must not exceed the model's context window size $W$.

Let us plug in some simple numbers. Suppose we have a 128K-token model. We allocate: system prompt = 2K tokens, conversation history = 15K, RAG documents = 50K, tool results = 10K, user message = 1K, and reserved output = 35K. The total is:

$$T_{\text{total}} = 2K + 15K + 50K + 10K + 1K + 35K = 113K$$

This fits within our 128K window with 15K tokens to spare. But now suppose our retrieval system returns 80K tokens of documents instead of 50K. Suddenly we need 143K tokens — 15K over budget. Something has to give.

This is where **priority scoring** comes in. Not all tokens contribute equally to the answer. A single sentence from a highly relevant document might be worth more than an entire page of tangentially related material. The optimization question is: given a fixed budget, which tokens deliver the most value?


![Token budget allocation across context window components](figures/figure_1.png)
*Token budget allocation across context window components*


This brings us to our first optimization technique: prompt compression.

---

## Prompt Compression

One of the most effective ways to stay within your token budget is to compress the retrieved context before injecting it into the prompt. Raw documents are verbose — they contain filler words, redundant phrasing, tangential details, and formatting artifacts that consume tokens without adding information.

The goal of prompt compression is simple: preserve the meaning while reducing the token count. There are three main strategies.

**Strategy 1: Extractive Compression**

This is the simplest approach. Instead of including the entire document, you select only the most important sentences. Think of it like highlighting a textbook — you do not read the whole page during the exam, just the highlighted parts.

The idea is to score each sentence by how much information it carries, then keep only the top-scoring sentences. A classic approach uses TF-IDF (Term Frequency-Inverse Document Frequency) to identify sentences that contain rare, informative terms.

Here is a simple implementation:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extractive_compress(document: str, query: str, keep_ratio: float = 0.3) -> str:
    """Select the most query-relevant sentences from a document."""
    sentences = document.split('. ')
    if len(sentences) <= 3:
        return document

    # Score each sentence by relevance to the query
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = [query] + sentences
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Cosine similarity between query and each sentence
    query_vec = tfidf_matrix[0:1]
    sentence_vecs = tfidf_matrix[1:]
    scores = cosine_similarity(query_vec, sentence_vecs).flatten()

    # Keep top-k sentences in original order
    k = max(1, int(len(sentences) * keep_ratio))
    top_indices = np.argsort(scores)[-k:]
    top_indices = sorted(top_indices)  # preserve order

    return '. '.join(sentences[i] for i in top_indices) + '.'
```

Let us trace through this code. Suppose we have a 20-sentence document and set `keep_ratio=0.3`. We keep the top $0.3 \times 20 = 6$ sentences. If each sentence averages 20 tokens, we go from 400 tokens down to 120 tokens — a 70% reduction while preserving the most relevant information.

**Strategy 2: Abstractive Compression**

Here, instead of selecting sentences verbatim, we use an LLM to generate a concise summary of the document that preserves the key facts relevant to the query. This typically achieves even higher compression ratios because the summary eliminates redundancy and rephrases information more concisely.

The tradeoff is that abstractive compression requires an additional LLM call, which adds latency and cost. It also risks introducing hallucinations into the summary.

**Strategy 3: Selective Token Dropping**

This is the most aggressive approach. Techniques like LLMLingua (developed by Microsoft Research) analyze each token's contribution to the prompt and drop tokens that carry low information content. The result reads like a telegram — grammatically broken but semantically preserved.

For example, the sentence "The retrieval-augmented generation system uses a vector database to store and retrieve relevant document chunks based on semantic similarity" might be compressed to "retrieval-augmented generation uses vector database store retrieve relevant chunks semantic similarity."

We can quantify compression with a simple ratio:


$$
\text{Compression Ratio} = 1 - \frac{T_{\text{compressed
$$
}{T_{\text{original}}}}}

Let us plug in numbers. If a document has $T_{\text{original}} = 4{,}000$ tokens and after extractive compression we have $T_{\text{compressed}} = 800$ tokens:

$$\text{Compression Ratio} = 1 - \frac{800}{4{,}000} = 1 - 0.2 = 0.8$$

This means we achieved 80% compression — we removed 80% of the tokens while (ideally) retaining the essential information. This is exactly what we want.


![Compression strategies compared: original, extractive, and abstractive](figures/figure_2.png)
*Compression strategies compared: original, extractive, and abstractive*


But compression raises an important question: how much can you compress before you start losing critical information? This is where evaluation becomes essential — and we will get there shortly. First, let us look at how to allocate our token budget dynamically.

---

## Context Window Optimization Strategies

So far we have talked about compressing individual documents. But there is a higher-level optimization problem: how do you allocate tokens across the different components of your context window?

A simple factual question like "What is the capital of France?" needs very little context — maybe a short system prompt and no retrieved documents at all. But a complex research question like "Compare the environmental policies of three countries and recommend a synthesis approach" might need dozens of retrieved documents, a detailed system prompt, and extensive conversation history.

The insight is: **the optimal token allocation depends on the query**. Static budgets waste tokens on simple queries and starve complex ones.

**Strategy 1: Progressive Disclosure**

Load context lazily, not eagerly. Instead of pre-loading every potentially relevant document, start with minimal context and expand only if the model needs more. This is the "start small, grow as needed" approach.

Think of it like a doctor's visit. The doctor does not order every possible test on day one. They start with basic questions, then order specific tests based on what they hear. Each test adds information (context) only when it becomes relevant.

**Strategy 2: Context Layering**

Organize context into priority layers:
- **Layer 1 (Always present):** System prompt, user message — these are non-negotiable
- **Layer 2 (Per-query):** Retrieved documents — fetched based on the specific question
- **Layer 3 (Per-step):** Tool results — generated during multi-step reasoning

Each layer has a different lifecycle and different compression tolerance. System prompts should rarely be compressed. Tool results can often be summarized after use.

**Strategy 3: Dynamic Budget Reallocation**

Here is the most powerful approach. We assign a priority weight to each context component and allocate tokens proportionally, adjusting based on query complexity.

We can express the allocation for component $i$ as:


$$
T_i = \frac{w_i \cdot s_i}{\sum_{j} w_j \cdot s_j} \times (W - T_{\text{reserved
$$
)}}

where $w_i$ is the base priority weight, $s_i$ is a query-dependent scaling factor (how important this component is for this particular query), $W$ is the total context window, and $T_{\text{reserved}}$ is the output reservation.

Let us work through two examples.

**Example 1: Simple factual query** — "What year was BERT published?"

We set scaling factors: $s_{\text{system}}=1.0$, $s_{\text{history}}=0.2$ (minimal history needed), $s_{\text{RAG}}=0.5$ (one document suffices), $s_{\text{tools}}=0.0$ (no tools needed). With base weights $w_{\text{system}}=2, w_{\text{history}}=1, w_{\text{RAG}}=3, w_{\text{tools}}=1$ and available budget $W - T_{\text{reserved}} = 93K$:

Numerator sum: $(2 \times 1.0) + (1 \times 0.2) + (3 \times 0.5) + (1 \times 0.0) = 2 + 0.2 + 1.5 + 0 = 3.7$

$T_{\text{RAG}} = \frac{3 \times 0.5}{3.7} \times 93K = \frac{1.5}{3.7} \times 93K \approx 37.7K$ tokens for RAG. But we only need one short document — so most of this budget goes unused, and the model responds quickly.

**Example 2: Complex research query** — "Compare RAG architectures across three papers"

Now scaling factors shift: $s_{\text{history}}=0.8$, $s_{\text{RAG}}=1.0$, $s_{\text{tools}}=0.5$. Numerator sum: $(2 \times 1.0) + (1 \times 0.8) + (3 \times 1.0) + (1 \times 0.5) = 2 + 0.8 + 3 + 0.5 = 6.3$

$T_{\text{RAG}} = \frac{3 \times 1.0}{6.3} \times 93K = \frac{3}{6.3} \times 93K \approx 44.3K$ tokens for RAG — almost half the total budget, which makes sense for a research-heavy query.


![Dynamic budget reallocation for simple vs complex queries](figures/figure_3.png)
*Dynamic budget reallocation for simple vs complex queries*


Here is a simple implementation:

```python
def allocate_budget(
    window_size: int,
    reserved_output: int,
    weights: dict[str, float],
    scaling: dict[str, float]
) -> dict[str, int]:
    """Dynamically allocate token budget across context components."""
    available = window_size - reserved_output
    weighted_scores = {k: weights[k] * scaling.get(k, 1.0) for k in weights}
    total_score = sum(weighted_scores.values())

    allocation = {}
    for component, ws in weighted_scores.items():
        allocation[component] = int((ws / total_score) * available)

    return allocation

# Example: complex research query
budget = allocate_budget(
    window_size=128_000,
    reserved_output=35_000,
    weights={"system": 2, "history": 1, "rag": 3, "tools": 1},
    scaling={"system": 1.0, "history": 0.8, "rag": 1.0, "tools": 0.5}
)
print(budget)
# {'system': 26571, 'history': 10628, 'rag': 39857, 'tools': 6571}
```

Now we have optimized our context — compressed documents, allocated budgets dynamically. But how do we know it is actually working? This brings us to evaluation.

---

## Automated Evaluation Frameworks

Here is the uncomfortable truth about context engineering: you can spend weeks optimizing your retrieval pipeline, compressing documents, and tuning budget allocations, and still have no idea whether any of it made the answer better. Without evaluation, optimization is just guessing.

The question is: how do you measure context quality at scale, without needing a human to review every single response?

This brings us to **RAGAS** (Retrieval Augmented Generation Assessment) — the most widely adopted framework for evaluating RAG systems. RAGAS defines four metrics that together give you a comprehensive picture of your context quality.

**Metric 1: Context Precision**

This measures: of the context chunks we retrieved, what fraction was actually relevant to answering the question? If you retrieve 10 chunks but only 3 are relevant, your precision is low — you are wasting tokens on noise.

Formally, context precision is calculated by checking each retrieved chunk for relevance and weighting by rank position (higher-ranked chunks matter more):


$$
\text{Context Precision} = \frac{1}{K} \sum_{k=1}^{K} \frac{\text{Number of relevant chunks in top } k}{k
$$
}

This is essentially the average precision metric from information retrieval, applied to context chunks.

Let us plug in a concrete example. Suppose we retrieve 5 chunks, and their relevance labels are: [Relevant, Not Relevant, Relevant, Relevant, Not Relevant].

At position 1: $\frac{1}{1} = 1.0$ (1 relevant out of top 1)
At position 2: skip (not relevant)
At position 3: $\frac{2}{3} = 0.667$ (2 relevant out of top 3)
At position 4: $\frac{3}{4} = 0.75$ (3 relevant out of top 4)
At position 5: skip (not relevant)

Context Precision = $\frac{1}{3}(1.0 + 0.667 + 0.75) = \frac{2.417}{3} = 0.806$

A context precision of 0.806 tells us that our retrieval system is doing a good job — most of the top-ranked chunks are relevant. This is exactly what we want.

**Metric 2: Context Recall**

This is the flip side: of all the information needed to answer the question, how much did our retrieval actually capture? High precision with low recall means you retrieved clean but incomplete context.

Context recall is measured by checking whether each claim in the ground-truth answer can be attributed to the retrieved context.

**Metric 3: Faithfulness**

This measures whether the generated answer is actually grounded in the provided context, or whether the LLM is hallucinating. An answer is faithful if every claim it makes can be traced back to the context.


$$
\text{Faithfulness} = \frac{\text{Number of claims supported by context
$$
{\text{Total number of claims in the answer}}}}

Let us work through an example. Suppose the model generates an answer with 8 distinct claims, and when we check each one against the retrieved context, 6 of them are supported and 2 are not (the model made them up).

$$\text{Faithfulness} = \frac{6}{8} = 0.75$$

A faithfulness of 0.75 means 75% of the answer is grounded in context, and 25% is hallucinated. For a production system, you would want this above 0.90.

**Metric 4: Answer Relevancy**

This measures whether the answer actually addresses the question that was asked. A faithful answer that goes off-topic is not useful. Answer relevancy checks semantic similarity between the question and the generated answer.


![The four RAGAS metrics and how they are computed](figures/figure_4.png)
*The four RAGAS metrics and how they are computed*


Here is how you might compute these metrics in practice:

```python
def compute_faithfulness(answer_claims: list[str], context: str) -> float:
    """Compute faithfulness: fraction of answer claims supported by context."""
    supported = 0
    for claim in answer_claims:
        # In practice, use an LLM to check if context supports each claim
        if claim.lower() in context.lower():  # simplified check
            supported += 1
    return supported / len(answer_claims) if answer_claims else 0.0

def compute_context_precision(relevance_labels: list[bool]) -> float:
    """Compute context precision from ranked relevance labels."""
    precisions = []
    relevant_count = 0
    for k, is_relevant in enumerate(relevance_labels, 1):
        if is_relevant:
            relevant_count += 1
            precisions.append(relevant_count / k)
    return sum(precisions) / len(precisions) if precisions else 0.0

# Example
labels = [True, False, True, True, False]
print(f"Context Precision: {compute_context_precision(labels):.3f}")
# Context Precision: 0.806

claims = ["BERT was published in 2018", "It uses bidirectional attention",
          "It was trained on ImageNet"]  # last claim is hallucinated
context = "BERT was published in 2018 and uses bidirectional attention."
print(f"Faithfulness: {compute_faithfulness(claims, context):.3f}")
# Faithfulness: 0.667
```

These automated metrics give us a quantitative signal. But they have limitations — they rely on LLM-as-judge or heuristic checks, which can miss nuance. For high-stakes applications, we need human evaluation too.

---

## Human Evaluation Protocols

Automated metrics are excellent for catching large regressions and running at scale. But they cannot fully capture the subtlety of context quality. A context might score well on precision and recall but produce an answer that feels awkward, misses the point, or is technically correct but unhelpful.

This is where human evaluation protocols come in. There are two primary approaches.

**Approach 1: Likert Scale Ratings**

Present human evaluators with a question, the retrieved context, and the generated answer. Ask them to rate on a 1-5 scale across multiple dimensions:
- **Relevance** (1-5): Is the retrieved context relevant to the question?
- **Completeness** (1-5): Does the context contain all the information needed?
- **Coherence** (1-5): Is the generated answer well-structured and logical?
- **Faithfulness** (1-5): Does the answer stick to what the context says?

By averaging across evaluators and questions, you get a robust picture of context quality.

**Approach 2: Pairwise Comparison**

Instead of absolute ratings, show evaluators two different context configurations for the same question and ask: "Which one leads to a better answer?" This is often more reliable than Likert scales because humans are better at comparative judgments than absolute ones.

For example, you might compare: Configuration A (top-5 chunks, no compression) vs Configuration B (top-10 chunks, extractive compression). The evaluator sees both answers side-by-side and picks the winner.

**Measuring Agreement: Cohen's Kappa**

When multiple annotators evaluate the same examples, we need to measure whether they agree. Raw agreement percentage is misleading because some agreement happens by chance. Cohen's Kappa adjusts for this:


$$
\kappa = \frac{p_o - p_e}{1 - p_e
$$
}

where $p_o$ is the observed agreement (fraction of cases where annotators agree) and $p_e$ is the expected agreement by chance.

Let us plug in some numbers. Suppose two annotators each label 100 context-answer pairs as "Good" or "Bad." Their labels look like this:

|  | Annotator B: Good | Annotator B: Bad |
|--|-------------------|------------------|
| Annotator A: Good | 40 | 10 |
| Annotator A: Bad | 15 | 35 |

Observed agreement: $p_o = \frac{40 + 35}{100} = 0.75$

To compute expected chance agreement: Annotator A says "Good" 50 times, Annotator B says "Good" 55 times. So the chance of both saying "Good" is $\frac{50}{100} \times \frac{55}{100} = 0.275$. Similarly, both saying "Bad": $\frac{50}{100} \times \frac{45}{100} = 0.225$. So $p_e = 0.275 + 0.225 = 0.5$.

$$\kappa = \frac{0.75 - 0.5}{1 - 0.5} = \frac{0.25}{0.5} = 0.5$$

A Kappa of 0.5 indicates moderate agreement. Generally, $\kappa > 0.6$ is considered substantial agreement, and $\kappa > 0.8$ is near-perfect. If your Kappa is below 0.4, your evaluation rubric needs refinement — the annotators are interpreting the criteria differently.


![Human evaluation workflow from sampling to agreement calculation](figures/figure_5.png)
*Human evaluation workflow from sampling to agreement calculation*


The combination of automated RAGAS metrics for broad coverage and human evaluation for nuanced assessment gives you a robust evaluation system. But evaluation is only useful if you act on the results. This brings us to experimentation.

---

## A/B Testing Context Strategies

Once you have evaluation metrics — both automated and human — you can treat context engineering like a product optimization problem. You have a baseline configuration, you hypothesize that a change will improve quality, and you run a controlled experiment to test it.

The process is straightforward:

1. **Hypothesis**: "Adding 3 few-shot examples to the system prompt will increase answer accuracy by at least 5 percentage points."
2. **Variant design**: Group A gets the current system prompt (control). Group B gets the system prompt with few-shot examples (treatment).
3. **Metric selection**: Primary metric is answer accuracy (judged by automated evaluation). Secondary metrics are faithfulness and latency.
4. **Traffic splitting**: Randomly assign 50% of queries to each group.
5. **Statistical testing**: After collecting enough data, test whether the difference is statistically significant.

For binary outcomes (correct/incorrect), we use a two-proportion z-test:


$$
z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)
$$
}}

where $\hat{p}_A$ and $\hat{p}_B$ are the observed accuracy rates, $\hat{p}$ is the pooled accuracy, and $n_A, n_B$ are sample sizes.

Let us work through a complete example. Suppose we run an A/B test for 1,000 queries (500 per group):

- **Group A** (no few-shot): 300 correct out of 500, so $\hat{p}_A = 0.60$
- **Group B** (with few-shot): 335 correct out of 500, so $\hat{p}_B = 0.67$

Pooled accuracy: $\hat{p} = \frac{300 + 335}{500 + 500} = \frac{635}{1000} = 0.635$

$$z = \frac{0.67 - 0.60}{\sqrt{0.635 \times 0.365 \times \left(\frac{1}{500} + \frac{1}{500}\right)}} = \frac{0.07}{\sqrt{0.635 \times 0.365 \times 0.004}} = \frac{0.07}{\sqrt{0.000927}} = \frac{0.07}{0.0305} = 2.30$$

A z-score of 2.30 corresponds to a p-value of approximately 0.021, which is below the standard significance threshold of 0.05. We can reject the null hypothesis — the few-shot examples do improve accuracy, and the improvement is statistically significant. Not bad right?


![A/B test comparing control vs few-shot context strategy with confidence intervals](figures/figure_6.png)
*A/B test comparing control vs few-shot context strategy with confidence intervals*


Here is a simple implementation:

```python
import math

def ab_test_significance(
    successes_a: int, total_a: int,
    successes_b: int, total_b: int,
    alpha: float = 0.05
) -> dict:
    """Run a two-proportion z-test for A/B test significance."""
    p_a = successes_a / total_a
    p_b = successes_b / total_b
    p_pooled = (successes_a + successes_b) / (total_a + total_b)

    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/total_a + 1/total_b))
    z_score = (p_b - p_a) / se

    # Approximate two-tailed p-value using normal CDF
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))

    return {
        "control_rate": p_a,
        "treatment_rate": p_b,
        "lift": p_b - p_a,
        "z_score": z_score,
        "p_value": p_value,
        "significant": p_value < alpha
    }

result = ab_test_significance(300, 500, 335, 500)
print(f"Control: {result['control_rate']:.1%}")
print(f"Treatment: {result['treatment_rate']:.1%}")
print(f"Lift: +{result['lift']:.1%}")
print(f"Z-score: {result['z_score']:.2f}, p-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
# Control: 60.0%, Treatment: 67.0%, Lift: +7.0%
# Z-score: 2.30, p-value: 0.0214, Significant: True
```

A/B testing turns context engineering from art into science. But experiments happen periodically — what about monitoring context quality continuously in production?

---

## Production Monitoring for Context Quality

Once your context engineering system is deployed, the work is not over. Context quality can degrade over time for many reasons: the document corpus changes, user query patterns shift, embedding model drift, or upstream data quality drops. You need continuous monitoring to catch these regressions before they impact users.

There are four key metrics to track in production.

**Metric 1: Token Utilization Rate**

This measures what fraction of your allocated budget you are actually using. Consistently low utilization means your budget is too generous (wasting money on large context windows). Consistently high utilization means you are at risk of truncating important context.

**Metric 2: Context Relevance Score**

Run the RAGAS context precision metric on a sample of production queries. Track this score over time. A declining relevance score often signals that your retrieval index is stale or that user queries have shifted to topics not well-covered by your corpus.

**Metric 3: Retrieval Latency**

How long does it take to retrieve and assemble context? Spikes in latency could indicate infrastructure issues, or that your retrieval queries have become more complex.

**Metric 4: Answer Quality Score**

Use automated evaluation (faithfulness + answer relevancy) on sampled production responses. This is your bottom-line metric — everything else feeds into this.

To detect gradual degradation (as opposed to sudden crashes), use an **Exponential Moving Average (EMA)** to smooth noisy daily metrics:


$$
\text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1
$$
}

where $x_t$ is the metric value on day $t$ and $\alpha$ is the smoothing factor (typically 0.1 to 0.3). A smaller $\alpha$ gives more weight to history (smoother line), while a larger $\alpha$ is more responsive to recent changes.

Let us work through a numerical example. Suppose we are tracking daily context relevance scores, and our EMA starts at 0.85. Over the next 5 days, the raw scores are: [0.83, 0.81, 0.79, 0.82, 0.78]. With $\alpha = 0.2$:

Day 1: $\text{EMA}_1 = 0.2 \times 0.83 + 0.8 \times 0.85 = 0.166 + 0.680 = 0.846$
Day 2: $\text{EMA}_2 = 0.2 \times 0.81 + 0.8 \times 0.846 = 0.162 + 0.677 = 0.839$
Day 3: $\text{EMA}_3 = 0.2 \times 0.79 + 0.8 \times 0.839 = 0.158 + 0.671 = 0.829$
Day 4: $\text{EMA}_4 = 0.2 \times 0.82 + 0.8 \times 0.829 = 0.164 + 0.663 = 0.827$
Day 5: $\text{EMA}_5 = 0.2 \times 0.78 + 0.8 \times 0.827 = 0.156 + 0.662 = 0.818$

The EMA has dropped from 0.85 to 0.818 — a clear downward trend that the raw noisy scores make harder to spot. If we set an alert threshold at 0.82, we would trigger an alert on Day 5.


![Production monitoring dashboard with four key context quality metrics](figures/figure_7.png)
*Production monitoring dashboard with four key context quality metrics*


Here is a monitoring class that implements this:

```python
class ContextQualityMonitor:
    def __init__(self, alpha: float = 0.2, alert_threshold: float = 0.82):
        self.alpha = alpha
        self.alert_threshold = alert_threshold
        self.ema = None
        self.history = []

    def record(self, relevance_score: float) -> dict:
        """Record a new relevance score and check for alerts."""
        self.history.append(relevance_score)
        if self.ema is None:
            self.ema = relevance_score
        else:
            self.ema = self.alpha * relevance_score + (1 - self.alpha) * self.ema

        alert = self.ema < self.alert_threshold
        return {
            "raw_score": relevance_score,
            "ema": round(self.ema, 4),
            "alert": alert,
            "message": f"Context relevance EMA dropped to {self.ema:.3f}" if alert else "OK"
        }

# Example usage
monitor = ContextQualityMonitor(alpha=0.2, alert_threshold=0.82)
daily_scores = [0.85, 0.83, 0.81, 0.79, 0.82, 0.78, 0.76]
for day, score in enumerate(daily_scores, 1):
    result = monitor.record(score)
    status = "ALERT" if result["alert"] else "OK"
    print(f"Day {day}: raw={score:.2f}, EMA={result['ema']:.3f} [{status}]")
```

When alerts fire, the response should be systematic: first, check if the document corpus has changed. Second, verify the embedding model is functioning correctly. Third, review a sample of recent queries to see if the distribution has shifted. Only after diagnosis should you take action — reindexing, retraining retrieval models, or adjusting compression parameters.

---

## The Optimization-Evaluation Cycle

Let us step back and see the big picture. Everything we have covered in this article forms a continuous cycle:

1. **Measure** your current context quality using RAGAS metrics and human evaluation
2. **Identify** the weakest link — is it precision? faithfulness? token waste?
3. **Optimize** — apply compression, adjust budgets, change retrieval parameters
4. **Experiment** — A/B test the optimization against the baseline
5. **Deploy** the winning variant and monitor in production
6. **Repeat** when metrics drift


![The continuous cycle of measuring, optimizing, testing, and monitoring context quality](figures/figure_8.png)
*The continuous cycle of measuring, optimizing, testing, and monitoring context quality*


This cycle is what separates production-grade context engineering from one-shot prompt hacking. The difference between a system that stays good and a system that gets better is whether you close this loop.


![Context quality improving across successive optimization iterations](figures/figure_9.png)
*Context quality improving across successive optimization iterations*


That is it! Context optimization and evaluation is where context engineering becomes a true engineering discipline — with measurements, experiments, and continuous improvement. The tools are straightforward: compression to make the most of your tokens, RAGAS to measure quality, human evaluation for nuance, A/B testing for rigorous experimentation, and production monitoring to keep things running smoothly.

Thanks!
