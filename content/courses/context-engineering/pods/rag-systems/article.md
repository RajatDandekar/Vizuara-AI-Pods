# RAG Systems

*Building retrieval-augmented generation pipelines from chunking to evaluation*

---

Let us start with a familiar problem. Imagine you have a brilliant colleague who graduated five years ago with encyclopedic knowledge of their field. They can answer almost any question about the fundamentals — but ask them about a paper published last month, a company policy updated yesterday, or the contents of a document they have never read, and they are stuck. Their knowledge is frozen at the time they last studied.

This is exactly the problem with large language models. GPT-4, Claude, and other LLMs have vast parametric knowledge — billions of facts encoded in their weights during training. But that knowledge has a cutoff date. Ask about events after training, proprietary documents, or domain-specific data, and the model either confesses ignorance or, worse, confabulates a plausible-sounding but incorrect answer.

Retrieval-Augmented Generation (RAG) solves this by giving the model a "phone" to call. Instead of relying solely on what it memorized during training, the model first retrieves relevant documents from an external knowledge base, then generates its answer grounded in those documents. The knowledge is no longer frozen — it is live, updatable, and verifiable.

In this article, we will build a complete RAG system from first principles — from how documents are split and embedded, to how vectors are indexed and searched, to how retrieved context shapes the final generation, and how we measure whether the whole pipeline actually works.

---

## Why Retrieval Matters

Before we build anything, let us understand why retrieval is so powerful by looking at the mathematics.

A standard LLM computes the probability of an answer $y$ given a question $x$ using only its parametric knowledge:

$$P(y \mid x)$$

This works well for questions within the training distribution — "What is the capital of France?" — but fails for anything outside it.

RAG introduces a retrieval step. Given a question $x$, we first retrieve a set of relevant documents $z$ from an external corpus, then generate the answer conditioned on both the question and the retrieved documents:

$$P(y \mid x) = \sum_{z \in \mathcal{Z}} P(y \mid x, z) \cdot P(z \mid x)$$

where $P(z \mid x)$ is the retrieval model (how likely is document $z$ to be relevant to question $x$?) and $P(y \mid x, z)$ is the generation model (given the question and the retrieved document, what is the answer?).

Let us plug in some simple numbers to see why this matters. Suppose without retrieval, the model has a 30% chance of correctly answering a question about a recent company policy — it might guess based on general patterns, but the specific policy is not in its training data.

Now suppose we retrieve the correct policy document with 90% probability, and given the correct document, the model answers correctly 95% of the time. The overall accuracy becomes:

$$P(\text{correct}) \approx 0.90 \times 0.95 = 0.855$$

We went from 30% to 85.5% accuracy — not by making the model smarter, but by giving it access to the right information. This is exactly what we want.

There is a second, equally important benefit: **verifiability**. When the model generates an answer from retrieval, it can cite its sources. The user can click through to the original document and verify the answer. This is transformative for high-stakes applications like legal research, medical question answering, and financial compliance — domains where "trust me, I'm an LLM" is not acceptable.

![RAG vs Parametric Knowledge](figures/figure_1.png)

---

## The RAG Pipeline End-to-End

A RAG system has two phases: an **offline phase** (prepare the knowledge base) and an **online phase** (answer questions at query time).

**Offline Phase (Indexing):**
1. **Document Loading** — Ingest documents from various sources (PDFs, web pages, databases, APIs)
2. **Chunking** — Split documents into smaller, semantically meaningful pieces
3. **Embedding** — Convert each chunk into a dense vector representation
4. **Indexing** — Store vectors in a vector database for fast retrieval

**Online Phase (Query):**
1. **Query Embedding** — Convert the user's question into the same vector space
2. **Retrieval** — Find the $k$ most similar chunks using vector similarity search
3. **Reranking** (optional) — Reorder retrieved chunks by relevance using a cross-encoder
4. **Generation** — Feed the question + retrieved chunks to the LLM to generate the answer

Each step in this pipeline has design choices that significantly impact the final answer quality. A bad chunking strategy can split critical information across chunk boundaries. A weak embedding model can miss semantic relationships. A naive retrieval strategy can return superficially similar but irrelevant documents. And poor prompt construction can cause the model to ignore the retrieved context entirely.

Let us walk through each step in detail.

![The RAG Pipeline](figures/figure_2.png)

---

## Chunking — Breaking Documents into Retrievable Pieces

Chunking is the first and arguably most underrated step in the RAG pipeline. The quality of your chunks directly determines the quality of your retrieval, which in turn determines the quality of your generation.

The core question is: **how do you split a document so that each piece is self-contained enough to be useful, but small enough to be retrieved precisely?**

There are four main strategies:

### Fixed-Size Chunking

The simplest approach: split the document into chunks of $n$ tokens (or characters) with an overlap of $m$ tokens between consecutive chunks.

```python
def fixed_size_chunk(text, chunk_size=500, overlap=50):
    """Split text into fixed-size chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # Overlap with previous chunk
    return chunks
```

The overlap is critical — without it, information that spans a chunk boundary is lost. If a key fact starts at word 490 and ends at word 510, a 500-word chunk with no overlap will split it across two chunks, and neither chunk alone contains the full fact.

**When to use:** Quick prototyping, documents with uniform structure (logs, data records).

**When to avoid:** Documents with strong semantic structure (research papers, legal contracts) where splitting at arbitrary positions destroys meaning.

### Sentence-Level Chunking

Split at sentence boundaries, then group sentences until a target chunk size is reached.

```python
import re

def sentence_chunk(text, max_chunk_size=500):
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(sentence)
        current_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

This preserves sentence integrity but still has no awareness of document structure.

### Recursive Chunking

Split hierarchically: first try to split at paragraph boundaries. If a paragraph is too large, split at sentence boundaries within it. If a sentence is too large, fall back to fixed-size splitting.

This is the approach used by LangChain's `RecursiveCharacterTextSplitter` — it respects document structure at multiple levels.

### Semantic Chunking

The most sophisticated approach: use embeddings to detect topic boundaries within the document. Compute embeddings for each sentence, then split where the cosine similarity between consecutive sentence embeddings drops below a threshold — indicating a topic shift.

```python
def semantic_chunk(sentences, embeddings, threshold=0.5):
    """Split based on semantic similarity between consecutive sentences."""
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

This produces the most semantically coherent chunks, but it is slower (requires embedding every sentence) and the threshold requires tuning.

![Chunking Strategies Compared](figures/figure_3.png)

**Practical guidance:** Start with recursive chunking at 500-1000 tokens with 10-20% overlap. Move to semantic chunking only if retrieval quality is a bottleneck and you have the compute budget for the embedding step.

---

## Embeddings — Turning Text into Vectors

Once documents are chunked, each chunk must be converted into a dense vector — an embedding — that captures its semantic meaning. Two chunks about the same topic should have similar embeddings, even if they use different words.

The key idea behind embeddings is simple but powerful. Traditional keyword search (like BM25) matches documents to queries based on shared words. If your query says "automobile" but the document says "car," keyword search misses the match. Embedding-based search maps both to nearby points in a high-dimensional space, capturing that "automobile" and "car" are semantically equivalent.

Formally, an embedding model $E$ maps a text string to a vector in $\mathbb{R}^d$:

$$E: \text{string} \rightarrow \mathbb{R}^d$$

where $d$ is typically 384, 768, or 1024 dimensions. The model is trained so that semantically similar texts produce vectors with high cosine similarity:

$$\text{sim}(a, b) = \frac{E(a) \cdot E(b)}{\|E(a)\| \cdot \|E(b)\|}$$

This similarity score ranges from -1 (opposite meaning) to 1 (identical meaning), with 0 indicating no relationship.

Here is a practical implementation:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional

# Embed some example texts
texts = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "Stock prices fell sharply today",
]

embeddings = model.encode(texts)
print(f"Embedding shape: {embeddings.shape}")  # (3, 384)

# Compute pairwise similarities
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        sim = np.dot(embeddings[i], embeddings[j]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
        )
        print(f"  sim('{texts[i][:30]}...', '{texts[j][:30]}...') = {sim:.3f}")
```

You will find that the cat/feline pair has high similarity (around 0.7-0.8) despite different words, while the stock prices sentence has low similarity (around 0.1) to both — exactly what we want.

**Choosing an embedding model:** The MTEB leaderboard (Massive Text Embedding Benchmark) ranks models by retrieval quality. For most RAG applications:
- **all-MiniLM-L6-v2** (384d) — fast, good for prototyping
- **bge-large-en-v1.5** (1024d) — strong retrieval, good balance
- **text-embedding-3-large** (3072d) — OpenAI's model, highest quality but API-dependent

![Semantic Space: Embeddings Cluster by Meaning](figures/figure_4.png)

---

## Vector Databases and Indexing

Once you have embeddings for all your chunks, you need to store them and search them efficiently. This is the job of a vector database.

The naive approach is brute-force: compute the cosine similarity between the query embedding and every chunk embedding, then return the top $k$. This works for small corpora (under 10,000 chunks) but becomes impractical at scale. With 1 million chunks and 768-dimensional embeddings, a single query requires 768 million multiply-add operations.

The solution is **Approximate Nearest Neighbor (ANN)** search — algorithms that trade a small amount of accuracy for dramatic speedups.

### FAISS and HNSW

Facebook AI Similarity Search (FAISS) is the most widely used library for vector search. It supports several indexing strategies:

**Flat Index (brute-force):** Exact search, no approximation. $O(n \cdot d)$ per query.

**IVF (Inverted File Index):** Partition the vector space into $k$ clusters using k-means. At query time, only search the $m$ nearest clusters. Reduces search from $n$ to roughly $n \cdot m / k$ vectors.

**HNSW (Hierarchical Navigable Small World):** Build a multi-layer graph where each node is a vector and edges connect nearby vectors. Query traverses the graph from coarse to fine layers. Achieves sub-linear search time with high recall.

```python
import faiss
import numpy as np

# Create embeddings (simulated)
n_chunks = 100_000
d = 384
chunk_embeddings = np.random.randn(n_chunks, d).astype("float32")
faiss.normalize_L2(chunk_embeddings)  # Normalize for cosine similarity

# Build HNSW index
index = faiss.IndexHNSWFlat(d, 32)  # 32 neighbors per node
index.add(chunk_embeddings)

# Search
query = np.random.randn(1, d).astype("float32")
faiss.normalize_L2(query)

k = 5  # Return top 5
distances, indices = index.search(query, k)

print(f"Top {k} results: {indices[0]}")
print(f"Similarities: {distances[0]}")
```

The key trade-off is between **recall** (what fraction of true nearest neighbors are found) and **queries per second (QPS)**. Brute-force gives perfect recall but low QPS. HNSW gives 95-99% recall at 100-1000x higher QPS.

![Indexing Strategy: Speed vs Accuracy](figures/figure_5.png)

**Practical guidance:** For corpora under 100K chunks, `IndexFlatIP` (brute-force inner product) is fine. For 100K-10M chunks, use HNSW. For larger corpora, use IVF with product quantization (IVFPQ) or a managed vector database like Pinecone, Weaviate, or Qdrant.

---

## Retrieval Strategies

Retrieval is where the most impactful design decisions happen. The right retrieval strategy can make or break your RAG system.

### Dense Retrieval

This is what we have been building: embed the query, embed the chunks, find the nearest neighbors by vector similarity. Dense retrieval excels at semantic matching — it finds documents that are *about* the same thing as the query, even if they use different words.

**Strength:** Handles synonyms, paraphrases, and conceptual similarity.
**Weakness:** Can miss exact keyword matches that a user explicitly asks for.

### Sparse Retrieval (BM25)

BM25 is the classic information retrieval algorithm. It scores documents based on term frequency (TF) and inverse document frequency (IDF) — essentially, how often query words appear in a document, weighted by how rare those words are across the corpus.

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{|d|_{\text{avg}}}\right)}$$

where $f(t, d)$ is the frequency of term $t$ in document $d$, $|d|$ is the document length, $|d|_{\text{avg}}$ is the average document length, and $k_1$ and $b$ are tuning parameters (typically $k_1 = 1.2$, $b = 0.75$).

**Strength:** Excellent at exact keyword matching, especially for proper nouns, technical terms, and codes.
**Weakness:** No semantic understanding — "automobile" and "car" are unrelated words to BM25.

### Hybrid Search

The insight is that dense and sparse retrieval have complementary strengths. Hybrid search combines both:

1. Run dense retrieval and sparse retrieval independently
2. Normalize and merge the scores
3. Return the top $k$ from the merged results

```python
def hybrid_search(query, dense_index, sparse_index, alpha=0.5, k=5):
    """
    Combine dense and sparse retrieval.
    alpha: weight for dense scores (1-alpha for sparse).
    """
    # Dense retrieval
    query_embedding = embed(query)
    dense_scores, dense_ids = dense_index.search(query_embedding, k * 2)

    # Sparse retrieval (BM25)
    sparse_results = sparse_index.search(query, k * 2)

    # Normalize scores to [0, 1]
    dense_norm = normalize(dense_scores)
    sparse_norm = normalize(sparse_results.scores)

    # Merge using Reciprocal Rank Fusion (RRF)
    combined = {}
    for rank, doc_id in enumerate(dense_ids):
        combined[doc_id] = combined.get(doc_id, 0) + alpha / (rank + 60)
    for rank, doc_id in enumerate(sparse_results.ids):
        combined[doc_id] = combined.get(doc_id, 0) + (1 - alpha) / (rank + 60)

    # Sort by combined score
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
```

The `alpha` parameter controls the balance between dense and sparse. For technical documentation with many specific terms, lean toward sparse ($\alpha < 0.5$). For conversational queries, lean toward dense ($\alpha > 0.5$).

### Reranking with Cross-Encoders

There is a fundamental tension in retrieval: the most accurate models are too slow for first-stage retrieval, and the fastest models are not accurate enough for final ranking.

**Bi-encoders** (like sentence-transformers) embed the query and document independently, enabling fast vector search. But they miss fine-grained interactions between the query and document.

**Cross-encoders** take the query and document as a single input and attend to both simultaneously. This captures interactions that bi-encoders miss (negation, qualification, context-dependent meaning) but requires running the model once per query-document pair — far too slow for first-stage retrieval.

The solution: **two-stage retrieval**. Use a fast bi-encoder to retrieve the top 50-100 candidates, then rerank with a cross-encoder to select the final top $k$.

```python
from sentence_transformers import CrossEncoder

# Load reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# First stage: retrieve top 50 candidates
candidates = dense_index.search(query_embedding, 50)

# Second stage: rerank with cross-encoder
pairs = [(query, chunks[doc_id]) for doc_id in candidates]
scores = reranker.predict(pairs)

# Sort by reranker score and take top 5
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
top_k = reranked[:5]
```

![Retrieval Strategy Comparison](figures/figure_6.png)

![Two-Stage Retrieval with Reranking](figures/figure_7.png)

---

## Generation with Retrieved Context

Retrieval is only half the battle. The other half is constructing the prompt so the LLM actually uses the retrieved context effectively.

The simplest approach is to concatenate the retrieved chunks into the prompt:

```python
def build_rag_prompt(query, retrieved_chunks):
    """Build a RAG generation prompt."""
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = f"""Answer the question based ONLY on the following context.
If the context does not contain enough information to answer,
say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
    return prompt
```

But there are several important design decisions here:

**1. Context window management.** If you retrieve 10 chunks of 500 tokens each, that is 5,000 tokens of context — a significant fraction of the model's context window. More context is not always better: research by Liu et al. (2023) showed that models struggle to use information in the middle of long contexts (the "lost in the middle" phenomenon). Retrieve fewer, higher-quality chunks rather than many mediocre ones.

**2. Source attribution.** Number your sources and instruct the model to cite them. This makes the output verifiable:

```
Answer the question using the provided sources.
Cite sources using [Source N] notation.
```

**3. Faithfulness instructions.** Explicitly tell the model not to use knowledge outside the retrieved context. Without this, the model may blend parametric knowledge with retrieved information, producing answers that are partially supported and partially hallucinated — the worst of both worlds.

**4. Handling conflicting sources.** When retrieved chunks contradict each other, the model should flag the conflict rather than silently choosing one version.

---

## Evaluation — Measuring RAG Quality

RAG evaluation is more complex than standard LLM evaluation because there are multiple points of failure: the retrieval might return the wrong documents, the generation might ignore the right documents, or the final answer might be correct but unfaithful to the sources.

The RAGAS framework (Retrieval Augmented Generation Assessment) defines three key metrics:

### Faithfulness

Does the generated answer stick to the information in the retrieved context? An answer is faithful if every claim it makes can be traced back to a retrieved chunk. This catches hallucination — the model generating claims that are not supported by any source.

$$\text{Faithfulness} = \frac{\text{Number of claims supported by context}}{\text{Total number of claims in the answer}}$$

A faithfulness score of 0.8 means 80% of the answer's claims are grounded in the retrieved documents, while 20% may be hallucinated.

### Context Relevance

Are the retrieved chunks actually relevant to the question? This evaluates the retrieval component independently of the generation. Even if the answer is correct, retrieving irrelevant documents wastes context tokens and can confuse the model.

$$\text{Context Relevance} = \frac{\text{Number of relevant chunks retrieved}}{\text{Total chunks retrieved}}$$

### Answer Correctness

Is the final answer correct? This is the end-to-end metric that evaluates the full pipeline. It can be measured against ground-truth answers using exact match, F1, or semantic similarity.

```python
def evaluate_rag_pipeline(test_set, rag_pipeline, judge_model):
    """Evaluate a RAG pipeline using LLM-as-judge."""
    results = []

    for question, ground_truth in test_set:
        # Run the pipeline
        answer, sources = rag_pipeline(question)

        # Evaluate faithfulness
        faithfulness_prompt = f"""Given the following sources and answer,
determine what fraction of claims in the answer are supported by the sources.

Sources: {sources}
Answer: {answer}

Return a score between 0.0 and 1.0."""

        faithfulness = judge_model(faithfulness_prompt)

        # Evaluate correctness
        correctness_prompt = f"""Compare the generated answer to the ground truth.
Score the correctness from 0.0 to 1.0.

Ground truth: {ground_truth}
Generated answer: {answer}

Score:"""

        correctness = judge_model(correctness_prompt)

        results.append({
            "faithfulness": faithfulness,
            "correctness": correctness,
        })

    return results
```

The LLM-as-judge approach works surprisingly well for RAG evaluation. Research has shown that GPT-4 and Claude as judges correlate strongly with human evaluations — especially for faithfulness, where detecting unsupported claims is a task LLMs excel at.

![RAG Quality by Pipeline Configuration](figures/figure_8.png)

---

## Putting It All Together — A Decision Framework

We have covered the complete RAG pipeline. But when should you use RAG in the first place? And when should you reach for alternatives?

**Use RAG when:**
- Your knowledge changes frequently (company docs, news, product catalogs)
- You need source attribution and verifiability
- The knowledge base is too large for the context window (millions of documents)
- You cannot or do not want to fine-tune a model
- You need to combine information from multiple sources

**Use fine-tuning instead when:**
- The task requires specialized behavior or tone (not just knowledge)
- The same information is needed for every query (no retrieval variance)
- Latency is critical and you cannot afford the retrieval step
- The knowledge base is stable and well-defined

**Use long context instead when:**
- The entire knowledge base fits in the context window (under 100K tokens)
- You need the model to reason across the full corpus simultaneously
- The cost of long-context inference is acceptable

In practice, these approaches often combine: a RAG system retrieves relevant chunks, which are then fed to a fine-tuned model that has been trained to follow specific formatting and citation conventions. The decision is not "RAG or fine-tuning" — it is "what role does each approach play in the overall system?"

![When to Use RAG: A Decision Framework](figures/figure_9.png)

---

## Conclusion

Let us take a step back and see the full picture. RAG is context engineering at the retrieval level — it is about getting the right documents into the context window so the model can do its best work.

We started with the fundamental equation — conditioning generation on retrieved documents rather than relying solely on parametric knowledge. We walked through each stage of the pipeline: chunking strategies that preserve semantic coherence, embedding models that map text to meaning-preserving vectors, vector databases that enable sub-millisecond search at scale, and retrieval strategies that balance semantic understanding with keyword precision.

We saw how two-stage retrieval with reranking improves precision without sacrificing recall, and how prompt construction determines whether the model actually uses the context it receives. Finally, we built an evaluation framework around faithfulness, relevance, and correctness — because a RAG system that cannot be measured cannot be improved.

The key takeaway: RAG is not just "search plus LLM." It is a carefully designed pipeline where every component — from chunk boundaries to embedding models to retrieval strategies to prompt structure — compounds to determine the quality of the final answer. Get any one step wrong, and the whole system suffers. Get them all right, and you have a system that is accurate, verifiable, and perpetually up to date.

That's it!

---

**References:**

- Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
- Es, Shahul, et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)
- Liu, Nelson, et al. "Lost in the Middle: How Language Models Use Long Contexts" (2023)
- Johnson, Jeff, et al. "Billion-scale similarity search with GPUs" (IEEE TBD 2019) — FAISS
- Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP 2020)
- Robertson, Stephen, and Hugo Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)
