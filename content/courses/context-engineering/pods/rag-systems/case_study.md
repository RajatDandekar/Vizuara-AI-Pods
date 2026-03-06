# Case Study: Building a Regulatory Compliance Assistant with RAG

---

## Section 1: Industry Context and Business Problem

### Industry: Financial Regulatory Compliance (RegTech)

Financial institutions operate under an ever-expanding web of regulations — Basel III, MiFID II, Dodd-Frank, GDPR, AML directives, and hundreds of jurisdiction-specific rules that change quarterly. Compliance teams spend thousands of hours manually searching through regulatory documents, internal policies, and enforcement actions to answer questions like "Does this trade structure comply with the latest SEC guidance on cross-border derivatives?"

The global RegTech market was valued at \$12.8 billion in 2024 and is projected to reach \$38.3 billion by 2030. The core driver: compliance failures are extraordinarily expensive. In 2023 alone, global financial institutions paid over \$6.6 billion in regulatory fines.

### Company Profile: Meridian Compliance Technologies

- **Founded**: 2021, San Francisco, CA
- **Team**: 47 employees — 18 engineers, 8 compliance domain experts, 6 ML engineers, 15 operations/sales
- **Funding**: Series B (\$34M raised; \$12M Series A in 2022, \$22M Series B in 2024)
- **Product**: "Meridian Advisor" — a compliance question-answering platform used by mid-size banks and asset managers
- **Customers**: 28 financial institutions, primarily \$5B-\$50B AUM asset managers and regional banks
- **Current approach**: Keyword-based search (Elasticsearch) over a curated corpus of 140,000+ regulatory documents

### Business Challenge

Meridian's keyword search is failing as regulations grow more complex. Three critical pain points:

1. **Semantic gap**: Compliance officers ask natural language questions ("Can we accept soft-dollar arrangements for research from EU counterparties?") but keyword search requires them to know the exact regulatory terminology ("MiFID II unbundling requirements for investment research payments"). Customer satisfaction surveys show 62% of users reformulate their query at least 3 times before finding the relevant regulation.

2. **Cross-reference blindness**: Regulations frequently reference other regulations. A single compliance question might require synthesizing information from an SEC no-action letter, a FINRA guidance notice, and an internal policy memo. Keyword search returns each document independently with no synthesis.

3. **Hallucination risk**: Meridian explored using GPT-4 directly for compliance Q&A, but in a pilot with 200 questions, the model confabulated regulatory citations in 23% of responses — inventing plausible-sounding but nonexistent regulation numbers. In a compliance context, a single fabricated citation could trigger regulatory action.

### Why It Matters

- **Revenue at risk**: Three enterprise customers (\$2.1M ARR) have signaled they will not renew unless accuracy improves. Two competitors (Ascent and Reg-AI) have launched RAG-based products.
- **Compliance liability**: If Meridian's system provides incorrect regulatory guidance that a client acts upon, the liability chain extends to Meridian under professional services agreements.
- **Regulatory pressure**: The OCC and FRB have issued joint guidance requiring that AI-assisted compliance tools maintain auditable source attribution — meaning every answer must cite its sources with verifiable links.

### Constraints

- **Compute budget**: \$8,000/month cloud spend (AWS). Cannot justify GPU instances exceeding 4x A10G.
- **Latency**: Answers must return within 8 seconds for interactive use. Batch analysis (overnight compliance sweeps) can take up to 60 seconds per query.
- **Privacy**: Documents include client-specific internal policies (SOC 2 Type II compliance required). No data can leave the customer's VPC for the on-premise tier.
- **Data**: 140,000 regulatory documents (averaging 12 pages each, ~1.7 million pages total). Documents range from 200-word guidance notices to 900-page omnibus regulations. Updated weekly with ~200 new documents.
- **Accuracy threshold**: The board has mandated that the system must achieve >90% faithfulness (answers grounded in retrieved sources) and >85% answer correctness on a held-out evaluation set of 500 expert-annotated Q&A pairs.
- **Team expertise**: The ML team is strong on NLP fundamentals but has limited production RAG experience. Two engineers have deployed Elasticsearch-based search; none have operated a vector database in production.

---

## Section 2: Technical Problem Formulation

### Problem Type: Retrieval-Augmented Question Answering

At its core, this is a **conditional text generation** problem where the generation must be strictly grounded in retrieved evidence. We are not building a classifier or a regression model — we are building a system that, given a natural language question, retrieves the most relevant regulatory passages and generates a faithful, citation-backed answer.

Why not pure generation (just use an LLM)? Because the compliance domain demands verifiability. Every claim in the answer must trace back to a specific document, section, and paragraph. Pure generation cannot provide this — the model's parametric knowledge is a black box with no audit trail.

Why not pure retrieval (just return the top-k documents)? Because regulatory questions often require synthesis across multiple documents. Returning a list of documents forces the compliance officer to read and synthesize manually, which is exactly the bottleneck Meridian is trying to eliminate.

RAG gives us both: the retrieval provides the evidence trail, and the generation provides the synthesis.

### Input Specification

- **Query** $x$: A natural language compliance question (typically 15-80 tokens). Examples:
  - "What are the reporting requirements for swap dealers under Dodd-Frank Title VII?"
  - "Does the EU AI Act affect our algorithmic trading systems?"
  - "What is the maximum fine for late SAR filing under BSA?"
- **Document corpus** $\mathcal{D}$: 140,000 regulatory documents in PDF, HTML, and plain text formats. Each document has metadata: source authority (SEC, FINRA, OCC, etc.), publication date, document type (rule, guidance, enforcement action, no-action letter), and jurisdiction.

### Output Specification

- **Answer** $y$: A natural language response (typically 100-400 tokens) that directly addresses the question
- **Citations** $c$: A list of (document_id, section, paragraph) tuples for every factual claim in the answer
- **Confidence score** $s$: A calibrated probability estimate of answer correctness

The system must NEVER generate an answer without at least one supporting citation. If no relevant documents are found, the system must respond with "I could not find relevant regulatory guidance for this question" rather than generating from parametric knowledge.

### Mathematical Foundation

The RAG formulation from the article provides our theoretical framework. Given question $x$, we compute:

$$P(y \mid x) = \sum_{z \in \mathcal{Z}} P(y \mid x, z) \cdot P(z \mid x)$$

In practice, we approximate this sum by retrieving only the top-$k$ documents (typically $k=5$ to $k=20$) rather than summing over the entire corpus:

$$P(y \mid x) \approx \sum_{i=1}^{k} P(y \mid x, z_i) \cdot P(z_i \mid x)$$

The retrieval model $P(z \mid x)$ is implemented as a two-stage pipeline:
1. **Dense retrieval**: Encode both query and documents into a shared embedding space, then retrieve by cosine similarity. This captures semantic meaning — "soft-dollar arrangements" matches "research payment accounts" even though they share no keywords.
2. **Sparse retrieval (BM25)**: Keyword-based scoring that captures exact term matches. Critical for regulatory identifiers like "Rule 15c3-5" or "Section 13(d)".
3. **Hybrid fusion**: Combine dense and sparse scores using Reciprocal Rank Fusion (RRF):

$$\text{RRF}(z) = \sum_{r \in \{dense, sparse\}} \frac{1}{k_0 + \text{rank}_r(z)}$$

where $k_0 = 60$ is a damping constant. RRF is rank-based rather than score-based, which means it naturally handles the different score distributions of dense and sparse retrievers without calibration.

4. **Cross-encoder reranking**: After fusion, the top candidates are reranked by a cross-encoder that processes the (query, document) pair jointly:

$$\text{relevance}(x, z) = \text{CrossEncoder}([x; z])$$

Cross-encoders are much more accurate than bi-encoders (they see the full interaction between query and document) but too slow for first-stage retrieval over 140K documents. Using them as a second-stage reranker over 50-100 candidates gives the best accuracy-latency tradeoff.

### Loss Function

For the embedding model (fine-tuned on regulatory domain data), we use InfoNCE contrastive loss:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{j=1}^{N} \exp(\text{sim}(q, d_j^-) / \tau)}$$

- **Numerator**: The similarity between the query $q$ and the positive (relevant) document $d^+$. This term pushes relevant pairs closer together in embedding space.
- **Denominator**: The sum over the positive and all negative documents. This pushes irrelevant documents away.
- **Temperature** $\tau$: Controls the sharpness of the distribution. Lower $\tau$ makes the model more discriminative but harder to train. We use $\tau = 0.05$.
- **Hard negatives**: The quality of negative samples $d_j^-$ is critical. Random negatives are too easy — the model learns nothing. We use BM25 to mine hard negatives: documents that share keywords with the query but are not relevant. This forces the embedding model to go beyond keyword matching.

If we remove the hard negatives and use only random negatives, Recall@10 drops from 0.89 to 0.71 in our experiments — a 20% degradation. The model learns a trivially easy task (distinguish "What is Basel III?" from a document about recipe ingredients) rather than the hard task (distinguish it from a document about Basel II).

### Evaluation Metrics

Following the RAGAS framework from the article:

1. **Faithfulness** (primary): What fraction of claims in the generated answer are supported by the retrieved context? Target: >0.90. This is the most critical metric — an unfaithful answer in a compliance context is worse than no answer at all.

2. **Context Relevance**: What fraction of the retrieved chunks are actually relevant to the question? Target: >0.80. Low context relevance means we are stuffing the context window with noise, which wastes tokens and can mislead the generator.

3. **Answer Correctness**: Does the generated answer match the ground truth answer? Target: >0.85. Measured against expert-annotated Q&A pairs using a combination of semantic similarity and factual overlap.

4. **Retrieval Recall@10**: What fraction of the relevant documents appear in the top 10 retrieved results? Target: >0.85. Measured on a held-out set of query-document relevance judgments.

5. **Latency (P95)**: 95th percentile end-to-end response time. Target: <8 seconds for interactive queries.

### Baseline

Meridian's current system is **Elasticsearch BM25 + manual synthesis**. The compliance officer types a keyword query, gets a ranked list of documents, reads the top 5-10, and writes the answer manually.

Performance on the 500-question evaluation set:
- Retrieval Recall@10: 0.58 (keyword queries miss semantically relevant documents)
- End-to-end accuracy: 0.71 (limited by both retrieval quality and manual synthesis errors)
- Average response time: 23 minutes per question (including human reading and synthesis)

The RAG system needs to beat 0.71 accuracy while being orders of magnitude faster.

### Why RAG Is the Right Approach

1. **Verifiability**: RAG provides source attribution by construction — every generated claim maps to a retrieved document. This satisfies the OCC/FRB auditability requirement.
2. **Freshness**: New regulations can be added to the corpus within hours (embed and index) without retraining any model. Fine-tuning would require expensive retraining cycles.
3. **Domain adaptation**: By combining domain-specific embeddings with a general-purpose LLM, we get the best of both worlds — specialized retrieval with sophisticated generation.
4. **Cost**: RAG is dramatically cheaper than fine-tuning a large LLM on regulatory data. The embedding model is small (110M parameters) and the vector index fits in memory.

### Technical Constraints

- **Embedding model**: Must run on CPU for the on-premise tier. Maximum model size: 500M parameters.
- **Vector index**: Must support 140K documents x 10 chunks/doc = 1.4M vectors with sub-100ms query time.
- **LLM**: Claude 3.5 Sonnet via API for cloud tier; Llama 3.1 8B for on-premise tier.
- **Context window**: Maximum 8,192 tokens for retrieved context (leaving room for system prompt and generation).

---

## Section 3: Implementation Notebook Structure

### 3.1 Data Acquisition Strategy

**Dataset**: We use the [CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad) — a corpus of 510 real commercial legal contracts with 13,101 expert annotations across 41 legal clause types. Each annotation identifies specific contract provisions (termination clauses, indemnification, IP rights, etc.) with the exact text span.

This dataset is ideal because:
- It contains real legal language (not synthetic)
- The annotations provide ground truth for retrieval evaluation
- The clause-type labels enable us to build meaningful Q&A pairs
- It is publicly available under CC BY 4.0

**Data loading pipeline**:
- Download CUAD from the Atticus Project
- Parse contracts from PDF/text format
- Extract annotated clause spans as ground truth passages
- Generate Q&A pairs from annotations (e.g., annotation "Termination for Convenience: Either party may terminate upon 30 days written notice" becomes Q: "What are the termination provisions?" A: "Either party may terminate upon 30 days written notice")

**TODO**: Students implement the Q&A pair generation function that converts CUAD annotations into (question, answer, evidence_passage) triples.

### 3.2 Exploratory Data Analysis

- Distribution of document lengths (tokens per contract)
- Distribution of clause types across the corpus
- Vocabulary overlap between different clause categories
- Average annotation span length by clause type
- Identify contracts with unusual structure (tables, nested clauses, multi-language sections)

**TODO**: Students write EDA code and answer: "Which clause types have the highest variance in span length? What does this tell us about chunking strategy?"

### 3.3 Baseline Approach

Implement a BM25 keyword search baseline:
- Tokenize documents and build an inverted index
- Implement BM25 scoring from scratch (the article provides the formula)
- Evaluate Recall@5 and Recall@10 on the Q&A evaluation set
- Measure precision and MRR (Mean Reciprocal Rank)

**TODO**: Students implement BM25 scoring and evaluate it. Expected Recall@10: ~0.55-0.65.

### 3.4 Model Design

Build the full RAG pipeline component by component:

1. **Chunking module**: Implement recursive character splitting with overlap, respecting legal section boundaries (identified by section numbering patterns like "3.1", "3.2", "Article IV")
2. **Embedding module**: Use `sentence-transformers/all-MiniLM-L6-v2` as the embedding model. Encode all chunks into 384-dimensional vectors.
3. **Vector index**: Build a FAISS index (IVF with 100 centroids + PQ compression) for approximate nearest neighbor search
4. **Hybrid retriever**: Combine dense retrieval scores with BM25 scores using Reciprocal Rank Fusion
5. **Reranker**: Use a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank the top-50 hybrid results down to top-5
6. **Generator**: Construct a prompt with the question, top-5 retrieved chunks, and a system instruction requiring citation

**TODO**: Students implement the hybrid retriever fusion function and the prompt construction function.

### 3.5 Training Strategy

For this case study, the primary "training" is the embedding fine-tuning:
- Fine-tune `all-MiniLM-L6-v2` on CUAD with contrastive learning
- Use annotated clause spans as positive pairs (question <-> clause text)
- Mine hard negatives using BM25 (clauses that share keywords but are from different clause types)
- Optimizer: AdamW with learning rate 2e-5, weight decay 0.01
- Cosine learning rate schedule with 10% warmup
- Train for 5 epochs with batch size 32

**TODO**: Students implement the contrastive training loop and hard negative mining.

### 3.6 Evaluation

- Compare retrieval quality: BM25 baseline vs dense-only vs hybrid vs hybrid+reranking
- Measure RAGAS metrics: faithfulness, context relevance, answer correctness
- Measure end-to-end latency breakdown (embedding, retrieval, reranking, generation)
- Ablation: impact of chunk size (256 vs 512 vs 1024 tokens) on retrieval quality

**TODO**: Students generate evaluation plots and write a 1-paragraph interpretation of results.

### 3.7 Error Analysis

- Categorize retrieval failures: semantic misses, boundary splits, metadata-dependent queries
- Categorize generation failures: hallucinated citations, incomplete answers, wrong clause type
- Identify the top-3 failure modes and propose mitigations

**TODO**: Students examine 10 failure cases, categorize them, and propose one fix for the most common failure mode.

### 3.8 Scalability and Deployment Considerations

- Profile inference latency by component
- Estimate memory requirements for scaling to 1.4M vectors (Meridian's full corpus)
- Compare FAISS index types (Flat, IVF, HNSW) on query latency vs recall tradeoff
- Write a simple inference benchmarking script that measures P50/P95/P99 latency

**TODO**: Students implement the benchmarking script and report latency percentiles.

### 3.9 Ethical and Regulatory Analysis

- **Bias**: Legal language has historical biases (gendered pronouns, culturally specific assumptions). How might the embedding model propagate or amplify these?
- **Fairness**: Does the system perform equally well on contracts from different jurisdictions, industries, and time periods?
- **Accountability**: When the RAG system provides incorrect compliance guidance, who is responsible — the tool vendor, the user, or the LLM provider?
- **Regulatory compliance**: How does the system satisfy the OCC/FRB auditability requirements? What logging is needed?

**TODO**: Students write a brief (300-word) ethical impact assessment addressing the accountability question.

---

## Section 4: Production and System Design Extension

### Architecture

```
                        +------------------+
                        |   Load Balancer  |
                        +--------+---------+
                                 |
                    +------------+------------+
                    |                         |
           +-------v-------+       +---------v--------+
           | API Gateway   |       | API Gateway       |
           | (Interactive) |       | (Batch)           |
           +-------+-------+       +---------+---------+
                   |                          |
           +-------v-------+       +---------v---------+
           | Query Service  |       | Batch Processor   |
           | (FastAPI)     |       | (Celery Workers)  |
           +--+----+---+---+       +---------+---------+
              |    |   |                     |
     +--------+  +-+  +--------+   +---------+---------+
     |           |             |   |                    |
+----v----+ +---v----+ +------v-----+  +----------------+
|Embedding| |Hybrid  | |Cross-      |  | Result Cache   |
|Service  | |Search  | |Encoder     |  | (Redis)        |
|(CPU)    | |Service | |Reranker    |  +----------------+
+---------+ +---+----+ +------+-----+
                |              |
          +-----v------+ +----v-----+
          |FAISS Index | |LLM       |
          |+ BM25 Index| |Service   |
          |(Milvus)    | |(Claude/  |
          +------------+ |Llama)    |
                         +----------+
```

### API Design

**Interactive Query Endpoint**:
```
POST /v1/query
{
  "question": "string",
  "filters": {
    "source_authority": ["SEC", "FINRA"],
    "date_range": {"start": "2020-01-01", "end": "2024-12-31"},
    "document_type": ["rule", "guidance"]
  },
  "max_citations": 5,
  "confidence_threshold": 0.7
}

Response:
{
  "answer": "string",
  "citations": [
    {
      "document_id": "string",
      "document_title": "string",
      "section": "string",
      "text_span": "string",
      "relevance_score": 0.95
    }
  ],
  "confidence": 0.88,
  "latency_ms": 3200
}
```

**Batch Analysis Endpoint**:
```
POST /v1/batch
{
  "questions": ["string"],
  "webhook_url": "string",
  "priority": "normal"
}
```

### Serving Infrastructure

- **Embedding service**: 2x c5.2xlarge (CPU) running sentence-transformers. Batch queries for throughput.
- **Vector search**: Milvus cluster (3 nodes) with IVF_HNSW index. Supports hot-swapping indices for zero-downtime updates.
- **Reranker**: 1x g5.xlarge (A10G GPU) running the cross-encoder. Batches of 50 candidates per query.
- **LLM**: Claude 3.5 Sonnet via API (cloud tier) or Llama 3.1 8B on 1x g5.2xlarge (on-premise tier).
- **Scaling**: Horizontal scaling via Kubernetes HPA. Embedding and reranker services scale on CPU/GPU utilization. Query service scales on request queue depth.

### Latency Budget

| Component | P50 (ms) | P95 (ms) | Budget (ms) |
|-----------|----------|----------|-------------|
| Query embedding | 15 | 25 | 50 |
| FAISS retrieval | 8 | 15 | 30 |
| BM25 retrieval | 12 | 20 | 30 |
| RRF fusion | 2 | 5 | 10 |
| Cross-encoder rerank | 180 | 250 | 300 |
| LLM generation | 1800 | 3500 | 5000 |
| Network overhead | 50 | 100 | 200 |
| **Total** | **2065** | **3915** | **5620** |

The LLM generation dominates latency. For the on-premise tier with Llama 8B, generation P95 drops to ~1200ms but answer quality is lower.

### Monitoring

- **Retrieval quality**: Track RAGAS faithfulness and context relevance on a rolling sample of 100 queries/day (using LLM-as-judge evaluation)
- **Latency**: P50/P95/P99 per component, alerting on P95 > 6 seconds
- **Usage**: Queries per customer, peak QPS, cache hit rate
- **Index freshness**: Time since last document ingestion, document count delta
- **Error rate**: Failed queries (timeout, LLM refusal, no results), alerting on >2% error rate
- **Dashboards**: Grafana with panels for latency histogram, RAGAS scores over time, retrieval recall trend, and cost per query

### Model Drift Detection

- **Query distribution shift**: Monitor embedding centroid of incoming queries. If the centroid shifts significantly (cosine distance > 0.15 from the training query distribution), alert the ML team — users may be asking about regulation types not covered by the current embedding model.
- **Retrieval performance drift**: Weekly evaluation on a held-out set of 50 annotated queries. If Recall@10 drops below 0.80, trigger a retraining pipeline.
- **Document corpus shift**: Track the rate of new document ingestion. If regulatory language patterns shift (detected via vocabulary overlap analysis), schedule embedding model fine-tuning.

### Model Versioning

- **Embedding models**: Versioned in MLflow with metadata (training dataset version, evaluation metrics, training config). Each model version produces a new FAISS index.
- **FAISS indices**: Stored in S3 with version tags. Old indices retained for 30 days for rollback.
- **Rollback**: If a new model version degrades RAGAS metrics below thresholds, automatically roll back to the previous index version within 5 minutes via blue-green deployment.

### A/B Testing

- **Traffic splitting**: Route 10% of queries to the challenger model, 90% to the current champion.
- **Metrics**: Compare faithfulness, answer correctness, latency, and user satisfaction (thumbs up/down) between variants.
- **Statistical significance**: Require p < 0.05 on a two-sided t-test for the primary metric (faithfulness) before promoting a challenger. Minimum sample size: 500 queries per variant.
- **Guardrails**: Automatically halt the experiment if the challenger's faithfulness drops below 0.85 (absolute floor).

### CI/CD for ML

1. **Document ingestion pipeline** (runs weekly):
   - Fetch new documents from regulatory feeds
   - Chunk and embed new documents
   - Incrementally update the FAISS index
   - Run smoke tests (10 canary queries) to verify index integrity

2. **Embedding model retraining pipeline** (runs monthly or on drift alert):
   - Pull latest annotated Q&A pairs from the evaluation database
   - Mine hard negatives from the updated corpus
   - Fine-tune embedding model with contrastive learning
   - Evaluate on held-out test set
   - If metrics improve, promote to staging; if staging passes, promote to production

3. **Validation gates**:
   - Retrieval Recall@10 >= 0.85
   - RAGAS faithfulness >= 0.90
   - P95 latency <= 8 seconds
   - No regression on any metric > 2% from current production

### Cost Analysis

| Component | Monthly Cost |
|-----------|-------------|
| Embedding service (2x c5.2xlarge) | \$980 |
| Milvus cluster (3x r5.xlarge) | \$1,350 |
| Reranker GPU (1x g5.xlarge) | \$1,200 |
| LLM API (Claude, ~50K queries/month) | \$2,500 |
| S3 storage (indices + documents) | \$120 |
| Redis cache | \$180 |
| Monitoring (Grafana Cloud) | \$150 |
| **Total** | **\$6,480/month** |

This is well within Meridian's \$8,000/month budget. The LLM API cost scales linearly with query volume — at 100K queries/month, it would reach ~\$5,000, still under budget with the other components.

For the on-premise tier, replacing Claude API with a self-hosted Llama 8B on 1x g5.2xlarge adds \$1,800/month but eliminates the per-query LLM cost, making it more economical above ~60K queries/month.
