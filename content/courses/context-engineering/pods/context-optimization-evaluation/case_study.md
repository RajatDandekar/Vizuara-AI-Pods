# Case Study: Optimizing Context Quality for MedAssist AI — A Clinical Decision Support System

## 1. Industry Context and Business Problem

**Company:** MedAssist AI, a Series B health-tech startup building an AI-powered clinical decision support system for emergency departments across 45 hospitals in the United States.

**System Overview:** MedAssist AI retrieves relevant medical literature, clinical guidelines, and patient history to help emergency physicians make faster, evidence-based decisions. The system processes approximately 12,000 physician queries per day, each requiring retrieval from a corpus of 2.3 million medical documents.

**The Problem:** After 8 months in production, MedAssist's answer quality has degraded significantly:

- **Context precision** dropped from 0.87 to 0.69 — nearly one-third of retrieved documents are now irrelevant, wasting tokens on noise
- **Faithfulness** declined from 0.94 to 0.82 — the system is hallucinating clinical recommendations that are not grounded in the retrieved evidence
- **Token utilization** increased by 40% — the average query now consumes 85,000 tokens, approaching the 128K context window limit
- **Latency** increased from 1.8s to 4.2s — physicians are abandoning the tool during time-critical situations

**Financial Impact:** Each hallucinated recommendation that reaches a physician costs an estimated \$15,000 in liability review. With 12,000 daily queries and a faithfulness score of 0.82, approximately 2,160 queries per day contain at least one unsupported claim. The legal team estimates annual liability exposure at \$8.2 million.

**Business Constraint:** MedAssist must restore answer quality to baseline levels (faithfulness > 0.92, precision > 0.85) within 6 weeks without increasing per-query costs beyond \$0.05.

---

## 2. Technical Problem Formulation

### 2.1 Formal Objective

We define the optimization problem as maximizing a composite quality score $Q$ subject to cost and latency constraints:

$$\max_{\theta} Q(\theta) = w_f \cdot F(\theta) + w_p \cdot P(\theta) + w_r \cdot R(\theta)$$

$$\text{subject to: } C(\theta) \leq \$0.05/\text{query}, \quad L(\theta) \leq 2.5\text{s}$$

where:
- $F(\theta)$ = faithfulness score (target > 0.92)
- $P(\theta)$ = context precision (target > 0.85)
- $R(\theta)$ = answer relevancy (target > 0.88)
- $w_f = 0.5, w_p = 0.3, w_r = 0.2$ (faithfulness weighted highest for clinical safety)
- $\theta$ represents the pipeline configuration parameters

### 2.2 Root Cause Analysis

The degradation has three identified root causes:

1. **Corpus staleness**: 340,000 new medical documents were added to the corpus over 8 months, but the embedding index was only refreshed quarterly. New documents are poorly indexed, and embedding drift means older embeddings no longer align well with the updated model.

2. **Query distribution shift**: Physician queries shifted from 70% pharmacology / 30% diagnostics to 45% pharmacology / 40% diagnostics / 15% procedural. The retrieval system was optimized for the original distribution.

3. **Token bloat**: As the corpus grew, the retriever returns more borderline-relevant chunks. Without compression, these consume 40% more tokens, crowding out the system prompt and clinical guidelines.

### 2.3 Metrics Framework

We adopt the RAGAS framework with clinical extensions:

| Metric | Formula | Target | Current |
|--------|---------|--------|---------|
| Faithfulness | $\frac{\text{supported claims}}{\text{total claims}}$ | > 0.92 | 0.82 |
| Context Precision | $\frac{1}{K}\sum_k \frac{\text{relevant}_k}{k}$ | > 0.85 | 0.69 |
| Answer Relevancy | $\text{sim}(q, a)$ | > 0.88 | 0.85 |
| Token Efficiency | $\frac{T_{\text{used}}}{T_{\text{budget}}}$ | < 0.70 | 0.91 |
| Latency | $L_{p99}$ | < 2.5s | 4.2s |

### 2.4 Loss Function

We define a composite loss that penalizes both quality degradation and constraint violations:

$$\mathcal{L} = -Q(\theta) + \lambda_c \cdot \max(0, C(\theta) - 0.05) + \lambda_l \cdot \max(0, L(\theta) - 2.5)$$

where $\lambda_c = 100$ and $\lambda_l = 10$ are penalty coefficients for cost and latency violations respectively.

---

## 3. Implementation Notebook Structure

### 3.1 Environment Setup and Data Loading

Load the clinical query dataset (500 annotated queries with ground-truth answers), configure API connections, and set up the evaluation framework.

### 3.2 Baseline Measurement

Run the current pipeline on the test set and compute all RAGAS metrics:

```python
# TODO: Implement baseline_evaluation()
# - Process 500 test queries through current pipeline
# - Compute faithfulness, precision, relevancy for each
# - Record token usage and latency
# - Generate baseline report
```

### 3.3 Compression Pipeline

Implement extractive compression with medical-domain TF-IDF:

```python
# TODO: Implement MedicalCompressor class
# - Train TF-IDF on medical corpus (handle medical terminology)
# - Implement extractive_compress() with clinical-aware sentence splitting
# - Target 60% compression ratio while maintaining faithfulness > 0.90
```

### 3.4 Dynamic Budget Allocator

Build a query-complexity-aware budget allocator:

```python
# TODO: Implement ClinicalBudgetAllocator
# - Classify query complexity (triage, differential diagnosis, treatment plan)
# - Allocate tokens: guidelines (always), evidence (per-query), history (if complex)
# - Ensure clinical guidelines are never compressed
```

### 3.5 RAGAS Evaluation Pipeline

Implement the full RAGAS evaluation with clinical extensions:

```python
# TODO: Implement ClinicalRAGASEvaluator
# - Standard RAGAS metrics (precision, recall, faithfulness, relevancy)
# - Clinical extension: drug interaction check (is every mentioned drug in context?)
# - Clinical extension: contraindication flag (does answer mention relevant contraindications?)
```

### 3.6 A/B Test Framework

Design and run the A/B test comparing old vs optimized pipeline:

```python
# TODO: Implement clinical_ab_test()
# - Control: current pipeline (no compression, static budget)
# - Treatment: optimized pipeline (compression + dynamic budget)
# - Primary metric: faithfulness
# - Secondary: precision, latency, cost
# - Required sample size: 500 per group (MDE = 5%, power = 80%)
```

### 3.7 Production Monitor

Build the monitoring dashboard with clinical alert thresholds:

```python
# TODO: Implement ClinicalMonitor
# - Track faithfulness with EMA (alpha=0.15, alert below 0.90)
# - Track precision with EMA (alpha=0.2, alert below 0.80)
# - Compound alert: faithfulness AND precision both degrading
# - Automatic reindexing trigger when precision < 0.75 for 3 consecutive days
```

### 3.8 End-to-End Optimization

Run the complete optimization-evaluation cycle:

```python
# TODO: Implement optimization_cycle()
# 1. Measure baseline
# 2. Apply compression (target 60% ratio)
# 3. Apply dynamic budgeting
# 4. Measure improvement
# 5. A/B test against baseline
# 6. Deploy if significant
# 7. Monitor for 7 days
```

### 3.9 Results Analysis and Reporting

Generate the final report with before/after comparison:

```python
# TODO: Implement generate_clinical_report()
# - Before/after table for all metrics
# - Cost analysis (per-query and annual)
# - Latency distribution comparison
# - Faithfulness improvement visualization
# - Recommendation for production deployment
```

---

## 4. Production and System Design Extension

### 4.1 System Architecture

The optimized MedAssist pipeline follows this architecture:

1. **Query Ingestion**: Physician query enters through the API gateway with priority classification (P0: emergency, P1: urgent, P2: standard)
2. **Query Analysis**: A lightweight classifier determines query complexity and domain (pharmacology, diagnostics, procedural)
3. **Dynamic Budget Allocation**: Based on complexity class, allocate tokens across: clinical guidelines (fixed 3K), retrieved evidence (variable 20-50K), patient history (variable 5-15K), system prompt (fixed 2K)
4. **Retrieval + Compression**: Retrieve top-20 chunks, rerank with clinical cross-encoder, apply extractive compression to fit budget
5. **Generation**: LLM generates response with explicit grounding instruction
6. **Post-Generation Audit**: Automated faithfulness check; flag responses below 0.85 for human review
7. **Monitoring**: EMA-based drift detection on all RAGAS metrics

### 4.2 Scaling Considerations

- **Batch compression**: Pre-compress frequently retrieved documents during off-peak hours
- **Embedding refresh**: Incremental index updates daily (full rebuild weekly)
- **Cache strategy**: Cache compressed versions of the top-1000 most-retrieved documents
- **Fallback**: If latency exceeds 3s, serve from a pre-computed answer cache for common queries

### 4.3 Expected Outcomes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Faithfulness | 0.82 | 0.93 | +13.4% |
| Context Precision | 0.69 | 0.87 | +26.1% |
| Token Usage | 85K | 52K | -38.8% |
| Latency (p99) | 4.2s | 2.1s | -50.0% |
| Cost/query | \$0.062 | \$0.038 | -38.7% |
| Annual liability exposure | \$8.2M | \$1.4M | -82.9% |

### 4.4 Monitoring and Maintenance

- Weekly RAGAS evaluation on 200 sampled production queries
- Monthly human evaluation (50 samples, 3 annotators, Kappa > 0.7 required)
- Quarterly A/B tests for any pipeline changes
- Automated corpus reindexing triggered by precision drops
- Annual full pipeline audit with clinical domain experts
