# Case Study: Building an AML Transaction Screening Pipeline with Prompt Design Principles

---

## Section 1: Industry Context and Business Problem

### Industry: Financial Compliance and RegTech

Financial institutions in the United States process over USD 5 trillion in wire transfers daily. Federal regulations under the Bank Secrecy Act (BSA) require every institution to monitor these transactions for suspicious activity — potential money laundering, terrorist financing, sanctions evasion, and fraud. Failure to detect and report suspicious transactions carries severe penalties: in 2023 alone, FinCEN levied over USD 1.5 billion in fines against institutions with inadequate AML programs.

The compliance monitoring market is valued at approximately USD 3.4 billion and growing at 15% annually, driven by increasing regulatory scrutiny and the rising sophistication of financial crime.

### Company Profile

**Meridian Compliance, Inc.**
- Founded: 2019, New York City
- Team: 120 employees (35 engineers, 25 compliance domain experts, 15 data scientists)
- Stage: Series B (USD 48M raised; USD 22M Series A in 2021, USD 26M Series B in 2023)
- Product: "MeridianShield" — a compliance monitoring platform serving 85 mid-market banks and credit unions (assets between USD 1B and USD 50B)
- Revenue: USD 18M ARR, growing 60% year-over-year

MeridianShield currently uses a rule-based transaction monitoring system with approximately 1,400 hand-coded rules. These rules flag transactions based on keywords, thresholds, geographic risk scores, and known typologies. The flagged transactions are then reviewed by human compliance analysts who decide whether to file a Suspicious Activity Report (SAR) with FinCEN.

### Business Challenge

The rule-based system is breaking under scale and sophistication pressures:

**Volume:** Each client institution generates between 15,000 and 80,000 transaction narratives daily. Across 85 clients, MeridianShield processes approximately 2.1 million transaction narratives per day. Of these, the rule engine flags roughly 42,000 (2%) for human review.

**False Positive Rate:** Of the 42,000 flagged transactions, only about 1,200 (2.9%) result in actual SAR filings. The remaining 97.1% are false positives — transactions that triggered rules but were ultimately benign. This means compliance analysts spend over 95% of their time reviewing transactions that require no action.

**Analyst Fatigue:** Each of Meridian's client institutions employs between 5 and 20 compliance analysts. At a review rate of approximately 25 transactions per analyst per hour, the backlog grows faster than analysts can clear it. Analyst turnover in compliance departments runs at 35% annually, largely due to the repetitive nature of the work.

**Missed True Positives:** More concerning, internal audits estimate that the rule-based system misses approximately 8-12% of genuinely suspicious transactions — those that do not match any existing rule pattern but exhibit novel laundering typologies. Regulators have flagged this gap in two recent examination cycles.

**Financial Impact:** Each false positive costs approximately USD 25 in analyst time. At 40,800 false positives per day, this represents over USD 1 million per day in wasted analyst capacity across the platform. Meanwhile, each missed SAR filing carries regulatory risk valued at USD 50,000-USD 500,000 per incident in potential fines.

### Why It Matters

Meridian's board has mandated a 60% reduction in false positives within 12 months while maintaining or improving the true positive detection rate. The VP of Engineering has proposed an LLM-powered screening layer that sits between the rule engine and human analysts, using sophisticated prompt design to analyze transaction narratives with near-human judgment.

The core insight is that transaction narrative analysis is fundamentally a natural language understanding task — exactly the kind of task where well-designed prompts dramatically outperform rigid rules.

### Constraints

- **Compute Budget:** USD 0.15 per transaction maximum (including all LLM API calls in the pipeline). At 42,000 flagged transactions per day, the daily LLM budget is USD 6,300.
- **Latency:** Each transaction must be screened within 30 seconds end-to-end. Analysts receive transactions in batches, so throughput matters more than individual latency, but no single transaction should block for more than 30 seconds.
- **Compliance:** All processing must occur within SOC 2 Type II certified infrastructure. Transaction data cannot leave the compliance perimeter. The LLM must be deployed on-premises or via a BAA-covered API endpoint.
- **Auditability:** Every screening decision must include a full reasoning trace. Regulators require explainability — a model that says "suspicious" without justification is worse than useless.
- **Data Availability:** Meridian has 3 years of historical transaction data with analyst decisions (approximately 2.3 million labeled transactions: 67,000 SAR-filed, 2.23 million cleared). They also have access to FinCEN's published SAR narratives and AML typology guides.
- **Team Expertise:** The data science team has experience with classical ML (XGBoost, logistic regression) but limited experience with LLM prompt engineering. The solution must be maintainable by the existing team.
- **Regulatory Oversight:** The OCC (Office of the Comptroller of the Currency) requires that any AI/ML system used in compliance must have a documented model risk management framework. The system cannot be a black box.

---

## Section 2: Technical Problem Formulation

### Problem Type: Multi-Stage Classification with Structured Reasoning

At first glance, this looks like a binary classification problem: is this transaction suspicious or not? But framing it as simple binary classification misses critical nuances that make the problem interesting and that prompt design techniques are uniquely suited to address.

**Why not a traditional classifier?** A fine-tuned BERT or XGBoost model could classify transactions as suspicious/not suspicious. But regulators require *reasoning* — not just a label, but an explanation of *why* a transaction is suspicious. Traditional classifiers produce probabilities, not reasoning traces. You could add a post-hoc explanation layer (SHAP values, attention visualization), but these explain what features the model attended to, not the logical reasoning chain that a compliance analyst would follow.

**Why prompt design?** The prompt-based approach naturally produces both a decision and a reasoning trace. By using chain-of-thought prompting, the model's reasoning is explicit and auditable. By using structured output, the reasoning is parseable and can be fed into downstream systems. By using prompt chaining, the complex analysis is decomposed into reliable sub-steps. This is the fundamental advantage: prompt design techniques produce *interpretable, structured, multi-step reasoning* — exactly what regulators require.

### Input Specification

Each transaction record contains:

- **Narrative** (string, 50-500 words): Free-text description of the transaction, including originator, beneficiary, purpose, and any notes from the processing bank. Example: "Wire transfer from Oceanic Trading Ltd (BVI) to account ending 4471 at First National Bank. Purpose stated as 'consulting services Q3.' Amount: USD 847,000. Originator bank: Royal Caribbean International Bank, Cayman Islands."

- **Structured fields** (JSON object):
  - `amount`: Transaction amount in USD (float)
  - `currency`: Original currency code (string)
  - `originator_country`: ISO 3166-1 alpha-2 code (string)
  - `beneficiary_country`: ISO 3166-1 alpha-2 code (string)
  - `transaction_type`: One of {wire, ACH, check, cash_deposit, cash_withdrawal} (string)
  - `originator_entity_type`: One of {individual, corporation, government, ngo} (string)
  - `rule_triggers`: List of rule IDs that flagged this transaction (array of strings)

- **Context** (JSON object):
  - `account_history_summary`: Brief summary of the account's recent activity (string, max 200 words)
  - `customer_risk_rating`: One of {low, medium, high, prohibited} (string)
  - `sanctions_screening_result`: One of {clear, potential_match, confirmed_match} (string)

### Output Specification

The pipeline must produce a structured decision object:

```json
{
  "transaction_id": "TXN-2024-0847291",
  "risk_assessment": "high" | "medium" | "low" | "clear",
  "recommendation": "file_sar" | "escalate_to_senior" | "clear_with_note" | "clear",
  "confidence": 0.0 to 1.0,
  "reasoning": {
    "risk_factors_identified": ["list of specific risk indicators found"],
    "typology_match": "closest AML typology pattern, if any",
    "mitigating_factors": ["factors that reduce suspicion"],
    "reasoning_chain": "full step-by-step reasoning trace"
  },
  "regulatory_citations": ["BSA/AML regulation sections applicable"]
}
```

**Why this output structure?** Each field serves a specific downstream purpose:
- `risk_assessment` feeds into the analyst's triage dashboard (sorting by priority)
- `recommendation` determines the workflow routing (direct to SAR filing vs. senior review vs. clear)
- `confidence` enables threshold-based automation (high-confidence clears can be auto-processed)
- `reasoning` satisfies regulatory explainability requirements
- `regulatory_citations` helps analysts draft SAR narratives faster

### Mathematical Foundation

The core mathematical framework combines Bayesian reasoning with information-theoretic concepts from prompt design.

**Prior probability from rules:** The rule engine provides an initial suspicion signal. Let $P(S \mid R)$ denote the probability that a transaction is truly suspicious given that it triggered rule set $R$. From historical data, we know this prior is weak: $P(S \mid R) \approx 0.029$ (the 2.9% SAR filing rate among flagged transactions).

**Posterior probability with LLM analysis:** The LLM pipeline acts as a likelihood function. Given the transaction narrative $x$, context $c$, and few-shot examples $E$, the LLM produces a risk assessment. We model the LLM's output as updating the prior:

$$P(S \mid x, c, E, R) = \frac{P(x, c \mid S, E) \cdot P(S \mid R)}{P(x, c \mid E)}$$

The few-shot examples $E$ sharpen the likelihood $P(x, c \mid S, E)$ by calibrating the LLM's understanding of what "suspicious" means in this specific regulatory context. Without examples (zero-shot), the LLM's notion of "suspicious" is broad and miscalibrated. With domain-specific examples, the likelihood becomes sharply discriminative.

**Chain reliability in prompt chaining:** As discussed in the article, if the pipeline has $n$ steps each with accuracy $p_i$, the chain accuracy is:

$$P(\text{chain correct}) = \prod_{i=1}^{n} p_i$$

For our 4-step pipeline, we need each step to achieve at least 97% accuracy to maintain overall accuracy above 88%:

$$0.97^4 = 0.885$$

Gate checks between steps mitigate cascading errors. If a gate check catches and corrects errors with probability $g$, the effective per-step accuracy becomes:

$$p_i^{\text{effective}} = p_i + (1 - p_i) \cdot g$$

With $p_i = 0.95$ and $g = 0.7$ (the gate catches 70% of errors):

$$p_i^{\text{effective}} = 0.95 + 0.05 \times 0.7 = 0.985$$

$$P(\text{chain correct}) = 0.985^4 = 0.941$$

Gate checks improve chain reliability from 81.5% to 94.1%.

### Loss Function

Since we are using an LLM-based pipeline (not training a model from scratch), our "loss function" is defined in terms of the decision quality metrics we optimize during prompt design and calibration:

**Primary objective — Weighted classification loss:**

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_{\text{FN}} \cdot y_i \log(\hat{y}_i) + w_{\text{FP}} \cdot (1 - y_i) \log(1 - \hat{y}_i) \right]$$

where $y_i \in \{0, 1\}$ is the ground truth (1 = truly suspicious), $\hat{y}_i$ is the pipeline's predicted probability, $w_{\text{FN}} = 10$ (false negatives are 10x more costly than false positives — a missed SAR filing has severe regulatory consequences), and $w_{\text{FP}} = 1$.

**Why asymmetric weights?** If we remove the asymmetry ($w_{\text{FN}} = w_{\text{FP}} = 1$), the pipeline optimizes for overall accuracy, which would clear most transactions (since 97% are benign) and miss many truly suspicious ones. The 10:1 weighting ensures the pipeline is conservative — it would rather flag a clean transaction for human review than miss a genuinely suspicious one. This matches regulatory expectations: compliance systems should err on the side of caution.

**Secondary objective — Reasoning quality:**

We also evaluate the quality of the reasoning trace using LLM-as-judge scoring. A separate evaluation prompt grades each reasoning chain on:
- Completeness (did it consider all relevant risk factors?): scored 0-5
- Logical coherence (does each step follow from the previous?): scored 0-5
- Regulatory grounding (does it cite specific BSA/AML requirements?): scored 0-5

The reasoning quality score is: $Q_{\text{reasoning}} = \frac{1}{3}(Q_{\text{complete}} + Q_{\text{coherent}} + Q_{\text{grounded}})$

Minimum acceptable threshold: $Q_{\text{reasoning}} \geq 3.5$ (out of 5.0).

### Evaluation Metrics

**Primary metrics:**
- **SAR Capture Rate (Recall):** Percentage of truly suspicious transactions correctly identified. Target: >= 95% (current rule system: ~90%)
- **False Positive Reduction:** Percentage reduction in false positives compared to the rule engine. Target: >= 60% (from 97.1% FP rate to <= 38.8%)
- **Precision at Recall 95%:** How precise the system is while maintaining 95% recall. Target: >= 15% (up from current 2.9%)

**Secondary metrics:**
- **Reasoning Quality Score:** Average LLM-as-judge score across the test set. Target: >= 3.5/5.0
- **Latency P95:** 95th percentile end-to-end processing time. Target: <= 25 seconds
- **Cost per Transaction:** Total LLM API cost per screened transaction. Target: <= USD 0.12

### Baseline

The current baseline is the rule-based system alone:
- SAR Capture Rate: ~90% (misses 8-12% of suspicious transactions)
- False Positive Rate: 97.1% (only 2.9% of flagged transactions result in SARs)
- Precision: 2.9%
- Cost per flagged transaction: USD 0 (rules are free to execute) + USD 25 in analyst time = USD 25
- Reasoning: None (rules produce trigger IDs, not explanations)

A naive LLM baseline (single zero-shot prompt asking "Is this transaction suspicious?") achieves approximately:
- SAR Capture Rate: ~75% (model lacks domain calibration)
- False Positive Rate: ~60% (better than rules, but still high)
- Precision: ~8%
- Reasoning: Unstructured, inconsistent quality

### Why Prompt Design Principles

The prompt design techniques from the article map directly to the failure modes of both baselines:

1. **System prompts** solve the role calibration problem. A well-designed system prompt establishes the LLM as a BSA/AML compliance analyst with specific regulatory knowledge, output format expectations, and risk tolerance calibration. Without this, the model defaults to generic text analysis.

2. **Few-shot learning** solves the domain calibration problem. By showing 3-5 examples of real SAR-filed transactions alongside cleared transactions, we sharpen the model's discriminative boundary for this specific compliance context. This is what takes recall from 75% (zero-shot) to 95%+ (few-shot).

3. **Chain-of-thought** solves the auditability problem. Regulators require explainable decisions. CoT forces the model to produce step-by-step reasoning that compliance officers can review and that examiners can audit. It also improves accuracy on complex multi-factor risk assessments.

4. **Structured output** solves the integration problem. The pipeline must feed into existing case management systems. Structured JSON output enables automated triage, dashboard population, and SAR pre-filling.

5. **Prompt chaining** solves the complexity problem. Transaction screening involves multiple analysis dimensions (sanctions check, typology matching, narrative analysis, risk scoring). No single prompt can reliably handle all of these. Chaining with gate checks decomposes the task and catches errors at each boundary.

### Technical Constraints

- **Model:** Claude 3.5 Sonnet or equivalent (balances cost, speed, and reasoning quality)
- **Max tokens per step:** 1,500 (controls cost; most reasoning chains complete within 800 tokens)
- **Inference latency budget:** 25 seconds total across all pipeline steps (approximately 6 seconds per step in a 4-step chain)
- **Training compute:** None required (prompt-based approach; calibration uses historical data for example selection)
- **Data volume:** 42,000 transactions per day through the LLM pipeline

---

## Section 3: Implementation Notebook Structure

### 3.1 Data Acquisition Strategy

We will use a synthetic dataset of transaction narratives modeled on real AML typologies published by FinCEN and the Financial Action Task Force (FATF). The dataset includes 5,000 transaction records: 500 suspicious (covering 8 common AML typologies) and 4,500 benign, with realistic narrative text, structured fields, and ground-truth labels.

**Why synthetic?** Real SAR data is confidential under 31 USC 5318(g)(2). However, FinCEN publishes detailed typology descriptions and anonymized case studies that we use to generate realistic synthetic narratives. The synthetic data preserves the statistical properties of real transaction data (amount distributions, country risk profiles, narrative patterns) without exposing protected information.

**Dataset structure:**
- `transactions.json`: 5,000 records with narrative text, structured fields, and labels
- `typologies.json`: 8 AML typologies with descriptions and example indicators
- `few_shot_examples.json`: 10 curated examples (5 suspicious, 5 benign) for few-shot prompts

**TODO:** Students load the dataset, inspect the schema, and verify class balance. Students implement a data augmentation function that generates additional narrative variations for underrepresented typologies.

### 3.2 Exploratory Data Analysis

Key distributions to analyze:
- Transaction amount distribution (log-scale) by suspicious vs. benign
- Country risk distribution (originator and beneficiary countries)
- Narrative length distribution
- Rule trigger frequency analysis
- Typology distribution among suspicious transactions

**TODO:** Students write EDA code to generate these visualizations and answer guided questions:
- What is the median transaction amount for suspicious vs. benign transactions?
- Which originator countries appear most frequently in suspicious transactions?
- Is there a correlation between narrative length and suspicion?
- Which rule triggers have the highest and lowest precision?

### 3.3 Baseline Approach

Implement two baselines:
1. **Rule-based baseline:** Apply a simplified rule set (amount thresholds, high-risk country lists, keyword matching) to the test set
2. **Zero-shot LLM baseline:** Send each transaction narrative to the LLM with a minimal prompt ("Is this transaction suspicious? Answer yes or no.")

Evaluate both baselines using the defined metrics (SAR capture rate, false positive rate, precision).

**TODO:** Students implement the rule-based baseline and the zero-shot LLM baseline, then compute and compare metrics.

### 3.4 Model Design — The Prompt Chain Architecture

The core architecture is a 4-step prompt chain with gate checks:

**Step 1 — Risk Factor Extraction (System Prompt + Structured Output):**
Design a system prompt that establishes the LLM as a BSA/AML compliance analyst. The prompt instructs the model to extract specific risk factors from the transaction narrative and structured fields, outputting a JSON object with identified risk indicators.

**Step 2 — Typology Matching (Few-Shot Learning):**
Using the extracted risk factors from Step 1, match against known AML typologies. The prompt includes 3-5 few-shot examples showing how risk factors map to specific typologies (structuring, layering, trade-based laundering, etc.).

**Step 3 — Risk Assessment (Chain-of-Thought):**
Given the risk factors and typology match, perform a step-by-step risk assessment. The CoT prompt forces the model to reason through each factor, weigh mitigating circumstances, and arrive at a risk level with explicit justification.

**Step 4 — Decision and SAR Recommendation (Structured Output):**
Synthesize all prior analysis into the final structured decision object, including the recommendation, confidence score, and regulatory citations.

**Gate checks** between each step validate output format and logical consistency.

**TODO:** Students implement the system prompt for Step 1, the few-shot prompt builder for Step 2, the CoT prompt for Step 3, and the synthesis prompt for Step 4. Each prompt function receives specific inputs and must produce outputs matching the expected schema. Students also implement the gate check functions.

### 3.5 Training Strategy

Since this is a prompt-based system (no model training), this section covers prompt calibration and optimization:

- **Example selection:** How to choose the best few-shot examples from the labeled dataset using embedding similarity
- **Prompt iteration:** Systematic A/B testing of prompt variants on a held-out validation set
- **Threshold calibration:** Finding the optimal confidence threshold for each recommendation level using the validation set
- **Cost optimization:** Measuring token usage per step and optimizing prompt length

**Why embedding-based example selection over random?** Random selection may include examples that are irrelevant to the current transaction. By computing embeddings of transaction narratives and selecting the $k$ most similar examples from the labeled set, we provide the LLM with maximally informative demonstrations. This is analogous to choosing a learning rate schedule in gradient-based training — it controls the quality of the optimization signal.

**TODO:** Students implement the embedding-based example selector, run prompt variants on the validation set, and plot performance curves for different numbers of few-shot examples (0, 1, 3, 5, 8).

### 3.6 Evaluation

Quantitative evaluation on the held-out test set (1,000 transactions):
- Compute SAR Capture Rate, False Positive Rate, and Precision for each pipeline variant
- Compare against both baselines (rule-based and zero-shot)
- Generate confusion matrices and precision-recall curves
- Evaluate reasoning quality using LLM-as-judge scoring

**TODO:** Students run the full pipeline on the test set, compute all metrics, generate comparison plots, and interpret the results. Students also implement the LLM-as-judge evaluator for reasoning quality.

### 3.7 Error Analysis

Systematic categorization of pipeline errors:
- **False Negatives:** Suspicious transactions the pipeline missed — what risk factors were not detected?
- **False Positives:** Benign transactions incorrectly flagged — what pattern caused the false alarm?
- **Gate Check Failures:** Transactions where intermediate steps produced malformed output — what prompt design issues caused the failure?
- **Reasoning Errors:** Transactions where the final decision was correct but the reasoning chain contained logical errors

**TODO:** Students sample 20 errors from each category, analyze the root causes, and propose prompt modifications to address the top 3 failure modes.

### 3.8 Scalability and Deployment Considerations

- Throughput profiling: measure transactions per minute at different concurrency levels
- Cost analysis: compute actual cost per transaction and project monthly costs at production volume
- Latency optimization: identify bottlenecks in the pipeline and explore caching strategies (e.g., caching few-shot example embeddings)
- Batching strategy: how to batch transactions for efficient API utilization

**TODO:** Students write a benchmarking script that measures throughput, latency distribution, and cost per transaction. Students also implement a simple caching layer for few-shot example retrieval.

### 3.9 Ethical and Regulatory Analysis

- **Bias in training examples:** Do the few-shot examples overrepresent certain countries, entity types, or transaction patterns? Could this create discriminatory screening behavior?
- **Regulatory compliance:** Does the pipeline meet OCC Model Risk Management (SR 11-7) requirements? What documentation is needed?
- **Transparency:** Can the reasoning traces be presented to examined parties during regulatory audits?
- **Fairness metrics:** Compute false positive rates across different originator country groups and entity types. Are there statistically significant disparities?

**TODO:** Students compute disaggregated false positive rates by country group and entity type, write a brief ethical impact assessment, and draft a model risk management summary addressing SR 11-7 requirements.

---

## Section 4: Production and System Design Extension

### Architecture Overview

The production system consists of five major components:

```
[Transaction Feed] → [Rule Engine] → [LLM Screening Pipeline] → [Case Management System] → [SAR Filing]
                                            |
                                     [Audit Log Store]
```

**Transaction Feed:** Receives real-time transaction data from client institutions via secure SFTP or API integration. Transactions are normalized into the standard schema and queued for processing.

**Rule Engine (existing):** Applies the 1,400-rule screening logic. Flagged transactions are sent to the LLM pipeline. Non-flagged transactions bypass the LLM (cost optimization — only screen pre-flagged transactions).

**LLM Screening Pipeline:** The 4-step prompt chain processes each flagged transaction. Runs on dedicated GPU infrastructure (for on-premises LLM) or through a BAA-covered API endpoint. Produces the structured decision object.

**Case Management System (existing):** Receives LLM decisions and routes to appropriate analyst queues. Transactions recommended for "clear" with high confidence are auto-processed (with random sampling for quality assurance). All others go to human analysts.

**Audit Log Store:** Every LLM call (prompt, response, latency, token count) is logged to an immutable audit store for regulatory examination.

### API Design

```
POST /api/v1/screen
```

Request:
```json
{
  "transaction_id": "TXN-2024-0847291",
  "narrative": "Wire transfer from...",
  "structured_fields": {
    "amount": 847000.00,
    "currency": "USD",
    "originator_country": "VG",
    "beneficiary_country": "US",
    "transaction_type": "wire",
    "originator_entity_type": "corporation",
    "rule_triggers": ["RULE-042", "RULE-117"]
  },
  "context": {
    "account_history_summary": "Account opened 6 months ago...",
    "customer_risk_rating": "high",
    "sanctions_screening_result": "clear"
  }
}
```

Response:
```json
{
  "transaction_id": "TXN-2024-0847291",
  "risk_assessment": "high",
  "recommendation": "file_sar",
  "confidence": 0.91,
  "reasoning": {
    "risk_factors_identified": [
      "Shell company jurisdiction (BVI)",
      "Vague transaction purpose",
      "Amount just below reporting threshold",
      "New account with large transaction"
    ],
    "typology_match": "layering_through_shell_companies",
    "mitigating_factors": ["Sanctions screening clear"],
    "reasoning_chain": "Step 1: The originator is a BVI-registered company..."
  },
  "regulatory_citations": ["31 CFR 1020.320", "FinCEN Advisory FIN-2019-A002"],
  "pipeline_metadata": {
    "total_latency_ms": 18420,
    "total_tokens": 3847,
    "cost_usd": 0.089,
    "steps_completed": 4,
    "gate_check_failures": 0
  }
}
```

### Serving Infrastructure

- **Primary:** Kubernetes cluster with 4 worker nodes, each running 2 inference containers
- **Scaling:** Horizontal Pod Autoscaler triggered at 70% CPU utilization and 500ms average latency
- **Queue:** Redis-backed task queue (Celery or BullMQ) for transaction buffering during peak hours
- **LLM Provider:** Claude 3.5 Sonnet via Anthropic API with BAA agreement, or on-premises vLLM deployment with Llama 3.1 70B for institutions requiring on-prem
- **Failover:** If the LLM pipeline times out or errors, the transaction is automatically routed to human review (fail-safe, not fail-silent)

### Latency Budget

| Component | Target P95 | Allocation |
|-----------|-----------|------------|
| Input validation and preprocessing | 50ms | 0.2% |
| Step 1: Risk Factor Extraction | 5,000ms | 20% |
| Gate Check 1 | 100ms | 0.4% |
| Step 2: Typology Matching | 5,000ms | 20% |
| Gate Check 2 | 100ms | 0.4% |
| Step 3: Risk Assessment (CoT) | 8,000ms | 32% |
| Gate Check 3 | 100ms | 0.4% |
| Step 4: Decision Synthesis | 4,000ms | 16% |
| Output formatting and logging | 150ms | 0.6% |
| **Total** | **22,500ms** | **~90%** |

Buffer of 2,500ms (10%) for retries and network variability.

### Monitoring

**Real-time dashboards:**
- Transaction throughput (screened/minute, by client institution)
- Latency distribution (P50, P95, P99 per pipeline step)
- Gate check failure rate (target: < 2%)
- LLM error rate (API timeouts, malformed responses)
- Cost tracking (daily spend, cost per transaction trend)

**Quality metrics (daily batch):**
- Recommendation distribution (file_sar / escalate / clear_with_note / clear)
- Confidence score distribution
- Reasoning quality scores (sampled LLM-as-judge evaluation)
- Agreement rate with human analyst decisions (for transactions that reach human review)

**Alerting thresholds:**
- Gate check failure rate > 5% for 15 minutes → page on-call engineer
- P95 latency > 30 seconds → scale up inference pods
- SAR recommendation rate deviates > 2x from 30-day moving average → alert compliance team
- LLM API error rate > 1% → switch to failover provider or route to human review

### Model Drift Detection

- **Feature drift:** Monitor embedding distributions of incoming transaction narratives. If the mean embedding distance from the training distribution exceeds a threshold (measured via Maximum Mean Discrepancy), trigger a prompt recalibration review.
- **Label drift:** Track the ratio of human analyst overrides (analyst disagrees with pipeline recommendation). If override rate exceeds 15% for any recommendation category, trigger few-shot example refresh.
- **Concept drift:** Quarterly review of new FinCEN advisories and typology updates. New typologies must be added to the typology matching examples within 30 days of publication.

### Model Versioning

- Each prompt configuration (system prompts, few-shot examples, gate check logic) is version-controlled in Git
- Prompt versions are tagged with semantic versioning (e.g., `prompt-v2.3.1`)
- Every LLM call in the audit log includes the prompt version hash
- Rollback: revert to previous prompt version via feature flag (< 5 minute rollback time)

### A/B Testing

- **Shadow mode:** New prompt versions run in parallel with the production version for 7 days. Both produce decisions, but only the production version's decision is surfaced to analysts.
- **Comparison metrics:** SAR capture rate, false positive rate, reasoning quality, cost per transaction
- **Statistical significance:** Require p < 0.01 (chi-squared test) on the primary metric (SAR capture rate) before promoting a new version
- **Guardrail metrics:** New version must not increase false negative rate by more than 0.5 percentage points, even if other metrics improve

### CI/CD for ML

- **Prompt testing:** Every prompt change triggers an automated evaluation on a fixed test set of 500 labeled transactions
- **Regression gates:** PR cannot merge if SAR capture rate drops below 93% or false positive rate increases by more than 5 percentage points on the test set
- **Staging environment:** New prompt versions deploy to staging first, processing a mirrored feed of real transactions for 48 hours before production promotion
- **Automated evaluation reports:** Each PR includes a generated comparison report showing metric changes against the current production version

### Cost Analysis

**Per-transaction cost breakdown:**
| Component | Tokens (avg) | Cost |
|-----------|-------------|------|
| Step 1: Risk Factor Extraction | ~800 input + 400 output | USD 0.018 |
| Step 2: Typology Matching | ~1,200 input + 300 output | USD 0.021 |
| Step 3: Risk Assessment (CoT) | ~900 input + 600 output | USD 0.024 |
| Step 4: Decision Synthesis | ~700 input + 500 output | USD 0.019 |
| Embedding computation | ~300 tokens | USD 0.001 |
| **Total per transaction** | **~4,700 tokens** | **USD 0.083** |

**Monthly projection:**
- 42,000 flagged transactions/day x 30 days = 1,260,000 transactions/month
- LLM cost: 1,260,000 x USD 0.083 = USD 104,580/month
- Infrastructure (Kubernetes, Redis, monitoring): ~USD 8,000/month
- **Total: ~USD 112,580/month**

**ROI calculation:**
- Current analyst cost for false positive review: 40,800 FP/day x USD 25 x 30 days = USD 30,600,000/month
- With 60% FP reduction: 16,320 FP/day x USD 25 x 30 days = USD 12,240,000/month
- Monthly savings: USD 18,360,000 - USD 112,580 = USD 18,247,420/month
- The pipeline pays for itself within the first day of operation
