# Case Study: Building a Hybrid Memory System for an AI Clinical Assistant

---

## Section 1: Industry Context and Business Problem

### Industry: Healthcare / Clinical AI

Healthcare is one of the highest-stakes environments for AI deployment. In the United States, over 1 billion outpatient visits occur annually, generating an enormous volume of clinical documentation -- progress notes, lab results, medication lists, allergy records, referral letters, and discharge summaries. A single patient with a chronic condition may accumulate hundreds of pages of medical history across dozens of providers over the course of a decade.

Physicians managing complex patients must hold a staggering amount of context in their heads: current medications, past adverse reactions, prior diagnoses that were ruled out, family history, social determinants, and the patient's own preferences and concerns. Cognitive overload is a well-documented contributor to medical errors -- the Institute of Medicine estimates that preventable adverse drug events alone affect 1.5 million Americans per year, with an annual cost exceeding \$3.5 billion.

This is exactly the kind of problem where AI assistants with robust memory could transform care quality -- if they can be built correctly.

### Company Profile: MedFlow AI

**MedFlow AI** is a Series A health technology startup headquartered in San Francisco, California. Founded in 2023 by a former chief medical informatics officer, a clinical NLP researcher from Stanford, and a senior engineer from a major EHR vendor, MedFlow AI builds an AI-powered clinical assistant designed for outpatient specialty clinics (cardiology, endocrinology, oncology, and rheumatology).

- **Founded**: 2023
- **Employees**: ~60 (22 engineers, 8 clinical informatics specialists, 5 compliance/regulatory staff, remainder in sales, ops, and support)
- **Funding**: \$28M raised across Seed and Series A rounds. Most recent round led by a top digital health fund at a \$120M valuation
- **Product**: An AI assistant embedded within the EHR workflow that helps physicians prepare for patient visits by synthesizing the patient's full medical history, flagging relevant clinical context, and supporting real-time clinical decision-making during multi-session consultations
- **Customers**: 34 specialty clinics across 9 U.S. states, representing approximately 1,800 active physician users
- **Revenue**: \$4.1M ARR, growing 22% quarter-over-quarter
- **Key Product Lines**:
  - **MedFlow Prep** -- Pre-visit patient summary generation that synthesizes the last 6-12 months of clinical activity into a structured briefing
  - **MedFlow Live** -- Real-time conversational assistant that physicians can query during patient encounters ("What was the last A1C?" or "Has this patient ever had a reaction to metformin?")
  - **MedFlow Follow-Up** -- Post-visit follow-up planning that remembers what was discussed, what was ordered, and what needs to happen next

### Business Challenge

MedFlow AI's product has strong early traction -- physicians love the concept. But a critical, dangerous problem has emerged as usage has scaled: **the AI forgets.**

The original system was built in early 2024 using a straightforward sliding window memory architecture: the most recent conversation turns are passed to the LLM as context, supplemented by a one-paragraph patient summary pulled from the EHR at the start of each session. This was sufficient for simple, single-session interactions during the MVP phase.

But as physicians began using MedFlow Live for multi-session follow-up care with complex patients, four critical failure patterns surfaced:

1. **Allergy amnesia.** A patient told her endocrinologist during Session 1 that she had a severe allergic reaction to sulfonamide antibiotics. In Session 3 (two weeks later), the physician asked MedFlow for medication suggestions for a urinary tract infection. MedFlow recommended trimethoprim-sulfamethoxazole -- a sulfonamide. The physician caught the error, but filed an incident report. The allergy had been mentioned in Session 1, which was no longer in the sliding window by Session 3.

2. **Medication history loss.** A cardiology patient was being titrated on a beta-blocker over four visits. At each visit, the dosage was adjusted based on the patient's blood pressure readings and side effects. By Visit 4, the AI had no memory of the Visit 1 and Visit 2 dosage decisions. When the physician asked "What dosage were we on two visits ago?", the AI responded with "I do not have information about previous dosage adjustments." The physician had to manually review the chart.

3. **Diagnostic context evaporation.** In oncology, treatment decisions depend heavily on the diagnostic journey -- what tests were run, what was ruled out, and why a particular treatment path was chosen. A patient's AI assistant lost the context that a genetic test (BRCA2) had come back positive three sessions ago, failing to flag this when the physician was discussing prophylactic surgery options.

4. **Contradictory recommendations across sessions.** Because each session starts with a near-blank memory, the AI sometimes offers advice in Session 4 that contradicts what it recommended in Session 2. A rheumatology patient was told in one session that exercise would help manage her fibromyalgia symptoms. Two sessions later, a different framing of the same question led the AI to suggest rest and activity limitation.

### Why It Matters

The consequences span patient safety, financial viability, and regulatory compliance:

- **Patient safety risk**: Medication errors due to forgotten allergies or drug interactions are among the most dangerous failure modes in clinical care. The Joint Commission identified communication failures as the leading root cause of sentinel events (the most serious category of patient harm). An AI assistant that forgets critical patient information is not merely unhelpful -- it is a liability.
- **Physician trust erosion**: Physicians reported in internal surveys that they check MedFlow's accuracy on "at least half" of critical queries. Trust scores dropped from 4.2/5.0 to 3.1/5.0 over a 90-day period as memory failures accumulated. Three large customer accounts (representing \$620K in combined ARR) have requested a formal remediation plan.
- **Regulatory exposure**: Under HIPAA, clinical decision support tools that influence care decisions are subject to regulatory scrutiny. The FDA's guidance on Clinical Decision Support (CDS) software specifies that tools which provide patient-specific recommendations based on medical history must demonstrate reliability. MedFlow's memory failures could trigger an FDA review if they contribute to an adverse event.
- **Competitive pressure**: Two well-funded competitors (one backed by a major EHR vendor, another by a large health system) are building memory-enabled clinical assistants. MedFlow's differentiation depends on solving this problem first.

### Constraints

The engineering team must operate within these real-world constraints:

- **Compute budget**: \$35K/month for cloud GPU inference (currently running Anthropic Claude API plus self-hosted embedding models on AWS in a HIPAA-compliant environment)
- **Latency**: Physicians querying MedFlow Live during patient encounters expect responses within 5 seconds. Pre-visit summaries (MedFlow Prep) can tolerate up to 30 seconds. Any architectural change must maintain these latency targets
- **Data compliance**: All patient data is Protected Health Information (PHI) under HIPAA. MedFlow operates under a Business Associate Agreement (BAA) with each clinic. All data must remain within HIPAA-compliant infrastructure (AWS GovCloud). No patient data may transit through non-BAA-covered services. All data at rest and in transit must be encrypted (AES-256 and TLS 1.3)
- **Data volume**: Across all customers, MedFlow stores approximately 840,000 patient session transcripts, 2.1 million clinical notes, and 380,000 lab results. The vector index currently holds 5.2 million embeddings
- **Team**: 6 ML engineers, 3 infrastructure engineers, 4 clinical informatics specialists embedded in the ML team, 2 security/compliance engineers
- **Model access**: MedFlow uses Claude 3.5 Sonnet via API (with BAA) for generation and a fine-tuned BGE-large-en-v1.5 model for medical text embeddings
- **Session characteristics**: The average patient has 4.2 sessions over a 6-month period. Each session produces 800-2,000 tokens of conversation. Multi-session memory must span weeks to months

---

## Section 2: Technical Problem Formulation

### Problem Type: Multi-Session Memory-Augmented Conversational AI

This is fundamentally a **multi-session conversational memory** problem. The AI must maintain accurate, accessible, and up-to-date knowledge about each patient across an indefinite number of clinical sessions, and retrieve the right subset of that knowledge in response to physician queries during live encounters.

Why not a simpler framing?

- **Pure sliding window** (keeping the last K turns) fails because critical patient information is stated once and must be remembered indefinitely. Allergies mentioned in Session 1 are just as critical in Session 10.
- **Pure RAG over clinical notes** (retrieving from EHR documents) is insufficient because much of the relevant context comes from the conversational interaction itself -- what the physician discussed, what was decided, what the patient expressed as preferences -- not just from structured clinical data.
- **Simple entity extraction** alone misses temporal relationships between clinical events. Knowing that a medication was started is not enough -- you need to know when, at what dosage, what it was changed to, and why.

A hybrid memory architecture is the right framing because it combines multiple memory types -- each capturing a different dimension of clinical knowledge -- to provide comprehensive, reliable context for every physician query.

### Input Specification

The system receives four categories of input at each interaction:

1. **Current session messages** (streaming, 50-200 tokens per turn): The real-time conversation between the physician and MedFlow during a patient encounter. This includes physician queries ("What medications is this patient on?") and the system's responses.

2. **Patient clinical record** (structured data, variable size): Demographics, problem list, medication list, allergy list, lab results, imaging reports, and procedure history pulled from the EHR via FHIR API. Typically 500-5,000 tokens when serialized.

3. **Prior session transcripts** (historical text, variable size): Full transcripts of all previous MedFlow sessions for this patient. A patient with 6 prior sessions might have 6,000-15,000 tokens of historical conversation.

4. **Extracted entities and facts** (structured, accumulated): Key clinical facts extracted from prior sessions -- medications, allergies, diagnoses, decisions, and their timestamps. Stored in a structured entity store.

Each input carries distinct clinical information:
- Current session messages provide **immediate context** -- what the physician is asking right now
- The clinical record provides **ground truth** medical facts from the EHR
- Prior session transcripts provide **conversational context** -- what was discussed, decided, and planned
- Extracted entities provide **distilled facts** -- the most critical information in a compact, structured format

### Output Specification

The system produces two categories of output:

1. **Real-time query responses** (MedFlow Live): Direct answers to physician questions during encounters. Must be accurate, concise (50-300 tokens), and cite the source of information (which session, which clinical record).

2. **Pre-visit summaries** (MedFlow Prep): Structured briefings generated before a scheduled visit. Includes: active problem list, current medications with change history, recent lab trends, outstanding follow-up items from prior sessions, and patient-reported concerns.

Outputs are constrained by:
- **Factual accuracy**: Every clinical fact stated must be traceable to a source (EHR record or prior session). Hallucinated clinical information is a patient safety hazard.
- **Temporal consistency**: The system must correctly represent the chronological sequence of clinical events. If a medication was started, then dosage-adjusted, then discontinued, all three events and their dates must be correct.
- **Completeness for safety-critical information**: Allergies, active medications, and known adverse reactions must ALWAYS be surfaced when relevant, regardless of how long ago they were recorded.
- **Recency awareness**: When information conflicts across sessions (e.g., a medication dosage changed), the most recent value must take precedence, with change history available on request.

### Mathematical Foundation

The core mathematical machinery combines embedding-based semantic retrieval, structured entity scoring, and a memory fusion mechanism that weights contributions from different memory subsystems.

**Embedding and Cosine Similarity**

Each message or clinical note is mapped to a dense vector via an embedding model $f_\theta$:

$$f_\theta(x) = \mathbf{v} \in \mathbb{R}^d$$

Retrieval uses cosine similarity to find the most relevant historical messages:

$$\text{sim}(\mathbf{q}, \mathbf{m}_i) = \frac{\mathbf{q} \cdot \mathbf{m}_i}{\|\mathbf{q}\| \|\mathbf{m}_i\|}$$

For a physician query embedding $\mathbf{q}$ and a store of $N$ memory embeddings $\{\mathbf{m}_1, \ldots, \mathbf{m}_N\}$, we retrieve the top-$k$ most similar:

$$\mathcal{M}_{\text{semantic}} = \text{top-}k \left( \{ \text{sim}(\mathbf{q}, \mathbf{m}_i) \}_{i=1}^{N} \right)$$

**TF-IDF for Clinical Term Matching**

Clinical text contains domain-specific terms (drug names, procedure codes, diagnostic labels) where exact keyword matching outperforms semantic similarity. We compute TF-IDF scores as a complementary retrieval signal:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

where:

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}, \quad \text{IDF}(t, D) = \log \frac{|D|}{1 + |\{d \in D : t \in d\}|}$$

Here $f_{t,d}$ is the frequency of term $t$ in document $d$, and $|D|$ is the total number of documents. TF-IDF gives high weight to terms that are frequent within a specific document but rare across the corpus -- exactly what we want for identifying sessions where a specific drug or diagnosis was discussed.

**Hybrid Memory Scoring**

Each memory candidate $m_i$ receives a composite score that blends semantic similarity, keyword relevance, recency, and clinical importance:

$$S(m_i, q) = \alpha \cdot \text{sim}(\mathbf{q}, \mathbf{m}_i) + \beta \cdot \text{TF-IDF}(q, m_i) + \gamma \cdot \text{Recency}(m_i) + \delta \cdot \text{Importance}(m_i)$$

where:

- $\text{Recency}(m_i) = \exp(-\lambda \cdot \Delta t_i)$ is an exponential decay function with $\Delta t_i$ being the time (in days) since memory $m_i$ was created, and $\lambda$ is the decay rate
- $\text{Importance}(m_i) \in \{0.0, 0.5, 1.0\}$ is a categorical weight assigned based on clinical criticality (1.0 for allergies/adverse reactions, 0.5 for active medications/diagnoses, 0.0 for general conversation)
- $\alpha + \beta + \gamma + \delta = 1$ are tunable weights (empirically: $\alpha = 0.4, \beta = 0.2, \gamma = 0.15, \delta = 0.25$)

The importance weight is critical: it ensures that safety-relevant memories (allergies, adverse reactions) always receive a score boost, even when they are old and semantically distant from the current query.

**Memory Staleness Detection**

Clinical information can become stale. A lab result from 6 months ago may no longer reflect the patient's current state. We define a staleness score:

$$\text{Staleness}(m_i) = 1 - \text{Recency}(m_i) = 1 - \exp(-\lambda \cdot \Delta t_i)$$

Memories with staleness above a threshold $\tau_s$ are flagged in the context with a warning: "[NOTE: This information is from X days ago and may be outdated.]"

For clinical data, the staleness threshold varies by type:
- Allergies: $\tau_s = \infty$ (never stale -- allergies persist indefinitely)
- Active medications: $\tau_s = 90$ days
- Lab results: $\tau_s = 30$ days
- Vital signs: $\tau_s = 7$ days

### Loss Function and Evaluation

The generation model (Claude 3.5 Sonnet) is used via API, so we do not train it directly. However, two components are trained:

**1. Embedding Model Fine-tuning Loss**

The BGE-large embedding model is fine-tuned on clinical text pairs using InfoNCE (contrastive) loss:

$$\mathcal{L}_{\text{embed}} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{d}^+) / \tau)}{\exp(\text{sim}(\mathbf{q}, \mathbf{d}^+) / \tau) + \sum_{j=1}^{N^-} \exp(\text{sim}(\mathbf{q}, \mathbf{d}_j^-) / \tau)}$$

where $\mathbf{d}^+$ is a clinically relevant passage, $\mathbf{d}_j^-$ are irrelevant passages, and $\tau = 0.07$ is the temperature.

**2. Entity Extraction Accuracy**

Entity extraction quality is measured by precision, recall, and F1 on a held-out set of expert-annotated clinical conversations:

$$\text{Precision} = \frac{|\text{Extracted} \cap \text{Gold}|}{|\text{Extracted}|}, \quad \text{Recall} = \frac{|\text{Extracted} \cap \text{Gold}|}{|\text{Gold}|}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**3. End-to-End Memory Recall Accuracy**

The critical metric: when a physician asks a question whose answer exists in a prior session, does the system retrieve and correctly present the information?

$$\text{Memory Recall Accuracy} = \frac{\text{Queries where correct memory was retrieved and used}}{\text{Total queries requiring historical information}}$$

**4. Factual Consistency Score**

We measure whether the system's responses are consistent with the ground truth in the EHR and prior sessions. A response is factually inconsistent if it states a clinical fact that contradicts the source data (wrong dosage, wrong date, wrong allergy).

### Evaluation Metrics

**Primary Metrics:**

| Metric | Definition | Target |
|--------|-----------|--------|
| Memory Recall Accuracy | % of history-dependent queries answered correctly | > 92% |
| Factual Consistency | % of clinical facts in responses that match source data | > 97% |
| Allergy/Safety Recall | % of safety-critical facts surfaced when relevant | > 99% |

**Secondary Metrics:**

| Metric | Definition | Target |
|--------|-----------|--------|
| Query Latency (Live) | Time from physician query to response delivery | < 5 seconds |
| Summary Generation Time | Time to generate a pre-visit summary | < 30 seconds |
| Entity Extraction F1 | F1 score on clinical entity extraction | > 0.88 |
| Memory Staleness Rate | % of surfaced memories that are outdated without flagging | < 3% |
| Context Token Efficiency | % of context window tokens that contribute to the response | > 65% |

### Baseline: Sliding Window + EHR Snapshot

Without a hybrid memory system, the simplest approach is:
1. Keep the last 10 conversation turns in a sliding window (approximately 2,000-4,000 tokens)
2. Pull a static patient summary from the EHR at session start (500-1,000 tokens)
3. Concatenate both into the context window
4. Use a generic clinical assistant prompt

This baseline achieves:
- Memory Recall Accuracy: ~38% (most prior-session information is lost)
- Factual Consistency: ~81% (the EHR snapshot provides some ground truth, but the model sometimes hallucinates when asked about undocumented clinical details)
- Allergy/Safety Recall: ~52% (allergies mentioned only in conversation, not in the structured EHR allergy list, are forgotten)
- Query Latency: 2.1 seconds (fast, because minimal context is assembled)

The baseline fails for three fundamental reasons: (1) the sliding window discards all information beyond the last 10 turns, (2) the EHR snapshot is static and does not include conversational context from prior sessions, and (3) there is no mechanism to identify which historical information is relevant to the current query.

### Why a Hybrid Memory Architecture

A hybrid memory architecture is the right approach because no single memory type captures all dimensions of clinical knowledge:

1. **Sliding window** preserves immediate conversational flow -- the physician's current line of questioning and the assistant's recent responses
2. **Vector-backed long-term memory** enables semantic retrieval of relevant past conversations -- finding the session where a specific medication was discussed, even months later
3. **Entity store** captures structured clinical facts (allergies, medications, diagnoses) with high precision and zero tolerance for loss
4. **Running summary** provides the "big picture" of the patient's care trajectory -- what happened overall, without requiring retrieval

Each memory type compensates for the others' weaknesses. The sliding window cannot remember Session 1 from Session 5, but the entity store can. The entity store cannot capture the nuance of why a treatment decision was made, but the vector store can retrieve the relevant conversation. The vector store might miss an allergy if the query is not semantically similar to the allergy mention, but the entity store guarantees its presence.

### Technical Constraints

- **Model**: Claude 3.5 Sonnet (200K context window), accessed via Anthropic API with BAA
- **Embedding model**: Fine-tuned BGE-large-en-v1.5 (1024-dimensional embeddings, ~335M parameters)
- **Vector index**: 5.2 million embeddings in FAISS (IVF-PQ index) on AWS, with per-patient partitioning
- **Inference latency budget**: < 5 seconds for live queries (embedding < 100ms, retrieval < 200ms, generation < 4.5s)
- **Training compute**: 2x A100 GPUs for embedding fine-tuning (not for LLM training)
- **Context token budget**: 200K total, 4K reserved for response, remainder available for context assembly
- **HIPAA compliance**: All vector stores, entity stores, and session transcripts encrypted at rest (AES-256) and in transit (TLS 1.3)

---

## Section 3: Implementation Notebook Structure

This section defines the structure of the Google Colab notebook that students will implement. The notebook builds a complete hybrid memory system for clinical conversations, progressing from data exploration through a fully evaluated multi-session memory pipeline.

### 3.1 Data Acquisition Strategy

**Dataset**: MedQuAD (Medical Question Answering Dataset) + Synthetic Clinical Sessions

MedQuAD is a publicly available medical question-answering dataset containing over 47,000 QA pairs sourced from trusted medical institutions (NIH, NLM, CDC). It covers diseases, medications, treatments, side effects, and risk factors. For this case study, we supplement MedQuAD with synthetic multi-session clinical conversations that simulate realistic patient follow-up scenarios.

We use this combination because:
- MedQuAD provides real, medically accurate content (drug names, disease descriptions, treatment protocols)
- Synthetic sessions let us control the ground truth for evaluation -- we know exactly what facts were stated in which session
- The combination is small enough for Colab but realistic enough to expose real memory architecture challenges

**Data loading pipeline:**
1. Load a curated subset of MedQuAD focusing on cardiology, endocrinology, and oncology
2. Generate synthetic multi-session patient conversations using the MedQuAD content as a knowledge base
3. Create ground-truth entity annotations for each session
4. Build evaluation queries that require cross-session memory retrieval

**TODO (Student):**
- Implement the session generator that creates realistic multi-turn clinical conversations
- Implement entity annotation extraction from generated sessions
- Validate that each patient has at least 3 sessions with cross-session information dependencies

### 3.2 Exploratory Data Analysis

Students will explore the dataset to understand its structure and identify challenges for memory retrieval:

- **Distribution of session lengths** (tokens per session) -- how much conversation context does each session generate?
- **Entity frequency distribution** -- which clinical entities (medications, allergies, diagnoses) appear most frequently?
- **Cross-session dependency rate** -- what percentage of queries in later sessions require information from earlier sessions?
- **Entity type breakdown** -- what proportion of entities are safety-critical (allergies, adverse reactions) vs. informational (preferences, social history)?
- **Temporal spacing** -- how are sessions distributed over time for a typical patient?

**TODO (Student):**
- Plot the distribution of session lengths (in tokens) across all patients
- Create a bar chart showing entity type frequencies (medication, allergy, diagnosis, lab result, preference)
- Compute the cross-session dependency rate: what fraction of Session N queries require facts from Session 1 through Session N-1?
- Identify the most common medications and diagnoses in the dataset
- Answer: If we use a sliding window of K=10 turns, what percentage of cross-session facts would be lost by Session 3?

### 3.3 Baseline Approach

Implement a sliding window + static summary baseline to establish a performance floor.

**Implementation:**
- Use a deque-based sliding window that retains the last K conversation turns
- Pull a fixed patient summary at session start (simulated EHR snapshot)
- Concatenate window + summary and send to the LLM
- Evaluate on cross-session memory queries

**TODO (Student):**
- Implement the `SlidingWindowBaseline` class with configurable window size K
- Write a function that assembles the baseline context (summary + recent turns)
- Evaluate the baseline on 30 cross-session queries using Memory Recall Accuracy
- Measure Factual Consistency on the baseline's responses
- Answer: At what window size K does the baseline first achieve > 60% Memory Recall Accuracy? What is the token cost at that K?

### 3.4 Model Design: Hybrid Memory Pipeline

This is the core implementation section. Students will build a hybrid memory system with four subsystems, directly applying the memory architectures from the article.

**Architecture Overview:**

The pipeline consists of four memory subsystems and an orchestration layer:

1. **Sliding Window Memory** -- Last K turns of the current session, verbatim
2. **Vector-Backed Long-Term Memory** -- All prior session messages embedded and indexed in FAISS for semantic retrieval
3. **Entity Store** -- Structured dictionary of clinical facts (allergies, medications, diagnoses) extracted from every session
4. **Running Summary** -- Compressed summary of the patient's full clinical trajectory, updated after each session

The orchestration layer queries all four subsystems, scores and ranks the results using the hybrid scoring function, and assembles the final context with clear section boundaries.

**Subsystem 1: Sliding Window Memory**

**TODO (Student):**
- Implement `SlidingWindowMemory` with configurable window size
- The implementation must match the article's `SlidingWindowMemory` class structure

```python
from collections import deque
from typing import List, Dict

class SlidingWindowMemory:
    """Maintains the last K conversation turns for immediate context.

    Args:
        window_size: Number of turn pairs (user + assistant) to retain

    Hints:
        1. Use a deque with maxlen = window_size * 2 (user + assistant messages)
        2. add_message appends a role/content dict to the deque
        3. get_context joins all messages with role prefixes
        4. get_token_count estimates tokens using len(text) // 4
    """

    def __init__(self, window_size: int = 10):
        # TODO: Implement
        pass

    def add_message(self, role: str, content: str):
        """Add a message to the sliding window.

        Args:
            role: 'physician' or 'assistant'
            content: The message text
        """
        # TODO: Implement
        pass

    def get_context(self) -> str:
        """Return all messages in the window as a formatted string."""
        # TODO: Implement
        pass

    def get_token_count(self) -> int:
        """Estimate the token count of the current window contents."""
        # TODO: Implement
        pass

# Verification cell
sw = SlidingWindowMemory(window_size=3)
sw.add_message("physician", "What medications is the patient on?")
sw.add_message("assistant", "The patient is currently on metformin 500mg twice daily.")
sw.add_message("physician", "Any allergies?")
sw.add_message("assistant", "The patient has a documented allergy to penicillin.")
sw.add_message("physician", "What about sulfa drugs?")
sw.add_message("assistant", "No documented sulfa drug allergies.")
# With window_size=3, all 6 messages (3 pairs) should be retained
context = sw.get_context()
assert context is not None, "get_context returned None"
assert "metformin" in context, "Expected metformin in context"
assert "penicillin" in context, "Expected penicillin in context"
print(f"Window context ({sw.get_token_count()} estimated tokens):")
print(context)
```

**Subsystem 2: Vector-Backed Long-Term Memory**

**TODO (Student):**
- Implement `VectorMemory` using FAISS for similarity search
- Must support adding messages with metadata (session_id, timestamp, importance)
- Must support retrieval by cosine similarity with top-k results

```python
import numpy as np

class VectorMemory:
    """Stores all conversation messages as embeddings for semantic retrieval.

    Args:
        embedding_fn: Function that maps text -> numpy array of shape (dim,)
        dim: Embedding dimension (default 384 for all-MiniLM-L6-v2)
        top_k: Number of results to retrieve

    Hints:
        1. Use faiss.IndexFlatIP for inner product (cosine sim on normalized vectors)
        2. Normalize vectors before adding to index: vec / np.linalg.norm(vec)
        3. Store message metadata in a parallel list (same index as FAISS)
        4. In retrieve(), normalize the query vector, search, and return messages
           with their scores
        5. Filter out results with negative indices (FAISS returns -1 for
           unfilled slots)
    """

    def __init__(self, embedding_fn, dim: int = 384, top_k: int = 5):
        # TODO: Implement -- create FAISS index and message storage
        pass

    def add_message(self, role: str, content: str, metadata: dict = None):
        """Embed and store a message.

        Args:
            role: 'physician' or 'assistant'
            content: The message text
            metadata: Optional dict with 'session_id', 'timestamp', 'importance'
        """
        # TODO: Implement
        pass

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve the top-k most similar messages to the query.

        Args:
            query: The physician's current question
            top_k: Override default top_k if provided

        Returns:
            List of dicts with keys: 'role', 'content', 'score', 'metadata'
        """
        # TODO: Implement
        pass

    def get_store_size(self) -> int:
        """Return the number of messages in the store."""
        # TODO: Implement
        pass

# Verification cell
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

vm = VectorMemory(embedding_fn=lambda t: model.encode(t), dim=384, top_k=3)
vm.add_message("physician", "Patient has a severe allergy to penicillin",
               metadata={"session_id": 1, "importance": 1.0})
vm.add_message("assistant", "Noted. I will flag penicillin allergy for all future sessions.",
               metadata={"session_id": 1, "importance": 1.0})
vm.add_message("physician", "Start metformin 500mg twice daily for diabetes management",
               metadata={"session_id": 1, "importance": 0.5})
vm.add_message("physician", "How is the patient responding to the new exercise regimen?",
               metadata={"session_id": 2, "importance": 0.0})

results = vm.retrieve("Does this patient have any drug allergies?")
assert results is not None, "retrieve returned None"
assert len(results) > 0, "retrieve returned empty list"
assert any("penicillin" in r["content"].lower() for r in results), \
    "Expected penicillin allergy to be retrieved for allergy query"
print(f"Retrieved {len(results)} memories for allergy query:")
for r in results:
    print(f"  [{r['score']:.3f}] {r['role']}: {r['content'][:80]}...")
```

**Subsystem 3: Entity Store**

**TODO (Student):**
- Implement `EntityStore` that extracts and stores structured clinical facts
- Must categorize entities by type: allergy, medication, diagnosis, lab_result, preference
- Must assign importance scores based on clinical criticality
- Must support lookup by entity type and by keyword

```python
from dataclasses import dataclass, field
from typing import Optional
import json

@dataclass
class ClinicalEntity:
    name: str
    entity_type: str  # 'allergy', 'medication', 'diagnosis', 'lab_result', 'preference'
    value: str  # e.g., "500mg twice daily" for medications, "positive" for lab results
    session_id: int
    timestamp: str
    importance: float  # 1.0 for safety-critical, 0.5 for active clinical, 0.0 for informational
    is_active: bool = True
    notes: str = ""

class EntityStore:
    """Structured store for clinical entities extracted from conversations.

    Hints:
        1. Store entities in a dict keyed by (entity_type, name) for fast lookup
        2. When updating an existing entity, preserve the history (append to notes)
        3. Importance scores: allergy=1.0, medication=0.5, diagnosis=0.5,
           lab_result=0.5, preference=0.0
        4. get_by_type should return all entities of a given type
        5. get_safety_critical should return all entities with importance >= 1.0
        6. format_for_context should produce a compact string representation
    """

    def __init__(self):
        # TODO: Implement -- initialize entity storage
        pass

    def add_entity(self, entity: ClinicalEntity):
        """Add or update a clinical entity.

        If an entity with the same (type, name) exists, update its value
        and append a note about the change with timestamp.
        """
        # TODO: Implement
        pass

    def extract_entities(self, text: str, session_id: int, timestamp: str) -> List[ClinicalEntity]:
        """Extract clinical entities from conversation text using pattern matching.

        Args:
            text: The conversation text to extract from
            session_id: Which session this text is from
            timestamp: ISO timestamp of the message

        Returns:
            List of extracted ClinicalEntity objects

        Hints:
            1. Look for allergy patterns: "allerg(y|ic) to DRUG_NAME"
            2. Look for medication patterns: "start(ed)? DRUG_NAME DOSAGE"
               or "taking DRUG_NAME" or "prescribe(d)? DRUG_NAME"
            3. Look for diagnosis patterns: "diagnos(ed|is) (with)? CONDITION"
            4. Look for lab results: "LAB_NAME (is|was|came back) VALUE"
            5. Use regex with case-insensitive matching
            6. Return empty list if no entities found (do not return None)
        """
        # TODO: Implement
        pass

    def get_by_type(self, entity_type: str) -> List[ClinicalEntity]:
        """Return all entities of a given type."""
        # TODO: Implement
        pass

    def get_safety_critical(self) -> List[ClinicalEntity]:
        """Return all entities with importance >= 1.0 (allergies, adverse reactions)."""
        # TODO: Implement
        pass

    def format_for_context(self, max_tokens: int = 500) -> str:
        """Format all entities as a compact string for context injection.

        Format:
        ## Active Allergies (SAFETY-CRITICAL)
        - Penicillin: severe reaction (Session 1, 2024-01-15)

        ## Active Medications
        - Metformin 500mg twice daily (Session 1, 2024-01-15)

        ## Active Diagnoses
        - Type 2 Diabetes (Session 1, 2024-01-15)
        """
        # TODO: Implement
        pass

# Verification cell
es = EntityStore()
entities = es.extract_entities(
    "Patient has a severe allergy to penicillin. Started metformin 500mg twice daily. "
    "Diagnosed with Type 2 Diabetes. A1C came back 7.2%.",
    session_id=1,
    timestamp="2024-01-15T10:00:00"
)
assert entities is not None, "extract_entities returned None"
assert len(entities) >= 3, f"Expected at least 3 entities, got {len(entities)}"
for e in entities:
    es.add_entity(e)
safety = es.get_safety_critical()
assert len(safety) >= 1, "Expected at least 1 safety-critical entity (penicillin allergy)"
print(f"Extracted {len(entities)} entities:")
for e in entities:
    print(f"  [{e.entity_type}] {e.name}: {e.value} (importance={e.importance})")
print(f"\nSafety-critical entities: {len(safety)}")
print(f"\nFormatted context:\n{es.format_for_context()}")
```

**Subsystem 4: Running Summary**

**TODO (Student):**
- Implement `RunningSummary` that maintains a compressed overview of the patient's clinical trajectory
- Must update after each session by incorporating new session content into the existing summary
- Must preserve safety-critical facts even during compression

```python
class RunningSummary:
    """Maintains a running summary of the patient's clinical trajectory.

    Args:
        max_summary_tokens: Maximum token budget for the summary

    Hints:
        1. Store the current summary as a string
        2. update_summary takes new session text and the current summary,
           and produces an updated summary that incorporates the new information
        3. For this exercise, implement summarization using extractive methods
           (selecting the most important sentences) rather than calling an LLM
        4. Prioritize sentences containing safety-critical keywords
           (allergy, reaction, contraindicated, adverse, warning)
        5. Use sentence scoring: safety keywords get +2.0, medication mentions
           get +1.0, diagnosis mentions get +1.0, other sentences get +0.5
    """

    def __init__(self, max_summary_tokens: int = 500):
        # TODO: Implement
        pass

    def update_summary(self, session_text: str, session_id: int):
        """Update the running summary with content from a new session.

        Args:
            session_text: Full text of the new session
            session_id: Session number for tracking
        """
        # TODO: Implement
        pass

    def get_summary(self) -> str:
        """Return the current summary."""
        # TODO: Implement
        pass

    def get_token_count(self) -> int:
        """Estimate the token count of the current summary."""
        # TODO: Implement
        pass

# Verification cell
rs = RunningSummary(max_summary_tokens=300)
rs.update_summary(
    "Session began with patient reporting increased fatigue. "
    "Patient has a severe allergy to penicillin -- documented and flagged. "
    "Started metformin 500mg twice daily for Type 2 Diabetes management. "
    "A1C was 7.2%, target is below 7.0%. "
    "Follow-up scheduled in 4 weeks to reassess blood glucose control.",
    session_id=1
)
summary = rs.get_summary()
assert summary is not None and len(summary) > 0, "Summary should not be empty"
assert "allergy" in summary.lower() or "penicillin" in summary.lower(), \
    "Summary should preserve allergy information"
print(f"Running summary ({rs.get_token_count()} tokens):")
print(summary)
```

**Orchestration: Hybrid Memory System**

**TODO (Student):**
- Implement `HybridMemorySystem` that orchestrates all four subsystems
- Must implement the hybrid scoring function from Section 2
- Must implement context assembly with clear section boundaries and token budgeting
- Must ensure safety-critical entities are ALWAYS included, regardless of relevance score

```python
class HybridMemorySystem:
    """Orchestrates four memory subsystems into a unified clinical memory.

    This is the central class that ties everything together. When a physician
    asks a question, this system:
    1. Checks the entity store for directly relevant clinical facts
    2. Queries the vector store for semantically similar past conversations
    3. Gets the current sliding window for immediate context
    4. Gets the running summary for big-picture trajectory
    5. Scores and ranks all retrieved memories using the hybrid scoring function
    6. Assembles the final context with clear section boundaries

    Args:
        embedding_fn: Function that maps text -> numpy array
        window_size: Sliding window size (turn pairs)
        vector_top_k: Number of vector retrieval results
        max_context_tokens: Total token budget for assembled context

    Hints:
        1. Initialize all four subsystems in __init__
        2. add_message should propagate to ALL subsystems
        3. In build_context:
           a. ALWAYS include safety-critical entities first (non-negotiable budget)
           b. Then add running summary
           c. Then add scored vector retrieval results
           d. Then add sliding window (most recent context)
           e. Track token usage and stop when budget is reached
        4. The hybrid scoring function should combine similarity, recency,
           and importance as defined in Section 2
        5. Use XML-style tags for section boundaries:
           <safety_alerts>...</safety_alerts>
           <patient_summary>...</patient_summary>
           <relevant_history>...</relevant_history>
           <recent_conversation>...</recent_conversation>
    """

    def __init__(self, embedding_fn, window_size=10, vector_top_k=5,
                 max_context_tokens=8000):
        # TODO: Implement -- initialize all four subsystems
        pass

    def add_message(self, role: str, content: str, session_id: int,
                    timestamp: str):
        """Add a message to all memory subsystems.

        Args:
            role: 'physician' or 'assistant'
            content: The message text
            session_id: Current session number
            timestamp: ISO timestamp
        """
        # TODO: Implement
        pass

    def end_session(self, session_id: int, session_text: str):
        """Called at the end of each session to update the running summary.

        Args:
            session_id: The session that just ended
            session_text: Full text of the session
        """
        # TODO: Implement
        pass

    def compute_hybrid_score(self, similarity: float, recency_days: float,
                              importance: float) -> float:
        """Compute the hybrid memory score from Section 2.

        S = alpha * similarity + beta * tfidf_score + gamma * recency + delta * importance

        For simplicity, we drop the TF-IDF term and use:
        S = 0.45 * similarity + 0.20 * exp(-0.01 * recency_days) + 0.35 * importance

        Args:
            similarity: Cosine similarity score [0, 1]
            recency_days: Days since the memory was created
            importance: Clinical importance [0.0, 0.5, 1.0]

        Returns:
            Hybrid score (higher is better)
        """
        # TODO: Implement
        pass

    def build_context(self, query: str, current_session_id: int) -> str:
        """Build the complete context for a physician query.

        Args:
            query: The physician's current question
            current_session_id: The current session number

        Returns:
            Assembled context string with XML-tagged sections
        """
        # TODO: Implement
        pass

# Verification cell
# This test simulates 3 sessions and checks cross-session memory
model = SentenceTransformer('all-MiniLM-L6-v2')
hm = HybridMemorySystem(
    embedding_fn=lambda t: model.encode(t),
    window_size=5,
    vector_top_k=5,
    max_context_tokens=4000
)

# Session 1: Initial visit
hm.add_message("physician", "Patient reports severe allergy to penicillin.",
               session_id=1, timestamp="2024-01-15T10:00:00")
hm.add_message("assistant", "Documented. Penicillin allergy flagged as safety-critical.",
               session_id=1, timestamp="2024-01-15T10:01:00")
hm.add_message("physician", "Start metformin 500mg twice daily for Type 2 Diabetes.",
               session_id=1, timestamp="2024-01-15T10:05:00")
hm.add_message("assistant", "Metformin 500mg BID initiated. Will monitor renal function.",
               session_id=1, timestamp="2024-01-15T10:06:00")
hm.end_session(1, "Patient has penicillin allergy. Started metformin 500mg for T2DM.")

# Session 2: Follow-up (2 weeks later)
hm.add_message("physician", "Patient reports GI upset with metformin. Reduce to 250mg.",
               session_id=2, timestamp="2024-01-29T10:00:00")
hm.add_message("assistant", "Metformin reduced from 500mg to 250mg due to GI side effects.",
               session_id=2, timestamp="2024-01-29T10:01:00")
hm.end_session(2, "Metformin reduced to 250mg due to GI upset.")

# Session 3: Follow-up (4 weeks later) -- test cross-session memory
# Query about allergy from Session 1
context = hm.build_context("Can we prescribe amoxicillin for the patient's infection?",
                           current_session_id=3)
assert context is not None, "build_context returned None"
assert "penicillin" in context.lower() or "allergy" in context.lower(), \
    "CRITICAL: Penicillin allergy from Session 1 must appear in Session 3 context"

# Query about medication history
context2 = hm.build_context("What was the original metformin dosage?",
                            current_session_id=3)
assert "500mg" in context2 or "500" in context2, \
    "Original metformin dosage from Session 1 should be retrievable"

print("All cross-session memory tests passed!")
print(f"\nContext for amoxicillin query (showing safety alert):")
print(context[:500])
```

### 3.5 Training Strategy

In this pipeline, we do not train the generation LLM. Instead, we focus on two trainable components:

**Embedding Model Fine-tuning:**

- **Model**: `BAAI/bge-large-en-v1.5` (335M parameters), but we use `all-MiniLM-L6-v2` (22M parameters) for Colab compatibility
- **Optimizer**: AdamW with weight decay 0.01
- **Learning rate**: 2e-5 with linear warmup (10% of steps) followed by cosine decay
- **Batch size**: 16 (with in-batch negatives for contrastive learning)
- **Loss**: InfoNCE with temperature $\tau = 0.07$
- **Training data**: Clinical query-passage pairs derived from the synthetic sessions (a physician question paired with the session passage that contains the answer is a positive pair)
- **Epochs**: 3

**Entity Extraction Fine-tuning:**

- **Approach**: Pattern-based extraction with handcrafted regex rules (no ML model to fine-tune for this exercise)
- **Evaluation**: Precision, Recall, F1 on a held-out set of 100 annotated conversations

**TODO (Student):**
- Implement the contrastive training loop for the embedding model
- Create positive pairs from the synthetic sessions (question -> answer passage)
- Create hard negatives (clinically similar but incorrect passages)
- Train for 3 epochs and plot the training loss curve
- Compare retrieval quality (Recall@5) before and after fine-tuning

### 3.6 Evaluation

**Quantitative Evaluation:**

Evaluate the complete hybrid memory system on a held-out test set of 30 cross-session queries with known ground truth answers.

**TODO (Student):**
- Compute Memory Recall Accuracy: for each query, check if the correct memory was retrieved and used
- Compute Factual Consistency: compare system responses against ground truth
- Compute Allergy/Safety Recall: for queries where allergies are relevant, check if they are surfaced
- Compare all metrics against the sliding window baseline from Section 3.3
- Create a summary table showing baseline vs. hybrid system performance
- Plot a bar chart comparing the two systems across all metrics

### 3.7 Error Analysis

**Systematic Error Categorization:**

After running the hybrid memory system on the test queries, categorize errors into four failure types:

1. **Amnesia errors**: The correct memory exists in the store but was not retrieved (retrieval failure)
2. **Staleness errors**: An outdated memory was surfaced without appropriate flagging (e.g., old medication dosage presented as current)
3. **Hallucination errors**: The system generated clinical information that does not exist in any memory (fabrication)
4. **Contradiction errors**: The system's response contradicts information in the memory stores

**TODO (Student):**
- Run the hybrid system on all 30 test queries and manually categorize any errors
- For each error type, compute: (a) frequency, (b) severity (1-3 scale: 1=inconvenience, 2=clinical risk, 3=patient safety hazard), (c) root cause
- Identify the top 2 failure modes and propose a specific fix for each
- Answer: Which failure mode is most dangerous in a clinical setting? Why?

### 3.8 Scalability and Deployment Considerations

**Latency Profiling:**

Profile the end-to-end pipeline to identify bottlenecks:

- Entity lookup: expected < 5ms
- Embedding query: expected < 100ms
- FAISS retrieval: expected < 50ms
- Hybrid scoring: expected < 10ms
- Context assembly: expected < 20ms
- LLM generation: expected < 4.5s

**TODO (Student):**
- Instrument each pipeline stage with timing measurements
- Run the pipeline on 10 queries and compute mean and P95 latency per stage
- Identify the top 2 latency bottlenecks
- Propose one optimization for each bottleneck
- Answer: If the vector store grows to 1 million memories, how would retrieval latency change? What index type would you switch to?

### 3.9 Ethical and Regulatory Analysis

**Clinical AI Ethics:**

Healthcare AI raises specific ethical and regulatory concerns:

- **Patient safety**: Memory failures in clinical AI can directly harm patients. An incorrect drug recommendation based on a forgotten allergy is a medical error, even if the physician catches it.
- **HIPAA compliance**: All patient memory data is PHI. The memory system must implement access controls (only the treating physician can access a patient's memory), audit logging (every memory access is recorded), and data retention policies (memories must be purged per clinic retention schedules).
- **Bias in clinical memory**: If the entity extraction system is better at recognizing medications common in one demographic group vs. another, memory quality will vary by patient population.
- **Informed consent**: Patients should be informed that an AI assistant is being used in their care and that their conversation data is being stored for memory purposes.
- **Right to deletion**: Under some state laws and future federal regulations, patients may have the right to request deletion of their AI memory data.

**TODO (Student):**
- Design an audit logging schema for the memory system: what events should be logged, what fields should each log entry contain, and how long should logs be retained?
- Analyze whether the entity extraction system has differential performance across different medical specialties. Test extraction on cardiology vs. oncology vs. endocrinology text.
- Write a 500-word ethical impact assessment addressing: (a) who benefits from this system, (b) who could be harmed, (c) what safeguards should be in place, and (d) whether this system should be deployed without a physician always reviewing AI recommendations

---

## Section 4: Production and System Design Extension

This section extends the implementation into a production system design. It is intended for advanced students who want to understand how the hybrid memory system would operate at scale in MedFlow AI's production environment.

### Architecture Diagram

The production system follows a layered architecture:

```
+---------------------------------------------------------------------+
|                         CLIENT LAYER                                 |
|  [Physician EHR Plugin] --- [API Gateway (Kong)] --- [Auth (Okta)]  |
+---------------------------------------------------------------------+
                               |
+---------------------------------------------------------------------+
|                     ORCHESTRATION LAYER                               |
|  [Clinical Memory Orchestrator Service]                              |
|     |-- Query Router (Live vs. Prep mode)                            |
|     |-- Memory Assembler                                             |
|     |-- Safety Validator                                             |
|     |-- Audit Logger                                                 |
+---------------------------------------------------------------------+
                               |
        +----------------------+----------------------+
        |                      |                      |
+----------------+   +-------------------+   +------------------+
| MEMORY LAYER   |   | GENERATION LAYER  |   | PERSISTENCE      |
|                |   |                   |   | LAYER            |
| [FAISS Vector  |   | [Claude API       |   | [PostgreSQL      |
|  Store]        |   |  (Generation)]    |   |  (Entities)]     |
| [Entity Store  |   | [Embedding        |   | [Redis           |
|  Service]      |   |  Service]         |   |  (Session Cache)]|
| [Summary       |   |                   |   | [S3 (Transcripts,|
|  Service]      |   |                   |   |  Audit Logs)]    |
+----------------+   +-------------------+   +------------------+
```

The orchestration layer is the heart of the memory system. The Clinical Memory Orchestrator routes queries to the appropriate memory subsystems, assembles context according to the hybrid scoring function, validates safety-critical information, and logs every access for HIPAA compliance.

### API Design

**REST API Endpoints:**

```
POST /api/v1/sessions/{patient_id}/messages
  Request:
    {
      "role": "physician",
      "content": "What medications is this patient currently on?",
      "session_id": "uuid"
    }
  Response (streaming, SSE):
    event: safety_check
    data: {"alerts": [{"type": "allergy", "drug": "penicillin", "severity": "severe"}]}

    event: chunk
    data: {"text": "The patient is currently on metformin 250mg..."}

    event: sources
    data: {"memories_used": [{"session": 2, "date": "2024-01-29", "relevance": 0.94}]}

    event: done
    data: {"response_id": "uuid", "latency_ms": 3200}

POST /api/v1/sessions/{patient_id}/end
  Request:
    {
      "session_id": "uuid",
      "session_transcript": "Full session text..."
    }
  Response:
    {
      "entities_extracted": 5,
      "summary_updated": true,
      "memories_stored": 12
    }

GET /api/v1/patients/{patient_id}/prep
  Query params: visit_date, specialty
  Response:
    {
      "summary": "Patient is a 58-year-old with T2DM...",
      "active_medications": [...],
      "allergies": [...],
      "pending_followups": [...],
      "recent_labs": [...],
      "last_visit_highlights": "..."
    }

GET /api/v1/patients/{patient_id}/entities
  Query params: type (allergy|medication|diagnosis|lab_result), active_only
  Response:
    {
      "entities": [
        {"name": "Penicillin", "type": "allergy", "value": "severe reaction",
         "session": 1, "date": "2024-01-15", "importance": 1.0}
      ]
    }

DELETE /api/v1/patients/{patient_id}/memory
  Requires: admin auth + patient consent documentation
  Response:
    {
      "deleted": true,
      "records_removed": 847,
      "audit_log_entry": "uuid"
    }
```

### HIPAA Compliance Architecture

HIPAA compliance is non-negotiable for MedFlow AI. The memory system must implement:

**1. Access Controls**
- Role-based access: only the treating physician and authorized clinical staff can access a patient's memory
- Per-patient access logging: every memory query generates an audit log entry with physician ID, patient ID, timestamp, query text, and memories accessed
- Break-the-glass protocol: emergency access by non-assigned physicians is allowed but generates a high-priority alert for compliance review

**2. Data Encryption**
- At rest: AES-256 encryption for all patient data in PostgreSQL, FAISS indices, S3 objects, and Redis cache
- In transit: TLS 1.3 for all API communication
- Key management: AWS KMS with per-customer encryption keys (customer-managed CMKs)
- Vector embeddings are treated as PHI (they can potentially be inverted to reconstruct text) and are encrypted with the same rigor as raw text

**3. Audit Logging**
- Every memory read, write, update, and delete operation is logged
- Audit logs are immutable (append-only S3 bucket with Object Lock)
- Log retention: 7 years (per HIPAA requirements)
- Log schema: `{timestamp, physician_id, patient_id, operation, query_text_hash, memories_accessed_ids, response_summary_hash, ip_address, session_id}`
- Quarterly audit reviews by compliance team

**4. Data Retention and Deletion**
- Default retention: patient memories are retained for 7 years after the last session
- Patient-initiated deletion: upon verified request, all memory data (vectors, entities, summaries, transcripts) are permanently deleted within 30 days
- Deletion verification: a post-deletion audit confirms that all data has been purged, including from backups and replicas

### Serving Infrastructure

- **Model serving**: Claude 3.5 Sonnet via Anthropic API with BAA (HIPAA-compliant). The embedding model is served via Triton Inference Server on HIPAA-compliant GPU instances in AWS GovCloud.
- **Embedding service**: 2x g5.xlarge instances (NVIDIA A10G) behind an ALB, serving the fine-tuned BGE-large model. Dynamic batching (max batch size 32, max latency 50ms).
- **Vector database**: Per-patient FAISS indices stored in encrypted S3, loaded into memory on-demand with LRU caching. Hot patients (recently active) are pre-loaded. Cold patients are loaded in < 500ms from S3.
- **Entity store**: PostgreSQL (RDS) with row-level security policies enforcing per-physician access control.
- **Summary store**: PostgreSQL with versioning (every summary update creates a new version; old versions are soft-deleted but retained for audit).
- **Scaling strategy**: Horizontal scaling of the embedding service based on request queue depth. The orchestrator is stateless and scales horizontally. The Claude API handles scaling automatically.
- **Auto-scaling triggers**: Scale up when P95 query latency exceeds 5 seconds or queue depth exceeds 20 requests. Scale down when utilization drops below 25% for 10 minutes.

### Latency Budget

End-to-end target for MedFlow Live: < 5 seconds (P95).

| Stage | Target (P50) | Target (P95) | Notes |
|-------|-------------|-------------|-------|
| Authentication + routing | 10ms | 30ms | Okta token validation, Kong routing |
| Entity store lookup | 3ms | 10ms | PostgreSQL indexed query |
| Embedding (query) | 15ms | 40ms | Single vector, GPU |
| FAISS retrieval | 20ms | 80ms | Per-patient index, pre-loaded |
| Hybrid scoring + ranking | 5ms | 15ms | CPU, NumPy operations |
| Context assembly | 15ms | 40ms | Token counting + formatting |
| LLM generation | 2,500ms | 4,200ms | Claude API, streaming |
| Safety validation | 10ms | 30ms | Entity cross-reference |
| Audit logging | 5ms | 15ms | Async write to S3 |
| **Total** | **2,583ms** | **4,460ms** | **Within 5s budget** |

The LLM generation stage dominates latency. Optimization strategies: (a) prompt caching for system prompts and common entity formats, (b) streaming responses so physicians see text appearing immediately, (c) pre-computing safety alerts before the LLM call so they display instantly.

### Monitoring

**Key Metrics Dashboard:**

- **Memory quality**: Memory Recall Accuracy (sampled daily from expert annotations), Allergy/Safety Recall (automated check on every query involving medication recommendations)
- **Clinical safety**: Safety alert trigger rate, false negative rate for allergy checks, medication interaction detection rate
- **Latency**: P50, P95, P99 for each pipeline stage. Alert if P95 > 5s for Live queries.
- **Error rates**: 5xx error rate, entity extraction failure rate, memory retrieval empty rate
- **Data health**: Vector store size per patient, entity store growth rate, summary staleness distribution
- **HIPAA compliance**: Audit log completeness (100% target), access anomaly detection, encryption status

**Alerting Thresholds:**

| Alert | Threshold | Severity |
|-------|-----------|----------|
| Safety recall < 95% | Rolling 1-hour window | P0 (immediate page) |
| P95 latency > 5s (Live) | 5-minute sustained | P1 |
| Factual consistency < 93% | Rolling 4-hour window | P1 |
| 5xx error rate > 1% | 5-minute window | P2 |
| Entity extraction F1 < 0.80 | Daily batch evaluation | P2 |
| Audit log gap detected | Real-time | P1 (HIPAA) |

### Model Drift Detection

Clinical memory systems face three types of drift:

1. **Vocabulary drift**: New medications, treatments, and clinical terminology enter usage over time. If the embedding model has not seen new drug names, retrieval quality degrades for recent medications.
   - **Detection**: Monthly comparison of entity extraction coverage on recent sessions vs. historical sessions. Alert if extraction F1 drops by > 5%.
   - **Mitigation**: Quarterly embedding model updates with recent clinical text. Maintain a drug name dictionary that is updated monthly from FDA drug database.

2. **Usage pattern drift**: Physicians may change how they interact with the system over time (shorter queries, different specialties, different question patterns).
   - **Detection**: Monitor query embedding distributions weekly using Maximum Mean Discrepancy (MMD). Alert if MMD between current week and baseline exceeds threshold.
   - **Mitigation**: Retrain embedding model quarterly on recent query-passage pairs from production traffic.

3. **Memory accumulation drift**: As patients accumulate more sessions, the vector store grows, retrieval noise increases, and context assembly becomes more challenging.
   - **Detection**: Track retrieval precision as a function of patient history length. Alert if precision drops below 70% for patients with > 20 sessions.
   - **Mitigation**: Implement memory consolidation: after every 10 sessions, merge highly similar memories, archive low-importance memories, and refresh the running summary.

### Model Versioning

- **Embedding model**: Versioned by date and training data hash (e.g., `bge-clinical-v2-20250315-def456`). Each version stored in encrypted S3 with metadata.
- **Entity extraction rules**: Version-controlled in Git. Each rule set has a version tag and changelog.
- **Memory scoring weights**: Version-controlled as configuration (alpha, beta, gamma, delta values). Changes require A/B test validation before deployment.
- **Rollback strategy**: Each deployment tags the current model and config versions. Rollback replaces the serving model within 3 minutes. FAISS indices are compatible across embedding model versions of the same dimensionality.

### A/B Testing

**Framework:**

- Traffic splitting at the API gateway level. Physicians are randomly assigned to experiment or control groups (sticky per physician to ensure consistent experience).
- **Primary metric**: Memory Recall Accuracy on history-dependent queries.
- **Guardrail metrics**: Allergy/Safety Recall must not decrease by any amount (zero-tolerance). P95 latency must not increase by more than 500ms.
- **Statistical significance**: Two-tailed t-test with $\alpha = 0.05$ and minimum 300 queries per group.
- **Minimum experiment duration**: 2 weeks.
- **Kill switch**: If any safety guardrail is violated, the experiment group is immediately routed back to the control configuration and a compliance review is triggered.

### Cost Analysis

**Monthly Cloud Compute Costs (Estimated):**

| Component | Instance/Service | Count | Monthly Cost |
|-----------|-----------------|-------|-------------|
| Embedding service | g5.xlarge (A10G, GovCloud) | 2 | \$3,200 |
| FAISS index hosting | r6g.2xlarge (GovCloud) | 2 | \$2,400 |
| Orchestrator | c6g.xlarge (GovCloud) | 3 | \$720 |
| PostgreSQL (RDS, encrypted) | db.r6g.large | 1 | \$520 |
| Redis (ElastiCache) | r6g.large | 1 | \$350 |
| S3 storage (encrypted, audit logs) | ~500GB | - | \$30 |
| Claude API (BAA) | ~8M input + 2M output tokens/day | - | \$22,000 |
| Training (quarterly, amortized) | 2x A100 spot, ~4 hours | - | \$180 |
| Compliance/security tooling | GuardDuty, CloudTrail, KMS | - | \$600 |
| **Total** | | | **\$30,000/month** |

The Claude API cost dominates at approximately 73% of total infrastructure spend. Key cost optimization strategies:

1. **Prompt caching**: Cache the system prompt and per-patient entity summaries. Anthropic's prompt caching can reduce input token costs by up to 90% for cached prefixes. Estimated savings: \$6,000-8,000/month.
2. **Memory efficiency**: Better memory scoring means fewer irrelevant memories in context. Improving context utilization from 55% to 75% reduces input tokens by ~30%. Estimated savings: \$3,000-4,000/month.
3. **Tiered model routing**: Use Claude Haiku for entity extraction and safety validation (fast, cheap). Reserve Sonnet for physician-facing responses. Estimated savings: \$2,000/month.
4. **Pre-computed summaries**: Generate MedFlow Prep summaries during off-peak hours (nightly batch) rather than on-demand. Reduces peak API usage.

With all optimizations, the target monthly cost is approximately \$20,000-22,000 -- within MedFlow AI's \$35K/month compute budget with headroom for growth.
