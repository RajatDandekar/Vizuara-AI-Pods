# Context Management & Reliability

*Building production AI systems that handle long contexts, escalate intelligently, recover from failures, and track where every claim came from.*

---

Let us start with a concrete scenario. Imagine you are building a customer support agent for a large insurance company. A customer calls in with a complex claim spanning 15 pages of policy documents, 3 prior conversations, and a stack of medical records. Your AI agent needs to process all of this, extract the right facts, make a decision, and -- critically -- know when it is out of its depth and should hand off to a human.

This is the reality of production AI systems. The context window is not infinite. Models have blind spots in long documents. Errors in one agent can cascade through an entire pipeline. And if you cannot trace where a claim came from, you cannot trust it.

In this article, we will build the six core reliability skills that the Claude Certified Architect exam tests: context window management, escalation criteria, error propagation, information provenance, stratified sampling for human review, and scratchpad files for intermediate state. These are not theoretical concerns -- they are the difference between a demo that works and a production system that a business can depend on.

---

## 1. Context Window Management

### The Problem: Your Context Is Not a Filing Cabinet

Every language model has a finite context window. Claude's is large -- up to 200K tokens -- but in production, you will often need to process documents that exceed even this limit. And even within the window, not all positions are treated equally.

Let us understand the two core risks.

**Risk 1: Progressive Summarization Loses Critical Details.** A common approach to handling long documents is progressive summarization -- take a 50-page document, summarize it to 5 pages, then summarize again to 1 page. The problem? Each summarization step is lossy. A specific policy exclusion buried on page 37 might be the single most important fact for a customer's claim, and progressive summarization can silently drop it.

Consider this scenario. The agent reads the first 10 pages and summarizes them: *"Customer filed a claim on March 5 for water damage. Policy covers structural repairs."* Then it reads pages 11-20 and summarizes: *"Contractor estimates $45,000 in repairs. Adjuster visited on March 12."*

The problem? The original document on page 7 mentioned a **policy exclusion for pre-existing foundation cracks** -- a critical detail that the summary quietly dropped. When the agent later recommends full approval, it has made a decision based on incomplete information. This is the fundamental risk: each summarization step is lossy, and critical details can vanish without warning.

**Risk 2: The "Lost in the Middle" Effect.** Research has shown that language models pay stronger attention to information at the **beginning** and **end** of their context window, with weaker attention to material in the middle. If a critical fact sits in the middle of a 100K-token context, the model may effectively ignore it.

![Progressive summarization vs structured extraction -- the key tradeoff in context management.](figures/figure_1.png)
*Progressive summarization vs structured extraction -- the key tradeoff in context management.*

### The Solution: Structured Extraction Over Summarization

Instead of summarizing, extract the specific facts you need into a structured format:

```python
extraction_schema = {
    "claim_id": "CLM-2024-78432",
    "policy_exclusions": [
        {"text": "Pre-existing conditions diagnosed within 12 months",
         "source": "policy_doc.pdf, page 37, paragraph 3"}
    ],
    "medical_facts": [
        {"diagnosis": "Lumbar disc herniation",
         "date": "2024-01-15",
         "source": "medical_records.pdf, page 2"}
    ],
    "coverage_limits": [
        {"type": "outpatient_surgery",
         "limit": 50000,
         "source": "policy_doc.pdf, page 12"}
    ],
    "prior_decisions": [
        {"date": "2024-02-01",
         "decision": "Initial claim approved for diagnostic imaging",
         "source": "conversation_log_001.txt"}
    ]
}
```

This is exactly what we want. Every critical detail is preserved, and every fact is linked to its source. No information is lost because we never asked the model to decide what is "important enough" to keep.

### Placing Critical Information at Context Boundaries

Given the "lost in the middle" effect, production systems should place critical instructions and facts at the **beginning** and **end** of the context -- what we call the "Sandwich Pattern":

- **Start of context:** System instructions and critical rules (e.g., "NEVER approve claims with unresolved exclusions")
- **Middle of context:** Reference material, documents, previous conversation history
- **End of context:** The specific question or task, plus a reminder of the most important rules

```python
def build_context(system_instructions, extracted_facts, user_query):
    """Place critical info at context boundaries."""
    return f"""
{system_instructions}

CRITICAL RULES (read these carefully):
{extracted_facts['policy_exclusions']}
{extracted_facts['coverage_limits']}

--- Supporting Documentation ---
{extracted_facts['medical_facts']}
{extracted_facts['prior_decisions']}

--- End of Documentation ---

REMINDER - CRITICAL RULES:
{extracted_facts['policy_exclusions']}

User Query: {user_query}
"""
```

Notice that the policy exclusions appear both at the top and bottom of the context. This redundancy is intentional -- it ensures the model sees critical constraints regardless of where its attention is strongest. The model's natural attention pattern now works in our favor.

---

## 2. Escalation Criteria and Human-in-the-Loop

### When Should an Agent Say "I Need a Human"?

Not every decision should be made by an AI. The exam tests your ability to define **explicit escalation criteria with concrete thresholds** -- not vague guidelines like "escalate when uncertain."

Let us take an example. Our insurance agent is handling a routine reimbursement -- the customer submitted receipts for a covered procedure. This is straightforward: verify the amounts, check the coverage, approve.

But what if the customer says: *"I want to escalate this to a manager, and my lawyer will be contacting you about the denied claim from last month"?*

This is no longer routine. This requires human judgment. The question is: how do we build explicit criteria so the agent knows when to escalate?

### Defining Concrete Escalation Thresholds

Vague rules like "escalate when things get complex" are useless in production. The system needs **concrete, measurable thresholds**:

| Category | Escalation Trigger | Threshold |
|---|---|---|
| **Safety concerns** | Threats of harm, legal action, or urgent situations | Keyword detection + sentiment analysis |
| **Customer request** | Customer explicitly asks for a human | Any mention of "manager," "human," "supervisor" |
| **High-value decisions** | Financial decisions above a threshold | Claim value > $100,000 |
| **Policy gaps** | No authoritative source covers this situation | Confidence score < 0.7 on policy match |
| **Repeated failures** | Agent has failed to resolve after N attempts | 3 consecutive unsuccessful attempts |

The key insight is: **escalation criteria must be concrete and testable**, not subjective. "This seems complicated" is not a valid escalation rule. "The customer has mentioned legal action" is.

![Escalation decision tree -- concrete thresholds for when AI should hand off to humans.](figures/figure_2.png)
*Escalation decision tree -- concrete thresholds for when AI should hand off to humans.*

### Structured Handoff Summaries

When an agent escalates, it should not simply say "I could not handle this." A good handoff includes **what was attempted** and **what remains**:

```python
handoff_summary = {
    "escalation_reason": "policy_gap",
    "confidence_score": 0.62,
    "customer_id": "CUST-44891",
    "conversation_summary": "Customer claims water damage to basement. "
        "Policy covers flood damage but does not define whether "
        "groundwater seepage qualifies as 'flood.'",
    "what_was_attempted": [
        "Searched policy document for 'groundwater' -- no matches",
        "Searched policy document for 'seepage' -- no matches",
        "Checked FAQ database -- no relevant entries"
    ],
    "what_remains": [
        "Determine if groundwater seepage falls under flood coverage",
        "Check company precedent for similar claims"
    ],
    "relevant_context": {
        "policy_number": "HO-2024-33892",
        "claim_amount": 28500,
        "policy_sections_reviewed": ["Section 4.2 -- Water Damage",
                                      "Section 4.3 -- Flood Coverage"]
    },
    "suggested_action": "Human reviewer should determine if groundwater "
        "seepage falls under flood coverage per company precedent."
}
```

This is exactly what we want. The human reviewer does not need to start from scratch -- they can read the summary, see what the agent already tried, and focus on the specific gap.

---

## 3. Error Propagation Across Agents

### The Cascading Failure Problem

Now let us look at a multi-agent research system:

- **Agent A** (Document Retriever): Fetches 10 relevant papers
- **Agent B** (Fact Extractor): Extracts key claims from each paper
- **Agent C** (Synthesizer): Combines the claims into a research summary

Agent A successfully finds 10 papers. Agent B processes 8 of them but fails on 2 -- one because the PDF was corrupted, and another because the paper was in German (a language it was not configured to handle).

Now the question is: **what should Agent B report to Agent C?**

The worst approach is to silently process only the 8 papers and pass them to Agent C as if nothing went wrong. Agent C would then produce a research summary that appears complete but is actually missing two potentially critical papers.

### Structured Error Propagation

The right approach is to report exactly what succeeded, what failed, and why:

```python
agent_b_result = {
    "status": "partial_success",
    "completed": [
        {"paper_id": "arxiv:2301.001", "findings": "...", "confidence": 0.95},
        {"paper_id": "arxiv:2301.002", "findings": "...", "confidence": 0.88},
        # ... 6 more successful extractions
    ],
    "failed": [
        {
            "paper_id": "arxiv:2301.009",
            "error_type": "transient",
            "error_detail": "PDF download returned 503 Service Unavailable",
            "retry_recommended": True,
            "attempted_actions": ["download_pdf", "retry_after_30s", "retry_after_60s"]
        },
        {
            "paper_id": "arxiv:2301.010",
            "error_type": "permanent",
            "error_detail": "Paper is in German; extraction needs German language support",
            "retry_recommended": False,
            "attempted_actions": ["detect_language", "check_translation_service"]
        }
    ],
    "metadata": {
        "total_requested": 10,
        "success_count": 8,
        "failure_count": 2,
        "success_rate": 0.8
    }
}
```

This structure tells Agent C everything it needs: **transient failures** (like a 503 error) can be retried. **Permanent failures** (like an unsupported language) need a different approach. And 8 out of 10 papers is much better than 0.

![Error propagation in multi-agent systems -- structured errors with partial results.](figures/figure_3.png)
*Error propagation in multi-agent systems -- structured errors with partial results.*

### Crash Recovery Manifests

For long-running multi-step processes, a crash recovery manifest records what has been completed so that a restart does not redo finished work:

```python
recovery_manifest = {
    "task_id": "extraction_batch_042",
    "started_at": "2024-03-15T10:30:00Z",
    "last_checkpoint": "2024-03-15T10:45:00Z",
    "completed_steps": [
        {"paper_id": "arxiv:2301.001", "status": "done",
         "output_location": "/results/001.json"},
        {"paper_id": "arxiv:2301.002", "status": "done",
         "output_location": "/results/002.json"},
        {"paper_id": "arxiv:2301.003", "status": "done",
         "output_location": "/results/003.json"},
    ],
    "in_progress": {
        "paper_id": "arxiv:2301.004",
        "step": "extracting_findings",
        "partial_output": "/results/004_partial.json"
    },
    "pending": ["arxiv:2301.005", "arxiv:2301.006", "arxiv:2301.007"],
    "config": {
        "extraction_model": "claude-sonnet-4-5-20250514",
        "max_retries": 3
    }
}
```

When Agent B restarts, it reads the manifest and resumes from paper `arxiv:2301.004` instead of starting over. No work is wasted, and the system is resilient to crashes. This makes sense because redoing completed work wastes time and money, especially when each step involves expensive API calls.

### Graceful Degradation

The principle behind all of this is **graceful degradation**: return the best possible result even when something goes wrong. The hierarchy is:

1. **Complete success** -- all results returned
2. **Partial success with transparency** -- most results returned, failures clearly documented
3. **Informative failure** -- no results, but a clear explanation of what went wrong
4. **Last resort: escalation** -- the system hands off to a human

The worst outcome -- which should never happen -- is **silent failure**: returning incomplete results as if they were complete.

---

## 4. Information Provenance and Temporal Data

### Every Claim Needs a Source

Let us move to our data extraction scenario. Imagine an agent processing 50 legal contracts simultaneously to extract key terms -- parties, payment amounts, deadlines, and obligations.

The agent extracts: *"The payment deadline is March 31, 2024."*

But which document did this come from? If the agent is processing 50 contracts at once, losing track of the source is a recipe for disaster. This is where **claim-source mapping** becomes essential.

### Implementing Claim-Source Mappings

Every extracted piece of information should be linked to its source document, page, and paragraph:

```python
extracted_claim = {
    "claim": "Payment deadline is March 31, 2024",
    "value": "2024-03-31",
    "field_type": "deadline",
    "source": {
        "document_id": "contract_2024_047",
        "document_title": "Service Agreement - Acme Corp",
        "page": 12,
        "paragraph": 3,
        "exact_quote": "All payments shall be remitted no later than March 31, 2024",
        "extraction_timestamp": "2024-03-15T14:22:00Z"
    },
    "confidence": 0.97
}
```

This mapping serves three critical purposes:

1. **Auditability** -- anyone can verify the claim by checking the original source
2. **Hallucination prevention** -- if the agent cannot point to a source, the claim is flagged
3. **Dispute resolution** -- when two documents conflict, the source mappings make it visible

### Temporal Data Handling

Information has a shelf life. A contract valid in 2023 may have been amended in 2024. Medical guidelines change. Insurance policies get updated. A fact extracted in January might be outdated by March.

Production systems must handle temporal data explicitly:

```python
def check_temporal_validity(claim):
    """Flag claims that may be outdated."""
    source_date = claim["source"].get("document_date")
    extraction_date = claim["source"]["extraction_timestamp"]
    warnings = []

    # Flag if source document is more than 1 year old
    if source_date and (now - source_date).days > 365:
        warnings.append({
            "type": "outdated_source",
            "message": f"Source document is {(now - source_date).days} days old",
            "recommendation": "Verify against current version"
        })

    # Flag if extraction is older than document's update cycle
    if claim.get("update_frequency") == "quarterly":
        if (now - extraction_date).days > 90:
            warnings.append({
                "type": "stale_extraction",
                "message": "Extraction older than quarterly update cycle",
                "recommendation": "Re-extract from latest version"
            })

    return warnings
```

![Information provenance -- every claim traced to its source with temporal metadata.](figures/figure_4.png)
*Information provenance -- every claim traced to its source with temporal metadata.*

The strongest defense against hallucination is simple: **require the model to cite its source for every claim.** If the model cannot point to a specific document, page, and paragraph, the claim is flagged for review. This transforms hallucination from a silent failure into a detectable one.

---

## 5. Human Review with Stratified Sampling

### Why Random Sampling Falls Short

Suppose your data extraction pipeline processes 10,000 insurance claims per day. You have the budget to manually review 200 of them (2%). If you pick those 200 randomly, you might end up reviewing 195 straightforward claims and only 5 complex ones. But the complex claims are where errors hide.

**Stratified sampling** solves this by ensuring you sample from each category proportionally to its risk, not its volume:

```python
def stratified_sample(claims, total_budget=200):
    """Sample proportionally to error risk, not volume."""
    categories = {
        "simple_auto_approved": {"claims": [], "sample_rate": 0.01},
        "medium_complexity":    {"claims": [], "sample_rate": 0.05},
        "high_value":           {"claims": [], "sample_rate": 0.15},
        "edge_cases":           {"claims": [], "sample_rate": 0.30},
        "new_policy_types":     {"claims": [], "sample_rate": 0.50},
    }

    # Classify each claim into a stratum
    for claim in claims:
        category = classify_claim_complexity(claim)
        categories[category]["claims"].append(claim)

    # Sample from each category
    review_queue = []
    for cat_name, cat_data in categories.items():
        n_to_sample = max(1, int(
            len(cat_data["claims"]) * cat_data["sample_rate"]
        ))
        sampled = random.sample(
            cat_data["claims"],
            min(n_to_sample, len(cat_data["claims"]))
        )
        for claim in sampled:
            review_queue.append({
                "claim": claim,
                "category": cat_name,
                "priority": cat_data["sample_rate"]
            })

    review_queue.sort(key=lambda x: -x["priority"])
    return review_queue[:total_budget]
```

This ensures that edge cases and new policy types -- the categories most prone to errors -- get disproportionately more review attention. This is exactly what we want.

### Tracking Per-Category Accuracy

To continuously improve the system, track accuracy rates separately for each category:

```python
accuracy_tracker = {
    "simple_auto_approved": {"reviewed": 150, "correct": 148, "rate": 0.987},
    "medium_complexity":    {"reviewed": 89,  "correct": 82,  "rate": 0.921},
    "high_value":           {"reviewed": 45,  "correct": 38,  "rate": 0.844},
    "edge_cases":           {"reviewed": 30,  "correct": 21,  "rate": 0.700},
    "new_policy_types":     {"reviewed": 12,  "correct": 7,   "rate": 0.583},
}
```

Now you can see exactly where the system struggles: new policy types have 58% accuracy, while simple claims are at 99%. This tells you where to invest in better prompts, additional training data, or tighter escalation rules. Over time, the review strategy adapts -- categories with lower accuracy get higher sample rates in future batches.

---

## 6. Scratchpad Files and Intermediate State

### The Problem: Context Exhaustion

In a multi-step analysis -- say, processing 50 documents one by one -- each intermediate result adds to the context. By document 30, the context window is full, and the model starts losing earlier results.

**Scratchpad files** solve this by offloading intermediate results to persistent storage, keeping the context window lean.

### Designing a Scratchpad Schema

```python
scratchpad = {
    "task_id": "insurance-batch-2024-03-15",
    "created_at": "2024-03-15T10:00:00Z",
    "current_step": "document_analysis",
    "document_index": 15,
    "accumulated_results": {
        "total_documents": 50,
        "processed": 15,
        "findings": [
            {"doc_id": "doc_001", "key_facts": ["..."],
             "risk_flags": []},
            {"doc_id": "doc_002", "key_facts": ["..."],
             "risk_flags": ["pre-existing condition"]},
        ]
    },
    "running_statistics": {
        "total_claims_value": 2450000,
        "flagged_for_review": 3,
        "average_confidence": 0.89
    },
    "notes": [
        "Doc 007 had unusual formatting -- extracted with lower confidence",
        "Doc 012 references a policy amendment not in our database"
    ]
}
```

### When to Use Scratchpads vs Context

The decision is straightforward:

| Use Context | Use Scratchpad |
|---|---|
| Current conversation with user | Results from previous research steps |
| Active instructions and rules | Cross-referenced data from multiple sources |
| Immediate task details | Running calculations and intermediate totals |
| Small, focused datasets | Large datasets being processed in chunks |

The key question is: **will I need this information later, but not right now?** If yes, write it to a scratchpad. If you need it for the current reasoning step, keep it in context.

### Implementing Read-Write Patterns

The agent follows a simple loop for each step of a multi-step analysis:

```python
def process_documents_with_scratchpad(documents, scratchpad_path):
    """Process documents one by one, saving to scratchpad."""
    scratchpad = load_or_create_scratchpad(scratchpad_path)
    start_index = scratchpad["document_index"]

    for i in range(start_index, len(documents)):
        doc = documents[i]

        # Process with ONLY current document in context
        result = analyze_document(
            document=doc,
            running_stats=scratchpad["running_statistics"],
            task_instructions=TASK_INSTRUCTIONS
        )

        # Save result to scratchpad (not to context)
        scratchpad["accumulated_results"]["findings"].append(result)
        scratchpad["document_index"] = i + 1
        scratchpad["running_statistics"] = update_stats(
            scratchpad["running_statistics"], result
        )

        # Persist after every document (crash recovery)
        save_scratchpad(scratchpad, scratchpad_path)

    return scratchpad
```

This pattern keeps the context window lean -- at any given moment, the model only has the current document, the task instructions, and the running statistics. All accumulated results live in the scratchpad file. And because the scratchpad is saved after every document, it doubles as a crash recovery checkpoint.

---

## Wrapping Up

Let us recap the six reliability pillars we covered:

1. **Context Window Management:** Extract structured facts instead of summarizing. Place critical information at context boundaries. Never rely on progressive summarization for detail-critical tasks.

2. **Escalation Criteria:** Define explicit thresholds -- safety concerns, customer requests, policy gaps, high-value decisions. Always provide structured handoff summaries.

3. **Error Propagation:** Distinguish transient from permanent failures. Return partial results with transparency notes. Build crash recovery manifests for long-running tasks.

4. **Information Provenance:** Link every claim to its source document, page, and paragraph. Track temporal metadata. Flag outdated sources automatically.

5. **Stratified Sampling:** Sample review queues by risk category, not randomly. Track per-category accuracy to find where the system struggles.

6. **Scratchpad Files:** Offload intermediate results to persistent storage. Use them for crash recovery and to prevent context exhaustion.

These are the patterns that separate a weekend prototype from a production system. The exam will test not just whether you know these concepts, but whether you can apply them to realistic scenarios -- a customer support agent handling ambiguous claims, a multi-agent research pipeline recovering from partial failures, or a data extraction system that needs to prove where every fact came from.

That's it!

---

**References:**

1. Anthropic. (2024). *Claude Documentation -- Context Window Best Practices.*
2. Liu, N. F., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." *Transactions of the ACL.*
3. Anthropic. (2025). *Building Reliable Agentic Systems -- Error Handling Patterns.*
