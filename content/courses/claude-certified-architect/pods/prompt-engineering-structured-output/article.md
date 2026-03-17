# Prompt Engineering & Structured Output

*Mastering explicit criteria, few-shot prompting, tool_use schemas, validation loops, and batch processing for production Claude applications*

---

Let us start with a scenario that every developer building with LLMs eventually faces. You have built an automated code review bot. It runs on every pull request in your CI/CD pipeline. The first version uses a simple prompt: "Review this code and flag any issues."

The results? A disaster. The bot flags missing docstrings as "critical" issues. It complains about variable naming preferences. It sometimes hallucinates bugs that do not exist. Your developers start ignoring every comment — even the legitimate ones about null pointer dereferences and SQL injection vulnerabilities. Within two weeks, someone proposes disabling the bot entirely.

The problem is not the model. The problem is the prompt. And in this article, we will build a complete toolkit for fixing it — from explicit review criteria that eliminate false positives, to few-shot examples that handle ambiguous cases, to JSON schemas that guarantee structured output, to validation loops that catch and correct errors automatically.

By the end, you will know exactly which technique to reach for when your Claude application produces inconsistent, unstructured, or unreliable output.

---

## Explicit Criteria Over Vague Instructions

Let us begin with the most common failure mode: vague instructions. Consider these two system prompts for our code review bot:

**Vague:**
```
Review this code. Be conservative. Flag anything that looks wrong.
```

**Explicit:**
```
Review this code for the following categories ONLY:
1. SECURITY: SQL injection, XSS, command injection, hardcoded secrets
2. BUGS: Null pointer access, off-by-one errors, race conditions, resource leaks
3. CORRECTNESS: Logic errors where claimed behavior contradicts actual code behavior

Do NOT flag:
- Style preferences (naming, formatting, line length)
- Missing documentation
- Performance optimizations unless they cause O(n squared) or worse complexity

For each issue found, provide:
- File and line number
- Category (SECURITY/BUGS/CORRECTNESS)
- Severity (CRITICAL/MAJOR/MINOR)
- One-sentence explanation
- Suggested fix as a code diff
```

The difference is night and day. The vague prompt tells the model to "be conservative" — but conservative about what? Should it flag fewer issues, or should it be conservative in its severity ratings? The model has no way to know, so it guesses. And when it guesses wrong, developers lose trust.

The explicit prompt does something fundamentally different: it gives the model **categorical criteria**. Instead of asking it to exercise judgment about what "looks wrong," it defines exactly three categories of issues to look for, explicitly excludes common false-positive generators (style, docs, minor performance), and specifies the output format.

This brings us to a critical insight about false positive rates. In a CI/CD pipeline, false positives are more damaging than false negatives. A missed bug is bad, but a flood of irrelevant comments makes developers stop reading reviews altogether. The explicit criteria approach works because it narrows the model's search space to high-value categories and eliminates the noise.

Let us look at this with a concrete example. Suppose the model reviews this Python function:

```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result.fetchone()
```

With the vague prompt, the model might produce five comments: missing docstring, no type hints, using `SELECT *` instead of named columns, no error handling, and — buried at the bottom — the SQL injection vulnerability. The developer skims the first three comments, rolls their eyes, and closes the review.

With the explicit prompt, the model produces exactly one comment:

```
File: user_service.py, Line 2
Category: SECURITY
Severity: CRITICAL
Issue: SQL injection — user_id is interpolated directly into query string
Fix: Use parameterized query: db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

This is exactly what we want. One critical finding, zero noise.

Now the question is: what happens when the boundary between categories is blurry? What if a piece of code is not clearly a "bug" but is certainly not a "style preference" either? This is where the model starts making inconsistent decisions — and that brings us to our next technique.

---

## Few-Shot Prompting — Teaching Through Examples

Few-shot prompting is the single most effective technique for getting consistent formatted output when instructions alone produce inconsistent results. Instead of describing every edge case with words, you *show* the model what you want by including 2-4 carefully chosen examples directly in the prompt.

Why does this work so well? When you include input-output examples, the model performs **in-context learning** — it identifies the pattern from your demonstrations and applies it to new inputs. The examples activate relevant patterns from training and act as a temporary task specification.

Let us formalize this. Given a new input $$x$$ and $$k$$ demonstration pairs $$\{(x_1, y_1), \ldots, (x_k, y_k)\}$$, the model computes:

$$P(y \mid x, \text{examples}) \gg P(y \mid x)$$

The examples sharpen the probability distribution over outputs. Without them, the model has many plausible output formats and reasoning strategies. With them, the distribution narrows to match the demonstrated pattern.

Let us plug in some simple numbers. Suppose we are building a document extractor that pulls invoice data from unstructured text. Without examples, the model might assign equal probability across several output formats — sometimes it uses JSON, sometimes plain text, sometimes it includes currency symbols, sometimes it does not. The consistency rate across 100 documents might be only 65%.

Now we add three examples showing our exact expected format. The model sees the pattern and the consistency rate jumps to 95%. The examples did not make the model smarter — they eliminated ambiguity about what we wanted. This is exactly what we want.

Here is the key insight for the exam: few-shot prompting is most valuable for **ambiguous-case handling**. Your examples should not demonstrate the easy cases — the model already gets those right. Instead, pick examples that show how to handle the grey areas.

For our code review bot, here is how we would add few-shot examples:

```python
examples = """
Example 1 - Acceptable pattern (NOT a bug):
  data = cache.get(key)
  if data is None:
      data = db.fetch(key)
      cache.set(key, data)
Review: No issues. The None check is a standard cache-miss pattern,
not a null safety issue.

Example 2 - Genuine bug:
  def process_items(items):
      for i in range(len(items)):
          if items[i].is_valid():
              items.remove(items[i])
Review:
- Line 3-4, Category: BUGS, Severity: MAJOR
- Issue: Modifying list while iterating causes skipped elements
- Fix: Use list comprehension:
  items = [x for x in items if not x.is_valid()]

Example 3 - Security issue disguised as normal code:
  template = f"Hello {user_input}, welcome to {site_name}!"
  return HttpResponse(template)
Review:
- Line 1, Category: SECURITY, Severity: CRITICAL
- Issue: user_input rendered without escaping, enabling XSS
- Fix: Use Django template engine or escape the input
"""
```

Notice what these examples accomplish. Example 1 teaches the model that a common pattern — checking for `None` after a cache lookup — is acceptable, not a null safety bug. Without this example, the model might flag it inconsistently. Example 2 shows a subtle real bug (modifying a list during iteration) with the exact output format we want. Example 3 demonstrates how to catch a security issue that looks like ordinary string formatting.

The power of few-shot prompting lies in this generalization capability: the model learns the *reasoning pattern* from your examples and applies it to novel situations it has never seen. You do not need to demonstrate every possible bug — three well-chosen examples teach the model your detection philosophy.

For the exam, remember these best practices:

1. **2-4 targeted examples** is the sweet spot. More examples consume context tokens with diminishing returns.
2. **Show reasoning, not just output.** Examples that include a one-line justification ("the None check is a standard cache-miss pattern") teach the model when and why to flag or skip.
3. **Cover the ambiguous cases.** Easy positives and easy negatives are already handled well. Pick examples at the decision boundary.
4. **Demonstrate varied document structures.** If you are extracting data from invoices, include examples with different layouts, missing fields, and unusual formatting.

![How few-shot examples resolve ambiguous cases that instructions alone handle inconsistently](figures/figure_1.png)
*How few-shot examples resolve ambiguous cases that instructions alone handle inconsistently*

---

## Structured Output with tool_use

Now let us tackle the most common question in production Claude applications: how do you guarantee that the model returns valid, schema-compliant JSON? The answer is Claude's **tool_use** feature — and understanding it is critical for the exam.

The core idea is elegant. Instead of asking the model to "return JSON" in a free-text response (which sometimes works and sometimes produces invalid syntax), you define a **tool** with a JSON schema. The model then "calls" this tool, which forces it to produce output that matches your schema exactly. The model is not literally executing code — it is generating structured output that conforms to the schema constraints you defined.

Let us build this step by step with our invoice extraction example:

```python
import anthropic

client = anthropic.Anthropic()

# Define the extraction tool with a JSON schema
extraction_tool = {
    "name": "extract_invoice_data",
    "description": "Extract structured data from an invoice document",
    "input_schema": {
        "type": "object",
        "properties": {
            "vendor_name": {
                "type": "string",
                "description": "Company name on the invoice"
            },
            "invoice_number": {
                "type": "string",
                "description": "Invoice/reference number"
            },
            "total_amount": {
                "type": "number",
                "description": "Total amount due in dollars"
            },
            "currency": {
                "type": "string",
                "enum": ["USD", "EUR", "GBP", "other"],
                "description": "Currency code"
            },
            "currency_detail": {
                "type": "string",
                "description": "If currency is 'other', specify here"
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "unit_price": {"type": "number"},
                        "line_total": {"type": "number"}
                    },
                    "required": ["description", "quantity",
                                 "unit_price", "line_total"]
                }
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Extraction confidence level"
            }
        },
        "required": ["vendor_name", "invoice_number", "total_amount",
                      "currency", "line_items", "confidence"]
    }
}

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[extraction_tool],
    tool_choice={"type": "tool", "name": "extract_invoice_data"},
    messages=[{
        "role": "user",
        "content": f"Extract the invoice data from this document:\n\n{invoice_text}"
    }]
)
```

Let us understand this code in detail. We define a tool called `extract_invoice_data` with a precise JSON schema. Every field has a type, a description, and constraints. The `currency` field uses an **enum** — it can only be one of four values. But we include an `"other"` option with a companion `currency_detail` field so the model has an escape hatch for unusual currencies instead of fabricating a match.

Now, the critical part: `tool_choice`. This is one of the most important concepts for the exam. There are three settings:

**`tool_choice: "auto"`** — The model decides whether to call a tool or respond with plain text. This is the default. Use it when the tool is optional — for example, a chatbot that can look up order status but usually just answers questions.

**`tool_choice: "any"`** — The model must call at least one tool, but it chooses which one. Use this when you have multiple tools and the model should pick the right one based on the input.

**`tool_choice: {"type": "tool", "name": "extract_invoice_data"}`** — The model must call this specific tool. This is what we use for structured extraction — we want guaranteed schema-compliant output, every time, with no exceptions.

Let us plug in some simple numbers to see why this matters. Suppose we process 1,000 invoices:

- With `"auto"`: 920 return structured JSON, 80 return free-text explanations like "I could not find a clear invoice number." We now need error handling for the 80 failures.
- With forced tool choice: 1,000 out of 1,000 return schema-compliant JSON. Zero format errors.

This is exactly what we want for extraction pipelines.

![Choosing between auto, any, and forced tool_choice settings](figures/figure_2.png)
*Choosing between auto, any, and forced tool_choice settings*

But here is the catch — and this is a point the exam tests specifically: **strict schemas eliminate syntax errors but NOT semantic errors.** The JSON will always be valid. The field types will always match. But the model can still produce semantically wrong values.

For example, an invoice with three line items at $100, $200, and $150 should have a `total_amount` of $450. But the model might extract `total_amount: 500` if the document is ambiguous or if it misreads a number. The schema guarantees that `total_amount` is a number — but it cannot guarantee that the number is the correct sum of line items.

There are several schema design patterns that reduce semantic errors:

1. **Nullable fields** — Use nullable for fields that might not exist in every document. This prevents the model from fabricating values to satisfy required fields.

2. **Enum + "other" pattern** — For categorical fields, define the common values as an enum and include `"other"` with a detail field. This is better than a free-text string (which invites inconsistency) or a closed enum (which forces incorrect matches).

3. **Confidence fields** — Add a `"confidence"` enum so the model can signal when it is uncertain. This lets your downstream pipeline flag low-confidence extractions for human review.

These patterns do not solve semantic errors completely — for that, we need validation. Which brings us to the next section.

---

## Validation-Retry Loops

Here is the scenario: you have built an invoice extractor with tool_use, and it works 92% of the time. But 8% of the time, the line items do not sum to the total, dates are in the wrong format, or a vendor name is misspelled. How do you catch and fix these errors automatically?

The answer is a **validation-retry loop** — also called retry-with-error-feedback. The pattern is simple but powerful:

1. Extract data from the document using tool_use
2. Run programmatic validation on the extracted data
3. If validation fails, send the original document, the failed extraction, and the specific validation errors back to the model
4. The model corrects its extraction based on the error feedback

Let us implement this:

```python
def validate_invoice(data):
    """Validate extracted invoice data. Returns list of error strings."""
    errors = []

    # Check that line items sum to total
    if data.get("line_items"):
        computed_total = sum(item["line_total"] for item in data["line_items"])
        if abs(computed_total - data["total_amount"]) > 0.01:
            errors.append(
                f"Line items sum to ${computed_total:.2f} but "
                f"total_amount is ${data['total_amount']:.2f}"
            )

    # Check each line item's internal consistency
    for i, item in enumerate(data.get("line_items", [])):
        expected = item["quantity"] * item["unit_price"]
        if abs(expected - item["line_total"]) > 0.01:
            errors.append(
                f"Line item {i+1}: {item['quantity']} x "
                f"${item['unit_price']:.2f} = ${expected:.2f}, "
                f"but line_total is ${item['line_total']:.2f}"
            )

    # Check required fields are not empty
    for field in ["vendor_name", "invoice_number"]:
        if not data.get(field) or data[field].strip() == "":
            errors.append(f"Required field '{field}' is empty")

    return errors

def extract_with_retry(invoice_text, max_retries=2):
    """Extract invoice data with validation-retry loop."""
    messages = [{
        "role": "user",
        "content": f"Extract the invoice data:\n\n{invoice_text}"
    }]

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[extraction_tool],
            tool_choice={"type": "tool", "name": "extract_invoice_data"},
            messages=messages
        )

        extracted = response.content[0].input
        errors = validate_invoice(extracted)

        if not errors:
            return {"data": extracted, "attempts": attempt + 1}

        if attempt < max_retries:
            # Append the failed extraction and errors as feedback
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": (
                    "The extraction has validation errors:\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\n\nPlease re-extract the data, fixing these "
                    "specific errors. Refer back to the original document."
                )
            })

    return {"data": extracted, "attempts": max_retries + 1, "errors": errors}
```

Let us understand what happens in this code. The `validate_invoice` function checks three things: that line items sum to the total, that each line item's quantity times unit price equals its line total, and that required fields are not empty. These are all structural and mathematical validations — things the model might get wrong but that we can verify programmatically.

The `extract_with_retry` function implements the retry loop. On the first attempt, it extracts normally. If validation fails, it appends the model's own failed response and the specific validation errors to the conversation, then asks the model to try again. The model sees what it got wrong and can correct it.

![Validation-retry loop: extract, validate, retry with error feedback, or escalate to human review](figures/figure_3.png)
*Validation-retry loop: extract, validate, retry with error feedback, or escalate to human review*

Now here is a crucial insight for the exam: **retries are only effective when the information exists in the source document.** If the invoice does not contain a vendor name, no amount of retrying will produce one. Retries fix format errors, mathematical errors, and misreading errors — cases where the model had the information but processed it incorrectly.

When should you NOT retry?

- **Missing information:** The source document simply does not contain the field you are looking for. Retrying wastes tokens and API calls. Instead, use nullable fields in your schema.
- **Ambiguous source data:** If the document has two possible invoice numbers and the model picks the wrong one, retrying might flip back and forth between them without converging.
- **Systematic model limitations:** If the model consistently fails at a specific extraction pattern (e.g., reading rotated text in a scanned PDF), retries will not help — you need a different approach.

For the exam, also understand the **detected_pattern field** concept. When building a code review system, you can add a `detected_pattern` field to your schema that captures what the model saw:

```json
{
    "issue": "Potential null pointer",
    "detected_pattern": "Variable 'user' accessed at line 15 without null check after fetch at line 10",
    "severity": "MAJOR"
}
```

This `detected_pattern` field serves two purposes. First, it makes the model's reasoning transparent — you can audit why it flagged something. Second, it enables systematic dismissal analysis. If you notice that 80% of flagged null-pointer issues have a `detected_pattern` mentioning "after cache lookup," you know you need to add a few-shot example showing that cache-miss patterns are acceptable.

---

## The Batch API — Processing at Scale

Now let us switch gears to a different problem: scale. Suppose your invoice extraction pipeline needs to process 10,000 documents overnight. You do not need real-time responses — you just need results by morning, and you want to minimize costs.

This is where the **Batch API** shines. It offers a 50% cost reduction compared to real-time API calls, with a 24-hour processing window. The trade-off is simple: you give up immediacy for savings.

Here is how it works:

```python
import anthropic
import json

client = anthropic.Anthropic()

# Step 1: Prepare batch requests with custom_id
batch_requests = []
for i, invoice in enumerate(invoices):
    batch_requests.append({
        "custom_id": f"invoice_{invoice['id']}",
        "params": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "tools": [extraction_tool],
            "tool_choice": {"type": "tool", "name": "extract_invoice_data"},
            "messages": [{
                "role": "user",
                "content": f"Extract invoice data:\n\n{invoice['text']}"
            }]
        }
    })

# Step 2: Write requests to JSONL file
with open("batch_input.jsonl", "w") as f:
    for req in batch_requests:
        f.write(json.dumps(req) + "\n")

# Step 3: Submit batch
batch = client.batches.create(
    input_file=open("batch_input.jsonl", "rb"),
    endpoint="/v1/messages",
    completion_window="24h"
)

print(f"Batch ID: {batch.id}")
print(f"Status: {batch.processing_status}")
```

Let us understand the key concepts. The `custom_id` field is critical — it is your tracking label for matching results back to original documents. When the batch completes, each result includes the `custom_id` you assigned, so you can map `invoice_12345` extraction result back to the original document in your database.

The processing happens asynchronously. You submit the batch, receive a batch ID, and poll for completion:

```python
# Step 4: Poll for completion
import time

while True:
    status = client.batches.retrieve(batch.id)
    if status.processing_status == "ended":
        break
    print(f"Progress: {status.request_counts.succeeded} / "
          f"{status.request_counts.total}")
    time.sleep(60)

# Step 5: Download and process results
results = client.batches.results(batch.id)
for result in results:
    doc_id = result.custom_id              # "invoice_12345"
    extracted = result.result.message       # The extraction response
    # Match back to original document and store
    store_extraction(doc_id, extracted)
```

![Batch API flow: preparing requests with custom_id, asynchronous processing, and matching results back to source documents](figures/figure_4.png)
*Batch API flow: preparing requests with custom_id, asynchronous processing, and matching results back to source documents*

When should you use batch vs real-time?

- **Batch:** Nightly document processing, bulk data extraction, periodic report generation, content moderation backlogs, migration tasks. Anything where you can wait up to 24 hours.
- **Real-time:** User-facing chat, interactive code review on pull request submission, live customer support, any workflow where the human is waiting for a response.

The exam specifically tests whether you understand this trade-off. A common mistake is using real-time API calls in a loop for batch-like workloads — this costs twice as much and often hits rate limits.

---

## Multi-Pass Review Architectures

Now let us return to our code review bot and add the final layer of sophistication: **multi-pass review**. Instead of asking a single prompt to do everything — find bugs, assess severity, check security, and suggest fixes — we split the work into specialized passes.

Why? Because different review tasks require different expertise and different prompt configurations. A security review pass needs deep knowledge of vulnerability patterns. A severity assessment pass needs context about the codebase and business logic. A cross-file integration pass needs to see how changes interact across multiple files.

Here is a three-pass architecture:

**Pass 1 — Detection (per-file):**
```
System: You are a code defect detector. For each file, identify ALL
potential issues. Be thorough — false positives are acceptable at
this stage. Categorize each as: SECURITY, BUG, LOGIC_ERROR, or
PERFORMANCE.

Output each finding as a structured tool call.
```

**Pass 2 — Severity Assessment (per-finding):**
```
System: You are a senior engineer assessing the severity of a code
finding. You are given:
- The original code with the finding highlighted
- The finding description from the detection pass
- The project test coverage for this file

Rate the severity as CRITICAL, MAJOR, MINOR, or FALSE_POSITIVE.
Provide a one-sentence justification.

A finding is FALSE_POSITIVE if:
- It follows a documented project pattern
- It is handled by a test
- The flagged behavior is intentional
```

**Pass 3 — Cross-File Integration:**
```
System: You are reviewing findings across multiple files in a pull
request. Look for:
- Related findings that should be grouped (e.g., a type change in
  one file that requires updates in callers)
- Findings that cancel each other out (e.g., a "missing null check"
  in a file where the caller already guarantees non-null)
- Patterns that suggest a systemic issue (same bug type in 3+ files)

Produce a final review summary with deduplicated, prioritized findings.
```

Each pass has a different personality, different instructions, and different context. Pass 1 casts a wide net — we want high recall, even if precision is low. Pass 2 filters aggressively — we want to eliminate false positives before they reach the developer. Pass 3 synthesizes across files — it catches issues that no single-file review could find.

![Multi-pass review architecture: three specialized passes progressively refine raw findings into actionable review comments](figures/figure_5.png)
*Multi-pass review architecture: three specialized passes progressively refine raw findings into actionable review comments*

The multi-pass approach also enables **domain-specific review passes**. If your codebase involves financial calculations, you can add a dedicated pass that checks for floating-point precision issues, rounding errors, and currency handling. If it involves medical data, you can add a HIPAA compliance pass. Each domain-specific pass uses a specialized prompt template with domain-relevant few-shot examples.

For the exam, understand that multi-pass architectures trade latency for quality. Each pass is a separate API call, so a three-pass review takes roughly three times longer than a single-pass review. This is acceptable for CI/CD pipelines (where the review runs in the background) but not for interactive code editors (where the developer is waiting).

---

## Putting It All Together

Let us step back and see how all these techniques connect. We started with a broken code review bot that developers wanted to disable. Here is the fixed architecture:

1. **Explicit Criteria** define what to look for and what to ignore — eliminating false positives from vague instructions
2. **Few-Shot Examples** handle the ambiguous cases at the decision boundary — teaching the model your team's conventions
3. **tool_use with JSON Schemas** guarantee structured output — every finding has a file, line, category, severity, and fix
4. **Validation-Retry Loops** catch mathematical and structural errors — with automatic correction via error feedback
5. **Batch API** handles scale — processing thousands of reviews overnight at half the cost
6. **Multi-Pass Architecture** separates concerns — detection, severity assessment, and cross-file integration each get specialized prompts

Each technique solves a specific failure mode. The art of prompt engineering is diagnosing which failure mode you are facing and reaching for the right tool.

For your exam preparation, practice identifying these patterns in real scenarios. When a system produces inconsistent output, ask: "Would 2-3 examples fix this?" When the output is unstructured, ask: "Should I force a tool_use schema?" When extractions have errors, ask: "Can I validate and retry programmatically?" When costs are too high, ask: "Can this be batched?"

These are not theoretical questions — they are the exact diagnostic patterns that separate a competent Claude architect from someone who just knows the API syntax.

---

## Key Takeaways

- **Explicit criteria beat vague instructions.** Define what to flag and what to skip with categorical specificity. False positive rates are the enemy of developer trust.
- **Few-shot examples resolve ambiguity.** Pick 2-4 examples at the decision boundary — where the model is most likely to be inconsistent.
- **tool_use guarantees structure, not semantics.** JSON schemas eliminate format errors. Semantic errors require validation.
- **Retry with error feedback fixes correctable mistakes.** But retries are futile when the information is absent from the source.
- **Batch API saves 50% on non-urgent workloads.** Use `custom_id` to match results back to source documents.
- **Multi-pass architectures trade latency for quality.** Separate detection from assessment from cross-file synthesis.

That is it for Prompt Engineering and Structured Output. These six techniques form the backbone of every production Claude application — and they represent 20% of your certification exam.
