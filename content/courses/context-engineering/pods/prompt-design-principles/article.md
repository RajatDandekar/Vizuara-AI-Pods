# Prompt Design Principles

*The science of steering LLMs through well-crafted instructions, examples, and structure*

---

Let us start with a simple analogy. Imagine you get into a taxi and say: "Take me somewhere nice for dinner." The driver might take you to a fast-food place, a sushi bar, or a steakhouse three towns over. Now imagine you say instead: "Take me to the Italian restaurant on 5th Street — the one with outdoor seating. Please avoid the highway, I prefer side streets."

Same driver, same car, but wildly different outcomes. The second instruction is not smarter — it is more precise. It gives the driver enough context to make the right decisions while leaving room for judgment on the exact route.

This is the difference between a bad prompt and a good one. The LLM is the driver. Your prompt is the directions. And the quality of the ride depends almost entirely on how well you give those directions.

In this article, we will build a complete toolkit for prompt design — from system prompts to few-shot learning, chain-of-thought reasoning, structured outputs, and prompt chaining. Each technique solves a specific failure mode, and by the end, you will know exactly when to reach for each one.

---

## What Makes a Prompt "Good"?

Before we dive into specific techniques, let us establish what we are optimizing for. A good prompt has three dimensions:

**Clarity** — The model should never have to guess what you want. Ambiguity in the prompt leads to ambiguity in the output.

**Specificity** — The prompt should constrain the output space to exactly what you need. Too broad and the model wanders. Too narrow and it cannot generalize.

**Structure** — The prompt should be organized so the model can parse it efficiently. Remember, the model attends to every token in the prompt — a well-structured prompt makes that attention productive.

Think of these three dimensions as a spectrum. On one end, you have "Write something about machine learning" — vague, unstructured, and wide open. On the other end, you have a 2,000-word system prompt with rigid if-else rules for every possible input. The sweet spot is in the middle: clear enough to guide, structured enough to parse, and specific enough to constrain — without being brittle.

![The prompt quality spectrum from Too Vague to Right Altitude to Too Rigid.](figures/figure_1.png)
*The prompt quality spectrum: too vague leads to ambiguous output, the right altitude produces reliable output, and too rigid creates brittle output.*

This brings us to our first technique.

---

## System Prompts — Setting the Stage

The system prompt is the most persistent piece of context in any LLM application. It is loaded at the start of every conversation and stays present throughout. Think of it as the personality blueprint — it tells the model who it is, what it should do, and how it should behave.

But here is where most developers go wrong: they write system prompts at the wrong level of abstraction.

Anthropic calls this the **"Right Altitude" principle**. Your system prompt should be specific enough to guide behavior effectively, yet flexible enough to give the model strong heuristics rather than brittle if-else logic.

Let us look at three levels to see the difference:

**Too Vague (30,000 feet):**
```
You are a helpful assistant. Answer questions accurately.
```

This tells the model almost nothing. What kind of assistant? What tone? What format? The model will default to generic behavior, and you will get generic results.

**Right Altitude (10,000 feet):**
```
You are a senior Python developer acting as a code reviewer.
Review code for: correctness, performance, and security.
For each issue found:
- State the problem clearly
- Explain why it matters
- Suggest a specific fix with code

Be direct and constructive. Skip praise — focus on what needs fixing.
If the code is clean, say so in one sentence.
```

This is specific enough to guide behavior (role, format, priorities) but flexible enough to handle any code the user submits. The model has strong heuristics — not rigid rules.

**Too Rigid (Ground Level):**
```
If the user submits Python code with a for loop, check if it
can be replaced with a list comprehension. If yes, output exactly:
"SUGGESTION: Replace for loop on line {N} with list comprehension."
If the code has more than 50 lines, split your review into sections
labeled SECTION_1, SECTION_2, etc. If the user asks a follow-up
question, respond only about the most recent code block...
```

This will break the moment the user does something unexpected — like submitting JavaScript, or asking a general question about code style. Rigid rules create brittle agents.

![The Right Altitude principle for system prompts.](figures/figure_2.png)
*The Right Altitude principle: 30,000 ft is too vague, 10,000 ft is the sweet spot, ground level is too rigid.*

The right altitude gives the model a compass, not a GPS route. It says "head north" rather than "turn left in 200 meters, then right in 50 meters, then..." — which breaks the moment there is a road closure.

---

## Few-Shot Learning — Teaching by Example

Now let us move to one of the most powerful prompting techniques: few-shot learning. The idea is beautifully simple — instead of describing what you want, you *show* what you want by including examples directly in the prompt.

Why does this work? When you include input-output examples in the prompt, the model performs what researchers call **in-context learning**. It identifies the pattern from your examples and applies it to the new input — without any weight updates or fine-tuning. The examples activate relevant patterns from training, acting as a temporary task specification.

We can write this formally. Given a new input $x$ and a set of $k$ demonstration examples $\{(x_1, y_1), (x_2, y_2), \ldots, (x_k, y_k)\}$, the model computes:

$$P(y \mid x) \approx P\big(y \mid x, (x_1, y_1), (x_2, y_2), \ldots, (x_k, y_k)\big)$$

The key insight is that the right-hand side — the probability conditioned on examples — is typically much sharper than the left-hand side. The examples reduce ambiguity about what kind of output is expected.

Let us plug in some simple numbers to see why this matters. Suppose we are building a sentiment classifier. Without examples (zero-shot), the model might assign: $P(\text{"positive"} \mid \text{review}) = 0.55$ and $P(\text{"negative"} \mid \text{review}) = 0.45$ — barely better than a coin flip because the model is unsure about our exact classification scheme.

Now we add three examples showing our labeling convention. The model sees the pattern and updates: $P(\text{"positive"} \mid \text{review, examples}) = 0.92$ and $P(\text{"negative"} \mid \text{review, examples}) = 0.08$. The examples sharpened the distribution dramatically. This is exactly what we want.

Here is a practical implementation:

```python
def build_few_shot_prompt(examples, new_input):
    """Build a few-shot prompt for sentiment classification."""
    prompt = "Classify the sentiment of each review as positive or negative.\n\n"

    # Add demonstration examples
    for text, label in examples:
        prompt += f"Review: {text}\nSentiment: {label}\n\n"

    # Add the new input
    prompt += f"Review: {new_input}\nSentiment:"
    return prompt

# --- Demonstration examples ---
examples = [
    ("The battery lasts all day and the camera is stunning.", "positive"),
    ("Crashed twice in the first hour. Waste of money.", "negative"),
    ("Fast shipping, exactly as described. Very happy.", "positive"),
]

new_review = "The interface is confusing and customer support never responds."
prompt = build_few_shot_prompt(examples, new_review)
print(prompt)
```

Let us understand this code in detail. The `build_few_shot_prompt` function constructs a prompt with three parts: a task instruction, the demonstration examples, and the new input. Each example follows a consistent format — `Review:` followed by `Sentiment:` — which teaches the model the exact input-output mapping we expect. The new input ends with `Sentiment:` but no answer, cueing the model to complete the pattern.

There are a few important best practices for few-shot learning:

1. **Number of examples:** 3–5 is the sweet spot for most tasks. More examples consume precious context tokens with diminishing returns.

2. **Ordering matters:** Research by Lu et al. (2022) showed that the order of few-shot examples can swing accuracy by up to 30 percentage points. When possible, put the most representative example last — models attend more strongly to recent tokens.

3. **Diversity:** Include examples that cover the range of expected inputs. If you are classifying sentiment, include clearly positive, clearly negative, and nuanced examples.

![Few-shot learning dramatically improves task performance across NLP tasks.](figures/figure_3.png)
*Few-shot learning dramatically improves task performance. Adding just 3 examples boosts accuracy by 20-30 percentage points.*

---

## Chain-of-Thought — Making the Model Think Aloud

Now let us look at a technique that was a genuine breakthrough: **Chain-of-Thought (CoT) prompting**, introduced by Wei et al. in 2022.

The core idea is this: instead of asking the model to jump directly from question to answer, you ask it to show its reasoning steps. This is surprisingly powerful — like asking a student to "show their work" on a math exam.

Here is a concrete example. Consider this problem: "A store has 45 apples. They sell 17 in the morning and receive a shipment of 23 in the afternoon. How many apples do they have?"

**Without CoT:** The model jumps directly to an answer — sometimes correct, sometimes not, and you cannot diagnose errors.

**With CoT:** The model writes: "Starting with 45 apples. After selling 17: 45 - 17 = 28. After receiving 23: 28 + 23 = 51. The store has 51 apples." Each intermediate step is visible and verifiable.

Why does this work? The key insight is that the intermediate reasoning tokens serve as a **computational scratchpad**. Each token the model generates becomes part of its context for generating the next token. By making the model write down intermediate steps, we give it more "working memory" for the computation.

We can formalize this. Without CoT, the model must compute the answer directly:

$$P(y \mid x)$$

With CoT, the model first generates a reasoning chain $z$ and then derives the answer:

$$P(y \mid x) = \sum_{z} P(y \mid z, x) \cdot P(z \mid x)$$

where $z$ represents the chain of reasoning steps.

Let us plug in some simple numbers. Suppose for a complex math problem, the direct probability of getting the right answer is $P(y \mid x) = 0.4$ — the model gets it right about 40% of the time by jumping to the answer.

Now, with chain-of-thought, suppose the model generates the correct reasoning chain with probability $P(z^* \mid x) = 0.85$, and given the correct reasoning chain, it produces the correct answer with probability $P(y \mid z^*, x) = 0.95$.

The overall probability becomes: $0.85 \times 0.95 = 0.81$.

We went from 40% accuracy to 81% accuracy — simply by asking the model to think aloud. This is exactly what we want.

There are two variants of CoT:

**Zero-shot CoT:** Simply append "Let's think step by step" to your prompt. Surprisingly effective — Kojima et al. (2022) showed this single phrase improves performance across a wide range of tasks.

**Few-shot CoT:** Provide examples that include the reasoning chain, not just the answer. This is more powerful because you show the model exactly what kind of reasoning you expect.

```python
def build_cot_prompt(question):
    """Build a chain-of-thought prompt with demonstrations."""
    prompt = """Solve each problem step by step, then give the final answer.

Question: A baker makes 12 loaves per hour. She works for 3 hours in the
morning and 2 hours in the afternoon. She sells 29 loaves during the day.
How many loaves remain?

Reasoning:
- Morning production: 12 × 3 = 36 loaves
- Afternoon production: 12 × 2 = 24 loaves
- Total produced: 36 + 24 = 60 loaves
- Loaves remaining: 60 - 29 = 31 loaves

Answer: 31

Question: {question}

Reasoning:"""
    return prompt.format(question=question)

problem = ("A warehouse has 3 shelves. Each shelf holds 48 boxes. "
           "Workers remove 35 boxes and add 22 new ones. "
           "How many boxes are in the warehouse?")
print(build_cot_prompt(problem))
```

This code demonstrates few-shot CoT. The demonstration example shows the full reasoning chain — each arithmetic step broken out — before stating the final answer. When the model encounters the new problem, it follows the same pattern: decompose, compute step by step, then conclude.

![Chain-of-Thought prompting: explicit reasoning steps improve accuracy.](figures/figure_4.png)
*Chain-of-Thought prompting adds explicit reasoning steps between input and output, dramatically improving accuracy on complex tasks.*

![Chain-of-Thought improves complex reasoning tasks.](figures/figure_5.png)
*Chain-of-Thought prompting provides the largest gains on tasks requiring multi-step reasoning.*

**When CoT hurts:** Chain-of-thought is not always the right choice. For simple factual lookups ("What is the capital of France?"), CoT adds unnecessary tokens and latency without improving accuracy. Use CoT when the task requires multi-step reasoning. Skip it when the answer is a direct recall.

---

## Structured Output — Controlling the Shape of Responses

Now let us tackle a problem that every developer building LLM applications faces: getting the model to output data in a consistent, parseable format.

By default, LLMs generate free-form text. This is great for chatbots but terrible for systems that need to parse the output programmatically. If your application expects JSON and the model returns a paragraph with the answer buried in the middle, your pipeline breaks.

The solution is **structured output** — using the prompt to constrain the format of the model's response.

There are three main approaches:

**1. Format instructions in the prompt:**
```
Extract the following fields from the text and return them as JSON:
- "name": the person's full name
- "age": their age as an integer
- "occupation": their job title

Return ONLY the JSON object, no other text.
```

**2. XML tags for clear section boundaries:**
```
Analyze the code and provide your review in this format:

<issues>
- Issue 1 description
- Issue 2 description
</issues>

<suggestion>
Your recommended fix here
</suggestion>

<severity>low | medium | high</severity>
```

**3. Constrained decoding:** Many inference frameworks (like Outlines or llama.cpp) can enforce output structure at the token level using formal grammars, guaranteeing valid JSON or other formats.

Here is a practical example:

```python
def build_extraction_prompt(text):
    """Build a prompt for structured information extraction."""
    prompt = f"""Extract key information from the following text.
Return your response as a JSON object with these exact fields:

{{
  "entities": ["list of named entities mentioned"],
  "sentiment": "positive" | "negative" | "neutral",
  "key_facts": ["list of main factual claims"],
  "confidence": 0.0 to 1.0
}}

Text: {text}

JSON:"""
    return prompt

text = ("Apple announced record Q4 revenue of $89.5 billion, "
        "driven by strong iPhone 16 sales. CEO Tim Cook called "
        "the results 'extraordinary' and highlighted growth in "
        "the services division.")
print(build_extraction_prompt(text))
```

The key design choice here is showing the exact JSON schema in the prompt. The model sees the field names, the expected types, and even example values for enums. This dramatically reduces format errors compared to vague instructions like "return the results as JSON."

![Structured output enables reliable downstream processing.](figures/figure_6.png)
*Structured output transforms fragile parsing into reliable JSON processing.*

---

## Prompt Chaining — Breaking Complex Tasks into Steps

So far, every technique we have discussed operates within a single prompt. But what happens when the task is too complex for one prompt to handle reliably?

This brings us to **prompt chaining**: the technique of breaking a complex task into a sequence of simpler sub-tasks, where the output of one prompt becomes the input to the next.

The intuition is straightforward. Consider a task like "Read this 50-page research paper and produce a structured literature review with citations." Asking a single prompt to do all of this is like asking someone to read a book, take notes, organize themes, and write a polished review — all in one breath. It is possible, but the quality will suffer.

Instead, we chain:
- **Prompt 1:** Extract key claims and findings from each section
- **Prompt 2:** Classify findings by theme
- **Prompt 3:** Synthesize themes into a coherent narrative
- **Prompt 4:** Format as a structured literature review with citations

Each prompt does one thing well, and the chain produces a result that no single prompt could match.

Here is the critical insight about reliability. If each step in a chain has an accuracy of $p_i$, then the overall chain accuracy is:

$$P(\text{chain correct}) = \prod_{i=1}^{n} p_i$$

Let us plug in some simple numbers. Suppose we have a 4-step chain where each step has 95% accuracy:

$$P(\text{chain correct}) = 0.95^4 = 0.815$$

That is 81.5% overall accuracy — a 13.5 percentage point drop from any individual step. Now suppose each step only has 85% accuracy:

$$P(\text{chain correct}) = 0.85^4 = 0.522$$

Only 52.2% — barely better than a coin flip! This tells us something critical: **each step must be highly reliable, or the chain falls apart.** This is why we add **gate checks** between steps — validation logic that catches errors before they propagate.

Let us implement a practical prompt chain:

```python
def step_extract(document):
    """Step 1: Extract key information from document."""
    return f"""Extract all factual claims from this document.
Return each claim as a numbered list.

Document: {document}

Claims:"""

def step_classify(claims):
    """Step 2: Classify claims by category."""
    return f"""Classify each claim into one of these categories:
- METHODOLOGY: how the research was conducted
- FINDING: key results or discoveries
- LIMITATION: acknowledged weaknesses or constraints

Claims:
{claims}

Return as JSON: {{"methodology": [...], "findings": [...], "limitations": [...]}}"""

def gate_check(classified_output):
    """Gate: Verify the classification output is valid."""
    import json
    try:
        data = json.loads(classified_output)
        required = {"methodology", "findings", "limitations"}
        if required.issubset(data.keys()):
            return True, data
    except (json.JSONDecodeError, AttributeError):
        pass
    return False, None

def step_synthesize(classified_data):
    """Step 3: Synthesize into a coherent summary."""
    return f"""Write a 3-paragraph summary based on these classified claims:

Methodology: {classified_data['methodology']}
Findings: {classified_data['findings']}
Limitations: {classified_data['limitations']}

Paragraph 1: What was done (methodology)
Paragraph 2: What was found (findings)
Paragraph 3: What remains open (limitations)"""
```

Let us understand this code. We have three prompt-building functions, each handling one step of the pipeline, plus a gate check between steps 2 and 3. The `step_extract` function pulls factual claims from a document. The `step_classify` function categorizes those claims. The `gate_check` function validates that the classification output is well-formed JSON with the required keys — if it fails, we can retry step 2 or flag an error instead of sending garbage to step 3. The `step_synthesize` function produces the final summary.

The gate check is the unsung hero here. Without it, a malformed classification in step 2 silently corrupts the synthesis in step 3. With it, we catch errors at the boundary and can retry or alert. In production, every chain should have gate checks between steps.

![Prompt chaining with gate checks.](figures/figure_7.png)
*Prompt chaining with gate checks catches errors at each boundary, dramatically improving end-to-end reliability.*

![Chained prompts with gate checks outperform single prompts.](figures/figure_8.png)
*Adding gate checks to a prompt chain pushes accuracy above 90% across all metrics.*

---

## Putting It All Together — A Decision Framework

We have covered five techniques: system prompts, few-shot learning, chain-of-thought, structured output, and prompt chaining. But the question every developer asks is: **which technique do I use when?**

Here is a practical decision framework:

**Start with the task complexity.** If the task is simple and well-defined (sentiment classification, data extraction, formatting), a good system prompt with structured output is usually sufficient.

**If accuracy is low,** add few-shot examples. This is the single highest-leverage improvement for most tasks — show the model what you want rather than describing it.

**If the task requires reasoning,** add chain-of-thought. Any task involving math, multi-step logic, planning, or causal reasoning benefits from CoT. But skip it for factual recall or simple classification.

**If the task is multi-step or complex,** use prompt chaining. Break it into subtasks, add gate checks, and compose the pipeline. This is essential when a single prompt cannot reliably handle the full task.

**If reliability is critical,** combine techniques. Few-shot CoT with structured output and gate checks is the gold standard for production systems.

![Prompt design decision framework.](figures/figure_9.png)
*A practical decision framework for choosing the right prompt design technique.*

Here is one final example that combines multiple techniques — few-shot CoT with structured output:

```python
def build_combined_prompt(code_snippet):
    """Few-shot CoT with structured output for code review."""
    return f"""Review the code below. Think through potential issues step by step,
then provide your review as structured JSON.

Example:
Code: `for i in range(len(lst)): print(lst[i])`
Reasoning: This iterates using index-based access. In Python, iterating
directly over the list is more idiomatic and slightly faster. No security
issues, but it reduces readability for experienced Python developers.
Review: {{"issues": ["Non-idiomatic loop pattern"], "severity": "low",
"suggestion": "Use `for item in lst: print(item)` instead"}}

Code: `{code_snippet}`
Reasoning:"""
```

Notice how this single prompt combines:
- A **system instruction** (review code, think step by step)
- A **few-shot example** (one demonstration of the expected flow)
- **Chain-of-thought** (explicit "Reasoning:" section before the answer)
- **Structured output** (JSON format with exact field names)

This is the power of composing techniques. Each one addresses a different failure mode, and together they produce reliable, parseable, well-reasoned output.

---

## Conclusion

Let us take a step back and see the full picture. Prompt design is context engineering at the atomic level — every technique we covered is about controlling what the model sees and how it processes that information.

We started with the three dimensions of prompt quality — clarity, specificity, and structure — and saw how the "right altitude" principle keeps system prompts flexible yet effective. We learned that few-shot examples sharpen the model's probability distribution by showing rather than telling. We discovered that chain-of-thought prompting gives the model a computational scratchpad, dramatically improving complex reasoning. We saw how structured output transforms unreliable free-text into parseable, deterministic data. And we learned that prompt chaining decomposes complex tasks into reliable sub-steps with gate checks that catch errors at the boundary.

The key takeaway: a well-designed prompt is not about being clever with words. It is about engineering the right information, in the right format, at the right level of abstraction. That is what separates a toy demo from a production system.

That's it!

---

**References:**

- Wei, Jason, et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (NeurIPS 2022)
- Kojima, Takeshi, et al. "Large Language Models are Zero-Shot Reasoners" (NeurIPS 2022)
- Brown, Tom, et al. "Language Models are Few-Shot Learners" (NeurIPS 2020)
- Lu, Yao, et al. "Fantastically Ordered Prompts and Where to Find Them" (ACL 2022)
- Anthropic. "Effective Context Engineering for AI Agents" (2025)
