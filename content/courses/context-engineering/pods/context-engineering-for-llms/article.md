# Context Engineering for LLMs

*The art and science of filling the context window with just the right information for the next step*

---

Let us start with a simple example. Imagine you are a brilliant student walking into an open-book exam. Your intelligence is fixed — you cannot suddenly become smarter halfway through. But your performance depends enormously on three things: what materials you brought, how your notes are organized, and whether you can find the right page fast enough before time runs out.

Now picture two students. The first one walks in with a messy backpack stuffed with six irrelevant textbooks, loose papers, and notes from last semester. The second student walks in with a slim, well-organized binder containing exactly the right notes — highlighted, tabbed, and ordered by topic.

Who do you think will score higher?

The second student, every time. Not because they are smarter, but because they have better **context**.

This is exactly the situation we face when building applications with Large Language Models. The LLM is the student. The context window is the open book. And your job — as the developer — is to design that book.

In June 2025, Shopify CEO Tobi Lutke captured this idea in a tweet that went viral: *"I really like the term 'context engineering' over prompt engineering. It describes the core skill better: the art of providing all the context for the task to be plausibly solvable by the LLM."* Within days, Andrej Karpathy amplified: *"Context engineering is the delicate art and science of filling the context window with just the right information for the next step."* MIT Technology Review later declared it the defining software development shift of 2025.

So here is the question we are asking in this article: **If prompt engineering is about writing the right question, what is the discipline of assembling the right information around that question?**

This brings us to Context Engineering.

---

## From Prompt Engineering to Context Engineering

Let us first understand what prompt engineering is. Prompt engineering is the practice of crafting the instruction text you give to an LLM — choosing the right words, formatting the request clearly, maybe adding a "think step by step" at the end. It is about optimizing a single message.

And it works — to a point.

But here is the problem: a perfect prompt with missing context will still fail. If you ask an LLM to debug your code but never show it the error message, the codebase, or the stack trace, even the most perfectly worded prompt will produce a useless answer. The prompt is fine. The **context** is wrong.

This is the fundamental limitation of thinking purely in terms of prompts. As Karpathy put it: *"People associate prompts with short task descriptions you'd give an LLM in your day-to-day use. When in every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window with just the right information for the next step."*

The shift is this:

- **Prompt engineering** = optimizing a single message
- **Context engineering** = designing the entire information environment across multiple inference turns

Context engineering includes everything: system prompts, tools and their outputs, retrieved documents, conversation history, memory from prior sessions, and even Model Context Protocol (MCP) resources. It is everything *beyond* the user's typed question.

Have a look at the diagram below:


![Prompt engineering optimizes a single input; context engineering orchestrates the entire information environment.](figures/figure_1.png)
*Prompt engineering optimizes a single input; context engineering orchestrates the entire information environment.*


Simon Willison made a sharp observation about why this terminology matters: *"Unlike 'prompt engineering,' context engineering has an inferred definition that's much closer to the intended meaning."* Prompt engineering sounds like typing into a chatbot. Context engineering signals systems-level work — and that is exactly what it is.

---

## Anatomy of the Context Window

Now let us look at what the context window actually contains. Think of it as a **working desk** — everything the LLM can see at inference time is laid out on this desk, and nothing else exists for it.

The context window is composed of six key components:

**1. System Prompt**

This is the persistent set of instructions that define the LLM's personality, constraints, and output format. It is always present and sets the tone for everything that follows.

There is an important design principle here that Anthropic calls the **"right altitude"** principle. Your system prompt should be specific enough to guide behavior effectively, yet flexible enough to give the model strong heuristics rather than brittle if-else logic. Too prescriptive and your agent breaks on edge cases. Too vague and it has no behavioral signal to follow.

**2. User Message**

This is the actual question or task — what the user is asking the LLM to do right now.

**3. Conversation History**

These are the prior turns of dialogue. In a multi-turn conversation, the model needs to remember what was said before to provide coherent responses.

**4. Retrieved Context (RAG)**

These are documents fetched from external knowledge bases — research papers, documentation, database records — that are relevant to the current query. We will dive deep into this in a later section.

**5. Tool Results**

These are the outputs from function calls — API responses, database queries, code execution results — that give the model access to real-time information.

**6. Memory**

These are persisted facts from prior sessions — user preferences, past decisions, learned patterns — that carry forward across conversations.

Have a look at the figure below which shows how these components stack inside the context window:


![The six components of a context window, with approximate token allocations for a 128K-token model.](figures/figure_2.png)
*The six components of a context window, with approximate token allocations for a 128K-token model.*


Now, here is the critical insight. All of these components compete for the same finite space. We can write this formally as:

$$T_{\text{total}} = T_{\text{system}} + T_{\text{history}} + T_{\text{RAG}} + T_{\text{tools}} + T_{\text{user}} + T_{\text{reserved}}$$

where $T_{\text{reserved}}$ is the tokens reserved for the model's output generation.

Let us plug in some simple numbers. Suppose we have a model with a 128K-token context window. We allocate: system prompt = 2K, conversation history = 20K, RAG documents = 60K, tool results = 10K, user message = 1K, and we reserve 35K for the model's output.

$$T_{\text{total}} = 2K + 20K + 60K + 10K + 1K + 35K = 128K$$

This fits perfectly. But now suppose our RAG retrieval returns 80K tokens instead of 60K. We are suddenly 20K over budget. What do we do? Drop some retrieved chunks? Compress the conversation history? Shorten the system prompt?

This budgeting decision is the heart of context engineering. And it gets made every single inference call.

---

## When Context Goes Wrong — Four Failure Modes

Before we learn how to engineer context well, let us understand how it fails. Drew Breunig published a widely-cited taxonomy of context failures that has become essential reading for anyone building with LLMs. There are four distinct failure modes.

**Failure Mode 1: Context Poisoning**

This is what happens when a hallucination or error enters the context and then gets referenced repeatedly in future responses. The model treats its own previous output as truth, and the mistake compounds.

Here is a striking example. Google built a Gemini-based agent to play Pokémon. At some point, the agent hallucinated about the game state — it believed it had certain items and objectives that did not actually exist. This false information poisoned the "goals" section of its context, and the agent began developing completely nonsensical strategies based on this fabricated reality.

Think of it like a student who accidentally writes the wrong formula on their reference sheet. Every subsequent problem they solve using that formula will be wrong — and they will have no idea why, because they trust their own notes.

**Failure Mode 2: Context Distraction**

As the context grows longer and longer, the model begins to over-focus on its history, neglecting the knowledge it learned during training. Research has shown that beyond 100K tokens, agents develop "a tendency toward repeating actions from their vast history rather than synthesizing novel plans."

This is closely related to the **"Lost in the Middle"** phenomenon discovered by Liu et al. (2023): LLMs attend strongly to information at the beginning and end of their context, but perform poorly on information buried in the middle. More context does not always help — it can actually hurt.

**Failure Mode 3: Context Confusion**

This occurs when superfluous or irrelevant information is dumped into the context, forcing the model to process noise alongside signal. The model cannot simply ignore irrelevant tokens — it must attend to everything.

Here is a concrete example: researchers tested a quantized Llama 3.1 8B model and found that it failed when given access to 46 tools, but succeeded perfectly when the same task was presented with only 19 tools. The extra 27 tools — none of which were needed — confused the model enough to cause complete failure. Less is more.

**Failure Mode 4: Context Clash**

This is what happens when new information or tools directly conflict with existing prompt instructions. The model receives contradictory signals and does not know which to follow.

A Microsoft/Salesforce study found that when prompts were sharded (split across multiple sections), performance dropped by an average of 39%. The model o3 saw its accuracy plummet from 98.1% to 64.1% on the same tasks — simply because the instructions were fragmented rather than unified.

Have a look at the figure below which summarizes these four failure modes:


![The four failure modes of context engineering, identified by Drew Breunig (2025).](figures/figure_3.png)
*The four failure modes of context engineering, identified by Drew Breunig (2025).*


So, how do we avoid these failure modes? This brings us to the four core strategies of context engineering.

---

## Four Core Strategies for Context Engineering

Lance Martin of LangChain and the team at Anthropic have codified four canonical strategies that address the failure modes we just discussed. Each strategy has a clear purpose, and the best systems use all four together.

**Strategy 1: Write Context (Persist Outside the Window)**

The idea here is simple but powerful: save important information *outside* the context window so it can be recalled later, even after the window is cleared or reset.

Let us take a real-world example. **Claude Code** — Anthropic's AI coding agent — uses a file called `CLAUDE.md` that sits at the root of your project. This Markdown file persists project conventions, build commands, code style rules, and workflow instructions. Every time you start a new conversation with Claude Code, it reads this file first. The context is written once and persisted permanently.

But it gets even more interesting. In an experiment where Anthropic had Claude play the game Pokémon, the agent maintained a `NOTES.md` file where it tracked game state — precise tallies of items, maps of explored regions, and strategic combat notes — across thousands of game steps. When the context window filled up and was reset, the agent simply read its notes and continued seamlessly. This is the "Write" strategy in action.

The analogy is straightforward: instead of memorizing every phone number, save them in your contacts and look them up when you need them.

**Strategy 2: Select Context (Retrieve Just-in-Time)**

This strategy says: do not pre-load everything into context. Instead, pull the right information in *when the agent needs it*.

The key idea is **just-in-time retrieval**. Rather than stuffing the context window with every potentially relevant document, maintain lightweight identifiers — file paths, search queries, URLs — and fetch the full content only when it becomes relevant.

**Cursor**, the AI code editor, is a great example of the manual approach. Developers use `@` symbols to explicitly reference specific files, folders, or code snippets. The developer is the context selector — they decide exactly what the AI sees.

**Claude Code** takes a hybrid approach. The `CLAUDE.md` loads upfront for speed, while `glob` and `grep` primitives enable just-in-time navigation, "effectively bypassing the issues of stale indexing and complex syntax trees." Some context is eager (loaded at start), other context is lazy (fetched on demand).

Anthropic calls this **progressive disclosure**: agents incrementally discover context through exploration. File sizes suggest complexity, naming conventions hint at purpose, timestamps can be a proxy for relevance. The agent does not need to see everything — it needs to know where to look.

**Strategy 3: Compress Context (Summarize and Compact)**

When the context window approaches its limit, you have two choices: truncate blindly, or compress intelligently. This strategy chooses the latter.

The principle is: preserve high-signal information and discard redundancy. Keep architectural decisions, unresolved bugs, and implementation details. Discard verbose tool outputs and repetitive intermediate steps.

**Claude Code's auto-compact** is a textbook example. When the context reaches approximately 95% capacity, it automatically summarizes the conversation history, clears older tool outputs while preserving key decisions, and reinitializes with a compressed version. Users can also trigger this manually with the `/compact` command.

A particularly clever technique is **tool result clearing**: after the agent has processed a tool's output and drawn conclusions, strip the verbose raw output and keep only the conclusions. The detailed data served its purpose — it does not need to occupy context forever.

**Strategy 4: Isolate Context (Sub-Agent Architectures)**

The final strategy is to split context across multiple agents, each with their own clean, focused context window.

The architecture looks like this: a main "lead" agent coordinates strategy and delegates specific research tasks to specialized sub-agents. Each sub-agent explores deeply — potentially using tens of thousands of tokens — but returns only a condensed summary of 1,000–2,000 tokens back to the lead agent.

Anthropic found that their multi-agent researcher — with isolated contexts for each sub-agent — outperformed single-agent approaches. The isolation prevents context confusion: the deep search context stays contained within the sub-agent and never pollutes the lead agent's strategic view.

This works best for parallelizable, read-only tasks like research gathering. One important caution: writing tasks that require coordination between agents can introduce conflicting decisions, so isolation is not always the answer.

Have a look at the figure below which summarizes all four strategies:


![The four core strategies of context engineering: Write, Select, Compress, Isolate.](figures/figure_4.png)
*The four core strategies of context engineering: Write, Select, Compress, Isolate.*


---

## Context Engineering in the Wild — Products That Got It Right

Now let us look at real products where context engineering is the key differentiator. These are not theoretical — these are tools that millions of developers use daily, and their success comes down to how well they manage context.

**OpenClaw (179K+ GitHub Stars)**

OpenClaw — originally named "Clawdbot" before being renamed after Anthropic trademark complaints — became the fastest repository in GitHub history to reach 100,000 stars, achieving this milestone in approximately two days in January 2026.

What makes it special? Its context system. OpenClaw uses **plain Markdown files as the source of truth** for agent memory, layered with hybrid semantic search. There are no hidden vector databases or opaque embeddings — everything is human-readable and debuggable. The chunking algorithm uses approximately 400 tokens per chunk with 80-token overlap, preserving context across chunk boundaries.

The architecture follows a gateway pattern: a WebSocket server dispatches messages to an Agent Runtime, which assembles context from session history and memory, invokes the model, executes tool calls, and persists updated state. A context compaction safeguard prevents overflow, and multi-agent routing isolates per-agent sessions.

Why did it win? Transparency. Developers can see and edit exactly what the agent remembers. In a world of black-box AI, that visibility is gold.

**Cursor vs. Windsurf: Two Philosophies**

These two AI code editors represent opposite ends of the context engineering spectrum, and both are wildly successful.

**Cursor** takes the manual curation approach. The developer picks what the AI sees using `@` references to specific files, folders, and code snippets. Under the hood, Cursor uses tree-sitter (an AST parser) to split code into semantic chunks — functions, classes, logical blocks — and stores over 100 billion vectors in Turbopuffer, a serverless vector database. Practical context sits around 10K–50K tokens. The strength is precision and control.

**Windsurf** takes the opposite approach: automatic retrieval. Its RAG-based context engine indexes the entire codebase without any manual curation. It operates with a ~200K-token context window, and its SWE-grep models deliver context retrieval 10x faster than frontier models. The "Cascade" system maintains real-time awareness of developer actions — you start a refactor, Cascade continues it. The strength is autonomy and repo-wide understanding.

The lesson? Both succeed because they solve context engineering well. They just put the control in different hands — human versus agent.

**Augment Code: "Context Is the New Compiler"**

Augment Code takes context engineering to the enterprise scale. Their Context Engine performs deep semantic indexing using dependency graphs, processing over 400,000 files across a codebase. It pulls context not just from code but from GitHub, Jira, and Notion — wherever relevant information lives.

The results speak for themselves: one customer reduced engineer onboarding from 18 months to just 2 weeks on a legacy Java monolith. Augment reports a 70%+ improvement in agentic coding performance across Claude Code, Cursor, and Codex when their Context Engine is used. Their tagline captures it well: "Context Is the New Compiler."

**Mem0 (41K+ GitHub Stars, 14M+ Downloads)**

Mem0 provides a dedicated memory layer for AI agents. Instead of managing memory inside the context window, Mem0 dynamically extracts, consolidates, and retrieves salient information from ongoing conversations.

The numbers are compelling: 26% accuracy improvement over OpenAI Memory on the LOCOMO benchmark, 91% faster responses than full-context approaches, and 90% lower token usage. This is the "Write" strategy at scale — a specialized system for persisting and recalling context efficiently.

Have a look at the comparison below:


![How five leading products approach context engineering differently.](figures/figure_5.png)
*How five leading products approach context engineering differently.*


---

## RAG — The Most Common Context Engineering Pattern

Now let us look at the single most common context engineering technique in production today: **Retrieval-Augmented Generation**, or RAG.

The core idea behind RAG is simple. Rather than relying solely on what the LLM learned during training (which is frozen at a point in time), we retrieve relevant documents from an external knowledge base and inject them into the context window alongside the user's query.

The pipeline works as follows:

1. The user sends a query
2. The query is converted into an embedding (a vector of numbers)
3. This embedding is compared against pre-computed embeddings of document chunks stored in a vector database
4. The top-K most similar chunks are retrieved
5. These chunks pass through a reranker for a second relevance check
6. The final selected chunks are injected into the context window
7. The LLM generates a response using both its training knowledge and the retrieved context

Have a look at the diagram below:


![The RAG pipeline: from user query to context-enriched LLM response.](figures/figure_6.png)
*The RAG pipeline: from user query to context-enriched LLM response.*


Let us go back to our open-book exam analogy. The three key design decisions in RAG correspond directly to how you organize your notes:

**Chunk size** — This determines how large each retrievable unit is. If your chunks are too small (say, a single sentence), you are ripping your notes into confetti — each piece has so little context that it is nearly meaningless on its own. If your chunks are too large (say, an entire chapter), you are wasting precious context window space on mostly irrelevant text. The sweet spot is typically 400–1,000 tokens per chunk.

**Top-K selection** — This is how many chunks you retrieve. Too few and you might miss the answer entirely. Too many and noise drowns the signal. This is a direct manifestation of the context confusion failure mode.

**Reranking** — This is a second pass where a specialized model reorders the retrieved chunks by true relevance to the query. Think of it as flipping through your retrieved notes and pulling the single most useful page to the very front. This is particularly important given the "Lost in the Middle" effect — you want the best information at the top.

Now, how does the retrieval actually work? The standard approach uses **cosine similarity** between the query embedding and each document embedding:

$$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{||\mathbf{q}|| \; ||\mathbf{d}||}$$

Let us plug in some simple numbers. Suppose our query embedding is $\mathbf{q} = [1, 2, 3]$ and we have a document embedding $\mathbf{d} = [2, 4, 6]$.

The dot product is: $1 \times 2 + 2 \times 4 + 3 \times 6 = 2 + 8 + 18 = 28$

The magnitudes are: $||\mathbf{q}|| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14} \approx 3.74$ and $||\mathbf{d}|| = \sqrt{2^2 + 4^2 + 6^2} = \sqrt{56} \approx 7.48$

So: $\text{sim} = \frac{28}{3.74 \times 7.48} \approx 1.0$

A cosine similarity of 1.0 means these vectors are perfectly aligned — this document is highly relevant to our query!

Now let us try a different document $\mathbf{d'} = [3, -1, 0]$:

Dot product = $1 \times 3 + 2 \times (-1) + 3 \times 0 = 3 - 2 + 0 = 1$

$||\mathbf{d'}|| = \sqrt{9 + 1 + 0} = \sqrt{10} \approx 3.16$

$\text{sim} = \frac{1}{3.74 \times 3.16} \approx 0.08$

A similarity of 0.08 — barely related at all. We would not retrieve this document. This is exactly what we want: high scores for relevant documents, low scores for irrelevant ones.

Let us now implement a simple RAG pipeline in Python:

```python
import numpy as np

# --- Document Embeddings (pre-computed) ---
documents = [
    "Batch normalization normalizes layer inputs to stabilize training.",
    "Dropout randomly deactivates neurons to prevent overfitting.",
    "The transformer architecture uses self-attention mechanisms.",
    "Learning rate scheduling adjusts the step size during training.",
    "Residual connections help gradients flow through deep networks.",
]
# Simulated embeddings (in practice, use an embedding model)
doc_embeddings = np.random.randn(len(documents), 128)
doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

def retrieve(query_embedding, top_k=3):
    """Retrieve top-K most similar documents using cosine similarity."""
    similarities = doc_embeddings @ query_embedding
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(documents[i], similarities[i]) for i in top_indices]

def build_context(query, retrieved_docs, system_prompt):
    """Assemble the final context from all components."""
    context = f"<system>{system_prompt}</system>\n\n"
    context += "<retrieved_context>\n"
    for doc, score in retrieved_docs:
        context += f"  [relevance={score:.2f}] {doc}\n"
    context += "</retrieved_context>\n\n"
    context += f"<user_query>{query}</user_query>"
    return context

# --- Usage ---
query_emb = np.random.randn(128)
query_emb = query_emb / np.linalg.norm(query_emb)

results = retrieve(query_emb, top_k=3)
context = build_context("What is batch normalization?", results,
                         "You are a helpful ML tutor.")
print(context)
```

Let us understand this code in detail. First, we have a set of documents and their pre-computed embeddings. In a real system, these embeddings would be generated by a model like `text-embedding-3-small` and stored in a vector database like Pinecone or Weaviate.

The `retrieve` function computes the cosine similarity between the query embedding and every document embedding (since both are already normalized, the dot product equals cosine similarity). It then returns the top-K most similar documents.

The `build_context` function assembles the final context string using XML tags for clear structure — the system prompt, the retrieved documents with their relevance scores, and the user query. This structured formatting makes it easy for the LLM to parse and attend to each section.

When we run this code with a query about "batch normalization," the output looks something like this:

```
<system>You are a helpful ML tutor.</system>

<retrieved_context>
  [relevance=0.87] Batch normalization normalizes layer inputs to stabilize training.
  [relevance=0.72] Residual connections help gradients flow through deep networks.
  [relevance=0.65] Learning rate scheduling adjusts the step size during training.
</retrieved_context>

<user_query>What is batch normalization?</user_query>
```

Notice how the most relevant document (0.87 similarity) appears first, and each section is clearly delimited with XML tags. The LLM can now attend to the right information for the next step. This is exactly what we want.

---

## Practical Implementation — Building a Context Engine

Now let us bring everything together. In this section, we will build a complete `ContextEngine` class that implements all the context engineering principles we have discussed.

```python
import numpy as np
import json
from pathlib import Path

class ContextEngine:
    """Assembles optimal context for LLM inference."""

    def __init__(self, max_tokens=128000, reserved_for_output=35000):
        self.max_tokens = max_tokens
        self.reserved = reserved_for_output
        self.available = max_tokens - reserved_for_output
        self.memory_path = Path("memory.json")

    def _estimate_tokens(self, text):
        return len(text) // 4  # ~4 chars per token

    def _load_memory(self):
        """Strategy 1: Write — load persisted memory."""
        if self.memory_path.exists():
            return json.loads(self.memory_path.read_text())
        return []

    def _retrieve_docs(self, query_emb, doc_embs, docs, top_k=5):
        """Strategy 2: Select — retrieve via cosine similarity."""
        scores = doc_embs @ query_emb
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(docs[i], scores[i]) for i in top_idx]

    def _compress_history(self, history, budget):
        """Strategy 3: Compress — keep recent, drop old."""
        compressed, used = [], 0
        for msg in reversed(history):
            t = self._estimate_tokens(msg)
            if used + t > budget:
                break
            compressed.insert(0, msg)
            used += t
        return compressed
```

The `assemble` method brings everything together:

```python
    def assemble(self, query, system_prompt, history,
                 query_emb, doc_embs, docs):
        """Assemble complete context within token budget."""
        sys_t = self._estimate_tokens(system_prompt)
        q_t = self._estimate_tokens(query)

        memories = self._load_memory()
        mem_text = "\n".join(memories)
        mem_t = self._estimate_tokens(mem_text)

        remaining = self.available - sys_t - q_t - mem_t
        hist_budget = int(remaining * 0.3)
        rag_budget = int(remaining * 0.7)

        history = self._compress_history(history, hist_budget)

        retrieved = self._retrieve_docs(query_emb, doc_embs, docs)
        rag_docs, rag_t = [], 0
        for doc, score in retrieved:
            dt = self._estimate_tokens(doc)
            if rag_t + dt > rag_budget:
                break
            rag_docs.append((doc, score))
            rag_t += dt

        # Assemble with clear structure
        context = f"<system>\n{system_prompt}\n</system>\n\n"
        if memories:
            context += f"<memory>\n{mem_text}\n</memory>\n\n"
        if history:
            context += "<history>\n"
            context += "\n".join(history)
            context += "\n</history>\n\n"
        context += "<retrieved_context>\n"
        for doc, score in rag_docs:
            context += f"  [{score:.2f}] {doc}\n"
        context += "</retrieved_context>\n\n"
        context += f"<query>{query}</query>"

        return context
```

Let us understand this code in detail. The `ContextEngine` class implements all four strategies:

**Strategy 1 (Write):** The `_load_memory` method reads persisted facts from an external JSON file. These memories survive across sessions and are loaded into every context assembly.

**Strategy 2 (Select):** The `_retrieve_documents` method performs cosine similarity search to find the most relevant documents — only fetching what is needed for the current query.

**Strategy 3 (Compress):** The `_compress_history` method keeps only the most recent conversation turns that fit within a token budget, preserving recency while discarding old context.

**Strategy 4 (Isolate):** While not shown in this single class, you would create multiple `ContextEngine` instances — one per sub-agent — each with their own isolated context window.

The `assemble` method brings everything together. It allocates the token budget across components (30% for history, 70% for RAG), loads memory, compresses history, retrieves relevant documents, and structures everything using XML tags for clear LLM parsing. Notice how every component has a clear boundary — `<system>`, `<memory>`, `<history>`, `<retrieved_context>`, `<query>` — so the model knows exactly what role each section plays.

---

## Learn the Art of Context Engineering

Now that we understand the theory and have seen the code, you might be asking: how do I actually develop this skill? Let us walk through a practical learning roadmap.

**Level 1: Start with System Prompts**

The simplest form of context engineering is writing a good system prompt. Practice writing system prompts at the "right altitude" — not a rigid if-else decision tree, and not a vague "be helpful."

Here is a concrete exercise: take any chatbot task and write a system prompt that constrains the model's behavior without hardcoding logic. For example, instead of "If the user asks about refunds, say 'Please contact support at support@company.com'", write "You are a customer service agent for Company X. You can help with product information and troubleshooting. For refund and billing inquiries, guide users to the support team."

Study how Claude Code's `CLAUDE.md` files are structured — they are a masterclass in persistent context design. They contain project conventions, build commands, and behavioral guidelines, all in clean Markdown.

**Level 2: Build a RAG Pipeline from Scratch**

Implement the basic pipeline we covered: chunk your documents, embed them, build a vector index, and retrieve top-K results for a query. The code from the previous sections gives you a starting point.

The key learning here: experiment with chunk sizes. Try 200 tokens, 500 tokens, and 1,000 tokens on the same document set and query, and watch how retrieval quality changes. You will develop an intuition for the precision-recall tradeoff that no tutorial can teach.

**Level 3: Master Token Budgeting**

For every LLM application you build, explicitly plan your token budget using the formula we derived. Ask yourself: how many tokens for instructions? For conversation history? For retrieved context? For output?

Build the habit of measuring what actually fills your context window. Most developers never look — and then they wonder why their agent degrades after a long conversation. The answer is almost always: the context filled up with noise.

**Level 4: Implement the Four Strategies**

Once you have the basics, start layering in the four strategies:

- **Write**: add a memory file to your agent that persists key facts across sessions
- **Select**: add just-in-time retrieval so your agent only fetches what it needs, when it needs it
- **Compress**: add a summarization step when conversation history exceeds a threshold
- **Isolate**: split a complex task into a lead agent and sub-agents with isolated contexts

Each strategy addresses a specific failure mode. Write prevents information loss. Select prevents confusion. Compress prevents distraction. Isolate prevents all four by giving each agent a clean slate.

**Level 5: Study the Products**

The best way to level up is to study what works in production:

- Read OpenClaw's source code — its memory architecture is remarkably clean and well-documented
- Use both Cursor and Windsurf and notice how differently they surface context — one asks you to curate, the other does it automatically
- Read Anthropic's "Effective Context Engineering for AI Agents" end to end — it is the most detailed practitioner guide available today
- Study Drew Breunig's failure mode taxonomy and actively diagnose which failure mode your application is hitting when things go wrong

**Essential Reading List:**

1. Anthropic, "Effective Context Engineering for AI Agents" (2025) — the definitive technical guide
2. Drew Breunig, "How Long Contexts Fail" (2025) — the failure mode taxonomy
3. LangChain / Lance Martin, "Context Engineering for Agents" (2025) — the four-strategy framework
4. Birgitta Bockeler / Martin Fowler, "Context Engineering for Coding Agents" (2026) — deep dive on coding tools
5. Addy Osmani, "Context Engineering: Bringing Engineering Discipline to Prompts" (O'Reilly, 2025) — the engineering discipline framing

Have a look at the learning roadmap below:


![The five-level learning roadmap for context engineering.](figures/figure_7.png)
*The five-level learning roadmap for context engineering.*


---

## The Future — Bigger Windows, Bigger Challenges

You might think that as context windows grow larger — from 4K tokens in early GPT-3 to 128K, 1M, and beyond — context engineering would become less important. After all, if you can fit everything in the window, why bother curating?

But the opposite is true. Larger windows make context engineering **more** important, not less.

Here is why. Recent research found that most models fall short of their advertised context window capabilities by over 99%. Some top models fail to accurately retrieve information with as few as 100 tokens of inserted context. Length does not equal capability. A million-token window with poorly organized context will perform worse than a 32K window with precisely curated, well-structured information.

Lance Martin of LangChain offers what he calls the "bitter lesson" for context engineering: as models improve, remove scaffolding. Simpler, more general approaches tend to outperform hand-tuned systems over time. Do not over-engineer your context pipeline for today's model limitations — tomorrow's model may handle things you are manually fixing.

But the core skill endures. As Anthropic puts it: *"Treating context as a precious, finite resource will remain central to building reliable, effective agents"* — regardless of how large the windows get.

Several emerging technologies are worth watching:

- **MCP (Model Context Protocol)** — Anthropic's open standard for connecting AI agents to external tools, now with 97M+ monthly SDK downloads and adoption by OpenAI, Google, and Microsoft
- **Learned context compression** — models that learn to compress context rather than relying on hand-written summarization rules
- **Context caching** — APIs that cache repeated context prefixes to reduce cost and latency
- **Attention steering** — techniques to explicitly guide where models attend in long contexts

Addy Osmani drew a compelling parallel: *"Context engineering will become similar to industrial engineering — those disciplines emerged to optimize machines that took over mechanical labor; context engineering is emerging to optimize machines taking over cognitive labor."*

---

## Conclusion

Let us take a step back and see the full picture. Context engineering is the discipline of designing the entire information environment that an LLM operates within. It goes far beyond writing good prompts — it encompasses what information to include, how to structure it, when to retrieve it, when to compress it, and how to persist it across sessions.

We learned that context has four distinct failure modes — poisoning, distraction, confusion, and clash — and four core strategies to combat them: write, select, compress, and isolate. We saw how products like OpenClaw, Cursor, Windsurf, Claude Code, and Augment Code have each found their own path through this design space, and we built our own context engine from scratch.

The Cognition team (the makers of Devin) put it best: context engineering has become *"effectively the #1 job of engineers building AI agents."*

Let us return to our open-book exam one last time. The smartest student with the worst notes will lose to a prepared student with the right notes every time. The same is true for LLMs. The model is only as good as the context you give it.

That's it! The next time you build with LLMs, remember — design the context, and the model will do the rest.

---

**References:**

- Lutke, Tobi, "Context engineering over prompt engineering" (June 2025)
- Karpathy, Andrej, "Context engineering is the delicate art and science of filling the context window" (June 2025)
- Anthropic, "Effective Context Engineering for AI Agents" (September 2025)
- Breunig, Drew, "How Long Contexts Fail" and "How to Fix Your Context" (June 2025)
- Martin, Lance / LangChain, "Context Engineering for Agents" (June 2025)
- Willison, Simon, "Context engineering is going to stick" (June 2025)
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023)
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- Bockeler, Birgitta / Martin Fowler, "Context Engineering for Coding Agents" (February 2026)
- Osmani, Addy, "Context Engineering: Bringing Engineering Discipline to Prompts" (O'Reilly, 2025)
- MIT Technology Review, "From Vibe Coding to Context Engineering" (November 2025)
- Steinberger, Peter, OpenClaw (github.com/openclaw/openclaw, 179K+ stars)
