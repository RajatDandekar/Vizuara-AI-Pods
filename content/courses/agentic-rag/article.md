# Agentic RAG Explained From Scratch

*How giving an LLM the ability to think before it retrieves transforms RAG from a lookup tool into a reasoning engine.*

---

## (1) The Two Librarians

Let us start with a simple example.

Imagine you walk into a massive library — thousands of shelves, millions of books — and you ask the librarian:

> "What are the latest breakthroughs in cancer immunotherapy?"

**Librarian A** hears your question, sprints to the nearest shelf labeled "cancer," grabs the top 5 books, and hands them to you. Fast? Absolutely. But when you open them, you find a 2003 textbook on skin cancer statistics, a pamphlet about sunscreen, and a novel where the protagonist has cancer. Not exactly what you wanted.

**Librarian B** takes a different approach. She pauses, thinks for a moment, and says: "You want *recent breakthroughs* in *immunotherapy* specifically. Let me first check the medical journals section for papers from 2023–2025. Then I will cross-reference with the latest conference proceedings. If I do not find enough, I will search the online database."

She goes, retrieves a few papers, flips through them, discards one that is about chemotherapy instead, and then goes back for one more search — this time specifically looking for CAR-T cell therapy results. Only when she is satisfied does she compile everything and give you a concise answer.

What is common in both these examples? Both librarians are doing *retrieval*. But Librarian B has something extra — **agency**. The ability to plan, decide, evaluate, and act. She reasons about *what* to retrieve, *evaluates* what she found, and *decides* whether to keep searching.

This is exactly the difference between RAG and Agentic RAG.


![Naive RAG vs Agentic RAG](figures/figure_1.png)
*Naive RAG is a simple linear pipeline (Query → Retrieve → Generate). Agentic RAG adds a reasoning loop — the agent evaluates results and re-retrieves if needed.*


---

## (2) A Quick RAG Refresher

Before we understand Agentic RAG, let us first make sure we have a solid understanding of standard RAG.

Large Language Models are incredibly powerful, but they have a fundamental limitation: **their knowledge is frozen at the time of training.** If you ask an LLM about something that happened after its training cutoff, it simply will not know. Even for information within its training data, the model might hallucinate — confidently generating plausible-sounding but incorrect answers.

Retrieval-Augmented Generation (RAG) solves this by giving the LLM access to external knowledge at inference time. Instead of relying solely on what it memorized during training, the LLM can *look things up* before answering.

The standard RAG pipeline has three steps:

**Step 1: Indexing**

We take our knowledge base — documents, PDFs, web pages, whatever we have — and break them into smaller chunks (typically 200–500 words each). Each chunk is converted into a numerical vector (an embedding) using an embedding model. These vectors are stored in a vector database like FAISS, Pinecone, or ChromaDB.

**Step 2: Retrieval**

When a user asks a question, we convert their query into an embedding using the same embedding model. Then we perform a similarity search in the vector database to find the top-k chunks that are most similar to the query. The most common similarity metric used here is cosine similarity.


$$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{||\mathbf{q}|| \cdot ||\mathbf{d}||}$$

Let us plug in some simple numbers to see how this works. Suppose our query embedding is $\mathbf{q} = [1, 2]$ and we have two document chunk embeddings: $\mathbf{d}_1 = [2, 3]$ and $\mathbf{d}_2 = [0, -1]$.

For $\mathbf{d}_1$:

$$\text{sim}(\mathbf{q}, \mathbf{d}_1) = \frac{(1)(2) + (2)(3)}{\sqrt{1^2 + 2^2} \cdot \sqrt{2^2 + 3^2}} = \frac{2 + 6}{\sqrt{5} \cdot \sqrt{13}} = \frac{8}{8.06} = 0.992$$

For $\mathbf{d}_2$:

$$\text{sim}(\mathbf{q}, \mathbf{d}_2) = \frac{(1)(0) + (2)(-1)}{\sqrt{5} \cdot \sqrt{1}} = \frac{-2}{2.236} = -0.894$$

This tells us that $\mathbf{d}_1$ is extremely similar to our query (cosine similarity close to 1) while $\mathbf{d}_2$ points in the opposite direction (negative similarity). So we would retrieve $\mathbf{d}_1$. This is exactly what we want.

**Step 3: Generation**

The retrieved chunks are stuffed into the LLM's prompt along with the original query. The LLM then generates an answer grounded in the retrieved context.

Have a look at the diagram below for the entire RAG pipeline:


![Standard RAG Pipeline](figures/figure_2.png)
*The standard RAG pipeline: documents are chunked and embedded offline (top row), then at query time the user's question is embedded, matched against the vector database, and the retrieved context is fed to the LLM (bottom row).*


---

## (3) Where Standard RAG Breaks Down

Now you might be thinking: this looks pretty good — what could possibly go wrong?

Well, quite a lot actually. Let us look at the four major failure modes of standard RAG.

**Failure Mode 1: Wrong Retrieval**

Suppose a user asks: "How does RLHF work for aligning language models?"

The retriever converts this to an embedding and does a similarity search. It finds chunks about "RL in robotics," "HF Transformers library documentation," and "language modeling basics." Why? Because there is keyword overlap — "RL," "language models" — but the retrieved chunks have nothing to do with Reinforcement Learning from Human Feedback specifically.

The retriever is matching surface-level similarity, not understanding *intent*.

**Failure Mode 2: Insufficient Retrieval**

Consider a complex question: "Compare the economic impact of COVID-19 on India, the United States, and Brazil, and explain how their government responses differed."

This question needs information from at least three different geographical contexts, covering both economics and government policy. A single retrieval pass that grabs the top-5 chunks will almost certainly not cover all three countries with both dimensions. The answer will be incomplete.

**Failure Mode 3: Unnecessary Retrieval**

User asks: "What is 2 + 2?"

Standard RAG dutifully retrieves 5 chunks from the vector database anyway. These chunks — maybe about number theory or arithmetic teaching methods — are now injected into the LLM's context. At best, this wastes tokens and adds latency. At worst, the irrelevant context actually *confuses* the model into giving a worse answer than it would have on its own.

**Failure Mode 4: Contradictory Information**

The vector database contains two chunks. Chunk A (from 2021) says: "The Pfizer vaccine requires two doses." Chunk B (from 2023) says: "The updated Pfizer vaccine is recommended as a single annual dose." Both are retrieved. The LLM has no way of knowing which one is more current or trustworthy — and might blend them into a confused, incorrect answer.

The core problem is this: **standard RAG has no brain.** It retrieves blindly and generates blindly. There is no step where the system *thinks* about whether retrieval is even needed, whether the retrieved results are good enough, or whether it should try a different search strategy.


![Four Failure Modes of Standard RAG](figures/figure_3.png)
*The four failure modes of standard RAG: wrong retrieval (irrelevant chunks), insufficient retrieval (not enough coverage), unnecessary retrieval (wastes tokens), and contradictory chunks (conflicting information).*


---

## (4) Enter Agentic RAG

This brings us to the main character in our story — **Agentic RAG**.

The idea is beautifully simple: instead of running a fixed retrieve-then-generate pipeline, we wrap the entire RAG process inside an **agent** — an LLM that can reason, plan, use tools, and make decisions in a loop.

Let us think about what this means concretely. In standard RAG, the flow is:

$$\text{Query} \rightarrow \text{Retrieve} \rightarrow \text{Generate}$$

It is a straight line. No thinking. No evaluation. No course correction.

In Agentic RAG, the flow becomes a *reasoning loop* with six stages:

1. **Route:** Does this query even need retrieval? Or can I answer it directly from my own knowledge?
2. **Plan:** What exactly should I search for? Should I decompose this complex question into simpler sub-queries?
3. **Retrieve:** Execute the search — possibly across multiple different sources.
4. **Evaluate:** Look at the retrieved chunks. Are they actually relevant? Are they sufficient to answer the question?
5. **Decide:** If the results are not good enough, should I refine my query and try again? Should I search a different source? Or do I have enough?
6. **Generate:** Synthesize the final answer from the accumulated context.

The critical insight is that steps 2 through 5 can repeat multiple times. The agent keeps looping until it is satisfied with the retrieved context. This is exactly what Librarian B was doing — she did not just grab books once. She searched, evaluated, discarded bad results, refined her search, and only then compiled her answer.

Have a look at the following diagram which shows the Agentic RAG reasoning loop:


![Agentic RAG Reasoning Loop](figures/figure_4.png)
*The Agentic RAG reasoning loop. The agent first decides if retrieval is needed. If yes, it enters a loop: Plan → Retrieve → Evaluate → and if results are insufficient, loop back and try again. Only when satisfied does it generate the final answer.*


---

## (5) The Building Blocks of Agentic RAG

Now let us understand the building blocks that make Agentic RAG work. There are four key components.

### (5a) The Router

The router is the first decision point. When a query comes in, the agent needs to decide: does this query need retrieval at all?

Consider these three queries:

- "What is the capital of France?" — The LLM already knows this. No retrieval needed.
- "What were our Q3 2024 revenue numbers?" — This requires searching the company's internal documents. Vector store retrieval.
- "What is the current weather in Mumbai?" — This needs a live API call, not a static document search.

The router directs each query down the right path. It is implemented as an LLM call with structured output — the LLM reads the query and returns a routing decision like `{"route": "vector_store"}` or `{"route": "direct_answer"}` or `{"route": "web_search"}`.

This single component already eliminates Failure Mode 3 (unnecessary retrieval) from our earlier list.


![Router Decision Tree](figures/figure_5.png)
*The router directs each query to the appropriate path: answer directly from LLM knowledge, search the vector store, or call a live API.*


### (5b) Query Decomposition

Some questions are too complex for a single retrieval pass. The agent can decompose a complex query into simpler sub-queries, retrieve for each one independently, and then merge the results.

Let us take an example. Suppose the user asks:

> "Compare the training approach of GPT-4 with Llama 3 and explain which one is more cost-efficient."

This is actually three questions packed into one:

- $q_1$: "What is the training approach of GPT-4?"
- $q_2$: "What is the training approach of Llama 3?"
- $q_3$: "Cost comparison of training GPT-4 vs Llama 3"

We can formalize this as:


$$Q_{\text{complex}} \rightarrow \{q_1, q_2, \ldots, q_n\}$$

$$C_{\text{final}} = \bigcup_{i=1}^{n} \text{Retrieve}(q_i)$$

Let us plug in some simple numbers. Suppose we decompose our complex query into $n = 3$ sub-queries, and for each sub-query we retrieve the top $k = 3$ most relevant chunks.

With a single query (standard RAG): we retrieve $k = 3$ chunks total. These 3 chunks might all be about GPT-4 and completely miss Llama 3.

With decomposition (Agentic RAG): we retrieve $3 \times 3 = 9$ chunks total — 3 about GPT-4's training, 3 about Llama 3's training, and 3 about cost comparisons. Even after removing duplicates, we end up with far better coverage.

This tells us that query decomposition dramatically improves recall for complex, multi-faceted questions. This is exactly what we want.

### (5c) The Grader

After retrieval, the agent does not blindly pass all chunks to the generator. Instead, it *grades* each retrieved chunk: "Is this chunk actually relevant to my query?"

This is implemented as an LLM call. The agent looks at each chunk and the original query, and returns a simple judgment — relevant or not relevant.

Here is what happens:

1. We retrieve 5 chunks
2. The grader examines each one
3. It marks 3 as relevant and 2 as irrelevant
4. The irrelevant chunks are discarded
5. If the number of relevant chunks falls below a threshold — say, fewer than 2 — the agent triggers a re-retrieval with a refined query

This grading step eliminates Failure Mode 1 (wrong retrieval). Even if the vector similarity search returns some bad results, the grader catches them before they pollute the final answer.


![The Grading Pipeline](figures/figure_6.png)
*The grading pipeline: each retrieved chunk is evaluated for relevance by the LLM. Relevant chunks proceed to generation; irrelevant ones are discarded. If too few relevant chunks remain, the agent refines the query and re-retrieves.*


### (5d) Self-Reflection

The final building block is the most powerful one. After the agent generates an answer, it can *reflect* on its own output:

- "Does my answer actually address the original question?"
- "Did I make any unsupported claims?"
- "Is there a gap in my answer that suggests I need more context?"

If the answer fails this self-check, the agent loops back — retrieves more context, and generates again. This is the safety net that catches hallucinations and incomplete answers.

Together, these four components — Router, Query Decomposition, Grader, and Self-Reflection — transform a rigid pipeline into a flexible, intelligent reasoning system.

---

## (6) Practical Implementation

Enough theory — let us look at a practical implementation now.

We will build a minimal Agentic RAG system in Python. Our system will have the following components:

1. A FAISS vector store with a few sample documents
2. A router that decides whether retrieval is needed
3. A grader that evaluates chunk relevance
4. An orchestrator that runs the agentic loop

Let us start by setting up the vector store.

```python
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI()

# Our knowledge base — a few sample documents
documents = [
    "RLHF uses human preferences to fine-tune language models via a reward model.",
    "RAG retrieves external documents to augment LLM generation at inference time.",
    "Transformers use self-attention to process sequences in parallel.",
    "FAISS is a library for efficient similarity search developed by Meta.",
    "PPO is a policy gradient method that clips updates to prevent large policy changes.",
    "Vector databases store embeddings for fast approximate nearest neighbor search.",
    "LoRA reduces fine-tuning cost by training low-rank adapter matrices.",
    "Chain-of-thought prompting improves reasoning by asking the model to show its work.",
]

# Create embeddings and build FAISS index
def get_embeddings(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([e.embedding for e in response.data], dtype="float32")

doc_embeddings = get_embeddings(documents)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])  # Inner product (cosine sim for normalized vecs)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

def retrieve(query, k=3):
    q_emb = get_embeddings([query])
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    return [(documents[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
```

Let us understand this code. We define 8 sample documents as our knowledge base. We use OpenAI's embedding model to convert each document into a vector. These vectors are normalized and stored in a FAISS index that uses inner product search (which is equivalent to cosine similarity for normalized vectors). The `retrieve` function takes a query, embeds it, and returns the top-k most similar chunks along with their similarity scores.

Now let us build the router. The router decides whether a query should go through retrieval or be answered directly.

```python
def route_query(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a query router. Given a query, decide:
- "retrieve" if the query needs external document search
- "direct" if the LLM can answer from general knowledge
Respond with ONLY one word: retrieve or direct."""},
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()
```

Simple and effective. The LLM reads the query and returns either "retrieve" or "direct." No fancy framework needed — just a well-crafted system prompt.

Next, the grader. After we retrieve chunks, the grader evaluates whether each chunk is actually relevant.

```python
def grade_chunk(query, chunk):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a relevance grader. Given a user query
and a retrieved document chunk, determine if the chunk is relevant to answering the query.
Respond with ONLY one word: relevant or irrelevant."""},
            {"role": "user", "content": f"Query: {query}\n\nChunk: {chunk}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower() == "relevant"
```

Again, a simple LLM call with a focused system prompt. It returns `True` if the chunk is relevant, `False` otherwise.

Now let us put it all together with the orchestrator — this is the brain of our Agentic RAG system.

```python
def agentic_rag(query, max_retries=2):
    print(f"Query: {query}\n")

    # Step 1: Route
    route = route_query(query)
    print(f"Router decision: {route}")

    if route == "direct":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content

    # Step 2: Retrieve and Grade (with retry loop)
    relevant_chunks = []
    for attempt in range(max_retries):
        retrieved = retrieve(query, k=5)
        print(f"\nAttempt {attempt + 1}: Retrieved {len(retrieved)} chunks")

        for chunk, score in retrieved:
            if grade_chunk(query, chunk):
                relevant_chunks.append(chunk)
                print(f"  [RELEVANT] (score={score:.3f}) {chunk[:60]}...")
            else:
                print(f"  [FILTERED] (score={score:.3f}) {chunk[:60]}...")

        if len(relevant_chunks) >= 2:
            break
        print("Not enough relevant chunks — refining search...")
        query = f"More details about: {query}"  # Simple query refinement

    # Step 3: Generate with relevant context
    context = "\n".join(relevant_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer using ONLY this context:\n{context}"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content
```

Let us understand this code in detail. The orchestrator follows the exact reasoning loop we discussed earlier:

1. **Route:** It first calls `route_query` to decide if retrieval is needed. If the answer is "direct," it skips retrieval entirely and answers from the LLM's own knowledge.
2. **Retrieve + Grade:** If retrieval is needed, it retrieves 5 chunks and grades each one. Relevant chunks are kept; irrelevant ones are discarded.
3. **Retry loop:** If fewer than 2 chunks pass the grading step, the agent refines the query and tries again — up to `max_retries` times.
4. **Generate:** Finally, the relevant chunks are assembled into context, and the LLM generates a grounded answer.

This is a minimal implementation, but it captures the essential pattern. In production, you would add query decomposition, multiple retrieval sources, more sophisticated query refinement, and self-reflection on the generated answer.


![Implementation Architecture](figures/figure_7.png)
*Architecture of our Agentic RAG implementation. The orchestrator routes queries, retrieves from FAISS, grades chunks for relevance, and loops back if insufficient relevant chunks are found.*


---

## (7) Agentic RAG in the Wild

Now that we understand how Agentic RAG works, let us look at where it is being used in the real world.

**Multi-Document Research Assistants**

In legal, medical, and financial domains, answering a single question often requires synthesizing information from dozens of documents. A lawyer asking "What are the precedents for data privacy violations in the EU?" needs the agent to search across case law databases, regulatory documents, and legal commentary — evaluating and cross-referencing as it goes. Standard RAG would grab 5 chunks and hope for the best. Agentic RAG decomposes the query, searches multiple sources, and iterates until coverage is sufficient.

**Customer Support Bots**

A customer asks: "My order #12345 hasn't arrived and I want a refund." The agentic system routes this to: (1) the order tracking API for shipment status, (2) the refund policy documents for eligibility rules, and (3) the customer's account history for prior interactions. It synthesizes all three before responding. If the order tracking API returns an error, the agent falls back to the shipping FAQ documents instead of giving up.

**Code Assistants**

A developer asks: "How do I implement pagination with our custom ORM?" The agent decides to search the project's internal documentation first. If the docs do not cover pagination, it falls back to searching the codebase for existing pagination implementations. If that also fails, it searches external documentation. This multi-source, fallback-driven retrieval is classic Agentic RAG.

**Corrective RAG (CRAG)**

A particularly interesting framework is Corrective RAG (CRAG), which formalizes the self-correction loop. In CRAG, after retrieval, a lightweight evaluator scores the relevance of results. If the confidence is high, it proceeds. If ambiguous, it refines the retrieval. If confidence is low, it falls back entirely to web search. This is Agentic RAG with a well-defined decision boundary.


![Agentic RAG in the Real World](figures/figure_8.png)
*Three real-world applications of Agentic RAG: a research assistant that searches multiple legal sources, a customer support bot that combines APIs with policy documents, and a code assistant that cascades through internal docs, codebase search, and external documentation.*


---

## (8) Standard RAG vs. Agentic RAG

Let us now put everything together and compare standard RAG with Agentic RAG side by side.

| Feature | Standard RAG | Agentic RAG |
|---|---|---|
| **Retrieval** | Single-pass, fixed | Multi-pass, adaptive |
| **Query Handling** | Uses query as-is | Decomposes, refines, reformulates |
| **Evaluation** | None — all chunks used | Grading + self-reflection |
| **Routing** | Always retrieves | Decides *if* retrieval is needed |
| **Error Recovery** | None | Re-retrieval, source fallback |
| **Complexity** | Low | Moderate |
| **Latency** | Low (1 LLM call) | Higher (multiple LLM calls) |
| **Cost** | Lower | Higher (more API calls) |
| **Best For** | Simple factual lookups | Complex, multi-hop questions |

An important point: Agentic RAG is not *always* better. For simple factual lookups where the retriever consistently returns good results, standard RAG is faster, cheaper, and perfectly adequate. You do not need an agent to answer "What is our company's return policy?" — a single retrieval pass will do.

But for complex, multi-hop questions where accuracy matters — questions that require information from multiple sources, that need disambiguation, or where retrieval quality is unpredictable — Agentic RAG wins. The extra LLM calls are a small price to pay for dramatically better answers.

The way I think about it is this: standard RAG is like a vending machine — you press a button and get what it gives you. Agentic RAG is like a personal assistant — it thinks about what you need, goes and finds it, checks if it is right, and tries again if it is not.

---

## (9) Conclusion

Let us take a step back and see what we have covered.

We started with a simple analogy — two librarians, one that grabs books blindly and one that *thinks* before retrieving. We then reviewed the standard RAG pipeline and identified its four fundamental failure modes: wrong retrieval, insufficient retrieval, unnecessary retrieval, and contradictory information.

This motivated us to introduce Agentic RAG — wrapping the RAG pipeline inside a reasoning agent. We broke down its four building blocks: the **Router** (decides if retrieval is needed), **Query Decomposition** (breaks complex questions into sub-queries), the **Grader** (evaluates chunk relevance), and **Self-Reflection** (checks the generated answer for gaps).

We then built a working implementation from scratch in Python — a router, a grader, and an agentic loop — and saw how these components work together in real-world applications from legal research to customer support.

The key takeaway is this: **Agentic RAG is standard RAG with a brain.** The agent decides *when* to retrieve, *what* to retrieve, whether the results are *good enough*, and whether to *try again*. This is exactly what we want.

In the next article, we will take this further and look at how we can build a **multi-agent RAG system** where specialized agents — a planner, a retriever, a critic, and a synthesizer — collaborate to answer complex research questions that no single agent could handle alone. Stay tuned!

---

## References

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- Yan et al., "Corrective Retrieval Augmented Generation" (2024)
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)
- Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey" (2024)
- Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (2023)
