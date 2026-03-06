# Memory Architectures for LLM Applications

*How to give your LLM the ability to remember — from simple buffers to hybrid long-term memory systems*

---

Let us start with a simple experiment. Open any chatbot — ChatGPT, Claude, Gemini — and have a long conversation with it. Talk about your project, mention your budget, share your team members' names, discuss your preferences. After about 50 messages, ask it: "What was the budget I mentioned at the start?"

If you are lucky, it will remember. If you are not, it will either hallucinate a number or politely admit it does not recall.

This is the **goldfish problem**. Large Language Models are, by default, completely stateless. Every single API call starts with a blank slate. The model has no memory of what you said five minutes ago — unless you explicitly pass it in the prompt.

Think of it like calling a brilliant consultant on the phone who has amnesia. Every time you call, they have no idea who you are. You have to re-explain everything from scratch. The consultant is incredibly smart — they just cannot remember anything between calls.


![LLMs are stateless — no information carries between API calls.](figures/figure_1.png)
*LLMs are stateless — no information carries between API calls.*


Now, the context window gives us a partial solution. Modern LLMs accept very long prompts — GPT-4 supports 128K tokens, Claude supports 200K tokens. We could, in principle, stuff the entire conversation history into the prompt every time. But this is expensive, slow, and eventually runs into the context window ceiling.

So how do we give our LLM the ability to remember? This is where **memory architectures** come in. Let us walk through them one by one, from the simplest to the most sophisticated.

---

## The Simplest Memory: Conversation Buffer

The most intuitive solution is the simplest one: just keep a list of every message in the conversation and send the entire list to the LLM every time.

Here is a minimal implementation:

```python
class ConversationBufferMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_context(self):
        return "\n".join(
            f"{m['role']}: {m['content']}" for m in self.messages
        )

    def build_prompt(self, system_prompt, user_input):
        history = self.get_context()
        return f"{system_prompt}\n\nConversation so far:\n{history}\n\nUser: {user_input}\nAssistant:"
```

Let us trace through a short conversation to see how this works. Suppose the user sends 5 messages, and the assistant responds to each:

| Turn | User Message | Token Count (approx) |
|------|-------------|---------------------|
| 1 | "My name is Raj and I'm building a chatbot." | 12 |
| 2 | "The budget is $5,000." | 8 |
| 3 | "I want to use Python and FastAPI." | 10 |
| 4 | "The target audience is university students." | 10 |
| 5 | "Can you suggest a database?" | 8 |

By turn 5, our buffer contains all 5 user messages plus 5 assistant responses. The total context we send to the LLM grows with every turn.

The token cost of the history grows linearly:


$$
C(n) = \sum_{i=1}^{n} (t_i^{\text{user
$$
 + t_i^{\text{assistant}})}}

where $t_i^{\text{user}}$ and $t_i^{\text{assistant}}$ are the token counts of the user and assistant messages at turn $i$.

Let us plug in some simple numbers. Suppose each turn averages 200 tokens (user + assistant combined). After 5 turns, we send $C(5) = 5 \times 200 = 1{,}000$ tokens of history. After 50 turns, that becomes $C(50) = 50 \times 200 = 10{,}000$ tokens. After 500 turns, we would need $C(500) = 500 \times 200 = 100{,}000$ tokens — just for history alone. This is a problem.


![Token usage grows linearly — eventually hitting the context window ceiling.](figures/figure_2.png)
*Token usage grows linearly — eventually hitting the context window ceiling.*


This clearly will not scale. Can we be smarter about what we keep?

---

## Sliding Window Memory

The simplest fix is to keep only the last $K$ turns. Old messages fall off. New messages enter. The memory has a fixed size, like a window sliding across the conversation.

```python
from collections import deque

class SlidingWindowMemory:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.messages = deque(maxlen=window_size * 2)  # user + assistant pairs

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_context(self):
        return "\n".join(
            f"{m['role']}: {m['content']}" for m in self.messages
        )
```

Now the token budget is bounded:


$$
C_{\text{max
$$
 = K \times \bar{t}}}

where $K$ is the window size and $\bar{t}$ is the average tokens per turn.

Let us plug in numbers. If $K = 10$ and each turn averages 200 tokens, then $C_{\text{max}} = 10 \times 200 = 2{,}000$ tokens. No matter how long the conversation runs — 100 turns, 1,000 turns — we never use more than 2,000 tokens for history. This is exactly what we want.


![Sliding window keeps only the last K turns — old messages are dropped.](figures/figure_3.png)
*Sliding window keeps only the last K turns — old messages are dropped.*


But there is an obvious problem. If the user stated their name in turn 1 and we are now on turn 20 with $K = 5$, the name is gone. The LLM has no idea who it is talking to.

What if, instead of keeping recent messages verbatim, we could compress the old ones?

---

## Summary Memory

Here is a clever idea: use the LLM itself to summarize the conversation history. Instead of storing every message, we maintain a running summary of the conversation so far, and append the recent messages verbatim.

```python
from collections import deque

class SummaryMemory:
    def __init__(self, llm_client, recent_turns=5):
        self.llm_client = llm_client
        self.recent_turns = recent_turns
        self.summary = ""
        self.recent_messages = deque(maxlen=recent_turns * 2)
        self.all_messages = []

    def add_message(self, role, content):
        self.all_messages.append({"role": role, "content": content})
        self.recent_messages.append({"role": role, "content": content})
        # Re-summarize when old messages accumulate
        if len(self.all_messages) > self.recent_turns * 2:
            self._update_summary()

    def _update_summary(self):
        old_messages = self.all_messages[:-self.recent_turns * 2]
        history = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)
        self.summary = self.llm_client.generate(
            f"Summarize this conversation concisely:\n{history}"
        )

    def get_context(self):
        parts = []
        if self.summary:
            parts.append(f"Summary of earlier conversation:\n{self.summary}")
        recent = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.recent_messages
        )
        parts.append(f"Recent messages:\n{recent}")
        return "\n\n".join(parts)
```

The token budget now looks like this:


$$
C = t_{\text{summary
$$
 + \sum_{i=n-K+1}^{n} t_i}}

where $t_{\text{summary}}$ is the token count of the compressed summary and the sum covers the last $K$ turns.

Let us work through a numerical example. Suppose we have had 50 turns. The summary of the first 40 turns compresses to 300 tokens. The last 10 turns use 2,000 tokens verbatim. Total: $C = 300 + 2{,}000 = 2{,}300$ tokens — regardless of whether the conversation has been 50 or 500 turns long. This is a massive improvement over the buffer approach.


![Old messages are compressed into a summary; recent messages are kept verbatim.](figures/figure_4.png)
*Old messages are compressed into a summary; recent messages are kept verbatim.*


But summaries are lossy. When you compress 40 turns into 300 tokens, specific details inevitably get dropped. The LLM summarizer might write "The user is building a chatbot project" but lose the fact that the budget was exactly \$5,000 or that the deployment deadline is March 15.


![Summaries preserve themes but lose specific facts like numbers and dates.](figures/figure_5.png)
*Summaries preserve themes but lose specific facts like numbers and dates.*


What if we could extract and store specific facts from the conversation?

---

## Entity Memory

Entity memory takes a different approach. Instead of summarizing everything into prose, we extract **specific entities** — people, numbers, preferences, dates — and store them in a structured dictionary.

After each turn, we ask the LLM: "What new entities or facts were mentioned in this message?" The extracted entities are stored in a key-value store, and relevant ones are injected into the prompt.

```python
import json
from collections import deque

class EntityMemory:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.entities = {}
        self.recent_messages = deque(maxlen=10)

    def add_message(self, role, content):
        self.recent_messages.append({"role": role, "content": content})
        self._extract_entities(content)

    def _extract_entities(self, text):
        prompt = f"""Extract key entities and facts from this text.
Return as JSON: {{"entity_name": "description"}}
Text: {text}"""
        result = self.llm_client.generate(prompt)
        new_entities = json.loads(result)
        self.entities.update(new_entities)

    def get_relevant_entities(self, query):
        relevant = {}
        for name, desc in self.entities.items():
            if name.lower() in query.lower():
                relevant[name] = desc
        return relevant
```


![Entity memory extracts specific facts and retrieves them when relevant.](figures/figure_6.png)
*Entity memory extracts specific facts and retrieves them when relevant.*


The token cost of entity memory is predictable:


$$
C_{\text{entity
$$
 = t_{\text{entities}} + t_{\text{recent}}}}

Let us plug in numbers. Suppose we have extracted 15 entities, each averaging 30 tokens of description. That is $15 \times 30 = 450$ tokens. Plus 1,000 tokens of recent messages. Total: $C = 450 + 1{,}000 = 1{,}450$ tokens. Not bad, right?

But entity extraction is imperfect. The LLM might miss subtle entities, or struggle to capture relationships between them. And what about semantic similarity — finding memories that are *relevant* to the current query even if they do not share exact entity names?

This brings us to the most powerful memory architecture.

---

## Vector-Backed Long-Term Memory

The idea is elegant: take every message in the conversation, convert it into a numerical vector using an embedding model, and store it in a vector database. When the user asks a question, embed the query and find the most similar messages from the entire conversation history.


![Messages are embedded as vectors; retrieval finds semantically similar memories.](figures/figure_7.png)
*Messages are embedded as vectors; retrieval finds semantically similar memories.*


The core of this approach is **cosine similarity** — measuring how "similar" two vectors are by computing the cosine of the angle between them:


$$
\text{sim}(\mathbf{q}, \mathbf{m}_i) = \frac{\mathbf{q} \cdot \mathbf{m}_i}{\|\mathbf{q}\| \|\mathbf{m}_i\|
$$
}

Let us work through a concrete example. Suppose we have a 2D embedding space (real embeddings have hundreds of dimensions, but the math is identical). Our query vector is $\mathbf{q} = [0.8, 0.6]$ and we have two memory vectors:

- $\mathbf{m}_1 = [0.7, 0.7]$ — a message about "Python web development"
- $\mathbf{m}_2 = [0.1, 0.9]$ — a message about "vacation plans"

For $\mathbf{m}_1$:

$$\text{sim}(\mathbf{q}, \mathbf{m}_1) = \frac{(0.8)(0.7) + (0.6)(0.7)}{\sqrt{0.64 + 0.36} \times \sqrt{0.49 + 0.49}} = \frac{0.56 + 0.42}{1.0 \times 0.99} = \frac{0.98}{0.99} = 0.99$$

For $\mathbf{m}_2$:

$$\text{sim}(\mathbf{q}, \mathbf{m}_2) = \frac{(0.8)(0.1) + (0.6)(0.9)}{\sqrt{0.64 + 0.36} \times \sqrt{0.01 + 0.81}} = \frac{0.08 + 0.54}{1.0 \times 0.91} = \frac{0.62}{0.91} = 0.68$$

Memory $\mathbf{m}_1$ has a similarity of 0.99 while $\mathbf{m}_2$ has only 0.68. The system correctly retrieves $\mathbf{m}_1$ as the more relevant memory. This is exactly what we want.

We then retrieve the top-$k$ most similar memories:


$$
\mathcal{M}_{\text{retrieved
$$
 = \text{top-}k \left( \{ \text{sim}(\mathbf{q}, \mathbf{m}_i) \}_{i=1}^{N} \right)}}

Here is a practical implementation using FAISS:

```python
import numpy as np
import faiss

class VectorMemory:
    def __init__(self, embedding_fn, dim=384, top_k=5):
        self.embedding_fn = embedding_fn
        self.top_k = top_k
        self.index = faiss.IndexFlatIP(dim)  # Inner product index
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        vec = self.embedding_fn(content)
        vec = vec / np.linalg.norm(vec)  # Normalize for cosine similarity
        self.index.add(vec.reshape(1, -1).astype("float32"))

    def retrieve(self, query):
        q_vec = self.embedding_fn(query)
        q_vec = q_vec / np.linalg.norm(q_vec)
        scores, indices = self.index.search(
            q_vec.reshape(1, -1).astype("float32"), self.top_k
        )
        return [self.messages[i] for i in indices[0] if i < len(self.messages)]
```


![The query finds semantically similar memories regardless of exact keyword match.](figures/figure_8.png)
*The query finds semantically similar memories regardless of exact keyword match.*


The beauty of vector memory is that it scales to thousands of messages without increasing the token count sent to the LLM. We always retrieve only the top-$k$ most relevant memories. And the retrieval is *semantic* — even if the user asks "How much can I spend?" the system will find the message where they said "The budget is \$5,000" because the embeddings capture meaning, not just keywords.

But purely similarity-based retrieval has its own limitations. Sometimes the most important context is not the most similar — it is the most *recent*, or the most *important*. Can we combine the best of all approaches?

---

## Hybrid Memory Architecture

The most robust approach is to combine multiple memory types, each contributing a different kind of context. Think of it as giving your LLM several different memory systems — just like the human brain has short-term memory, long-term memory, and episodic memory working together.

A hybrid architecture typically combines:

1. **Short-term memory** (sliding window) — the last few turns, verbatim
2. **Long-term memory** (vector store) — semantically retrieved older messages
3. **Entity store** — structured facts extracted from the conversation
4. **Running summary** — compressed overview of the full conversation arc


![Hybrid memory combines four subsystems — each fills a different role in context.](figures/figure_9.png)
*Hybrid memory combines four subsystems — each fills a different role in context.*


The key design decision is **budget allocation** — how to divide the context window across memory types:


$$
T_{\text{total
$$
 = T_{\text{system}} + T_{\text{summary}} + T_{\text{entities}} + T_{\text{retrieved}} + T_{\text{recent}} + T_{\text{response}}}}

Let us plug in a concrete example. Suppose our model supports 8,000 tokens:

| Component | Token Budget |
|-----------|-------------|
| System prompt | 500 |
| Running summary | 300 |
| Entity store | 200 |
| Retrieved memories (top-5) | 1,000 |
| Recent turns (last 5) | 2,000 |
| Response budget | 4,000 |
| **Total** | **8,000** |

This allocation ensures that the LLM always has the big picture (summary), specific facts (entities), relevant past context (vector retrieval), immediate conversational flow (recent turns), and enough room to generate a response.

Here is a simplified orchestrator:

```python
class HybridMemory:
    def __init__(self, llm_client, embedding_fn):
        self.window = SlidingWindowMemory(window_size=5)
        self.vector = VectorMemory(embedding_fn)
        self.entities = EntityMemory(llm_client)
        self.summary = SummaryMemory(llm_client, recent_turns=5)

    def add_message(self, role, content):
        self.window.add_message(role, content)
        self.vector.add_message(role, content)
        self.entities.add_message(role, content)
        self.summary.add_message(role, content)

    def build_context(self, user_query):
        retrieved = self.vector.retrieve(user_query)
        relevant_entities = self.entities.get_relevant_entities(user_query)

        context_parts = []
        if self.summary.summary:
            context_parts.append(f"## Conversation Summary\n{self.summary.summary}")
        if relevant_entities:
            entity_str = "\n".join(f"- {k}: {v}" for k, v in relevant_entities.items())
            context_parts.append(f"## Key Facts\n{entity_str}")
        if retrieved:
            mem_str = "\n".join(f"- {m['role']}: {m['content']}" for m in retrieved)
            context_parts.append(f"## Relevant Past Messages\n{mem_str}")
        context_parts.append(f"## Recent Conversation\n{self.window.get_context()}")

        return "\n\n".join(context_parts)
```

---

## Putting It All Together

Now that we have seen five different memory architectures, how do we choose the right one? Here is a practical decision framework:


![Each architecture trades off between token efficiency, detail, and complexity.](figures/figure_10.png)
*Each architecture trades off between token efficiency, detail, and complexity.*


Here are the quick guidelines:

- **Simple chatbot** with short conversations → Sliding window is sufficient
- **Customer support bot** that needs to remember specifics → Entity memory + sliding window
- **Knowledge assistant** searching over long histories → Vector memory
- **Production personal assistant** → Full hybrid architecture

In production systems, you will also want to consider:
- **Persistence** — storing memories across sessions in a database
- **Multi-session memory** — remembering facts from previous conversations
- **Memory decay** — gradually reducing the importance of very old memories
- **Memory deduplication** — avoiding storing the same fact multiple times

---

## Closing

We started with the goldfish problem — LLMs that forget everything between API calls. We then walked through five progressively more sophisticated solutions: conversation buffers that grow without bound, sliding windows that trade history for predictability, summary memory that compresses the past, entity memory that preserves specific facts, and vector-backed memory that retrieves relevant context semantically.

The key insight is this: **memory architecture is context engineering.** Every approach we discussed is ultimately about the same thing — filling the context window with the right information at the right time. The "right" approach depends entirely on what your application needs to remember and how much context budget you have.

That's it!

---

**References:**
- LangChain Memory Documentation
- MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)
- Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)
