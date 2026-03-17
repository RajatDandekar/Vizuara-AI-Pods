# Agentic Architecture & Orchestration

*Design production agentic loops, multi-agent coordinator patterns, and session management — the largest exam domain at 27%.*

---

Let us start with a concrete scenario. Imagine you are building a customer support agent for an e-commerce company. This agent can look up order details, check shipping status, process refunds, and — when things get complicated — escalate to a human representative. The remarkable thing? You never wrote a single if/else branch to decide which action to take. The model itself decides what to do, when to do it, and when it is done.

This is the essence of agentic architecture. In this article, we will build up from the simplest possible agent loop to sophisticated multi-agent systems with hooks, session management, and guaranteed compliance. These concepts form the largest domain on the Claude Certified Architect exam at 27%, so understanding them deeply is essential.

## 1. The Agentic Loop Lifecycle

Let us begin with the most fundamental concept: the agentic loop. Think of an agent like a chef in a kitchen. The chef reads the recipe (the prompt), checks what ingredients are available (the tools), starts cooking (executes tool calls), and — critically — decides when the dish is done. No one stands behind the chef saying "stop after exactly 5 steps." The chef tastes the food and decides.

The core agentic loop works as follows:

1. Send the conversation (system prompt + messages) to the model
2. Receive the model's response
3. Check the `stop_reason` in the response
4. If `stop_reason == "tool_use"` → the model wants to call a tool. Execute it, append the tool result to the conversation, and go back to step 1
5. If `stop_reason == "end_turn"` → the model has decided it is done. Return the final response

![Agentic loop lifecycle — the model drives the loop through stop_reason](figures/figure_1.png)
*Agentic loop lifecycle — the model drives the loop through stop_reason*

Here is a minimal implementation:

```python
import anthropic

client = anthropic.Anthropic()
tools = [{"name": "lookup_order", "description": "Look up order by ID",
          "input_schema": {"type": "object",
                           "properties": {"order_id": {"type": "string"}}}}]

messages = [{"role": "user",
             "content": "What is the status of order ORD-1234?"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # The model decides what happens next — not our code
    if response.stop_reason == "end_turn":
        print(response.content[0].text)
        break

    if response.stop_reason == "tool_use":
        tool_block = next(
            b for b in response.content if b.type == "tool_use"
        )
        result = execute_tool(tool_block.name, tool_block.input)

        # Append assistant response AND tool result to history
        messages.append({"role": "assistant",
                         "content": response.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result",
             "tool_use_id": tool_block.id,
             "content": result}
        ]})
```

Let us trace through a concrete execution to see how this works in practice.

**Turn 1:** The user asks "What is the status of order ORD-1234?" The model responds with `stop_reason: "tool_use"` and calls `lookup_order(order_id="ORD-1234")`. We execute the tool, get back `{"status": "shipped", "tracking": "TRK-5678"}`, and append it to the conversation.

**Turn 2:** The model now sees the tool result. It responds with `stop_reason: "end_turn"` and says: "Your order ORD-1234 has been shipped! The tracking number is TRK-5678." The loop exits.

Notice how the model itself decided it needed one tool call and then decided it was done. We never told it "call exactly one tool" or "stop after 2 turns." This is exactly what we want.

### Anti-Patterns to Avoid

There are two critical anti-patterns that the exam tests:

**Anti-Pattern 1: Parsing natural language for loop termination.** Never write code like `if "I'm done" in response.text: break`. The `stop_reason` field exists precisely for this purpose. Natural language is ambiguous — the model might say "I'm done checking the database" (meaning it finished one step) versus "I'm done" (meaning the task is complete). Using `stop_reason` removes all ambiguity.

**Anti-Pattern 2: Arbitrary iteration caps as the primary stopping mechanism.** Writing `for i in range(5): call_model()` treats the iteration limit as the main control flow. Instead, let the model drive termination through `stop_reason == "end_turn"`, and use iteration limits only as a safety net — a guard rail, not a steering wheel.

## 2. Multi-Agent Coordinator Patterns

Now that we understand how a single agent loop works, let us scale up. Imagine a hospital emergency room. When a patient arrives, a triage nurse does not treat them directly. Instead, the nurse assesses the situation, routes the patient to the right specialist (cardiologist, orthopedist, radiologist), and then coordinates the results from multiple specialists into a treatment plan.

This is exactly how multi-agent coordinator patterns work. We call this **hub-and-spoke architecture**: one coordinator agent at the hub, multiple specialized subagents as the spokes.

![Hub-and-spoke multi-agent architecture — coordinator decomposes, delegates, and aggregates](figures/figure_2.png)
*Hub-and-spoke multi-agent architecture — coordinator decomposes, delegates, and aggregates*

The coordinator has three responsibilities:

1. **Task Decomposition:** Breaking the overall task into subtasks suitable for each specialist
2. **Delegation:** Assigning each subtask to the right subagent with the right context
3. **Result Aggregation:** Combining the results from all subagents into a coherent final output

Here is the critical principle that the exam tests: **subagents have isolated context.** They do not inherit the coordinator's conversation history. Each subagent starts with a fresh context window containing only the specific instructions and data the coordinator passes to it.

Why does this matter? Consider our customer support scenario. If the coordinator has been handling 20 different customers, passing that entire conversation to a subagent that only needs to check one refund policy would waste context window space and could confuse the subagent with irrelevant information. Isolated context means focused, high-quality results.

```python
# Coordinator dispatches to a research subagent
research_result = await run_subagent(
    system_prompt="You are a research specialist.",
    prompt=f"Research the return policy for category: {category}",
    tools=["search_knowledge_base", "lookup_policy"]
    # Note: NO conversation history from coordinator is passed
)

# Coordinator dispatches to an analysis subagent
analysis_result = await run_subagent(
    system_prompt="You are a policy analyst.",
    prompt=f"Policy: {research_result}\nComplaint: {complaint}\n"
           f"Is a refund warranted?",
    tools=["calculate_refund"]
)
```

### The Risk of Over-Decomposition

A natural temptation is to make subagents hyper-specialized — one agent that only validates email addresses, one that only formats phone numbers, one that only checks zip codes. But this creates significant overhead. Each subagent invocation requires a full API call, context construction, and response parsing.

The exam guidance is clear: decompose into meaningful, coherent subtasks — not into the smallest possible units. A "customer data validation agent" that handles all input validation is better than five separate single-field validators.

## 3. Subagent Invocation and the Task Tool

Now the question is: how do we actually spawn these subagents in practice? This brings us to the Task tool and the Agent SDK.

In the Claude Agent SDK, subagents are spawned using the **Task tool**. For an agent to spawn subagents, its `allowedTools` configuration must include `"Task"`. Without this, the agent cannot create child agents.

Each subagent is defined through an **AgentDefinition** configuration:

```python
from claude_agent_sdk import AgentDefinition

research_agent = AgentDefinition(
    model="claude-sonnet-4-20250514",
    system_prompt="You are a research specialist. "
                  "Search for and summarize information.",
    tools=["web_search", "read_document"],
    max_tokens=4096
)
```

When the coordinator invokes the Task tool, it must pass all necessary context explicitly in the prompt. The subagent will not magically know what the coordinator knows — you must tell it:

```python
# Explicit context passing — subagent gets ONLY this prompt
task_result = coordinator.invoke_tool("Task", {
    "agent": research_agent,
    "prompt": f"""Research the following customer issue:
    Customer ID: {customer_id}
    Order ID: {order_id}
    Complaint: {complaint_text}
    Find the relevant return policy and precedent cases."""
})
```

### fork_session for Divergent Exploration

Sometimes you want to explore multiple approaches in parallel. The `fork_session` mechanism creates a branch point: the forked session gets a copy of the current context but can diverge without affecting the original.

Think of it like a save point in a video game. You save your progress, try a risky strategy, and if it does not work, you go back to the save point. In agent terms, you fork the session, let the forked agent explore one approach (say, solving a bug by refactoring), while the original continues with a different approach (say, adding a patch). Then you compare results.

## 4. Multi-Step Workflows and Compliance

Let us now look at a scenario where the stakes are high. Imagine a financial operations agent that can transfer funds, process refunds, and modify account settings. Before any of these actions, the agent must verify the customer's identity. This is not optional — it is a regulatory requirement.

The question is: how do we enforce this? There are two approaches:

**Approach 1: Prompt-based guidance.** We tell the agent in its system prompt: "Always verify the customer's identity before processing any financial transaction." This works most of the time, but "most of the time" is not acceptable for regulatory compliance. The model might occasionally skip verification if the customer sounds frustrated, if the conversation is long, or if the request seems straightforward.

**Approach 2: Programmatic enforcement.** We use hooks or prerequisite gates in code to make it impossible to execute a financial tool without prior identity verification:

```python
class IdentityGate:
    def __init__(self):
        self.verified = False

    def verify_identity(self, customer_id, verification_code):
        if check_verification(customer_id, verification_code):
            self.verified = True
            return "Identity verified successfully."
        return "Verification failed."

    def process_refund(self, order_id, amount):
        if not self.verified:
            return "ERROR: Identity must be verified first."
        return execute_refund(order_id, amount)
```

The exam is clear on this: **use programmatic enforcement for deterministic compliance requirements.** Identity verification before financial operations must never be left to prompt-based guidance alone. Save prompt-based guidance for softer requirements like "maintain a professional tone" or "ask clarifying questions when the request is ambiguous."

### Structured Handoff Protocols

When an agent needs to escalate to a human representative, it should not just say "transferring you now." A proper handoff protocol includes:

1. **Customer details:** Name, account ID, contact preferences
2. **Root cause analysis:** What the agent determined the issue to be
3. **Actions already taken:** What tools were called, what results were returned
4. **Recommended next steps:** What the agent thinks the human should do

```python
handoff = {
    "customer": {"id": "C-789", "name": "Alice Smith",
                 "tier": "premium"},
    "root_cause": "Duplicate charge on order ORD-456 "
                  "due to payment gateway timeout",
    "actions_taken": [
        {"tool": "lookup_order",
         "result": "Order found, payment processed twice"},
        {"tool": "check_refund_eligibility",
         "result": "Eligible for full refund"}
    ],
    "recommendation": "Process refund of $149.99 for duplicate charge"
}
```

This is exactly what we want — the human representative gets a complete picture and can act immediately rather than asking the customer to repeat everything.

## 5. Agent SDK Hooks — Guaranteed Compliance

We touched on programmatic enforcement in the previous section. Now let us go deeper into the specific mechanism the Agent SDK provides: **hooks**.

Hooks are functions that execute automatically at specific points in the agent's lifecycle. The two most important types for compliance are:

**PostToolUse hooks** fire after a tool has been called and returned a result. They can inspect and transform the result before the model sees it. A classic use case is **data normalization**.

Let us say your `lookup_order` tool returns timestamps as Unix epoch integers, but your system expects ISO 8601 strings. A PostToolUse hook handles this automatically:

```python
from datetime import datetime, timezone

def normalize_timestamps(tool_name, tool_result):
    """PostToolUse hook: convert Unix timestamps to ISO 8601."""
    if "timestamp" in tool_result:
        unix_ts = tool_result["timestamp"]
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        tool_result["timestamp"] = dt.isoformat()
    return tool_result
```

Let us trace through a concrete example. Suppose the tool returns `{"order_id": "ORD-1234", "timestamp": 1710000000}`. The Unix timestamp `1710000000` is not human-readable. After the hook runs, the model sees `{"order_id": "ORD-1234", "timestamp": "2024-03-09T16:00:00+00:00"}`. The model never has to parse raw Unix timestamps — it always gets clean, human-readable dates. This is exactly what we want.

![Hook interception pipeline — PostToolUse normalizes data, PreToolUse enforces policy](figures/figure_3.png)
*Hook interception pipeline — PostToolUse normalizes data, PreToolUse enforces policy*

**Tool call interception** hooks fire before a tool is executed and can block the call entirely. This is where you enforce hard business rules:

```python
def enforce_refund_policy(tool_name, tool_input):
    """PreToolUse hook: block refunds over $500."""
    if tool_name == "process_refund":
        amount = tool_input.get("amount", 0)
        if amount > 500:
            return {
                "blocked": True,
                "reason": f"Refund of ${amount} exceeds $500 limit. "
                          f"Manager approval required."
            }
    return {"blocked": False}
```

Why not just tell the model "never process refunds over \$500"? Because prompts are probabilistic. The model might process a \$750 refund if the customer provides a compelling reason. A hook is deterministic — it will block that refund 100% of the time, regardless of what the model wants to do.

This is the key exam insight: **hooks are for guaranteed compliance; prompts are for best-effort guidance.**

## 6. Task Decomposition Strategies

Let us now look at how agents break complex tasks into manageable pieces. There are two fundamentally different approaches.

### Fixed Sequential Pipelines (Prompt Chaining)

In a fixed pipeline, each step is predetermined. The output of step 1 feeds into step 2, which feeds into step 3. This is also called **prompt chaining**:

```
Step 1: Extract key entities from the document
    ↓ (entities)
Step 2: Research each entity in the knowledge base
    ↓ (research results)
Step 3: Write a summary incorporating the research
    ↓ (final summary)
```

This approach is predictable, debuggable, and easy to monitor. You know exactly what each step will do and in what order. It works well for workflows that are well-understood and rarely change.

### Dynamic Adaptive Decomposition

In dynamic decomposition, the model decides what to do next based on what it has learned so far. There is no predetermined pipeline — the agent maps the territory, assesses what it finds, and creates a prioritized plan:

```
Step 1: Map the codebase structure → discovers 47 files, 8 modules
Step 2: (Agent decides) Focus on the 3 most relevant modules
Step 3: (Agent decides) Analyze file A → finds suspicious pattern
Step 4: (Agent decides) Check if same pattern in file B → confirms bug
Step 5: (Agent decides) Generate and verify a fix
```

![Fixed sequential pipeline vs dynamic adaptive decomposition — choosing the right strategy](figures/figure_4.png)
*Fixed sequential pipeline vs dynamic adaptive decomposition — choosing the right strategy*

A particularly powerful pattern for code analysis is the **per-file analysis + cross-file integration pass**. First, the agent analyzes each relevant file independently, extracting key patterns and issues. Then, in a separate integration pass, it looks across all per-file results to find cross-cutting concerns, dependencies, and systemic patterns. This two-phase approach prevents the agent from getting lost in details while ensuring nothing is missed.

### When to Use Each

The exam tests your judgment here. A simple framework:

- **Fixed pipelines** when the workflow is well-defined, steps are known in advance, and reliability matters more than flexibility. Example: processing a loan application (verify identity → check credit → calculate terms → generate offer).
- **Dynamic decomposition** when the task is exploratory, the number of steps is unknown, and the agent needs to adapt based on what it discovers. Example: debugging a complex software issue or conducting open-ended research.

## 7. Session Management and Recovery

The final piece of agentic architecture is session management: how agents maintain continuity across interactions and recover from interruptions.

### Resuming Sessions with --resume

The `--resume` flag with a session name lets you continue an agent session exactly where it left off. The agent retains its full conversation history, tool results, and context:

```bash
# Start a session with a name
claude --session "refactoring-auth-module"

# Later, resume where you left off
claude --resume "refactoring-auth-module"
```

But there is an important subtlety: if files have changed since the last session, the agent's context may be stale. The tool results it remembers from the previous session might reference code that no longer exists. The exam tests whether you know to **inform the resumed session about changes**:

```
"Since your last session, these files were modified:
- src/auth/login.py (added 2FA support)
- src/auth/session.py (new file)
Please re-read these files before continuing."
```

### fork_session for Parallel Exploration

When you fork a session, you create a branch — like a git branch for conversations. The forked session starts with a copy of the current context but evolves independently.

This is powerful for exploring multiple approaches in parallel:

```
Original session: "Fix the bug by refactoring the parser"
Forked session:   "Fix the bug by adding input validation"
```

Both sessions work independently. When they finish, you compare results and pick the better approach. The forked session cannot affect the original session's state.

### When to Resume vs Start Fresh

The decision between resuming and starting fresh depends on how stale the context is:

**Resume** when the interruption was brief, files have not changed significantly, and the agent was mid-task.

**Start fresh with an injected summary** when significant time has passed, many files have changed, or tool results from the old session would be misleading. The injected summary approach means starting a new session but providing a concise recap: "In the previous session, we identified the authentication bug in the token refresh logic. We attempted retry logic but it did not resolve the issue. Please try a different approach."

## Putting It All Together

Let us revisit our customer support agent and see how all seven concepts combine:

1. **Agentic Loop (1.1):** The agent runs in a while loop, checking `stop_reason` to decide whether to call tools or return a response
2. **Multi-Agent Coordination (1.2):** Complex cases are delegated to specialist subagents via hub-and-spoke architecture
3. **Task Tool (1.3):** Subagents are spawned with explicit context — they do not inherit the main conversation
4. **Workflow Enforcement (1.4):** Identity verification is enforced programmatically before any financial operation
5. **Hooks (1.5):** PostToolUse hooks normalize data formats; PreToolUse hooks block policy-violating actions
6. **Task Decomposition (1.6):** Simple queries use direct tool calls; complex cases use dynamic decomposition
7. **Session Management (1.7):** Long-running cases use `--resume`; parallel investigation uses `fork_session`

### Exam Anti-Patterns Checklist

- Parsing natural language to determine loop termination instead of using `stop_reason`
- Using arbitrary iteration caps as the primary stopping mechanism
- Passing the coordinator's full conversation history to subagents
- Decomposing tasks into too many tiny subagents (over-decomposition)
- Using prompt-based guidance for hard compliance requirements
- Resuming stale sessions without informing the agent about file changes

That's it! You now have a comprehensive understanding of agentic architecture — from the fundamental while loop that drives every agent to the sophisticated multi-agent patterns, hooks, and session management strategies that make production agents reliable.
