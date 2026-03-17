# Tool Design & MCP Integration

*Craft tool descriptions that guide reliable selection, structure error responses for intelligent recovery, and configure MCP servers for production agent workflows*

---

Let us start with a debugging story. You have built a customer support agent using the Claude Agent SDK. It has four MCP tools: `get_customer`, `lookup_order`, `process_refund`, and `escalate_to_human`. The agent works well in simple cases — a customer asks about their order, the agent calls `lookup_order`, retrieves the details, and responds.

But then something strange happens. A customer writes: "I need to update my shipping address." The agent calls `lookup_order`. That is not right — it should call `get_customer` to pull up the customer profile. You check the logs. It happens again and again: whenever the request is even slightly ambiguous, the agent picks the wrong tool.

You look at the tool descriptions:

```
get_customer: "Retrieves customer information"
lookup_order: "Looks up order information"
```

There is your problem. These descriptions tell the model almost nothing about *when* to use each tool. "Customer information" and "order information" overlap heavily — an order contains customer data, and a customer record references orders. The model is essentially flipping a coin.

This is the central lesson of Domain 2 of the Claude Certified Architect exam: **the quality of your tools is determined not by their implementation, but by how clearly you communicate their purpose to the model.** In this article, we will cover the five core skills: writing tool descriptions that differentiate, structuring error responses for intelligent recovery, distributing tools across agents, configuring MCP servers, and selecting the right built-in tools.

---

## Tool Descriptions Are the Selection Mechanism

Let us be very precise about what happens when Claude decides which tool to call. The model receives a list of available tools, each with a name, a description, and an input schema. It reads the user's request, reads the tool descriptions, and selects the tool whose description best matches the intent.

This means the description is not documentation for humans — it is the **primary selection signal** for the model. A minimal description leads to unreliable selection. A detailed, differentiated description leads to reliable selection.

Think of it like a restaurant menu. If every dish is listed as "food — tastes good," you would order randomly. But if one entry says "Wood-fired margherita pizza — thin crust, San Marzano tomatoes, fresh mozzarella, basil. Best for: a quick, classic Italian meal" and another says "Slow-braised lamb shank — 8-hour cook, red wine reduction, root vegetables. Best for: a hearty, warming dinner" — you know exactly what you are getting.


![How Claude uses tool descriptions to select among available tools](figures/figure_1.png)
*How Claude uses tool descriptions to select among available tools*


### What Makes a Good Tool Description

A good tool description answers five questions:

1. **What does this tool do?** — One sentence, specific and unambiguous
2. **What are the expected inputs?** — Format, types, constraints
3. **What does it return?** — Structure and content of the response
4. **When should you use it?** — Positive examples and use cases
5. **When should you NOT use it?** — Boundary with similar tools

Let us rewrite our customer support tools:

**Before (bad):**
```
get_customer: "Retrieves customer information"
lookup_order: "Looks up order information"
```

**After (good):**
```
get_customer: "Retrieves a customer's profile by email or customer ID.
Returns: name, email, shipping address, account status, loyalty tier.
Use for: address changes, account questions, loyalty inquiries.
Do NOT use for: order-specific questions (use lookup_order instead)."

lookup_order: "Retrieves order details by order ID or tracking number.
Returns: items, quantities, prices, shipping status, delivery date.
Use for: order status, delivery tracking, item-specific questions.
Do NOT use for: customer profile updates (use get_customer instead)."
```

Now the model has clear signals. "Update my shipping address" maps to `get_customer` because the description explicitly mentions address changes. "Where is my package?" maps to `lookup_order` because it mentions delivery tracking.

### Renaming and Splitting Overlapping Tools

Sometimes the problem is not just descriptions — the tools themselves overlap. Consider a research agent with two tools:

```
analyze_content: "Analyzes content and returns insights"
analyze_document: "Analyzes a document and returns insights"
```

These are functionally identical from the model's perspective. The fix has two steps:

**Step 1: Rename to differentiate.** Change `analyze_content` to `extract_web_results` with a web-specific description. Now the names alone tell the model which context each tool serves.

**Step 2: Split generic tools into purpose-specific ones.** If `analyze_document` does three different things — extracting data points, summarizing content, and fact-checking — split it into three tools:

- `extract_data_points` — "Extracts structured data (names, dates, amounts) from a document"
- `summarize_content` — "Generates a concise summary of a document's main arguments"
- `verify_claim_against_source` — "Checks whether a specific claim is supported by the source document"

Each tool now has a single, clear purpose. The model will select the right one because there is no ambiguity.

### The System Prompt Keyword Trap

Here is a subtle failure mode. You write a system prompt that says: "When the user mentions analysis, always analyze the content thoroughly before responding."

The word "analyze" in the system prompt creates an unintended association with any tool containing "analyze" in its name or description. The model sees the keyword match and preferentially selects `analyze_content` even when `lookup_order` would be more appropriate.

The fix: review your system prompts for keyword-sensitive instructions that might override well-written tool descriptions. Use task-oriented language ("retrieve the customer's profile") rather than generic verbs that match tool names ("analyze the customer's situation").

---

## Structured Error Responses

Now let us consider what happens when a tool call fails. Your customer support agent calls `process_refund` and gets back:

```
"Operation failed"
```

What should the agent do? Retry? Tell the customer it cannot help? Escalate to a human? The agent has no idea, because the error message contains no actionable information. It is like a doctor receiving a lab report that just says "abnormal" — without knowing *what* is abnormal, no treatment decision is possible.

### The MCP isError Flag

The Model Context Protocol provides an `isError` flag that explicitly tells the model a tool call failed. When your MCP server returns a response with `isError: true`, the model knows this is not a normal result — it is a failure that needs handling.

But the flag alone is not enough. The model also needs to know *what kind* of failure occurred, because different failures require different responses.

### Four Error Categories

Let us define four categories that cover virtually every tool failure:

**1. Transient errors** — The service is temporarily unavailable. A database timeout, a rate limit, a network blip. These are retryable — the same request will likely succeed if you wait and try again.

**2. Validation errors** — The input was malformed. An invalid email format, a missing required field, an order ID that does not match the expected pattern. These are NOT retryable with the same input — the agent needs to fix the input first.

**3. Business errors** — The request violated a business rule. The refund was denied because the return window expired, or the customer has already received a refund for this order. These are NOT retryable — the policy will not change.

**4. Permission errors** — The agent lacks authorization for this action. A refund above \$500 requires manager approval, or the customer's account is restricted. These need escalation.


![Error response taxonomy — transient, validation, business, and permission errors with recovery actions](figures/figure_2.png)
*Error response taxonomy — transient, validation, business, and permission errors with recovery actions*


### Returning Structured Error Metadata

Here is what a well-structured error response looks like:

```json
{
  "isError": true,
  "content": "Refund denied: return window expired (90 days)",
  "metadata": {
    "errorCategory": "business",
    "isRetryable": false,
    "errorCode": "RETURN_WINDOW_EXPIRED",
    "customerMessage": "Unfortunately, this order is past the 90-day return window and is no longer eligible for a refund."
  }
}
```

Now the agent knows exactly what happened:
- `errorCategory: "business"` — this is a policy issue, not a technical glitch
- `isRetryable: false` — do not attempt the same refund again
- `customerMessage` — a pre-written, customer-friendly explanation to relay

Compare this to `"Operation failed"`. The structured response gives the agent everything it needs to make a good decision.

### Retryable vs Non-Retryable: Why It Matters

This distinction prevents wasted retry attempts. Consider two failures:

**Failure A:** Database timeout (transient). `isRetryable: true`. The agent should wait briefly and retry.

**Failure B:** Refund denied by policy (business). `isRetryable: false`. The agent should NOT retry — the refund will be denied every time. Instead, it should explain the policy to the customer.

Without this distinction, the agent might retry the business failure three times, waste tokens, and still end up telling the customer the same thing. Or worse, it might give up on a transient failure that would have succeeded on the second try.

### Local Error Recovery in Subagents

In a multi-agent system, error handling has another dimension: where does recovery happen?

**Transient errors** should be handled locally within the subagent. If a database call times out, the subagent retries once or twice before propagating the failure. The coordinator should not be bothered with every network hiccup.

**Business and permission errors** should be propagated to the coordinator along with:
- What was attempted
- What failed and why
- Any partial results obtained before the failure

This gives the coordinator the context to make a higher-level decision — perhaps trying a different approach, or escalating to a human.

### Access Failures vs Valid Empty Results

One critical distinction the exam tests: a query that returns zero results is NOT an error. If you search for orders placed by a customer and find none, that is a **valid empty result** — the query succeeded, the answer is simply "no orders found."

An access failure — where the query could not execute because of a permission issue or service outage — IS an error. The agent needs to distinguish between "I checked and there are no orders" (inform the customer) and "I could not check because the system is down" (retry or escalate).

---

## Tool Distribution Across Agents

Let us now zoom out from individual tools to how tools are distributed across a multi-agent system.

Imagine you give a single agent access to 18 tools: customer lookup, order search, refund processing, inventory checking, shipping tracking, email sending, ticket creation, knowledge base search, sentiment analysis, language translation, calendar scheduling, report generation, data export, user authentication, payment processing, discount application, feedback collection, and escalation routing.

The agent will struggle. Not because it cannot understand what each tool does, but because the decision space is too large. With 18 options, every tool call requires evaluating 18 descriptions and picking the best match. The probability of misselection increases with every tool you add.

The empirical guideline: **4–5 tools per agent** gives reliable selection. Beyond that, accuracy degrades.

### Scoped Tool Access

The solution is to scope tools by agent role. In a multi-agent customer support system:

- **Triage agent** (3 tools): `get_customer`, `classify_intent`, `route_to_specialist`
- **Order specialist** (4 tools): `lookup_order`, `track_shipment`, `process_return`, `escalate_to_human`
- **Billing specialist** (4 tools): `get_invoice`, `process_refund`, `apply_credit`, `escalate_to_human`

Each agent has a focused toolset that matches its role. The triage agent never sees `process_refund` — it cannot accidentally call it. The billing specialist never sees `track_shipment` — it will not be distracted by shipping-related options.


![Tool distribution across specialized agents — scoped access prevents misuse](figures/figure_3.png)
*Tool distribution across specialized agents — scoped access prevents misuse*


### Replacing Generic Tools with Constrained Alternatives

Sometimes you cannot reduce the total number of tools, but you can make each one more constrained. Instead of giving a research agent a generic `fetch_url` tool that can hit any URL on the internet, replace it with `load_document` that only accepts URLs matching your document repository pattern.

This is a safety and reliability win: the tool itself enforces boundaries, so the model cannot wander outside its intended scope even if it tries.

### tool_choice Configuration

The Claude API provides three `tool_choice` options that give you fine-grained control over tool selection:

**`tool_choice: "auto"`** (default) — The model decides whether to call a tool, and if so, which one. This is the standard mode for most agent interactions.

**`tool_choice: "any"`** — The model MUST call a tool. It cannot return conversational text without a tool call. Use this when you want to guarantee the agent takes an action rather than chatting.

**`tool_choice: {"type": "tool", "name": "extract_metadata"}`** — The model MUST call the specified tool. Use this for enforcing ordering — for example, forcing the agent to call `extract_metadata` before any enrichment tools.

Let us see how forced selection works in practice. Suppose you have a document processing pipeline where metadata extraction must happen first:

```python
# First turn: force metadata extraction
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_metadata"},
    messages=messages
)

# Process the metadata result...
messages.append({"role": "assistant", "content": response.content})
messages.append({"role": "user", "content": [{"type": "tool_result", ...}]})

# Subsequent turns: let the model choose freely
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=tools,
    tool_choice={"type": "auto"},
    messages=messages
)
```

The first call forces `extract_metadata`. Once that completes, subsequent calls use `"auto"` so the model can select the appropriate enrichment tool based on the metadata it extracted.

---

## MCP Server Configuration

The Model Context Protocol (MCP) is how you connect external tools and data sources to Claude. Let us understand the two configuration levels and when to use each.

### Project-Level vs User-Level Configuration

**Project-level: `.mcp.json`** — Lives in the project root. Committed to version control. Every team member who clones the repository gets the same MCP server configuration. Use this for shared team tooling: your company's Jira server, your GitHub integration, your internal knowledge base.

```json
{
  "mcpServers": {
    "jira": {
      "command": "npx",
      "args": ["@anthropic/jira-mcp-server"],
      "env": {
        "JIRA_URL": "https://yourcompany.atlassian.net",
        "JIRA_TOKEN": "${JIRA_TOKEN}"
      }
    },
    "github": {
      "command": "npx",
      "args": ["@anthropic/github-mcp-server"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

Notice the `${JIRA_TOKEN}` and `${GITHUB_TOKEN}` syntax. MCP supports environment variable expansion — the actual token values come from each developer's local environment, not the committed file. This is how you manage credentials without putting secrets in your repository.

**User-level: `~/.claude.json`** — Lives in the user's home directory. Not committed to version control. Not shared with teammates. Use this for personal or experimental servers: a local database explorer, a personal note-taking integration, a prototype tool you are testing.

```json
{
  "mcpServers": {
    "my-notes": {
      "command": "node",
      "args": ["/Users/me/tools/notes-server/index.js"]
    }
  }
}
```


![MCP server scoping — project-level (shared) vs user-level (personal) configuration](figures/figure_4.png)
*MCP server scoping — project-level (shared) vs user-level (personal) configuration*


### Tool Discovery at Connection Time

When Claude connects to your MCP servers, it discovers all available tools from all configured servers simultaneously. There is no priority ordering — every tool from every server is available in a flat list. This means tool names must be unique across servers, and descriptions must clearly differentiate tools from different servers.

### MCP Resources

MCP is not just about tools — it also supports **resources**. A resource is a read-only data source that the agent can browse without making tool calls. Think of it as a content catalog.

For example, an MCP server for your issue tracker might expose:
- A resource listing all open issues with their summaries
- A resource showing the documentation hierarchy
- A resource describing database schemas

Without resources, the agent would need to make exploratory tool calls to discover what data exists: "List all issues," "Show me the docs structure," "What tables are in the database?" With resources, the agent already knows what is available and can make targeted, efficient tool calls.

### Community vs Custom MCP Servers

A practical decision you will face: should you build a custom MCP server or use an existing community one?

**Use community servers** for standard integrations — Jira, GitHub, Slack, databases. These are well-tested, maintained by the community, and cover common use cases.

**Build custom servers** for team-specific workflows — your proprietary API, your internal deployment pipeline, your custom data format. Custom servers make sense when no community server covers your needs, or when you need tighter integration with internal systems.

### Making MCP Tools Beat Built-in Tools

Here is a subtle issue. Claude has built-in tools like Grep for searching file contents. If you also have an MCP tool that searches a code index (faster, more semantic, better results), the agent might still prefer Grep because it knows Grep well and the MCP tool description is vague.

The fix: write MCP tool descriptions that explicitly explain why the tool is better than the built-in alternative. For example:

```
search_code_index: "Searches the pre-built code index for semantic matches.
Faster than Grep for large codebases (returns in <100ms vs seconds).
Supports natural language queries like 'authentication middleware'.
Use this INSTEAD of Grep when searching for concepts rather than exact strings."
```

Now the model has a clear reason to prefer the MCP tool over the built-in one.

---

## Built-in Tools: Grep, Glob, Read, Write, Edit

Claude Code provides five built-in tools for working with codebases. Each has a specific purpose, and selecting the right one matters for efficiency and reliability.

### Grep — Content Search

Grep searches the **contents** of files. Use it to find:
- All callers of a specific function: `Grep("processPayment")`
- Error messages: `Grep("Connection refused")`
- Import statements: `Grep("from auth import")`

Grep answers the question: "Which files contain this text?"

### Glob — File Path Patterns

Glob searches for files by **name or path pattern**. Use it to find:
- All test files: `Glob("**/*.test.tsx")`
- Configuration files: `Glob("**/config.*")`
- Files in a specific directory: `Glob("src/components/**/*.tsx")`

Glob answers the question: "Which files match this naming pattern?"

### Read and Write — Full File Operations

Read loads the complete contents of a file. Write replaces the complete contents. These are your general-purpose file tools for when you need to see or replace everything.

### Edit — Targeted Modifications

Edit makes surgical changes to a file by finding a unique text anchor and replacing it. This is more efficient than Read + Write when you only need to change a few lines — the model does not need to reproduce the entire file.

But Edit has a constraint: the old text must be **unique** within the file. If the text appears multiple times, Edit fails because it cannot determine which occurrence to change.

**The fallback pattern:** When Edit fails because the anchor text is not unique, switch to Read + Write. Read the full file, find the specific occurrence you want to change (using surrounding context to disambiguate), and Write the complete modified file back.

### Building Codebase Understanding Incrementally

The exam tests an important pattern for how an agent should explore an unfamiliar codebase. The wrong approach is to read every file upfront — this floods the context window with irrelevant code.

The right approach is incremental:

**Step 1: Grep to find entry points.** Search for the function name, error message, or concept you are investigating. This gives you a small set of relevant files.

**Step 2: Read to follow imports and trace flows.** Open the files Grep found. Look at their imports to find related modules. Read those modules to understand the full call chain.

**Step 3: Trace function usage across wrappers.** If a function is re-exported through wrapper modules, first identify all exported names from the original module, then search for each name across the codebase to find all callers.

This pattern — search, read, trace — builds understanding efficiently without overwhelming the context window.

---

## Putting It All Together

Let us revisit our customer support agent, now applying every concept from this article.

**Tool descriptions** are detailed and differentiated — each tool explains its purpose, inputs, outputs, and boundaries with similar tools.

**Error responses** are structured — every failure returns an `errorCategory`, `isRetryable` flag, and a customer-friendly message. The agent retries transient failures, explains business rules to customers, and escalates permission errors to humans.

**Tool distribution** is scoped — the triage agent has 3 tools, the order specialist has 4, and the billing specialist has 4. No agent has more than 5. Cross-specialization misuse is impossible because each agent only sees its own tools.

**MCP configuration** is split — shared tools (customer database, order system) are in `.mcp.json` with environment variable expansion for credentials. Personal debugging tools are in `~/.claude.json`.

**Built-in tools** are used incrementally — Grep finds relevant code, Read traces the full context, Edit makes targeted changes.

### Quick Reference

| Exam Task | Core Concept | Key Anti-Pattern |
|-----------|-------------|-----------------|
| 2.1 | Detailed tool descriptions with boundaries | Minimal descriptions that overlap |
| 2.2 | Structured errors with errorCategory + isRetryable | Generic "Operation failed" messages |
| 2.3 | 4-5 tools per agent, scoped by role | 18 tools on a single agent |
| 2.4 | Project .mcp.json vs user ~/.claude.json | Committing secrets to version control |
| 2.5 | Grep→Read→trace incrementally | Reading all files upfront |

### Key Anti-Patterns to Avoid

1. **Vague tool descriptions** — "Retrieves information" tells the model nothing
2. **Overlapping tool names** — analyze_content vs analyze_document
3. **Uniform error messages** — "Operation failed" with no category or retry guidance
4. **Retrying non-retryable errors** — Business rule denials will fail every time
5. **Too many tools per agent** — Beyond 5, selection degrades
6. **Generic tools where constrained ones would work** — fetch_url vs load_document
7. **Committing secrets in .mcp.json** — Use ${ENV_VAR} expansion
8. **Reading all files upfront** — Use Grep to find entry points first

That's it for Domain 2. These five skills — descriptions, errors, distribution, MCP config, and built-in tools — form the foundation for building reliable tool interfaces in production agent systems.
