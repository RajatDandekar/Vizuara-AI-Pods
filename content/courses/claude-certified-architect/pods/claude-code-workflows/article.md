# Claude Code Configuration & Workflows

*Master the layered configuration hierarchy, custom commands, execution strategies, and CI/CD patterns that turn Claude Code from a generic assistant into a team-aligned development partner*

---

Let us start with a scenario you have probably encountered. You join a new team and open Claude Code for the first time in their monorepo. You type: "Add a new API endpoint for user preferences." Claude generates the code — functional, syntactically correct — but completely wrong for this project. It uses Express instead of Fastify. It puts the route in `src/routes/` instead of `src/api/v2/`. It writes tests with Jest instead of Vitest. It skips the team's mandatory error-handling middleware.

Your teammate sits next to you, types the exact same prompt, and Claude produces code that follows every convention perfectly. Same model, same prompt, wildly different results.

The difference is not magic. It is **configuration**. Your teammate's project has a carefully structured set of CLAUDE.md files, path-specific rules, and custom commands that shape Claude's behavior at every level — from global personal preferences down to file-specific conventions. This is Domain 3 of the Claude Certified Architect exam, and it covers 20% of the questions. Let us build that configuration system from the ground up.

---

## The CLAUDE.md Hierarchy

The foundation of Claude Code configuration is the CLAUDE.md file. Think of it as a system prompt that lives in your filesystem instead of an API call. Claude reads these files automatically when it starts a session, and the instructions inside shape every response.

But here is the key insight: there is not just one CLAUDE.md. There are **three levels**, and they form a hierarchy.

**Level 1: User-level** — Located at `~/.claude/CLAUDE.md`. This file contains your personal preferences: your preferred coding style, your editor shortcuts, your testing habits. It applies to every project you open with Claude Code, regardless of the repository.

**Level 2: Project-level** — Located at `.claude/CLAUDE.md` or `CLAUDE.md` in the repository root. This file contains team conventions: the framework you use, naming patterns, architectural rules. It ships with version control, so every teammate gets the same instructions.

**Level 3: Directory-level** — A `CLAUDE.md` file inside any subdirectory. This applies only when Claude is working on files within that directory. If your backend uses different conventions than your frontend, you put a CLAUDE.md in each directory.


![CLAUDE.md configuration hierarchy showing user, project, and directory levels with precedence flow](figures/figure_1.png)
*CLAUDE.md configuration hierarchy — user, project, and directory levels with @import modular organization*


### Precedence: Most Specific Wins

When multiple levels contain conflicting instructions, the most specific level wins. If your user-level CLAUDE.md says "use tabs for indentation" but the project-level says "use 2-space indentation," the project-level wins because it is more specific to the current context. A directory-level CLAUDE.md overrides both.

Let us see a concrete example. Suppose your user-level file says:

```
Always add JSDoc comments to exported functions.
Prefer functional components in React.
```

Your project-level file says:

```
Use TypeScript strict mode. All files must have explicit return types.
Use Vitest for testing. Never use Jest.
Framework: Next.js 14 with App Router.
```

And inside `src/api/`, a directory-level file says:

```
All API routes must use the withErrorHandler middleware.
Response format: { data: T, error: null } | { data: null, error: string }
```

When Claude edits a file in `src/api/`, it sees all three levels merged together. It will write TypeScript with explicit return types (project), add JSDoc comments (user), and wrap the route in `withErrorHandler` (directory). This is exactly what we want.

### One Critical Rule: User-Level Is Not Shared

The user-level CLAUDE.md at `~/.claude/CLAUDE.md` is **not checked into version control**. It lives on your machine only. This means you should never put team conventions here — they will not travel with the repository. Use the user-level file exclusively for personal preferences that should apply everywhere: your preferred comment style, your debugging habits, how verbose you want Claude's explanations to be.

### @import for Modular Organization

As your project-level CLAUDE.md grows, it becomes unwieldy. A 500-line CLAUDE.md is hard to read, hard to maintain, and hard to reason about.

The solution is `@import`. You can split your instructions into separate files and import them:

```markdown
# Project Instructions

@import ./claude/rules/testing.md
@import ./claude/rules/api-conventions.md
@import ./claude/rules/database.md
@import ./claude/rules/frontend.md
```

Each imported file contains a focused set of rules for one domain. When you update your testing conventions, you edit `testing.md` — you do not scroll through a monolithic file looking for the right section.

This also makes code review easier. When someone changes `api-conventions.md`, reviewers know exactly what changed and why. A diff to a 500-line CLAUDE.md is much harder to parse.

### Verifying What Claude Sees

How do you confirm that Claude is reading the right configuration? Use the `/memory` command. This shows you exactly which CLAUDE.md files Claude has loaded and what instructions it is following. If your directory-level rules are not appearing, `/memory` will tell you — perhaps the file is in the wrong directory, or the filename has a typo.

---

## Path-Specific Rules with .claude/rules/

Directory-level CLAUDE.md files work well when your conventions follow the directory structure. But what about conventions that cut across directories?

Consider test files. In many projects, test files are co-located with source files — `Button.tsx` and `Button.test.tsx` live in the same directory. You cannot put a CLAUDE.md in every directory that contains test files. And you cannot put test-specific rules in the project-level CLAUDE.md without polluting the instructions for non-test files.

This is where `.claude/rules/` comes in. Files in this directory use YAML frontmatter with a `paths` field to specify **glob patterns**. The rules inside only activate when Claude is editing files that match those patterns.

Here is an example. Create a file at `.claude/rules/testing.md`:

```markdown
---
paths:
  - "**/*.test.tsx"
  - "**/*.test.ts"
  - "**/*.spec.ts"
---

## Testing Conventions

- Use `describe` blocks to group related tests
- Use `it` for individual test cases (not `test`)
- Mock external services with `vi.mock()`, never real network calls
- Each test file must have at least one snapshot test for UI components
- Use `screen.getByRole` over `getByTestId` for accessibility
```

Now, when Claude edits `Button.test.tsx` — regardless of which directory it is in — these rules activate. When Claude edits `Button.tsx`, they do not. This is exactly what we want.

### Glob Patterns by File Type

The glob syntax supports rich matching patterns:

- `**/*.test.tsx` — all test files, any directory
- `terraform/**/*` — everything under the terraform directory
- `src/api/**/*.ts` — TypeScript files in the API directory tree
- `**/*.{css,scss}` — all CSS and SCSS files
- `migrations/*.sql` — SQL migration files in the migrations directory

Let us look at another practical example. Suppose you have Terraform infrastructure code scattered across multiple directories:

```markdown
---
paths:
  - "terraform/**/*"
  - "infra/**/*.tf"
---

## Terraform Conventions

- Always use `terraform fmt` style formatting
- Tag all resources with `environment`, `team`, and `managed-by` tags
- Use `count` for simple conditionals, `for_each` for collections
- Never hardcode AMI IDs — use data sources
- State files must use S3 backend with DynamoDB locking
```

### Path-Specific Rules vs Directory-Level CLAUDE.md

When should you use each approach? The decision is straightforward:

**Use directory-level CLAUDE.md** when the rules apply to an entire subtree of your repository and the conventions align with directory boundaries. Example: your `frontend/` directory uses React and your `backend/` directory uses Express — put a CLAUDE.md in each.

**Use path-specific rules** when the conventions are based on file type or naming patterns that span multiple directories. Example: test files, migration files, configuration files, Terraform files — these appear throughout the codebase.

In practice, most teams use both. Directory-level CLAUDE.md for broad architectural boundaries, path-specific rules for cross-cutting concerns like testing, linting, and documentation.

---

## Custom Slash Commands and Skills

CLAUDE.md files tell Claude *how* to behave. Custom commands and skills tell Claude *what to do* — they are reusable action templates that you trigger on demand.

### Slash Commands

Slash commands are Markdown files that become invocable actions. There are two places to put them:

**Project-scoped:** `.claude/commands/` — These ship with the repository. Every teammate gets them. Use for team-standard workflows like "generate a migration," "create a component," or "run the full test suite with coverage."

**Personal:** `~/.claude/commands/` — These live on your machine only. Use for personal workflows like "format this code the way I like it" or "explain this function in simple terms."

Let us create a project command. Place this in `.claude/commands/new-api-endpoint.md`:

```markdown
Create a new API endpoint with the following specifications:

1. Create the route handler in `src/api/v2/$ARGUMENTS.ts`
2. Add input validation using Zod schemas
3. Wrap the handler in `withErrorHandler` middleware
4. Create a test file at `src/api/v2/$ARGUMENTS.test.ts`
5. Add the route to the OpenAPI spec in `docs/openapi.yaml`
6. Register the route in `src/api/v2/index.ts`

Follow all conventions in CLAUDE.md and the api-conventions rules.
```

Now any teammate can type `/new-api-endpoint users/preferences` and Claude will create a fully configured endpoint with tests, validation, OpenAPI documentation, and proper middleware — all following the team's conventions.

The `$ARGUMENTS` placeholder captures whatever the user types after the command name. This makes commands flexible without requiring complex parameterization.

### Skills: Commands with Superpowers

Skills are like commands, but with additional capabilities defined through a `SKILL.md` frontmatter. They live in `.claude/skills/` (project-scoped) or `~/.claude/skills/` (personal).

What makes skills different from commands? Three things:

**1. context:fork** — This is the most important property. When a skill runs with `context:fork`, it executes in an **isolated sub-agent context**. The sub-agent gets a fresh context window, runs the skill, and returns just the result. This prevents a verbose skill from filling up your main conversation's context window.

Think of it like calling a function vs inlining code. A command inlines its work into your current conversation. A skill with `context:fork` calls a separate function and returns the result.

**2. allowed-tools** — You can restrict which tools the skill can use. A code-generation skill might only need file read/write access. A deployment skill might need shell access but not file editing. This follows the principle of least privilege.

**3. argument-hint** — A description of what arguments the skill expects, shown to the user when they invoke it. This makes skills self-documenting.


![Comparison of slash commands vs skills — showing scope, context, and capabilities](figures/figure_4.png)
*Slash commands vs skills — when to use each and how they differ in scope and capabilities*


Here is a complete SKILL.md example for a database migration skill:

```markdown
---
context: fork
allowed-tools:
  - read
  - write
  - bash
argument-hint: "migration name (e.g., add-user-preferences-table)"
---

# Generate Database Migration

Create a new database migration with the given name:

1. Create migration file at `migrations/{timestamp}_{$ARGUMENTS}.sql`
2. Include UP and DOWN sections
3. Follow the column naming conventions in .claude/rules/database.md
4. Add a corresponding test in `migrations/tests/`
5. Update the migration index
```

When a developer types `/generate-migration add-user-preferences-table`, Claude spawns a sub-agent (because of `context:fork`) that creates the migration, writes the test, and returns. The main conversation stays clean.

### When to Use Skills vs CLAUDE.md

This is a common exam question. The rule is simple:

- **CLAUDE.md** is for passive rules — conventions that should always be followed. "Use 2-space indentation." "Always validate inputs." These are background instructions.
- **Skills** are for active workflows — specific multi-step procedures triggered on demand. "Generate a migration." "Deploy to staging." "Run the full review checklist." These are foreground actions.

If you find yourself writing a CLAUDE.md instruction that starts with "When the developer asks you to..." — that is a skill, not a rule. Extract it into `.claude/skills/`.

---

## Plan Mode vs Direct Execution

Claude Code operates in two modes: **plan mode** and **direct execution**. Choosing the right mode for the right task is a core exam skill.

### When to Use Plan Mode

Plan mode is for tasks where you need to think before you act. The model explores the codebase, reasons about the approach, and presents a plan for approval before making any changes.

Use plan mode when:

1. **The change spans many files** — If you are modifying 10, 20, or 45+ files (like a library migration), plan mode prevents you from getting halfway through and realizing your approach is wrong.

2. **Multiple valid approaches exist** — Should you refactor the authentication system to use JWTs or session tokens? Plan mode lets the model analyze the tradeoffs before committing.

3. **Architectural decisions are involved** — Restructuring a module, changing a data model, introducing a new pattern. These decisions cascade through the codebase.

4. **The scope is unclear** — "Improve the performance of the dashboard." Plan mode lets the model investigate first: profile the code, identify bottlenecks, and propose targeted fixes.


![Decision tree for choosing between plan mode and direct execution based on task characteristics](figures/figure_2.png)
*Plan mode decision tree — when to plan first vs execute directly*


### When to Use Direct Execution

Direct execution is for tasks where the scope is clear and the approach is obvious:

- Fix a specific bug with a known root cause
- Add a single function to an existing module
- Update a configuration value
- Rename a variable across the codebase
- Write a test for an existing function

The key question is: **Do I already know what files need to change and how?** If yes, skip planning and execute directly.

### The Explore Subagent

Here is a pattern that combines the best of both modes. Sometimes you need to investigate the codebase before you even know whether to use plan mode or direct execution.

The **Explore subagent** isolates verbose discovery. It reads files, searches for patterns, and maps the codebase — all in a separate context. This prevents your main conversation from being filled with grep results and file listings.

The workflow looks like this:

1. **Explore** — Use the Explore subagent to investigate. "How does the authentication system work? What files are involved?"
2. **Decide** — Based on the findings, choose plan mode (complex) or direct execution (simple).
3. **Execute** — Either present a plan for approval or make the changes directly.

This three-step pattern — explore, decide, execute — appears frequently on the exam.

---

## Iterative Refinement Patterns

Configuration tells Claude *what* to do. But getting consistently high-quality output requires the right interaction patterns. Domain 3 tests five specific refinement techniques.

### Pattern 1: Concrete Input/Output Examples

When you need Claude to perform a consistent transformation — reformatting code, standardizing API responses, converting between formats — provide 2-3 concrete examples of input and expected output.

```
Transform these function signatures from callback style to async/await:

Example 1:
Input:  function getUser(id, callback) { ... }
Output: async function getUser(id): Promise<User> { ... }

Example 2:
Input:  function saveOrder(order, onSuccess, onError) { ... }
Output: async function saveOrder(order): Promise<Order> { ... }

Now transform all functions in src/legacy/api.ts.
```

The examples anchor Claude's behavior far more precisely than a verbal description. "Convert callbacks to async/await" is ambiguous. Two concrete examples eliminate the ambiguity entirely.

### Pattern 2: Test-Driven Iteration

Write the tests first, then let Claude iterate on the implementation until the tests pass.

```
Here are the tests for the new caching layer (see tests/cache.test.ts).
Implement src/lib/cache.ts so that all 12 tests pass.
```

This inverts the typical workflow. Instead of writing code and then checking if it works, you define the contract first and let Claude iterate toward it. Each test failure gives Claude specific, actionable feedback about what is wrong.

This pattern is especially powerful for refactoring. You write tests that capture the existing behavior, then ask Claude to refactor the implementation while keeping all tests green.

### Pattern 3: The Interview Pattern

For unfamiliar domains or ambiguous requirements, use the interview pattern: instruct Claude to **ask questions before implementing**.

```
I need to add rate limiting to our API. Before implementing anything,
ask me questions to understand our requirements. What do you need to know?
```

Claude will ask about: which endpoints need rate limiting, what the limits should be, whether limits are per-user or per-IP, what should happen when limits are exceeded, whether you need distributed rate limiting across multiple servers.

This prevents Claude from making assumptions. In an unfamiliar domain, assumptions lead to rework. Questions lead to correct implementations.

### Pattern 4: Batching Interacting Issues

When you have multiple related changes that interact — they affect the same files, the same data structures, the same interfaces — batch them into a single prompt. This lets Claude reason about the interactions holistically.

```
Make these three changes together (they all affect the User model):
1. Add an `emailVerified` boolean field
2. Update the registration flow to send a verification email
3. Add a middleware that blocks unverified users from protected routes
```

If you submitted these as three separate prompts, Claude might implement change #1 without considering the middleware in change #3, leading to an inconsistent state.

Conversely, independent changes that do not interact should be submitted **sequentially**. A CSS bugfix and an API endpoint addition have no interaction — submit them separately so each gets Claude's full attention.

---

## CI/CD Integration

The final skill in Domain 3 is running Claude Code in automated pipelines. This is where configuration meets automation.

### The -p Flag: Non-Interactive Mode

In a CI/CD pipeline, there is no human to approve tool calls or answer questions. The `-p` flag (or `--print`) runs Claude Code in **non-interactive mode**: it reads the prompt from stdin or the command line, executes, and prints the result to stdout.

```bash
claude -p "Review the changes in this PR for security vulnerabilities" \
  --output-format json
```

This is the foundation of every CI integration. Without `-p`, Claude Code would wait for interactive input and your pipeline would hang.

### Structured Output with --output-format and --json-schema

For CI/CD, you almost always want structured output that downstream tools can parse. The `--output-format json` flag tells Claude to return JSON instead of freeform text.

But raw JSON is not enough — you need to control the shape. The `--json-schema` flag lets you provide a JSON Schema that Claude's output must conform to:

```bash
claude -p "Review this PR for issues" \
  --output-format json \
  --json-schema '{
    "type": "object",
    "properties": {
      "issues": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "file": { "type": "string" },
            "line": { "type": "integer" },
            "severity": { "type": "string", "enum": ["critical", "warning", "info"] },
            "description": { "type": "string" },
            "suggestion": { "type": "string" }
          },
          "required": ["file", "line", "severity", "description"]
        }
      },
      "summary": { "type": "string" },
      "approve": { "type": "boolean" }
    },
    "required": ["issues", "summary", "approve"]
  }'
```

Now your CI pipeline can parse the output, post individual comments on the PR at the exact file and line, and auto-approve or block the merge based on the `approve` field.


![CI/CD pipeline flow showing Claude Code with -p flag, structured output, and PR integration](figures/figure_3.png)
*CI/CD pipeline — Claude Code reviewing PRs with -p flag, --output-format json, and structured schema output*


### CLAUDE.md for CI Context

Your CI pipeline should include the project's CLAUDE.md in Claude's context. This ensures that automated reviews follow the same conventions as interactive development. Without it, Claude's CI reviews might flag code that is perfectly acceptable under your team's conventions.

The simplest approach: run Claude Code from the project root, where it will automatically detect and load the CLAUDE.md files.

### Session Context Isolation

Here is a subtle but important point the exam tests: **a model reviewing its own output in the same session is less effective than a fresh session**.

If Claude generates code in one step and reviews it in the next step within the same conversation, it has a bias toward approving its own work. The code "looks right" because the model just wrote it.

For CI/CD code review, always use a **separate session** — a fresh Claude Code invocation with `-p` that sees only the diff. This gives you an unbiased review because the reviewer has no memory of writing the code.

### Avoiding Duplicate Findings

When running iterative reviews (e.g., reviewing after each push to a PR), include prior review findings in the prompt:

```bash
claude -p "Review this PR. Previous review findings that have been addressed:
- Line 42: missing null check (FIXED)
- Line 78: SQL injection risk (FIXED)
Focus on NEW issues only." \
  --output-format json
```

Without this context, Claude will re-flag issues that have already been fixed, creating noise for the developer.

### Documenting Testing Standards in CLAUDE.md

For CI/CD to work smoothly, your CLAUDE.md should document:

1. **How to run tests** — `npm test`, `pytest`, or whatever your command is
2. **What test coverage is expected** — Minimum percentage, which directories must be covered
3. **What testing frameworks to use** — Vitest, Jest, pytest, etc.
4. **Test file naming conventions** — `*.test.ts`, `*.spec.ts`, `*_test.py`
5. **Where to find existing test examples** — So Claude can follow patterns

This information shapes both interactive development and automated CI reviews.

---

## Bringing It All Together

Let us trace through a complete workflow to see how all these pieces connect.

A developer joins the team and clones the repository. Claude Code loads three levels of CLAUDE.md: their personal preferences (user-level), the team conventions (project-level), and the API-specific rules (directory-level). Path-specific rules activate whenever they touch test files, migration files, or Terraform configs.

They type `/new-api-endpoint users/preferences` — a project-scoped slash command. Claude creates the endpoint following every convention.

For a larger task — migrating the authentication library — they use plan mode. Claude explores the codebase with the Explore subagent, proposes a migration plan, and executes after approval.

They write tests first for the new preference validation logic, then ask Claude to implement until all tests pass. For an unfamiliar domain (rate limiting), they use the interview pattern to clarify requirements before any code is written.

When they push a PR, the CI pipeline runs Claude Code with `-p` and `--output-format json`. The structured output posts review comments directly on the PR. A fresh session ensures unbiased review. Prior findings are included to avoid duplicates.

That is the complete Domain 3 story: configuration hierarchy at the bottom, execution strategies in the middle, and CI/CD automation at the top. Each layer builds on the previous one, and together they transform Claude Code from a generic assistant into a team-aligned development partner.

---

*This article covers Domain 3 of the Claude Certified Architect — Foundations exam (20% of questions). For hands-on practice, work through the four accompanying notebooks: CLAUDE.md Hierarchy & Rules, Custom Commands & Skills, Plan Mode & Iterative Refinement, and CI/CD Integration.*
