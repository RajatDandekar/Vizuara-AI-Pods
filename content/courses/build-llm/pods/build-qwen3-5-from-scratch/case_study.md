# Building a Local AI Coding Assistant for a Privacy-Conscious Healthcare Startup

## MedCode Health -- HIPAA-Compliant On-Device Code Intelligence

---

## Section 1: Industry Context and Business Problem

### The Healthcare Software Compliance Crisis

Healthcare software companies operate under some of the strictest data regulations in the world. In the United States, the Health Insurance Portability and Accountability Act (HIPAA) mandates that any system handling Protected Health Information (PHI) must ensure that patient data never leaves controlled environments without explicit authorization. Violations carry penalties of up to \$1.5 million per incident category per year.

**MedCode Health** is a 45-person healthtech startup based in Austin, Texas, building an Electronic Health Record (EHR) system for mid-size clinics (50-200 providers). Their platform processes 2.8 million patient records, including clinical notes, lab results, prescriptions, and diagnostic codes. All data is stored on-premises at each clinic site — a non-negotiable requirement from their healthcare clients.

### The Problem

MedCode's engineering team of 12 developers spends approximately 35% of their time on repetitive coding tasks: writing database queries for patient record retrieval, implementing FHIR (Fast Healthcare Interoperability Resources) API endpoints, writing data validation logic for clinical forms, and debugging HL7 message parsers. The VP of Engineering estimates this represents \$840,000 per year in engineering salary spent on tasks that could be accelerated with AI-powered code assistance.

The obvious solution — using cloud-based AI coding assistants like GitHub Copilot or ChatGPT — is **not an option**. MedCode's development environment has access to their production database schemas, sample patient data in test fixtures, HIPAA-regulated API keys, and proprietary clinical algorithms. Sending any of this context to a cloud API would constitute a HIPAA violation, regardless of the provider's data processing agreements.

| Challenge | Impact | Constraint |
|---|---|---|
| **Repetitive coding tasks** | 35% of engineering time wasted | Cannot use cloud AI (HIPAA) |
| **FHIR API boilerplate** | 8 hours per new endpoint | Code references PHI schemas |
| **Database query writing** | 4 hours per complex query | Queries reference patient tables |
| **Code review bottleneck** | 2-day average review turnaround | Reviewers need domain context |
| **Onboarding new developers** | 6 weeks to productivity | Proprietary codebase patterns |

The CTO has approved a budget of \$15,000 for hardware and \$0 for recurring API costs to deploy a **fully local AI coding assistant** that runs entirely on MedCode's on-premises infrastructure. No patient data, no code, and no queries may leave the local network.

### Success Criteria

| Metric | Current (No AI) | Target (Local AI) |
|---|---|---|
| Time per FHIR endpoint | 8 hours | 3 hours |
| Time per complex SQL query | 4 hours | 1.5 hours |
| Code review turnaround | 2 days | 0.5 days |
| New developer onboarding | 6 weeks | 3 weeks |
| Monthly API costs | \$0 | \$0 (must remain zero) |
| First-token latency | N/A | < 100ms |
| Data leaving premises | N/A | Exactly 0 bytes |

---

## Section 2: Technical Problem Formulation

### 2.1 Problem Statement

Design and deploy a local code completion and chat assistant that:

1. Runs entirely on a single on-premises GPU server (NVIDIA RTX 4090, 24 GB VRAM)
2. Provides code completion with < 100ms first-token latency
3. Handles context windows of at least 8,192 tokens (to include file context + conversation)
4. Understands healthcare coding patterns (FHIR, HL7, ICD-10, clinical data models)
5. Supports both inline completion and interactive chat modes
6. Costs \$0 in recurring fees after initial hardware investment

### 2.2 Why Qwen3.5 Is the Right Model

The model selection criteria are unusually constrained: the model must be (a) open-weight (no API dependency), (b) small enough to run on a single RTX 4090 in 4-bit quantization, (c) strong at code generation, and (d) capable of tool use for code search and documentation retrieval.

After evaluating Llama 4-8B, Gemma 3-9B, DeepSeek-Coder-7B, and Qwen3.5-9B, MedCode selected **Qwen3.5-9B** for three architectural reasons:

**1. Hybrid attention for long-context code.** The 3:1 pattern of Gated DeltaNet and Gated Attention layers gives Qwen3.5-9B a native context length of 262,144 tokens — far more than needed — while maintaining fast inference. The Gated DeltaNet layers (75% of the model) process tokens in $O(1)$ time, meaning the model does not slow down as the code context grows. For a developer pasting a 2,000-line file into the chat, this is critical.

**2. Superior coding benchmarks.** On HumanEval+, Qwen3.5-9B scores 82.3, compared to Llama 4-8B (74.1) and Gemma 3-9B (76.8). On MBPP+, it scores 78.9. Most importantly, on BFCL-V4 (the tool-use benchmark that tests function calling, which MedCode needs for code search), Qwen3.5 models dominate the leaderboard.

**3. Efficient quantization.** The 9B dense model quantizes to Q4_K_M (4-bit) at approximately 5.2 GB, leaving 18.8 GB of VRAM on the RTX 4090 for KV cache and activations. The hybrid attention architecture reduces KV cache requirements by 75% (only the Gated Attention layers need traditional KV cache), enabling longer effective context than a standard transformer of the same size.

| Model | HumanEval+ | MBPP+ | VRAM (Q4) | Native Context | KV Cache Reduction |
|---|---|---|---|---|---|
| Llama 4-8B | 74.1 | 71.2 | 4.8 GB | 128K | None |
| Gemma 3-9B | 76.8 | 73.5 | 5.1 GB | 128K | None |
| DeepSeek-Coder-7B | 79.2 | 75.8 | 4.2 GB | 32K | None |
| **Qwen3.5-9B** | **82.3** | **78.9** | **5.2 GB** | **262K** | **75%** |

### 2.3 System Architecture

The deployment architecture consists of four components:

**1. Inference Server (llama.cpp).** The Qwen3.5-9B model is served via llama.cpp compiled with CUDA support. This provides an OpenAI-compatible API at `http://localhost:8080/v1/` that any IDE plugin can connect to.

Configuration:
- Model: `Qwen3.5-9B-Q4_K_M.gguf` (5.2 GB)
- Context length: 16,384 tokens
- Batch size: 512
- GPU layers: all (fully offloaded to RTX 4090)
- Threads: 8 (for prompt processing on CPU)

**2. Code Search Tool (local).** A lightweight semantic search index over MedCode's codebase, built using a local embedding model (Qwen3.5-0.8B). When a developer asks about a function, the system retrieves relevant code files and injects them into the context.

**3. IDE Integration.** A VS Code extension that connects to the local llama.cpp server, providing:
- Inline code completion (Fill-in-the-Middle mode)
- Chat panel for interactive coding questions
- Automatic context injection (current file + imports + relevant files from code search)

**4. Prompt Engineering Layer.** A middleware service that constructs prompts with:
- System prompt describing MedCode's coding conventions (FHIR resource patterns, database schema naming, error handling standards)
- Retrieved code context from the search tool
- The developer's query or current code context

### 2.4 Key Technical Decisions

**Why llama.cpp over vLLM?** For a single-GPU deployment serving 12 developers (not hundreds of concurrent users), llama.cpp provides lower latency and simpler deployment. vLLM excels at high-throughput multi-user scenarios; llama.cpp excels at low-latency single-user inference with GGUF quantization.

**Why Q4_K_M quantization?** This format uses 4-bit quantization with k-means clustering for the most important layers, achieving near-FP16 quality. On the Qwen3.5-9B model specifically, Q4_K_M degrades perplexity by only 0.3 points compared to FP16, while reducing model size from 18 GB to 5.2 GB. The RTX 4090's 24 GB VRAM comfortably holds the model plus a 16K-token KV cache.

**Why 16,384 context instead of the full 262K?** While Qwen3.5-9B supports 262K tokens natively, the KV cache memory for the Gated Attention layers (25% of the model) still grows linearly with context length. At 16K tokens, the KV cache uses approximately 2 GB, leaving ample headroom. For coding tasks, 16K tokens (roughly 12,000 lines of code) is more than sufficient.

---

## Section 3: Implementation Details

### 3.1 Hardware Setup

| Component | Specification | Cost |
|---|---|---|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) | \$1,599 |
| CPU | AMD Ryzen 9 7950X (16 cores) | \$549 |
| RAM | 64 GB DDR5-5600 | \$180 |
| Storage | 2 TB NVMe SSD | \$150 |
| **Total** | | **\$2,478** |

This is a one-time capital expenditure. Monthly operating cost (electricity): approximately \$35 at 350W average draw, 24/7 operation, at \$0.12/kWh.

### 3.2 Model Deployment

```bash
# Download the quantized model
wget https://huggingface.co/Qwen/Qwen3.5-9B-GGUF/resolve/main/qwen3.5-9b-q4_k_m.gguf

# Start the inference server
./llama-server \
  --model qwen3.5-9b-q4_k_m.gguf \
  --ctx-size 16384 \
  --n-gpu-layers 999 \
  --host 0.0.0.0 \
  --port 8080 \
  --batch-size 512 \
  --threads 8 \
  --flash-attn
```

### 3.3 Performance Tuning

The hybrid attention architecture provides a unique optimization opportunity. During inference:

- **Gated DeltaNet layers (45 of 60):** These maintain a fixed-size state matrix per head. No KV cache needed. Memory is constant regardless of sequence length.
- **Gated Attention layers (15 of 60):** These require a traditional KV cache that grows with sequence length.

This means the effective KV cache memory is 75% smaller than a standard transformer of equivalent depth:

$$
\text{KV cache} = \frac{15}{60} \times 2 \times L \times n_{\text{heads}} \times d_{\text{head}} \times \text{sizeof(float16)}
$$

For $L = 16{,}384$, $n_{\text{heads}} = 32$, $d_{\text{head}} = 128$:

$$
\text{KV cache} = 0.25 \times 2 \times 16{,}384 \times 32 \times 128 \times 2 \text{ bytes} = 0.5 \text{ GB}
$$

Compare this to a standard transformer: $2.0$ GB. The 75% savings leaves more VRAM for larger batch sizes or longer contexts when needed.

### 3.4 Thinking Mode for Complex Tasks

Qwen3.5's dual thinking/non-thinking mode is particularly useful for the coding assistant:

- **Non-thinking mode** (fast): Used for inline code completion. The model generates the next few tokens directly, with first-token latency under 50ms.
- **Thinking mode** (deep): Used for complex queries like "Explain why this FHIR endpoint is returning a 422 error" or "Design a database migration for adding prescription history." The model generates internal chain-of-thought reasoning before producing the answer, typically taking 2-5 seconds but producing significantly better results.

The IDE integration automatically selects the mode: inline completions use non-thinking mode, chat queries use thinking mode with a 2,048-token thinking budget.

---

## Section 4: Results and Business Impact

### 4.1 Performance Benchmarks

After one month of deployment, MedCode measured the following results:

| Metric | Before (No AI) | After (Qwen3.5-9B Local) | Improvement |
|---|---|---|---|
| Time per FHIR endpoint | 8 hours | 2.5 hours | **69% faster** |
| Time per complex SQL query | 4 hours | 1.2 hours | **70% faster** |
| Code review turnaround | 2 days | 8 hours | **75% faster** |
| New developer onboarding | 6 weeks | 2.5 weeks | **58% faster** |
| Monthly API costs | \$0 | \$0 | No change |
| Data leaving premises | 0 bytes | 0 bytes | **Full compliance** |

### 4.2 Inference Performance

| Metric | Value |
|---|---|
| First-token latency (non-thinking) | 38ms |
| First-token latency (thinking mode) | 42ms |
| Tokens per second (generation) | 45 tok/s |
| Context processing speed | 2,800 tok/s |
| Max concurrent users | 3 (with <100ms latency) |
| Uptime (30-day period) | 99.97% |

### 4.3 Financial Impact

**Productivity gains:**
- Engineering time saved: 35% → 12% on repetitive tasks = 23% recovery
- 12 developers × \$70,000 avg salary × 23% = **\$193,200/year saved**
- Code review cycle improvement saves approximately **\$48,000/year** in reduced context-switching costs

**Total annual savings: \$241,200**

**Total cost:**
- Hardware (one-time): \$2,478
- Electricity (annual): \$420
- Setup and configuration: ~20 hours of DevOps time ≈ \$3,000

**ROI: 97:1 in the first year.** The hardware investment pays for itself in less than 4 days of operation.

### 4.4 Qualitative Feedback

After the first month, MedCode conducted a developer satisfaction survey:

- **92% of developers** reported using the assistant daily
- **88%** said it "significantly" or "very significantly" improved their productivity
- **100%** reported zero concerns about data privacy (the #1 reason they could not use cloud alternatives)
- Most popular use cases: (1) FHIR endpoint boilerplate generation, (2) SQL query writing with schema context, (3) explaining unfamiliar parts of the codebase to new hires

The most impactful feedback came from a junior developer: "I used to spend 2 hours reading through our codebase to understand how we handle medication interactions. Now I just ask the assistant, and it pulls up the relevant code and explains it. It's like having a senior developer available 24/7."

### 4.5 Comparison: What if MedCode Used a Cloud API?

For context, let us estimate what the cost would have been if MedCode could have used a cloud-based API (ignoring HIPAA constraints):

| Service | Cost per 1M tokens | Estimated monthly usage | Monthly cost |
|---|---|---|---|
| GPT-4o | \$2.50 input / \$10 output | ~80M tokens | \$520 |
| Claude Sonnet | \$3.00 input / \$15 output | ~80M tokens | \$720 |
| Qwen3.5 (via Alibaba Cloud) | \$0.50 input / \$2 output | ~80M tokens | \$120 |

Even the cheapest cloud option would cost \$1,440/year — more than half the cost of the hardware that will last 3-5 years. And none of these options would be HIPAA-compliant without expensive BAA agreements and security audits.

---

### Lessons Learned and Production Considerations

### What Worked Well

1. **Qwen3.5's hybrid attention was the right architectural choice.** The 75% KV cache reduction meant the difference between fitting 8K and 16K context on the RTX 4090. For coding tasks where developers paste large files, this extra context is critical.

2. **llama.cpp's simplicity was an advantage.** A single binary, a single model file, and a single startup command. No Python environment, no dependency conflicts, no container orchestration. The server has been running for 30 days with 99.97% uptime.

3. **Thinking mode differentiation was valuable.** Using non-thinking mode for completions (fast) and thinking mode for chat (thorough) gave the best user experience. Developers did not want to wait 3 seconds for every code completion, but appreciated the deeper reasoning for complex questions.

4. **The code search tool was essential.** Without it, the model could only work with the current file. With semantic search over the full codebase, the model could answer questions about distant files and cross-reference patterns across the project.

### Challenges and Gotchas

1. **Quantization can degrade code quality on edge cases.** Q4_K_M quantization occasionally produces slightly garbled output for very long code completions (>500 tokens). MedCode addressed this by limiting completion length to 256 tokens and using Q5_K_M for the chat mode model (which needed higher quality for longer outputs). Running two model instances (Q4 for completions, Q5 for chat) fit within the 24 GB budget.

2. **Context window management is non-trivial.** With 16K tokens of context, you must carefully decide what to include. MedCode's solution: current file (up to 4K tokens) + retrieved context (up to 4K tokens) + conversation history (up to 4K tokens) + system prompt (1K tokens) + generation budget (3K tokens). This required building a token-counting middleware to prevent context overflow.

3. **The model does not know about MedCode-specific libraries.** Qwen3.5-9B was trained on public code, not MedCode's proprietary libraries. The code search tool partially addresses this by injecting relevant code into the context, but the model sometimes suggests patterns from popular open-source healthcare libraries instead of MedCode's internal ones. Fine-tuning on MedCode's codebase (using QLoRA) is planned for Q2 2026.

4. **Multi-user contention.** When 3+ developers query the model simultaneously, latency increases to 200-300ms. MedCode solved this by implementing a request queue with priority levels (completions get highest priority due to latency sensitivity).

### Deployment Recommendations for Similar Organizations

1. **Start with Qwen3.5-9B for single-GPU deployments.** It is the sweet spot of quality vs hardware requirements. If you have 2+ GPUs, consider Qwen3.5-35B-A3B (MoE) — the 3B active parameters run as fast as a 3B dense model but with 35B quality.

2. **Use llama.cpp for < 50 users, vLLM for > 50 users.** llama.cpp excels at low-latency single-user scenarios. vLLM's PagedAttention and continuous batching are essential for high-throughput multi-user deployments.

3. **Invest in a code search tool.** The model alone is not enough — it needs access to your codebase to be useful. A local embedding model (Qwen3.5-0.8B is excellent at ~1.6 GB) plus a vector database (ChromaDB or Qdrant, both run locally) transforms the assistant from "generic code helper" to "teammate who knows our codebase."

4. **Budget for fine-tuning.** After 3 months of usage, you will have enough interaction data to fine-tune the model on your specific coding patterns. QLoRA fine-tuning of Qwen3.5-9B takes approximately 4 hours on a single RTX 4090 and can improve domain-specific code quality by 15-25%.

---

## Section 5: Companion Notebook Overview

The companion Colab notebook implements a simplified version of MedCode's deployment:

1. **Download and serve Qwen3.5-0.8B** locally (smaller model for Colab's T4 GPU)
2. **Build a semantic code search tool** using embeddings
3. **Implement the context management pipeline** (file context + search results + system prompt)
4. **Implement thinking/non-thinking mode switching**
5. **Benchmark inference performance** (latency, throughput, memory)
6. **Compare quantization levels** (FP16, Q8, Q4) on code generation quality

Students will walk away with a working local AI coding assistant running entirely in their Colab notebook — the same architecture MedCode uses in production, just on a smaller scale.
