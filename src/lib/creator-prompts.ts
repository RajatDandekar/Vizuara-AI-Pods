import { readFile, readdir } from 'fs/promises';
import path from 'path';

const STYLE_PROFILE_PATH = path.join(process.cwd(), 'style-profile.md');
const REFERENCE_DIR = path.join(process.cwd(), 'reference-articles');

async function loadStyleProfile(): Promise<string> {
  try {
    return await readFile(STYLE_PROFILE_PATH, 'utf-8');
  } catch {
    return '';
  }
}

async function loadReferenceArticles(): Promise<string> {
  try {
    const files = await readdir(REFERENCE_DIR);
    const mdFiles = files.filter((f) => f.endsWith('.md')).sort().slice(0, 3);
    const articles: string[] = [];

    for (const file of mdFiles) {
      try {
        const content = await readFile(path.join(REFERENCE_DIR, file), 'utf-8');
        // Include substantial excerpts for proper style matching
        articles.push(
          `--- Reference Article (${file}) ---\n${content.slice(0, 4000)}\n---`
        );
      } catch {
        // Skip missing files
      }
    }

    return articles.join('\n\n');
  } catch {
    return '';
  }
}

export async function getArticleSystemPrompt(): Promise<string> {
  const styleProfile = await loadStyleProfile();
  const referenceArticles = await loadReferenceArticles();

  return `You are an expert AI/ML educator writing for the Vizuara publication. You must internalize the writing style completely before producing any content.

## Writing Style Profile (READ EVERY WORD)
${styleProfile}

## Reference Articles (match this voice EXACTLY)
${referenceArticles}

## Critical Rules

### Voice & Structure
1. Follow the "Guided Discovery" pedagogy: Hook → Intuition → Formalize → Validate → Complicate → Resolve → Implement
2. Open with "Let us start with..." + concrete, relatable example
3. Write in conversational professor tone — use "we" and "Let us..."
4. Each article must be fully self-contained — explain all prerequisites inline
5. Use the signature phrase "This is exactly what we want." after validating results
6. Use transitions like "This brings us to...", "Now the question is...", "But how do we..."

### Equations
- Inline math: use single dollar signs $...$  (e.g., $x^2$)
- Block/display math: use DOUBLE dollar signs $$...$$ on their own lines
- CRITICAL: Any equation containing \\begin{bmatrix}, \\begin{pmatrix}, \\begin{align}, \\begin{cases}, \\\\, or spanning multiple lines MUST use $$ delimiters, NEVER single $
- Block equations must have $$ on their own line before and after the equation
- After EVERY formula/equation, include a numerical worked example with simple concrete values
- Pattern: show formula → "Let us plug in some simple numbers..." → walk through computation → interpret result

### Figure Placeholders
- Use format: {{FIGURE: detailed generation description || short caption}}
- The || separator is MANDATORY — description and caption are different things
- **Description** (before ||): VERY specific — list exact elements, labels, arrows, panels, colors, layout. Enough detail for a designer to recreate it precisely
- **Caption** (after ||): Short (1 sentence, under 15 words) — describes the takeaway, NOT the visual layout
- BAD caption: "Two-row comparison diagram with 4 pipelines showing noise flowing through boxes"
- GOOD caption: "Independent frame generation fails at temporal coherence; joint video diffusion succeeds."
- Place each figure IMMEDIATELY after the paragraph that introduces/motivates it

### Code Blocks
- Python exclusively, pedagogical style
- 2-4 code blocks in practical implementation sections
- Code always follows math, never precedes it
- Include explanatory comments
- Each code block should demonstrate one clear concept`;
}

export function getOutlinePrompt(concept: string): string {
  return `Create a detailed outline for an article about: "${concept}"

The outline should follow this structure:

1. **Title** — compelling, specific title
2. **Subtitle** — one-line hook
3. **Opening** — describe the concrete example/analogy that will ground the reader (be specific — don't just say "use an analogy," describe the actual analogy in detail)

4. **Sections** (4-7 sections) — for each section provide:
   - Section title
   - Key concepts covered
   - Specific analogies/examples to use (describe them)
   - Equations to introduce (write the actual LaTeX)
   - Figure descriptions with the exact format: {{FIGURE: detailed description || short caption}}
   - Code blocks planned (what they demonstrate)
   - Estimated word count for this section

5. **Conclusion** — brief sign-off pattern ("That's it!" or resources list)

## Figure Rules
- Include 6-10 figure placeholders throughout the outline
- Each figure description must be VERY specific: list exact elements, labels, arrows, panels, layout
- Each caption must be short (1 sentence, under 15 words) — the takeaway, not the visual description
- Use the || separator: {{FIGURE: generation description || reader-facing caption}}
- Place figures where visual aids would most help understanding

## Important
- Be specific about everything — this outline will be used to generate the full article
- Match the section flow pattern: Intuitive Example → Concept Introduction → Mathematical Formulation → More Examples → Practical Implementation → Transition

Format the outline as clean markdown.`;
}

export function getArticlePrompt(concept: string, outline: string): string {
  return `Write a complete article about "${concept}" following this approved outline:

${outline}

## Requirements
- Write the FULL article — every section, fully fleshed out, not a summary
- Target length: 3,000-5,000 words
- Follow the Guided Discovery pedagogy strictly
- Match the Vizuara voice from the style profile and reference articles EXACTLY

## Figure Placeholders
- Include all figure placeholders exactly as: {{FIGURE: detailed description || short caption}}
- Description must be VERY specific — list exact elements, labels, arrows, panels, layout
- Caption must be short (1 sentence, under 15 words) — the takeaway
- Place each figure IMMEDIATELY after the paragraph that introduces it
- Never separate a figure from its context

## Equations
- Inline math: single dollar signs $...$ (e.g., $x^2$)
- Block/display math: DOUBLE dollar signs $$...$$ on their OWN lines
- CRITICAL: Matrices (\\begin{bmatrix}), aligned equations (\\begin{align}), and ANY multi-line math MUST use $$ delimiters, NEVER single $
- After EVERY equation, include a numerical worked example:
  "Let us plug in some simple numbers. Suppose [assign small values]... [walk through computation]... This tells us that..."

## Code
- Include 2-4 Python code blocks in the practical sections
- Code always follows the math it implements
- Pedagogical style with explanatory comments
- Each code block should produce a visible result

## Style Checklist
- Open with "Let us start with..." + concrete example
- Use "we" and "Let us..." throughout
- Pose questions to the reader: "But how do we...?", "You might be thinking..."
- Use "This is exactly what we want." after validating results
- Transitions: "This brings us to...", "Now the question is..."
- End with "That's it!" or brief sign-off with references

Write the article now. Output ONLY the article content in markdown format.`;
}

export function getNotebookConceptPlanPrompt(
  concept: string,
  articleContent: string
): string {
  return `Given this article about "${concept}", split the content into 3-5 notebook concepts. Each notebook should teach one coherent topic that can be explored hands-on in Google Colab.

Article content:
${articleContent.slice(0, 8000)}

Return a JSON object with this structure:
{
  "concepts": [
    {
      "id": "01_concept_slug",
      "title": "Notebook Title",
      "objective": "What the student will learn and build",
      "topics": ["topic1", "topic2", "topic3"]
    }
  ]
}

Rules:
- Each notebook should take 15-30 minutes
- Order from foundational to advanced
- Use snake_case for IDs, prefixed with 01_, 02_, etc.
- Each notebook should produce a visible result (plot, model output, etc.)
- Include practical implementations, not just theory

Output ONLY valid JSON.`;
}

export function getNotebookPrompt(
  title: string,
  objective: string,
  topics: string[],
  articleContext: string
): string {
  return `Generate a complete Google Colab teaching notebook for: "${title}"

Objective: ${objective}
Topics to cover: ${topics.join(', ')}

Relevant article context:
${articleContext.slice(0, 6000)}

## OUTPUT FORMAT — CRITICAL

Write the notebook as a **markdown file** using these conventions (a Python script converts it to .ipynb):

- **Code cells**: Use fenced \`\`\`python ... \`\`\` blocks. Each block becomes one code cell.
- **Bash cells**: Use fenced \`\`\`bash ... \`\`\` blocks (lines auto-prefixed with ! for Colab).
- **Markdown cells**: Everything outside code blocks is markdown. Use \`#%%\` on its own line to force a cell break within markdown.
- Do NOT use ===MARKDOWN_CELL=== or ===CODE_CELL=== delimiters.

## NOTEBOOK STRUCTURE — Follow this exact 9-section layout:

# 🚀 ${title}

*Part of the Vizuara series*
*Estimated time: 20-30 minutes*

#%%

## 1. Why Does This Matter?
[Big picture motivation — WHY should the student care? Connect to real-world applications. Show a teaser of the final output.]

#%%

## 2. Building Intuition
[NO CODE in this section. Use analogies, thought experiments, and plain-English explanations to build a mental model.]

### 🤔 Think About This
[Pose a reflective question that primes the student for the math]

#%%

## 3. The Mathematics
[Present equations with LaTeX. After EVERY equation, explain what it means computationally. Connect to the intuition from section 2.]

#%%

## 4. Let's Build It — Component by Component
### 4.1 [First Component]
[Explain what this does and why, then a code cell, then a visualization checkpoint]

### 4.2 [Second Component]
[Continue building incrementally...]

#%%

## 5. 🔧 Your Turn
### TODO: Implement [key function]
[Provide function signature, docstring, step-by-step hints, but NOT the solution. Follow with a verification cell.]

#%%

## 6. Putting It All Together
[Combine all components into a complete pipeline]

#%%

## 7. Training and Results
[Training loop, loss curves, evaluation]

#%%

## 8. 🎯 Final Output
[Generate the final tangible, visually satisfying output]

#%%

## 9. Reflection and Next Steps
### 🤔 Reflection Questions
### 🏆 Optional Challenges

## PEDAGOGY RULES — Follow strictly:

1. **Ground in fundamentals.** If a library hides the core mechanic, implement it manually. Use torch.nn.Linear/Conv2d as building blocks, but build architectures from those primitives.

2. **Every equation gets a computational explanation.** After every equation, add: "This equation says: take X, combine with Y, and produce Z. Computationally, this means..."

3. **Incremental building.** Never dump the full architecture in one cell. Build piece by piece. Test each piece. Visualize each piece.

4. **Strategic TODO sections (3-5 per notebook).** Provide scaffolding:
\`\`\`
def function_name(args):
    """Docstring explaining what to implement"""
    # ============ TODO ============
    # Step 1: ...
    # Step 2: ...
    # ============ END TODO ========
    result = ???  # YOUR CODE HERE
    return result
\`\`\`
Follow each TODO with a ✅ verification cell using asserts.

5. **Visualization checkpoints.** After every major step, add a 📊 visualization code cell. Never go more than 3 code cells without one.

6. **Satisfying final output.** The last code cell must produce something the student can screenshot and feel proud of — generated images, reward curves, attention maps, etc.

7. **Vizuara voice.** Warm, clear, pedagogical. Use "we" and "Let us...", conversational professor tone. Be encouraging at TODO sections.

8. **Colab-ready.** All code must run on Google Colab with a T4 GPU. Keep training times under 10 minutes. Set random seeds for reproducibility. The script auto-prepends a GPU setup cell, so do NOT include one yourself.

9. **Alternation rule.** Never have more than 2 consecutive code cells without markdown. Never have more than 2 consecutive markdown cells without code.

10. **Target 20-30 cells total**, roughly half markdown, half code.

Write the complete notebook now. Output ONLY the markdown content.`;
}

export function getCaseStudyPrompt(
  concept: string,
  articleContent: string
): string {
  return `Create a comprehensive, real-world case study for the topic: "${concept}"

Article context:
${articleContent.slice(0, 8000)}

## OUTPUT STRUCTURE — Follow this EXACT 4-section layout:

# Case Study: [Compelling Title]

## Section 1: Industry Context and Business Problem

Write this as an internal company project brief:

- **Industry**: Which sector and why it matters
- **Company Profile**: A fictional but believable company — include name, founding year, team size, funding stage, key product lines. Make it feel real.
- **Business Challenge**: The specific problem in business terms. What is breaking? What opportunity is being missed?
- **Why It Matters**: Financial impact, user impact, competitive pressure. Include rough dollar figures or user counts.
- **Constraints**: Real-world constraints the team must work within:
  - Budget (compute, headcount)
  - Latency requirements
  - Privacy/compliance (HIPAA, GDPR, etc.)
  - Data availability and quality
  - Deployment environment (edge, cloud, on-prem)

## Section 2: Technical Problem Formulation

This is the MOST IMPORTANT section. Convert the business problem into a rigorous ML/AI formulation built from FIRST PRINCIPLES:

- **Problem Type**: Classification, regression, generation, RL, etc. Explain WHY this framing over alternatives.
- **Input Specification**: Data format, dimensions, modalities. Explain what information each input carries.
- **Output Specification**: What the model produces, format, constraints. Explain design choices.
- **Mathematical Foundation**: Fundamental concepts the student needs — probability distributions, optimization principles, key theorems. Include derivations or intuitive explanations of WHY the math works. Connect back to the business problem.
- **Loss Function**: Mathematical formulation with DETAILED justification. For each term: what it optimizes for, what happens if removed (ablation intuition), how weighting affects training.
- **Evaluation Metrics**: Primary and secondary metrics with target thresholds. Explain why each matters for the business.
- **Baseline**: What naive/traditional approach would work? What performance does it achieve? Why is it insufficient?
- **Why This Concept**: Explain specifically why the article's concept is the right technical approach. What fundamental properties match the problem?

## Section 3: Implementation Notebook Structure

Create a detailed implementation guide with 9 subsections. For each, describe what students must do and provide code scaffolding with TODO stubs.

### 3.1 Data Acquisition Strategy
- What dataset to use (REAL public datasets — NEVER MNIST, CIFAR-10, SVHN, or other toy datasets)
- Data loading and preprocessing
- TODO: Students implement data augmentation or cleaning

### 3.2 Exploratory Data Analysis
- Key distributions to plot
- Anomalies to investigate
- TODO: Students write EDA code and answer guided questions

### 3.3 Baseline Approach
- Implement a simple baseline (rule-based, linear, etc.)
- Evaluate with defined metrics
- TODO: Students implement and evaluate baseline

### 3.4 Model Design
- Architecture connecting back to the article's concept
- Each component explained from first principles — WHY, not just WHAT
- TODO: Students implement core model components (function signatures + docstrings + step hints, NOT full solutions)

### 3.5 Training Strategy
- Optimizer, LR schedule, regularization — explain WHY each choice
- TODO: Students implement training loop with logging

### 3.6 Evaluation
- Quantitative evaluation on held-out test set
- Comparison against baseline
- TODO: Students generate evaluation plots and interpret results

### 3.7 Error Analysis
- Systematic error categorization
- TODO: Students identify and categorize top 3 failure modes

### 3.8 Scalability and Deployment
- How would this model be served in production?
- TODO: Students write inference benchmarking script

### 3.9 Ethical and Regulatory Analysis
- Bias considerations specific to the industry
- TODO: Students write brief ethical impact assessment

For EVERY TODO, provide:
- Clear function signature
- Detailed docstring explaining what to implement
- Step-by-step hints (NOT the solution)
- Verification cells to check correctness

## Section 4: Production and System Design Extension

Write as a system design document for advanced students:

- **Architecture Diagram** (described in text — boxes, arrows, components)
- **API Design**: REST/gRPC endpoints, request/response schemas
- **Serving Infrastructure**: Model serving framework, scaling strategy
- **Latency Budget**: End-to-end breakdown by component
- **Monitoring**: Metrics to track, alerting thresholds
- **Model Drift Detection**: How to detect and handle distribution shift
- **A/B Testing**: Statistical significance, guardrail metrics
- **CI/CD for ML**: Training pipeline automation, model validation gates
- **Cost Analysis**: Estimated cloud compute costs

## CRITICAL RULES
- BANNED datasets: NEVER use MNIST, FashionMNIST, CIFAR-10, CIFAR-100, SVHN, or classroom toys
- Currency: ALWAYS escape dollar signs (\\$23M, \\$1.2B) to prevent LaTeX conflicts
- Every technical choice must be rooted in fundamentals — students should understand WHY
- The case study must feel like it came from a real company's ML team
- Target difficulty: final-year undergraduate / early graduate level
- All 4 sections must be substantial — no placeholder sentences
- The implementation must be solvable on a T4 GPU in under 90 minutes

Output the complete case study in markdown format.`;
}

export function getCaseStudyNotebookPrompt(
  caseStudyContent: string,
  concept: string
): string {
  return `Generate a Google Colab teaching notebook from the Implementation section (Section 3) of this case study about "${concept}".

Case study content:
${caseStudyContent.slice(0, 12000)}

## OUTPUT FORMAT — CRITICAL

Write the notebook as a **markdown file** using these conventions (a Python script converts it to .ipynb):

- **Code cells**: Use fenced \`\`\`python ... \`\`\` blocks. Each block becomes one code cell.
- **Bash cells**: Use fenced \`\`\`bash ... \`\`\` blocks (lines auto-prefixed with ! for Colab).
- **Markdown cells**: Everything outside code blocks is markdown. Use \`#%%\` on its own line to force a cell break within markdown.
- Do NOT use ===MARKDOWN_CELL=== or ===CODE_CELL=== delimiters.

## NOTEBOOK STRUCTURE

# Case Study: [Title from the case study]
## Implementation Notebook

*Vizuara Case Study Series*
*Estimated time: 60-90 minutes*

#%%

## Context

[Brief summary of the industry scenario, company, and problem from Section 1 — enough context so the notebook is self-contained]

#%%

## 3.1 Data Acquisition

[Explanation of the dataset and why it was chosen]

\`\`\`python
# Setup and imports
import torch
import torch.nn as nn
...
\`\`\`

[Continue through all 9 subsections (3.1-3.9) from the case study]

## PEDAGOGY RULES

1. **Include ALL TODO stubs** from Section 3 as actual code cells with function signatures, docstrings, and step-by-step hints. Do NOT provide solutions.

2. **After every TODO**, add a verification cell:
\`\`\`
# ✅ Verification
assert ..., "❌ Check your implementation"
print("✅ Correct!")
\`\`\`

3. **Visualization checkpoints** after every major step. Never go 3+ code cells without a plot or print.

4. **Incremental building.** Never dump full architectures. Build piece by piece, test each piece.

5. **Colab-ready.** All code must run on T4 GPU. Keep training under 10 minutes. Set random seeds.

6. **Vizuara voice.** Warm, conversational professor tone. Use "we" and "Let us...".

7. **Alternation rule.** Never more than 2 consecutive code cells without markdown. Never more than 2 consecutive markdown cells without code.

8. **Target 25-35 cells**, roughly half markdown, half code.

9. **End with a summary** reviewing what the student built and linking back to the business problem.

Write the complete notebook now. Output ONLY the markdown content.`;
}

export function getNarrationScriptPrompt(
  notebookContent: string
): string {
  return `Generate a narration script for this Colab notebook. The narration will be converted to audio using text-to-speech and played alongside the notebook cells.

Each narration audio player will be INJECTED into the notebook right BEFORE the cell it narrates about. So each segment must clearly map to a specific cell or group of cells.

Notebook content:
${notebookContent}

## Requirements:
- Generate one narration segment per major section or group of 1-3 related cells
- Each segment MUST map to a specific location in the notebook
- For markdown cells: explain the concept in a conversational, teaching tone
- For code cells: walk through the code step by step, explain what each part does
- Keep each segment 30-90 seconds when spoken (roughly 75-225 words)
- Use natural, spoken language — avoid reading equations verbatim
- Include transitions between sections

## Output Format

Output a JSON array where each element has:
- "segment_id": a short, descriptive identifier for this segment (e.g., "00_intro", "01_motivation", "02_architecture_code", "03_training_loop"). Use snake_case. Start with a 2-digit number followed by a descriptive name. This becomes the audio filename AND the title shown to students (e.g., "02_architecture_code" displays as "Listen: Architecture Code").
- "cell_indices": array of cell index numbers this segment covers (0-based)
- "insert_before": a short, unique text snippet copied VERBATIM from the notebook content above. This must exactly match text that appears in the FIRST cell this narration should appear BEFORE. The injection script searches the notebook for this exact text to determine where to place the audio player.
- "narration_text": the text to speak aloud
- "duration_estimate_seconds": estimated duration in seconds

Example:
[
  {
    "segment_id": "00_intro",
    "cell_indices": [],
    "insert_before": "",
    "narration_text": "Hey everyone, welcome back to Vizuara. Today we're going to...",
    "duration_estimate_seconds": 45
  },
  {
    "segment_id": "01_why_it_matters",
    "cell_indices": [1, 2],
    "insert_before": "## 1. Why Does This Matter?",
    "narration_text": "Let's start with why this concept matters...",
    "duration_estimate_seconds": 60
  },
  {
    "segment_id": "02_building_encoder",
    "cell_indices": [3],
    "insert_before": "class Encoder(nn.Module):",
    "narration_text": "Now take a look at this code cell. We're building the encoder...",
    "duration_estimate_seconds": 50
  }
]

## CRITICAL rules for insert_before:
- For the FIRST segment (intro), leave insert_before as "" — it will be placed at the top of the notebook
- For ALL other segments, you MUST provide insert_before text
- Copy a heading or first distinctive line from the notebook content above EXACTLY — character for character
- Prefer section headings like "## 1. Why Does This Matter?" or "### 4.1 Building the Encoder"
- For code cells, use the first line of code (e.g., "class Encoder(nn.Module):" or "def train_step(")
- The text must be UNIQUE across all notebook cells — if it appears in multiple cells, use a longer snippet
- NEVER paraphrase or approximate — the injection script does an exact substring match
- If the insert_before text doesn't match any cell, the narration will be placed in the WRONG location

## CRITICAL rules for segment_id:
- Use descriptive names that reflect the section content (e.g., "03_loss_function", "04_training_results")
- The segment_id becomes the title students see: "Listen: Loss Function", "Listen: Training Results"
- Do NOT use generic names like "seg_00", "seg_01" — they display as meaningless "Seg 00"

## Narration style:
- Warm, conversational, like talking to a student one-on-one
- Use "we" and "let's" and "you"
- After important concepts, add pacing: "Take a moment to think about that."
- After code cells: "Go ahead and run that cell now."
- Build anticipation before visualizations: "When you run this cell, you should see..."

Output ONLY valid JSON.`;
}
