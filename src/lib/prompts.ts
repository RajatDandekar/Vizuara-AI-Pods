import { NUM_PAPERS } from './constants';

export function paperDiscoveryPrompt(concept: string): string {
  return `You are an AI research historian and educator. Given an AI/ML concept, identify the most historically significant papers that a student should read to deeply understand that concept.

CONCEPT: "${concept}"

Identify exactly ${NUM_PAPERS} papers that are foundational to understanding "${concept}". Order them chronologically (earliest first). These should be landmark papers that introduced key ideas, not survey papers.

For each paper, provide:
- title: The exact paper title
- authors: Lead author(s) (use "et al." for >3 authors)
- year: Publication year
- venue: Conference or journal (e.g., "NeurIPS 2017", "ICML 2014", "arXiv preprint")
- arxivUrl: The arxiv.org URL if available, otherwise the most canonical URL
- oneLiner: A single sentence describing the paper's core contribution (max 15 words)
- significance: 2-3 sentences explaining why this paper matters for understanding ${concept}

CRITICAL RULES:
- Only include real, published papers. Do not fabricate papers.
- If you are not confident a paper exists with the exact title, use your best knowledge but stay factual.
- The papers should tell a coherent story: how the concept evolved from early ideas to its current form.
- Include the seminal/original paper that introduced the concept if one exists.
- Generate a short kebab-case id for each paper based on its title and year.

Respond with ONLY valid JSON matching this exact schema (no markdown, no code fences, no explanation):
{
  "concept": "${concept}",
  "papers": [
    {
      "id": "kebab-case-short-name-year",
      "title": "...",
      "authors": ["..."],
      "year": 0,
      "venue": "...",
      "arxivUrl": "...",
      "oneLiner": "...",
      "significance": "..."
    }
  ]
}`;
}

export function paperSummaryPrompt(
  concept: string,
  paper: { title: string; authors: string[]; year: number; venue: string }
): string {
  return `You are an expert AI educator creating a study guide. Write a comprehensive summary of the following research paper, specifically in the context of helping a student understand "${concept}".

PAPER: "${paper.title}" by ${paper.authors.join(', ')} (${paper.year}, ${paper.venue})

Write your summary in markdown format with the following structure:

## Overview
A 2-3 paragraph overview of what this paper is about, what problem it solves, and why it was important at the time of publication.

## Key Contributions
A bulleted list of 3-5 main contributions of this paper. Each bullet should be 1-2 sentences.

## Main Results
Summarize the key experimental results and benchmarks. What did the paper demonstrate? Include specific numbers/metrics where relevant.

## Impact & Legacy
How did this paper influence the field? What subsequent work did it enable? How does it connect to modern approaches?

## Connection to ${concept}
Explicitly explain how this paper relates to and advances the understanding of "${concept}". This is the most important section for the student.

STYLE RULES:
- Write for a graduate student who has basic ML knowledge but is learning ${concept} for the first time.
- Be precise and technical but always explain jargon.
- Use concrete examples where possible.
- Keep the total length to approximately 600-800 words.`;
}

export function paperArchitecturePrompt(
  concept: string,
  paper: { title: string; authors: string[]; year: number; venue: string }
): string {
  return `You are an expert AI educator who specializes in explaining neural network architectures. Create a detailed architecture deep dive for the following paper.

PAPER: "${paper.title}" by ${paper.authors.join(', ')} (${paper.year}, ${paper.venue})
CONTEXT: The student is learning about "${concept}"

Write in markdown with the following structure:

## Architecture Overview
A high-level description of the model architecture. What are the main components? How do they connect?

## Step-by-Step Data Flow
Walk through how data flows through the architecture step by step. For each step:
1. Describe the input shape/dimensions (e.g., "Input: batch of sequences, shape [B, T] where B=batch size, T=sequence length")
2. Describe the transformation that happens
3. Describe the output shape/dimensions
4. Explain WHY this transformation is important

Use concrete dimension examples. For instance, if the paper uses specific hidden dimensions (like d_model=512), use those.

## Key Innovation Deep Dive
Take the single most important architectural innovation from this paper and explain it in extreme detail:
- What is the mathematical formulation?
- Show the key equations in LaTeX (use $...$ for inline and $$...$$  for block)
- What is the intuition behind it?
- A small numerical walkthrough with made-up but realistic numbers

## Practical Considerations
- What are the computational complexity considerations?
- What are common implementation pitfalls?
- What hyperparameters are most important and why?

## Architecture Diagram
Create a clear ASCII art diagram showing the key components and data flow. Use box-drawing characters for clarity.

STYLE RULES:
- Be extremely precise with tensor dimensions. Always use the notation [dim1, dim2, ...].
- When you reference a mathematical operation, always specify what shapes go in and come out.
- Keep the total length to approximately 800-1200 words.
- Target a graduate student audience.`;
}

export function paperNotebookPrompt(
  concept: string,
  paper: { title: string; authors: string[]; year: number; venue: string }
): string {
  return `You are an expert AI educator creating a Google Colab notebook. Generate the COMPLETE content for a Jupyter notebook that implements the key ideas from the following paper.

PAPER: "${paper.title}" by ${paper.authors.join(', ')} (${paper.year}, ${paper.venue})
CONTEXT: Teaching the student about "${concept}"

NOTEBOOK STRUCTURE (use exactly this structure with the markers shown):

===MARKDOWN_CELL===
# ${paper.title} - Implementation Notebook
**Paper:** ${paper.title} (${paper.year})
**Concept:** ${concept}

This notebook implements the core ideas from this paper to help you understand ${concept}.
===END_CELL===

===CODE_CELL===
# Install dependencies (if needed)
!pip install torch numpy matplotlib -q
===END_CELL===

===MARKDOWN_CELL===
## 1. Setup and Imports
First, let's import the necessary libraries.
===END_CELL===

===CODE_CELL===
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
print("Setup complete!")
===END_CELL===

Continue with this pattern. Create a notebook with approximately 12-18 cells that covers:

1. **Setup** (installs + imports) - 2-3 cells
2. **Core Implementation** - Implement the key architectural component from scratch using PyTorch
   - Build each sub-component as a separate class/function
   - Each code cell should be preceded by a markdown cell explaining what it does and why
   - Include shape assertions (assert tensor.shape == (...)) to help students verify dimensions
   - Print intermediate results so students can see what's happening
   - 6-8 cells
3. **Demonstration** - Create a small example showing the implementation in action
   - Use small, tractable dimensions (e.g., d_model=64, not 512)
   - Print intermediate shapes and values
   - 2-3 cells
4. **Visualization** - Create 1-2 matplotlib visualizations showing something meaningful
   - E.g., attention weights heatmap, loss curves, feature visualizations
   - 2-3 cells
5. **Exercises** - End with 2-3 small exercises for the student (markdown cells with TODO code cells)
   - 2-3 cells

CRITICAL RULES:
- Every code cell MUST be syntactically valid Python that runs in Google Colab.
- Use PyTorch as the primary framework.
- Use small dimensions so everything runs quickly on Colab's free tier (no GPU required if possible).
- Include comments in code explaining non-obvious lines.
- Each markdown cell should be educational, not just a header.
- Use the exact ===MARKDOWN_CELL===, ===CODE_CELL===, and ===END_CELL=== delimiters.
- Do not include any text outside of cells.
- Make sure all code cells produce visible output (print statements, plots, etc.).`;
}
