---
name: analyze-style
description: Analyze Vizuara Substack articles to extract writing style profile
allowed-tools: Read, Bash, Write
argument-hint: (no arguments needed)
---

Read ALL files in `/Users/raj/Desktop/Course_Creator/reference-articles` and perform a deep analysis 
of the writing style. Create a comprehensive style profile at 
`/Users/raj/Desktop/Course_Creator/style-profile.md` covering:

## Analysis Dimensions

1. **Article Structure**
   - How articles typically open (hook style, context-setting pattern)
   - Section flow pattern (e.g., Intro → Intuition → Math → Code → Visualization → Summary)
   - How transitions between sections work
   - How articles conclude
   - Typical article length (word count range)
   - Number and placement of figures, code blocks, equations

2. **Voice & Tone**
   - Formality level (academic vs conversational)
   - Use of first person ("I", "we", "let's")
   - How the reader is addressed ("you", "we together", etc.)
   - Humor and personality patterns
   - Confidence level in claims (hedging vs assertive)

3. **Technical Explanation Style**
   - How new concepts are introduced (analogy-first? definition-first? example-first?)
   - How math is motivated before being shown
   - Depth of mathematical derivations
   - How code relates to math (does code follow equations? precede them?)
   - Use of intuition-building before formalism

4. **Visual & Formatting Patterns**
   - When and why figures are used
   - Figure caption style
   - Code block placement patterns
   - Use of bold, italics, blockquotes
   - Bullet/list usage patterns

5. **Signature Phrases & Patterns**
   - Recurring transitional phrases
   - How questions are posed to the reader
   - Opening and closing patterns
   - Any catchphrases or distinctive expressions

6. **Code Style**
   - Primary language(s) used
   - Comment density and style
   - Whether code is minimal/pedagogical or production-ready
   - Import patterns, variable naming conventions

Write the style profile as a structured document that can be used as a reference
for future article generation. Include SPECIFIC EXAMPLES from the articles for 
each dimension — quote actual sentences as evidence.