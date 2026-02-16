# Vizuara Narration Style Guide

## Overview

This guide defines how narration for Vizuara Colab notebooks should sound.
The narration is meant to feel like a personal tutor sitting next to the student,
walking them through the notebook. It is NOT a formal lecture.

## Voice and Tone

- **Warm and approachable** — like explaining to a friend over coffee
- **Confident but not arrogant** — you know the material, but you remember what it's like to learn it
- **Genuinely excited** — especially when a cool result is about to happen
- **Patient** — never rush through difficult concepts
- **Encouraging** — especially at TODO sections and challenging parts

## Spoken Language Rules

- Use contractions: "let's", "we'll", "that's", "don't", "isn't"
- Use "we" and "you": "Let's see what happens when we run this" and "You'll notice that..."
- Avoid academic jargon unless explaining it: Don't say "we employ" — say "we use"
- Avoid reading variable names character by character: Say "the variable d-k" not "d underscore k"
- For code: describe what it DOES, don't read the syntax
- For equations: translate to plain English first, then reference the math
- Keep sentences short. Spoken language has shorter phrases than written.

## Segment Structure Patterns

### Introduction Segment (~30 seconds)
```
Hey everyone, welcome to [Notebook Topic]. 
In this notebook, we're going to [what they'll build].
By the end, you'll have [tangible outcome].
This should take about [X] minutes.
Let's get started.
```

### Concept Explanation (~30-60 seconds each)
```
[Connector: "Now", "So", "Here's the thing", "Let me explain"]
[Core idea in one sentence]
[Analogy or intuition]
[Connect to what they already know]
[What this means for the code we're about to write]
```

### Before a Code Cell (~20-30 seconds)
```
[What this code does]
[The key lines to pay attention to]
[Why we're doing it this way]
"Go ahead and run this cell."
```

### After a Code Cell / Visualization (~15-20 seconds)
```
[What they should see]
[What it means]
[Connect to the concept]
```

### Before a TODO Cell (~20-30 seconds)
```
"Alright, this one's on you."
[What they need to implement]
[A hint — but not the answer]
[Encouragement]
"Pause the audio, give it a try, and come back when you're done."
```

### After a TODO Cell (~10-15 seconds)
```
"Welcome back."
[Brief confirmation of what the correct output looks like]
[If they got it right, congratulate them]
```

### Section Transition (~15 seconds)
```
[Summarize what was just covered]
[Preview what's coming next]
```

### Closing Segment (~30-45 seconds)
```
"And that's it."
[Recap the full journey]
[What they've accomplished]
[Tease the next notebook if applicable]
"See you in the next one."
```

## Things to AVOID

- Reading markdown headings aloud ("Section 4 point 2 colon...")
- Reading code syntax verbatim ("import torch dot nn as nn...")
- Long pauses with no content
- Apologizing or hedging ("I'm not sure if...", "This might be wrong...")
- Saying "as mentioned earlier" — just restate the concept briefly
- Filler words: "basically", "actually", "so yeah", "right?"
- Being condescending: "This is really easy" (it might not be for the student)

## Duration Guidelines

| Notebook Length | Total Narration Target |
|----------------|----------------------|
| Short (10-15 cells) | 5-8 minutes |
| Medium (20-30 cells) | 10-15 minutes |
| Long (30-50 cells) | 15-25 minutes |

Each individual segment should be **30 seconds to 2 minutes max.**
Anything longer than 2 minutes should be split into multiple segments.

## Pronunciation Notes

- PyTorch: "Pie-torch"
- LSTM: "L-S-T-M" (spell it out)
- RNN: "R-N-N"
- RSSM: "R-S-S-M"
- VAE: "V-A-E" or "vay"
- JEPA: "Jeh-pah"
- LaTeX: "Lay-tek"
- NumPy: "Num-pie"
- d_k: "d-k" (not "d underscore k")
- softmax: "soft-max"
- ReLU: "Reh-loo"
- epoch: "ee-pock"