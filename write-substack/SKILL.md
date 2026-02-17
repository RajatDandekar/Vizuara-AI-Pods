---
name: write-substack
description: Write an in-depth Vizuara Substack article on a given topic with 
  figures, code, and LaTeX equations in the Vizuara writing style
argument-hint: <topic description>
---

# Vizuara Substack Article Writer

You are writing an in-depth Substack article for the Vizuara publication.

## MANDATORY FIRST STEPS

1. **Read the style profile**: Read `/Users/raj/Desktop/Course_Creator/style-profile.md` completely
2. **Read 2-3 reference articles**: Read at least 2-3 articles from 
   `/Users/raj/Desktop/Course_Creator/reference-articles` that are closest in topic to $ARGUMENTS
3. **Internalize the voice** before writing a single word

## ARTICLE CREATION PROCESS

### Step 1: Outline
Create a detailed outline at `/Users/raj/Desktop/Course_Creator/output/articles/[topic-slug]/outline.md` with:
- Proposed title and subtitle
- Section-by-section plan
- Where each figure will go and what it will show
- Where each code block will go and what it will demonstrate  
- Where each equation will appear
- Estimated word count per section

**STOP HERE and show the outline to the user for approval before proceeding.**

### Step 2: Draft Prose
Write the full article prose in the Vizuara style to 
`/Users/raj/Desktop/Course_Creator/output/articles/[topic-slug]/draft.md`. 

Rules:
- Follow the structure, voice, and patterns from the style profile EXACTLY
- Use placeholder tags for figures: `{{FIGURE: description of what to generate}}`
  - Figure descriptions must be VERY specific — list exact elements, labels, arrows, panels, and layout so the generated image will match the caption
- Use placeholder tags for equations: `{{EQUATION: LaTeX code here}}`
- After EVERY equation/formula, include a numerical worked example with simple concrete values (see style profile "Numerical Worked Examples" section)
- Write all code blocks inline (Claude generates these directly)
- Match the typical article length from the style profile
- Place each figure placeholder immediately after the paragraph that introduces/motivates it — never separate a figure from its context

### Step 3: Generate Figures

**IMPORTANT: Choose the right tool for each figure type.**

For each `{{FIGURE: ...}}` placeholder, first classify the figure:

- **Excalidraw** — Use for: flowcharts, technical diagrams, architecture maps, timeline/schedule diagrams, side-by-side comparisons, hardware topology diagrams, pipeline diagrams, any figure with boxes/arrows/flow. Generate using `generate_excalidraw.py`.
- **PaperBanana** — Use for: conceptual illustrations, analogy-based visuals, artistic/creative figures, figures with real-world imagery (kitchens, restaurants, people). Generate using `generate_figure.py`.

For **PaperBanana** figures:
```bash
python /Users/raj/.claude/skills/write-substack/scripts/generate_figure.py \
  --description "detailed description of the figure" \
  --style "clean, publication-ready, Vizuara aesthetic" \
  --output "output/articles/[topic-slug]/figures/figure_N.png"
```

For **Excalidraw** figures, design the elements JSON and generate:
```bash
python /Users/raj/.claude/skills/write-substack/scripts/generate_excalidraw.py \
  --elements "output/articles/[topic-slug]/figures/figure_N_elements.json" \
  --output "output/articles/[topic-slug]/figures/figure_N.png"
```
See `/Users/raj/.claude/skills/write-substack/scripts/excalidraw_rules.md` for element sizing guidelines.

### Step 4: Render LaTeX Equations
For each `{{EQUATION: ...}}` placeholder, render to image:
```bash
python .claude/skills/write-substack/scripts/render_latex.py \
  --latex "E = mc^2" \
  --output "output/articles/[topic-slug]/equations/eq_N.png"
```

### Step 5: Assemble Final Article
Run the assembler to replace all placeholders with actual images and produce 
the final Substack-ready markdown:
```bash
python .claude/skills/write-substack/scripts/assemble_article.py \
  --draft "output/articles/[topic-slug]/draft.md" \
  --figures-dir "/Users/raj/Desktop/Course_Creator/output/articles/[topic-slug]/figures/" \
  --equations-dir "output/articles/[topic-slug]/equations/" \
  --output "/Users/raj/Desktop/Course_Creator/output/articles/[topic-slug]/final.md"
```

### Step 6: Quality Check
Read the final article and compare against the style profile. Perform ALL of these checks:

**Voice & Style:**
- Does it sound like the Vizuara articles?
- Are transitions smooth?
- Is the technical depth appropriate?

**Figure Captions (CRITICAL — common mistake):**
- Captions MUST be short (1 sentence, under 15 words) — they describe the takeaway, not the visual layout
- NEVER use the generation description as the caption. "Two-row comparison diagram with 4 pipelines showing noise flowing through boxes..." is NOT a caption. "Independent frame generation fails at temporal coherence; joint video diffusion succeeds." IS a caption.
- Every caption in the final article must read naturally as a standalone sentence a reader would understand
- If any caption is longer than ~20 words or reads like a figure specification, rewrite it immediately
- The `{{FIGURE: ... || caption}}` format separates description from caption — verify the `||` separator is present in every figure tag

**Figure-Caption-Image Consistency:**
- For EVERY figure: read the caption text, then verify the generated image actually depicts what the caption describes
- If a caption says "Three-panel diagram showing X, Y, Z" — confirm the image has three panels with those specific elements
- If a caption mentions specific labels, arrows, or annotations — confirm they appear in the image
- If ANY mismatch is found: regenerate the figure with a more precise description, or rewrite the caption to match what the image actually shows

**Figure Placement:**
- Each figure must appear immediately after the paragraph that introduces/discusses it
- A figure about "training pipeline" must NOT appear next to text about "data collection"
- Read the text before and after each figure — does the figure illustrate the surrounding context?
- If a figure is misplaced, move it to the correct location in final.md

**Numerical Examples:**
- Every formula, equation, or mathematical approach MUST have a worked numerical example
- Check: after each equation, is there a "Let us plug in some simple numbers..." walkthrough?
- If missing, add one with small, easy-to-follow values

**References:**
- References must be formatted as a clean list — one reference per line
- Use format: `Author et al., "Title" (Year)` — each on its own bullet or line

Report any concerns to the user.

### Step 7: Upload Figures to Google Drive

Before publishing to Notion, upload all figure images with **article-slug prefixed names** to avoid filename collisions in the shared Drive folder:

```bash
SLUG="[topic-slug]"
for fig in /Users/raj/Desktop/Course_Creator/output/articles/${SLUG}/figures/figure_*.png; do
  NEWNAME="${SLUG}-$(basename $fig)"
  cp "$fig" "/tmp/$NEWNAME"
  /Users/raj/.local/bin/rclone copy "/tmp/$NEWNAME" gdrive:/ \
    --drive-root-folder-id 1kvHB2fPlnbH6-SXd-SP0hoahcr8yaIER -v
  /Users/raj/.local/bin/rclone link "gdrive:/${NEWNAME}" \
    --drive-root-folder-id 1kvHB2fPlnbH6-SXd-SP0hoahcr8yaIER
done
```

For each figure, extract the Drive file ID and construct the **Notion-compatible** direct image URL:
`https://lh3.googleusercontent.com/d/{FILE_ID}=w2000`

**CRITICAL:** Do NOT use `https://drive.google.com/uc?export=view&id=...` — Notion cannot follow those redirects and images will fail to load.

Keep a mapping of figure filename → image URL for use in Step 8.

### Step 8: Publish to Notion

Use the centralized publish script at `/Users/raj/.claude/skills/write-substack/scripts/publish_to_notion.py`. Do NOT write a new publish script per article.

First, save the figure URL mapping from Step 7 to a JSON file:
```bash
# Save figure URLs as JSON (generated during Step 7)
cat > output/articles/[topic-slug]/figure_urls.json << 'EOF'
{
  "figures/figure_1.png": "https://lh3.googleusercontent.com/d/FILE_ID_1=w2000",
  "figures/figure_2.png": "https://lh3.googleusercontent.com/d/FILE_ID_2=w2000"
}
EOF
```

Then run the publish script:
```bash
python /Users/raj/.claude/skills/write-substack/scripts/publish_to_notion.py \
  --article-dir output/articles/[topic-slug] \
  --title "Article Title Here" \
  --figure-urls output/articles/[topic-slug]/figure_urls.json
```

The script handles everything:
- Creating a Notion page under parent `30501fbe477180d893bbf1079b9d432a`
- Converting markdown to Notion blocks (headings, paragraphs with rich text, images, equations, code, tables, lists, dividers)
- Batching blocks (max 100 per API call)
- Using `lh3.googleusercontent.com` image URLs

### Step 9: Share and Report

After creating the Notion page:

1. **Get the page URL** from the Notion API response
2. **Tell the user** the article has been published with the Notion link
3. The user can then:
   - Open the link to review
   - Click "Share" → "Publish" in Notion to make it publicly accessible
   - Share the public URL with students

**Ask the user to review** using AskUserQuestion:
- Ask: "The article has been published to Notion. Does it look good, or are there changes needed?"
- Options: "Looks great!" / "Needs changes"

If changes are needed, use the Notion MCP tools to update the page content directly.