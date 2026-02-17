# Course Creator Skills

## Skill: Add New Concept Suggestion
When asked to add a new concept to the suggestion list:
1. Open `src/lib/constants.ts`
2. Add the concept string to the `SUGGESTED_CONCEPTS` array
3. Keep the array alphabetically sorted

## Skill: Modify Prompt Templates
When asked to change how papers, summaries, architectures, or notebooks are generated:
1. Open `src/lib/prompts.ts` — this is the single source of truth for all AI prompts
2. There are 4 prompt functions:
   - `paperDiscoveryPrompt(concept)` — identifies historically significant papers
   - `paperSummaryPrompt(concept, paper)` — generates paper summaries
   - `paperArchitecturePrompt(concept, paper)` — generates architecture deep dives
   - `paperNotebookPrompt(concept, paper)` — generates Colab notebook content
3. Edit the relevant function and rebuild

## Skill: Change AI Model
When asked to change the AI model used:
1. Open `src/lib/constants.ts`
2. Update `MODEL_NAME` to the desired model (e.g., `claude-sonnet-4-5-20250929`, `claude-opus-4-6`)
3. Adjust max token limits if needed

## Skill: Add New Paper Section
When asked to add a new section type to papers (beyond summary/architecture/notebook):
1. Add the new section type to `SectionKey` in `src/context/CourseContext.tsx`
2. Add the section state to `PaperState` in `src/types/paper.ts`
3. Create a new API route in `src/app/api/<section>/route.ts`
4. Create a new prompt function in `src/lib/prompts.ts`
5. Create a new component in `src/components/papers/Paper<Section>.tsx`
6. Wire it into `PaperCardExpanded.tsx`

## Skill: Customize UI Theme
When asked to change colors, fonts, or styling:
1. Open `src/app/globals.css`
2. Modify the CSS variables in `:root` and `@theme inline`
3. Key variables: `--accent-blue`, `--background`, `--foreground`, `--card-bg`, `--card-border`

## Skill: Add Number of Papers Config
When asked to change the number of papers discovered:
1. Open `src/lib/constants.ts`
2. Change `NUM_PAPERS` to the desired count (default: 5)
