# Course Creator

An AI-powered educational tool that helps students understand AI/ML concepts through historically significant papers.

## Project Overview
- **Stack**: Next.js 16 (App Router), React 19, Tailwind CSS v4, Framer Motion, Google GenAI SDK
- **Purpose**: Takes an AI concept as input, discovers landmark papers, and for each generates summaries, architecture deep dives, and Colab notebooks

## Architecture
- `src/app/` — Next.js pages and API routes
- `src/components/` — React components (ui/, concept/, papers/, layout/, animations/)
- `src/lib/` — Core libraries (Gemini client, prompts, notebook parser, SSE utilities)
- `src/hooks/` — Custom React hooks (streaming, paper discovery)
- `src/context/` — React Context for global state management
- `src/types/` — TypeScript type definitions

## Key Files
- `src/lib/prompts.ts` — All prompt templates (most important file for output quality)
- `src/lib/stream.ts` — SSE streaming utilities for API routes
- `src/lib/notebook.ts` — Jupyter notebook (.ipynb) parser and builder
- `src/context/CourseContext.tsx` — Global state with useReducer
- `src/hooks/useStreamingResponse.ts` — SSE consumption hook with buffering

## Commands
- `npm run dev` — Start development server
- `npm run build` — Production build
- `npm run lint` — Run ESLint

## Environment
- Requires `GEMINI_API_KEY` in `.env.local`
- Uses `gemini-2.5-flash` model (configurable in `src/lib/constants.ts`)
- Get a free API key at https://aistudio.google.com

## Conventions
- All client components marked with `'use client'` directive
- Tailwind CSS v4 with CSS-based configuration in `globals.css` (no tailwind.config)
- Framer Motion for all animations
- SSE (Server-Sent Events) for streaming API responses
- Notebook cells use `===MARKDOWN_CELL===` / `===CODE_CELL===` / `===END_CELL===` delimiters

## Vizuara Substack Writing Tool

This project includes a Substack article writing tool for the Vizuara publication.

### Key files:
- `/Users/raj/Desktop/Course_Creator/style-profile.md` — Writing style guide (read this before any article)
- `/Users/raj/Desktop/Course_Creator/reference-articles` — 10+ reference articles for style matching
- `/Users/raj/.claude/skills/write-substack` — Main article writing skill
- `/Users/raj/.claude/skills/analyze-style` — One-time style analysis skill

### Writing rules:
- ALWAYS read the style profile before writing
- ALWAYS show the outline for approval before drafting
- Match the Vizuara voice precisely — review reference articles if unsure
- Use Claude (yourself) for all code generation
- Use PaperBanana/Gemini for figure generation
- Use LaTeX (inline $$...$$ for Substack) for equations
- Generate figures one at a time, confirming each looks right

### Publishing:
- Articles are published to Notion as child pages under "Vizuara Notes" (parent page ID: `30501fbe477180d893bbf1079b9d432a`)
- Each article gets its own Notion page with a public URL
- Figures are uploaded to Google Drive via rclone, then referenced by Drive URL in Notion
- Equations use Notion's native KaTeX support (no LaTeX-to-PNG needed)
- Notion MCP server configured in `.mcp.json`

### Dependencies:
- Python 3.10+
- Google Gemini API key (for figures): set GOOGLE_API_KEY env var
- matplotlib, Pillow (for equation rendering fallback)
- rclone (for uploading figures to Google Drive)
- Notion MCP server (`@notionhq/notion-mcp-server`) for publishing
