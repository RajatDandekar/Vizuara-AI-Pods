'use client';

import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface MarkdownRendererProps {
  content: string;
  streaming?: boolean;
}

const katexOptions = {
  strict: 'ignore' as const,
  trust: true,
  errorColor: '#888888',
};

/**
 * Fix LaTeX issues so remarkMath + rehypeKatex render correctly.
 *
 * remarkMath block math rules:
 *   - Opening $$ MUST be the only content on its line
 *   - Closing $$ MUST be the only content on its line
 *   - $$content$$ on a single line is treated as INLINE math (not block)
 *
 * Claude often generates:
 *   $$J = \begin{bmatrix}        ← content after opening $$
 *   ...
 *   \end{bmatrix}$$              ← content before closing $$
 *
 * This function:
 *   1. Converts multi-line $...$ to $$...$$ (Claude sometimes uses single $)
 *   2. Ensures all $$ markers are alone on their own line
 */
function fixLatexDelimiters(text: string): string {
  // Step 1: Fix multi-line $...$ → $$...$$
  let fixed = fixMultiLineSingleDollar(text);

  // Step 2: Ensure all $$ are on their own lines
  fixed = normalizeBlockMathDelimiters(fixed);

  return fixed;
}

/**
 * Find $...$ that span multiple lines and convert to $$...$$.
 */
function fixMultiLineSingleDollar(text: string): string {
  const lines = text.split('\n');
  const result: string[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Skip lines inside fenced code blocks
    if (line.trimStart().startsWith('```')) {
      result.push(line);
      i++;
      // Push everything until closing ```
      while (i < lines.length && !lines[i].trimStart().startsWith('```')) {
        result.push(lines[i]);
        i++;
      }
      if (i < lines.length) {
        result.push(lines[i]); // closing ```
        i++;
      }
      continue;
    }

    // Find positions of lone $ signs (not part of $$) on this line
    const loneDollarPositions = findLoneDollarPositions(line);

    // If even number, they're all paired on this line — fine
    if (loneDollarPositions.length % 2 === 0) {
      result.push(line);
      i++;
      continue;
    }

    // Odd: the last $ opens a multi-line block. Look for closing $ on a later line.
    const openPos = loneDollarPositions[loneDollarPositions.length - 1];
    let closingLineIdx = -1;
    let closingCharPos = -1;

    for (let k = i + 1; k < lines.length; k++) {
      // Skip code blocks when searching
      if (lines[k].trimStart().startsWith('```')) break;

      const positions = findLoneDollarPositions(lines[k]);
      if (positions.length > 0) {
        closingLineIdx = k;
        closingCharPos = positions[0]; // first lone $ on that line
        break;
      }
    }

    if (closingLineIdx <= i) {
      result.push(line);
      i++;
      continue;
    }

    // Convert multi-line $...$ to $$...$$
    const beforeOpen = line.slice(0, openPos).trimEnd();
    const afterOpen = line.slice(openPos + 1);
    if (beforeOpen) result.push(beforeOpen);
    result.push('$$');
    if (afterOpen.trim()) result.push(afterOpen);

    for (let k = i + 1; k < closingLineIdx; k++) {
      result.push(lines[k]);
    }

    const closingLine = lines[closingLineIdx];
    const beforeClose = closingLine.slice(0, closingCharPos);
    const afterClose = closingLine.slice(closingCharPos + 1).trimStart();
    if (beforeClose.trim()) result.push(beforeClose);
    result.push('$$');
    if (afterClose) result.push(afterClose);

    i = closingLineIdx + 1;
  }

  return result.join('\n');
}

/**
 * Ensure every $$ is alone on its own line.
 * remarkMath requires this for block math parsing.
 *
 * Handles:
 *   $$content...    → $$\ncontent...
 *   ...content$$    → ...content\n$$
 *   $$content$$     → $$\ncontent\n$$
 */
function normalizeBlockMathDelimiters(text: string): string {
  const lines = text.split('\n');
  const result: string[] = [];
  let inCodeBlock = false;

  for (const line of lines) {
    // Track fenced code blocks — don't modify content inside them
    if (line.trimStart().startsWith('```')) {
      inCodeBlock = !inCodeBlock;
      result.push(line);
      continue;
    }
    if (inCodeBlock) {
      result.push(line);
      continue;
    }

    const trimmed = line.trim();

    // Already just $$ on its own line — perfect
    if (trimmed === '$$') {
      result.push(line);
      continue;
    }

    // Single-line $$...$$  (e.g., $$x^2 + y^2$$)
    // Split into three lines: $$\ncontent\n$$
    if (trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length > 4) {
      const inner = trimmed.slice(2, -2).trim();
      if (inner) {
        result.push('$$');
        result.push(inner);
        result.push('$$');
        continue;
      }
    }

    // Line starts with $$ but has content after (e.g., $$J = \begin{bmatrix})
    if (trimmed.startsWith('$$') && !trimmed.startsWith('$$$')) {
      const content = trimmed.slice(2).trim();
      if (content) {
        result.push('$$');
        result.push(content);
        continue;
      }
    }

    // Line ends with $$ but has content before (e.g., \end{bmatrix}$$)
    if (trimmed.endsWith('$$') && !trimmed.endsWith('$$$') && trimmed !== '$$') {
      const content = trimmed.slice(0, -2).trim();
      if (content) {
        result.push(content);
        result.push('$$');
        continue;
      }
    }

    result.push(line);
  }

  return result.join('\n');
}

/**
 * Find positions of lone $ signs (not part of $$) in a line.
 */
function findLoneDollarPositions(line: string): number[] {
  const positions: number[] = [];
  for (let j = 0; j < line.length; j++) {
    if (line[j] === '$') {
      const prev = j > 0 && line[j - 1] === '$';
      const next = j + 1 < line.length && line[j + 1] === '$';
      if (!prev && !next) {
        positions.push(j);
      }
    }
  }
  return positions;
}

export default function MarkdownRenderer({ content, streaming = false }: MarkdownRendererProps) {
  const processedContent = useMemo(() => {
    if (streaming) return content;
    return fixLatexDelimiters(content);
  }, [content, streaming]);

  const remarkPlugins = useMemo(
    () => (streaming ? [remarkGfm] : [remarkGfm, remarkMath]),
    [streaming]
  );

  const rehypePlugins = useMemo(
    () =>
      streaming
        ? [[rehypeHighlight] as [typeof rehypeHighlight]]
        : [
            [rehypeHighlight] as [typeof rehypeHighlight],
            [rehypeKatex, katexOptions] as [typeof rehypeKatex, typeof katexOptions],
          ],
    [streaming]
  );

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
}
