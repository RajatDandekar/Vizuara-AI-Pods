import fs from 'fs';

interface NotebookCell {
  cell_type: string;
  source: string[];
  metadata?: { tags?: string[] };
}

interface NotebookJSON {
  cells: NotebookCell[];
}

/**
 * Extract text content from a .ipynb file for use as AI chat context.
 * Skips chatbot and narration injection cells. Caps at 100K chars.
 */
export function extractNotebookContext(filePath: string): string {
  const raw = fs.readFileSync(filePath, 'utf-8');
  const nb: NotebookJSON = JSON.parse(raw);

  const parts: string[] = [];
  for (const cell of nb.cells) {
    const tags = cell.metadata?.tags ?? [];
    if (tags.includes('chatbot') || tags.includes('narration')) continue;

    const src = cell.source.join('');
    if (!src.trim()) continue;

    const label = cell.cell_type.toUpperCase();
    parts.push(`[${label}]\n${src}`);
  }

  const context = parts.join('\n\n---\n\n');
  return context.slice(0, 100_000);
}

/**
 * Extract markdown headings from a notebook for a table of contents.
 */
export function extractNotebookTOC(filePath: string): string[] {
  const raw = fs.readFileSync(filePath, 'utf-8');
  const nb: NotebookJSON = JSON.parse(raw);

  const headings: string[] = [];
  for (const cell of nb.cells) {
    if (cell.cell_type !== 'markdown') continue;
    const tags = cell.metadata?.tags ?? [];
    if (tags.includes('chatbot') || tags.includes('narration')) continue;

    const src = cell.source.join('');
    for (const line of src.split('\n')) {
      const match = line.match(/^(#{1,3})\s+(.+)/);
      if (match) {
        headings.push(match[2].trim());
      }
    }
  }

  return headings;
}
