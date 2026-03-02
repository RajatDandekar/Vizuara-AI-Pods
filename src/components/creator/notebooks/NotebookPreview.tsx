'use client';

import MarkdownRenderer from '@/components/ui/MarkdownRenderer';

interface NotebookPreviewProps {
  content: string;
  title: string;
}

export default function NotebookPreview({ content, title }: NotebookPreviewProps) {
  // Parse cell delimiters to show cell-by-cell preview
  const cells = parseCells(content);

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold text-foreground">{title}</h3>
      <p className="text-base text-text-muted">{cells.length} cells</p>

      <div className="space-y-2">
        {cells.map((cell, i) => (
          <div
            key={i}
            className={`border rounded-lg overflow-hidden ${
              cell.type === 'code'
                ? 'border-accent-blue/30'
                : 'border-card-border'
            }`}
          >
            {/* Cell header */}
            <div
              className={`px-3 py-2 text-base font-medium ${
                cell.type === 'code'
                  ? 'bg-accent-blue/5 text-accent-blue'
                  : 'bg-background text-text-muted'
              }`}
            >
              {cell.type === 'code' ? `Code [${i + 1}]` : `Markdown [${i + 1}]`}
            </div>

            {/* Cell content */}
            <div className="p-4">
              {cell.type === 'code' ? (
                <pre className="font-mono text-foreground whitespace-pre-wrap overflow-x-auto leading-relaxed">
                  {cell.content}
                </pre>
              ) : (
                <MarkdownRenderer content={cell.content} />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

interface Cell {
  type: 'markdown' | 'code';
  content: string;
}

/**
 * Parse markdown into notebook cells.
 * Matches notebook_generator.py's parse_markdown_to_cells():
 * - ```python ... ``` blocks → code cells
 * - ```bash ... ``` blocks → code cells (prefixed with !)
 * - #%% on its own line → markdown cell break
 * - Everything else → markdown cells
 * Also supports legacy ===MARKDOWN_CELL=== / ===CODE_CELL=== delimiters.
 */
function parseCells(content: string): Cell[] {
  // Try legacy format first
  const legacyRegex = /===(MARKDOWN_CELL|CODE_CELL)===\n([\s\S]*?)===END_CELL===/g;
  let match;
  const legacyCells: Cell[] = [];
  while ((match = legacyRegex.exec(content)) !== null) {
    legacyCells.push({
      type: match[1] === 'CODE_CELL' ? 'code' : 'markdown',
      content: match[2].trim(),
    });
  }
  if (legacyCells.length > 0) return legacyCells;

  // Parse markdown format (matching notebook_generator.py)
  const cells: Cell[] = [];
  const lines = content.split('\n');
  let currentBlock: string[] = [];
  let currentType: 'markdown' | 'code' = 'markdown';

  const flushBlock = () => {
    const text = currentBlock.join('\n').trim();
    if (text) {
      cells.push({ type: currentType, content: text });
    }
    currentBlock = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Code block start
    if (
      (line.startsWith('```python') || line.startsWith('```Python')) &&
      currentType === 'markdown'
    ) {
      flushBlock();
      currentType = 'code';
      continue;
    }

    // Bash block start
    if (
      (line.startsWith('```bash') || line.startsWith('```shell')) &&
      currentType === 'markdown'
    ) {
      flushBlock();
      currentType = 'code';
      continue;
    }

    // Code block end
    if (line.startsWith('```') && currentType === 'code') {
      flushBlock();
      currentType = 'markdown';
      continue;
    }

    // Explicit cell break in markdown
    if (line.trim() === '#%%' && currentType === 'markdown') {
      flushBlock();
      continue;
    }

    currentBlock.push(line);
  }

  // Flush remaining
  flushBlock();

  // Fallback: if nothing parsed, show as single markdown cell
  if (cells.length === 0 && content.trim()) {
    cells.push({ type: 'markdown', content: content.trim() });
  }

  return cells;
}
