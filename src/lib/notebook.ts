import type { NotebookCell, NotebookDocument } from '@/types/paper';

export function buildNotebook(
  rawText: string,
  concept: string,
  paperTitle: string
): NotebookDocument {
  const cells = parseCells(rawText);

  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3',
      },
      language_info: {
        name: 'python',
        version: '3.10.0',
        codemirror_mode: {
          name: 'ipython',
          version: 3,
        },
      },
      colab: {
        name: `${concept}_${paperTitle.replace(/[^a-zA-Z0-9]/g, '_')}.ipynb`,
        provenance: [],
      },
    },
    cells,
  };
}

function parseCells(rawText: string): NotebookCell[] {
  const cells: NotebookCell[] = [];
  const cellPattern = /===(MARKDOWN_CELL|CODE_CELL)===\n([\s\S]*?)===END_CELL===/g;
  let match: RegExpExecArray | null;
  let cellIndex = 0;

  while ((match = cellPattern.exec(rawText)) !== null) {
    const cellType = match[1] === 'CODE_CELL' ? 'code' : 'markdown';
    const source = match[2].trim();

    // nbformat requires source as array of strings (lines ending with \n)
    const sourceLines = source.split('\n').map((line, i, arr) =>
      i < arr.length - 1 ? line + '\n' : line
    );

    const cell: NotebookCell = {
      cell_type: cellType,
      source: sourceLines,
      metadata: {},
      id: `cell-${cellIndex++}`,
    };

    if (cellType === 'code') {
      cell.execution_count = null;
      cell.outputs = [];
    }

    cells.push(cell);
  }

  // If no cells were parsed, create a fallback notebook with the raw content
  if (cells.length === 0) {
    cells.push({
      cell_type: 'markdown',
      source: ['# Notebook Content\n', '\n', 'The notebook content could not be parsed into cells. Here is the raw content:\n'],
      metadata: {},
      id: 'cell-0',
    });
    cells.push({
      cell_type: 'code',
      source: rawText.split('\n').map((line, i, arr) =>
        i < arr.length - 1 ? line + '\n' : line
      ),
      metadata: {},
      id: 'cell-1',
      execution_count: null,
      outputs: [],
    });
  }

  return cells;
}
