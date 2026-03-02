/**
 * Converts markdown with ```python fences and #%% markers to .ipynb JSON.
 * TypeScript port of notebook_generator.py
 */

const NBFORMAT_VERSION = 4;
const NBFORMAT_MINOR = 5;

interface NotebookCell {
  cell_type: 'code' | 'markdown';
  source: string[];
  metadata: Record<string, unknown>;
  id?: string;
  execution_count?: null;
  outputs?: unknown[];
}

interface NotebookJson {
  nbformat: number;
  nbformat_minor: number;
  metadata: Record<string, unknown>;
  cells: NotebookCell[];
}

function ensureList(source: string): string[] {
  const lines = source.split('\n');
  return lines.map((line, i) => (i < lines.length - 1 ? line + '\n' : line));
}

function createMarkdownCell(source: string, cellId?: string): NotebookCell {
  const cell: NotebookCell = {
    cell_type: 'markdown',
    metadata: {},
    source: ensureList(source),
  };
  if (cellId) cell.id = cellId;
  return cell;
}

function createCodeCell(source: string, cellId?: string): NotebookCell {
  const cell: NotebookCell = {
    cell_type: 'code',
    execution_count: null,
    metadata: {},
    outputs: [],
    source: ensureList(source),
  };
  if (cellId) cell.id = cellId;
  return cell;
}

function createEmptyNotebook(title: string): NotebookJson {
  return {
    nbformat: NBFORMAT_VERSION,
    nbformat_minor: NBFORMAT_MINOR,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3',
      },
      language_info: {
        name: 'python',
        version: '3.10.0',
        mimetype: 'text/x-python',
        file_extension: '.py',
        codemirror_mode: { name: 'ipython', version: 3 },
        pygments_lexer: 'ipython3',
        nbconvert_exporter: 'python',
      },
      colab: {
        provenance: [],
        gpuType: 'T4',
        name: title,
      },
      accelerator: 'GPU',
    },
    cells: [],
  };
}

function parseMarkdownToCells(markdownContent: string): NotebookCell[] {
  const cells: NotebookCell[] = [];
  const lines = markdownContent.split('\n');
  let currentBlock: string[] = [];
  let currentType: 'markdown' | 'code' = 'markdown';
  let codeLanguage = '';
  let cellCounter = 0;

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Check for code block start
    if (line.startsWith('```python') || line.startsWith('```Python')) {
      if (currentBlock.length > 0 && currentType === 'markdown') {
        const content = currentBlock.join('\n').trim();
        if (content) {
          cellCounter++;
          cells.push(createMarkdownCell(content, `cell_${cellCounter}`));
        }
        currentBlock = [];
      }
      currentType = 'code';
      codeLanguage = 'python';
      i++;
      continue;
    }

    if (line.startsWith('```bash') || line.startsWith('```shell')) {
      if (currentBlock.length > 0 && currentType === 'markdown') {
        const content = currentBlock.join('\n').trim();
        if (content) {
          cellCounter++;
          cells.push(createMarkdownCell(content, `cell_${cellCounter}`));
        }
        currentBlock = [];
      }
      currentType = 'code';
      codeLanguage = 'bash';
      i++;
      continue;
    }

    if (line.startsWith('```') && currentType === 'code') {
      let codeContent: string;

      if (codeLanguage === 'bash') {
        const bashLines = currentBlock.map((bl) =>
          bl.trim() && !bl.trim().startsWith('#') ? `!${bl}` : bl
        );
        codeContent = bashLines.join('\n');
      } else {
        codeContent = currentBlock.join('\n');
      }

      if (codeContent.trim()) {
        cellCounter++;
        cells.push(createCodeCell(codeContent, `cell_${cellCounter}`));
      }

      currentBlock = [];
      currentType = 'markdown';
      i++;
      continue;
    }

    if (line.startsWith('```') && currentType === 'markdown') {
      // Non-python code block — keep as markdown
      currentBlock.push(line);
      i++;
      continue;
    }

    // Check for explicit cell break
    if (line.trim() === '#%%' && currentType === 'markdown') {
      const content = currentBlock.join('\n').trim();
      if (content) {
        cellCounter++;
        cells.push(createMarkdownCell(content, `cell_${cellCounter}`));
      }
      currentBlock = [];
      i++;
      continue;
    }

    currentBlock.push(line);
    i++;
  }

  // Flush remaining block
  if (currentBlock.length > 0) {
    const content = currentBlock.join('\n').trim();
    if (content) {
      cellCounter++;
      if (currentType === 'code') {
        cells.push(createCodeCell(content, `cell_${cellCounter}`));
      } else {
        cells.push(createMarkdownCell(content, `cell_${cellCounter}`));
      }
    }
  }

  return cells;
}

function addColabSetupCell(cells: NotebookCell[]): NotebookCell[] {
  const setupCode = `# 🔧 Setup: Run this cell first!
# Check GPU availability and install dependencies

import torch
import sys

# Check GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    device = torch.device('cpu')
    print("⚠️ No GPU detected. Some cells may run slowly.")
    print("   Go to Runtime → Change runtime type → GPU")

print(f"\\n📦 Python {sys.version.split()[0]}")
print(f"🔥 PyTorch {torch.__version__}")

# Set random seeds for reproducibility
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"🎲 Random seed set to {SEED}")

%matplotlib inline`;

  const setupCell = createCodeCell(setupCode, 'setup_cell');
  return [setupCell, ...cells];
}

/**
 * Convert markdown content to a complete .ipynb JSON string.
 */
export function generateNotebook(
  title: string,
  markdownContent: string,
  includeSetup: boolean = true
): string {
  const notebook = createEmptyNotebook(title);
  let cells = parseMarkdownToCells(markdownContent);

  if (includeSetup) {
    cells = addColabSetupCell(cells);
  }

  notebook.cells = cells;
  return JSON.stringify(notebook, null, 1);
}

/**
 * Parse markdown content into cells without creating a full notebook.
 * Useful for inspecting the cell structure.
 */
export function parseMarkdown(markdownContent: string): NotebookCell[] {
  return parseMarkdownToCells(markdownContent);
}
