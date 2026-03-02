/**
 * Injects narration audio player cells into a Colab notebook (.ipynb JSON).
 * TypeScript port of inject_narration.py
 */

interface NarrationSegmentInput {
  segment_id: string;
  cell_indices?: number[];
  insert_before?: string;
}

interface NotebookCellJson {
  cell_type: string;
  source: string | string[];
  metadata?: Record<string, unknown>;
  execution_count?: null;
  outputs?: unknown[];
  id?: string;
}

interface NotebookJson {
  cells: NotebookCellJson[];
  [key: string]: unknown;
}

function makeLabel(segmentId: string): string {
  const parts = segmentId.split('_');
  let labelParts: string[];

  if (parts.length >= 3 && /^\d+$/.test(parts[0]) && /^\d+$/.test(parts[1])) {
    labelParts = parts.slice(2);
  } else if (parts.length >= 2 && /^\d+$/.test(parts[0])) {
    labelParts = parts.slice(1);
  } else {
    labelParts = parts;
  }

  return labelParts
    .join(' ')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function makeDownloadCell(
  driveFileId: string,
  introSegmentId: string = '00_intro'
): NotebookCellJson {
  const source = [
    '#@title 🎧 Download Narration Audio & Play Introduction\n',
    'import os as _os\n',
    'if not _os.path.exists("/content/narration"):\n',
    '    !pip install -q gdown\n',
    '    import gdown\n',
    `    gdown.download(id="${driveFileId}", output="/content/narration.zip", quiet=False)\n`,
    '    !unzip -q /content/narration.zip -d /content/narration\n',
    '    !rm /content/narration.zip\n',
    '    print(f"Loaded {len(_os.listdir(\'/content/narration\'))} narration segments")\n',
    'else:\n',
    '    print("Narration audio already loaded.")\n',
    '\n',
    'from IPython.display import Audio, display\n',
    `display(Audio("/content/narration/${introSegmentId}.mp3"))`,
  ];

  return {
    cell_type: 'code',
    execution_count: null,
    metadata: { tags: ['narration'], cellView: 'form' },
    outputs: [],
    source,
    id: 'narration_download',
  };
}

function makeAudioCell(segmentId: string): NotebookCellJson {
  const label = makeLabel(segmentId);
  const source = [
    `#@title 🎧 Listen: ${label}\n`,
    'from IPython.display import Audio, display\n',
    'import os as _os\n',
    `_f = "/content/narration/${segmentId}.mp3"\n`,
    'if _os.path.exists(_f):\n',
    '    display(Audio(_f))\n',
    'else:\n',
    '    print("Run the first cell to download narration audio.")',
  ];

  return {
    cell_type: 'code',
    execution_count: null,
    metadata: { tags: ['narration'], cellView: 'form' },
    outputs: [],
    source,
    id: `narration_${segmentId}`,
  };
}

function getCellSource(cell: NotebookCellJson): string {
  const source = cell.source;
  if (Array.isArray(source)) return source.join('');
  return source || '';
}

function findCellByText(cells: NotebookCellJson[], text: string): number | null {
  for (let i = 0; i < cells.length; i++) {
    if (getCellSource(cells[i]).includes(text)) {
      return i;
    }
  }
  return null;
}

/**
 * Inject narration audio player cells into a notebook.
 *
 * @param notebookJson - The parsed .ipynb JSON object
 * @param segments - Array of narration segments with positioning info
 * @param driveFileId - Google Drive file ID for the narration zip
 * @returns Modified notebook JSON with injected audio cells
 */
export function injectNarration(
  notebookJson: NotebookJson,
  segments: NarrationSegmentInput[],
  driveFileId: string
): NotebookJson {
  const nb = JSON.parse(JSON.stringify(notebookJson)) as NotebookJson;

  // Remove any previously injected narration cells (idempotency)
  nb.cells = nb.cells.filter(
    (c) => !(c.metadata?.tags as string[] | undefined)?.includes('narration')
  );

  const cleanCount = nb.cells.length;

  // Build insertion list: [position, segment_id]
  const introSegmentId = segments.length > 0 ? segments[0].segment_id : '00_intro';
  const insertions: Array<[number, string]> = [];

  for (const seg of segments) {
    const sid = seg.segment_id;
    if (sid === introSegmentId) continue;

    let pos: number | null = null;

    // Primary: use insert_before text
    const insertBefore = seg.insert_before || '';
    if (insertBefore) {
      pos = findCellByText(nb.cells, insertBefore);
    }

    // Fallback: use min(cell_indices)
    if (pos === null && seg.cell_indices && seg.cell_indices.length > 0) {
      pos = Math.min(...seg.cell_indices);
      pos = Math.min(pos, cleanCount);
    }

    if (pos === null) pos = 0;

    insertions.push([pos, sid]);
  }

  // Sort descending so we insert bottom-to-top
  insertions.sort((a, b) => {
    if (a[0] !== b[0]) return b[0] - a[0];
    return b[1] < a[1] ? -1 : 1;
  });

  // Insert audio cells bottom-to-top
  for (const [pos, sid] of insertions) {
    nb.cells.splice(pos, 0, makeAudioCell(sid));
  }

  // Insert download+intro cell at position 0
  nb.cells.splice(0, 0, makeDownloadCell(driveFileId, introSegmentId));

  return nb;
}
