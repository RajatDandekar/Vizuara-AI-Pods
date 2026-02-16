export interface Paper {
  id: string;
  title: string;
  authors: string[];
  year: number;
  venue: string;
  arxivUrl: string;
  oneLiner: string;
  significance: string;
}

export type SectionStatus = 'idle' | 'loading' | 'streaming' | 'complete' | 'error';

export interface SectionState {
  status: SectionStatus;
  content: string;
  error?: string;
}

export interface NotebookSectionState extends SectionState {
  notebookJson: NotebookDocument | null;
}

export interface PaperState extends Paper {
  summary: SectionState;
  architecture: SectionState;
  notebook: NotebookSectionState;
}

export type CourseStatus = 'idle' | 'discovering' | 'ready' | 'error';

export interface CourseState {
  concept: string;
  status: CourseStatus;
  papers: PaperState[];
  activePaperId: string | null;
  error?: string;
}

// Re-export notebook types for convenience
export interface NotebookCell {
  cell_type: 'code' | 'markdown';
  source: string[];
  metadata: Record<string, unknown>;
  id: string;
  execution_count?: number | null;
  outputs?: Array<Record<string, unknown>>;
}

export interface NotebookDocument {
  nbformat: number;
  nbformat_minor: number;
  metadata: {
    kernelspec: {
      display_name: string;
      language: string;
      name: string;
    };
    language_info: {
      name: string;
      version: string;
      codemirror_mode: {
        name: string;
        version: number;
      };
    };
    colab?: {
      name: string;
      provenance: never[];
    };
  };
  cells: NotebookCell[];
}
