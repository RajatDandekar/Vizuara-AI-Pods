// Pod Creator — internal tool types

export type ProjectStep =
  | 'article'
  | 'figures'
  | 'preview'
  | 'export'
  | 'notebooks'
  | 'case-study'
  | 'narration'
  | 'publish';

export type StepStatus = 'pending' | 'in-progress' | 'complete' | 'error';

export type FigureMethod = 'paperbanana';

export interface FigureState {
  id: string; // e.g. "figure_1"
  description: string; // from {{FIGURE: description || caption}}
  caption: string;
  method: FigureMethod;
  status: 'pending' | 'generating' | 'ready' | 'approved' | 'error';
  jobId?: string;
  storagePath?: string; // Supabase Storage path: {projectId}/figures/{id}.png
  driveFileId?: string;
  driveUrl?: string;
  error?: string;
}

export interface NotebookState {
  id: string; // e.g. "01_concept_name"
  title: string;
  objective: string;
  status: 'pending' | 'generating' | 'converting' | 'ready' | 'uploaded' | 'error';
  markdownContent?: string;
  ipynbStoragePath?: string; // Supabase Storage path: {projectId}/notebooks/{id}.ipynb
  colabUrl?: string;
  driveFileId?: string;
  error?: string;
}

export interface CaseStudyState {
  status: 'pending' | 'generating' | 'ready' | 'error';
  content?: string;
  pdfStoragePath?: string; // Supabase Storage path: {projectId}/case-study/case_study.pdf

  // Notebook generated from Section 3
  notebookMarkdown?: string;
  notebookIpynbStoragePath?: string; // Supabase Storage path
  notebookColabUrl?: string;
  notebookDriveFileId?: string;
  notebookStatus?: 'pending' | 'generating' | 'converting' | 'ready' | 'uploaded' | 'error';
  notebookError?: string;

  error?: string;
}

export interface NarrationSegment {
  id: string;           // e.g. "01_flow_intuition_seg_00"
  segment_id: string;   // same as id — used by narration injector
  notebookId: string;
  cellIndex: number;
  cell_indices: number[];  // used by narration injector
  insert_before: string;   // used by narration injector for positioning
  text: string;
  narration_text: string;  // same as text — used by narration injector
  audioStoragePath?: string; // Supabase Storage path: {projectId}/narration/{id}.mp3
  duration?: number;
}

export interface NarrationState {
  status: 'pending' | 'scripting' | 'script-ready' | 'synthesizing' | 'injecting' | 'complete' | 'error';
  segments: NarrationSegment[];
  zipDriveFileId?: string;
  error?: string;
}

export interface StepProgress {
  step: ProjectStep;
  status: StepStatus;
  completedAt?: string;
}

export interface CreatorProject {
  id: string;
  concept: string;
  podTitle: string;
  podSlug: string;
  courseSlug: string;
  createdAt: string;
  updatedAt: string;
  createdBy: string; // uid
  currentStep: ProjectStep;
  steps: StepProgress[];

  // Article
  outline?: string;
  outlineApproved?: boolean;
  articleDraft?: string;
  articleApproved?: boolean;

  // Figures
  figures: FigureState[];

  // Notebooks
  conceptPlan?: ConceptPlan;
  conceptPlanApproved?: boolean;
  notebooks: NotebookState[];

  // Case Study
  caseStudy?: CaseStudyState;

  // Narration
  narration?: NarrationState;

  // Publishing
  publishedAt?: string;
}

export interface ConceptPlan {
  concepts: ConceptPlanItem[];
}

export interface ConceptPlanItem {
  id: string;
  title: string;
  objective: string;
  topics: string[];
}

// Job system for long-running tasks
export type JobStatus = 'pending' | 'running' | 'complete' | 'error';

export interface Job {
  id: string;
  projectId: string;
  type: 'figure-generate' | 'notebook-convert' | 'audio-synthesize' | 'publish';
  status: JobStatus;
  progress?: number; // 0-100
  message?: string;
  result?: Record<string, unknown>;
  error?: string;
  createdAt: string;
  updatedAt: string;
}

// API request/response types
export interface CreateProjectRequest {
  concept: string;
  podTitle: string;
  podSlug: string;
  courseSlug: string;
}

export interface GenerateOutlineRequest {
  projectId: string;
}

export interface GenerateArticleRequest {
  projectId: string;
  outline: string;
}

export interface GenerateFigureRequest {
  projectId: string;
  figureId: string;
  description: string;
}

export interface GenerateNotebookRequest {
  projectId: string;
  conceptId: string;
  title: string;
  objective: string;
  topics: string[];
}

// SSE event types for creator streams
export interface CreatorSSETextEvent {
  type: 'text';
  content: string;
}

export interface CreatorSSEDoneEvent {
  type: 'done';
}

export interface CreatorSSEErrorEvent {
  type: 'error';
  message: string;
}

export interface CreatorSSEProgressEvent {
  type: 'progress';
  step: string;
  message: string;
  progress?: number;
}

export type CreatorSSEEvent =
  | CreatorSSETextEvent
  | CreatorSSEDoneEvent
  | CreatorSSEErrorEvent
  | CreatorSSEProgressEvent;

// Creator context state
export interface CreatorState {
  projects: CreatorProject[];
  activeProject: CreatorProject | null;
  loading: boolean;
  error?: string;
  streamingContent: string;
  streamingStatus: 'idle' | 'streaming' | 'complete' | 'error';
}

export type CreatorAction =
  | { type: 'SET_PROJECTS'; projects: CreatorProject[] }
  | { type: 'SET_ACTIVE_PROJECT'; project: CreatorProject | null }
  | { type: 'UPDATE_PROJECT'; project: CreatorProject }
  | { type: 'SET_LOADING'; loading: boolean }
  | { type: 'SET_ERROR'; error: string }
  | { type: 'CLEAR_ERROR' }
  | { type: 'STREAM_START' }
  | { type: 'STREAM_APPEND'; content: string }
  | { type: 'STREAM_COMPLETE' }
  | { type: 'STREAM_ERROR'; error: string }
  | { type: 'STREAM_RESET' };
