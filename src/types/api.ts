import type { Paper } from './paper';

export interface DiscoverPapersRequest {
  concept: string;
}

export interface DiscoverPapersResponse {
  concept: string;
  papers: Paper[];
}

export interface PaperContentRequest {
  concept: string;
  paper: {
    title: string;
    authors: string[];
    year: number;
    venue: string;
  };
}

export interface SSETextEvent {
  type: 'text';
  content: string;
}

export interface SSEDoneEvent {
  type: 'done';
}

export interface SSENotebookEvent {
  type: 'notebook';
  content: string; // JSON string of NotebookDocument
}

export interface SSEErrorEvent {
  type: 'error';
  message: string;
}

export type SSEEvent = SSETextEvent | SSEDoneEvent | SSENotebookEvent | SSEErrorEvent;
