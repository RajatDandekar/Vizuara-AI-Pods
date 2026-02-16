'use client';

import React, { createContext, useContext, useReducer, type ReactNode } from 'react';
import type {
  CourseState,
  Paper,
  PaperState,
  NotebookDocument,
} from '@/types/paper';

// --- Actions ---

type CourseAction =
  | { type: 'SET_CONCEPT'; concept: string }
  | { type: 'PAPERS_LOADING' }
  | { type: 'PAPERS_LOADED'; papers: Paper[] }
  | { type: 'PAPERS_ERROR'; error: string }
  | { type: 'SET_ACTIVE_PAPER'; paperId: string | null }
  | { type: 'SECTION_START'; paperId: string; section: SectionKey }
  | { type: 'SECTION_APPEND'; paperId: string; section: SectionKey; content: string }
  | { type: 'SECTION_COMPLETE'; paperId: string; section: SectionKey }
  | { type: 'SECTION_ERROR'; paperId: string; section: SectionKey; error: string }
  | { type: 'NOTEBOOK_GENERATED'; paperId: string; notebookJson: NotebookDocument }
  | { type: 'SECTION_RESET'; paperId: string; section: SectionKey }
  | { type: 'RESET' };

type SectionKey = 'summary' | 'architecture' | 'notebook';

// --- Initial State ---

const initialState: CourseState = {
  concept: '',
  status: 'idle',
  papers: [],
  activePaperId: null,
};

function createPaperState(paper: Paper): PaperState {
  return {
    ...paper,
    summary: { status: 'idle', content: '' },
    architecture: { status: 'idle', content: '' },
    notebook: { status: 'idle', content: '', notebookJson: null },
  };
}

// --- Reducer ---

function courseReducer(state: CourseState, action: CourseAction): CourseState {
  switch (action.type) {
    case 'SET_CONCEPT':
      return { ...state, concept: action.concept };

    case 'PAPERS_LOADING':
      return { ...state, status: 'discovering', papers: [], activePaperId: null, error: undefined };

    case 'PAPERS_LOADED':
      return {
        ...state,
        status: 'ready',
        papers: action.papers.map(createPaperState),
      };

    case 'PAPERS_ERROR':
      return { ...state, status: 'error', error: action.error };

    case 'SET_ACTIVE_PAPER':
      return { ...state, activePaperId: action.paperId };

    case 'SECTION_START':
      return {
        ...state,
        papers: state.papers.map((p) =>
          p.id === action.paperId
            ? {
                ...p,
                [action.section]: {
                  ...p[action.section],
                  status: 'streaming' as const,
                  content: '',
                  error: undefined,
                },
              }
            : p
        ),
      };

    case 'SECTION_APPEND':
      return {
        ...state,
        papers: state.papers.map((p) =>
          p.id === action.paperId
            ? {
                ...p,
                [action.section]: {
                  ...p[action.section],
                  content: p[action.section].content + action.content,
                },
              }
            : p
        ),
      };

    case 'SECTION_COMPLETE':
      return {
        ...state,
        papers: state.papers.map((p) =>
          p.id === action.paperId
            ? {
                ...p,
                [action.section]: {
                  ...p[action.section],
                  status: 'complete' as const,
                },
              }
            : p
        ),
      };

    case 'SECTION_ERROR':
      return {
        ...state,
        papers: state.papers.map((p) =>
          p.id === action.paperId
            ? {
                ...p,
                [action.section]: {
                  ...p[action.section],
                  status: 'error' as const,
                  error: action.error,
                },
              }
            : p
        ),
      };

    case 'SECTION_RESET':
      return {
        ...state,
        papers: state.papers.map((p) =>
          p.id === action.paperId
            ? {
                ...p,
                [action.section]: {
                  ...p[action.section],
                  status: 'idle' as const,
                  content: '',
                  error: undefined,
                },
              }
            : p
        ),
      };

    case 'NOTEBOOK_GENERATED':
      return {
        ...state,
        papers: state.papers.map((p) =>
          p.id === action.paperId
            ? {
                ...p,
                notebook: {
                  ...p.notebook,
                  notebookJson: action.notebookJson,
                },
              }
            : p
        ),
      };

    case 'RESET':
      return initialState;

    default:
      return state;
  }
}

// --- Context ---

interface CourseContextValue {
  state: CourseState;
  dispatch: React.Dispatch<CourseAction>;
}

const CourseContext = createContext<CourseContextValue | null>(null);

export function CourseProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(courseReducer, initialState);

  return (
    <CourseContext.Provider value={{ state, dispatch }}>
      {children}
    </CourseContext.Provider>
  );
}

export function useCourse() {
  const context = useContext(CourseContext);
  if (!context) {
    throw new Error('useCourse must be used within a CourseProvider');
  }
  return context;
}

export type { CourseAction, SectionKey };
