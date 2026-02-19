'use client';

import { createContext, useContext, useReducer, useEffect, useCallback, type ReactNode } from 'react';
import type { PodProgress } from '@/types/course';
import {
  getPodProgress,
  setPodProgress,
  markPodArticleRead as markArticleReadStorage,
  markPodNotebookComplete as markNotebookCompleteStorage,
  markPodCaseStudyComplete as markCaseStudyCompleteStorage,
} from '@/lib/progress';

const PROGRESS_DEFAULTS: PodProgress = {
  articleRead: false,
  completedNotebooks: [],
  caseStudyComplete: false,
  lastVisited: '',
};

interface ProgressState {
  /** keyed by `{courseSlug}__{podSlug}` */
  pods: Record<string, PodProgress>;
}

type ProgressAction =
  | { type: 'LOAD'; courseSlug: string; podSlug: string; progress: PodProgress }
  | { type: 'MARK_ARTICLE_READ'; courseSlug: string; podSlug: string }
  | { type: 'MARK_NOTEBOOK_COMPLETE'; courseSlug: string; podSlug: string; notebookSlug: string }
  | { type: 'MARK_CASE_STUDY_COMPLETE'; courseSlug: string; podSlug: string };

function makeKey(courseSlug: string, podSlug: string) {
  return `${courseSlug}__${podSlug}`;
}

function progressReducer(state: ProgressState, action: ProgressAction): ProgressState {
  switch (action.type) {
    case 'LOAD': {
      const key = makeKey(action.courseSlug, action.podSlug);
      return { ...state, pods: { ...state.pods, [key]: action.progress } };
    }
    case 'MARK_ARTICLE_READ': {
      const key = makeKey(action.courseSlug, action.podSlug);
      const current = state.pods[key] || { ...PROGRESS_DEFAULTS };
      return {
        ...state,
        pods: {
          ...state.pods,
          [key]: { ...current, articleRead: true, lastVisited: new Date().toISOString() },
        },
      };
    }
    case 'MARK_NOTEBOOK_COMPLETE': {
      const key = makeKey(action.courseSlug, action.podSlug);
      const current = state.pods[key] || { ...PROGRESS_DEFAULTS };
      const notebooks = current.completedNotebooks.includes(action.notebookSlug)
        ? current.completedNotebooks
        : [...current.completedNotebooks, action.notebookSlug];
      return {
        ...state,
        pods: {
          ...state.pods,
          [key]: { ...current, completedNotebooks: notebooks, lastVisited: new Date().toISOString() },
        },
      };
    }
    case 'MARK_CASE_STUDY_COMPLETE': {
      const key = makeKey(action.courseSlug, action.podSlug);
      const current = state.pods[key] || { ...PROGRESS_DEFAULTS };
      return {
        ...state,
        pods: {
          ...state.pods,
          [key]: { ...current, caseStudyComplete: true, lastVisited: new Date().toISOString() },
        },
      };
    }
    default:
      return state;
  }
}

interface ProgressContextValue {
  getPodProgressState: (courseSlug: string, podSlug: string) => PodProgress;
  markArticleRead: (courseSlug: string, podSlug: string) => void;
  markNotebookComplete: (courseSlug: string, podSlug: string, notebookSlug: string) => void;
  markCaseStudyComplete: (courseSlug: string, podSlug: string) => void;
}

const ProgressContext = createContext<ProgressContextValue | null>(null);

export function ProgressProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(progressReducer, { pods: {} });

  const getPodProgressState = useCallback(
    (courseSlug: string, podSlug: string): PodProgress => {
      return state.pods[makeKey(courseSlug, podSlug)] || { ...PROGRESS_DEFAULTS };
    },
    [state.pods]
  );

  const handleMarkArticleRead = useCallback((courseSlug: string, podSlug: string) => {
    markArticleReadStorage(courseSlug, podSlug);
    dispatch({ type: 'MARK_ARTICLE_READ', courseSlug, podSlug });
  }, []);

  const handleMarkNotebookComplete = useCallback(
    (courseSlug: string, podSlug: string, notebookSlug: string) => {
      markNotebookCompleteStorage(courseSlug, podSlug, notebookSlug);
      dispatch({ type: 'MARK_NOTEBOOK_COMPLETE', courseSlug, podSlug, notebookSlug });
    },
    []
  );

  const handleMarkCaseStudyComplete = useCallback((courseSlug: string, podSlug: string) => {
    markCaseStudyCompleteStorage(courseSlug, podSlug);
    dispatch({ type: 'MARK_CASE_STUDY_COMPLETE', courseSlug, podSlug });
  }, []);

  // This context is currently unused — progress is managed via direct localStorage calls.
  // Kept for potential future use.
  useEffect(() => {
    // No-op on mount for now
  }, []);

  return (
    <ProgressContext.Provider
      value={{
        getPodProgressState,
        markArticleRead: handleMarkArticleRead,
        markNotebookComplete: handleMarkNotebookComplete,
        markCaseStudyComplete: handleMarkCaseStudyComplete,
      }}
    >
      {children}
    </ProgressContext.Provider>
  );
}

export function useProgress(): ProgressContextValue {
  const context = useContext(ProgressContext);
  if (!context) {
    throw new Error('useProgress must be used within a ProgressProvider');
  }
  return context;
}
