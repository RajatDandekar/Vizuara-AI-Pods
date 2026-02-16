'use client';

import { createContext, useContext, useReducer, useEffect, useCallback, type ReactNode } from 'react';
import type { CourseProgress } from '@/types/course';
import {
  getProgress,
  setProgress,
  markArticleRead as markArticleReadStorage,
  markNotebookComplete as markNotebookCompleteStorage,
  markCaseStudyComplete as markCaseStudyCompleteStorage,
} from '@/lib/progress';

const PROGRESS_DEFAULTS: CourseProgress = {
  articleRead: false,
  completedNotebooks: [],
  caseStudyComplete: false,
  lastVisited: '',
};

interface ProgressState {
  courses: Record<string, CourseProgress>;
}

type ProgressAction =
  | { type: 'LOAD'; courseSlug: string; progress: CourseProgress }
  | { type: 'MARK_ARTICLE_READ'; courseSlug: string }
  | { type: 'MARK_NOTEBOOK_COMPLETE'; courseSlug: string; notebookSlug: string }
  | { type: 'MARK_CASE_STUDY_COMPLETE'; courseSlug: string };

function progressReducer(state: ProgressState, action: ProgressAction): ProgressState {
  switch (action.type) {
    case 'LOAD':
      return {
        ...state,
        courses: { ...state.courses, [action.courseSlug]: action.progress },
      };
    case 'MARK_ARTICLE_READ': {
      const current = state.courses[action.courseSlug] || { ...PROGRESS_DEFAULTS };
      const updated: CourseProgress = {
        ...current,
        articleRead: true,
        lastVisited: new Date().toISOString(),
      };
      return {
        ...state,
        courses: { ...state.courses, [action.courseSlug]: updated },
      };
    }
    case 'MARK_NOTEBOOK_COMPLETE': {
      const current = state.courses[action.courseSlug] || { ...PROGRESS_DEFAULTS };
      const notebooks = current.completedNotebooks.includes(action.notebookSlug)
        ? current.completedNotebooks
        : [...current.completedNotebooks, action.notebookSlug];
      const updated: CourseProgress = {
        ...current,
        completedNotebooks: notebooks,
        lastVisited: new Date().toISOString(),
      };
      return {
        ...state,
        courses: { ...state.courses, [action.courseSlug]: updated },
      };
    }
    case 'MARK_CASE_STUDY_COMPLETE': {
      const current = state.courses[action.courseSlug] || { ...PROGRESS_DEFAULTS };
      const updated: CourseProgress = {
        ...current,
        caseStudyComplete: true,
        lastVisited: new Date().toISOString(),
      };
      return {
        ...state,
        courses: { ...state.courses, [action.courseSlug]: updated },
      };
    }
    default:
      return state;
  }
}

interface ProgressContextValue {
  getCourseProgress: (courseSlug: string) => CourseProgress;
  markArticleRead: (courseSlug: string) => void;
  markNotebookComplete: (courseSlug: string, notebookSlug: string) => void;
  markCaseStudyComplete: (courseSlug: string) => void;
}

const ProgressContext = createContext<ProgressContextValue | null>(null);

export function ProgressProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(progressReducer, { courses: {} });

  const getCourseProgress = useCallback(
    (courseSlug: string): CourseProgress => {
      return state.courses[courseSlug] || { ...PROGRESS_DEFAULTS };
    },
    [state.courses]
  );

  const handleMarkArticleRead = useCallback((courseSlug: string) => {
    markArticleReadStorage(courseSlug);
    dispatch({ type: 'MARK_ARTICLE_READ', courseSlug });
  }, []);

  const handleMarkNotebookComplete = useCallback(
    (courseSlug: string, notebookSlug: string) => {
      markNotebookCompleteStorage(courseSlug, notebookSlug);
      dispatch({ type: 'MARK_NOTEBOOK_COMPLETE', courseSlug, notebookSlug });
    },
    []
  );

  const handleMarkCaseStudyComplete = useCallback((courseSlug: string) => {
    markCaseStudyCompleteStorage(courseSlug);
    dispatch({ type: 'MARK_CASE_STUDY_COMPLETE', courseSlug });
  }, []);

  // Load progress for known courses on mount
  useEffect(() => {
    const loadAllProgress = () => {
      // Scan localStorage for all vizuara_progress_ keys
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith('vizuara_progress_')) {
          const slug = key.replace('vizuara_progress_', '');
          const progress = getProgress(slug);
          dispatch({ type: 'LOAD', courseSlug: slug, progress });
        }
      }
    };
    loadAllProgress();
  }, []);

  return (
    <ProgressContext.Provider
      value={{
        getCourseProgress,
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
