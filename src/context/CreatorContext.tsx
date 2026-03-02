'use client';

import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  type ReactNode,
} from 'react';
import type { CreatorProject, CreatorState, CreatorAction } from '@/types/creator';

const initialState: CreatorState = {
  projects: [],
  activeProject: null,
  loading: false,
  streamingContent: '',
  streamingStatus: 'idle',
};

function creatorReducer(
  state: CreatorState,
  action: CreatorAction
): CreatorState {
  switch (action.type) {
    case 'SET_PROJECTS':
      return { ...state, projects: action.projects, loading: false };
    case 'SET_ACTIVE_PROJECT':
      return { ...state, activeProject: action.project };
    case 'UPDATE_PROJECT': {
      const updated = action.project;
      return {
        ...state,
        activeProject:
          state.activeProject?.id === updated.id
            ? updated
            : state.activeProject,
        projects: state.projects.map((p) =>
          p.id === updated.id ? updated : p
        ),
      };
    }
    case 'SET_LOADING':
      return { ...state, loading: action.loading };
    case 'SET_ERROR':
      return { ...state, error: action.error, loading: false };
    case 'CLEAR_ERROR':
      return { ...state, error: undefined };
    case 'STREAM_START':
      return {
        ...state,
        streamingContent: '',
        streamingStatus: 'streaming',
      };
    case 'STREAM_APPEND':
      return {
        ...state,
        streamingContent: state.streamingContent + action.content,
      };
    case 'STREAM_COMPLETE':
      return { ...state, streamingStatus: 'complete' };
    case 'STREAM_ERROR':
      return {
        ...state,
        streamingStatus: 'error',
        error: action.error,
      };
    case 'STREAM_RESET':
      return {
        ...state,
        streamingContent: '',
        streamingStatus: 'idle',
        error: undefined,
      };
    default:
      return state;
  }
}

interface CreatorContextValue {
  state: CreatorState;
  dispatch: React.Dispatch<CreatorAction>;
  fetchProjects: () => Promise<void>;
  fetchProject: (projectId: string) => Promise<void>;
  refreshProject: (projectId: string) => Promise<void>;
}

const CreatorContext = createContext<CreatorContextValue | null>(null);

export function CreatorProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(creatorReducer, initialState);

  const fetchProjects = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', loading: true });
    try {
      const res = await fetch('/api/admin/creator/projects');
      if (!res.ok) throw new Error('Failed to fetch projects');
      const data = await res.json();
      dispatch({ type: 'SET_PROJECTS', projects: data.projects });
    } catch (err) {
      dispatch({
        type: 'SET_ERROR',
        error: err instanceof Error ? err.message : 'Failed to fetch projects',
      });
    }
  }, []);

  const fetchProject = useCallback(async (projectId: string) => {
    dispatch({ type: 'SET_LOADING', loading: true });
    try {
      const res = await fetch(`/api/admin/creator/projects/${projectId}`);
      if (!res.ok) throw new Error('Failed to fetch project');
      const data = await res.json();
      dispatch({ type: 'SET_ACTIVE_PROJECT', project: data.project });
      dispatch({ type: 'SET_LOADING', loading: false });
    } catch (err) {
      dispatch({
        type: 'SET_ERROR',
        error: err instanceof Error ? err.message : 'Failed to fetch project',
      });
    }
  }, []);

  const refreshProject = useCallback(
    async (projectId: string) => {
      try {
        const res = await fetch(`/api/admin/creator/projects/${projectId}`);
        if (!res.ok) return;
        const data = await res.json();
        dispatch({ type: 'UPDATE_PROJECT', project: data.project });
      } catch {
        // Silent refresh failure
      }
    },
    []
  );

  return (
    <CreatorContext.Provider
      value={{ state, dispatch, fetchProjects, fetchProject, refreshProject }}
    >
      {children}
    </CreatorContext.Provider>
  );
}

export function useCreator() {
  const context = useContext(CreatorContext);
  if (!context)
    throw new Error('useCreator must be used within CreatorProvider');
  return context;
}
