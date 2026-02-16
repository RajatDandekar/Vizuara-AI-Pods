'use client';

import { useCallback } from 'react';
import { useCourse } from '@/context/CourseContext';
import type { DiscoverPapersResponse } from '@/types/api';

export function usePaperDiscovery() {
  const { dispatch } = useCourse();

  const discoverPapers = useCallback(
    async (concept: string) => {
      dispatch({ type: 'SET_CONCEPT', concept });
      dispatch({ type: 'PAPERS_LOADING' });

      try {
        const response = await fetch('/api/papers', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ concept }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.error || `Failed to discover papers (HTTP ${response.status})`
          );
        }

        const data: DiscoverPapersResponse = await response.json();
        dispatch({ type: 'PAPERS_LOADED', papers: data.papers });
      } catch (error) {
        dispatch({
          type: 'PAPERS_ERROR',
          error: (error as Error).message || 'Failed to discover papers',
        });
      }
    },
    [dispatch]
  );

  return { discoverPapers };
}
