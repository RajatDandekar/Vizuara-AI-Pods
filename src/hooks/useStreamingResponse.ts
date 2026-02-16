'use client';

import { useCallback, useRef } from 'react';
import { useCourse, type SectionKey } from '@/context/CourseContext';
import type { NotebookDocument } from '@/types/paper';

interface UseStreamingResponseOptions {
  endpoint: string;
  paperId: string;
  section: SectionKey;
}

export function useStreamingResponse({
  endpoint,
  paperId,
  section,
}: UseStreamingResponseOptions) {
  const { dispatch } = useCourse();
  const abortControllerRef = useRef<AbortController | null>(null);
  const bufferRef = useRef('');
  const rafRef = useRef<number | null>(null);

  const flushBuffer = useCallback(() => {
    if (bufferRef.current) {
      dispatch({
        type: 'SECTION_APPEND',
        paperId,
        section,
        content: bufferRef.current,
      });
      bufferRef.current = '';
    }
    rafRef.current = null;
  }, [dispatch, paperId, section]);

  const startStreaming = useCallback(
    async (body: Record<string, unknown>) => {
      // Abort any existing stream
      abortControllerRef.current?.abort();
      abortControllerRef.current = new AbortController();

      dispatch({ type: 'SECTION_START', paperId, section });

      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        if (!response.body) {
          throw new Error('No response body');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let lineBuffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          lineBuffer += decoder.decode(value, { stream: true });

          const lines = lineBuffer.split('\n');
          lineBuffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'text') {
                // Buffer text chunks and flush via rAF for performance
                bufferRef.current += data.content;
                if (!rafRef.current) {
                  rafRef.current = requestAnimationFrame(flushBuffer);
                }
              } else if (data.type === 'notebook') {
                // Flush any remaining buffer first
                flushBuffer();
                dispatch({
                  type: 'NOTEBOOK_GENERATED',
                  paperId,
                  notebookJson: data.content as NotebookDocument,
                });
              } else if (data.type === 'done') {
                // Flush remaining buffer
                flushBuffer();
                dispatch({ type: 'SECTION_COMPLETE', paperId, section });
              } else if (data.type === 'error') {
                flushBuffer();
                dispatch({
                  type: 'SECTION_ERROR',
                  paperId,
                  section,
                  error: data.message,
                });
              }
            } catch {
              // Skip malformed JSON lines
            }
          }
        }

        // Final flush
        flushBuffer();

      } catch (error) {
        if ((error as Error).name === 'AbortError') {
          // Reset to idle so the component can retry on remount (e.g., React strict mode)
          dispatch({ type: 'SECTION_RESET', paperId, section });
        } else {
          flushBuffer();
          dispatch({
            type: 'SECTION_ERROR',
            paperId,
            section,
            error: (error as Error).message || 'Connection failed',
          });
        }
      }
    },
    [endpoint, paperId, section, dispatch, flushBuffer]
  );

  const cancel = useCallback(() => {
    abortControllerRef.current?.abort();
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  return { startStreaming, cancel };
}
