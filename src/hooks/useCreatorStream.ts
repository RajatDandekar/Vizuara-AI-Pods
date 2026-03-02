'use client';

import { useRef, useCallback } from 'react';
import { useCreator } from '@/context/CreatorContext';

interface UseCreatorStreamOptions {
  onComplete?: (fullText: string) => void;
  onError?: (error: string) => void;
}

export function useCreatorStream(options?: UseCreatorStreamOptions) {
  const { dispatch } = useCreator();
  const abortRef = useRef<AbortController | null>(null);
  const bufferRef = useRef('');
  const fullTextRef = useRef('');
  const rafRef = useRef<number>(0);

  const flushBuffer = useCallback(() => {
    if (bufferRef.current) {
      dispatch({ type: 'STREAM_APPEND', content: bufferRef.current });
      bufferRef.current = '';
    }
    rafRef.current = 0;
  }, [dispatch]);

  const startStream = useCallback(
    async (endpoint: string, body: Record<string, unknown>) => {
      // Cancel any existing stream
      if (abortRef.current) {
        abortRef.current.abort();
      }

      const controller = new AbortController();
      abortRef.current = controller;
      bufferRef.current = '';
      fullTextRef.current = '';

      dispatch({ type: 'STREAM_START' });

      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) {
          const errText = await res.text();
          throw new Error(errText || `HTTP ${res.status}`);
        }

        const reader = res.body?.getReader();
        if (!reader) throw new Error('No response body');

        const decoder = new TextDecoder();
        let lineBuffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          lineBuffer += decoder.decode(value, { stream: true });
          const lines = lineBuffer.split('\n');
          lineBuffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data: ')) continue;

            try {
              const event = JSON.parse(trimmed.slice(6));

              if (event.type === 'text') {
                bufferRef.current += event.content;
                fullTextRef.current += event.content;
                if (!rafRef.current) {
                  rafRef.current = requestAnimationFrame(flushBuffer);
                }
              } else if (event.type === 'done') {
                // Flush remaining buffer
                if (bufferRef.current) {
                  dispatch({
                    type: 'STREAM_APPEND',
                    content: bufferRef.current,
                  });
                  bufferRef.current = '';
                }
                dispatch({ type: 'STREAM_COMPLETE' });
                options?.onComplete?.(fullTextRef.current);
              } else if (event.type === 'error') {
                dispatch({
                  type: 'STREAM_ERROR',
                  error: event.message,
                });
                options?.onError?.(event.message);
              }
            } catch {
              // Skip unparseable lines
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          dispatch({ type: 'STREAM_RESET' });
          return;
        }
        const message =
          err instanceof Error ? err.message : 'Stream failed';
        dispatch({ type: 'STREAM_ERROR', error: message });
        options?.onError?.(message);
      }
    },
    [dispatch, flushBuffer, options]
  );

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
    bufferRef.current = '';
    dispatch({ type: 'STREAM_RESET' });
  }, [dispatch]);

  return { startStream, cancel };
}
