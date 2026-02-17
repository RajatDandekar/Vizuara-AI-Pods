'use client';

import { useCallback, useRef } from 'react';

interface UseChatStreamOptions {
  onChunk: (text: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

export function useChatStream({ onChunk, onDone, onError }: UseChatStreamOptions) {
  const abortRef = useRef<AbortController | null>(null);
  const bufferRef = useRef('');
  const rafRef = useRef<number | null>(null);

  const flushBuffer = useCallback(() => {
    if (bufferRef.current) {
      onChunk(bufferRef.current);
      bufferRef.current = '';
    }
    rafRef.current = null;
  }, [onChunk]);

  const send = useCallback(
    async (body: { question: string; context: string; history: { role: string; content: string }[] }) => {
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      try {
        const response = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: abortRef.current.signal,
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
                bufferRef.current += data.content;
                if (!rafRef.current) {
                  rafRef.current = requestAnimationFrame(flushBuffer);
                }
              } else if (data.type === 'done') {
                flushBuffer();
                onDone();
              } else if (data.type === 'error') {
                flushBuffer();
                onError(data.message);
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }

        // Final flush
        flushBuffer();
      } catch (error) {
        if ((error as Error).name === 'AbortError') return;
        flushBuffer();
        onError((error as Error).message || 'Connection failed');
      }
    },
    [flushBuffer, onDone, onError]
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    bufferRef.current = '';
  }, []);

  return { send, cancel };
}
