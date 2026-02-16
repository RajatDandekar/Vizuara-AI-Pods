'use client';

import { useEffect, useRef } from 'react';
import StreamingText from '@/components/ui/StreamingText';
import Button from '@/components/ui/Button';
import { useStreamingResponse } from '@/hooks/useStreamingResponse';
import type { PaperState } from '@/types/paper';

interface PaperArchitectureProps {
  paper: PaperState;
  concept: string;
}

export default function PaperArchitecture({
  paper,
  concept,
}: PaperArchitectureProps) {
  const { startStreaming } = useStreamingResponse({
    endpoint: '/api/architecture',
    paperId: paper.id,
    section: 'architecture',
  });

  const hasStartedRef = useRef(false);

  useEffect(() => {
    if (paper.architecture.status === 'idle' && !hasStartedRef.current) {
      hasStartedRef.current = true;
      startStreaming({
        concept,
        paper: {
          title: paper.title,
          authors: paper.authors,
          year: paper.year,
          venue: paper.venue,
        },
      });
    }
  }, [paper.architecture.status, concept, paper, startStreaming]);

  return (
    <div>
      <StreamingText
        content={paper.architecture.content}
        isStreaming={paper.architecture.status === 'streaming'}
      />

      {paper.architecture.status === 'error' && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-xl">
          <p className="text-sm text-accent-red mb-2">
            Failed to generate architecture analysis: {paper.architecture.error}
          </p>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              hasStartedRef.current = false;
              startStreaming({
                concept,
                paper: {
                  title: paper.title,
                  authors: paper.authors,
                  year: paper.year,
                  venue: paper.venue,
                },
              });
            }}
          >
            Retry
          </Button>
        </div>
      )}
    </div>
  );
}
