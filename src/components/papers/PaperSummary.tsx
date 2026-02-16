'use client';

import { useEffect, useRef } from 'react';
import StreamingText from '@/components/ui/StreamingText';
import Button from '@/components/ui/Button';
import { useStreamingResponse } from '@/hooks/useStreamingResponse';
import type { PaperState } from '@/types/paper';

interface PaperSummaryProps {
  paper: PaperState;
  concept: string;
}

export default function PaperSummary({
  paper,
  concept,
}: PaperSummaryProps) {
  const { startStreaming } = useStreamingResponse({
    endpoint: '/api/summary',
    paperId: paper.id,
    section: 'summary',
  });

  const hasStartedRef = useRef(false);

  // Auto-start streaming when component mounts
  useEffect(() => {
    if (paper.summary.status === 'idle' && !hasStartedRef.current) {
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
  }, [paper.summary.status, concept, paper, startStreaming]);

  return (
    <div>
      <StreamingText
        content={paper.summary.content}
        isStreaming={paper.summary.status === 'streaming'}
      />

      {paper.summary.status === 'error' && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-xl">
          <p className="text-sm text-accent-red mb-2">
            Failed to generate summary: {paper.summary.error}
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
