'use client';

import { useEffect, useRef, useState } from 'react';
import type { Job } from '@/types/creator';

interface JobProgressProps {
  projectId: string;
  jobId: string;
  onComplete?: (result: Record<string, unknown>) => void;
  onError?: (error: string) => void;
  label?: string;
}

export default function JobProgress({
  projectId,
  jobId,
  onComplete,
  onError,
  label = 'Processing',
}: JobProgressProps) {
  const [job, setJob] = useState<Job | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(
          `/api/admin/creator/jobs/${jobId}?projectId=${projectId}`
        );
        if (!res.ok) return;
        const data = await res.json();
        setJob(data.job);

        if (data.job.status === 'complete') {
          if (intervalRef.current) clearInterval(intervalRef.current);
          onComplete?.(data.job.result || {});
        } else if (data.job.status === 'error') {
          if (intervalRef.current) clearInterval(intervalRef.current);
          onError?.(data.job.error || 'Job failed');
        }
      } catch {
        // Silent polling failure
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 2000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [projectId, jobId, onComplete, onError]);

  const progress = job?.progress ?? 0;
  const message = job?.message || `${label}...`;

  return (
    <div className="p-4 bg-card-bg border border-card-border rounded-xl">
      <div className="flex items-center justify-between mb-2">
        <span className="text-base font-medium text-foreground">{message}</span>
        <span className="text-sm text-text-muted">{progress}%</span>
      </div>
      <div className="h-2 bg-background rounded-full overflow-hidden">
        <div
          className="h-full bg-accent-blue rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
