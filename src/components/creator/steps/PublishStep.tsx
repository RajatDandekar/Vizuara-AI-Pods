'use client';

import { useState, useCallback } from 'react';
import { useCreator } from '@/context/CreatorContext';
import StepHeader from '../shared/StepHeader';
import FadeIn from '@/components/animations/FadeIn';
import type { FigureState, NotebookState } from '@/types/creator';

interface ProgressEvent {
  step: string;
  progress: number;
}

export default function PublishStep() {
  const { state, refreshProject } = useCreator();
  const project = state.activeProject;
  const [publishing, setPublishing] = useState(false);
  const [progress, setProgress] = useState<ProgressEvent[]>([]);
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);

  const handlePublish = useCallback(async () => {
    if (!project) return;
    setPublishing(true);
    setProgress([]);
    setError('');
    setDone(false);

    try {
      const res = await fetch('/api/admin/creator/publish', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: project.id }),
      });

      if (!res.body) return;

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let lineBuffer = '';

      while (true) {
        const { done: streamDone, value } = await reader.read();
        if (streamDone) break;

        lineBuffer += decoder.decode(value, { stream: true });
        const lines = lineBuffer.split('\n');
        lineBuffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(trimmed.slice(6));
            if (event.type === 'progress') {
              setProgress((prev) => [...prev, { step: event.message, progress: event.progress }]);
            } else if (event.type === 'done') {
              setDone(true);
              await refreshProject(project.id);
            } else if (event.type === 'error') {
              setError(event.message);
            }
          } catch {
            // skip
          }
        }
      }
    } finally {
      setPublishing(false);
    }
  }, [project, refreshProject]);

  if (!project) return null;

  // Pre-publish checklist
  const checks = [
    {
      label: 'Article draft approved',
      ok: !!project.articleApproved,
    },
    {
      label: 'All figures approved',
      ok:
        project.figures.length > 0 &&
        project.figures.every((f: FigureState) => f.status === 'approved' || f.status === 'ready'),
    },
    {
      label: 'Notebooks generated',
      ok: project.notebooks.length > 0,
    },
    {
      label: 'Case study generated',
      ok: !!project.caseStudy?.content,
    },
  ];

  const allReady = checks.every((c) => c.ok);

  return (
    <div className="p-8">
      <StepHeader
        title="Publish"
        description="Deploy the pod to the Vizuara platform"
        stepNumber={8}
      />

      <FadeIn>
        {/* Checklist */}
        <div className="border border-card-border rounded-xl p-6 bg-card-bg mb-6">
          <h3 className="text-base font-semibold text-foreground mb-4">
            Pre-publish Checklist
          </h3>
          <div className="space-y-3">
            {checks.map((check) => (
              <div key={check.label} className="flex items-center gap-3">
                <span
                  className={`w-5 h-5 rounded-full flex items-center justify-center text-white text-xs
                    ${check.ok ? 'bg-accent-green' : 'bg-card-border'}`}
                >
                  {check.ok ? (
                    <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                      <path d="M2 6L5 9L10 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                  ) : null}
                </span>
                <span
                  className={`text-base ${check.ok ? 'text-foreground' : 'text-text-muted'}`}
                >
                  {check.label}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Publish summary */}
        <div className="border border-card-border rounded-xl p-6 bg-card-bg mb-6">
          <h3 className="text-base font-semibold text-foreground mb-3">
            Publish Details
          </h3>
          <div className="grid grid-cols-2 gap-3 text-base">
            <div>
              <span className="text-text-muted">Course:</span>{' '}
              <span className="text-foreground font-medium">{project.courseSlug}</span>
            </div>
            <div>
              <span className="text-text-muted">Pod slug:</span>{' '}
              <span className="text-foreground font-mono">{project.podSlug}</span>
            </div>
            <div>
              <span className="text-text-muted">Figures:</span>{' '}
              <span className="text-foreground">{project.figures.length}</span>
            </div>
            <div>
              <span className="text-text-muted">Notebooks:</span>{' '}
              <span className="text-foreground">{project.notebooks.length}</span>
            </div>
          </div>
        </div>

        {/* Progress */}
        {progress.length > 0 && (
          <div className="border border-card-border rounded-xl p-6 bg-card-bg mb-6">
            <h3 className="text-base font-semibold text-foreground mb-3">
              Progress
            </h3>
            <div className="space-y-2">
              {progress.map((p, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="w-5 h-5 rounded-full bg-accent-green flex items-center justify-center">
                    <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                      <path d="M2 6L5 9L10 3" stroke="white" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                  </span>
                  <span className="text-base text-foreground">{p.step}</span>
                </div>
              ))}
            </div>

            {progress.length > 0 && !done && (
              <div className="mt-3 h-2 bg-background rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent-blue rounded-full transition-all duration-500"
                  style={{
                    width: `${progress[progress.length - 1]?.progress || 0}%`,
                  }}
                />
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="p-4 bg-accent-red/10 text-accent-red rounded-xl mb-6">
            <p className="font-medium">Error</p>
            <p className="text-base mt-1">{error}</p>
          </div>
        )}

        {/* Done */}
        {done && (
          <div className="p-6 bg-accent-green-light rounded-xl mb-6 text-center">
            <p className="text-accent-green font-bold text-lg">
              Pod Published Successfully!
            </p>
            <p className="text-accent-green text-base mt-1">
              Your pod is now live at /courses/{project.courseSlug}/{project.podSlug}
            </p>
          </div>
        )}

        {/* Publish button */}
        {!done && (
          <button
            onClick={handlePublish}
            disabled={!allReady || publishing}
            className="w-full px-6 py-3 bg-accent-blue text-white rounded-xl font-medium text-lg
                       hover:bg-accent-blue/90 transition-colors disabled:opacity-50
                       disabled:cursor-not-allowed"
          >
            {publishing
              ? 'Publishing...'
              : project.publishedAt
                ? 'Republish Pod'
                : 'Publish Pod'}
          </button>
        )}

        {!allReady && !publishing && (
          <p className="text-sm text-text-muted mt-2 text-center">
            Complete all checklist items before publishing
          </p>
        )}
      </FadeIn>
    </div>
  );
}
