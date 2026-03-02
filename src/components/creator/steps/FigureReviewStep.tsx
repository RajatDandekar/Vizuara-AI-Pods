'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';
import StepHeader from '../shared/StepHeader';
import FigureGrid from '../figures/FigureGrid';
import FadeIn from '@/components/animations/FadeIn';
import type { FigureState } from '@/types/creator';

export default function FigureReviewStep() {
  const router = useRouter();
  const { state, dispatch, refreshProject } = useCreator();
  const project = state.activeProject;
  const autoTriggeredRef = useRef(false);
  const [autoGenerating, setAutoGenerating] = useState(false);

  const triggerGeneration = useCallback(
    async (figures: FigureState[]) => {
      if (!project) return;
      setAutoGenerating(true);
      for (const fig of figures) {
        await fetch('/api/admin/creator/figures/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            projectId: project.id,
            figureId: fig.id,
            description: fig.description,
          }),
        });
      }
      await refreshProject(project.id);
      setAutoGenerating(false);
    },
    [project, refreshProject]
  );

  // Auto-generate all pending figures when entering this step
  useEffect(() => {
    if (!project || autoTriggeredRef.current) return;

    const pendingFigures = project.figures.filter(
      (f: FigureState) => f.status === 'pending'
    );
    if (pendingFigures.length === 0) return;

    autoTriggeredRef.current = true;
    triggerGeneration(pendingFigures);
  }, [project, triggerGeneration]);

  // Auto-poll while any figure is generating
  useEffect(() => {
    if (!project) return;

    const hasGenerating = project.figures.some(
      (f: FigureState) => f.status === 'generating'
    );
    if (!hasGenerating) return;

    const interval = setInterval(async () => {
      await refreshProject(project.id);
    }, 5000);

    return () => clearInterval(interval);
  }, [project, refreshProject]);

  const handleRetryAll = useCallback(async () => {
    if (!project) return;

    // Reset error figures to pending first
    const errorFigures = project.figures.filter(
      (f: FigureState) => f.status === 'error'
    );
    if (errorFigures.length === 0) return;

    const figures = project.figures.map((f: FigureState) =>
      f.status === 'error' ? { ...f, status: 'pending' as const, error: undefined } : f
    );
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ figures }),
    });

    // Now generate them
    await triggerGeneration(errorFigures);
  }, [project, triggerGeneration]);

  const handleRegenerate = useCallback(
    async (figureId: string, description: string) => {
      if (!project) return;

      const res = await fetch('/api/admin/creator/figures/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: project.id,
          figureId,
          description,
        }),
      });

      if (res.ok) {
        await refreshProject(project.id);
      }
    },
    [project, refreshProject]
  );

  const handleApprove = useCallback(
    async (figureId: string) => {
      if (!project) return;

      const figures = project.figures.map((f: FigureState) =>
        f.id === figureId ? { ...f, status: 'approved' as const } : f
      );

      await fetch(`/api/admin/creator/projects/${project.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ figures }),
      });

      dispatch({
        type: 'UPDATE_PROJECT',
        project: { ...project, figures },
      });
    },
    [project, dispatch]
  );

  const handleUpload = useCallback(
    async (figureId: string, file: File) => {
      if (!project) return;

      const formData = new FormData();
      formData.append('projectId', project.id);
      formData.append('figureId', figureId);
      formData.append('file', file);

      const res = await fetch('/api/admin/creator/figures/upload', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        await refreshProject(project.id);
      }
    },
    [project, refreshProject]
  );

  const handleProceed = useCallback(async () => {
    if (!project) return;
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ currentStep: 'preview' }),
    });
    dispatch({
      type: 'UPDATE_PROJECT',
      project: { ...project, currentStep: 'preview' },
    });
    router.push(`/admin/creator/${project.id}/preview`);
  }, [project, dispatch, router]);

  if (!project) return null;

  const generatingCount = project.figures.filter(
    (f: FigureState) => f.status === 'generating'
  ).length;
  const readyCount = project.figures.filter(
    (f: FigureState) => f.status === 'ready'
  ).length;
  const errorCount = project.figures.filter(
    (f: FigureState) => f.status === 'error'
  ).length;
  const approvedCount = project.figures.filter(
    (f: FigureState) => f.status === 'approved'
  ).length;
  const allApproved =
    project.figures.length > 0 &&
    project.figures.every((f: FigureState) => f.status === 'approved');

  return (
    <div className="p-8">
      <StepHeader
        title="Figures"
        description="Figures are auto-generated via PaperBanana (multi-agent critique pipeline). Review, regenerate, or upload replacements."
        stepNumber={2}
      />

      <FadeIn>
        {/* Auto-generation progress banner */}
        {(autoGenerating || generatingCount > 0) && (
          <div className="mb-4 p-4 bg-accent-blue-light rounded-xl flex items-center gap-3">
            <div className="w-5 h-5 border-2 border-accent-blue border-t-transparent rounded-full animate-spin" />
            <p className="text-accent-blue text-base font-medium">
              Generating figures via PaperBanana... {generatingCount} in progress,{' '}
              {readyCount + approvedCount} / {project.figures.length} complete
            </p>
          </div>
        )}

        {/* Error banner with retry */}
        {errorCount > 0 && generatingCount === 0 && !autoGenerating && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center justify-between">
            <p className="text-red-600 text-base font-medium">
              {errorCount} figure{errorCount > 1 ? 's' : ''} failed to generate
            </p>
            <button
              onClick={handleRetryAll}
              className="px-4 py-2 bg-red-600 text-white text-base rounded-lg font-medium
                         hover:bg-red-700 transition-colors"
            >
              Retry Failed ({errorCount})
            </button>
          </div>
        )}

        <FigureGrid
          figures={project.figures}
          projectId={project.id}
          onRegenerate={handleRegenerate}
          onApprove={handleApprove}
          onUpload={handleUpload}
        />

        {allApproved && (
          <div className="mt-6 flex items-center justify-between p-4 bg-accent-green-light rounded-xl">
            <p className="text-accent-green font-medium text-base">
              All figures approved
            </p>
            <button
              onClick={handleProceed}
              className="px-5 py-2 bg-accent-green text-white rounded-lg font-medium
                         hover:bg-accent-green/90 transition-colors"
            >
              Continue to Preview
            </button>
          </div>
        )}
      </FadeIn>
    </div>
  );
}
