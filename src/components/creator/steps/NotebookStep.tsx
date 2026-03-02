'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';
import { useCreatorStream } from '@/hooks/useCreatorStream';
import StepHeader from '../shared/StepHeader';
import ConceptPlanReview from '../notebooks/ConceptPlanReview';
import NotebookPreview from '../notebooks/NotebookPreview';
import FadeIn from '@/components/animations/FadeIn';
import type { ConceptPlan, NotebookState } from '@/types/creator';

type NotebookPhase = 'plan' | 'plan-review' | 'generation';

function derivePhase(project: { notebooks: NotebookState[]; conceptPlan?: ConceptPlan } | null): NotebookPhase {
  if (!project) return 'plan';
  if (project.conceptPlan) {
    if (project.notebooks.length > 0) return 'generation';
    return 'plan-review';
  }
  return 'plan';
}

export default function NotebookStep() {
  const router = useRouter();
  const { state, dispatch, refreshProject } = useCreator();
  const project = state.activeProject;
  const [phase, setPhase] = useState<NotebookPhase>(() => derivePhase(project));
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [generatingId, setGeneratingId] = useState<string | null>(null);
  const [autoGenerateAll, setAutoGenerateAll] = useState(false);
  const [resetting, setResetting] = useState(false);

  // Keep phase in sync with project state
  useEffect(() => {
    if (generatingId) return;
    setPhase(derivePhase(project));
  }, [project?.notebooks.length, project?.conceptPlan, generatingId]);

  const notebookStream = useCreatorStream({
    onComplete: useCallback(() => {
      if (project) {
        refreshProject(project.id);
        setGeneratingId(null);
      }
    }, [project, refreshProject]),
  });

  // Auto-chain: generate next notebook when current finishes
  useEffect(() => {
    if (!autoGenerateAll || generatingId || !project?.conceptPlan) return;
    const remaining = project.conceptPlan.concepts.filter(
      (c) => !project.notebooks.find((n: NotebookState) => n.id === c.id)
    );
    if (remaining.length > 0) {
      const timer = setTimeout(() => generateNotebook(remaining[0]), 500);
      return () => clearTimeout(timer);
    } else {
      setAutoGenerateAll(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoGenerateAll, generatingId, project?.notebooks.length]);

  if (!project) return null;

  const handleGeneratePlan = async () => {
    setPlanLoading(true);
    try {
      const res = await fetch('/api/admin/creator/notebooks/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: project.id }),
      });
      if (res.ok) {
        const data = await res.json();
        dispatch({
          type: 'UPDATE_PROJECT',
          project: { ...project, conceptPlan: data.conceptPlan },
        });
        setPhase('plan-review');
      }
    } finally {
      setPlanLoading(false);
    }
  };

  const handleApprovePlan = async (plan: ConceptPlan) => {
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conceptPlan: plan, conceptPlanApproved: true }),
    });
    dispatch({
      type: 'UPDATE_PROJECT',
      project: { ...project, conceptPlan: plan, conceptPlanApproved: true },
    });
    setPhase('generation');
    // Start generating all notebooks
    setAutoGenerateAll(true);
    if (plan.concepts.length > 0) {
      generateNotebook(plan.concepts[0]);
    }
  };

  const generateNotebook = (concept: { id: string; title: string; objective: string; topics: string[] }) => {
    setGeneratingId(concept.id);
    setExpandedId(concept.id);
    dispatch({ type: 'STREAM_RESET' });
    notebookStream.startStream('/api/admin/creator/notebooks/generate', {
      projectId: project.id,
      conceptId: concept.id,
      title: concept.title,
      objective: concept.objective,
      topics: concept.topics,
    });
  };

  const handleConvert = async (conceptId: string) => {
    const notebooks = project.notebooks.map((n: NotebookState) =>
      n.id === conceptId ? { ...n, status: 'converting' as const } : n
    );
    dispatch({ type: 'UPDATE_PROJECT', project: { ...project, notebooks } });
    await fetch('/api/admin/creator/notebooks/convert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ projectId: project.id, conceptId }),
    });
    setTimeout(() => refreshProject(project.id), 3000);
    setTimeout(() => refreshProject(project.id), 8000);
  };

  const handleConvertAll = async () => {
    const toConvert = project.notebooks.filter(
      (n: NotebookState) => n.status === 'ready' && n.markdownContent && !n.ipynbStoragePath
    );
    for (const nb of toConvert) {
      await handleConvert(nb.id);
    }
  };

  const handleUpload = async (conceptId: string) => {
    const res = await fetch('/api/admin/creator/notebooks/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ projectId: project.id, conceptId }),
    });
    if (res.ok) await refreshProject(project.id);
  };

  const handleUploadAll = async () => {
    const toUpload = project.notebooks.filter(
      (n: NotebookState) => n.ipynbStoragePath && n.status !== 'uploaded'
    );
    for (const nb of toUpload) {
      await handleUpload(nb.id);
    }
  };

  const handleReset = async () => {
    setResetting(true);
    try {
      await fetch(`/api/admin/creator/projects/${project.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          notebooks: [],
          conceptPlan: undefined,
          conceptPlanApproved: false,
        }),
      });
      dispatch({
        type: 'UPDATE_PROJECT',
        project: { ...project, notebooks: [], conceptPlan: undefined, conceptPlanApproved: false },
      });
      setExpandedId(null);
      setGeneratingId(null);
      setAutoGenerateAll(false);
      setPhase('plan');
    } finally {
      setResetting(false);
    }
  };

  const handleProceed = async () => {
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ currentStep: 'case-study' }),
    });
    dispatch({
      type: 'UPDATE_PROJECT',
      project: { ...project, currentStep: 'case-study' },
    });
    router.push(`/admin/creator/${project.id}/case-study`);
  };

  // Derived state
  const allConcepts = project.conceptPlan?.concepts || [];
  const totalConcepts = allConcepts.length;
  const generatedCount = project.notebooks.length;
  const readyCount = project.notebooks.filter(
    (n: NotebookState) => n.status === 'ready' || n.status === 'uploaded'
  ).length;
  const convertedCount = project.notebooks.filter(
    (n: NotebookState) => n.ipynbStoragePath
  ).length;
  const uploadedCount = project.notebooks.filter(
    (n: NotebookState) => n.status === 'uploaded'
  ).length;
  const allReady = totalConcepts > 0 && readyCount === totalConcepts;
  const needsConvert = project.notebooks.some(
    (n: NotebookState) => n.status === 'ready' && n.markdownContent && !n.ipynbStoragePath
  );
  const needsUpload = project.notebooks.some(
    (n: NotebookState) => n.ipynbStoragePath && n.status !== 'uploaded'
  );

  return (
    <div className="p-8">
      <div className="flex items-start justify-between">
        <StepHeader
          title="Notebooks"
          description="Split the article into concepts and generate Colab notebooks"
          stepNumber={5}
        />
        {phase !== 'plan' && (
          <button
            onClick={handleReset}
            disabled={resetting}
            className="px-4 py-2 text-sm border border-card-border rounded-lg
                       text-text-muted hover:text-accent-red hover:border-accent-red/50
                       transition-colors disabled:opacity-50"
          >
            {resetting ? 'Resetting...' : 'Reset Notebooks'}
          </button>
        )}
      </div>

      <FadeIn>
        {/* Phase: Generate Concept Plan */}
        {phase === 'plan' && (
          <div className="text-center py-16 border border-dashed border-card-border rounded-xl">
            <p className="text-text-muted text-base mb-4">
              Analyze the article structure and create a concept plan for notebooks
            </p>
            <button
              onClick={handleGeneratePlan}
              disabled={planLoading}
              className="px-6 py-3 bg-accent-blue text-white rounded-lg font-medium
                         hover:bg-accent-blue/90 transition-colors disabled:opacity-50"
            >
              {planLoading ? 'Analyzing article...' : 'Generate Concept Plan'}
            </button>
          </div>
        )}

        {/* Phase: Review Concept Plan */}
        {phase === 'plan-review' && project.conceptPlan && (
          <ConceptPlanReview
            plan={project.conceptPlan}
            onApprove={handleApprovePlan}
            onRegenerate={handleGeneratePlan}
          />
        )}

        {/* Phase: Generation & Review */}
        {phase === 'generation' && (
          <div className="space-y-4">
            {/* Progress bar */}
            <div className="border border-card-border rounded-xl p-4 bg-card-bg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-foreground">
                  {generatingId
                    ? `Generating notebook ${generatedCount + 1} of ${totalConcepts}...`
                    : allReady
                      ? 'All notebooks generated'
                      : `${generatedCount} of ${totalConcepts} notebooks generated`}
                </span>
                <div className="flex items-center gap-2 text-base text-text-muted">
                  {convertedCount > 0 && (
                    <span>{convertedCount} converted</span>
                  )}
                  {uploadedCount > 0 && (
                    <span>{uploadedCount} uploaded</span>
                  )}
                </div>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent-blue rounded-full transition-all duration-700"
                  style={{ width: `${totalConcepts > 0 ? (generatedCount / totalConcepts) * 100 : 0}%` }}
                />
              </div>

              {/* Batch action buttons */}
              <div className="flex gap-2 mt-3">
                {!allReady && !autoGenerateAll && !generatingId && generatedCount < totalConcepts && (
                  <button
                    onClick={() => {
                      setAutoGenerateAll(true);
                      const remaining = allConcepts.filter(
                        (c) => !project.notebooks.find((n: NotebookState) => n.id === c.id)
                      );
                      if (remaining.length > 0) generateNotebook(remaining[0]);
                    }}
                    className="px-4 py-2 text-base bg-accent-blue text-white rounded-lg
                               hover:bg-accent-blue/90 transition-colors"
                  >
                    Generate Remaining ({totalConcepts - generatedCount})
                  </button>
                )}
                {needsConvert && !generatingId && (
                  <button
                    onClick={handleConvertAll}
                    className="px-4 py-2 text-base bg-accent-amber text-white rounded-lg
                               hover:bg-accent-amber/90 transition-colors"
                  >
                    Convert All to .ipynb
                  </button>
                )}
                {needsUpload && !generatingId && (
                  <button
                    onClick={handleUploadAll}
                    className="px-4 py-2 text-base bg-accent-green text-white rounded-lg
                               hover:bg-accent-green/90 transition-colors"
                  >
                    Upload All to Drive
                  </button>
                )}
              </div>
            </div>

            {/* Notebook cards */}
            <div className="space-y-3">
              {allConcepts.map((concept, idx) => {
                const notebook = project.notebooks.find(
                  (n: NotebookState) => n.id === concept.id
                );
                const isGenerating = generatingId === concept.id;
                const isExpanded = expandedId === concept.id;
                const status = isGenerating
                  ? 'generating'
                  : notebook?.status || 'pending';

                const statusConfig: Record<string, { label: string; color: string; bg: string }> = {
                  pending: { label: 'Pending', color: 'text-text-muted', bg: 'bg-card-border' },
                  generating: { label: 'Generating...', color: 'text-accent-blue', bg: 'bg-accent-blue' },
                  converting: { label: 'Converting...', color: 'text-accent-amber', bg: 'bg-accent-amber' },
                  ready: { label: 'Ready', color: 'text-accent-green', bg: 'bg-accent-green' },
                  uploaded: { label: 'Uploaded', color: 'text-accent-green', bg: 'bg-accent-green' },
                  error: { label: 'Error', color: 'text-accent-red', bg: 'bg-accent-red' },
                };

                const st = statusConfig[status] || statusConfig.pending;

                return (
                  <div
                    key={concept.id}
                    className={`border rounded-xl overflow-hidden transition-colors
                      ${isGenerating ? 'border-accent-blue/50 bg-accent-blue/5' : 'border-card-border bg-card-bg'}`}
                  >
                    {/* Card header */}
                    <div
                      className="flex items-center gap-4 p-5 cursor-pointer hover:bg-background/50 transition-colors"
                      onClick={() => setExpandedId(isExpanded ? null : concept.id)}
                    >
                      {/* Number badge */}
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white flex-shrink-0 ${st.bg}`}
                      >
                        {status === 'generating' ? (
                          <span className="animate-spin">⟳</span>
                        ) : (
                          idx + 1
                        )}
                      </div>

                      {/* Title and objective */}
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-foreground truncate">
                          {concept.title}
                        </h4>
                        <p className="text-base text-text-muted truncate mt-0.5">
                          {concept.objective}
                        </p>
                      </div>

                      {/* Status + actions */}
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className={`text-base font-medium ${st.color}`}>
                          {st.label}
                        </span>

                        {/* Action buttons */}
                        {notebook?.status === 'ready' && notebook.markdownContent && !notebook.ipynbStoragePath && (
                          <button
                            onClick={(e) => { e.stopPropagation(); handleConvert(concept.id); }}
                            className="text-base px-3 py-1.5 bg-accent-amber text-white rounded-lg hover:bg-accent-amber/90"
                          >
                            Convert
                          </button>
                        )}
                        {notebook?.ipynbStoragePath && notebook.status !== 'uploaded' && (
                          <button
                            onClick={(e) => { e.stopPropagation(); handleUpload(concept.id); }}
                            className="text-base px-3 py-1.5 bg-accent-blue text-white rounded-lg hover:bg-accent-blue/90"
                          >
                            Upload
                          </button>
                        )}
                        {notebook?.colabUrl && (
                          <a
                            href={notebook.colabUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-base px-3 py-1.5 text-accent-green border border-accent-green/30 rounded-lg hover:bg-accent-green/10"
                          >
                            Colab
                          </a>
                        )}

                        {/* Expand chevron */}
                        <svg
                          width="16" height="16" viewBox="0 0 16 16" fill="none"
                          className={`text-text-muted transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                        >
                          <path d="M4 6L8 10L12 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                        </svg>
                      </div>
                    </div>

                    {/* Expanded content */}
                    {isExpanded && (
                      <div className="border-t border-card-border">
                        {isGenerating && state.streamingStatus === 'streaming' ? (
                          <div className="p-5 max-h-[500px] overflow-auto">
                            <div className="flex items-center gap-2 mb-3">
                              <div className="w-2 h-2 bg-accent-blue rounded-full animate-pulse" />
                              <span className="text-base text-accent-blue font-medium">Streaming...</span>
                            </div>
                            <pre className="whitespace-pre-wrap text-base text-foreground font-mono leading-relaxed">
                              {state.streamingContent}
                            </pre>
                          </div>
                        ) : notebook?.markdownContent ? (
                          <div className="p-4 max-h-[500px] overflow-auto">
                            <NotebookPreview
                              content={notebook.markdownContent}
                              title={notebook.title}
                            />
                          </div>
                        ) : status === 'pending' ? (
                          <div className="p-6 text-center">
                            <p className="text-base text-text-muted mb-3">Not yet generated</p>
                            {!generatingId && (
                              <button
                                onClick={() => generateNotebook(concept)}
                                className="px-4 py-2 text-base bg-accent-blue text-white rounded-lg
                                           hover:bg-accent-blue/90 transition-colors"
                              >
                                Generate This Notebook
                              </button>
                            )}
                          </div>
                        ) : null}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Continue button */}
            {allReady && (
              <div className="flex justify-end pt-2">
                <button
                  onClick={handleProceed}
                  className="px-6 py-2.5 bg-accent-green text-white rounded-lg font-medium
                             hover:bg-accent-green/90 transition-colors"
                >
                  Continue to Case Study
                </button>
              </div>
            )}
          </div>
        )}
      </FadeIn>
    </div>
  );
}
