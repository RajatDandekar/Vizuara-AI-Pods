'use client';

import { useState, useCallback, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';
import StepHeader from '../shared/StepHeader';
import FadeIn from '@/components/animations/FadeIn';
import type { NotebookState, NarrationSegment } from '@/types/creator';

/** Per-notebook narration pipeline status */
interface NotebookNarrationStatus {
  hasScript: boolean;
  hasAllAudio: boolean;
  injected: boolean;
  reUploaded: boolean;
  segmentCount: number;
  audioCount: number;
}

function getNotebookNarrationStatus(
  notebookId: string,
  segments: NarrationSegment[],
  injectedSet: Set<string>,
  reUploadedSet: Set<string>
): NotebookNarrationStatus {
  const nbSegments = segments.filter((s) => s.notebookId === notebookId);
  const hasScript = nbSegments.length > 0;
  const audioCount = nbSegments.filter((s) => s.audioStoragePath).length;
  const hasAllAudio = hasScript && audioCount === nbSegments.length;
  return {
    hasScript,
    hasAllAudio,
    injected: injectedSet.has(notebookId),
    reUploaded: reUploadedSet.has(notebookId),
    segmentCount: nbSegments.length,
    audioCount,
  };
}

function getPipelineStep(status: NotebookNarrationStatus): number {
  if (status.reUploaded) return 4;
  if (status.injected) return 3;
  if (status.hasAllAudio) return 2;
  if (status.hasScript) return 1;
  return 0;
}

const PIPELINE_LABELS = ['Script', 'Audio', 'Inject', 'Upload'];

export default function NarrationStep() {
  const router = useRouter();
  const { state, dispatch, refreshProject } = useCreator();
  const project = state.activeProject;

  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [scriptLoading, setScriptLoading] = useState<string | null>(null);
  const [synthesizing, setSynthesizing] = useState<string | null>(null);
  const [synthProgress, setSynthProgress] = useState('');
  const [injecting, setInjecting] = useState<string | null>(null);
  const [reUploading, setReUploading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Track per-notebook post-API states (inject and re-upload aren't stored in project.json)
  const [injectedSet, setInjectedSet] = useState<Set<string>>(new Set());
  const [reUploadedSet, setReUploadedSet] = useState<Set<string>>(new Set());

  // Auto-expand first notebook
  useEffect(() => {
    if (!expandedId && project?.notebooks.length) {
      setExpandedId(project.notebooks[0].id);
    }
  }, [project?.notebooks.length, expandedId]);

  if (!project) return null;

  const narration = project.narration;
  const segments = narration?.segments || [];
  const notebooks = project.notebooks;

  // --- Handlers ---

  const handleGenerateScript = async (notebookId: string) => {
    setError(null);
    setScriptLoading(notebookId);
    try {
      const res = await fetch('/api/admin/creator/narration/script', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: project.id, notebookId }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to generate script');
      }
      await refreshProject(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Script generation failed');
    } finally {
      setScriptLoading(null);
    }
  };

  const handleSynthesize = async (notebookId: string) => {
    setError(null);
    setSynthesizing(notebookId);
    setSynthProgress('Starting...');

    try {
      const res = await fetch('/api/admin/creator/narration/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: project.id, notebookId }),
      });

      if (!res.body) throw new Error('No response body');

      const reader = res.body.getReader();
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
            if (event.type === 'progress') {
              setSynthProgress(event.step);
            } else if (event.type === 'done') {
              setSynthProgress('');
              await refreshProject(project.id);
            } else if (event.type === 'error') {
              throw new Error(event.message);
            }
          } catch (e) {
            if (e instanceof Error && e.message !== 'Unexpected end of JSON input') {
              setError(e.message);
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Synthesis failed');
    } finally {
      setSynthesizing(null);
      setSynthProgress('');
    }
  };

  const handleInject = async (notebookId: string) => {
    setError(null);
    setInjecting(notebookId);
    try {
      const res = await fetch('/api/admin/creator/narration/inject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: project.id, notebookId }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Injection failed');
      }
      setInjectedSet((prev) => new Set([...prev, notebookId]));
      await refreshProject(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Injection failed');
    } finally {
      setInjecting(null);
    }
  };

  const handleReUpload = async (notebookId: string) => {
    setError(null);
    setReUploading(notebookId);
    try {
      const res = await fetch('/api/admin/creator/notebooks/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: project.id, conceptId: notebookId }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Upload failed');
      }
      setReUploadedSet((prev) => new Set([...prev, notebookId]));
      await refreshProject(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setReUploading(null);
    }
  };

  const handleProceed = async () => {
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ currentStep: 'publish' }),
    });
    dispatch({
      type: 'UPDATE_PROJECT',
      project: { ...project, currentStep: 'publish' },
    });
    router.push(`/admin/creator/${project.id}/publish`);
  };

  // Derived state
  const notebookStatuses = useMemo(() => {
    return notebooks.map((nb: NotebookState) => ({
      id: nb.id,
      ...getNotebookNarrationStatus(nb.id, segments, injectedSet, reUploadedSet),
    }));
  }, [notebooks, segments, injectedSet, reUploadedSet]);

  const completedCount = notebookStatuses.filter((s) => s.reUploaded).length;
  const totalCount = notebooks.length;
  const allDone = totalCount > 0 && completedCount === totalCount;

  return (
    <div className="p-8">
      <StepHeader
        title="Narration"
        description="Generate voice narration and inject audio players into each notebook"
        stepNumber={7}
      />

      {/* Error banner */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-base">
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-3 text-red-400/60 hover:text-red-400"
          >
            Dismiss
          </button>
        </div>
      )}

      <FadeIn>
        {/* Overall progress */}
        {totalCount > 0 && (
          <div className="border border-card-border rounded-xl p-5 bg-card-bg mb-6">
            <div className="flex items-center justify-between mb-3">
              <span className="font-medium text-foreground">
                {allDone
                  ? 'All notebooks narrated and uploaded'
                  : `${completedCount} of ${totalCount} notebooks complete`}
              </span>
            </div>
            <div className="h-2.5 bg-background rounded-full overflow-hidden">
              <div
                className="h-full bg-accent-green rounded-full transition-all duration-700"
                style={{ width: `${totalCount > 0 ? (completedCount / totalCount) * 100 : 0}%` }}
              />
            </div>
          </div>
        )}

        {/* No notebooks */}
        {totalCount === 0 && (
          <div className="text-center py-16 border border-dashed border-card-border rounded-xl">
            <p className="text-text-muted">
              No notebooks found. Complete the Notebooks step first.
            </p>
          </div>
        )}

        {/* Notebook cards */}
        <div className="space-y-3">
          {notebooks.map((nb: NotebookState, idx: number) => {
            const nStatus = notebookStatuses[idx];
            const pipelineStep = getPipelineStep(nStatus);
            const isExpanded = expandedId === nb.id;
            const nbSegments = segments.filter((s: NarrationSegment) => s.notebookId === nb.id);

            const isActive =
              scriptLoading === nb.id ||
              synthesizing === nb.id ||
              injecting === nb.id ||
              reUploading === nb.id;

            return (
              <div
                key={nb.id}
                className={`border rounded-xl overflow-hidden transition-colors ${
                  isActive
                    ? 'border-accent-blue/50 bg-accent-blue/5'
                    : nStatus.reUploaded
                      ? 'border-accent-green/30 bg-card-bg'
                      : 'border-card-border bg-card-bg'
                }`}
              >
                {/* Card header */}
                <div
                  className="flex items-center gap-4 p-5 cursor-pointer hover:bg-background/50 transition-colors"
                  onClick={() => setExpandedId(isExpanded ? null : nb.id)}
                >
                  {/* Number badge */}
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white flex-shrink-0 ${
                      nStatus.reUploaded ? 'bg-accent-green' : pipelineStep > 0 ? 'bg-accent-blue' : 'bg-card-border'
                    }`}
                  >
                    {nStatus.reUploaded ? (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      idx + 1
                    )}
                  </div>

                  {/* Title */}
                  <div className="flex-1 min-w-0">
                    <h4 className="font-semibold text-foreground truncate">{nb.title}</h4>
                    <p className="text-base text-text-muted mt-0.5">
                      {nStatus.reUploaded
                        ? 'Narration injected and uploaded'
                        : nStatus.injected
                          ? 'Injected — needs re-upload to Drive'
                          : nStatus.hasAllAudio
                            ? `${nStatus.audioCount} audio segments ready`
                            : nStatus.hasScript
                              ? `${nStatus.segmentCount} script segments`
                              : 'No narration yet'}
                    </p>
                  </div>

                  {/* Mini pipeline dots */}
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    {PIPELINE_LABELS.map((label, i) => (
                      <div
                        key={label}
                        className={`w-2.5 h-2.5 rounded-full ${
                          i < pipelineStep ? 'bg-accent-green' : 'bg-card-border'
                        }`}
                        title={label}
                      />
                    ))}
                  </div>

                  {/* Colab link if uploaded */}
                  {nb.colabUrl && nStatus.reUploaded && (
                    <a
                      href={nb.colabUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="text-base px-3 py-1.5 text-accent-green border border-accent-green/30 rounded-lg
                                 hover:bg-accent-green/10 transition-colors flex-shrink-0"
                    >
                      Colab
                    </a>
                  )}

                  {/* Expand chevron */}
                  <svg
                    width="20" height="20" viewBox="0 0 16 16" fill="none"
                    className={`text-text-muted transition-transform flex-shrink-0 ${isExpanded ? 'rotate-180' : ''}`}
                  >
                    <path d="M4 6L8 10L12 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                </div>

                {/* Expanded content */}
                {isExpanded && (
                  <div className="border-t border-card-border">
                    {/* Pipeline steps */}
                    <div className="px-5 py-4 bg-background/50">
                      <div className="flex items-center gap-2 mb-4">
                        {PIPELINE_LABELS.map((label, i) => (
                          <div key={label} className="flex items-center gap-2">
                            <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-base font-medium ${
                              i < pipelineStep
                                ? 'bg-accent-green/10 text-accent-green'
                                : i === pipelineStep
                                  ? 'bg-accent-blue/10 text-accent-blue'
                                  : 'bg-card-bg text-text-muted'
                            }`}>
                              {i < pipelineStep ? (
                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                              ) : (
                                <span className="w-3.5 text-center">{i + 1}</span>
                              )}
                              {label}
                            </div>
                            {i < PIPELINE_LABELS.length - 1 && (
                              <div className={`w-6 h-0.5 ${i < pipelineStep ? 'bg-accent-green' : 'bg-card-border'}`} />
                            )}
                          </div>
                        ))}
                      </div>

                      {/* Action buttons */}
                      <div className="flex gap-3">
                        {/* Step 1: Generate Script */}
                        <button
                          onClick={(e) => { e.stopPropagation(); handleGenerateScript(nb.id); }}
                          disabled={!!scriptLoading}
                          className={`px-4 py-2.5 rounded-lg text-base font-medium transition-colors disabled:opacity-50 ${
                            pipelineStep === 0
                              ? 'bg-accent-blue text-white hover:bg-accent-blue/90'
                              : 'border border-card-border text-text-secondary hover:bg-background'
                          }`}
                        >
                          {scriptLoading === nb.id ? (
                            <span className="flex items-center gap-2">
                              <Spinner /> Generating Script...
                            </span>
                          ) : nStatus.hasScript ? 'Regenerate Script' : 'Generate Script'}
                        </button>

                        {/* Step 2: Synthesize Audio */}
                        {nStatus.hasScript && (
                          <button
                            onClick={(e) => { e.stopPropagation(); handleSynthesize(nb.id); }}
                            disabled={!!synthesizing}
                            className={`px-4 py-2.5 rounded-lg text-base font-medium transition-colors disabled:opacity-50 ${
                              pipelineStep === 1
                                ? 'bg-accent-blue text-white hover:bg-accent-blue/90'
                                : 'border border-card-border text-text-secondary hover:bg-background'
                            }`}
                          >
                            {synthesizing === nb.id ? (
                              <span className="flex items-center gap-2">
                                <Spinner /> {synthProgress || 'Synthesizing...'}
                              </span>
                            ) : nStatus.hasAllAudio ? 'Re-synthesize' : 'Synthesize Audio'}
                          </button>
                        )}

                        {/* Step 3: Inject */}
                        {nStatus.hasAllAudio && (
                          <button
                            onClick={(e) => { e.stopPropagation(); handleInject(nb.id); }}
                            disabled={!!injecting}
                            className={`px-4 py-2.5 rounded-lg text-base font-medium transition-colors disabled:opacity-50 ${
                              pipelineStep === 2
                                ? 'bg-accent-blue text-white hover:bg-accent-blue/90'
                                : 'border border-card-border text-text-secondary hover:bg-background'
                            }`}
                          >
                            {injecting === nb.id ? (
                              <span className="flex items-center gap-2">
                                <Spinner /> Injecting...
                              </span>
                            ) : nStatus.injected ? 'Re-inject' : 'Inject into Notebook'}
                          </button>
                        )}

                        {/* Step 4: Re-upload to Drive */}
                        {nStatus.injected && (
                          <button
                            onClick={(e) => { e.stopPropagation(); handleReUpload(nb.id); }}
                            disabled={!!reUploading}
                            className={`px-4 py-2.5 rounded-lg text-base font-medium transition-colors disabled:opacity-50 ${
                              pipelineStep === 3
                                ? 'bg-accent-green text-white hover:bg-accent-green/90'
                                : 'border border-card-border text-text-secondary hover:bg-background'
                            }`}
                          >
                            {reUploading === nb.id ? (
                              <span className="flex items-center gap-2">
                                <Spinner /> Uploading...
                              </span>
                            ) : nStatus.reUploaded ? 'Re-upload' : 'Upload to Drive'}
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Script segments preview */}
                    {nbSegments.length > 0 && (
                      <div className="border-t border-card-border">
                        <div className="px-5 py-3 flex items-center justify-between bg-card-bg">
                          <span className="text-base font-medium text-foreground">
                            Narration Script — {nbSegments.length} segments
                          </span>
                          <span className="text-base text-text-muted">
                            ~{Math.round(nbSegments.reduce((sum: number, s: NarrationSegment) => sum + (s.duration || 0), 0) / 60)} min total
                          </span>
                        </div>
                        <div className="max-h-[400px] overflow-auto divide-y divide-card-border">
                          {nbSegments.map((seg: NarrationSegment, i: number) => (
                            <div key={seg.id} className="px-5 py-3">
                              <div className="flex items-center justify-between mb-1.5">
                                <span className="text-base font-medium text-accent-blue">
                                  Segment {i + 1} — Cell {seg.cellIndex + 1}
                                </span>
                                <div className="flex items-center gap-3">
                                  {seg.duration && (
                                    <span className="text-base text-text-muted">
                                      ~{seg.duration}s
                                    </span>
                                  )}
                                  {seg.audioStoragePath ? (
                                    <span className="text-base text-accent-green font-medium flex items-center gap-1">
                                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                      </svg>
                                      Audio
                                    </span>
                                  ) : (
                                    <span className="text-base text-text-muted">No audio</span>
                                  )}
                                </div>
                              </div>
                              <p className="text-text-secondary leading-relaxed">{seg.text}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Colab URL after re-upload */}
                    {nStatus.reUploaded && nb.colabUrl && (
                      <div className="border-t border-card-border px-5 py-4 bg-accent-green/5">
                        <div className="flex items-center gap-3">
                          <svg className="w-5 h-5 text-accent-green flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          <div className="flex-1">
                            <p className="font-medium text-foreground">Narration injected and uploaded</p>
                            <a
                              href={nb.colabUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-base text-accent-green hover:underline"
                            >
                              {nb.colabUrl}
                            </a>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Empty state */}
                    {nbSegments.length === 0 && (
                      <div className="px-5 py-8 text-center">
                        <p className="text-text-muted">
                          Click &ldquo;Generate Script&rdquo; to create the narration for this notebook.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Continue button */}
        {totalCount > 0 && (
          <div className="flex items-center justify-between mt-6 pt-4 border-t border-card-border">
            <p className="text-base text-text-muted">
              {allDone
                ? 'All notebooks have narration. Ready to publish.'
                : 'You can continue to publish even without narrating all notebooks.'}
            </p>
            <button
              onClick={handleProceed}
              className="px-6 py-3 bg-accent-green text-white rounded-lg font-medium
                         hover:bg-accent-green/90 transition-colors"
            >
              Continue to Publish
            </button>
          </div>
        )}
      </FadeIn>
    </div>
  );
}

function Spinner() {
  return (
    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
  );
}
