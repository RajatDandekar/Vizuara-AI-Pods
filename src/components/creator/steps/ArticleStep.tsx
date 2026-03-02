'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';
import { useCreatorStream } from '@/hooks/useCreatorStream';
import StepHeader from '../shared/StepHeader';
import OutlineEditor from '../article/OutlineEditor';
import ArticleEditor from '../article/ArticleEditor';
import FadeIn from '@/components/animations/FadeIn';

type ArticlePhase = 'initial' | 'outline-streaming' | 'outline-review' | 'draft-streaming' | 'draft-review';

export default function ArticleStep() {
  const router = useRouter();
  const { state, dispatch, refreshProject } = useCreator();
  const project = state.activeProject;
  const [phase, setPhase] = useState<ArticlePhase>(() => {
    if (!project) return 'initial';
    if (project.articleApproved) return 'draft-review';
    if (project.articleDraft) return 'draft-review';
    if (project.outlineApproved) return 'draft-streaming';
    if (project.outline) return 'outline-review';
    return 'initial';
  });

  const outlineStream = useCreatorStream({
    onComplete: useCallback(
      (fullText: string) => {
        setPhase('outline-review');
        if (project) {
          dispatch({
            type: 'UPDATE_PROJECT',
            project: { ...project, outline: fullText },
          });
        }
      },
      [project, dispatch]
    ),
  });

  const draftStream = useCreatorStream({
    onComplete: useCallback(
      () => {
        setPhase('draft-review');
        if (project) refreshProject(project.id);
      },
      [project, refreshProject]
    ),
  });

  if (!project) return null;

  const handleGenerateOutline = () => {
    setPhase('outline-streaming');
    dispatch({ type: 'STREAM_RESET' });
    outlineStream.startStream('/api/admin/creator/article/outline', {
      projectId: project.id,
    });
  };

  const handleApproveOutline = async (outline: string) => {
    // Save approved outline
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ outline, outlineApproved: true }),
    });

    dispatch({
      type: 'UPDATE_PROJECT',
      project: { ...project, outline, outlineApproved: true },
    });

    // Start draft generation
    setPhase('draft-streaming');
    dispatch({ type: 'STREAM_RESET' });
    draftStream.startStream('/api/admin/creator/article/generate', {
      projectId: project.id,
      outline,
    });
  };

  const handleApproveDraft = async (content: string) => {
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        articleDraft: content,
        articleApproved: true,
        currentStep: 'figures',
      }),
    });

    dispatch({
      type: 'UPDATE_PROJECT',
      project: {
        ...project,
        articleDraft: content,
        articleApproved: true,
        currentStep: 'figures',
      },
    });
    router.push(`/admin/creator/${project.id}/figures`);
  };

  return (
    <div className="p-8">
      <StepHeader
        title="Article"
        description="Generate an outline, review it, then produce the full article draft"
        stepNumber={1}
      />

      {/* Phase: Initial — show generate button */}
      {phase === 'initial' && (
        <FadeIn>
          <div className="text-center py-16 border border-dashed border-card-border rounded-xl">
            <p className="text-text-muted text-base mb-2">
              Topic: <span className="text-foreground font-medium">{project.concept}</span>
            </p>
            <p className="text-text-muted text-base mb-6">
              Claude will first generate an outline for your review
            </p>
            <button
              onClick={handleGenerateOutline}
              className="px-6 py-3 bg-accent-blue text-white rounded-lg font-medium
                         hover:bg-accent-blue/90 transition-colors"
            >
              Generate Outline
            </button>
          </div>
        </FadeIn>
      )}

      {/* Phase: Outline streaming */}
      {phase === 'outline-streaming' && (
        <FadeIn>
          <div className="border border-card-border rounded-xl p-6 max-h-[600px] overflow-auto streaming-cursor">
            {state.streamingContent ? (
              <div className="markdown-content">
                <pre className="whitespace-pre-wrap text-base text-foreground font-sans">
                  {state.streamingContent}
                </pre>
              </div>
            ) : (
              <div className="flex items-center gap-3 text-text-muted">
                <span className="w-2 h-2 bg-accent-blue rounded-full pulse-dot" />
                Generating outline...
              </div>
            )}
          </div>
        </FadeIn>
      )}

      {/* Phase: Outline review */}
      {phase === 'outline-review' && (project.outline || state.streamingContent) && (
        <FadeIn>
          <OutlineEditor
            outline={project.outline || state.streamingContent}
            onApprove={handleApproveOutline}
            onRegenerate={handleGenerateOutline}
          />
        </FadeIn>
      )}

      {/* Phase: Draft streaming */}
      {phase === 'draft-streaming' && (
        <FadeIn>
          <ArticleEditor
            content={state.streamingContent}
            streaming={state.streamingStatus === 'streaming'}
          onApprove={() => {}}
          />
          {!state.streamingContent && state.streamingStatus !== 'streaming' && (
            <div className="flex items-center gap-3 text-text-muted p-6">
              <span className="w-2 h-2 bg-accent-blue rounded-full pulse-dot" />
              Generating full article draft...
            </div>
          )}
        </FadeIn>
      )}

      {/* Phase: Draft review */}
      {phase === 'draft-review' && project.articleDraft && (
        <FadeIn>
          <ArticleEditor
            content={project.articleDraft}
            onApprove={handleApproveDraft}
            onRegenerate={() => {
              const outline = project.outline || '';
              setPhase('draft-streaming');
              dispatch({ type: 'STREAM_RESET' });
              draftStream.startStream('/api/admin/creator/article/generate', {
                projectId: project.id,
                outline,
              });
            }}
          />
          {project.articleApproved && (
            <div className="mt-4 p-3 bg-accent-green-light text-accent-green rounded-lg text-base font-medium">
              Article approved. Proceed to Figures step.
            </div>
          )}
        </FadeIn>
      )}

      {/* Error state */}
      {state.streamingStatus === 'error' && state.error && (
        <div className="mt-4 p-4 bg-accent-red/10 text-accent-red rounded-xl">
          <p className="font-medium">Error</p>
          <p className="text-base mt-1">{state.error}</p>
          <button
            onClick={handleGenerateOutline}
            className="mt-3 text-base underline hover:no-underline"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
