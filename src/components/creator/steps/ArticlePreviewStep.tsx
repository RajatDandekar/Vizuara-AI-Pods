'use client';

import { useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';
import StepHeader from '../shared/StepHeader';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';
import FadeIn from '@/components/animations/FadeIn';
import type { FigureState } from '@/types/creator';

export default function ArticlePreviewStep() {
  const router = useRouter();
  const { state, dispatch } = useCreator();
  const project = state.activeProject;

  const processedContent = useMemo(() => {
    if (!project?.articleDraft) return '';

    let content = project.articleDraft;

    // Replace figure placeholders with actual images
    let figureIndex = 0;
    content = content.replace(
      /\{\{FIGURE:\s*(.+?)\s*(?:\|\|\s*(.+?))?\s*\}\}/g,
      (_match, _desc, caption) => {
        const figure = project.figures[figureIndex];
        figureIndex++;

        if (!figure) return `*[Figure missing]*`;

        if (figure.storagePath) {
          const src = `/api/admin/creator/figures/serve?projectId=${project.id}&figureId=${figure.id}`;
          return `![${caption || figure.caption}](${src})`;
        }
        return `*[${figure.caption} — not yet generated]*`;
      }
    );

    return content;
  }, [project]);

  if (!project) return null;

  const handleProceed = async () => {
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ currentStep: 'export' }),
    });
    dispatch({
      type: 'UPDATE_PROJECT',
      project: { ...project, currentStep: 'export' },
    });
    router.push(`/admin/creator/${project.id}/export`);
  };

  return (
    <div className="p-8">
      <StepHeader
        title="Article Preview"
        description="Review the full article with figures before exporting"
        stepNumber={3}
      />

      <FadeIn>
        <div className="border border-card-border rounded-xl p-8 bg-card-bg mb-6">
          <MarkdownRenderer content={processedContent} />
        </div>

        <div className="flex items-center justify-end gap-3">
          <button
            onClick={handleProceed}
            className="px-5 py-2.5 bg-accent-blue text-white rounded-lg font-medium
                       hover:bg-accent-blue/90 transition-colors"
          >
            Continue to Export
          </button>
        </div>
      </FadeIn>
    </div>
  );
}
