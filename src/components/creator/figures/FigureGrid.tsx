'use client';

import type { FigureState } from '@/types/creator';
import FigureCard from './FigureCard';

interface FigureGridProps {
  figures: FigureState[];
  projectId: string;
  onRegenerate: (figureId: string, description: string) => void;
  onApprove: (figureId: string) => void;
  onUpload: (figureId: string, file: File) => void;
}

export default function FigureGrid({
  figures,
  projectId,
  onRegenerate,
  onApprove,
  onUpload,
}: FigureGridProps) {
  if (figures.length === 0) {
    return (
      <div className="text-center py-12 border border-dashed border-card-border rounded-xl">
        <p className="text-text-muted">
          No figures detected. Generate the article draft first — figures are
          auto-extracted from <code>{'{{FIGURE: ...}}'}</code> placeholders.
        </p>
      </div>
    );
  }

  const approvedCount = figures.filter((f) => f.status === 'approved').length;
  const readyCount = figures.filter((f) => f.status === 'ready').length;
  const generatingCount = figures.filter((f) => f.status === 'generating').length;
  const pendingCount = figures.filter((f) => f.status === 'pending').length;

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4 text-base text-text-muted">
          <span>{approvedCount} / {figures.length} approved</span>
          {readyCount > 0 && (
            <span className="text-accent-amber">{readyCount} ready for review</span>
          )}
          {generatingCount > 0 && (
            <span className="text-accent-blue">{generatingCount} generating</span>
          )}
        </div>
        <div className="flex gap-2">
          {pendingCount > 0 && (
            <button
              onClick={() => {
                figures
                  .filter((f) => f.status === 'pending')
                  .forEach((f) => onRegenerate(f.id, f.description));
              }}
              className="text-base px-4 py-2 bg-accent-blue text-white rounded-lg hover:bg-accent-blue/90"
            >
              Generate {pendingCount} Pending
            </button>
          )}
          {readyCount > 0 && (
            <button
              onClick={() => {
                figures
                  .filter((f) => f.status === 'ready')
                  .forEach((f) => onApprove(f.id));
              }}
              className="text-base px-4 py-2 bg-accent-green text-white rounded-lg hover:bg-accent-green/90"
            >
              Approve All Ready
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {figures.map((figure) => (
          <FigureCard
            key={figure.id}
            figure={figure}
            projectId={projectId}
            onRegenerate={onRegenerate}
            onApprove={onApprove}
            onUpload={onUpload}
          />
        ))}
      </div>
    </div>
  );
}
