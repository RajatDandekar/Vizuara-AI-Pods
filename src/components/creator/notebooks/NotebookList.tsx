'use client';

import type { NotebookState } from '@/types/creator';

interface NotebookListProps {
  notebooks: NotebookState[];
  activeId?: string;
  onSelect: (id: string) => void;
  onConvert: (id: string) => void;
  onUpload: (id: string) => void;
}

const statusConfig: Record<
  NotebookState['status'],
  { label: string; color: string }
> = {
  pending: { label: 'Pending', color: 'text-text-muted' },
  generating: { label: 'Generating...', color: 'text-accent-blue' },
  converting: { label: 'Converting...', color: 'text-accent-amber' },
  ready: { label: 'Ready', color: 'text-accent-green' },
  uploaded: { label: 'Uploaded', color: 'text-accent-green' },
  error: { label: 'Error', color: 'text-accent-red' },
};

export default function NotebookList({
  notebooks,
  activeId,
  onSelect,
  onConvert,
  onUpload,
}: NotebookListProps) {
  return (
    <div className="space-y-2">
      {notebooks.map((nb) => {
        const status = statusConfig[nb.status];
        const isActive = nb.id === activeId;

        return (
          <div
            key={nb.id}
            className={`p-3 border rounded-lg cursor-pointer transition-colors
              ${isActive ? 'border-accent-blue bg-accent-blue/5' : 'border-card-border hover:bg-background'}`}
            onClick={() => onSelect(nb.id)}
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-foreground line-clamp-1">
                {nb.title}
              </h4>
              <span className={`text-xs font-medium ${status.color}`}>
                {status.label}
              </span>
            </div>
            <p className="text-xs text-text-muted mt-0.5 line-clamp-1">
              {nb.objective}
            </p>

            {/* Action buttons */}
            <div className="flex gap-2 mt-2">
              {nb.status === 'ready' && nb.markdownContent && !nb.ipynbStoragePath && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onConvert(nb.id);
                  }}
                  className="text-xs px-2 py-1 bg-accent-amber text-white rounded hover:bg-accent-amber/90"
                >
                  Convert to .ipynb
                </button>
              )}
              {nb.ipynbStoragePath && nb.status !== 'uploaded' && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onUpload(nb.id);
                  }}
                  className="text-xs px-2 py-1 bg-accent-blue text-white rounded hover:bg-accent-blue/90"
                >
                  Upload to Drive
                </button>
              )}
              {nb.colabUrl && (
                <a
                  href={nb.colabUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="text-xs px-2 py-1 text-accent-blue hover:underline"
                >
                  Open in Colab
                </a>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
