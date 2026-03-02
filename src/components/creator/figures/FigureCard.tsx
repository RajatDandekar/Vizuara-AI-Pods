'use client';

import { useState, useRef } from 'react';
import type { FigureState } from '@/types/creator';
import JobProgress from '../shared/JobProgress';

interface FigureCardProps {
  figure: FigureState;
  projectId: string;
  onRegenerate: (figureId: string, description: string) => void;
  onApprove: (figureId: string) => void;
  onUpload: (figureId: string, file: File) => void;
}

export default function FigureCard({
  figure,
  projectId,
  onRegenerate,
  onApprove,
  onUpload,
}: FigureCardProps) {
  const [editDesc, setEditDesc] = useState(figure.description);
  const [showEdit, setShowEdit] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const statusBadge = {
    pending: { label: 'Pending', color: 'bg-card-border text-text-muted' },
    generating: { label: 'Generating...', color: 'bg-accent-blue-light text-accent-blue' },
    ready: { label: 'Ready', color: 'bg-accent-amber-light text-accent-amber' },
    approved: { label: 'Approved', color: 'bg-accent-green-light text-accent-green' },
    error: { label: 'Error', color: 'bg-accent-red/10 text-accent-red' },
  }[figure.status];

  return (
    <div className="border border-card-border rounded-xl overflow-hidden bg-card-bg">
      {/* Image area */}
      <div className="aspect-video bg-background flex items-center justify-center relative">
        {figure.storagePath ? (
          <img
            src={`/api/admin/creator/figures/serve?projectId=${projectId}&figureId=${figure.id}`}
            alt={figure.caption}
            className="w-full h-full object-contain"
          />
        ) : figure.status === 'generating' && figure.jobId ? (
          <div className="p-4 w-full">
            <JobProgress
              projectId={projectId}
              jobId={figure.jobId}
              label="Generating figure"
            />
          </div>
        ) : (
          <span className="text-text-muted text-base">No image yet</span>
        )}

        {/* Status badge */}
        <span
          className={`absolute top-2 right-2 px-2.5 py-1 rounded-full text-sm font-medium ${statusBadge.color}`}
        >
          {statusBadge.label}
        </span>
      </div>

      {/* Info and actions */}
      <div className="p-4">
        <div className="flex items-center gap-2 mb-1">
          <h4 className="text-base font-semibold text-foreground">
            {figure.id.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
          </h4>
          <span className="text-xs px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded font-medium">
            PaperBanana
          </span>
        </div>
        <p className="text-sm text-text-muted line-clamp-2 mb-3">
          {figure.caption}
        </p>

        {figure.error && (
          <p className="text-sm text-accent-red mb-2">{figure.error}</p>
        )}

        {showEdit && (
          <div className="mb-3">
            <textarea
              value={editDesc}
              onChange={(e) => setEditDesc(e.target.value)}
              rows={3}
              className="w-full px-2 py-1.5 text-sm border border-card-border rounded-lg
                         bg-background text-foreground resize-none
                         focus:outline-none focus:ring-2 focus:ring-accent-blue/30"
            />
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          {figure.status === 'pending' && (
            <button
              onClick={() => onRegenerate(figure.id, figure.description)}
              className="text-sm px-3 py-1.5 bg-accent-blue text-white rounded-lg hover:bg-accent-blue/90"
            >
              Generate
            </button>
          )}

          {figure.status === 'ready' && (
            <>
              <button
                onClick={() => onApprove(figure.id)}
                className="text-sm px-3 py-1.5 bg-accent-green text-white rounded-lg hover:bg-accent-green/90"
              >
                Approve
              </button>
              <button
                onClick={() => setShowEdit(!showEdit)}
                className="text-sm px-3 py-1.5 border border-card-border rounded-lg hover:bg-background"
              >
                Edit & Regen
              </button>
              {showEdit && (
                <button
                  onClick={() => {
                    onRegenerate(figure.id, editDesc);
                    setShowEdit(false);
                  }}
                  className="text-sm px-3 py-1.5 bg-accent-amber text-white rounded-lg hover:bg-accent-amber/90"
                >
                  Regenerate
                </button>
              )}
            </>
          )}

          {(figure.status === 'error' || figure.status === 'ready' || figure.status === 'pending') && (
            <>
              <button
                onClick={() => fileRef.current?.click()}
                className="text-sm px-3 py-1.5 border border-card-border rounded-lg hover:bg-background"
              >
                Upload
              </button>
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) onUpload(figure.id, file);
                }}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
