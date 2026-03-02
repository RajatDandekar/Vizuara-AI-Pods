'use client';

import { useState } from 'react';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';
import ApprovalBar from '../shared/ApprovalBar';

interface ArticleEditorProps {
  content: string;
  streaming?: boolean;
  onApprove: (content: string) => void;
  onRegenerate?: () => void;
}

export default function ArticleEditor({
  content,
  streaming = false,
  onApprove,
  onRegenerate,
}: ArticleEditorProps) {
  const [editing, setEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(content);
  const [viewMode, setViewMode] = useState<'preview' | 'split' | 'raw'>('preview');

  // Keep editor in sync when new streaming content arrives
  if (!editing && editedContent !== content) {
    setEditedContent(content);
  }

  return (
    <div className="space-y-4">
      {/* View mode tabs */}
      {!streaming && (
        <div className="flex items-center gap-1 bg-background p-1 rounded-lg w-fit">
          {(['preview', 'split', 'raw'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => {
                setViewMode(mode);
                if (mode === 'split' || mode === 'raw') setEditing(true);
                else setEditing(false);
              }}
              className={`px-3 py-1.5 rounded-md text-base font-medium transition-colors
                ${viewMode === mode ? 'bg-card-bg text-foreground shadow-sm' : 'text-text-muted hover:text-text-secondary'}`}
            >
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>
      )}

      {/* Content area */}
      <div
        className={`${viewMode === 'split' ? 'grid grid-cols-2 gap-4' : ''}`}
      >
        {(viewMode === 'raw' || viewMode === 'split') && (
          <textarea
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            className="w-full h-[calc(100vh-320px)] px-4 py-3 border border-card-border rounded-xl
                       bg-background text-foreground font-mono text-sm resize-none
                       focus:outline-none focus:ring-2 focus:ring-accent-blue/30"
          />
        )}

        {(viewMode === 'preview' || viewMode === 'split') && (
          <div
            className={`border border-card-border rounded-xl p-6 overflow-auto
                        ${viewMode === 'split' ? 'h-[calc(100vh-320px)]' : 'max-h-[calc(100vh-320px)]'}
                        ${streaming ? 'streaming-cursor' : ''}`}
          >
            <MarkdownRenderer content={editing ? editedContent : content} streaming={streaming} />
          </div>
        )}
      </div>

      {/* Approval bar (hidden during streaming) */}
      {!streaming && content && (
        <ApprovalBar
          onApprove={() => onApprove(editing ? editedContent : content)}
          onRegenerate={onRegenerate}
          approveLabel="Approve Article"
        />
      )}
    </div>
  );
}
