'use client';

import { useState } from 'react';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';
import ApprovalBar from '../shared/ApprovalBar';

interface OutlineEditorProps {
  outline: string;
  onApprove: (outline: string) => void;
  onRegenerate: () => void;
}

export default function OutlineEditor({
  outline,
  onApprove,
  onRegenerate,
}: OutlineEditorProps) {
  const [editing, setEditing] = useState(false);
  const [editedOutline, setEditedOutline] = useState(outline);

  const handleApprove = () => {
    onApprove(editing ? editedOutline : outline);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-foreground">Article Outline</h3>
        {!editing && (
          <button
            onClick={() => setEditing(true)}
            className="text-base text-accent-blue hover:underline"
          >
            Edit outline
          </button>
        )}
      </div>

      {editing ? (
        <div className="grid grid-cols-2 gap-4">
          {/* Editor */}
          <textarea
            value={editedOutline}
            onChange={(e) => setEditedOutline(e.target.value)}
            className="w-full h-[600px] px-4 py-3 border border-card-border rounded-xl bg-background
                       text-foreground font-mono text-sm resize-none
                       focus:outline-none focus:ring-2 focus:ring-accent-blue/30"
          />
          {/* Preview */}
          <div className="border border-card-border rounded-xl p-6 overflow-auto max-h-[600px]">
            <MarkdownRenderer content={editedOutline} />
          </div>
        </div>
      ) : (
        <div className="border border-card-border rounded-xl p-6 max-h-[600px] overflow-auto">
          <MarkdownRenderer content={outline} />
        </div>
      )}

      <ApprovalBar
        onApprove={handleApprove}
        onRegenerate={onRegenerate}
        onEdit={editing ? () => setEditing(false) : () => setEditing(true)}
        approveLabel="Approve & Generate Draft"
        editLabel={editing ? 'View Rendered' : 'Edit Outline'}
      />
    </div>
  );
}
