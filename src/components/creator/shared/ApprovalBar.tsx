'use client';

interface ApprovalBarProps {
  onApprove: () => void;
  onRegenerate?: () => void;
  onEdit?: () => void;
  approveLabel?: string;
  regenerateLabel?: string;
  editLabel?: string;
  disabled?: boolean;
  loading?: boolean;
}

export default function ApprovalBar({
  onApprove,
  onRegenerate,
  onEdit,
  approveLabel = 'Approve',
  regenerateLabel = 'Regenerate',
  editLabel = 'Edit',
  disabled = false,
  loading = false,
}: ApprovalBarProps) {
  return (
    <div className="flex items-center gap-3 py-4 px-6 bg-card-bg border border-card-border rounded-xl">
      <button
        onClick={onApprove}
        disabled={disabled || loading}
        className="px-5 py-2.5 bg-accent-green text-white rounded-lg font-medium
                   hover:bg-accent-green/90 transition-colors disabled:opacity-50"
      >
        {approveLabel}
      </button>

      {onRegenerate && (
        <button
          onClick={onRegenerate}
          disabled={disabled || loading}
          className="px-5 py-2.5 bg-accent-amber text-white rounded-lg font-medium
                     hover:bg-accent-amber/90 transition-colors disabled:opacity-50"
        >
          {loading ? 'Generating...' : regenerateLabel}
        </button>
      )}

      {onEdit && (
        <button
          onClick={onEdit}
          disabled={disabled}
          className="px-5 py-2.5 border border-card-border text-text-secondary rounded-lg font-medium
                     hover:bg-background transition-colors disabled:opacity-50"
        >
          {editLabel}
        </button>
      )}
    </div>
  );
}
