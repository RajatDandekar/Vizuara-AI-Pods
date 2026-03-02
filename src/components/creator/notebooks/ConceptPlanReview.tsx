'use client';

import { useState } from 'react';
import type { ConceptPlan, ConceptPlanItem } from '@/types/creator';
import ApprovalBar from '../shared/ApprovalBar';

interface ConceptPlanReviewProps {
  plan: ConceptPlan;
  onApprove: (plan: ConceptPlan) => void;
  onRegenerate: () => void;
}

export default function ConceptPlanReview({
  plan,
  onApprove,
  onRegenerate,
}: ConceptPlanReviewProps) {
  const [concepts, setConcepts] = useState<ConceptPlanItem[]>(plan.concepts);
  const [editing, setEditing] = useState(false);

  const handleRemove = (id: string) => {
    setConcepts(concepts.filter((c) => c.id !== id));
  };

  const handleUpdate = (id: string, field: keyof ConceptPlanItem, value: string) => {
    setConcepts(
      concepts.map((c) =>
        c.id === id ? { ...c, [field]: value } : c
      )
    );
  };

  const handleMoveUp = (index: number) => {
    if (index === 0) return;
    const newList = [...concepts];
    [newList[index - 1], newList[index]] = [newList[index], newList[index - 1]];
    setConcepts(newList);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-foreground">Concept Plan</h3>
        <button
          onClick={() => setEditing(!editing)}
          className="text-sm text-accent-blue hover:underline"
        >
          {editing ? 'Done editing' : 'Edit plan'}
        </button>
      </div>

      <p className="text-text-muted text-base">
        {concepts.length} notebooks planned
      </p>

      <div className="space-y-3">
        {concepts.map((concept, index) => (
          <div
            key={concept.id}
            className="border border-card-border rounded-xl p-4 bg-card-bg"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <span className="text-base font-medium text-accent-blue bg-accent-blue-light px-3 py-1 rounded-full">
                  {index + 1}
                </span>
                {editing ? (
                  <input
                    value={concept.title}
                    onChange={(e) =>
                      handleUpdate(concept.id, 'title', e.target.value)
                    }
                    className="font-semibold text-foreground bg-transparent border-b border-card-border
                               focus:outline-none focus:border-accent-blue"
                  />
                ) : (
                  <h4 className="font-semibold text-foreground">
                    {concept.title}
                  </h4>
                )}
              </div>

              {editing && (
                <div className="flex gap-1">
                  <button
                    onClick={() => handleMoveUp(index)}
                    disabled={index === 0}
                    className="text-sm px-2 py-1 text-text-muted hover:text-foreground disabled:opacity-30"
                  >
                    Move up
                  </button>
                  <button
                    onClick={() => handleRemove(concept.id)}
                    className="text-sm px-2 py-1 text-accent-red hover:bg-accent-red/10 rounded"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>

            {editing ? (
              <textarea
                value={concept.objective}
                onChange={(e) =>
                  handleUpdate(concept.id, 'objective', e.target.value)
                }
                rows={2}
                className="w-full mt-2 text-sm text-text-secondary bg-transparent border border-card-border
                           rounded-lg p-2 resize-none focus:outline-none focus:ring-2 focus:ring-accent-blue/30"
              />
            ) : (
              <p className="text-text-secondary mt-1 ml-10">
                {concept.objective}
              </p>
            )}

            <div className="flex flex-wrap gap-1.5 mt-2 ml-10">
              {concept.topics.map((topic) => (
                <span
                  key={topic}
                  className="text-base px-2.5 py-0.5 bg-background text-text-muted rounded"
                >
                  {topic}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <ApprovalBar
        onApprove={() => onApprove({ concepts })}
        onRegenerate={onRegenerate}
        approveLabel="Approve Plan & Start Generation"
      />
    </div>
  );
}
