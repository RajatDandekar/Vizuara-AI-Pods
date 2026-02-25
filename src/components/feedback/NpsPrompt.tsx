'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useFeedback } from '@/hooks/useFeedback';
import type { FeedbackSubmission } from '@/types/feedback';

interface Props {
  courseSlug: string;
  podSlug?: string;
  contentType: 'pod' | 'course';
}

export default function NpsPrompt({ courseSlug, podSlug, contentType }: Props) {
  const { user } = useAuth();
  const { submit, checkExisting, submitting } = useFeedback();
  const [selected, setSelected] = useState<number | null>(null);
  const [comment, setComment] = useState('');
  const [showComment, setShowComment] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!user) return;
    checkExisting({ type: 'nps', courseSlug, podSlug, contentType }).then(existing => {
      if (existing) {
        setSelected(existing.rating);
        if (existing.comment) setComment(existing.comment);
        setDone(true);
      }
    });
  }, [checkExisting, courseSlug, podSlug, contentType, user]);

  if (!user) return null;

  async function handleSelect(value: number) {
    setSelected(value);
    setShowComment(true);
    const data: FeedbackSubmission = {
      type: 'nps',
      courseSlug,
      podSlug,
      contentType,
      rating: value,
      comment: comment || undefined,
    };
    await submit(data);
  }

  async function handleCommentSubmit() {
    if (selected === null) return;
    const data: FeedbackSubmission = {
      type: 'nps',
      courseSlug,
      podSlug,
      contentType,
      rating: selected,
      comment: comment || undefined,
    };
    await submit(data);
    setDone(true);
  }

  return (
    <div className="border border-card-border rounded-xl p-5 my-4 bg-card-bg">
      <p className="text-sm font-medium text-foreground mb-3">
        How likely are you to recommend Vizuara to a friend?
      </p>
      <div className="flex items-center gap-1 mb-2">
        {Array.from({ length: 10 }, (_, i) => i + 1).map(value => (
          <button
            key={value}
            onClick={() => handleSelect(value)}
            disabled={submitting}
            className={`w-8 h-8 text-xs font-medium rounded-lg transition-all cursor-pointer ${
              selected === value
                ? value >= 9
                  ? 'bg-accent-green text-white'
                  : value >= 7
                    ? 'bg-accent-amber text-white'
                    : 'bg-red-500 text-white'
                : 'bg-gray-100 text-text-secondary hover:bg-gray-200'
            }`}
          >
            {value}
          </button>
        ))}
      </div>
      <div className="flex justify-between text-xs text-text-muted mb-3">
        <span>Not likely</span>
        <span>Very likely</span>
      </div>

      {(showComment || done) && (
        <div className="mt-3">
          <textarea
            value={comment}
            onChange={e => setComment(e.target.value)}
            placeholder="Any additional thoughts? (optional)"
            className="w-full text-sm border border-gray-200 rounded-lg p-2.5 bg-white focus:outline-none focus:ring-2 focus:ring-accent-blue/30 resize-none"
            rows={2}
            disabled={done}
          />
          {!done && (
            <button
              onClick={handleCommentSubmit}
              disabled={submitting}
              className="mt-2 text-xs font-medium px-3 py-1.5 rounded-lg bg-foreground text-white hover:bg-gray-800 transition-colors cursor-pointer"
            >
              Submit
            </button>
          )}
          {done && (
            <p className="text-xs text-accent-green mt-1">Thank you for your feedback!</p>
          )}
        </div>
      )}
    </div>
  );
}
