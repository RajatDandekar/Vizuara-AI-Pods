'use client';

import { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useFeedback } from '@/hooks/useFeedback';
import type { FeedbackCategory, FeedbackSubmission } from '@/types/feedback';

const CATEGORIES: { value: FeedbackCategory; label: string }[] = [
  { value: 'bug', label: 'Bug Report' },
  { value: 'suggestion', label: 'Suggestion' },
  { value: 'content', label: 'Content Issue' },
  { value: 'other', label: 'Other' },
];

export default function FeedbackTab() {
  const { user } = useAuth();
  const { submit, submitting } = useFeedback();
  const [open, setOpen] = useState(false);
  const [category, setCategory] = useState<FeedbackCategory>('suggestion');
  const [comment, setComment] = useState('');
  const [done, setDone] = useState(false);

  if (!user) return null;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!comment.trim()) return;

    const data: FeedbackSubmission = {
      type: 'general',
      category,
      comment: comment.trim(),
      pageUrl: window.location.href,
    };
    const result = await submit(data);
    if (result) {
      setDone(true);
      setTimeout(() => {
        setOpen(false);
        setDone(false);
        setComment('');
      }, 2000);
    }
  }

  return (
    <>
      {/* Side tab button */}
      <button
        onClick={() => setOpen(true)}
        className="fixed right-0 top-1/2 -translate-y-1/2 z-40 bg-foreground text-white text-xs font-medium px-2 py-3 rounded-l-lg shadow-lg hover:bg-gray-800 transition-colors cursor-pointer"
        style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}
      >
        Feedback
      </button>

      {/* Slide-out panel */}
      {open && (
        <>
          <div className="fixed inset-0 bg-black/30 z-40" onClick={() => setOpen(false)} />
          <div className="fixed right-0 top-0 bottom-0 w-full max-w-sm bg-white shadow-2xl z-50 flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-sm font-semibold text-foreground">Send Feedback</h2>
              <button
                onClick={() => setOpen(false)}
                className="p-1 text-text-muted hover:text-foreground cursor-pointer"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {done ? (
              <div className="flex-1 flex items-center justify-center p-8">
                <div className="text-center">
                  <div className="text-3xl mb-2">✓</div>
                  <p className="text-sm font-medium text-foreground">Thank you!</p>
                  <p className="text-xs text-text-muted mt-1">Your feedback has been submitted.</p>
                </div>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="flex-1 flex flex-col p-4 gap-4">
                <div>
                  <label className="text-xs font-medium text-text-secondary mb-1.5 block">Category</label>
                  <div className="flex flex-wrap gap-1.5">
                    {CATEGORIES.map(({ value, label }) => (
                      <button
                        key={value}
                        type="button"
                        onClick={() => setCategory(value)}
                        className={`text-xs px-3 py-1.5 rounded-full border transition-colors cursor-pointer ${
                          category === value
                            ? 'bg-accent-blue-light border-accent-blue text-accent-blue'
                            : 'border-gray-200 text-text-muted hover:border-gray-300'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex-1 flex flex-col">
                  <label className="text-xs font-medium text-text-secondary mb-1.5 block">
                    Your feedback
                  </label>
                  <textarea
                    value={comment}
                    onChange={e => setComment(e.target.value)}
                    placeholder="Tell us what's on your mind..."
                    className="flex-1 min-h-[120px] text-sm border border-gray-200 rounded-lg p-3 bg-white focus:outline-none focus:ring-2 focus:ring-accent-blue/30 resize-none"
                    required
                  />
                </div>

                <button
                  type="submit"
                  disabled={submitting || !comment.trim()}
                  className="w-full text-sm font-medium py-2.5 rounded-lg bg-foreground text-white hover:bg-gray-800 transition-colors disabled:opacity-50 cursor-pointer"
                >
                  {submitting ? 'Sending...' : 'Send Feedback'}
                </button>
              </form>
            )}
          </div>
        </>
      )}
    </>
  );
}
