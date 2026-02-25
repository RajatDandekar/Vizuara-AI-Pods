'use client';

import { useState, useEffect } from 'react';
import { useFeedback } from '@/hooks/useFeedback';
import type { FeedbackSubmission } from '@/types/feedback';

const EMOJIS = [
  { value: 1, emoji: '😕', label: 'Confused' },
  { value: 2, emoji: '😐', label: 'Meh' },
  { value: 3, emoji: '🙂', label: 'Okay' },
  { value: 4, emoji: '😊', label: 'Good' },
  { value: 5, emoji: '🤩', label: 'Amazing' },
];

interface Props {
  courseSlug: string;
  podSlug: string;
  contentType: 'article' | 'notebook';
  notebookOrder?: number;
}

export default function EmojiRating({ courseSlug, podSlug, contentType, notebookOrder }: Props) {
  const { submit, checkExisting, submitting } = useFeedback();
  const [selected, setSelected] = useState<number | null>(null);

  useEffect(() => {
    checkExisting({ type: 'emoji', courseSlug, podSlug, contentType, notebookOrder }).then(existing => {
      if (existing?.rating) setSelected(existing.rating);
    });
  }, [checkExisting, courseSlug, podSlug, contentType, notebookOrder]);

  async function handleSelect(value: number) {
    setSelected(value);
    const data: FeedbackSubmission = {
      type: 'emoji',
      courseSlug,
      podSlug,
      contentType,
      notebookOrder,
      rating: value,
    };
    await submit(data);
  }

  return (
    <div className="flex items-center gap-1">
      {EMOJIS.map(({ value, emoji, label }) => (
        <button
          key={value}
          onClick={() => handleSelect(value)}
          disabled={submitting}
          title={label}
          className={`text-2xl p-1.5 rounded-lg transition-all cursor-pointer ${
            selected === value
              ? 'bg-accent-amber-light scale-110'
              : 'hover:bg-gray-100 opacity-60 hover:opacity-100'
          }`}
        >
          {emoji}
        </button>
      ))}
    </div>
  );
}
