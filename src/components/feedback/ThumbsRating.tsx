'use client';

import { useState, useEffect } from 'react';
import { useFeedback } from '@/hooks/useFeedback';
import type { FeedbackSubmission, FeedbackTag } from '@/types/feedback';

const TAG_OPTIONS: { value: FeedbackTag; label: string }[] = [
  { value: 'great_examples', label: 'Great examples' },
  { value: 'needs_more_code', label: 'Needs more code' },
  { value: 'too_easy', label: 'Too easy' },
  { value: 'too_hard', label: 'Too hard' },
  { value: 'confusing', label: 'Confusing' },
];

interface Props {
  courseSlug: string;
  podSlug: string;
  contentType: 'article' | 'notebook';
  notebookOrder?: number;
}

export default function ThumbsRating({ courseSlug, podSlug, contentType, notebookOrder }: Props) {
  const { submit, checkExisting, submitting } = useFeedback();
  const [selected, setSelected] = useState<0 | 1 | null>(null);
  const [tags, setTags] = useState<FeedbackTag[]>([]);
  const [showTags, setShowTags] = useState(false);

  useEffect(() => {
    checkExisting({ type: 'thumbs', courseSlug, podSlug, contentType, notebookOrder }).then(existing => {
      if (existing) {
        setSelected(existing.rating as 0 | 1);
        if (existing.tags) setTags(existing.tags);
      }
    });
  }, [checkExisting, courseSlug, podSlug, contentType, notebookOrder]);

  async function handleThumb(value: 0 | 1) {
    setSelected(value);
    setShowTags(true);
    const data: FeedbackSubmission = {
      type: 'thumbs',
      courseSlug,
      podSlug,
      contentType,
      notebookOrder,
      rating: value,
      tags,
    };
    await submit(data);
  }

  async function toggleTag(tag: FeedbackTag) {
    const newTags = tags.includes(tag) ? tags.filter(t => t !== tag) : [...tags, tag];
    setTags(newTags);
    if (selected !== null) {
      const data: FeedbackSubmission = {
        type: 'thumbs',
        courseSlug,
        podSlug,
        contentType,
        notebookOrder,
        rating: selected,
        tags: newTags,
      };
      await submit(data);
    }
  }

  return (
    <div>
      <div className="flex items-center gap-2">
        <button
          onClick={() => handleThumb(1)}
          disabled={submitting}
          className={`p-2 rounded-lg transition-all cursor-pointer ${
            selected === 1
              ? 'bg-green-50 text-accent-green'
              : 'text-text-muted hover:bg-gray-100 hover:text-foreground'
          }`}
          title="Helpful"
        >
          <svg className="w-5 h-5" fill={selected === 1 ? 'currentColor' : 'none'} viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6.633 10.25c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 012.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 00.322-1.672V3a.75.75 0 01.75-.75 2.25 2.25 0 012.25 2.25c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282m0 0h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 01-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 00-1.423-.23H5.904m10.598-9.75H14.25M5.904 18.5c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 01-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 9.953 4.167 9.5 5 9.5h1.053c.472 0 .745.556.5.96a8.958 8.958 0 00-1.302 4.665c0 1.194.232 2.333.654 3.375z" />
          </svg>
        </button>
        <button
          onClick={() => handleThumb(0)}
          disabled={submitting}
          className={`p-2 rounded-lg transition-all cursor-pointer ${
            selected === 0
              ? 'bg-red-50 text-red-500'
              : 'text-text-muted hover:bg-gray-100 hover:text-foreground'
          }`}
          title="Not helpful"
        >
          <svg className="w-5 h-5" fill={selected === 0 ? 'currentColor' : 'none'} viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M7.498 15.25H4.372c-1.026 0-1.945-.694-2.054-1.715A12.137 12.137 0 012.25 12.25c0-2.848.992-5.464 2.649-7.521C5.287 4.247 5.886 4 6.504 4h4.016a4.5 4.5 0 011.423.23l3.114 1.04a4.5 4.5 0 001.423.23h1.294M7.498 15.25c.618 0 .991.724.725 1.282A7.471 7.471 0 007.5 19.75 2.25 2.25 0 009.75 22a.75.75 0 00.75-.75v-.633c0-.573.11-1.14.322-1.672.304-.76.93-1.33 1.653-1.715a9.04 9.04 0 002.861-2.4c.498-.634 1.226-1.08 2.032-1.08h.384m-10.253 1.5H9.7m8.075-9.75c.083-.205.173-.405.27-.602.197-.4-.078-.898-.523-.898h-.908c-.889 0-1.713.518-1.972 1.368a12 12 0 00-.521 3.507c0 1.553.295 3.036.831 4.398.306.774 1.086 1.227 1.918 1.227h1.053c.472 0 .745-.556.5-.96a8.958 8.958 0 01-1.302-4.665c0-1.194.232-2.333.654-3.375z" />
          </svg>
        </button>
      </div>

      {showTags && (
        <div className="flex flex-wrap gap-1.5 mt-2">
          {TAG_OPTIONS.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => toggleTag(value)}
              className={`text-xs px-2.5 py-1 rounded-full border transition-colors cursor-pointer ${
                tags.includes(value)
                  ? 'bg-accent-blue-light border-accent-blue text-accent-blue'
                  : 'border-gray-200 text-text-muted hover:border-gray-300'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
