'use client';

import { useAuth } from '@/context/AuthContext';
import EmojiRating from './EmojiRating';
import ThumbsRating from './ThumbsRating';

interface Props {
  courseSlug: string;
  podSlug: string;
  contentType: 'article' | 'notebook';
  notebookOrder?: number;
}

export default function InlineFeedback({ courseSlug, podSlug, contentType, notebookOrder }: Props) {
  const { user } = useAuth();

  if (!user) return null;

  return (
    <div className="border border-card-border rounded-xl p-5 my-6 bg-card-bg">
      <p className="text-sm font-medium text-foreground mb-3">
        How was this {contentType === 'notebook' ? 'notebook' : 'article'}?
      </p>
      <div className="flex flex-col sm:flex-row sm:items-center gap-4">
        <EmojiRating
          courseSlug={courseSlug}
          podSlug={podSlug}
          contentType={contentType}
          notebookOrder={notebookOrder}
        />
        <div className="hidden sm:block w-px h-8 bg-gray-200" />
        <ThumbsRating
          courseSlug={courseSlug}
          podSlug={podSlug}
          contentType={contentType}
          notebookOrder={notebookOrder}
        />
      </div>
    </div>
  );
}
