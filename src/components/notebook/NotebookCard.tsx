'use client';

import Link from 'next/link';
import Badge from '@/components/ui/Badge';
import type { NotebookMeta } from '@/types/course';

interface NotebookCardProps {
  notebook: NotebookMeta;
  courseSlug: string;
  podSlug: string;
  status: 'completed' | 'current' | 'locked';
}

export default function NotebookCard({ notebook, courseSlug, podSlug, status }: NotebookCardProps) {
  const isLocked = status === 'locked';

  const card = (
    <div
      className={`
        bg-card-bg border rounded-xl p-5 transition-all duration-300
        ${status === 'completed'
          ? 'border-accent-green/30 hover:shadow-md cursor-pointer'
          : status === 'current'
            ? 'border-accent-blue shadow-md ring-1 ring-accent-blue/20 cursor-pointer'
            : 'border-card-border opacity-60'
        }
        ${!isLocked ? 'hover:shadow-md' : ''}
      `}
    >
      <div className="flex items-start gap-4">
        {/* Order circle */}
        <div
          className={`
            w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0
            ${status === 'completed'
              ? 'bg-accent-green text-white'
              : status === 'current'
                ? 'bg-accent-blue text-white'
                : 'bg-gray-200 text-text-muted'
            }
          `}
        >
          {status === 'completed' ? (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          ) : isLocked ? (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          ) : (
            notebook.order
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-foreground text-sm truncate">
              {notebook.title}
            </h3>
          </div>
          <p className="text-xs text-text-secondary leading-relaxed mb-2">
            {notebook.objective}
          </p>
          <div className="flex items-center gap-3">
            <span className="text-xs text-text-muted flex items-center gap-1">
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {notebook.estimatedMinutes} min
            </span>
            {notebook.todoCount > 0 && (
              <Badge variant="blue" size="sm">
                {notebook.todoCount} exercises
              </Badge>
            )}
            {notebook.hasNarration && (
              <span className="text-xs text-text-muted flex items-center gap-1">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                Narrated
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  if (isLocked) return card;

  return (
    <Link href={`/courses/${courseSlug}/${podSlug}/practice/${notebook.order}`}>
      {card}
    </Link>
  );
}
