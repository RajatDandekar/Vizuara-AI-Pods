'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import type { NotebookMeta } from '@/types/course';
import { getProgress, onProgressChange } from '@/lib/progress';

interface Props {
  courseSlug: string;
  courseTitle: string;
  notebooks: NotebookMeta[];
  hasCaseStudy: boolean;
  /** Which step is currently active (for highlighting). e.g. 'article', 'nb-1', 'case-study', 'certificate' */
  activeStep: string;
}

export default function CourseProgressBar({
  courseSlug,
  courseTitle,
  notebooks,
  hasCaseStudy,
  activeStep,
}: Props) {
  const [progress, setProgressState] = useState(() => ({
    articleRead: false,
    completedNotebooks: [] as string[],
    caseStudyComplete: false,
    lastVisited: '',
  }));

  useEffect(() => {
    setProgressState(getProgress(courseSlug));
    return onProgressChange(() => setProgressState(getProgress(courseSlug)));
  }, [courseSlug]);

  const allNbDone = notebooks.length > 0 && progress.completedNotebooks.length === notebooks.length;
  const articleAndNbDone = progress.articleRead && (notebooks.length === 0 || allNbDone);

  type Step = { id: string; label: string; href: string; status: 'completed' | 'current' | 'locked' };
  const steps: Step[] = [];

  // Article
  steps.push({
    id: 'article',
    label: 'Article',
    href: `/courses/${courseSlug}/article`,
    status: progress.articleRead ? 'completed' : 'current',
  });

  // Notebooks
  for (const nb of notebooks) {
    const done = progress.completedNotebooks.includes(nb.slug);
    const accessible = progress.articleRead;
    steps.push({
      id: `nb-${nb.order}`,
      label: `NB ${nb.order}`,
      href: `/courses/${courseSlug}/practice/${nb.order}`,
      status: done ? 'completed' : accessible ? 'current' : 'locked',
    });
  }

  // Case study
  if (hasCaseStudy) {
    steps.push({
      id: 'case-study',
      label: 'Case Study',
      href: `/courses/${courseSlug}/case-study`,
      status: progress.caseStudyComplete ? 'completed' : articleAndNbDone ? 'current' : 'locked',
    });
  }

  // Certificate
  const allDone = articleAndNbDone && (!hasCaseStudy || progress.caseStudyComplete);
  steps.push({
    id: 'certificate',
    label: 'Certificate',
    href: `/courses/${courseSlug}/certificate`,
    status: allDone ? 'current' : 'locked',
  });

  return (
    <div className="sticky top-0 z-30 bg-white/95 backdrop-blur-sm border-b border-card-border">
      <div className="max-w-4xl mx-auto px-4 sm:px-6">
        {/* Course title row */}
        <div className="flex items-center gap-2 pt-2 pb-1">
          <Link href={`/courses/${courseSlug}`} className="text-xs font-medium text-accent-blue hover:underline truncate">
            {courseTitle}
          </Link>
        </div>

        {/* Steps row */}
        <div className="flex items-center gap-1 pb-2 overflow-x-auto scrollbar-hide">
          {steps.map((step, i) => {
            const isActive = step.id === activeStep;
            const isClickable = step.status !== 'locked';

            const dot = (
              <div className="flex items-center gap-1 flex-shrink-0">
                {i > 0 && (
                  <div className={`w-3 sm:w-5 h-px flex-shrink-0 ${
                    step.status === 'completed' || (steps[i - 1]?.status === 'completed')
                      ? 'bg-accent-green'
                      : 'bg-card-border'
                  }`} />
                )}
                <div
                  className={`
                    flex items-center gap-1 px-2 py-1 rounded-full text-[11px] font-medium transition-all whitespace-nowrap
                    ${isActive
                      ? 'bg-accent-blue text-white'
                      : step.status === 'completed'
                        ? 'bg-accent-green-light text-accent-green'
                        : step.status === 'current'
                          ? 'bg-gray-100 text-text-secondary hover:bg-gray-200'
                          : 'bg-gray-50 text-text-muted/50'
                    }
                  `}
                >
                  {step.status === 'completed' && !isActive && (
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                    </svg>
                  )}
                  {step.status === 'locked' && (
                    <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                    </svg>
                  )}
                  {step.label}
                </div>
              </div>
            );

            return isClickable ? (
              <Link key={step.id} href={step.href} className="flex items-center">
                {dot}
              </Link>
            ) : (
              <div key={step.id} className="flex items-center">
                {dot}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
