'use client';

import { useEffect, useState } from 'react';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import CourseNav from '@/components/navigation/CourseNav';
import NotebookCard from '@/components/notebook/NotebookCard';
import type { NotebookMeta, CourseProgress } from '@/types/course';
import { getProgress } from '@/lib/progress';

interface Props {
  slug: string;
  title: string;
  notebooks: NotebookMeta[];
}

export default function PracticeHubClient({ slug, title, notebooks }: Props) {
  const [progress, setProgress] = useState<CourseProgress>({
    articleRead: false,
    completedNotebooks: [],
    caseStudyComplete: false,
    lastVisited: '',
  });

  useEffect(() => {
    setProgress(getProgress(slug));
  }, [slug]);

  function getNotebookStatus(
    nb: NotebookMeta
  ): 'completed' | 'current' | 'locked' {
    if (progress.completedNotebooks.includes(nb.slug)) return 'completed';
    // First notebook is always unlocked if article is read
    if (nb.order === 1) return 'current';
    // Unlock if previous notebook is completed
    const prev = notebooks.find((n) => n.order === nb.order - 1);
    if (prev && progress.completedNotebooks.includes(prev.slug)) return 'current';
    return 'locked';
  }

  const completedCount = progress.completedNotebooks.length;

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
      <FadeIn>
        <Breadcrumb
          items={[
            { label: 'Courses', href: '/' },
            { label: title, href: `/courses/${slug}` },
            { label: 'Practice' },
          ]}
        />
      </FadeIn>

      <FadeIn delay={0.1}>
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-foreground mb-2">Practice Notebooks</h1>
          <p className="text-text-secondary text-sm">
            Work through each notebook sequentially. Complete the exercises to unlock the next one.
          </p>
          <div className="mt-3 flex items-center gap-2">
            <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-accent-green rounded-full transition-all duration-500"
                style={{
                  width: `${notebooks.length > 0 ? (completedCount / notebooks.length) * 100 : 0}%`,
                }}
              />
            </div>
            <span className="text-xs text-text-muted font-medium">
              {completedCount}/{notebooks.length}
            </span>
          </div>
        </div>
      </FadeIn>

      <FadeIn delay={0.2}>
        <div className="space-y-3">
          {notebooks.map((nb) => (
            <NotebookCard
              key={nb.slug}
              notebook={nb}
              courseSlug={slug}
              status={getNotebookStatus(nb)}
            />
          ))}
        </div>
      </FadeIn>

      <FadeIn delay={0.25}>
        <CourseNav
          prevHref={`/courses/${slug}/article`}
          prevLabel="Back to Article"
          nextHref={
            notebooks.length > 0
              ? `/courses/${slug}/practice/${notebooks[0].order}`
              : undefined
          }
          nextLabel="Start First Notebook"
        />
      </FadeIn>
    </div>
  );
}
