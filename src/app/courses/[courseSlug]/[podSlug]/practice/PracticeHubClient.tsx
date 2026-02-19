'use client';

import { useEffect, useState } from 'react';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import CourseNav from '@/components/navigation/CourseNav';
import NotebookCard from '@/components/notebook/NotebookCard';
import type { NotebookMeta, PodProgress } from '@/types/course';
import { getPodProgress } from '@/lib/progress';

interface Props {
  courseSlug: string;
  courseTitle: string;
  podSlug: string;
  title: string;
  notebooks: NotebookMeta[];
}

export default function PracticeHubClient({ courseSlug, courseTitle, podSlug, title, notebooks }: Props) {
  const [progress, setProgress] = useState<PodProgress>({
    articleRead: false,
    completedNotebooks: [],
    caseStudyComplete: false,
    lastVisited: '',
  });

  useEffect(() => {
    setProgress(getPodProgress(courseSlug, podSlug));
  }, [courseSlug, podSlug]);

  function getNotebookStatus(nb: NotebookMeta): 'completed' | 'current' | 'locked' {
    if (progress.completedNotebooks.includes(nb.slug)) return 'completed';
    if (nb.order === 1) return 'current';
    const prev = notebooks.find((n) => n.order === nb.order - 1);
    if (prev && progress.completedNotebooks.includes(prev.slug)) return 'current';
    return 'locked';
  }

  const completedCount = progress.completedNotebooks.length;
  const basePath = `/courses/${courseSlug}/${podSlug}`;

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
      <FadeIn>
        <Breadcrumb
          items={[
            { label: 'Courses', href: '/' },
            { label: courseTitle, href: `/courses/${courseSlug}` },
            { label: title, href: basePath },
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
              courseSlug={courseSlug}
              podSlug={podSlug}
              status={getNotebookStatus(nb)}
            />
          ))}
        </div>
      </FadeIn>

      <FadeIn delay={0.25}>
        <CourseNav
          prevHref={`${basePath}/article`}
          prevLabel="Back to Article"
          nextHref={
            notebooks.length > 0
              ? `${basePath}/practice/${notebooks[0].order}`
              : undefined
          }
          nextLabel="Start First Notebook"
        />
      </FadeIn>
    </div>
  );
}
