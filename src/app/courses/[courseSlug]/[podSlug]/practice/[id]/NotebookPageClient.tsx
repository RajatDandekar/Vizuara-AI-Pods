'use client';

import { useState, useEffect, useCallback } from 'react';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import CourseNav from '@/components/navigation/CourseNav';
import CourseProgressBar from '@/components/course/CourseProgressBar';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import type { NotebookMeta } from '@/types/course';
import { getPodProgress, markPodNotebookComplete, isPodNotebookComplete } from '@/lib/progress';

interface Props {
  courseSlug: string;
  courseTitle: string;
  podSlug: string;
  podTitle: string;
  notebook: NotebookMeta;
  prevOrder?: number;
  nextOrder?: number;
  totalNotebooks: number;
  notebooks: NotebookMeta[];
  hasCaseStudy?: boolean;
}

export default function NotebookPageClient({
  courseSlug,
  courseTitle,
  podSlug,
  podTitle,
  notebook,
  prevOrder,
  nextOrder,
  totalNotebooks,
  notebooks,
  hasCaseStudy,
}: Props) {
  const [completed, setCompleted] = useState(false);

  useEffect(() => {
    setCompleted(isPodNotebookComplete(courseSlug, podSlug, notebook.slug));
  }, [courseSlug, podSlug, notebook.slug]);

  function handleMarkComplete() {
    markPodNotebookComplete(courseSlug, podSlug, notebook.slug);
    setCompleted(true);
  }

  const handleDownload = useCallback(() => {
    if (!notebook.downloadPath) return;
    const link = document.createElement('a');
    link.href = notebook.downloadPath;
    link.download = notebook.downloadPath.split('/').pop() || 'notebook.ipynb';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    if (!completed) {
      markPodNotebookComplete(courseSlug, podSlug, notebook.slug);
      setCompleted(true);
    }
  }, [notebook.downloadPath, notebook.slug, courseSlug, podSlug, completed]);

  const handleOpenInColab = useCallback(() => {
    if (notebook.colabUrl) {
      window.open(notebook.colabUrl, '_blank');
    } else {
      window.open('https://colab.research.google.com/#create=true', '_blank');
    }
    if (!completed) {
      markPodNotebookComplete(courseSlug, podSlug, notebook.slug);
      setCompleted(true);
    }
  }, [notebook.colabUrl, notebook.slug, courseSlug, podSlug, completed]);

  const isLast = !nextOrder;
  const progress = getPodProgress(courseSlug, podSlug);
  const completedCount = progress.completedNotebooks.length;
  const hasColabUrl = notebook.colabUrl && notebook.colabUrl.length > 0;
  const hasDownload = !!notebook.downloadPath;
  const basePath = `/courses/${courseSlug}/${podSlug}`;

  return (
    <>
      <CourseProgressBar
        courseSlug={courseSlug}
        podSlug={podSlug}
        courseTitle={podTitle}
        notebooks={notebooks}
        hasCaseStudy={!!hasCaseStudy}
        activeStep={`nb-${notebook.order}`}
      />
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
      <FadeIn>
        <Breadcrumb
          items={[
            { label: 'Courses', href: '/' },
            { label: courseTitle, href: `/courses/${courseSlug}` },
            { label: podTitle, href: basePath },
            { label: 'Practice', href: `${basePath}/practice` },
            { label: `Notebook ${notebook.order}` },
          ]}
        />
      </FadeIn>

      <FadeIn delay={0.1}>
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-sm text-text-muted font-medium">
              Notebook {notebook.order} of {totalNotebooks}
            </span>
            {completed && <Badge variant="green" size="sm">Completed</Badge>}
          </div>
          <h1 className="text-2xl font-bold text-foreground mb-2">{notebook.title}</h1>
          <p className="text-text-secondary leading-relaxed">{notebook.objective}</p>
        </div>
      </FadeIn>

      <FadeIn delay={0.2}>
        <div className="bg-card-bg border border-card-border rounded-2xl p-8 text-center mb-6">
          <div className="w-16 h-16 rounded-2xl bg-amber-50 text-accent-amber flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-foreground mb-2">Ready to Code</h2>
          <p className="text-sm text-text-secondary mb-6 max-w-md mx-auto">
            Download this notebook and open it in Google Colab. Work through the exercises
            {notebook.hasNarration ? ' — this notebook includes voice narration inside Colab.' : '.'}
          </p>

          <div className="flex items-center justify-center gap-3 flex-wrap">
            <Button size="lg" onClick={handleOpenInColab}>
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              {hasColabUrl ? 'Open in Colab' : 'Open Google Colab'}
            </Button>

            {hasDownload && (
              <Button size="lg" variant="secondary" onClick={handleDownload}>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download .ipynb
              </Button>
            )}

            <Button
              size="lg"
              variant="secondary"
              onClick={() => window.open(`${basePath}/practice/${notebook.order}/assistant`, '_blank')}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
              </svg>
              AI Assistant
            </Button>
          </div>

          <div className="mt-4 flex items-center justify-center gap-4 text-xs text-text-muted">
            <span className="flex items-center gap-1">
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              ~{notebook.estimatedMinutes} min
            </span>
            {notebook.todoCount > 0 && (
              <span className="flex items-center gap-1">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                {notebook.todoCount} exercises
              </span>
            )}
          </div>
        </div>
      </FadeIn>

      {!completed && (
        <FadeIn delay={0.25}>
          <div className="text-center mb-6">
            <button
              onClick={handleMarkComplete}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium bg-accent-green-light text-accent-green hover:bg-green-100 transition-colors cursor-pointer"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Mark as Complete
            </button>
          </div>
        </FadeIn>
      )}

      <FadeIn delay={0.28}>
        <div className="flex items-center gap-2 justify-center mb-4">
          <div className="flex-1 max-w-xs h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-accent-green rounded-full transition-all duration-500"
              style={{
                width: `${totalNotebooks > 0 ? (completedCount / totalNotebooks) * 100 : 0}%`,
              }}
            />
          </div>
          <span className="text-xs text-text-muted font-medium">
            {completedCount}/{totalNotebooks} complete
          </span>
        </div>
      </FadeIn>

      <FadeIn delay={0.3}>
        <CourseNav
          prevHref={
            prevOrder
              ? `${basePath}/practice/${prevOrder}`
              : `${basePath}/practice`
          }
          prevLabel={prevOrder ? `Notebook ${prevOrder}` : 'All Notebooks'}
          nextHref={
            nextOrder
              ? `${basePath}/practice/${nextOrder}`
              : isLast
                ? hasCaseStudy
                  ? `${basePath}/case-study`
                  : `${basePath}/certificate`
                : undefined
          }
          nextLabel={
            nextOrder
              ? `Notebook ${nextOrder}`
              : isLast
                ? hasCaseStudy
                  ? 'Continue to Case Study'
                  : 'Get Certificate'
                : undefined
          }
        />
      </FadeIn>
    </div>
    </>
  );
}
