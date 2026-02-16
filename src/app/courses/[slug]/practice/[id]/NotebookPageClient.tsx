'use client';

import { useState, useEffect, useCallback } from 'react';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import CourseNav from '@/components/navigation/CourseNav';
import CourseProgressBar from '@/components/course/CourseProgressBar';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import type { NotebookMeta } from '@/types/course';
import { getProgress, markNotebookComplete, isNotebookComplete } from '@/lib/progress';

interface Props {
  courseSlug: string;
  courseTitle: string;
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
  notebook,
  prevOrder,
  nextOrder,
  totalNotebooks,
  notebooks,
  hasCaseStudy,
}: Props) {
  const [completed, setCompleted] = useState(false);

  useEffect(() => {
    setCompleted(isNotebookComplete(courseSlug, notebook.slug));
  }, [courseSlug, notebook.slug]);

  function handleMarkComplete() {
    markNotebookComplete(courseSlug, notebook.slug);
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
    // Auto-mark notebook as complete on download
    if (!completed) {
      markNotebookComplete(courseSlug, notebook.slug);
      setCompleted(true);
    }
  }, [notebook.downloadPath, notebook.slug, courseSlug, completed]);

  const handleOpenInColab = useCallback(() => {
    if (notebook.colabUrl) {
      window.open(notebook.colabUrl, '_blank');
    } else {
      // If no Colab URL, open Colab upload page
      window.open('https://colab.research.google.com/#create=true', '_blank');
    }
    // Auto-mark notebook as complete when opening in Colab
    if (!completed) {
      markNotebookComplete(courseSlug, notebook.slug);
      setCompleted(true);
    }
  }, [notebook.colabUrl, notebook.slug, courseSlug, completed]);

  const isLast = !nextOrder;
  const progress = getProgress(courseSlug);
  const completedCount = progress.completedNotebooks.length;
  const hasColabUrl = notebook.colabUrl && notebook.colabUrl.length > 0;
  const hasDownload = !!notebook.downloadPath;

  return (
    <>
      <CourseProgressBar
        courseSlug={courseSlug}
        courseTitle={courseTitle}
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
            { label: 'Practice', href: `/courses/${courseSlug}/practice` },
            { label: `Notebook ${notebook.order}` },
          ]}
        />
      </FadeIn>

      {/* Header */}
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

      {/* Notebook Actions */}
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
            {/* Open in Colab — primary action */}
            <Button
              size="lg"
              onClick={handleOpenInColab}
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              {hasColabUrl ? 'Open in Colab' : 'Open Google Colab'}
            </Button>

            {/* Download button */}
            {hasDownload && (
              <Button size="lg" variant="secondary" onClick={handleDownload}>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download .ipynb
              </Button>
            )}
          </div>

          {hasDownload && !hasColabUrl && (
            <p className="mt-4 text-xs text-text-muted max-w-sm mx-auto">
              After downloading, open Google Colab and use File &rarr; Upload notebook to run it.
            </p>
          )}

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

      {/* Mark Complete */}
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

      {/* Progress overview */}
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

      {/* Navigation */}
      <FadeIn delay={0.3}>
        <CourseNav
          prevHref={
            prevOrder
              ? `/courses/${courseSlug}/practice/${prevOrder}`
              : `/courses/${courseSlug}/practice`
          }
          prevLabel={prevOrder ? `Notebook ${prevOrder}` : 'All Notebooks'}
          nextHref={
            nextOrder
              ? `/courses/${courseSlug}/practice/${nextOrder}`
              : isLast
                ? hasCaseStudy
                  ? `/courses/${courseSlug}/case-study`
                  : `/courses/${courseSlug}/certificate`
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
